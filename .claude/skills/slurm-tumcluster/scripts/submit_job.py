"""Validate, optionally dry-run, and submit an sbatch script, recording a run manifest.

Submission is only half of a reproducible workflow: the job id alone does not say
which script produced it, where its logs went, or what was requested. This module
writes a JSON manifest next to every submission so that `gather_job.py` can
reconstruct the full picture later without guessing. The runnable entry point is
`submit_job`.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import re
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from validate_sbatch import SEVERITY_ERROR, Directive, parse_sbatch_file, validate_sbatch_file

#---------------------------------------------------------------------
# Constants
#---------------------------------------------------------------------

SUBMITTED_JOB_PATTERN = re.compile(r'Submitted batch job (\d+)')
DEFAULT_RUNS_DIR = Path('slurm-runs')


#---------------------------------------------------------------------
# Data model
#---------------------------------------------------------------------

@dataclass
class RunManifest:
    """Everything needed to identify, reproduce and gather one submitted job.

    Attributes:
        job_id: SLURM job id returned by sbatch.
        job_name: Value of `--job-name`, or the script stem when unset.
        script_path: Absolute path of the submitted sbatch script.
        script_sha256: Digest of the script contents at submit time, so a later
            edit to the script cannot silently misrepresent what ran.
        submitted_at: UTC ISO-8601 timestamp of submission.
        submitted_by: Local username that ran sbatch.
        submitted_from: Hostname of the submitting machine.
        submit_dir: Working directory sbatch was invoked from; relative
            `--output` paths resolve against it.
        stdout_path: Resolved `--output` path with `%j`/`%x` substituted.
        stderr_path: Resolved `--error` path, equal to `stdout_path` when the
            script does not separate the streams.
        directives: Flat mapping of every honoured `#SBATCH` option to its value.
        extra_sbatch_args: Additional command-line arguments passed to sbatch,
            which override the in-file directives.
        validation_warnings: Warnings that were accepted at submit time.
    """

    job_id: str
    job_name: str
    script_path: str
    script_sha256: str
    submitted_at: str
    submitted_by: str
    submitted_from: str
    submit_dir: str
    stdout_path: str | None
    stderr_path: str | None
    directives: dict[str, str | None] = field(default_factory=dict)
    extra_sbatch_args: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)

    def save_to_json(self, runs_dir: Path) -> Path:
        """Write the manifest to `<runs_dir>/<job_id>.json` and return its path.

        Args:
            runs_dir: Directory that collects run manifests; created if missing.

        Returns:
            Path of the manifest file that was written.
        """
        runs_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = runs_dir / f'{self.job_id}.json'
        manifest_path.write_text(json.dumps(asdict(self), indent=2) + '\n', encoding='utf-8')
        return manifest_path


#---------------------------------------------------------------------
# Path resolution
#---------------------------------------------------------------------

def resolve_log_path(pattern: str | None, job_id: str, job_name: str, submit_dir: Path, user_name: str) -> str | None:
    """Substitute SLURM filename tokens so the log path can be read back later.

    Only the tokens knowable at submit time are substituted. Array tasks (`%a`)
    and node names (`%N`) are left in place, since they expand per task.

    Args:
        pattern: Raw `--output`/`--error` value, or None when unset.
        job_id: Job id assigned by sbatch.
        job_name: Resolved job name.
        submit_dir: Directory sbatch was invoked from, used to absolutise
            relative patterns.
        user_name: Submitting username, substituted for `%u`.

    Returns:
        The concrete log path, or None when `pattern` is None.
    """
    if pattern is None:
        return None

    resolved = pattern.replace('%j', job_id).replace('%A', job_id).replace('%x', job_name).replace('%u', user_name)

    path = Path(resolved)
    if not path.is_absolute():
        path = submit_dir / path

    return str(path)


def build_directive_map(directives: list[Directive]) -> dict[str, str | None]:
    """Collapse parsed directives into an option-to-value mapping.

    Later directives win, matching SLURM's own last-one-wins behaviour for
    repeated options.

    Args:
        directives: Directives in file order.

    Returns:
        Mapping from long-form option name to its value.
    """
    return {directive.option: directive.value for directive in directives}


def compute_sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


#---------------------------------------------------------------------
# sbatch invocation
#---------------------------------------------------------------------

def run_test_only(script_path: Path, extra_args: list[str]) -> tuple[bool, str]:
    """Ask SLURM when the job could start, without queueing it.

    Args:
        script_path: Path of the sbatch script.
        extra_args: Additional arguments forwarded to sbatch.

    Returns:
        A tuple of success flag and the combined sbatch output.

    Raises:
        RuntimeError: If sbatch is not installed on this machine.
    """
    command = ['sbatch', '--test-only', *extra_args, str(script_path)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    except FileNotFoundError as error:
        raise RuntimeError('sbatch not found on PATH — this machine is not a SLURM submit host.') from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError('sbatch --test-only timed out after 60s; the controller may be unreachable.') from error

    output = (completed.stdout + completed.stderr).strip()
    return completed.returncode == 0, output


def run_sbatch(script_path: Path, extra_args: list[str], submit_dir: Path) -> str:
    """Submit the script and return the assigned job id.

    Args:
        script_path: Path of the sbatch script.
        extra_args: Additional arguments forwarded to sbatch, which override
            in-file directives.
        submit_dir: Working directory to run sbatch from.

    Returns:
        The numeric job id as a string.

    Raises:
        RuntimeError: If sbatch is missing, fails, or its output cannot be parsed.
    """
    command = ['sbatch', *extra_args, str(script_path)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, cwd=submit_dir, timeout=60, check=False)
    except FileNotFoundError as error:
        raise RuntimeError('sbatch not found on PATH — this machine is not a SLURM submit host.') from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError('sbatch timed out after 60s; the job may or may not have been queued. Check `squeue -u $USER` before retrying.') from error

    if completed.returncode != 0:
        raise RuntimeError(f'sbatch failed with exit code {completed.returncode}:\n{completed.stderr.strip()}')

    match = SUBMITTED_JOB_PATTERN.search(completed.stdout)
    if match is None:
        raise RuntimeError(f'could not parse a job id from sbatch output:\n{completed.stdout.strip()}')

    return match.group(1)


#---------------------------------------------------------------------
# Orchestration
#---------------------------------------------------------------------

def submit_job(
    script_path: str,
    runs_dir: str = str(DEFAULT_RUNS_DIR),
    tier: str = 'wiss',
    dry_run: bool = False,
    force: bool = False,
    offline: bool = False,
    sbatch_arg: list[str] | None = None,
) -> None:
    """Validate an sbatch script, then submit it and write a run manifest.

    Validation errors block submission unless `force` is set; this is the safety
    interlock that keeps a malformed directive from silently changing what runs.

    Args:
        script_path: Path to the `.sbatch` file to submit.
        runs_dir: Directory in which to write the run manifest.
        tier: Account tier used for resource-ceiling validation.
        dry_run: Validate and run `sbatch --test-only`, but do not queue the job.
        force: Submit even when validation reports blocking errors.
        offline: Skip the live `sinfo` query during validation.
        sbatch_arg: Extra arguments forwarded verbatim to sbatch. These override
            the in-file directives and are recorded in the manifest.
    """
    target_path = Path(script_path).expanduser().resolve()
    extra_args = list(sbatch_arg or [])

    if not target_path.is_file():
        print(f'error: no such file: {target_path}', file=sys.stderr)
        raise SystemExit(2)

    #--- Gate 1: static validation ---
    issues = validate_sbatch_file(target_path, tier=tier, prefer_live=not offline)
    errors = [issue for issue in issues if issue.severity == SEVERITY_ERROR]
    warnings = [issue for issue in issues if issue.severity != SEVERITY_ERROR]

    for issue in issues:
        print(issue.format_line())

    if errors and not force:
        print(f'\nBLOCKED: {len(errors)} validation error(s). Fix them, or re-run with --force to override.', file=sys.stderr)
        raise SystemExit(1)
    if errors:
        print(f'\nWARNING: submitting with {len(errors)} validation error(s) because --force was given.', file=sys.stderr)

    #--- Gate 2: let SLURM itself confirm the request is schedulable ---
    try:
        test_ok, test_output = run_test_only(target_path, extra_args)
    except RuntimeError as error:
        print(f'\nerror: {error}', file=sys.stderr)
        raise SystemExit(3) from error

    print(f'\nsbatch --test-only: {"OK" if test_ok else "REJECTED"}')
    if test_output:
        print(test_output)
    if not test_ok and not force:
        print('\nBLOCKED: SLURM rejected the request. Fix it, or re-run with --force to override.', file=sys.stderr)
        raise SystemExit(1)

    if dry_run:
        print('\nDry run — job was not submitted.')
        raise SystemExit(0)

    #--- Submit and record ---
    submit_dir = Path.cwd()
    try:
        job_id = run_sbatch(target_path, extra_args, submit_dir)
    except RuntimeError as error:
        print(f'\nerror: {error}', file=sys.stderr)
        raise SystemExit(3) from error

    script_text = target_path.read_text(encoding='utf-8', errors='replace')
    directives, _ = parse_sbatch_file(script_text)
    directive_map = build_directive_map(directives)
    job_name = directive_map.get('job-name') or target_path.stem
    user_name = getpass.getuser()

    stdout_path = resolve_log_path(directive_map.get('output'), job_id, job_name, submit_dir, user_name)
    stderr_path = resolve_log_path(directive_map.get('error'), job_id, job_name, submit_dir, user_name) or stdout_path

    manifest = RunManifest(
        job_id=job_id,
        job_name=job_name,
        script_path=str(target_path),
        script_sha256=compute_sha256(target_path),
        submitted_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        submitted_by=user_name,
        submitted_from=socket.gethostname(),
        submit_dir=str(submit_dir),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        directives=directive_map,
        extra_sbatch_args=extra_args,
        validation_warnings=[issue.format_line() for issue in warnings],
    )
    manifest_path = manifest.save_to_json(Path(runs_dir).expanduser().resolve())

    print(f'\nSubmitted batch job {job_id}')
    print(f'  manifest : {manifest_path}')
    print(f'  stdout   : {stdout_path}')
    print(f'  stderr   : {stderr_path}')
    print(f'\nGather with: python {Path(__file__).parent / "gather_job.py"} {job_id} --runs-dir {runs_dir}')


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the submitter."""
    parser = argparse.ArgumentParser(description='Validate and submit an sbatch script, recording a run manifest.')
    parser.add_argument('script_path', help='Path to the .sbatch file to submit.')
    parser.add_argument('--runs-dir', default=str(DEFAULT_RUNS_DIR), help='Directory for run manifests (default: ./slurm-runs).')
    parser.add_argument('--tier', default='wiss', choices=['external', 'stud', 'wiss'], help='Account tier for resource-ceiling checks (default: wiss).')
    parser.add_argument('--dry-run', action='store_true', help='Validate and run sbatch --test-only without queueing the job.')
    parser.add_argument('--force', action='store_true', help='Submit even when validation or --test-only reports errors.')
    parser.add_argument('--offline', action='store_true', help='Skip the live sinfo query during validation.')
    parser.add_argument('--sbatch-arg', action='append', default=[], help='Extra argument forwarded to sbatch; repeatable. Overrides in-file directives.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    submit_job(**vars(args))
