"""Collect the state, accounting record and logs of submitted SLURM jobs.

Job state lives in two places that disagree: `squeue` knows only pending and
running jobs, while `sacct` knows finished ones but lags by seconds. This module
queries both, falls back to `scontrol` for log paths when no run manifest exists,
and renders one consistent report. It also surfaces preemption restarts, which
otherwise look like a job mysteriously starting over. The runnable entry point is
`gather_job`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

#---------------------------------------------------------------------
# Constants
#---------------------------------------------------------------------

DEFAULT_RUNS_DIR = Path('slurm-runs')

SQUEUE_FIELDS = ('JobID', 'Name', 'State', 'Reason', 'TimeUsed', 'TimeLimit', 'NodeList', 'Partition')
SQUEUE_FORMAT = '%i|%j|%T|%R|%M|%l|%N|%P'

SACCT_FIELDS = ('JobID', 'JobName', 'State', 'ExitCode', 'Elapsed', 'Timelimit', 'MaxRSS', 'ReqTRES', 'NodeList', 'Start', 'End', 'DerivedExitCode')

#--- States that mean the job is over and its record will not change again ---
TERMINAL_STATES = frozenset({'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'})

#--- States that mean SLURM took the job away and will (by default) requeue it ---
PREEMPTION_STATES = frozenset({'PREEMPTED', 'REQUEUED', 'RESIZING', 'SUSPENDED'})

#--- Matched case-sensitively: these are exact strings as SLURM, the kernel and PyTorch emit them.
#--- Case-insensitive matching would conflate "CUDA out of memory" (VRAM) with OUT_OF_MEMORY (system RAM).
FAILURE_HINTS = (
    ('CUDA error: no kernel image is available',
     'The GPU is a newer architecture than the installed PyTorch/CUDA build supports. Pin a matching `GPU_CC:` constraint, or install a build with that arch.'),
    ('CUDA out of memory',
     'VRAM exhausted. Raise the VRAM floor (`--gres=gpu:1,VRAM:48G`) or lower the batch size. VRAM is selected per GPU, not per node.'),
    ('OUT_OF_MEMORY',
     'System RAM exhausted, not VRAM — the job was killed by the cgroup. Raise `--mem`; a normal DL job wants about 32G.'),
    ('oom-kill',
     'System RAM exhausted, not VRAM — the job was killed by the cgroup. Raise `--mem`; a normal DL job wants about 32G.'),
    ('DUE TO TIME LIMIT',
     'Walltime exhausted. Raise `--time` and checkpoint so the next run resumes instead of restarting.'),
    ('DUE TO PREEMPTION',
     'A DEADLINE-partition job took the node. Handle `SLURM_RESTART_COUNT` to resume from a checkpoint.'),
    ('command not found',
     'The environment was not activated inside the job. The batch shell is non-interactive, so ~/.bashrc conda hooks do not run — source conda explicitly.'),
    ('bad interpreter',
     'CRLF line endings in the script. Run `dos2unix` on it.'),
)


#---------------------------------------------------------------------
# Data model
#---------------------------------------------------------------------

@dataclass
class JobReport:
    """The gathered state of a single job.

    Attributes:
        job_id: SLURM job id.
        source: Where the state came from — `squeue`, `sacct`, or `unknown`.
        state: Current or final job state, e.g. `RUNNING`, `COMPLETED`.
        fields: Raw key-value pairs from the querying command.
        manifest: Contents of the run manifest, or None when no manifest exists.
        stdout_path: Resolved stdout log path, or None when unknown.
        stderr_path: Resolved stderr log path, or None when unknown.
        stdout_tail: Last lines of stdout, or a message explaining its absence.
        stderr_tail: Last lines of stderr when it is a separate file.
        restart_count: Number of times the job was requeued after preemption.
        hints: Diagnostic suggestions matched against the logs and state.
    """

    job_id: str
    source: str
    state: str
    fields: dict[str, str] = field(default_factory=dict)
    manifest: dict | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    stdout_tail: str = ''
    stderr_tail: str = ''
    restart_count: int = 0
    hints: list[str] = field(default_factory=list)

    def is_terminal(self) -> bool:
        """Return True when the job has finished and its record is final."""
        return self.state.split()[0] in TERMINAL_STATES if self.state else False


#---------------------------------------------------------------------
# SLURM queries
#---------------------------------------------------------------------

def run_slurm_command(command: list[str], timeout: int = 30) -> str | None:
    """Run a SLURM query and return stdout, or None when the command is unusable.

    Args:
        command: Argument vector to execute.
        timeout: Seconds to wait before giving up.

    Returns:
        Captured stdout on success, or None if the binary is missing, the call
        times out, or the command exits non-zero.
    """
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None

    if completed.returncode != 0:
        return None

    return completed.stdout


def query_squeue(job_id: str) -> dict[str, str] | None:
    """Return live queue state for a job, or None when it is not queued.

    Args:
        job_id: Job id to look up.

    Returns:
        Mapping of `SQUEUE_FIELDS` to values, or None if the job has already left
        the queue (or squeue is unavailable).
    """
    output = run_slurm_command(['squeue', '-j', job_id, '-h', '-o', SQUEUE_FORMAT])
    if not output or not output.strip():
        return None

    #--- Array jobs yield several lines; the first is representative enough for a status check ---
    parts = output.strip().splitlines()[0].split('|')
    if len(parts) != len(SQUEUE_FIELDS):
        return None

    return dict(zip(SQUEUE_FIELDS, [part.strip() for part in parts]))


def query_sacct(job_id: str) -> dict[str, str] | None:
    """Return the accounting record for a finished job, or None when unavailable.

    Only the top-level job entry is returned; `.batch` and `.extern` steps are
    merged in for `MaxRSS`, which SLURM records only on the step.

    Args:
        job_id: Job id to look up.

    Returns:
        Mapping of `SACCT_FIELDS` to values, or None when sacct has no record.
    """
    output = run_slurm_command(['sacct', '-j', job_id, '-n', '-P', '--format', ','.join(SACCT_FIELDS)])
    if not output or not output.strip():
        return None

    primary: dict[str, str] | None = None
    step_max_rss = ''

    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split('|')]
        if len(parts) != len(SACCT_FIELDS):
            continue
        record = dict(zip(SACCT_FIELDS, parts))
        if '.' in record['JobID']:
            step_max_rss = step_max_rss or record.get('MaxRSS', '')
        elif primary is None:
            primary = record

    if primary is None:
        return None

    if not primary.get('MaxRSS') and step_max_rss:
        primary['MaxRSS'] = step_max_rss

    return primary


def query_scontrol_log_paths(job_id: str) -> tuple[str | None, str | None]:
    """Recover stdout/stderr paths from `scontrol` for a job with no manifest.

    Only works while SLURM still holds the job record in memory, which is
    typically a few minutes after completion.

    Args:
        job_id: Job id to look up.

    Returns:
        A tuple of stdout path and stderr path, each None when not found.
    """
    output = run_slurm_command(['scontrol', 'show', 'job', job_id, '-d'])
    if not output:
        return None, None

    stdout_path: str | None = None
    stderr_path: str | None = None

    for token in output.split():
        if token.startswith('StdOut='):
            stdout_path = token.split('=', 1)[1]
        elif token.startswith('StdErr='):
            stderr_path = token.split('=', 1)[1]

    return stdout_path, stderr_path


def query_restart_count(job_id: str) -> int:
    """Return how many times the job was requeued, e.g. after preemption.

    Args:
        job_id: Job id to look up.

    Returns:
        The restart count, or 0 when it cannot be determined.
    """
    output = run_slurm_command(['scontrol', 'show', 'job', job_id])
    if not output:
        return 0

    for token in output.split():
        if token.startswith('Restarts='):
            value = token.split('=', 1)[1]
            return int(value) if value.isdigit() else 0

    return 0


#---------------------------------------------------------------------
# Log handling
#---------------------------------------------------------------------

def read_log_tail(log_path: str | None, line_count: int) -> str:
    """Return the last lines of a log file, or an explanatory message.

    Args:
        log_path: Path to the log file, or None when unknown.
        line_count: Number of trailing lines to return.

    Returns:
        The trailing log text, or a bracketed explanation when it cannot be read.
    """
    if log_path is None:
        return '[log path unknown — no manifest and scontrol no longer holds the job record]'

    path = Path(log_path)
    if not path.exists():
        return f'[no log file at {path} — the job may not have started, or the output directory was not writable]'

    try:
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError as error:
        return f'[could not read {path}: {error}]'

    if not lines:
        return f'[log file {path} is empty]'

    return '\n'.join(lines[-line_count:])


def collect_hints(report: JobReport) -> list[str]:
    """Match known failure signatures against the job state and logs.

    Args:
        report: A partially populated report with state and log tails filled in.

    Returns:
        Human-readable diagnostic hints, empty when nothing matched.
    """
    haystack = '\n'.join([report.state, report.stdout_tail, report.stderr_tail, report.fields.get('Reason', '')])
    hints: list[str] = []
    for signature, hint in FAILURE_HINTS:
        if signature in haystack and hint not in hints:
            hints.append(hint)

    if report.restart_count > 0:
        hints.append(f'Job was requeued {report.restart_count} time(s). Check that the script resumes from a checkpoint when SLURM_RESTART_COUNT is set.')

    if report.state.startswith('PENDING'):
        reason = report.fields.get('Reason', '')
        if reason in ('Resources', '(Resources)'):
            hints.append('Pending on Resources: the request is schedulable but nothing is free yet. Widen `--constraint` to reach more nodes.')
        elif 'QOSMax' in reason or 'AssocMax' in reason:
            hints.append(f'Pending on `{reason}`: an account or user limit is saturated. Check running jobs with `squeue -u $USER`.')
        elif reason:
            hints.append(f'Pending reason `{reason}`. A reason that never changes usually means the request can never be satisfied.')

    return hints


#---------------------------------------------------------------------
# Orchestration
#---------------------------------------------------------------------

def load_manifest(job_id: str, runs_dir: Path) -> dict | None:
    """Load the run manifest for a job id, or None when it does not exist.

    Args:
        job_id: Job id whose manifest to load.
        runs_dir: Directory that collects run manifests.

    Returns:
        Parsed manifest contents, or None when absent or unreadable.
    """
    manifest_path = runs_dir / f'{job_id}.json'
    if not manifest_path.is_file():
        return None

    try:
        return json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None


def gather_one_job(job_id: str, runs_dir: Path, tail_lines: int) -> JobReport:
    """Build a complete report for one job from queue, accounting and log data.

    Args:
        job_id: Job id to gather.
        runs_dir: Directory holding run manifests.
        tail_lines: Number of trailing log lines to include.

    Returns:
        The assembled report.
    """
    manifest = load_manifest(job_id, runs_dir)

    #--- squeue is authoritative while the job lives; sacct after it finishes ---
    queue_record = query_squeue(job_id)
    if queue_record is not None:
        source, fields, state = 'squeue', queue_record, queue_record.get('State', '')
    else:
        accounting_record = query_sacct(job_id)
        if accounting_record is not None:
            source, fields, state = 'sacct', accounting_record, accounting_record.get('State', '')
        else:
            source, fields, state = 'unknown', {}, ''

    #--- Log paths: manifest first (always correct), scontrol second (expires) ---
    stdout_path = manifest.get('stdout_path') if manifest else None
    stderr_path = manifest.get('stderr_path') if manifest else None
    if stdout_path is None:
        stdout_path, scontrol_stderr = query_scontrol_log_paths(job_id)
        stderr_path = stderr_path or scontrol_stderr

    report = JobReport(
        job_id=job_id,
        source=source,
        state=state,
        fields=fields,
        manifest=manifest,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        restart_count=query_restart_count(job_id),
    )
    report.stdout_tail = read_log_tail(stdout_path, tail_lines)
    if stderr_path and stderr_path != stdout_path:
        report.stderr_tail = read_log_tail(stderr_path, tail_lines)
    report.hints = collect_hints(report)

    return report


def format_report(report: JobReport) -> str:
    """Render a job report as a human-readable block.

    Args:
        report: The report to render.

    Returns:
        Multi-line text suitable for printing to a terminal.
    """
    lines = [f'=== job {report.job_id} ===']

    if report.source == 'unknown':
        lines.append('state    : UNKNOWN — not in squeue and no sacct record. The id may be wrong, or accounting may have purged it.')
    else:
        lines.append(f'state    : {report.state}  (via {report.source})')
        for key, value in report.fields.items():
            if key in ('JobID', 'State') or not value:
                continue
            lines.append(f'{key.lower():9s}: {value}')

    if report.manifest:
        lines.append(f'script   : {report.manifest.get("script_path")}')
        lines.append(f'submitted: {report.manifest.get("submitted_at")} by {report.manifest.get("submitted_by")} on {report.manifest.get("submitted_from")}')
    else:
        lines.append('manifest : none — this job was not submitted through submit_job.py, so provenance is unavailable.')

    if report.restart_count:
        lines.append(f'restarts : {report.restart_count}')

    lines.append(f'\n--- stdout ({report.stdout_path}) ---')
    lines.append(report.stdout_tail)

    if report.stderr_tail:
        lines.append(f'\n--- stderr ({report.stderr_path}) ---')
        lines.append(report.stderr_tail)

    if report.hints:
        lines.append('\n--- hints ---')
        lines.extend(f'  * {hint}' for hint in report.hints)

    return '\n'.join(lines)


def gather_job(
    job_id: list[str] | None = None,
    runs_dir: str = str(DEFAULT_RUNS_DIR),
    tail_lines: int = 40,
    json_output: bool = False,
    all_runs: bool = False,
) -> None:
    """Gather and report the state and logs of one or more SLURM jobs.

    Exits 0 when every gathered job completed successfully, 1 when any job failed
    or is still running, so that automation can branch on the result.

    Args:
        job_id: Job ids to gather. Ignored when `all_runs` is set.
        runs_dir: Directory holding run manifests.
        tail_lines: Number of trailing log lines to include per stream.
        json_output: Emit machine-readable JSON instead of a human report.
        all_runs: Gather every job that has a manifest in `runs_dir`.
    """
    resolved_runs_dir = Path(runs_dir).expanduser().resolve()

    if all_runs:
        job_ids = sorted((path.stem for path in resolved_runs_dir.glob('*.json')), key=lambda item: int(item) if item.isdigit() else 0)
    else:
        job_ids = list(job_id or [])

    if not job_ids:
        print('error: no job ids given and no manifests found. Pass job ids, or use --all-runs with a populated --runs-dir.', file=sys.stderr)
        raise SystemExit(2)

    reports = [gather_one_job(single_id, resolved_runs_dir, tail_lines) for single_id in job_ids]

    if json_output:
        print(json.dumps([asdict(report) for report in reports], indent=2))
    else:
        print('\n\n'.join(format_report(report) for report in reports))

    #--- Exit 0 only when every job is known to have finished successfully ---
    all_completed = all(report.state.split()[0] == 'COMPLETED' if report.state else False for report in reports)
    raise SystemExit(0 if all_completed else 1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the gatherer."""
    parser = argparse.ArgumentParser(description='Gather SLURM job state, accounting data and logs into one report.')
    parser.add_argument('job_id', nargs='*', help='Job id(s) to gather.')
    parser.add_argument('--runs-dir', default=str(DEFAULT_RUNS_DIR), help='Directory holding run manifests (default: ./slurm-runs).')
    parser.add_argument('--tail-lines', type=int, default=40, help='Trailing log lines to show per stream (default: 40).')
    parser.add_argument('--json-output', action='store_true', help='Emit JSON instead of a human-readable report.')
    parser.add_argument('--all-runs', action='store_true', help='Gather every job that has a manifest in --runs-dir.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    gather_job(**vars(args))
