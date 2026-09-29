"""Static validation of an sbatch script against the CVAI/I9 TUM cluster configuration.

This module exists because SLURM fails *silently* on several classes of malformed
`#SBATCH` directive: it ignores everything after a whitespace-broken option, it
ignores directives placed after the first command, and it does not expand shell
variables in paths. A job submitted with such a script runs with resources the
author never asked for. The public API is `validate_sbatch_file`, which returns a
list of `Issue` records, and the runnable entry point `validate_sbatch`.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

#---------------------------------------------------------------------
# Constants
#---------------------------------------------------------------------

REFERENCE_PATH = Path(__file__).resolve().parent.parent / 'reference' / 'node_features.json'

SEVERITY_ERROR = 'error'
SEVERITY_WARNING = 'warning'

#--- SLURM accepts MM, MM:SS, HH:MM:SS, DD-HH, DD-HH:MM, DD-HH:MM:SS ---
TIME_PATTERN = re.compile(r'^(\d+|\d+:\d{1,2}|\d+:\d{1,2}:\d{1,2}|\d+-\d{1,2}|\d+-\d{1,2}:\d{1,2}|\d+-\d{1,2}:\d{1,2}:\d{1,2})$')
MEMORY_PATTERN = re.compile(r'^\d+(\.\d+)?[KMGT]?$')
VRAM_PATTERN = re.compile(r'^VRAM:(\d+)([KMGT]?)$', re.IGNORECASE)
GPU_GRES_PATTERN = re.compile(r'^gpu(:[A-Za-z0-9_.\-]+)?(:\d+)?$')
JOB_ID_TOKEN_PATTERN = re.compile(r'%[a-zA-Z]')
SHELL_VARIABLE_PATTERN = re.compile(r'(\$\{?[A-Za-z_][A-Za-z0-9_]*\}?|(?<![\w/])~)')

#--- Options whose value is a filesystem path that SLURM resolves literally ---
PATH_OPTIONS = frozenset({'output', 'error', 'chdir', 'input'})

#--- Long-form names for the short options we care about ---
SHORT_TO_LONG = {
    '-J': 'job-name',
    '-N': 'nodes',
    '-n': 'ntasks',
    '-c': 'cpus-per-task',
    '-t': 'time',
    '-p': 'partition',
    '-o': 'output',
    '-e': 'error',
    '-D': 'chdir',
    '-w': 'nodelist',
    '-x': 'exclude',
    '-C': 'constraint',
    '-a': 'array',
    '-A': 'account',
    '-i': 'input',
}

REQUIRED_OPTIONS = ('time', 'output')


#---------------------------------------------------------------------
# Data model
#---------------------------------------------------------------------

@dataclass(frozen=True)
class Directive:
    """A single parsed `#SBATCH` line.

    Attributes:
        line_number: 1-indexed line in the source file.
        option: Long-form option name without leading dashes, e.g. `cpus-per-task`.
            Short options are normalised via `SHORT_TO_LONG` when known.
        value: The option's value, or None for flag-style options such as `--exclusive`.
        raw: The directive text with `#SBATCH` and surrounding whitespace stripped.
    """

    line_number: int
    option: str
    value: str | None
    raw: str


@dataclass(frozen=True)
class Issue:
    """A single validation finding.

    Attributes:
        severity: Either `error` (submission must be blocked) or `warning`.
        line_number: 1-indexed source line, or 0 for whole-file findings.
        code: Stable machine-readable identifier, e.g. `SB001`.
        message: Human-readable explanation including the fix.
    """

    severity: str
    line_number: int
    code: str
    message: str

    def format_line(self) -> str:
        """Return a single-line `severity file:line [code] message` rendering."""
        location = f'line {self.line_number}' if self.line_number > 0 else 'file'
        return f'{self.severity.upper():7s} {location:>9s}  [{self.code}] {self.message}'


@dataclass
class ClusterFacts:
    """Known-good cluster configuration used to check requests against reality.

    Attributes:
        node_features: Mapping from node name to its list of feature strings.
        valid_features: Every feature string that exists on at least one node.
        max_gpu_memory_gb: Largest `GPU_MEM` value present in the cluster.
        tier_limits: Per-account-tier resource ceilings keyed by tier name.
        partitions: Partition metadata keyed by partition name.
        is_live: True when `node_features` came from a live `sinfo` call.
    """

    node_features: dict[str, list[str]]
    valid_features: frozenset[str] = field(default_factory=frozenset)
    max_gpu_memory_gb: int = 0
    tier_limits: dict[str, dict] = field(default_factory=dict)
    partitions: dict[str, dict] = field(default_factory=dict)
    is_live: bool = False

    def __post_init__(self) -> None:
        """Derive the feature set and maximum VRAM from the node table."""
        features: set[str] = set()
        for feature_list in self.node_features.values():
            features.update(feature_list)
        self.valid_features = frozenset(features)
        memory_values = [int(item.split(':')[1]) for item in features if item.startswith('GPU_MEM:')]
        self.max_gpu_memory_gb = max(memory_values) if memory_values else 0

    def list_gpu_models(self) -> list[str]:
        """Return every value appearing as `GPU_MODEL:<value>` across the cluster."""
        return sorted({item.split(':', 1)[1] for item in self.valid_features if item.startswith('GPU_MODEL:')})


#---------------------------------------------------------------------
# Cluster facts loading
#---------------------------------------------------------------------

def load_cluster_facts(reference_path: Path = REFERENCE_PATH, prefer_live: bool = True) -> ClusterFacts:
    """Load cluster node features, preferring live `sinfo` output over the snapshot.

    The bundled snapshot goes stale as the sysadmins add nodes, so a live query is
    always attempted first when running on a machine that has SLURM installed.

    Args:
        reference_path: Path to the bundled `node_features.json`.
        prefer_live: When False, skip the `sinfo` call and use the snapshot only.
            Useful for offline authoring and for reproducible tests.

    Returns:
        Cluster facts with `is_live` indicating whether node data came from `sinfo`.

    Raises:
        FileNotFoundError: If the reference snapshot is missing.
    """
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    node_features = dict(reference['nodes'])
    is_live = False

    if prefer_live:
        live_features = query_live_node_features()
        if live_features:
            node_features = live_features
            is_live = True

    return ClusterFacts(
        node_features=node_features,
        tier_limits=reference['tier_limits'],
        partitions=reference['partitions'],
        is_live=is_live,
    )


def query_live_node_features() -> dict[str, list[str]] | None:
    """Return node-to-feature mapping from `sinfo`, or None when SLURM is unavailable.

    Returns:
        Mapping from node name to feature list, or None if `sinfo` is missing,
        times out, or returns nothing usable.
    """
    try:
        completed = subprocess.run(
            ['sinfo', '-N', '-h', '-o', '%N %f'],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None

    if completed.returncode != 0 or not completed.stdout.strip():
        return None

    node_features: dict[str, list[str]] = {}
    for line in completed.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        node_name, feature_text = parts[0], parts[1].strip()
        if feature_text in ('(null)', ''):
            node_features.setdefault(node_name, [])
            continue
        node_features[node_name] = [item.strip() for item in feature_text.split(',') if item.strip()]

    return node_features or None


#---------------------------------------------------------------------
# Parsing
#---------------------------------------------------------------------

def parse_sbatch_file(script_text: str) -> tuple[list[Directive], list[Issue]]:
    """Parse `#SBATCH` directives out of a script, reporting structural problems.

    Mirrors SLURM's own parser: directives are only honoured in the contiguous
    comment block at the top of the file, and the value of an `--opt=value`
    directive must not contain unquoted whitespace.

    Args:
        script_text: Full contents of the sbatch script.

    Returns:
        A tuple of:
            The directives SLURM would actually honour, in file order.
            Structural issues found while parsing.
    """
    issues: list[Issue] = []
    directives: list[Directive] = []
    lines = script_text.splitlines()

    if not lines:
        issues.append(Issue(SEVERITY_ERROR, 0, 'SB001', 'Script is empty.'))
        return directives, issues

    #--- Shebang must be the very first line or SLURM cannot launch the script ---
    if not lines[0].startswith('#!'):
        issues.append(Issue(SEVERITY_ERROR, 1, 'SB002', 'First line must be a shebang, e.g. `#!/bin/bash`.'))

    first_command_line = find_first_command_line(lines)

    for index, line in enumerate(lines):
        line_number = index + 1
        stripped = line.strip()
        if not stripped.startswith('#SBATCH'):
            continue

        #--- Directives after the first command are silently dropped by SLURM ---
        if first_command_line is not None and line_number > first_command_line:
            issues.append(
                Issue(
                    SEVERITY_ERROR,
                    line_number,
                    'SB003',
                    f'#SBATCH directive appears after the first command (line {first_command_line}). '
                    'SLURM ignores it. Move every directive above the first command.',
                )
            )
            continue

        raw = stripped[len('#SBATCH'):].strip()
        if not raw:
            issues.append(Issue(SEVERITY_WARNING, line_number, 'SB004', 'Empty #SBATCH directive; remove it.'))
            continue

        directive, directive_issues = parse_directive(raw, line_number)
        issues.extend(directive_issues)
        if directive is not None:
            directives.append(directive)

    return directives, issues


def find_first_command_line(lines: list[str]) -> int | None:
    """Return the 1-indexed line number of the first executable statement.

    Args:
        lines: Script lines without trailing newlines.

    Returns:
        Line number of the first non-empty, non-comment line, or None if the
        script contains no commands at all.
    """
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        return index + 1
    return None


def parse_directive(raw: str, line_number: int) -> tuple[Directive | None, list[Issue]]:
    """Parse one directive body and detect the whitespace-truncation trap.

    SLURM splits a directive on whitespace and only honours the first option. A
    line such as `--gres=gpu:1, VRAM:12G` therefore requests a GPU but silently
    drops the VRAM constraint.

    Args:
        raw: Directive text with the `#SBATCH` prefix already removed.
        line_number: 1-indexed source line, used for issue reporting.

    Returns:
        A tuple of the parsed directive (None when unparseable) and any issues.
    """
    issues: list[Issue] = []

    try:
        tokens = shlex.split(raw, comments=True)
    except ValueError as error:
        return None, [Issue(SEVERITY_ERROR, line_number, 'SB005', f'Unbalanced quotes in directive `{raw}`: {error}.')]

    if not tokens:
        return None, [Issue(SEVERITY_WARNING, line_number, 'SB004', 'Empty #SBATCH directive; remove it.')]

    head = tokens[0]

    #--- The `--opt=value` form must be a single token; anything trailing is dropped ---
    if '=' in head and len(tokens) > 1:
        dropped = ' '.join(tokens[1:])
        issues.append(
            Issue(
                SEVERITY_ERROR,
                line_number,
                'SB006',
                f'Whitespace inside the value of `{head}` — SLURM stops parsing there and discards `{dropped}`. '
                'Remove the space (e.g. `--gres=gpu:1,VRAM:12G`) or quote the whole value.',
            )
        )
    elif '=' not in head and len(tokens) > 2:
        issues.append(
            Issue(
                SEVERITY_ERROR,
                line_number,
                'SB006',
                f'Directive `{raw}` has more than one option/value pair; SLURM honours only `{tokens[0]} {tokens[1]}`. '
                'Put one option per #SBATCH line.',
            )
        )

    #--- Normalise into (option, value) ---
    if '=' in head:
        option_text, value = head.split('=', 1)
    elif len(tokens) >= 2 and not tokens[1].startswith('-'):
        option_text, value = head, tokens[1]
    else:
        option_text, value = head, None

    if option_text.startswith('--'):
        option = option_text[2:]
    elif option_text.startswith('-'):
        option = SHORT_TO_LONG.get(option_text, option_text[1:])
    else:
        return None, issues + [Issue(SEVERITY_ERROR, line_number, 'SB007', f'Directive `{raw}` does not start with an option flag.')]

    return Directive(line_number=line_number, option=option, value=value, raw=raw), issues


#---------------------------------------------------------------------
# Semantic checks
#---------------------------------------------------------------------

def check_required_options(directives: list[Directive]) -> list[Issue]:
    """Report missing directives that this cluster's conventions treat as mandatory."""
    issues: list[Issue] = []
    present = {directive.option for directive in directives}

    for option in REQUIRED_OPTIONS:
        if option not in present:
            issues.append(
                Issue(
                    SEVERITY_ERROR,
                    0,
                    'SB010',
                    f'Missing required `--{option}`. Without it the job inherits a partition default that is '
                    'usually wrong (an unbounded walltime, or output dumped into the submit directory).',
                )
            )

    if 'job-name' not in present:
        issues.append(Issue(SEVERITY_WARNING, 0, 'SB011', 'No `--job-name`; the job will be listed under the script filename in squeue.'))

    return issues


def check_paths(directives: list[Directive]) -> list[Issue]:
    """Validate path-valued options: no shell expansion, absolute, parent exists.

    SLURM writes `--output`/`--error` from the slurmd process, which performs no
    shell expansion and will not create missing directories. Both failure modes
    produce a job that starts and then vanishes with no log at all.
    """
    issues: list[Issue] = []

    for directive in directives:
        if directive.option not in PATH_OPTIONS or directive.value is None:
            continue

        value = directive.value

        match = SHELL_VARIABLE_PATTERN.search(value)
        if match is not None:
            issues.append(
                Issue(
                    SEVERITY_ERROR,
                    directive.line_number,
                    'SB020',
                    f'`--{directive.option}={value}` contains `{match.group(0)}`. SLURM does not expand shell variables '
                    'or `~` in #SBATCH directives — the literal string becomes part of the path. Write the path out in full.',
                )
            )
            continue

        if not value.startswith('/'):
            issues.append(
                Issue(
                    SEVERITY_WARNING,
                    directive.line_number,
                    'SB021',
                    f'`--{directive.option}={value}` is relative; it resolves against the submit directory (or `--chdir`), '
                    'which makes the job non-reproducible from elsewhere. Prefer an absolute path.',
                )
            )
            continue

        #--- Strip SLURM filename patterns (%j, %x, %A, ...) before checking the parent ---
        parent_text = JOB_ID_TOKEN_PATTERN.sub('X', value)
        parent_dir = Path(parent_text).parent
        if not parent_dir.exists():
            issues.append(
                Issue(
                    SEVERITY_ERROR,
                    directive.line_number,
                    'SB022',
                    f'Directory `{parent_dir}` for `--{directive.option}` does not exist. SLURM will not create it and '
                    'the job fails with no log written. Run `mkdir -p {parent}` first.'.replace('{parent}', str(parent_dir)),
                )
            )
        elif directive.option in ('output', 'error') and '%j' not in value and '%A' not in value:
            issues.append(
                Issue(
                    SEVERITY_WARNING,
                    directive.line_number,
                    'SB023',
                    f'`--{directive.option}={value}` has no `%j` job-id token; concurrent jobs will overwrite each other\'s logs.',
                )
            )

    return issues


def check_time_and_memory(directives: list[Directive]) -> list[Issue]:
    """Validate `--time` and memory option formats.

    A malformed `--time` is rejected at submit; a malformed `--mem` is rejected
    too, but both are cheap to catch before burning a queue round-trip.
    """
    issues: list[Issue] = []

    for directive in directives:
        if directive.option == 'time':
            if directive.value is None or not TIME_PATTERN.match(directive.value):
                issues.append(
                    Issue(
                        SEVERITY_ERROR,
                        directive.line_number,
                        'SB030',
                        f'`--time={directive.value}` is not a valid SLURM walltime. Use MM, MM:SS, HH:MM:SS, '
                        'DD-HH, DD-HH:MM or DD-HH:MM:SS.',
                    )
                )
        elif directive.option in ('mem', 'mem-per-cpu', 'mem-per-gpu'):
            if directive.value is None or not MEMORY_PATTERN.match(directive.value):
                issues.append(
                    Issue(
                        SEVERITY_ERROR,
                        directive.line_number,
                        'SB031',
                        f'`--{directive.option}={directive.value}` is not a valid size. Use a number with an optional '
                        'K/M/G/T suffix, e.g. `32G`. A bare number means megabytes.',
                    )
                )

    return issues


def check_gres_and_gpus(directives: list[Directive], facts: ClusterFacts) -> list[Issue]:
    """Validate `--gres` / `--gpus` requests against real cluster hardware."""
    issues: list[Issue] = []
    gpu_models = set(facts.list_gpu_models())

    for directive in directives:
        if directive.option not in ('gres', 'gpus', 'gpus-per-node', 'gpus-per-task') or directive.value is None:
            continue

        for component in directive.value.split(','):
            component = component.strip()
            if not component:
                issues.append(Issue(SEVERITY_ERROR, directive.line_number, 'SB040', f'Empty component in `--{directive.option}={directive.value}`.'))
                continue

            vram_match = VRAM_PATTERN.match(component)
            if vram_match is not None:
                issues.extend(check_vram_component(vram_match, component, directive, facts))
                continue

            if directive.option != 'gres':
                #--- `--gpus=titan:1` or `--gpus=2` ---
                model = component.split(':')[0] if ':' in component else None
                if model is not None and not model.isdigit() and model not in gpu_models:
                    issues.append(
                        Issue(
                            SEVERITY_ERROR,
                            directive.line_number,
                            'SB041',
                            f'Unknown GPU type `{model}` in `--{directive.option}={directive.value}`. '
                            f'Known types: {", ".join(sorted(gpu_models))}.',
                        )
                    )
                continue

            if component.lower().startswith('gpu'):
                if not GPU_GRES_PATTERN.match(component):
                    issues.append(
                        Issue(
                            SEVERITY_ERROR,
                            directive.line_number,
                            'SB042',
                            f'`{component}` is not a valid gpu gres spec. Use `gpu:<count>` or `gpu:<model>:<count>`.',
                        )
                    )
                    continue
                parts = component.split(':')
                if len(parts) == 3 and parts[1] not in gpu_models:
                    issues.append(
                        Issue(
                            SEVERITY_ERROR,
                            directive.line_number,
                            'SB041',
                            f'Unknown GPU type `{parts[1]}` in `{component}`. Known types: {", ".join(sorted(gpu_models))}.',
                        )
                    )
            else:
                issues.append(
                    Issue(
                        SEVERITY_WARNING,
                        directive.line_number,
                        'SB043',
                        f'Unrecognised gres component `{component}`; this cluster defines `gpu` and `VRAM`.',
                    )
                )

    return issues


def check_vram_component(vram_match: re.Match[str], component: str, directive: Directive, facts: ClusterFacts) -> list[Issue]:
    """Check that a `VRAM:<size>` gres request is satisfiable by some GPU.

    Args:
        vram_match: Match object from `VRAM_PATTERN` over `component`.
        component: The raw `VRAM:...` text, used in messages.
        directive: The directive the component belongs to.
        facts: Cluster facts supplying the maximum installed VRAM.

    Returns:
        Issues raised for this component; empty when the request is satisfiable.
    """
    amount = int(vram_match.group(1))
    unit = vram_match.group(2).upper()

    if unit == '':
        return [
            Issue(
                SEVERITY_WARNING,
                directive.line_number,
                'SB044',
                f'`{component}` has no unit suffix; SLURM interprets bare gres counts as megabytes. Write `VRAM:{amount}G` if you meant gigabytes.',
            )
        ]

    amount_gb = amount if unit == 'G' else amount / 1024 if unit == 'M' else amount * 1024 if unit == 'T' else amount
    if facts.max_gpu_memory_gb and amount_gb > facts.max_gpu_memory_gb:
        return [
            Issue(
                SEVERITY_ERROR,
                directive.line_number,
                'SB045',
                f'`{component}` requests more VRAM than any GPU in the cluster has ({facts.max_gpu_memory_gb}G max). The job would queue forever.',
            )
        ]

    return []


def check_constraints(directives: list[Directive], facts: ClusterFacts) -> list[Issue]:
    """Check every feature named in `--constraint` exists on at least one node.

    A typo in a constraint does not fail at submit time — the job simply sits in
    the queue with reason `Resources` and never starts.
    """
    issues: list[Issue] = []

    for directive in directives:
        if directive.option != 'constraint' or directive.value is None:
            continue

        #--- Constraints combine features with |, &, comma, and optional [count] / bracket groups ---
        tokens = [token for token in re.split(r'[|&,\[\]()*]', directive.value) if token.strip()]
        for token in tokens:
            feature = token.strip()
            if not feature or feature.isdigit():
                continue
            if feature not in facts.valid_features:
                suggestion = suggest_feature(feature, facts.valid_features)
                hint = f' Did you mean `{suggestion}`?' if suggestion else ''
                issues.append(
                    Issue(
                        SEVERITY_ERROR,
                        directive.line_number,
                        'SB050',
                        f'Feature `{feature}` exists on no node ({"live sinfo" if facts.is_live else "snapshot"} data). '
                        f'The job would queue forever with reason `Resources`.{hint}',
                    )
                )

    return issues


def suggest_feature(feature: str, valid_features: frozenset[str]) -> str | None:
    """Return the closest known feature name, or None when nothing is close.

    Args:
        feature: The unrecognised feature string from a `--constraint`.
        valid_features: Every feature present on the cluster.

    Returns:
        The best match with a similarity ratio above 0.6, otherwise None.
    """
    matches = difflib.get_close_matches(feature, sorted(valid_features), n=1, cutoff=0.6)
    return matches[0] if matches else None


def check_partition_rules(directives: list[Directive], facts: ClusterFacts) -> list[Issue]:
    """Enforce the chair's deadline-partition policy.

    The DEADLINE partitions preempt other users' running jobs, so the chair
    requires a written justification. Additionally, a small-GPU job on DEADLINE
    is itself preemptible by DEADLINEBIG unless it is pinned to `NOTBIG`.
    """
    issues: list[Issue] = []
    values = {directive.option: directive.value for directive in directives}
    partition = values.get('partition')

    if partition is None:
        return issues

    known_partitions = set(facts.partitions)
    if partition not in known_partitions:
        issues.append(
            Issue(
                SEVERITY_WARNING,
                0,
                'SB060',
                f'Partition `{partition}` is not one of the documented partitions ({", ".join(sorted(known_partitions))}). Verify with `sinfo -s`.',
            )
        )
        return issues

    if facts.partitions[partition].get('requires_comment') and not values.get('comment'):
        issues.append(
            Issue(
                SEVERITY_ERROR,
                0,
                'SB061',
                f'Partition `{partition}` preempts other users\' jobs and requires '
                '`--comment="<deadline justification>"`. Add the actual deadline name.',
            )
        )

    if partition == 'DEADLINE':
        constraint = values.get('constraint') or ''
        gres = values.get('gres') or ''
        vram_match = VRAM_PATTERN.search(gres)
        wants_small_gpu = 'NOTBIG' in constraint or (vram_match is not None and int(vram_match.group(1)) < 48)
        if not wants_small_gpu and 'NOTBIG' not in constraint:
            issues.append(
                Issue(
                    SEVERITY_WARNING,
                    0,
                    'SB062',
                    'On DEADLINE without `--constraint=NOTBIG`, this job may land on a 48G GPU and then be preempted '
                    'by DEADLINEBIG. Add `--constraint=NOTBIG` unless you specifically need a big GPU.',
                )
            )

    return issues


def check_tier_limits(directives: list[Directive], facts: ClusterFacts, tier: str) -> list[Issue]:
    """Check the job's GPU and CPU request against the account tier's ceilings.

    A single job requesting more than the per-user ceiling can never be scheduled.

    Args:
        directives: Parsed directives for the job.
        facts: Cluster facts holding the tier limit table.
        tier: Account tier name, one of `external`, `stud`, `wiss`.

    Returns:
        Issues for requests that exceed the tier's per-user limits.
    """
    issues: list[Issue] = []
    limits = facts.tier_limits.get(tier)
    if limits is None:
        return [Issue(SEVERITY_WARNING, 0, 'SB070', f'Unknown account tier `{tier}`; skipping limit checks.')]

    gpu_count = count_requested_gpus(directives)
    max_gpu = limits.get('max_gpu_per_user')
    if gpu_count is not None and max_gpu is not None and gpu_count > max_gpu:
        issues.append(
            Issue(
                SEVERITY_ERROR,
                0,
                'SB071',
                f'Job requests {gpu_count} GPUs but tier `{tier}` allows at most {max_gpu} per user. The job would never start.',
            )
        )

    return issues


def count_requested_gpus(directives: list[Directive]) -> int | None:
    """Return the total GPU count the job asks for, or None when unspecified.

    Args:
        directives: Parsed directives for the job.

    Returns:
        Number of GPUs requested, scaled by `--nodes` for the per-node forms, or
        None when no GPU request is present.
    """
    nodes = 1
    for directive in directives:
        if directive.option == 'nodes' and directive.value is not None:
            head = directive.value.split('-')[0]
            if head.isdigit():
                nodes = int(head)

    for directive in directives:
        if directive.option == 'gres' and directive.value is not None:
            for component in directive.value.split(','):
                parts = component.strip().split(':')
                if parts[0].lower() == 'gpu' and parts[-1].isdigit():
                    return int(parts[-1]) * nodes
        elif directive.option == 'gpus' and directive.value is not None:
            parts = directive.value.split(':')
            if parts[-1].isdigit():
                return int(parts[-1])
        elif directive.option in ('gpus-per-node',) and directive.value is not None:
            parts = directive.value.split(':')
            if parts[-1].isdigit():
                return int(parts[-1]) * nodes

    return None


def check_hygiene(script_text: str, directives: list[Directive]) -> list[Issue]:
    """Check whole-file conventions that make jobs reproducible and debuggable."""
    issues: list[Issue] = []
    values = {directive.option: directive.value for directive in directives}

    if '\r\n' in script_text:
        issues.append(
            Issue(
                SEVERITY_ERROR,
                0,
                'SB080',
                'File has CRLF line endings; the kernel includes `\\r` in the shebang interpreter path and the job '
                'fails with "bad interpreter". Convert with `dos2unix`.',
            )
        )

    if 'mail-type' in values and 'mail-user' not in values:
        issues.append(
            Issue(
                SEVERITY_WARNING,
                0,
                'SB081',
                '`--mail-type` is set without `--mail-user`; mail goes to the submitting account\'s local mailbox, which is usually unread.',
            )
        )

    #--- Unbuffered output is the documented fix for logs arriving in bunches ---
    if 'srun' in script_text and '--unbuffered' not in script_text:
        issues.append(
            Issue(
                SEVERITY_WARNING,
                0,
                'SB082',
                'srun without `--unbuffered`: stdout is pipe-buffered by glibc, so logs arrive in bunches and a crashed '
                'job loses its last output. Use `srun --unbuffered ...`.',
            )
        )

    if 'set -e' not in script_text and 'set -euo' not in script_text:
        issues.append(
            Issue(
                SEVERITY_WARNING,
                0,
                'SB083',
                'No `set -euo pipefail`: a failing setup step (e.g. `conda activate`) is ignored and the job reports '
                'COMPLETED while having done nothing.',
            )
        )

    return issues


#---------------------------------------------------------------------
# Orchestration
#---------------------------------------------------------------------

def validate_sbatch_file(script_path: Path, tier: str = 'wiss', prefer_live: bool = True) -> list[Issue]:
    """Run every check against an sbatch script and return the findings.

    Args:
        script_path: Path to the `.sbatch` file to validate.
        tier: Account tier used for resource-ceiling checks.
        prefer_live: Whether to query `sinfo` for current node features.

    Returns:
        All issues found, sorted with errors first and then by line number.

    Raises:
        FileNotFoundError: If `script_path` does not exist.
    """
    # Decode from bytes rather than read_text(): universal-newline translation would
    # rewrite CRLF to LF and make the SB080 line-ending check unfireable.
    script_bytes = script_path.read_bytes()
    script_text = script_bytes.decode('utf-8', errors='replace')
    facts = load_cluster_facts(prefer_live=prefer_live)

    directives, issues = parse_sbatch_file(script_text)
    issues.extend(check_required_options(directives))
    issues.extend(check_paths(directives))
    issues.extend(check_time_and_memory(directives))
    issues.extend(check_gres_and_gpus(directives, facts))
    issues.extend(check_constraints(directives, facts))
    issues.extend(check_partition_rules(directives, facts))
    issues.extend(check_tier_limits(directives, facts, tier))
    issues.extend(check_hygiene(script_text, directives))

    return sorted(issues, key=lambda issue: (issue.severity != SEVERITY_ERROR, issue.line_number, issue.code))


def validate_sbatch(script_path: str, tier: str = 'wiss', json_output: bool = False, offline: bool = False, strict: bool = False) -> None:
    """Validate an sbatch script and report findings, exiting non-zero on failure.

    Exit codes are the contract for automation: 0 means safe to submit, 1 means
    at least one blocking error, 2 means the script could not be read.

    Args:
        script_path: Path to the `.sbatch` file to validate.
        tier: Account tier for resource-ceiling checks, one of external/stud/wiss.
        json_output: Emit machine-readable JSON instead of a human report.
        offline: Skip the live `sinfo` query and use the bundled snapshot.
        strict: Treat warnings as errors for the purpose of the exit code.
    """
    target_path = Path(script_path).expanduser().resolve()

    if not target_path.is_file():
        print(f'error: no such file: {target_path}', file=sys.stderr)
        raise SystemExit(2)

    issues = validate_sbatch_file(target_path, tier=tier, prefer_live=not offline)
    error_count = sum(1 for issue in issues if issue.severity == SEVERITY_ERROR)
    warning_count = len(issues) - error_count

    if json_output:
        payload = {
            'script_path': str(target_path),
            'tier': tier,
            'error_count': error_count,
            'warning_count': warning_count,
            'issues': [{'severity': issue.severity, 'line': issue.line_number, 'code': issue.code, 'message': issue.message} for issue in issues],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f'Validating {target_path} (tier={tier})')
        for issue in issues:
            print(issue.format_line())
        if not issues:
            print('OK — no issues found.')
        else:
            print(f'\n{error_count} error(s), {warning_count} warning(s)')

    failed = error_count > 0 or (strict and warning_count > 0)
    raise SystemExit(1 if failed else 0)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the validator."""
    parser = argparse.ArgumentParser(description='Validate an sbatch script against the CVAI/I9 TUM cluster configuration.')
    parser.add_argument('script_path', help='Path to the .sbatch file to validate.')
    parser.add_argument('--tier', default='wiss', choices=['external', 'stud', 'wiss'], help='Account tier for resource-ceiling checks (default: wiss).')
    parser.add_argument('--json-output', action='store_true', help='Emit JSON instead of a human-readable report.')
    parser.add_argument('--offline', action='store_true', help='Skip the live sinfo query and use the bundled node snapshot.')
    parser.add_argument('--strict', action='store_true', help='Exit non-zero on warnings as well as errors.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    validate_sbatch(**vars(args))
