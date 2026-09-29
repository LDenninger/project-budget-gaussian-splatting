#!/usr/bin/env python3
"""SessionStart hook — publish PROJECT_ROOT and a snapshot of the project's moving parts.

`CLAUDE.md` and the rule files are static. What a session actually needs on top
of them is what exists *right now*: which workspaces are checked out and on
which branch, which specs and plans have been written, what sits in
`references/`. This hook builds that snapshot once per session.

It publishes the root two ways:

  1. `env.PROJECT_ROOT` in `<root>/.claude/settings.local.json`, so
     `$PROJECT_ROOT` resolves inside Bash calls. Claude Code applies a saved
     `env` change to the running session.
  2. The first line of `additionalContext`, so file tools (Write, Edit, Glob),
     which perform no shell expansion, have the literal path.

The settings file is written only when the session started at the root itself,
so a session running inside a foreign repository under `workspaces/` is never
written to.

Resolution, in order:

  1. Nearest ancestor of the session's cwd holding both `docs/` and `workspaces/`.
  2. The git top level of the cwd.
  3. The cwd itself.

Stdlib only. Fails open: any exception prints nothing and exits 0.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

MAX_WORKSPACES = 10
MAX_DOCUMENTS = 6
MAX_REFERENCES = 12
GIT_TIMEOUT_S = 2

#---------------------------------------------------------------------
# Resolution
#---------------------------------------------------------------------


def find_project_root(session_dir: Path) -> Path:
    """Return the directory whose `docs/` holds the shared specs, plans and devlogs.

    Args:
        session_dir: Directory the session started in.

    Returns:
        Nearest ancestor holding both `docs/` and `workspaces/`; the git top
        level when there is none; otherwise `session_dir` unchanged.
    """
    for candidate in [session_dir, *session_dir.parents]:
        if (candidate / 'docs').is_dir() and (candidate / 'workspaces').is_dir():
            return candidate

    top_level = run_git(session_dir, ['rev-parse', '--show-toplevel'])
    return Path(top_level).resolve() if top_level else session_dir


def run_git(repo_dir: Path, args: list[str]) -> str:
    """Run a git command in `repo_dir` and return its stripped stdout.

    Args:
        repo_dir: Directory to run in. Need not be a repository.
        args: Git arguments after `git -C <repo_dir>`.

    Returns:
        Stripped stdout, or an empty string on any non-zero exit, timeout or
        missing git.
    """
    try:
        completed = subprocess.run(
            ['git', '-C', str(repo_dir), *args],
            capture_output=True, text=True, timeout=GIT_TIMEOUT_S, check=False,
        )
    except Exception:
        return ''
    return completed.stdout.strip() if completed.returncode == 0 else ''


#---------------------------------------------------------------------
# Environment
#---------------------------------------------------------------------


def write_env_setting(session_dir: Path, project_root: Path) -> None:
    """Set `env.PROJECT_ROOT` in the project's local settings file.

    Writes only when the session started at `project_root`, so a session inside
    a repository under `workspaces/` never has settings written into it. Other
    keys are preserved, and an unchanged value is left alone so the file's mtime
    does not move on every session.

    Args:
        session_dir: Directory whose `.claude/settings.local.json` Claude Code reads.
        project_root: Resolved project root, written as the value.
    """
    if session_dir != project_root:
        return

    settings_path = project_root / '.claude' / 'settings.local.json'
    settings: dict = {}
    if settings_path.is_file():
        try:
            settings = json.loads(settings_path.read_text())
        except json.JSONDecodeError:
            return                            # hand-edited and broken: leave it to its owner

    env_block = settings.setdefault('env', {})
    if env_block.get('PROJECT_ROOT') == str(project_root):
        return

    env_block['PROJECT_ROOT'] = str(project_root)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + '\n')


#---------------------------------------------------------------------
# Snapshot sections
#---------------------------------------------------------------------


def describe_age(path: Path, now: float) -> str:
    """Return the file's modification age as a terse token such as `today` or `12d`.

    Args:
        path: File to stat.
        now: Reference timestamp, seconds since the epoch.

    Returns:
        `today`, `<N>d` under a year, else `<N>mo`.
    """
    try:
        days = int((now - path.stat().st_mtime) // 86400)
    except OSError:
        return '?'
    if days <= 0:
        return 'today'
    return f'{days}d' if days < 365 else f'{days // 30}mo'


def describe_workspaces(workspaces_dir: Path) -> list[str]:
    """Return one line per checked-out workspace, with its branch and dirty count.

    Args:
        workspaces_dir: The project's `workspaces/` directory.

    Returns:
        Lines of the form `  <name>  [<branch>, <state>]`, at most
        `MAX_WORKSPACES` of them plus a `+N more` line. Empty when nothing is
        checked out.
    """
    entries = sorted(entry for entry in workspaces_dir.iterdir() if entry.is_dir() and not entry.name.startswith('.'))
    lines: list[str] = []
    for entry in entries[:MAX_WORKSPACES]:
        branch = run_git(entry, ['branch', '--show-current'])
        if not branch:
            lines.append(f'  {entry.name}  [not a git repo]')
            continue
        dirty = len([line for line in run_git(entry, ['status', '--porcelain']).splitlines() if line])
        state = f'{dirty} uncommitted' if dirty else 'clean'
        lines.append(f'  {entry.name}  [{branch}, {state}]')

    if len(entries) > MAX_WORKSPACES:
        lines.append(f'  +{len(entries) - MAX_WORKSPACES} more')
    return lines


def describe_documents(documents_dir: Path, now: float) -> list[str]:
    """Return one line per document, newest first.

    Args:
        documents_dir: A directory of markdown documents (`docs/specs`, `docs/plans`).
        now: Reference timestamp for the age column.

    Returns:
        Lines of the form `  <file name>  (<age>)`, at most `MAX_DOCUMENTS` of
        them plus a `+N more` line. Empty when the directory holds no files.
    """
    files = sorted(
        (entry for entry in documents_dir.iterdir() if entry.is_file() and not entry.name.startswith('.')),
        key=lambda entry: entry.stat().st_mtime,
        reverse=True,
    )
    lines = [f'  {entry.name}  ({describe_age(entry, now)})' for entry in files[:MAX_DOCUMENTS]]
    if len(files) > MAX_DOCUMENTS:
        lines.append(f'  +{len(files) - MAX_DOCUMENTS} more')
    return lines


def describe_references(references_dir: Path) -> list[str]:
    """Return the contents of `references/` as one comma-joined line.

    Args:
        references_dir: The project's `references/` directory.

    Returns:
        A single indented line naming each entry, directories suffixed with `/`,
        at most `MAX_REFERENCES` names plus a `+N more` suffix. Empty when the
        directory holds nothing.
    """
    entries = sorted(entry for entry in references_dir.iterdir() if not entry.name.startswith('.'))
    if not entries:
        return []
    names = [f'{entry.name}/' if entry.is_dir() else entry.name for entry in entries[:MAX_REFERENCES]]
    suffix = f', +{len(entries) - MAX_REFERENCES} more' if len(entries) > MAX_REFERENCES else ''
    return [f'  {", ".join(names)}{suffix}']


#---------------------------------------------------------------------
# Assembly
#---------------------------------------------------------------------


def build_context(project_root: Path, session_dir: Path) -> str:
    """Return the session's starting context: the root, then what currently exists.

    Args:
        project_root: Resolved project root.
        session_dir: Directory the session started in, named only when it differs
            from the root, which is the case a relative `docs/` path would get wrong.

    Returns:
        Text for `hookSpecificOutput.additionalContext`.
    """
    now = time.time()
    lines = [
        f'PROJECT_ROOT={project_root}',
        f'Specs, plans and devlogs live at {project_root}/docs/{{specs,plans,devlog}} — '
        'address them by absolute path, never through a worktree\'s own docs/ '
        '(.claude/rules/docs-rules.md).',
    ]
    if session_dir != project_root:
        lines.append(f'This session started in {session_dir}, whose own docs/ is not the shared one.')

    #--- one section per directory, empties collapsed into a single trailing line ---
    sections: list[tuple[str, list[str]]] = []
    empty: list[str] = []
    for label, builder in (
        ('workspaces/', describe_workspaces),
        ('docs/specs/', lambda path: describe_documents(path, now)),
        ('docs/plans/', lambda path: describe_documents(path, now)),
        ('references/', describe_references),
    ):
        directory = project_root / label.rstrip('/')
        if not directory.is_dir():
            continue
        body = builder(directory)
        if body:
            sections.append((label, body))
        else:
            empty.append(label)

    for label, body in sections:
        lines.append('')
        lines.append(label)
        lines.extend(body)

    if empty:
        lines.append('')
        lines.append(f'Empty: {", ".join(empty)}')
    return '\n'.join(lines)


def main() -> int:
    """Emit the SessionStart payload. Any failure is silent and non-blocking."""
    try:
        raw_input = sys.stdin.read() if not sys.stdin.isatty() else ''
        payload = json.loads(raw_input) if raw_input.strip().startswith('{') else {}
        session_dir = Path(payload.get('cwd') or Path.cwd()).resolve()
        project_root = find_project_root(session_dir)
        write_env_setting(session_dir, project_root)
        print(json.dumps({
            'hookSpecificOutput': {
                'hookEventName': 'SessionStart',
                'additionalContext': build_context(project_root, session_dir),
            }
        }))
    except Exception:
        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
