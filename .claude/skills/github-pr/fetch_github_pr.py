#!/usr/bin/env python3
"""GitHub pull-request reader for the repository of the current working directory.

Lists open pull requests, or fetches one PR's description and threaded review comments, and prints
the result as Markdown on stdout. Every network call goes through the `gh` CLI, so authentication is
`gh`'s job via `gh auth login` and no token is read here. The repository is whatever `gh` resolves
from the working directory's `origin` remote, so subdirectories and git worktrees work too.

GitHub splits PR discussion across three REST streams, and this script merges all of them: issue
comments on the conversation tab, review summary bodies, and inline review comments anchored to a
file and line. Thread resolution is not exposed by those REST endpoints, so it is read separately
through the GraphQL `reviewThreads` connection and used to hide resolved threads by default.

Usage:
    python fetch_github_pr.py                       # list all open PRs
    python fetch_github_pr.py --pr 13               # fetch PR #13 with its unresolved comments
    python fetch_github_pr.py --pr 13 --include-resolved
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from typing import Any, NoReturn

GH_INSTALL_URL = 'https://cli.github.com'

RESOLVED_THREADS_QUERY = '''
query($owner: String!, $repo: String!, $number: Int!, $endCursor: String) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $endCursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          isResolved
          comments(first: 100) { nodes { databaseId } }
        }
      }
    }
  }
}
'''


#---------------------------------------------------------------------
# Process helpers and preconditions
#---------------------------------------------------------------------

def fail(message: str) -> NoReturn:
    """Print an error to stderr and exit with a non-zero status."""
    print(f'error: {message}', file=sys.stderr)
    raise SystemExit(1)


def run_gh(arguments: list[str], action: str) -> str:
    """Run a `gh` subcommand and return its stdout.

    Args:
        arguments: Argument list passed to `gh`, without the `gh` executable itself.
        action: Short description of the call, used in the error message when it fails.

    Returns:
        The captured stdout of the command.

    Raises:
        SystemExit: When `gh` is missing from PATH, or the command exits non-zero.
    """
    try:
        result = subprocess.run(['gh', *arguments], capture_output=True, text=True, check=True)
    except FileNotFoundError:
        fail(f'the `gh` CLI is not installed. Install it from {GH_INSTALL_URL}, then run `gh auth login`')
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or '').strip() or f'exit status {exc.returncode}'
        fail(f'{action} failed: {detail}')
    return result.stdout


def parse_json_stream(payload: str) -> list[Any]:
    """Decode the one-or-more JSON values `gh api --paginate` concatenates on stdout.

    A paginated call emits one JSON value per page rather than a single merged document, so the
    payload is decoded value by value and every list is flattened into the result.

    Args:
        payload: Raw stdout of a `gh api` call, holding one JSON value per fetched page.

    Returns:
        Every decoded value, with the elements of list-valued pages spliced in individually.
    """
    decoder = json.JSONDecoder()
    values: list[Any] = []
    index = 0
    while index < len(payload):
        if payload[index].isspace():
            index += 1
            continue
        value, index = decoder.raw_decode(payload, index)
        if isinstance(value, list):
            values.extend(value)
        else:
            values.append(value)
    return values


def check_preconditions() -> str:
    """Verify that `gh` is installed, authenticated, and pointed at a GitHub repository.

    Returns:
        The `owner/repo` slug of the repository resolved from the working directory.

    Raises:
        SystemExit: When `gh` is absent, unauthenticated, or the directory has no GitHub remote.
    """
    if shutil.which('gh') is None:
        fail(f'the `gh` CLI is not installed. Install it from {GH_INSTALL_URL}, then run `gh auth login`')

    auth = subprocess.run(['gh', 'auth', 'status'], capture_output=True, text=True)
    if auth.returncode != 0:
        fail('the `gh` CLI is not authenticated. Run `gh auth login` and retry')

    repo = subprocess.run(['gh', 'repo', 'view', '--json', 'nameWithOwner', '--jq', '.nameWithOwner'], capture_output=True, text=True)
    if repo.returncode != 0 or not repo.stdout.strip():
        fail('cannot determine the GitHub repository. Run inside a git repo whose `origin` remote points at GitHub')
    return repo.stdout.strip()


#---------------------------------------------------------------------
# GitHub data access
#---------------------------------------------------------------------

def fetch_open_prs() -> list[dict[str, Any]]:
    """Return the open pull requests of the current repository, with the fields the list needs."""
    fields = 'number,title,author,headRefName,baseRefName,url'
    payload = run_gh(['pr', 'list', '--state', 'open', '--limit', '200', '--json', fields], 'listing open pull requests')
    return json.loads(payload)


def fetch_pr(pr: int) -> dict[str, Any]:
    """Return one pull request's header fields and description body.

    Args:
        pr: Pull-request number to fetch.

    Returns:
        The decoded `gh pr view --json` object for that pull request.
    """
    fields = 'number,title,author,state,url,body'
    payload = run_gh(['pr', 'view', str(pr), '--json', fields], f'fetching PR #{pr}')
    return json.loads(payload)


def fetch_comment_streams(repo_slug: str, pr: int) -> list[dict[str, Any]]:
    """Return the issue comments, review summaries, and inline review comments of one PR.

    Args:
        repo_slug: Repository in `owner/repo` form.
        pr: Pull-request number.

    Returns:
        Raw API objects from all three streams, each tagged with a `_stream` key naming its origin.
    """
    streams = {
        'issue': f'repos/{repo_slug}/issues/{pr}/comments',
        'review': f'repos/{repo_slug}/pulls/{pr}/reviews',
        'inline': f'repos/{repo_slug}/pulls/{pr}/comments',
    }

    entries: list[dict[str, Any]] = []
    for stream_name, endpoint in streams.items():
        payload = run_gh(['api', '--paginate', endpoint], f'fetching {stream_name} comments for PR #{pr}')
        for entry in parse_json_stream(payload):
            if isinstance(entry, dict):
                entry['_stream'] = stream_name
                entries.append(entry)
    return entries


def fetch_resolved_comment_ids(repo_slug: str, pr: int) -> set[int]:
    """Return the ids of every review comment sitting in a resolved review thread.

    Resolution is absent from the REST comment endpoints, so it is read from the GraphQL
    `reviewThreads` connection, which is the only place GitHub exposes it.

    Args:
        repo_slug: Repository in `owner/repo` form.
        pr: Pull-request number.

    Returns:
        Database ids of the comments belonging to resolved threads. Empty when nothing is resolved.
    """
    owner, _, name = repo_slug.partition('/')
    arguments = [
        'api', 'graphql', '--paginate',
        '-f', f'query={RESOLVED_THREADS_QUERY}',
        # `-f` keeps owner and repo as strings. `-F` types the number, which GraphQL needs as Int.
        '-f', f'owner={owner}',
        '-f', f'repo={name}',
        '-F', f'number={pr}',
    ]
    payload = run_gh(arguments, f'reading resolved review threads for PR #{pr}')

    resolved_ids: set[int] = set()
    for page in parse_json_stream(payload):
        threads = (((page.get('data') or {}).get('repository') or {}).get('pullRequest') or {}).get('reviewThreads') or {}
        for thread in threads.get('nodes') or []:
            if not thread.get('isResolved'):
                continue
            for comment in (thread.get('comments') or {}).get('nodes') or []:
                if comment.get('databaseId') is not None:
                    resolved_ids.add(int(comment['databaseId']))
    return resolved_ids


#---------------------------------------------------------------------
# Comment normalisation and threading
#---------------------------------------------------------------------

def normalize_comment(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Map one raw API object from any of the three streams onto a single comment shape.

    Args:
        entry: A raw issue comment, review, or inline review comment, tagged with `_stream`.

    Returns:
        A comment dict carrying `id`, `user`, `created_on`, `content`, `inline` and `parent`, or
        None for an entry that holds no readable body, such as the empty container review that
        GitHub creates around a batch of inline comments.
    """
    body = (entry.get('body') or '').strip()
    if not body:
        return None

    stream = entry.get('_stream')
    created_on = entry.get('created_at') or entry.get('submitted_at') or 'Unknown Date'

    inline: dict[str, Any] | None = None
    if stream == 'inline' and entry.get('path'):
        inline = {'path': entry['path'], 'to': entry.get('line'), 'from': entry.get('original_line')}

    parent: dict[str, Any] | None = None
    if entry.get('in_reply_to_id') is not None:
        parent = {'id': entry['in_reply_to_id']}

    return {
        'id': entry.get('id'),
        'stream': stream,
        'state': entry.get('state'),
        'user': {'display_name': (entry.get('user') or {}).get('login', 'Unknown User')},
        'created_on': created_on,
        'content': {'raw': body},
        'inline': inline,
        'parent': parent,
        'replies': [],
    }


def thread_comments(comments: list[dict[str, Any]], resolved_ids: set[int]) -> list[dict[str, Any]]:
    """Group flat comments into sorted top-level threads with nested replies.

    Args:
        comments: Normalized comment dicts from all three streams.
        resolved_ids: Comment ids belonging to resolved review threads. Any top-level comment in
            this set is dropped along with its replies. Pass an empty set to keep everything.

    Returns:
        Top-level comments sorted by creation time, each carrying a `replies` list.
    """
    by_id = {comment['id']: comment for comment in comments}

    top_level: list[dict[str, Any]] = []
    for comment in comments:
        parent = comment.get('parent') or {}
        parent_id = parent.get('id')
        if parent_id is not None and parent_id in by_id:
            by_id[parent_id]['replies'].append(comment)
        else:
            top_level.append(comment)

    for comment in comments:
        comment['replies'].sort(key=lambda item: item.get('created_on', ''))
    top_level.sort(key=lambda item: item.get('created_on', ''))

    if not resolved_ids:
        return top_level
    return [comment for comment in top_level if comment.get('id') not in resolved_ids]


#---------------------------------------------------------------------
# Markdown rendering
#---------------------------------------------------------------------

def render_pr_list(prs: list[dict[str, Any]], repo_slug: str) -> str:
    """Render the open-PR list as Markdown for the user to choose from.

    Args:
        prs: Open pull requests as returned by `gh pr list --json`.
        repo_slug: Repository in `owner/repo` form, used in the heading.

    Returns:
        A Markdown bullet list of the open PRs, newest number first.
    """
    if not prs:
        return f'No open pull requests in {repo_slug}.'

    lines = [f'# Open pull requests in {repo_slug}\n']
    for pull_request in sorted(prs, key=lambda item: item.get('number', 0), reverse=True):
        author = (pull_request.get('author') or {}).get('login', 'Unknown')
        source_branch = pull_request.get('headRefName', '?')
        dest_branch = pull_request.get('baseRefName', '?')
        link = pull_request.get('url', '')
        lines.append(
            f'- **#{pull_request.get("number")}**: {pull_request.get("title", "")} '
            f'_(by {author}, `{source_branch}` → `{dest_branch}`)_  {link}'
        )
    return '\n'.join(lines)


def render_comment(comment: dict[str, Any], level: int, lines: list[str]) -> None:
    """Append a single comment (and its replies) to `lines`, indented by reply depth."""
    prefix = '> ' * level
    user = comment.get('user', {}).get('display_name', 'Unknown User')
    created_on = comment.get('created_on', 'Unknown Date')
    content = comment.get('content', {}).get('raw', '')

    lines.append(f'{prefix}### Comment by {user} ({created_on})')

    if level == 0:
        inline = comment.get('inline')
        if inline and inline.get('path'):
            location = f'**File:** `{inline["path"]}`'
            if inline.get('to'):
                location += f' **Line:** {inline["to"]}'
            elif inline.get('from'):
                location += f' **Line:** {inline["from"]} (original)'
            lines.append(f'{prefix}{location}')
        elif comment.get('stream') == 'review':
            lines.append(f'{prefix}**Context:** Review summary ({comment.get("state", "COMMENTED")})')
        else:
            lines.append(f'{prefix}**Context:** General PR Comment')

    lines.append(f'\n{prefix}{content}\n')
    lines.append(f'{prefix}---\n')

    for reply in comment.get('replies', []):
        render_comment(reply, level + 1, lines)


def render_pr_markdown(pr_data: dict[str, Any], comments: list[dict[str, Any]]) -> str:
    """Render a PR's header, description, and threaded comments as Markdown."""
    title = pr_data.get('title', 'Unknown Title')
    pr_id = pr_data.get('number', 'Unknown ID')
    author = (pr_data.get('author') or {}).get('login', 'Unknown Author')
    state = pr_data.get('state', 'Unknown State')
    link = pr_data.get('url', '#')
    description = (pr_data.get('body') or '').strip()

    lines = [
        f'# PR #{pr_id}: {title}\n',
        f'- **Author:** {author}',
        f'- **State:** {state}',
        f'- **Link:** {link}\n',
        '## Description\n',
        description if description else '_No description_',
        '\n',
        '## Comments\n',
    ]

    if not comments:
        lines.append('_No comments found._')
        return '\n'.join(lines)

    for comment in comments:
        render_comment(comment, 0, lines)
    return '\n'.join(lines)


#---------------------------------------------------------------------
# Entry point
#---------------------------------------------------------------------

def fetch_github_pr(pr: int | None, include_resolved: bool) -> None:
    """List open PRs, or print one PR's details and comments, as Markdown on stdout.

    Args:
        pr: PR number to fetch. When None, all open PRs are listed instead.
        include_resolved: When True, resolved review threads are kept in the output and the
            GraphQL resolution lookup is skipped.
    """
    repo_slug = check_preconditions()

    if pr is None:
        print(render_pr_list(fetch_open_prs(), repo_slug))
        return

    pr_data = fetch_pr(pr)
    raw_entries = fetch_comment_streams(repo_slug, pr)
    normalized = [comment for comment in map(normalize_comment, raw_entries) if comment is not None]
    resolved_ids = set() if include_resolved else fetch_resolved_comment_ids(repo_slug, pr)
    print(render_pr_markdown(pr_data, thread_comments(normalized, resolved_ids)))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='List open GitHub PRs or fetch one PR with comments.')
    parser.add_argument('--pr', type=int, default=None, help='PR number to fetch. Omit to list all open PRs')
    parser.add_argument(
        '--include-resolved', dest='include_resolved', action='store_true', help='Include resolved review threads'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    fetch_github_pr(**vars(args))
