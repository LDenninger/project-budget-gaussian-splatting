#!/usr/bin/env python3
"""Self-contained Bitbucket pull-request reader for the repository of the current directory.

Lists open pull requests, or fetches one PR's description and threaded comments, by calling the
Bitbucket Cloud REST API directly with the Python standard library only. Authentication uses the
`BITBUCKET_API_TOKEN` environment variable together with the git `user.email`. The workspace and
the repository slug both come from the `origin` remote URL, so the script works from any directory
inside the repo, including subdirectories and git worktrees.

Usage:
    python fetch_bitbucket_pr.py                       # list all open PRs
    python fetch_bitbucket_pr.py --pr 13               # fetch PR #13 with its unresolved comments
    python fetch_bitbucket_pr.py --pr 13 --include-resolved
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any, NoReturn

BASE_URL = 'https://api.bitbucket.org/2.0/repositories'
ATLASSIAN_TOKEN_URL = 'https://id.atlassian.com/manage-profile/security/api-tokens'
TOKEN_ENV_VAR = 'BITBUCKET_API_TOKEN'


#---------------------------------------------------------------------
# Repository and credential discovery
#---------------------------------------------------------------------

def fail(message: str) -> NoReturn:
    """Print an error to stderr and exit with a non-zero status."""
    print(f'error: {message}', file=sys.stderr)
    raise SystemExit(1)


def get_git_email() -> str | None:
    """Return the git `user.email`, or None if it is not configured."""
    try:
        result = subprocess.run(['git', 'config', 'user.email'], capture_output=True, text=True, check=True)
        return result.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_repo_location() -> tuple[str, str] | None:
    """Return the Bitbucket workspace and repository slug taken from the `origin` remote URL.

    Returns:
        A `(workspace, repo_slug)` pair, or None when there is no `origin` remote or its URL
        carries too few path segments to name both.
    """
    try:
        result = subprocess.run(['git', 'remote', 'get-url', 'origin'], capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    remote_url = result.stdout.strip().removesuffix('/').removesuffix('.git')
    # Mapping ':' onto '/' folds SCP-style SSH (host:workspace/repo) onto the HTTPS form
    # (host/workspace/repo), so the last two segments are the workspace and the slug in both.
    segments = [segment for segment in remote_url.replace(':', '/').split('/') if segment]
    if len(segments) < 3:
        return None
    return segments[-2], segments[-1]


def build_auth_header(email: str, api_token: str) -> dict[str, str]:
    """Return an HTTP Basic Authorization header for the Bitbucket API."""
    token = base64.b64encode(f'{email}:{api_token}'.encode()).decode()
    return {'Authorization': f'Basic {token}'}


#---------------------------------------------------------------------
# Bitbucket API access
#---------------------------------------------------------------------

def http_get_json(url: str, auth_header: dict[str, str]) -> dict[str, Any]:
    """Perform an authenticated GET and return the parsed JSON body.

    Raises:
        SystemExit: On an authentication failure (401), a missing resource (404), or any other
            transport/HTTP error, after printing a caller-facing message.
    """
    request = urllib.request.Request(url, headers=auth_header)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            fail(f'authentication failed. Check git user.email and {TOKEN_ENV_VAR} ({ATLASSIAN_TOKEN_URL})')
        if exc.code == 404:
            fail('not found. Check the PR id, and that the workspace and repository slug match the Bitbucket repo')
        fail(f'API error {exc.code}: {exc.reason}')
    except urllib.error.URLError as exc:
        fail(f'network error: {exc.reason}')


def fetch_paginated(url: str, auth_header: dict[str, str]) -> list[dict[str, Any]]:
    """Return all `values` entries across every page of a paginated Bitbucket endpoint."""
    values: list[dict[str, Any]] = []
    next_url: str | None = url
    while next_url:
        data = http_get_json(next_url, auth_header)
        values.extend(data.get('values', []))
        next_url = data.get('next')
    return values


#---------------------------------------------------------------------
# Comment threading and Markdown rendering
#---------------------------------------------------------------------

def thread_comments(comments: list[dict[str, Any]], include_resolved: bool) -> list[dict[str, Any]]:
    """Group flat comments into sorted top-level threads with nested replies.

    Args:
        comments: Raw comment dicts from the Bitbucket API.
        include_resolved: When False, drop threads marked resolved.

    Returns:
        Top-level comments sorted by creation time, each carrying a `replies` list.
    """
    by_id = {comment['id']: comment for comment in comments}
    for comment in comments:
        comment['replies'] = []

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

    if include_resolved:
        return top_level
    return [
        comment
        for comment in top_level
        if not (comment.get('resolved', False) or comment.get('resolution') is not None)
    ]


def render_pr_list(prs: list[dict[str, Any]], workspace: str, repo_slug: str) -> str:
    """Render the open-PR list as Markdown for the user to choose from.

    Args:
        prs: Open pull requests as returned by the Bitbucket API.
        workspace: Bitbucket workspace owning the repository, used in the heading.
        repo_slug: Repository slug, used in the heading.

    Returns:
        A Markdown bullet list of the open PRs, newest id first.
    """
    if not prs:
        return f'No open pull requests in {workspace}/{repo_slug}.'

    lines = [f'# Open pull requests in {workspace}/{repo_slug}\n']
    for pull_request in sorted(prs, key=lambda item: item.get('id', 0), reverse=True):
        author = pull_request.get('author', {}).get('display_name', 'Unknown')
        source_branch = pull_request.get('source', {}).get('branch', {}).get('name', '?')
        dest_branch = pull_request.get('destination', {}).get('branch', {}).get('name', '?')
        link = pull_request.get('links', {}).get('html', {}).get('href', '')
        lines.append(
            f'- **#{pull_request.get("id")}**: {pull_request.get("title", "")} '
            f'_(by {author}, `{source_branch}` → `{dest_branch}`)_  {link}'
        )
    return '\n'.join(lines)


def render_comment(comment: dict[str, Any], level: int, lines: list[str]) -> None:
    """Append a single comment (and its replies) to `lines`, indented by reply depth."""
    if comment.get('deleted', False):
        return

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
        else:
            lines.append(f'{prefix}**Context:** General PR Comment')

    lines.append(f'\n{prefix}{content}\n')
    lines.append(f'{prefix}---\n')

    for reply in comment.get('replies', []):
        render_comment(reply, level + 1, lines)


def render_pr_markdown(pr_data: dict[str, Any], comments: list[dict[str, Any]]) -> str:
    """Render a PR's header, description, and threaded comments as Markdown."""
    title = pr_data.get('title', 'Unknown Title')
    pr_id = pr_data.get('id', 'Unknown ID')
    author = pr_data.get('author', {}).get('display_name', 'Unknown Author')
    state = pr_data.get('state', 'Unknown State')
    link = pr_data.get('links', {}).get('html', {}).get('href', '#')
    description = pr_data.get('description', '')

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

def fetch_bitbucket_pr(pr: int | None, include_resolved: bool) -> None:
    """List open PRs, or print one PR's details and comments, as Markdown on stdout.

    Args:
        pr: PR id to fetch. When None, all open PRs are listed instead.
        include_resolved: When True, resolved comment threads are kept in the output.
    """
    api_token = os.environ.get(TOKEN_ENV_VAR)
    if not api_token:
        fail(f'{TOKEN_ENV_VAR} is not set. Generate a scoped API token at {ATLASSIAN_TOKEN_URL}')

    email = get_git_email()
    if not email:
        fail('git user.email is not configured')

    location = get_repo_location()
    if location is None:
        fail('cannot determine the Bitbucket workspace and repository. Run inside a git repo whose `origin` remote points at Bitbucket')
    workspace, repo_slug = location

    auth_header = build_auth_header(email, api_token)
    repo_url = f'{BASE_URL}/{workspace}/{repo_slug}'

    if pr is None:
        prs = fetch_paginated(f'{repo_url}/pullrequests?state=OPEN', auth_header)
        print(render_pr_list(prs, workspace, repo_slug))
        return

    pr_data = http_get_json(f'{repo_url}/pullrequests/{pr}', auth_header)
    raw_comments = fetch_paginated(f'{repo_url}/pullrequests/{pr}/comments', auth_header)
    comments = thread_comments(raw_comments, include_resolved=include_resolved)
    print(render_pr_markdown(pr_data, comments))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='List open Bitbucket PRs or fetch one PR with comments.')
    parser.add_argument('--pr', type=int, default=None, help='PR id to fetch. Omit to list all open PRs')
    parser.add_argument(
        '--include-resolved', dest='include_resolved', action='store_true', help='Include resolved comment threads'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    fetch_bitbucket_pr(**vars(args))
