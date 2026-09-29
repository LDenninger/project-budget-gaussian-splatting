#!/bin/bash
# Claude Code status line, shipped with claude-env and symlinked into every
# project's .claude/ by `claude-env activate` (see envs/base/settings.json).
#
# Renders: <cwd>[<branch>] | <model>[<effort>] | ctx <bar> <pct> (<used>/<size>)
#          | 5h[<time>] <bar> <pct> | 7d[<time>] <bar> <pct> | <session cost>
#
# Reads the status-line payload as JSON on stdin. Every segment degrades
# gracefully: a field the payload does not carry drops its whole segment rather
# than rendering an empty bracket or a fabricated value.
#
# Requires `jq` to parse the payload. Without it the line explains itself once
# and exits 0 — a missing status line never fails the session.

command -v jq >/dev/null 2>&1 || { printf '%s\n' "claude-env: install jq for the status line"; exit 0; }

input=$(cat)

#---------------------------------------------------------------------
# colors (dim variants, safe against the dimmed status-line rendering)
#---------------------------------------------------------------------
RESET=$'\033[0m'
GRAY=$'\033[2;37m'
CYAN=$'\033[2;36m'
BLUE=$'\033[2;34m'
GREEN=$'\033[2;32m'
YELLOW=$'\033[2;33m'
MAGENTA=$'\033[2;35m'

#---------------------------------------------------------------------
# cwd + git branch
#---------------------------------------------------------------------
cwd=$(echo "$input" | jq -r '.workspace.current_dir // .cwd // empty')
cwd_seg=""
if [ -n "$cwd" ]; then
    cwd_name=$(basename "$cwd")
    branch=""
    if git -C "$cwd" --no-optional-locks rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        branch=$(git -C "$cwd" --no-optional-locks branch --show-current 2>/dev/null)
        [ -z "$branch" ] && branch=$(git -C "$cwd" --no-optional-locks rev-parse --short HEAD 2>/dev/null)
    fi
    cwd_seg="${CYAN}${cwd_name}${RESET}"
    [ -n "$branch" ] && cwd_seg="${cwd_seg}${GRAY}[${MAGENTA}${branch}${GRAY}]${RESET}"
fi

#---------------------------------------------------------------------
# model + effort
#---------------------------------------------------------------------
model=$(echo "$input" | jq -r '.model.display_name // empty' | sed -E 's/[[:space:]]*\([0-9]+[A-Za-z]*[[:space:]]*[Cc]ontext\)[[:space:]]*$//')
effort=$(echo "$input" | jq -r '.effort.level // empty')
model_seg=""
if [ -n "$model" ]; then
    model_seg="${BLUE}${model}${RESET}"
    [ -n "$effort" ] && model_seg="${model_seg}${GRAY}[${YELLOW}${effort}${GRAY}]${RESET}"
fi

#---------------------------------------------------------------------
# context window: progress bar colored by token category, with pct + totals
#---------------------------------------------------------------------
fmt_k() {
    awk -v n="$1" 'BEGIN { if (n >= 1000) printf "%.1fk", n / 1000; else printf "%d", n }'
}

ctx_size=$(echo "$input" | jq -r '.context_window.context_window_size // 0')
cache_read=$(echo "$input" | jq -r '.context_window.current_usage.cache_read_input_tokens // 0')
cache_write=$(echo "$input" | jq -r '.context_window.current_usage.cache_creation_input_tokens // 0')
in_tokens=$(echo "$input" | jq -r '.context_window.current_usage.input_tokens // 0')
out_tokens=$(echo "$input" | jq -r '.context_window.current_usage.output_tokens // 0')
used_pct=$(echo "$input" | jq -r '.context_window.used_percentage // empty')

bar_width=20
if [ "${ctx_size:-0}" -gt 0 ] 2>/dev/null && [ -n "$used_pct" ]; then
    #--- fill the bar segment by segment out of a shared budget, so the four
    #--- filled widths can never sum past bar_width (a >100% payload saturates
    #--- the bar instead of overflowing the status line)
    budget=$bar_width
    take_w() {
        # Set seg_w to the bar width for a token count, capped at the leftover
        # budget (assigns globals: must not run in a command-substitution subshell).
        seg_w=$(( $1 * bar_width / ctx_size ))
        [ "$seg_w" -gt "$budget" ] && seg_w=$budget
        budget=$(( budget - seg_w ))
    }
    take_w "$cache_read";   read_w=$seg_w
    take_w "$cache_write";  write_w=$seg_w
    take_w "$in_tokens";    in_w=$seg_w
    take_w "$out_tokens";   out_w=$seg_w
    free_w=$budget

    bar=""
    for ((ii = 0; ii < read_w; ii++)); do bar="${bar}${GRAY}▓${RESET}"; done
    for ((ii = 0; ii < write_w; ii++)); do bar="${bar}${BLUE}▓${RESET}"; done
    for ((ii = 0; ii < in_w; ii++)); do bar="${bar}${CYAN}▓${RESET}"; done
    for ((ii = 0; ii < out_w; ii++)); do bar="${bar}${GREEN}▓${RESET}"; done
    for ((ii = 0; ii < free_w; ii++)); do bar="${bar}${GRAY}░${RESET}"; done

    used_total=$(( cache_read + cache_write + in_tokens + out_tokens ))
    used_pct_int=$(awk -v p="$used_pct" 'BEGIN { printf "%.0f", p }')
    ctx_seg="[${bar}] ${YELLOW}${used_pct_int}%${RESET} ($(fmt_k "$used_total")/$(fmt_k "$ctx_size"))"
else
    ctx_seg="${GRAY}n/a${RESET}"
fi
ctx_seg="ctx ${ctx_seg}"

#---------------------------------------------------------------------
# rate limits: 5-hour session window and 7-day weekly window
# Each window exposes `used_percentage` (0-100) and `resets_at` (UNIX epoch
# seconds), so render a compact usage bar next to the countdown (bar omitted if
# the percentage is ever absent, falling back to the countdown alone).
#
# `resets_at` is only ever consumed as epoch SECONDS. Anything else (an ISO-8601
# string, a millisecond epoch) is not converted or guessed at — the window's
# segment is dropped whole, per the graceful-degradation contract.
#---------------------------------------------------------------------
now_epoch=$(date +%s)
rl_bar_width=7

seconds_until() {
    # Set `remain` to the seconds until an epoch-seconds `resets_at` ($1), and
    # return non-zero if $1 is not a plausible epoch-seconds value for a window
    # of $2 seconds (non-numeric, or so far out it must be another unit).
    local resets_at="$1" window="$2"
    [[ "$resets_at" =~ ^[0-9]+$ ]] || return 1
    remain=$(( resets_at - now_epoch ))
    [ "$remain" -lt 0 ] && remain=0
    [ "$remain" -gt $(( window * 2 )) ] && return 1      # e.g. a millisecond epoch
    return 0
}

render_rl_bar() {
    # Render a small filled/empty bar for a 0-100 percentage.
    local pct="$1"
    local filled
    filled=$(awk -v p="$pct" -v w="$rl_bar_width" 'BEGIN { n = int((p / 100) * w + 0.5); if (n > w) n = w; if (n < 0) n = 0; print n }')
    local empty=$(( rl_bar_width - filled ))
    local bar=""
    for ((ii = 0; ii < filled; ii++)); do bar="${bar}${GREEN}▓${RESET}"; done
    for ((ii = 0; ii < empty; ii++)); do bar="${bar}${GRAY}░${RESET}"; done
    echo "$bar"
}

five_pct=$(echo "$input" | jq -r '.rate_limits.five_hour.used_percentage // empty')
five_resets=$(echo "$input" | jq -r '.rate_limits.five_hour.resets_at // empty')
five_seg=""
if seconds_until "$five_resets" 18000; then
    hh=$(( remain / 3600 ))
    mm=$(( (remain % 3600) / 60 ))
    countdown=$(printf '%02d:%02d' "$hh" "$mm")
    if [ -n "$five_pct" ]; then
        pct_int=$(awk -v p="$five_pct" 'BEGIN { printf "%.0f", p }')
        bar=$(render_rl_bar "$five_pct")
        five_seg="${GREEN}5h${RESET}${GRAY}[${RESET}${countdown}${GRAY}]${RESET} ${bar} ${YELLOW}${pct_int}%${RESET}"
    else
        five_seg="${GREEN}5h[${countdown}]${RESET}"
    fi
fi

week_pct=$(echo "$input" | jq -r '.rate_limits.seven_day.used_percentage // empty')
week_resets=$(echo "$input" | jq -r '.rate_limits.seven_day.resets_at // empty')
week_seg=""
if seconds_until "$week_resets" 604800; then
    dd=$(( remain / 86400 ))
    hh=$(( (remain % 86400) / 3600 ))
    if [ "$dd" -gt 0 ]; then
        countdown=$(printf '%02d:%02d' "$dd" "$hh")
    else
        mm=$(( (remain % 3600) / 60 ))
        countdown=$(printf '%02d:%02d' "$hh" "$mm")
    fi
    if [ -n "$week_pct" ]; then
        pct_int=$(awk -v p="$week_pct" 'BEGIN { printf "%.0f", p }')
        bar=$(render_rl_bar "$week_pct")
        week_seg="${GREEN}7d${RESET}${GRAY}[${RESET}${countdown}${GRAY}]${RESET} ${bar} ${YELLOW}${pct_int}%${RESET}"
    else
        week_seg="${GREEN}7d[${countdown}]${RESET}"
    fi
fi

#---------------------------------------------------------------------
# total session api cost, straight off the payload (cost.total_cost_usd)
#---------------------------------------------------------------------
total_cost=$(echo "$input" | jq -r '.cost.total_cost_usd // empty')
cost_seg=""
if [ -n "$total_cost" ]; then
    cost_seg="${YELLOW}$(awk -v c="$total_cost" 'BEGIN { printf "$%.2f", c }')${RESET}"
fi

#---------------------------------------------------------------------
# assemble and print
#---------------------------------------------------------------------
segments=()
[ -n "$cwd_seg" ] && segments+=("$cwd_seg")
[ -n "$model_seg" ] && segments+=("$model_seg")
segments+=("$ctx_seg")
[ -n "$five_seg" ] && segments+=("$five_seg")
[ -n "$week_seg" ] && segments+=("$week_seg")
[ -n "$cost_seg" ] && segments+=("$cost_seg")

out=""
for seg in "${segments[@]}"; do
    if [ -z "$out" ]; then
        out="$seg"
    else
        out="${out}${GRAY} | ${RESET}${seg}"
    fi
done

printf '%s\n' "$out"
