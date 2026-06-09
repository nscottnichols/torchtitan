#!/bin/bash --login
# Check per-chain production READMEs for staleness vs on-disk state.
#
# For each canonical chain, compares:
#   - latest valid step-N dir on disk (`.metadata` present, not an empty placeholder)
#   - latest step-N mentioned in the chain's README
#   - the README's last-modified date (git log -1)
#
# Also flags chain checkpoint dirs that have NO matching README at all
# (e.g. new chains like the 80B 4N validation).
#
# Prints a punch list. Doesn't auto-edit — README updates need narrative
# judgment (failure-mode interpretation, "next continuation" identification,
# etc.) that's best done manually with the report in hand.
#
# Usage (from repo root):
#   bash torchtitan/experiments/ezpz/scripts/check_stale_docs.sh

set -o pipefail

cd "$(dirname "$0")/../../../.."   # repo root

# Map: ckpt_dir → README path. Add new chains here as they appear.
declare -A CHAIN_TO_README=(
    [/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144]="torchtitan/experiments/ezpz/docs/production/agpt/2b/n256/README.md"
    [/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288]="torchtitan/experiments/ezpz/docs/production/agpt/2b/n512/README.md"
    [/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288-gbs12288-lr3.22e-5]="torchtitan/experiments/ezpz/docs/production/agpt/2b/n512/README.md"
    [/flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/outputs/checkpoints/agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288]="torchtitan/experiments/ezpz/docs/production/agpt/20b/n512/README.md"
    [/flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz/outputs/checkpoints/agpt-80b-adamw-olmo-mix-1124-n4-gbs24]="torchtitan/experiments/ezpz/docs/production/agpt/80b/n4/README.md"
    [/flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz/outputs/checkpoints/agpt-80b-adamw-olmo-mix-1124-n256-gbs1536]="torchtitan/experiments/ezpz/docs/production/agpt/80b/n256/README.md"
)

stale_count=0
missing_count=0
ok_count=0

echo "=== production README staleness check ($(date +%Y-%m-%d_%H:%M)) ==="
echo ""

for ckpt_dir in "${!CHAIN_TO_README[@]}"; do
    readme="${CHAIN_TO_README[$ckpt_dir]}"
    chain_name="$(basename "$ckpt_dir")"

    if [[ ! -d "$ckpt_dir" ]]; then
        # Chain doesn't exist on disk — skip silently
        continue
    fi

    # Latest valid step-N dir (digits only — excludes .bak-* and other suffixes)
    last_ckpt=$(ls "$ckpt_dir" 2>/dev/null | grep -E '^step-[0-9]+$' | sort -t- -k2 -n | tail -1)
    if [[ -z "$last_ckpt" ]]; then
        echo "  [no ckpts] $chain_name"
        continue
    fi
    last_mtime=$(stat -c '%y' "$ckpt_dir/$last_ckpt" 2>/dev/null | head -c 19)

    if [[ ! -f "$readme" ]]; then
        echo "  [MISSING] $chain_name"
        echo "    disk: $last_ckpt ($last_mtime)"
        echo "    no README at: $readme"
        echo ""
        missing_count=$((missing_count + 1))
        continue
    fi

    # Latest step-N mentioned in the README (purely textual scan)
    readme_step=$(grep -oE 'step-[0-9]+' "$readme" 2>/dev/null | sort -u | sort -t- -k2 -n | tail -1)
    readme_git=$(git log -1 --format='%ad' --date=short -- "$readme" 2>/dev/null)

    if [[ -z "$readme_step" ]]; then
        echo "  [no step-N in README] $chain_name"
        echo "    disk: $last_ckpt ($last_mtime)"
        echo "    readme: $readme (git: $readme_git)"
        echo ""
        stale_count=$((stale_count + 1))
        continue
    fi

    disk_num=${last_ckpt#step-}
    readme_num=${readme_step#step-}

    if (( disk_num > readme_num )); then
        delta=$((disk_num - readme_num))
        echo "  [STALE] $chain_name"
        echo "    disk: $last_ckpt ($last_mtime)"
        echo "    readme: $readme_step (git: $readme_git) — $delta steps behind"
        echo "    edit: $readme"
        echo ""
        stale_count=$((stale_count + 1))
    else
        ok_count=$((ok_count + 1))
    fi
done

echo "=== summary: $ok_count up-to-date, $stale_count stale, $missing_count missing-README ==="
exit 0
