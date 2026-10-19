#!/usr/bin/env zsh
# Upload of an HF distribution repository and its fragmentation check (the script of docs/release-runbook.md §2).
#
#     tools/release/hf-upload.zsh upload <repo-dir-name> [extra arguments for hf upload…]
#         uploads models/<repo-dir-name> to hdae/<repo-dir-name> (first reads the provenance of every container and refuses an undeclared-licence marker), then checks the fragmentation of every safetensors / krm right after
#     tools/release/hf-upload.zsh check <repo-dir-name>
#         tabulates the reconstruction term count of every safetensors / krm of a published repository (no upload)
#
# The log is appended to outputs/release/upload-<repo-dir-name>.log (outputs/ is not tracked by git).
#
# MUST: use the hf of tools/.venv (huggingface_hub 1.27 / hf_xet 1.6.0) — nix's hf (hf_xet 1.4.3)
# has no knob to disable global dedup (the 4th env), and hits on other repositories' xorbs fragment
# even a first upload (observed with siglip2 on 2026-09-04; mechanism in docs/research/2026-08-09-xet-fragmentation.md).
# MUST: move the shard-cache aside every time — when a shard fetched by a global dedup hit remains, the
# next upload hits it and inherits the fragmentation.
set -u

SELF=${0:A}
ROOT=${SELF:h:h:h}
HF=$ROOT/tools/.venv/bin/hf
OWNER=hdae

usage() {
  # Inside a function $0 is the function name, so use the script's real path taken at the top.
  sed -n '2,9p' "$SELF" | sed 's/^# \{0,1\}//'
  exit 2
}

[[ $# -ge 2 ]] || usage
MODE=$1; NAME=$2; shift 2
LOG=$ROOT/outputs/release/upload-$NAME.log
mkdir -p "$ROOT/outputs/release"
cd "$ROOT"

# The size limit (10 MiB) of a file that HF keeps as a plain git blob without LFS / xet. A file below it
# without x-xet-hash is not on xet = fragmentation does not apply, so it gets a SKIP row.
# A file at or above it should be on xet, so a missing hash is FAILED.
NON_XET_MAX_BYTES=$(( 10 * 1048576 ))

# Reconstruction terms table of every safetensors / krm (healthy is close to 1 xorb = 1 term; rule of thumb ≥10 MiB/term).
# ranges = the total number of elements of the response's fetch_info (xorb hash → array of fetch ranges) = the number of ranges actually fetched.
# A term that points at the same range many times, like an all-zero chunk, raises only terms and not ranges, so
# ranges tells whether many terms come from repeated references or from fetch ranges that are themselves fragmented.
# The verdict (MiB/term and FAILED / SKIP) stays on terms — ranges is a column for the reader and does not loosen the check.
# MUST: fail non-zero when there are 0 local mirror files. The usage claims "every safetensors / krm of a published
# repository" while what it enumerates is models/<repo>, so without a local mirror it loops 0 times and prints an empty table as success —
# "checked, no problem" and "checked nothing" would look the same (runbook §2 requires every file).
# MUST: keep the same reason at row granularity. Silently moving past a row whose hash or reconstruction could not be fetched
# prints a row with terms=0 → MiB/term = size (the healthiest value in the table), and an unchecked part looks like a pass.
# A row that could not be fetched prints `### FAILED <rel>`; the table is printed to the end, then the function returns non-zero.
fragmentation_table() {
  local tok cas casUrl access f rel headers hash recon stats statsCode terms ranges xorbs size failed=0
  # (N.OL) = empty array when nothing matches / regular files only / descending size (covers both the `ls -S` order and word splitting).
  local files=(models/$NAME/**/*.(safetensors|krm)(N.OL))
  if (( ${#files} == 0 )); then
    echo "### models/$NAME has not a single .safetensors / .krm (fragmentation not checked — place the mirror)"
    return 1
  fi
  # A failure to get the token fails every row for the same reason, so fail before the table instead of scattering per-row FAILED lines.
  if ! tok=$(curl -sSf "https://huggingface.co/api/models/$OWNER/$NAME/xet-read-token/main"); then
    echo "### FAILED xet-read-token could not be obtained ($OWNER/$NAME) — fragmentation not checked"
    return 1
  fi
  if ! cas=$(print -r -- "$tok" | node -e 'const t=JSON.parse(require("node:fs").readFileSync(0,"utf8")); if (typeof t.casUrl!=="string"||typeof t.accessToken!=="string") process.exit(1); console.log(t.casUrl+" "+t.accessToken)'); then
    echo "### FAILED xet-read-token response has no casUrl / accessToken ($OWNER/$NAME) — fragmentation not checked"
    return 1
  fi
  casUrl=${cas%% *}; access=${cas##* }
  for f in $files; do
    rel=${f#models/$NAME/}
    size=$(stat -c %s "$f")
    if ! headers=$(curl -sSf -I -L "https://huggingface.co/$OWNER/$NAME/resolve/main/$rel"); then
      echo "### FAILED $rel HEAD failed (a part not on HF, or a fetch failure)"
      failed=1; continue
    fi
    hash=$(print -r -- "$headers" | grep -i '^x-xet-hash:' | awk '{print $2}' | tr -d '\r')
    if [[ -z $hash ]]; then
      if (( size < NON_XET_MAX_BYTES )); then
        echo "### SKIP $rel small file not on xet ($size bytes — fragmentation does not apply)"
      else
        echo "### FAILED $rel has no x-xet-hash ($(( size / 1048576 )) MiB — large enough that it should be on xet)"
        failed=1
      fi
      continue
    fi
    if ! recon=$(curl -sSf -H "Authorization: Bearer $access" "$casUrl/v1/reconstructions/$hash"); then
      echo "### FAILED $rel failed to fetch the reconstruction (hash $hash)"
      failed=1; continue
    fi
    # Exit code 1 = no terms (including a response that is not valid JSON) / 2 = fetch_info missing or of the wrong shape.
    # A row whose fetch_info cannot be read is FAILED too (a row with ranges=0 would look like a healthy row with few fetch ranges).
    stats=$(print -r -- "$recon" | node -e 'const t=JSON.parse(require("node:fs").readFileSync(0,"utf8")); if (!Array.isArray(t.terms)) process.exit(1); const fi=t.fetch_info; if (typeof fi!=="object"||fi===null||Array.isArray(fi)||!Object.values(fi).every(Array.isArray)) process.exit(2); console.log(t.terms.length+" "+Object.values(fi).reduce((n,r)=>n+r.length,0)+" "+new Set(t.terms.map(x=>x.hash)).size)')
    statsCode=$?
    if (( statsCode == 2 )); then
      echo "### FAILED $rel reconstruction response has no fetch_info (hash $hash)"
      failed=1; continue
    elif (( statsCode != 0 )); then
      echo "### FAILED $rel reconstruction response has no terms (hash $hash)"
      failed=1; continue
    fi
    terms=${stats%% *}; xorbs=${stats##* }; ranges=${${stats#* }%% *}
    if (( terms <= 0 )); then
      echo "### FAILED $rel reconstruction has 0 terms (hash $hash)"
      failed=1; continue
    fi
    printf '### fragmentation %-52s %5d MiB terms=%4d ranges=%4d xorbs=%3d MiB/term=%.1f\n' "$rel" $(( size / 1048576 )) $terms $ranges $xorbs $(( size / 1048576.0 / terms ))
  done
  return $failed
}

# Summarizes the `###` lines since this upload to the terminal (the same form on success and on failure).
summarize() {
  grep '^###' "$LOG" | tail -n +1 | awk -v s="### upload $NAME" 'index($0,s)==1{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}'
}

case $MODE in
  check)
    # The pipeline's exit code is tee's, so take the table's status from $pipestatus and return it
    # (so the caller can read that it failed on 0 files).
    fragmentation_table | tee -a "$LOG"
    exit ${pipestatus[1]}
    ;;
  upload)
    # MUST: before uploading, read provenance.license of every container (part 0 of .krm), and when an undeclared-licence
    # marker is present (NOASSERTION — including case variants and values such as unknown that do not identify the
    # redistribution terms), fail without uploading a single byte (ADR 0122 decision 6, guard 3 — closes on the containers'
    # content, not on the location; an experimental mirror copied by hand under models/ or a symlink is stopped too). It comes
    # before moving the shard-cache aside and before starting hf. An unreadable part 0 and a directory without a single
    # container fail too (what could not be checked must not look like a pass).
    node "${SELF:h}/container_license.ts" "models/$NAME" 2>&1 | tee -a "$LOG"
    if (( ${pipestatus[1]} != 0 )); then
      echo "### FAILED pre-publication check: cannot confirm the containers' provenance / an undeclared marker is present (models/$NAME) — not uploading" | tee -a "$LOG"
      exit 1
    fi
    export HF_XET_DEDUPLICATION_MIN_N_CHUNKS_PER_RANGE=1000000
    export HF_XET_DEDUPLICATION_MIN_N_CHUNKS_PER_RANGE_HYSTERESIS_FACTOR=1.0
    export HF_XET_DEDUPLICATION_NRANGES_IN_STREAMING_FRAGMENTATION_ESTIMATOR=1
    export HF_XET_DEDUPLICATION_GLOBAL_DEDUP_QUERY_ENABLED=false
    for C in ~/.cache/huggingface/xet/*/shard-cache(N); do
      # MUST: do not upload when moving aside fails (the remaining shard-cache would be hit and the fragmentation inherited).
      # Chaining with AND-OR lets a failure through (the same with set -e — a failure anywhere but the last of a list is
      # ignored), so fail explicitly one step at a time.
      mv "$C" "$C.bak-$(date +%Y%m%dT%H%M%S)" || { echo "### failed to move the shard-cache aside: $C" | tee -a "$LOG"; exit 1 }
      mkdir -p "$C" || { echo "### failed to recreate the shard-cache: $C" | tee -a "$LOG"; exit 1 }
      echo "### shard-cache moved: $C" >> "$LOG"
    done
    echo "### upload $NAME $(date +%T) hf=$($HF version 2>/dev/null | tail -1)" >> "$LOG"
    # Time marker — the reference that lets the check below look only at the xet logs this upload wrote (created right before the upload).
    MARKER=$(mktemp) || { echo "### cannot create the time marker" | tee -a "$LOG"; exit 1 }
    $HF upload "$OWNER/$NAME" "models/$NAME" . --repo-type model "$@" >> "$LOG" 2>&1; rc=$?
    echo "### upload exit=$rc $(date +%T)" >> "$LOG"
    # MUST: look only at the logs this upload wrote (newer than the marker). Taking the newest one would, when this
    # hf wrote no log, read the previous successful log and pass the check. There is one log per process, but
    # when there are several, every one is checked.
    XET_LOGS=(~/.cache/huggingface/xet/logs/*(N.e['[[ $REPLY -nt $MARKER ]]']))
    rm -f "$MARKER"
    [[ $rc -ne 0 ]] && { tail -3 "$LOG"; exit $rc }
    # Confirm from the hf_xet logs that the 4th env was read and that there were 0 chunk queries to the CAS.
    # The check below fails on any of: 0 logs, any log without the disabled line, a non-zero total of queries.
    DISABLED=$(( ${#XET_LOGS} > 0 )); QUERIES=0; ENABLED=
    for L in $XET_LOGS; do
      ENABLED=$(grep -o 'global_dedup_query_enabled = [a-z]* ([a-z ]*)' "$L" | head -1)
      Q=$(grep -c 'Completed query_dedup' "$L")
      [[ $ENABLED == *"= false"* ]] || DISABLED=0
      QUERIES=$(( QUERIES + Q ))
      echo "### xet log $L $ENABLED query_dedup=$Q" >> "$LOG"
    done
    (( ${#XET_LOGS} > 0 )) || echo "### xet log no log from this upload" >> "$LOG"
    SHA=$(curl -sS "https://huggingface.co/api/models/$OWNER/$NAME/revision/main" | node -e 'const t=require("node:fs").readFileSync(0,"utf8"); console.log(JSON.parse(t).sha)')
    echo "### main sha $SHA" >> "$LOG"
    # Print the table before the verdict (keep the observed fragmentation even when failing). With 0 files the table side fails.
    fragmentation_table >> "$LOG" || { echo "### FAILED could not check fragmentation" >> "$LOG"; summarize; exit 1 }
    # MUST: do not only copy the disabling of global dedup (the 4th env) into the log — **make it a check**. When it is not
    # disabled, the upload hits other repositories' xorbs and the publication completes fragmented, and recovery is the
    # destructive procedure of deleting and recreating the repository (runbook §2 — the checked values use the same spelling as there).
    if (( ! DISABLED || QUERIES != 0 )); then
      echo "### FAILED global dedup is not disabled (logs from this upload: ${#XET_LOGS} / ${ENABLED:-no line} / query_dedup=$QUERIES) — is hf the one in tools/.venv (hf_xet 1.6.0 or later)?" >> "$LOG"
      summarize
      exit 1
    fi
    echo "### DONE $NAME $(date +%T)" >> "$LOG"
    summarize
    ;;
  *)
    usage
    ;;
esac
