#!/usr/bin/env zsh
# HF 配布リポのアップロードと断片化検証（docs/release-runbook.md §2 の台本）。
#
#     tools/release/hf-upload.zsh upload <repo-dir-name> [hf upload の追加引数…]
#         models/<repo-dir-name> を hdae/<repo-dir-name> へ上げ、直後に全 safetensors の断片化を検証する
#     tools/release/hf-upload.zsh check <repo-dir-name>
#         公開済みリポの全 safetensors について reconstruction の term 数を表にする（アップロードしない）
#
# ログは outputs/release/upload-<repo-dir-name>.log へ追記する（outputs/ は git 追跡外）。
#
# MUST: hf は tools/.venv のもの（huggingface_hub 1.27 / hf_xet 1.6.0）を使う — nix の hf（hf_xet 1.4.3）
# には global dedup の停止ノブ（4 本目の env）が無く、他リポの xorb へのヒットで初回アップロードでも
# 断片化する（2026-09-04 siglip2 で実測・機序は docs/research/2026-08-09-xet-fragmentation.md）。
# MUST: shard-cache は毎回退避する — global dedup のヒットで取り寄せた shard が残っていると、次の
# アップロードがそれを引き当てて断片化を継承する。
set -u

SELF=${0:A}
ROOT=${SELF:h:h:h}
HF=$ROOT/tools/.venv/bin/hf
OWNER=hdae

usage() {
  # 関数内の $0 は関数名になるので、冒頭で取ったスクリプトの実パスを使う。
  sed -n '2,9p' "$SELF" | sed 's/^# \{0,1\}//'
  exit 2
}

[[ $# -ge 2 ]] || usage
MODE=$1; NAME=$2; shift 2
LOG=$ROOT/outputs/release/upload-$NAME.log
mkdir -p "$ROOT/outputs/release"
cd "$ROOT"

# 全 safetensors の reconstruction terms 表（healthy なら 1 xorb = 1 term に近い・目安 ≥10 MiB/term）。
fragmentation_table() {
  local tok cas casUrl access f rel hash stats terms xorbs size
  tok=$(curl -sS "https://huggingface.co/api/models/$OWNER/$NAME/xet-read-token/main")
  cas=$(echo "$tok" | deno eval 'const t=JSON.parse(await new Response(Deno.stdin.readable).text()); console.log(t.casUrl+" "+t.accessToken)')
  casUrl=${cas%% *}; access=${cas##* }
  for f in $(ls -S models/$NAME/**/*.safetensors 2>/dev/null); do
    rel=${f#models/$NAME/}
    hash=$(curl -sS -I -L "https://huggingface.co/$OWNER/$NAME/resolve/main/$rel" | grep -i '^x-xet-hash:' | awk '{print $2}' | tr -d '\r')
    stats=$(curl -sS -H "Authorization: Bearer $access" "$casUrl/v1/reconstructions/$hash" | deno eval 'const t=JSON.parse(await new Response(Deno.stdin.readable).text()); const terms=t.terms??[]; console.log(terms.length+" "+new Set(terms.map(x=>x.hash)).size)')
    terms=${stats%% *}; xorbs=${stats##* }
    size=$(stat -c %s "$f")
    printf '### fragmentation %-52s %5d MiB terms=%4d xorbs=%3d MiB/term=%.1f\n' "$rel" $(( size / 1048576 )) $terms $xorbs $(( size / 1048576.0 / (terms>0?terms:1) ))
  done
}

case $MODE in
  check)
    fragmentation_table | tee -a "$LOG"
    ;;
  upload)
    export HF_XET_DEDUPLICATION_MIN_N_CHUNKS_PER_RANGE=1000000
    export HF_XET_DEDUPLICATION_MIN_N_CHUNKS_PER_RANGE_HYSTERESIS_FACTOR=1.0
    export HF_XET_DEDUPLICATION_NRANGES_IN_STREAMING_FRAGMENTATION_ESTIMATOR=1
    export HF_XET_DEDUPLICATION_GLOBAL_DEDUP_QUERY_ENABLED=false
    for C in ~/.cache/huggingface/xet/*/shard-cache(N); do
      mv "$C" "$C.bak-$(date +%Y%m%dT%H%M%S)" && mkdir -p "$C" && echo "### shard-cache moved: $C" >> "$LOG"
    done
    echo "### upload $NAME $(date +%T) hf=$($HF version 2>/dev/null | tail -1)" >> "$LOG"
    $HF upload "$OWNER/$NAME" "models/$NAME" . --repo-type model "$@" >> "$LOG" 2>&1; rc=$?
    echo "### upload exit=$rc $(date +%T)" >> "$LOG"
    [[ $rc -ne 0 ]] && { tail -3 "$LOG"; exit $rc }
    # 4 本目の env が読まれ、CAS への chunk 照会が 0 回だったことを hf_xet のログで確かめる。
    L=$(ls -t ~/.cache/huggingface/xet/logs/* | head -1)
    echo "### xet log $L $(grep -o 'global_dedup_query_enabled = [a-z]* ([a-z ]*)' "$L" | head -1) query_dedup=$(grep -c 'Completed query_dedup' "$L")" >> "$LOG"
    SHA=$(curl -sS "https://huggingface.co/api/models/$OWNER/$NAME/revision/main" | deno eval 'const t=await new Response(Deno.stdin.readable).text(); console.log(JSON.parse(t).sha)')
    echo "### main sha $SHA" >> "$LOG"
    fragmentation_table >> "$LOG"
    echo "### DONE $NAME $(date +%T)" >> "$LOG"
    grep '^###' "$LOG" | tail -n +1 | awk -v s="### upload $NAME" 'index($0,s)==1{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}'
    ;;
  *)
    usage
    ;;
esac
