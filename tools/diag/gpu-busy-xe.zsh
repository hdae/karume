#!/usr/bin/env zsh
# Arc B570（xe ドライバ）の使用率の目安: gt0 の idle 滞在時間の増分から busy% を出す。
# 使い方: gpu-busy-xe.zsh [秒数=3]  → "busy=NN% act_freq=MMMM idle_status=..." を 1 行。
# コンテナ内からは他プロセス（学習）の VRAM 使用量は見えないので、この busy% と act_freq で判定する。
G=/sys/class/drm/card0/device/tile0/gt0
sec=${1:-3}
i0=$(cat $G/gtidle/idle_residency_ms); t0=$(date +%s%3N)
sleep $sec
i1=$(cat $G/gtidle/idle_residency_ms); t1=$(date +%s%3N)
busy=$(( 100 - 100 * (i1 - i0) / (t1 - t0) ))
(( busy < 0 )) && busy=0
# throttle= は xe の freq0/throttle/reasons（thermal / pl1 / ratl …・空なら無し）。1 submit の時間門が外れた回の切り分け用（2026-10-03）。
echo "busy=${busy}% act_freq=$(cat $G/freq0/act_freq) idle_status=$(cat $G/gtidle/idle_status) throttle=$(cat $G/freq0/throttle/reasons 2>/dev/null | tr -d "\n") window=${sec}s"
