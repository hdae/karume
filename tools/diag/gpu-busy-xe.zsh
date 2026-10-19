#!/usr/bin/env zsh
# A rough utilisation reading for the Arc B570 (xe driver): derives busy% from the increase in gt0's idle residency time.
# Usage: gpu-busy-xe.zsh [seconds=3]  → "busy=NN% act_freq=MMMM idle_status=..." on one line.
# Inside a container the VRAM use of other processes (training) is not visible, so decide from this busy% and act_freq.
G=/sys/class/drm/card0/device/tile0/gt0
sec=${1:-3}
i0=$(cat $G/gtidle/idle_residency_ms); t0=$(date +%s%3N)
sleep $sec
i1=$(cat $G/gtidle/idle_residency_ms); t1=$(date +%s%3N)
busy=$(( 100 - 100 * (i1 - i0) / (t1 - t0) ))
(( busy < 0 )) && busy=0
# throttle= is xe's freq0/throttle/reasons (thermal / pl1 / ratl …; nothing when empty). For isolating the runs where the per-submit time check failed (2026-10-03).
echo "busy=${busy}% act_freq=$(cat $G/freq0/act_freq) idle_status=$(cat $G/gtidle/idle_status) throttle=$(cat $G/freq0/throttle/reasons 2>/dev/null | tr -d "\n") window=${sec}s"
