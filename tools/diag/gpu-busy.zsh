#!/usr/bin/env zsh
# A rough GPU utilisation reading (RTX 3080 Ti, NVIDIA driver — nvidia-smi). Used to decide before using a GPU shared with training.
# Usage: gpu-busy.zsh [seconds=3]  → "busy=NN% max=NN% vram=used/totalMiB sm_clock=MMMMMHz temp=NN power=NNW throttle=... window=Ns" on one line.
#   Samples utilization.gpu once per second over the window and prints the mean (busy) and the maximum (max). vram is for the whole GPU (including other processes).
#   throttle is the names of the Active reasons among nvidia-smi's clocks_event_reasons (gpu_idle is left out; none when there are none).
# The version for the former Arc B570 (xe sysfs) is kept in gpu-busy-xe.zsh.
emulate -L zsh
sec=${1:-3}
(( sec < 1 )) && sec=1
local -a util
local sum=0 max=0 u
for ((i = 0; i < sec; i++)); do
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  util+=($u); (( sum += u )); (( u > max )) && max=$u
  (( i < sec - 1 )) && sleep 1
done
local busy=$(( sum / sec ))
local line=$(nvidia-smi --query-gpu=memory.used,memory.total,clocks.current.sm,temperature.gpu,power.draw,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown,clocks_event_reasons.hw_power_brake_slowdown,clocks_event_reasons.applications_clocks_setting,clocks_event_reasons.sync_boost --format=csv,noheader,nounits)
local -a f; f=(${(s:, :)line})
local -a names; names=(sw_thermal hw_thermal sw_power_cap hw_slowdown hw_power_brake app_clocks sync_boost)
local -a active
for ((k = 1; k <= 7; k++)); do [[ ${f[5 + k]} == Active ]] && active+=(${names[k]}); done
echo "busy=${busy}% max=${max}% vram=${f[1]}/${f[2]}MiB sm_clock=${f[3]}MHz temp=${f[4]} power=${f[5]%.*}W throttle=${(j:,:)active:-none} window=${sec}s"
