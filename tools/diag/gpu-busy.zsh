#!/usr/bin/env zsh
# GPU の使用率の目安（RTX 3080 Ti・NVIDIA ドライバ — nvidia-smi）。学習と共有の GPU を使う前の判定に使う。
# 使い方: gpu-busy.zsh [秒数=3]  → "busy=NN% max=NN% vram=used/totalMiB sm_clock=MMMMMHz temp=NN power=NNW throttle=... window=Ns" を 1 行。
#   窓の間 1 秒ごとに utilization.gpu を取り、平均（busy）と最大（max）を出す。vram は GPU 全体（他プロセスの分を含む）。
#   throttle は nvidia-smi の clocks_event_reasons のうち Active な理由の名前（gpu_idle は省く・無ければ none）。
# 旧 Arc B570（xe の sysfs）版は gpu-busy-xe.zsh に残す。
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
