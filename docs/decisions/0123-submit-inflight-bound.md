# 0123: 送り込み（`queue.submit`）の先行に上限を置く

- Status: accepted（利用者裁定 2026-10-07 — 「どちらも推奨案で OK、進めてください」）
- Date: 2026-10-07
- 関連: ADR [0004](0004-execution-model.md)（実行モデル — submit の時間予算分割。本 ADR は改訂 ⑤ として先行の上限を足す）/
  [0032](0032-deserialize-submit-and-optin-timing.md)（submit ごとの `onSubmittedWorkDone` の廃止 — 本 ADR はその MUST NOT を
  保ったまま、間隔を空けた完了印だけを足す）/ [0054](0054-resident-loop-and-fence.md)（通常 run のフェンスを `mapAsync` 1 本へ・
  BatchScope・追い越し不変条件の新形）/ [0121](0121-wan22-ti2v-5b.md) 追記（2026-10-07）「段 10」の「参照席の 121 の行を書く走行 —
  device lost」/ research [2026-08-04-host-overhead-recon](../research/2026-08-04-host-overhead-recon.md) §4.1（Deno の
  `onSubmittedWorkDone` は同期部分が完了までホストを止める）

## Context

### 起きたこと

開発機（RTX 3080 Ti・NVIDIA 615.71.09 の open kernel module・Deno 2.9.6・Vulkan・この GPU は画面も出している）で、Wan2.2 の
参照席 `f16+dit8` の 1280×704×121（DiT のトークン数 S = 27,280）・2 ステップが、2 本目の DiT の forward の途中で device lost になり、
Deno が panic した（`ext/webgpu/queue.rs:109` — `on_submitted_work_done` の `device_poll(wait_indefinitely).unwrap()`）。
5 回中 4 回・公開 API の `deno run` でも再現した。調査の記録は `.claude/reviews/2026-10-07_wan22-121f-device-lost/FINDINGS.md`
（git 追跡外）と `outputs/diag/wan22-121f-device-lost-2026-10-07/`。

### 切り分けた事実

- **1 回の処理の長さではない**: 製品と同じ歩調の計測で最長の submit 0.650〜0.669 s・最長の dispatch 0.356〜0.401 s（1 s 超は 0 本）。
  計測で歩調を崩した走行では 38 s の submit・19 s の dispatch でも落ちなかった。
- **熱ではない**: HW Thermal Slowdown の累計は落ちた走行の前後で不変。93 ℃ のまま 8.5 分回った走行も完走した。
- **カーネル側の回復ではない**: ホストの kernel log に NVRM の Xid が 1 件も無い（利用者が確認）。device lost はユーザー空間の
  Vulkan ドライバが返した形。
- **前兆**: 落ちた走行は全て、2 本目の forward の 1,084〜1,100 本目の `queue.submit` そのものが 11,008〜11,022 ms ブロックし、
  明けた後の submit は 1 本あたり約 10 倍速く返った（後続が GPU に届いていないと読める — 推測）。GPU がその位置に着いた時刻に
  device lost が表に出た。
- **ホストの先行に上限が無い**: runtime は submit ごとに完了を待たない（ADR 0032）。通常 run のフェンスは run の末尾の `mapAsync`
  1 本（ADR 0054）なので、1 本の run の間はホストが GPU より無制限に先行する。参照席の 1 forward は約 1,926 submit を 0.4 s で積み、
  GPU は約 80 s かかる。実用席の 1 forward は約 573 submit で、落ちた例が無い。

### 読み（推測）

ドライバの内部の何か（送り込みの列・コマンドバッファの資源）が先行のしすぎで溢れ、`vkQueueSubmit` の内部の待ちが時間切れになって
device lost を返した。機構はドライバの中で、確かめる手段が無い。先行の上限は、どの機構であっても溢れの元を断つ。

## Decision

1. **完了印**: run が dispatch を積む間、ステップの間で `SubmitScheduler.backpressure()` を問い、計測窓の中で前の印からの submit が
   **`IN_FLIGHT_MARKER_INTERVAL`（64）回に届いていれば** `queue.onSubmittedWorkDone()` を呼んで完了印を 1 本置く。submit ごとには
   呼ばない（ADR 0032 の MUST NOT はそのまま — 間隔を空けた印だけを足す）。印はフェンスとしては扱わない（計測窓を閉じない）。
2. **数える単位は計測窓**（Session のスケジューラごと）。窓はフェンスの直後に閉じる（flush・単一フェンスの run の `mapAsync`・batch の
   決着）ので、「最後にフェンスで GPU の完了を確かめてからの submit の数」がそのまま窓のチャンク数になる。窓を閉じる・捨てるときは
   印の位置も 0 へ戻す。フェンスの間の submit が 64 回に届かない run（decode のような短い run の繰り返し）は印を 1 本も置かず、待ちも
   生まない。
3. **待ち**: 未完了の印が **`IN_FLIGHT_MAX_MARKERS`（2）を超えたら**、最も古い印の完了を待つ（印は 1 回に 1 本しか置かないので、
   待つ時点の未完了は 3 本で、最も古い印が済めば 2 本に戻る。先行は最大で約 3 × 64 = 192 回）。上限の内では待ちを作らない（Promise を
   返さず、マイクロタスクも挟まない — 上限に届かない run は `executeBakedPlanPaced` も Promise を返さず、変更前と同じく同期で積み
   終える）。待つ前に未 submit のエンコードを出し切る — 待ちの間に出る `queue.writeBuffer` が未 submit の dispatch を追い越さない
   （ADR 0004 不変条件④を待ちの間も保つ）。消失は `raceDeviceLost` で `GpuDeviceLostError` へ変換する（消失後の
   `onSubmittedWorkDone` が解決しない実装がありうる）。
4. **印を置き待つのは run だけ（ステップの間 — `executeBakedPlanPaced`）**。run は errorScope 区間ロックの中で dispatch を積み、ロックは
   他の run と batch を締め出すので、待ちの間に他の誰の dispatch も割り込まない。
   **MUST NOT: enqueue は待たず、印も置かない**（変更前と同じ動き）。同じ batch の別 Session の enqueue は区間ロックでは直列化されず、
   本体の順序は Session ごとの直列化（`#chain`）の段数が揃っていることで保たれている。enqueue が待つと、待っている間に後から呼んだ
   別 Session の enqueue の本体が走って先に queue へ載る。本体の途中で待つ形は 1 周目で、末尾で待つ形も 2 周目で崩れる（A1 → B1 → A2 →
   B2 を非 await で積むと、A2 は A の直列化で A1 の待ちに止められ、B2 だけが先に走る）。常駐テンソル越しに前の enqueue の出力を読む
   形では、書かれる前の値を読む沈黙誤値になる（どちらも実 GPU の故障注入・独立レビューの模型で確認した）。
   印だけを置く形（待たない）も採らない — 下の「採らなかった案」。
5. **Deno での実際の形**: Deno 2.9.6 の `onSubmittedWorkDone` は同期部分が「それまでに submit した全作業の完了」までホストを止める
   （research 2026-08-04 §4.1）。印を置いた時点で GPU が追いつくので、Deno の run の先行は 64 回までになり、待ちの Promise は解決済みの
   ものを待つだけになる。GPU が遊ぶのは、印が明けてから次のチャンクを積み終えるまでの短い間だけ。ブラウザ（`onSubmittedWorkDone` が
   非同期の実装）では、run の先行は約 192 回まで。enqueue の先行はどちらでも従来どおり上限が無い。
6. **数値は変わらない**: 積むコマンドの中身と順序は変えない。変わりうるのは run のチャンクの切れ目だけ（待つ前に未 submit を出し切る
   ため）で、command buffer の境界は実行の意味を変えない。sha256 の参照値は全系列で不変であることをフル verify で確かめる。

### 値の根拠

- 落ちたのは先行が約 1,080 回の位置。計測の変種では約 300 回で送り込みが詰まり始めた走行もある（落ちてはいない）。実用席の 1 forward
  の約 573 回は落ちていない。上限はこれらより十分小さく、待ちの回数が速度に効かない大きさにする: 64 回ごと（Deno で 1 forward の
  待ちは参照席で約 30 回・実用席で約 9 回）。
- 待ちの代価の見込み（推測 — 下の「検収」で実測する）: 1 回の待ちで GPU が遊ぶのは 1 チャンクを積む時間（submit 1 本あたり約 0.2 ms —
  参照席の 1,926 本を 0.4 s で積んだ実測から）と待ちの解ける遅れ。1 forward（参照席 約 80 s・実用席 約 27 s）に対して 1 % を大きく下回る。

## 採らなかった案

- **device 全体の累計で数える**（`GpuContext` に数を持つ）: 同じ queue に複数のスケジューラが積む形でも 1 本の上限で済むが、フェンスで
  数え直す手段が無い。短い run を繰り返す decode でも累計が 64 に届くたびに印が出てホストが止まり（Deno）、フェンスの本数を数える
  テストが前に回したものしだいで変わる。窓で数えれば数え直しはフェンスと同時に起きる。待つ run は区間ロックで直列なので、
  スケジューラごとに数えても device 全体の上限と同じになる。
- **enqueue も待つ（本体の途中・末尾）**: 決定 4 の理由で呼び出し順が崩れる。順序を守ったまま待つには、batch の中の enqueue の本体を
  受理の順に 1 本ずつ流す仕組み（batch 単位の FIFO）が要る。BatchScope の契約を変える新しい直列化で、デッドロックの検討も増えるので、
  この ADR では採らない（今の失敗は run の経路）。同じ仕組みは、初回の enqueue（導出の `await` を持つ本体）が後から呼んだ別 Session の
  enqueue に追い越されうる既存の形も閉じる — 別の判断として扱う。
- **enqueue に印だけを置く（待たない）**: 上限が効くのは `onSubmittedWorkDone` が同期でホストを止める実装（Deno）だけで、仕様の保証では
  ない実装の癖に頼る。ブラウザでは何も変わらない。その一方で Deno では batch の生成ループ（irodori の DiT のループ・Wan の VAE の chunk）
  に、印のたびに GPU が遊ぶ止まりが入る。enqueue の長い batch（121 フレームの VAE の約 2 万 submit）は変更の前から落ちた例が無いので、
  enqueue は変更前と同じ動きに保つ（この案で一度実装し、独立レビューの後に外した）。
- **時間で上限を置く**（推定した GPU 時間の先行を X 秒まで）: 推定は run の 1 本目には無く（裏付け前）、窓の実測はホスト時間込みの過大側。
  溢れたのが本数か時間かも分からない。本数は推定に依らず決定的で、テストで固定できる。
- **開発機の画面を内蔵 GPU へ移す・ドライバの設定**: 開発機の参照値は安定する見込みだが、画面を出している NVIDIA の GPU を使う利用者の
  製品の問題が残る。
- **参照席の 121 フレームを受理から外す**: 先行の上限が無いことは席にも長さにも依らない性質で、外しても同じ形の大きい run で再発しうる。

## Consequences

- `SubmitScheduler.backpressure()` が増える（上限を超えたときだけ Promise を返す）。呼ぶのは `executeBakedPlanPaced`（run）だけ。
  スケジューラへ dispatch を積むのは run と enqueue の焼き込み実行だけで（`dispatchWithWork`）、重みの上げ込みや GenerationContext の
  書き込みは `queue.writeBuffer` で、スケジューラを通らない。
- 上限は Session のスケジューラごと。run は区間ロックで直列なので device 全体でも 1 本ぶんになる。
- 待つのはステップ（1 ノード — 融合を含む）の間なので、1 ステップの中の submit はブラウザでは上限の外にはみ出しうる。
- Deno では印の同期部分の間（参照席の DiT なら 64 submit ぶん — 約 2.7 s）JS のスレッドが止まり、タイマや進捗の通知が遅れる
  （それまでは積み終えた後の `mapAsync` の待ちで空いていた）。進捗の通知はステップの間で出るので、見え方は変わらない。
  また、この同期部分の中で device lost が起きると、Deno は `GpuDeviceLostError` ではなく panic する（flush と同じ — known-issues の
  Deno の device lost の項）。
- run の途中の待ちの間も errorScope は push したまま（区間ロックの中）。ロックの外から GPU 操作を出す層は同期区間で完結するスコープ
  しか使わない規約（`GpuContext` の errorScope 区間の不変条件）なので、帰属は変わらない。スコープを張らない操作（利用者の
  `ResidentTensor.write` など）の失敗がこの run に帰属しうる窓は、ミス run（パイプラインの生成の await）に元からあり、それがヒット run
  にも広がる。
- フェンスの本数を数えるテストのうち、フェンスの間に 64 回以上 submit する run は印の本数ぶん `onSubmittedWorkDone` が増える
  （enqueue は増えない）。リポ内の門（小さなグラフ）はどれも 64 回に届かない。

## 検収

- ホストのテスト（フェイクの device・本物の `GpuContext`）: 完了印は run が歩調を問うたときだけ、前の印から 64 回に届いていれば 1 本
  置き（submit だけでは置かない）、フェンス（flush・フェンスの後の窓の閉じ・窓の破棄）で窓のチャンク数も印の位置も数え直す・未完了の
  印が上限までは待たず、超えたら最も古い印を待ってその完了で解ける・待ちを返すときだけ未 submit を出し切る・消失すると
  `GpuDeviceLostError` で落ちる（ハングしない）・`executeBakedPlanPaced` は待ちの間に次のステップを積まず、上限の内では Promise を
  返さない・enqueue の経路（`executeBakedPlan`）は歩調を問わない。
- 実 GPU の門（`packages/runtime/tests/gpu_submit_backpressure_test.ts`）: 長い run の値と完了印の本数・run が上限を超えたら印を待ち、
  その間も区間ロックを離さない・enqueue は印を置かず待たない・非 await で 2 周続けた別 Session の enqueue が常駐テンソル越しの順序を
  追い越さない。
- 故障注入 10 通り（印を置かない・enqueue が途中で待つ・enqueue が末尾で待つ・待ちを返さない・待つ前に submit しない・run が待たない・
  run が上限の内でも譲る・enqueue の経路で印を置く・フェンスで印の位置を戻さない・enqueue の経路が歩調を問う）で、それぞれ対応する門が
  赤になることを確かめた。
- GPU: 参照席の 1280×704×121・2 ステップを 5 回回して 5 回とも完走（1 回目で開発機の行を書き、残りは同じ行と一致）。この 5 回は
  enqueue にも印を置いていた形で回した。最終形（`7fefb2ad`）でも 1 回完走し、同じ行と一致した（ADR 0121 の段 10 の節）。
- 速度: 前後のフル verify（`6da5288b` → `7fefb2ad`・同じ開発機・熱制限込み）で、テストファイルごとの所要の合計は 7,298 → 7,057 s
  （−3.3%）。ファイルごとの差は ±25% ほどまで散る。目立った 2 本（Wan2.1 の VAE タイルの縦長 9 フレーム・Gemma 4 の 598 トークンの
  prefill）は、変更前のコードでも同じ二峰（12 s 対 20 s 台・8 s 対 13 s 台）が出る既存のばらつきで、Gemma のテスト全体で置かれる印は
  1 本・待ちは 0 回（計測を足したコピーで確認）。A/B の記録は `.claude/reviews/2026-10-07_submit-inflight/`（git 追跡外）。
- フル `deno task verify` が緑（`7fefb2ad`: 4,003 passed・0 failed・12 ignored — sha256 の参照値は全系列で不変）。
