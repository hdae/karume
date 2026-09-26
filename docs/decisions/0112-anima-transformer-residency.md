# 0112: Anima の DiT 常駐 — opt-in・要求単位・OOM で退避して読み直す

- Status: accepted（実測済み・採用 — 2026-09-26・高速化の波 イテレーション 2「DiT 速度」B3・利用者承認。Arc B570 で
  2 回目以降の利得 2.45 s / 生成 = 壁の 10.4% が kill 線を超えた。opt-in のまま — 下の「kill 基準」節）
- Date: 2026-09-26
- 関連: ADR [0016](0016-anima-chain-export.md)（グラフは 1 本ずつ開いて閉じる — 既定として残す）/
  [0095](0095-plan-backing-budget.md)（backing の予算つき保持 — 解像度を変えたときの常駐量の上限）/
  [0107](0107-model-input-error.md)（未知の綴りの送出型）/ [0033](0033-vae-fixed-tile-decode.md)（VAE の常時タイル化 —
  チェーン最大が DiT 段へ移った理由）/ perf-ledger H-4・L-9。
  実装 = `packages/models/src/anima/residency.ts`（決定の純関数と状態機械）と
  `packages/models/src/anima/pipeline.ts`（結線）。調査の正本は
  `.claude/reviews/2026-09-25_perf-recon/deep/B3-anima-resident.md`（git 追跡外）。実測の正本は
  [research 2026-09-26](../research/2026-09-26-anima-residency-bench.md)。

## Context

`AnimaPipeline.generate` は段（text_encoder → text_conditioner → transformer → vae_decoder）ごとに Session を
作っては捨てる。既定の VRAM の前提は「最大の段 1 本ぶん」で、VAE を常時タイル化した後はチェーン最大が
DiT 段にある（既定席 1024² で runtime 集計 2,647 MiB）。

同じ pipeline で generate を繰り返す利用者（ブラウザのアプリ・サーバ・e2e）は、毎回 DiT の重みを読み直し、
計画を導出し、中間バッファ（backing）を作り直す。2026-09-10 の同一プロセス 6 連続生成（RTX 3080 Ti・
コンテナ移行前・turbo 1024²・CFG 1）で、これが 1 生成あたり **1.87〜2.25 s（壁の 15〜19%）** あった
（DiT 段の run 以外 1.46〜1.54 s + 初回 run の超過 0.30〜0.49 s + 2 回目 run の超過 0.11〜0.22 s）。
コンテナ移行後の Arc B570 での値は下の「kill 基準」節。

## Decision

1. **opt-in の常駐**: `AnimaPipelineOptions.residency?: "per-stage" | "transformer"`（既定 `"per-stage"` =
   従来の挙動）。真偽値にしないのは、将来 text 段の常駐を列挙で足せるようにするため。未知の綴りは
   admission 席（重みの part を取る前）で `ModelInputError`。構築は今までどおり Session を 1 本も張らない —
   常駐 DiT は**最初の generate の DiT 段で作る**。
2. **要求単位の決定**: `AnimaGenerateRequest.residency?`（同じ列挙）。実効値 = request ?? 構築オプション。
   意味は「**この generate の後に** DiT の Session を持ち続けるか」。generate の開始時に常駐 DiT が既に
   あれば、実効値に関わらずそれを使う（読み直さない）。連続生成の最後の 1 枚だけ `"per-stage"` にすれば、
   その generate の後に解放される。**generate が失敗しても**（DiT 段より前の text 段の失敗・`stage` イベントでの
   `onEvent` の throw 等）、実効値が `"per-stage"` なら持ち越した DiT を手放して `released` / `request` を名乗る
   （効かせないと、手放したつもりの VRAM が次の generate か dispose まで残る）。入力の検査で落ちた要求は
   GPU にも常駐の席にも触らない。
3. **OOM で退避して読み直す**（利用者の言う「スワップ」— 2026-09-26 に意味を確認: 「追加で読んで OOM したら
   解放して目的のモデルを読み直す」）。常駐 DiT がある状態で、他の段（text_encoder / text_conditioner /
   vae_decoder）の Session 構築か run、または**前の generate から持ち越した** DiT の DiT 段の run が
   `GpuOutOfMemoryError` を投げたら:
   ① 常駐 DiT を dispose ② `device.queue.onSubmittedWorkDone()` を待つ（Intel / wgpu は `destroy()` の解放が
   次の poll まで遅れる — known-issues「Intel Arc B570」節。device 消失とは競わせる）③ `residency` イベントで
   `evicted` / `out-of-memory` を名乗る ④ **その段を 1 回だけ最初からやり直す**（DiT 段なら段ごと運転で
   作り直す）。やり直しの失敗はそのまま投げる。判定は型（`instanceof GpuOutOfMemoryError`）で行い、
   validation や device 消失は退避しない（本当の原因が 2 度目の失敗に埋もれる）。
   - **同じ generate で作った DiT の run の OOM は退避しない**（格下げも通知もせず、段ごと運転の失敗と同じく
     畳んで元の例外を投げる）。そのとき pipeline が持つ GPU 資源はその DiT 1 本だけ（text 段は畳んだ後）で、
     VRAM の構成は段ごと運転と同じ — 常駐が原因の OOM ではないので、格下げして GB 級を読み直しても同じ構成を
     もう一度試すだけになる。実装上その DiT は段が成功するまで常駐の席に載せない。
   - ② の待ちの根拠: 素の WebGPU の probe（B570・2026-09-26 — [研究記録](../research/2026-09-26-anima-residency-bench.md)）で、
     満杯から 1 GiB を destroy した後 `onSubmittedWorkDone` **だけ**を待てば、同じ 1 GiB の再確保と 256 MiB の
     書き込みが通った（待ちは 11 ms。待たないと OOM）。固定の sleep は足さない。ただし退避経路を実機で踏むと、
     やり直しの段が device lost になった（Consequences）— 解放待ちの長さでは説明できない。
   - ①② が失敗しても（破棄・解放待ちの例外）格下げは立ち席は空くので、③ を名乗ってから元の OOM を先頭にした
     `AggregateError` を投げる（やり直さない）。名乗らないと、常駐を失ったことが次の generate まで見えない。
4. **格下げ**: 退避した pipeline は、その寿命の間は常駐しない。格下げ後に `"transformer"` を求められても
   持たず、`released` / `downgraded` を名乗る。**黙って遅い経路へ落ちない**（イベント必須）。
5. **失敗時は捨てる**: 前の generate から持ち越した DiT の段で例外が出たら（denoise ループ内の `onEvent` の
   throw による中断を含む）、常駐 DiT を捨てて `evicted` / `failure` を名乗り、元の例外を投げる。壊れうる
   Session を次の generate へ持ち越さない。格下げはしない。同じ generate で作った DiT の段の失敗は段ごと運転の
   失敗と同じ（席に載る前の失敗 — 畳むだけで名乗らない）。`stage` / `vae-tile` / `residency` の通知での throw は
   DiT 段の本体の外なので常駐 DiT を捨てない（決定 2 の手放しだけが効く）。後始末の失敗は元の例外を先頭にした
   `AggregateError`（文言「anima: DiT の後始末が失敗した」— 段ごと運転の DiT にも同じ文言）。通知の購読側が
   元と同じ例外を投げ直したとき（`signal.throwIfAborted()` を毎回呼ぶ中断の書き方）は並べない。
6. **イベント**: `AnimaGenerateEvent` に `{ kind: "residency"; component: "transformer"; action; reason }` を
   足す（`action` = `retained` / `released` / `evicted`、`reason` = `request` / `downgraded` / `out-of-memory` /
   `failure`）。既定の段ごと運転で常駐と無関係な generate では出さない（既存の購読側のイベント列を変えない）。
   `stage` イベントの意味は「Session 構築の前 / 解放の後」から「段の開始 / 終了」に変わる（型は不変）。
   持ち越した常駐 DiT を実効値 `"transformer"` で使う `transformer` 段の start → end にはロードも解放も入らない。
   実効値 `"per-stage"` で使う段では解放（手放し）が、OOM で退避した段では解放と段ごとのロードが start → end の
   中に入る。OOM の退避の後は、その段の進捗イベント（`denoise-step` / `vae-tile`）が 1 から出直す。
7. **dispose**: 直列化鎖の中で常駐 DiT を dispose してから GPU を破棄する（flush-before-destroy）。
   `options.gpu` を共有していても常駐 DiT は pipeline の所有物なので畳む — 共有 GPU でも dispose は必須
   （忘れると常駐 DiT が共有 device に残る）。device が失われていると常駐 DiT の破棄（Session の flush）が
   失敗して dispose が reject しうる（内部で取った GPU はそれでも破棄する）。
8. **決定は純関数 + 小さな状態機械に閉じる**（`residency.ts` の `transformerSource` /
   `decideAfterTransformer` / `decideAfterFailedGenerate` / `evictsForMemory` / `TransformerResidency`）。
   純関数は DiT 段の**開始時**の事実から決め、状態機械はその `action` を適用するだけ（真実は純関数 1 か所）。
   到達しない組（席に DiT があるまま格下げ済み）は既定値へ落とさず投げる。pipeline はそれを呼ぶだけ。
   GPU 無しの単体テストが故障注入で遷移を固定する（`anima_residency_test.ts`）。
9. **数値は変えない**: 参照行（`fixtures/references/anima.json`）を 1 行も足さず、全ケースを既存の
   turbo 1024 / 512 の行の双子として回す（`e2e_anima_test.ts` の「DiT 常駐」節）。

## VRAM（opt-in 側だけが上がる）

| 量（既定席 `f16+dit8-a8-attn8-s16`・1024²・CFG 1）            | 値                                                               | 出典                                |
| ------------------------------------------------------------- | ---------------------------------------------------------------- | ----------------------------------- |
| 常駐ぶん（重み 1,870.9 + backing 768.1 + 入力 7.1 MiB）       | **+2,646 MiB**                                                   | 2026-09-10 runtime 集計             |
| 段ごと運転の DiT 段ピーク（= 既定のチェーン最大）             | 2,647 MiB                                                        | 同                                  |
| 常駐時のチェーン最大（text 段 / VAE 段が常駐 DiT の上に乗る） | 約 3.9〜4.1 GiB（推測・確度: 中 — text 段の runtime 集計が無い） | B3 報告 §2-4                        |
| `f16` 席の DiT 重み                                           | 3,733 MiB（常駐すると text 段で +3.7 GB 超）                     | 2026-08-05 final-perf-bench VRAM 表 |

- **4 GB 級の GPU では使えない**（段ごと運転の DiT 段だけで 2.65 GiB あり、常駐はそこへ text 段を積む）。
  既定の段ごと運転では挙動は変わらない。
- 解像度を変えると backing は予算 256 MiB を超える形 1 本だけに絞られる（ADR 0095 決定 1）ので、**定常状態の**
  常駐量は「重み + 最大の backing 1 本」を超えない。解像度を切り替えた直後の run では、退役した古い backing
  （か arena）が flush の後まで生きるので、一時的なピークはこれを超える（推測: text 段のピークよりは小さく、
  チェーン最大は変わらない）。
- 退避は OOM を型（`GpuOutOfMemoryError`）で報告する環境でだけ発動する。Metal の out-of-memory errorScope が
  沈黙する環境（known-issues「Metal で out-of-memory errorScope が沈黙する」節）では、常駐が VRAM を超えても
  退避も格下げもイベントも起きず、遅延か device 消失として現れうる。

## kill 基準

オーケストレータが Arc B570（現況の開発機）で測る（台本 = `outputs/bench/karume/2026-09-26_anima-residency/bench.ts`
— per-stage / transformer をブロック単位の ABBA で回し、各ブロックの初回を除く）。
**2 回目以降の 1 生成の壁の差が 0.5 s 未満、または 1024² turbo の壁の 5% 未満**なら閉じる（opt-in を撤去する
かは閉じるときに裁定）。Chrome での同じ量は別に採る（ブラウザは CacheStorage から読むので利得が Deno より
大きい可能性がある — 推測・確度: 低）。

**判定（2026-09-26・B570 / Linux xe / Deno 2.9.6）: 閉じない — 採用（opt-in のまま・既定は `"per-stage"`）**。
既定モデル `anima-turbo-v1.1`・既定席・1024²・配布形の既定（8 step・CFG 1）。4 ブロック × 5 生成の ABBA で、
各ブロックの初回を除いた n = 8 ずつの中央値（[研究記録](../research/2026-09-26-anima-residency-bench.md)）:

| residency     | n | 壁      | DiT 段  | text 2 段 | VAE 段 |
| ------------- | - | ------- | ------- | --------- | ------ |
| `per-stage`   | 8 | 23.49 s | 16.77 s | 2.05 s    | 4.62 s |
| `transformer` | 8 | 21.05 s | 14.15 s | 2.23 s    | 4.63 s |

利得 = **2.45 s / 生成（`per-stage` の壁の 10.4%）**で、kill 線（0.5 s / 5%）を超える。PNG の sha256 は
20 枚すべて 1 種類で、参照行 `f16+dit8-a8-attn8-s16-1024` の B570 の値と一致した（決定 9 — 数値は変わらない）。

## 検討した代替案

- **B: A に加え、generate の末尾で backing だけ手放す**（常駐 +1,871 MiB）— 見送り。runtime に backing の明示
  退役の公開面が要り（ADR 0095 追記）、得るのは 768 MiB の節約だけ。6 GB 級の端末で A が収まらないと
  測れてから A の上に積む。
- **C: 跨がず、generate の中で DiT の構築を text 段と並走させる（先読み）** — 見送り。1 プロセス 1 生成の
  利用者にも効く唯一の案だが、text 段も DiT の構築も読み + アップロード律速で同じ帯域を取り合い、利得
  （0〜1.3 s・確度: 低）は測るまで分からない。A とは独立の候補として残す。
- **ホスト RAM に DiT の block を保持する** — 却下。読みと digest は消えるが、アップロード・導出・backing は
  毎回払う。ブラウザで 1.87 GB の ArrayBuffer を抱えるのは重い（perf-ledger L-9 の「opt-in の RAM 保持」）。
- **OOM を退避せずにそのまま投げる** — 却下（利用者の要求そのもの）。opt-in の機で VRAM が足りなければ
  生成ごと落ち、利用者は常駐を外して組み直すしかない。
- **格下げしない（毎回常駐を試みる）** — 却下。VRAM が足りない機で generate ごとに OOM を踏み、退避の
  費用（破棄 + 解放待ち + 段のやり直し）を毎回払う。
- **OOM した run だけをやり直す（段の途中から再開）** — 見送り。DiT のホスト側状態（latent・DPM の履歴）は
  保たれているので原理的には可能で、`denoise-step` の出直しも起きない。だが text 段は run を `Promise.all` で
  並走させており run 単位の再試行が並行の退避を要し、段ごとのやり直しより契約が込み入る。まれな経路なので
  段単位の単純さを採った（出る画素は同じ — 段の本体は seed から決定的）。

## Consequences

- 既定（`"per-stage"`）のイベント列・VRAM・数値は変わらない（`onEvent-1024` の門がイベント列を固定している）。
- `AnimaGenerateEvent` の union が 1 つ増えた（TS で 3 種を網羅して `else` で絞っていた購読側は型検査で落ちる —
  CHANGELOG の Breaking）。公開面に `AnimaResidency` / `AnimaResidencyAction` / `AnimaResidencyReason` の
  型 3 つが増えた（barrel と `./anima`）。
- `stage` イベントの `transformer` の start → end は、常駐時に GB 級ロードの進捗を表さない。
- 未計測: 常駐時の text / VAE 段のピーク（runtime 集計）、Chrome と B570 以外の機での利得。
- **B570 では退避 → やり直しが device lost になった**（2026-09-26・台本 =
  `outputs/bench/karume/2026-09-26_anima-residency/evict-probe.ts --dummy-gib 6`）。常駐 DiT の上で共有 device に
  6 GiB のダミーを積むと、2 枚目の text_encoder 段が OOM し、`evicted` / `out-of-memory` までは設計どおり出た。
  続くやり直しの text_encoder 構築（part 1 の重みアップロード中）で `GpuDeviceLostError`（reason unknown /
  device was lost）が投げられ、生成は失敗した（黙っては落ちない）。素の WebGPU では `onSubmittedWorkDone` だけで
  再確保が通るので、原因は解放待ちの長さではなく、**未特定**（known-issues「Intel Arc B570」節・調査は
  perf-ledger H-35）。他機での退避経路は未計測。
- B570（VRAM 9.93 GiB）で退避が起きたのはダミー 6 GiB のときだけで、4 / 5 GiB では常駐のまま 2 枚目が通った
  （20.96 s / 20.87 s・PNG sha は 1 枚目と一致）。
