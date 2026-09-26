# 0111: SessionOptions の合成規則 — 明示 > quant 宣言 > runtime 既定を全系列共通の 1 本にする

- Status: accepted（2026-09-26 — イテレーション 1「契約と土台」A3・利用者承認）
- Date: 2026-09-26
- 関連: ADR [0058](0058-numerics-opt-in-contract.md)（追記 2026-09-26 — 将来課題①の解き方）/
  [0104](0104-gemma-fast-quant.md)（gemma の合成規則の初出 — 本 ADR が全系列へ一般化する）/
  [0107](0107-model-input-error.md)（送出型の分類 — 決定 2）/ [0110](0110-practical-tier-numerics-contract.md)
  （参照層 / 実用層 — 束の正本は quant 席）/ [0038](0038-manifest-v1.md) §3（`gpuFeatures`）。
  実装 = `packages/models/src/session/options.ts`（`FamilySessionPolicy` / `resolveSessionOptions` /
  `assertSessionOverrides`）と `packages/runtime/src/runtime/session-build.ts`（`sessionOptionsViolation`）。

## Context

manifest の quant 席が宣言する `session`（実行ノブの束）を runtime の `SessionOptions` へ写す規則は、
gemma だけが「明示指定 > quant 宣言 > runtime 既定」の合成を持ち（ADR 0104・`gemma/session-options.ts`）、
他の 7 系列は宣言をそのまま写すだけで明示の上書き口が無かった。組合せの受理条件（`fuseLinearStaticQuantize`
は `linearGemvReduce: "parallel"` が要る 等）は runtime の Session 構築と gemma の写しの 2 か所にあり、
片方だけ直る形が作れた。2 層の契約（ADR 0110 決定 1「束の正本は quant 席・models は値の既定表を持たない」）
の下では、合成の規則そのものを 1 本にし、系列ごとに違うのは「どのキーを受けるか」だけにする必要がある。

## Decision

1. **合成規則は全系列共通の 1 本**（`resolveSessionOptions(policy, declared, overrides, where)`）。キーごとに
   明示（`undefined` 以外）があればそれ、無ければ quant 宣言、どちらも無ければ欄を作らない（= runtime 既定）。
   明示の `false` は宣言の `true` に勝つ。`null` のような不正な明示値は宣言へ戻さず拒否する（`??` は使わない）。
2. **系列が持つのは受理表だけ**（`FamilySessionPolicy` = `Required<SessionSpec>` の網羅写像・値は boolean）。
   値の既定は持たない（既定は runtime が 1 か所で持つ）。`SessionSpec` にキーが増えると全系列の表が型検査で
   落ちるので、「新しいノブを受けるか」を決め忘れる余地が無い。受理表: gemma = 並列 GEMV と融合の 4 欄・
   anima = `linearCompute` / `attentionCompute` / `attentionScoreStorage`・irodori = `linearCompute`（DiT の
   attention は実行時 bool マスクで融合 attention の契約に載らない）・sbv2 = `linearCompute`・f32 のみの 4 系列
   （birefnet / depth-anything / siglip2 / vowel-detector）= 全 false。
3. **組合せの受理条件は runtime の純関数 1 本**（`sessionOptionsViolation` — Session 構築が同じ関数で落とす）。
   models に写しを置かない。GPU の能力（shader-f16 / subgroups）を要る条件は device を見る構築側に残る。
   値域の門（`null` / 受理集合外の綴り）は models 側に置く — runtime は構築と同じく `??` で既定へ読むため。
4. **送出型は出所で分ける**（ADR 0107 決定 2）: 受理表が拒否するキーを manifest が宣言 → 素の `Error`（資産の
   齟齬）。受理表が拒否するキーを明示 → `ModelInputError`（黙って捨てると「指定したのに効かない」になる）。
   実効設定が受理条件を破ったら、同じ違反が宣言だけでも成立するなら `Error`、明示が関与して初めて成立するなら
   `ModelInputError`。
5. **合成は admission 席（重みの part を 1 バイトも取る前）で行う**。後段で行うと、未対応の宣言も誤った明示指定
   も GB 級の取得の後にしか落ちない。
6. **GPU feature の要求は実効設定から導く**（anima / irodori）: 明示の `"f16"` を shader-f16 を宣言しない quant
   に重ねたとき、quant 宣言だけ見ると自前で取る device が feature を持たず、重みを上げた後の構築で落ちる。
   `sessionGpuFeatures` が実効設定の要求を宣言へ足し、admission 席で名指しで落とす。上書き口の無い 6 系列への
   統一は未決（今のミラーに該当は無い）→ 追記 2026-09-26 で全系列へ統一した。
7. **runtime に一括軸（`numerics: "practical"` 等）は置かない**。束の中身は系列と格納型ごとに違い、manifest の
   quant 席だけが表せる（ADR 0058 追記 2026-09-26）。

## 検討した代替案

- **gemma の規則を各系列へ複製する** — 却下。複製は綴りの改名を型検査で捕まえられても、規則の変更（優先順位・
  送出型）は系列ごとに追随漏れを起こす（`toSessionOptions` の 1 本化と同じ理由 — `options.ts` の NOTE）。
- **models に値の既定表を持つ**（段 1 の案）— 却下。gemma の `stateAttentionReduce: "parallel"` の models 既定が
  `i4` 席を参照経路でなくしていた実例（ADR 0110 Context）。値は quant 席へ昇格する（別 ADR）。

## Consequences

- 同値: この機の配布ミラー 11 本・86 quant 全てで `resolveSessionOptions(policy, quant.session, {})` が従来の
  `toSessionOptions(quant.session)` と deep equal（`session_options_mirror_test.ts`）。既存の sha 行は動かない。
- gemma の組合せ違反の文言は runtime のものに揃った（テストの期待部分文字列 8 か所を追随・受理集合と送出型は不変）。
- runtime の公開面に `sessionOptionsViolation` が 1 つ増えた（ADR 0008 の薄い面 — 構築と同じ判定を取得前に借りる
  上位層のため）。
- runtime の検査順序は「GPU 非依存の検査を全て先・GPU 能力の門を後」になった。単一違反の入力では不変で、複数の
  違反を同時に持つ入力でだけ先に出る文言が変わりうる。

## 追記（2026-09-26）— GPU feature の要求を全系列で実効設定から導く（決定 6 の一般化）

利用者の回答（2026-09-26・逐語）は「7. b, 必要になるまでそのまま。(通した方が綺麗であればaだけど)」で、主回答は b
（上書き口の無い系列は据え置き）、a は「通した方が綺麗なら」の条件付きだった。オーケストレータはこの条件が成り立つと
判定して a を採った。根拠は 2 つ: ① 8 系列が同じ形になり、`gpu-features.ts` の MUST（要求と検査を実効設定から導く）
に「上書き口を持つ家族だけ」という例外が残らない ② 振る舞いの差が出るのは「f16 計算を宣言し `shaderF16` を書き
落とした manifest」だけで、今のミラーに該当は無い（下の「振る舞いの差が出る唯一の形」）。
決定 6 を全 8 系列の規則にした: 各系列の admission 席で
`sessionGpuFeatures(quant.gpuFeatures, 実効 SessionOptions)` を 1 度だけ導き、共有 GPU の検査・`acquireGpu` の要求・
取得後の検査の 3 か所がこの 1 つの値を見る（`packages/models/src/session/gpu-features.ts` の MUST）。

- **理由**: 「要求する feature = 宣言 ∪ 実効設定が要る feature」を系列ごとに書き分けない。書き分けると、受理表
  （決定 2）が f16 計算を受けるように変わった系列だけ要求が追随しない形が型検査を通る — 受理表を網羅写像にしたのと
  同じ理由で、規則そのものを 1 本にする。
- **振る舞いの差が出る唯一の形**: `session` で f16 計算（`linearCompute` / `attentionCompute` = `"f16"`）を宣言
  しながら `gpuFeatures.shaderF16` を宣言しない manifest。今の受理表でこの宣言を受け、かつ従来導いていなかったのは
  sbv2（`linearCompute`）だけ（anima / irodori は決定 6 で導き済み・f32 のみの 4 系列と gemma は f16 計算を受けない）。
  その manifest は従来 Session 構築の f16 門で落ちていたが、共有 GPU なら admission 席で名指しで落ち、自前で取る
  device なら feature を要求して通る。今のミラーに該当は無い（`session_options_mirror_test.ts` が全系列の全 quant で
  「導いた要求 = 宣言」を見る）。
- **gemma**: 従来は `quant.gpuFeatures` を一切読まず（宣言があっても要求も検査もしない — 沈黙劣化の形）、
  `acquireGpu({ subgroups })` だけだった。`fromPretrained` は admission の閉包（`resolveGemmaSessionOptions` の席）で
  導いて共有 GPU をその席で検査し、構築は `acquireGpu({ ...要求, subgroups })` と取得後の検査を通る。`fromAssets` は
  manifest を持たないので宣言は無く、実効設定が要る feature だけを要求する（今の受理表では空）。drafter Session は
  同じ device に張るので別に要求しない。今の gemma ミラーは `gpuFeatures` を宣言しないので、要求は従来と同じ。
