# 0069: packed 4bit 格納（ADR 0019 の w4 棄却を reopen）

- Status: accepted（2026-08-17 — ユーザー裁定 B〈zero-point 欄なし・予約つき〉/
  C〈scale f32 開始・f16 両対応を将来目標〉+ 委任チェック方式。Codex レビュー第 3〜5 巡を
  反映〈3 面分離・group scale 形・clamp tiny・f16 admission〉し第 6 巡で go）
- 関連: ADR [0019](0019-i8-weight-execution.md)（i8 経路 — 本 ADR が「w4 再測しない」を
  supersede し、実装規律は延長する）/ [0006](0006-quantization.md)（格納のみ量子化・
  fake-quant 正）/ [0063](0063-safetensors-physical-layout.md)（物理配置 — 本 ADR とセットで
  改訂）/ [0058](0058-numerics-opt-in-contract.md)（品質管理の分担）/
  [0066](0066-generation-context-state-slots.md)（検収モデル = LLM への波）
- 根拠:
  [research/2026-08-17-autoregressive-references.md](../research/2026-08-17-autoregressive-references.md)
  §4（以下「調査 §n」）

## Context

検収モデル級（1〜2B param）を配布・常駐するには f16 でも 2〜4GiB 級で、4bit 格納が実用条件
になる。一方 ADR 0019 は「per-tensor / w4 group 量子化: 不採用確定。再測しない」と書くが、
この射程は過大（調査 §4.3）: 旧測定は **SBV2 voice 1 ファミリ・1 時点の int4 group 実測**
（2026-08-03）であり、親 ADR 0006 は group-wise を「低優先 backlog」としか書いていない。
棄却根拠の一つ「発話長の系統的短縮」は**採用済みの w8 でも観測されており**（ADR 0029:53）
w4 固有の基準として機能しない。WebGPU 上の 4bit は ORT MatMulNBits / MLC q4f16_1 /
llama.cpp WebGPU が実運用している（調査 §4.1）。

## Decision

### 1. ADR 0019 の supersede 範囲（reopen の正確な形）

- 「**再測しない**」を撤回する。旧測定の適用範囲を「SBV2 voice の音声 SNR / 発話長・
  per-tensor RTN および int4 group の 1 時点実測」に限定し、**テキスト生成（検収モデル）へ
  一般化しない**。数値の帰属も訂正する: −1.5〜+5.1dB は int4 group の実測（0019:17 が正・
  素朴 RTN へ帰属させた 0006:45 の側が不正確 — 調査 §7 wt LB-6 の検証済み補正）。
- 0019 のその他（i8 経路・±127 対称・平坦添字・タイル読み込み時 dequant・fake-quant 規律）
  は**本 ADR の土台としてそのまま有効**。
- 派生同期（実装波で）: ir-v1.md「group_size は実行経路が無い」・limitations の w4 項・
  container の groupSize 拒否。

### 2. 格納の表現 = bit 幅一般化（shape は論理のまま — 裁定 3a）

新しい格納 dtype **`storage.dtype: "i4"`** を追加する。**テンソル shape（IR `values{}` /
safetensors ヘッダ）は論理形のまま**で、バイト数だけ bit 幅から導出する
（GGUF / safetensors 本家と同配置 — 調査 §4.2。ORT 型の「物理 shape + 論理 attrs」は
消費側 op 契約〈linear の `W[out,in]` 等〉が総崩れになるため採らない）。

- safetensors リーダの改訂は **3 面を分離**する（第 3 巡で精密化 — `DTYPE_BYTES` の
  単純置換では済まない。現行表は「バイト長の計算」と「絶対 offset の整列検査」の両方に
  使われている）: ①**サイズ表（bit 単位）** — 検証は `numel × bits / 8` の厳密一致
  ②**整列表（byte 単位）** — I4 は要素整列の概念を持たず「テンソル先頭が 4 byte 整列」を
  要求（u32 束縛のため。F32/I32 = 4・F16 = 2 は従来どおり）③**view の型** — I4 に対応する
  TypedArray は存在しないため、view は **raw バイト（`Uint8Array`）+ 論理 numel の
  メタデータ**で表す（`STORAGE_ENCODING` の突合・container 検査・適格外の CPU nibble
  展開器も同時に改訂対象として列挙する）。
- **端数を作らない制約で整列問題を消す**: `i4` は量子化軸（最終次元 = linear の in 軸）の
  要素数が `group_size` で割り切れることを MUST とし、`group_size` は
  **2 冪かつ ≥ 16**（ORT と同制約 — 調査 §4.1。single-scale 例外は作らない）。これで
  行境界・group 境界が常にバイト整列し、テンソル総バイト数は 4 の倍数（u32 束縛の
  自然整列 — 末尾ゼロ詰め不要）。
- ADR 0063 改訂（セット）: 書き出し順は I8 の後ろ末尾に I4 を足す・リーダ / ライタ /
  `assert_reader_layout` を対で改訂。

### 3. 量子化形 = K 方向 group・対称 15 準位・scale f32（裁定 B / C 反映済み）

- **group 対称量子化**: `scale = clamp(amax / 7, f32 tiny)`（group ごと — clamp は全ゼロ
  group の 0 除算を閉じる。ADR 0019 の `clamp(amax/127, f32 tiny)` と同文・第 3 巡で
  脱落を補正）・`q = clamp(round(w/scale), −7, +7)` を **offset 8 の unsigned nibble** で
  格納（値域 [1,15]・0 は未使用 = 15 準位）。amax 要素が厳密復元され fake-quant が冪等 —
  ADR 0019 の ±127 論証の 4bit 版（MLC q4f16_1 と同思想・調査 §4.1）。
- **scale の形を明文化**（第 3 巡の矛盾指摘の閉鎖 — ADR 0019 の「同 rank keepdim
  broadcast」は per-channel 専用で group には不適合）: scale の論理 shape は**重みと同 rank・
  量子化軸を group 数に置換**した形（linear `W[O,I]` → scale `[O, I/group_size]`）。
  container の scale 検査（現行 = keepdim broadcast 形）と WGSL の scale 束縛（現行 =
  出力チャネルあたり 1 値の `wscale[channel]` — weight-storage.ts:140-150）は **group 版の
  別分岐**として拡張する（添字 = `(row, k / group_size)`・タイル読み込み時に group 境界で
  引き直す）。CPU 展開経路も同じ添字式でビット一致を保つ。
- **zero-point 欄は作らない**（裁定済み 2026-08-17 — 対称のみで開始）。**後日追加で
  困らないための予約 3 点**: ①欄名と形をここで予告する — 追加時は `storage.zero_point`
  （scale と同形の companion `[O, I/group_size]`）で、既存資産・既存検査に影響しない
  **追加欄**になる ②dequant 式は `(u − 8) · scale` を「zero_point 省略時の既定 = 8」と
  読める形で規定する（ORT と同じ規約 — 非対称化は定数 8 を zero_point 読みへ差し替える
  だけで、pack 形式・pack 順・15 準位の値域は不変）③Phase 0 sweep（決定 6）に非対称の
  測定列を含め、対称の品質不足を**実装前に**検知する。
- **scale は f32 companion で開始・f16 受理を将来目標**（裁定済み 2026-08-17 — 最終的に
  両対応）。開始形は既存 `storage.scale` + `storage.group_size` の解禁で、ADR 0019 と
  同じく GPU dequant と CPU 展開のビット一致を保つ。**f16 scale の追加はビット一致体制を
  捨てずにできる**: f16 → f32 変換は厳密（拡張は無損失）なので、「f16 へ丸めた scale を
  CPU / GPU の両方が同じ f32 値として使う」形で検証規律がそのまま成立する（丸めは emit の
  1 回だけ・読みは厳密 — スコア格納 s16 と同じ手筋）。追加時は scale companion の dtype
  受理を F32 | F16 へ広げ、emit 側は fake-quant が f16 丸め済み scale で参照採取する。
  **f16 受理時の追加条件（第 4 巡）**: f32 → f16 の**丸めは**厳密でない — `f32 tiny` は
  f16 で 0 に、大きい scale は inf になり得る。emit 側 admission で「f16 丸め後の scale が
  **有限かつ非ゼロ**」を全 group で検査し（既存 f16 emit の inf 拒否と同族）、全ゼロ group
  の clamp 下限は **f16 の最小正規値**へ置き換える。
  f16 scale（MLC・GGUF の選択）とのサイズ差はサイズ試算表参照 — 帯域が問題になったら
  group_size を上げる側でも調整できる（bit 幅あたりのオーバヘッド: group 32 で
  f32 = +25% / f16 = +12.5%、group 128 で f32 = +6.25%）。既定 group_size は
  Phase 0 sweep（決定 6）で決める。

### 4. pack 順の明文固定 + 検出器

**「1 バイトに要素 2i（下位 nibble）/ 2i+1（上位 nibble）」**（ORT 型 — `unpack4xU8` 2 発の
展開順と対）。llama.cpp Q4_0 の split-half 順（byte j の下位 = 要素 j / 上位 = 要素 j+16）は
採らない — pack 順は上流でも割れている自由パラメータで、**間違えても形も型も合う沈黙誤値**
になる（調査 §4.1）。検出器 MUST: ①隣接要素が全て異なる非対称パターンのユニット
②group 長・行長が語境界と一致しない形（平坦添字の罠 — 0019 の同型検出器の 4bit 版）
③エクスポータ emit 直後の逆変換ビット一致門（`dequant(pack(w)) == fake_quant(w)`）。

### 5. 実行経路 = linear 限定で開始・タイル読み込み時 1 回展開

- 適格は **linear の重みスロットのみ**で開始する（LLM の支配項。0019 の全 5 op 展開は
  需要が出た op から追補 — embedding の w4 は語彙 262k 級で効果が大きいので最初の追補候補）。
  適格外の i4 宣言はロード時 CPU 展開（f32・GPU とビット一致 — 0019 と同じ受け皿）。
- WGSL は core builtin のみ（`unpack4xU8` + マスク / シフト — 調査 §4.1）・展開は
  **共有メモリタイルへの読み込み時 1 回**（0019 流儀・ORT の内積内展開は採らない）・
  語とレーンは平坦添字 MUST。
- scale 適用は要素ごと（`q·s` — 0019 決定と同一。縮約外形への変更は 0019 と同じ改訂条件）。

### 6. Phase 0 = Python fake-quant sweep が実装ゲート

runtime を触らない **format 候補 sweep** を先行する: 検収モデル（MiniCPM5-1B 先行 —
調査 §6.3）で group_size {32, 64, 128} × （必要なら）非対称の品質を fake-quant で比較し、
既定 group_size と〔裁定 B〕の実測根拠を得る。品質管理は ADR 0058 決定 5 の分担
（機械検証 = fake-quant 冪等・逆変換ビット一致 / 品質 = 人間レビュー + 助言的数値）。

## サイズ試算（Gemma 4 E2B 級・weight 2B param 概算）

| 形                              | bpw  | 概算      |
| ------------------------------- | ---- | --------- |
| f16                             | 16   | ~4.0 GiB  |
| i8 + per-channel f32 scale      | ~8.1 | ~2.0 GiB  |
| i4 group 128 + f32 scale        | 4.25 | ~1.06 GiB |
| i4 group 32 + f32 scale         | 5.0  | ~1.25 GiB |
| （参考）i4 group 32 + f16 scale | 4.5  | ~1.13 GiB |

## Consequences

- ADR 0019 は supersede 注記（範囲限定）を受ける。0006 の帰属記述は訂正注記。
- ADR 0063・ir-v1（storage dtype 語彙 + group_size 解禁）・container / safetensors
  リーダ・exporter emit / verify が実装波で対で動く。
- 受入条件: ①既存 f32/f16/i8 資産の読み書き・全 sha 門無風 ②逆変換ビット一致門
  ③pack 順検出器（決定 4 の 3 点）④GPU dequant と CPU 展開のビット一致 ⑤診断
  （`residentCompressedBytes` に i4 + scale が正しく載る）。

## 追記（2026-08-18・Phase 0 完了 — 既定 group_size と裁定 B の実測確定）

決定 6 の sweep を実施した（実測の正本 =
[research/2026-08-18-w4-fake-quant-sweep.md](../research/2026-08-18-w4-fake-quant-sweep.md)・
台本 = `tools/export-recipes/minicpm5/sweep_w4.py`）。確定 2 点（2026-08-18 ユーザー裁定）:

1. **既定 group_size = 32（対称）**。weight 相対 RMSE は 32 < 64 < 128 で単調、自由走行
   greedy の保持は g32 対称だけが相対的に持つ（23/48 — 他構成は 3〜8/48）。サイズ差は
   g128 比 +18%（bpw 5.0・2B 級 ~1.25GiB）で常駐可能化の目的を損なわない。group_size は
   格納欄であり、サイズ優先の資産が個別に 64 / 128 を選ぶことは妨げない（受理集合は
   決定 2 のまま = 2 冪かつ ≥ 16）。
2. **裁定 B（zero-point 欄なし）を実測で確認**: 連続 zero-point という上界測定でも
   weight 空間の改善（−11〜−15%）がモデル級の品質に乗らず、同幅の改善は group を 1 段
   細かくするだけで得られる。決定 3 の予約 3 点は変更なし（将来の追加条件のために残す）。

付随所見: w4 RTN は 1B 級 checkpoint で品質が明確に落ちる（teacher-forced 一致 31〜37/48）。
これは実装の検収方法論（fake-quant 済み重みとの一致 — ADR 0006）には影響しないが、w4 の
採用は品質と引き換えの選択であることが数値で立った。AWQ / GPTQ 級の重み調整は未測定
（実需が立てば別 ADR）。

## 追記 2（2026-08-18・Phase 1 第 2 便 — 書き出し順の訂正）

決定 2 の「書き出し順は I8 の後ろ末尾に I4 を足す」を**訂正する**。正しい並びは**整列単位の
降順** = `F32 → I32 → I4 → 偶数要素 F16 → 奇数要素 F16 → I8`（ADR 0063 を同時改訂）。理由 3 点:

1. **I8 節は任意長**（要素サイズ 1 で整列制約が無い）なので、その後ろに置いた I4 は「テンソル
   先頭が 4 バイト整列」（決定 2）を満たせない。データ節は隙間なく覆う MUST（ADR 0063）
   なので、詰め物で整列を回復する逃げ道も無い。
2. **I4 節を前置しても後続の整列は崩れない**: 量子化軸が 2 冪 ≥ 16 の `group_size` で
   割り切れる（決定 2）⇒ 論理要素数は 16 の倍数 ⇒ バイト長は 8 の倍数。F16 の偶奇トリック
   （奇数要素 F16 を末尾側へ寄せる）もそのまま成立する。
3. **既存資産のバイト列は不変**: I4 を持つ配布形はまだ 1 つも無く、順序は他の dtype の相対
   関係を変えない（f32 のみの資産が `save_file` と同一バイトになる性質も保つ）。

決定 2 の他の項（bit 単位のサイズ表・整列表・view の型・端数を作らない制約）は変更なし。

## 追記 3（2026-08-18・決定 3「amax 厳密復元」の精密化 — 実測）

決定 3 の「amax 要素が厳密復元され fake-quant が冪等」のうち、f32 で常に成り立つのは
**冪等の側**。amax の厳密復元 `fl(7·fl(amax/7)) == amax` は 1ulp ずれることがある
（group 16 の乱数 20 万 group で約 9.5% — 実測の正本は `tools/exporter/tests/test_quantize.py`
の NOTE）。冪等が保たれる機構は「各 group の最大絶対値要素が必ず `|q| = 7` に乗る ⇒
丸め済み重みから引き直した scale `fl(amax(|q·s|)/7) = s` が不動点」で、ADR 0019 の
±127 論証も同型（i8 側の実測 NOTE は `tests/test_emit.py`）。検証門の設計（逆変換ビット一致・
冪等）は不変。

## 追記 4（2026-08-19・波 H — 実モデル検収と混成格納の API 席）

Gemma 4 E2B（`tools/export-recipes/gemma4/` — 1-shot + states 形 decode の 2 系列）で
**実モデル w4 検収を完了**。linear 276 本 = i4 g32（1.88B 要素）・embedding 系 36 本
（主 embedding〈tied lm_head〉+ PLE 35 分割表）= i8（2.75B 要素）・残り f32 の**混成格納**で、
実 GPU の greedy K=16 が torch golden と厳密一致（門 = `e2e_gemma4_test.ts` /
`e2e_gemma4_greedy_test.ts`。census が `:wi4g32` / `:wi8` の両変種と hostExpandedBytes 0 =
適格落ちゼロを固定）。

実装確定 3 点:

1. **混成の API 席**: `export_to_file` / `write_model` に `weight_dtype_overrides`
   （テンソルキー → 格納 dtype・既定 `weight_dtype` に優先）、`fake_quant_int8` / `int4` に
   `include`（モジュール FQN 述語）。各 dtype の適格規則・scale 形・検証門は本 ADR と
   0018/0019 の**既存決定のまま**（割り付けの表現だけが増えた）。明示指定は**満たせなければ
   fail loudly**（未知キー・適格外・i4 × 非 linear）・既定側は従来の「適格外は静かに f32」
   （I4-ELIG-01 挙動）を維持。`"f32"` の明示 = 圧縮既定からの除外。
2. **tied lm_head の割り付け**: embedding と linear の両重みスロットで消費される tied 実体は
   i4 不適格（決定 5）・i8 は両者適格。fake-quant の include で **i4 側から除外して i8 に
   一本化**する（両方に通すと二重丸めで scale 台帳が実値と食い違う）。格納指定は「既定 i8 +
   linear を i4 明示」の向き — i4 側のキーが fake-quant 台帳そのものになり、tied 実体に
   export が付ける FQN を書く前に知らずに済む。
3. **RoPE 位置表は量子化しない**: decode 系列の表引き化で cos/sin 表が embedding の
   重みスロット（= i8 適格集合）に入るため、`"f32"` 明示除外が必須
   （`export_decode.rope_table_keys`）。位置表の丸めは重みの丸めと違い**角度誤差が位置に
   沿って蓄積する**。読み手側は census の「f32 embedding = 表 4 本ちょうど」で固定。

## 追記 5（2026-08-19・測定側の拡張 — 対象 op の opt-in と方式スクリーニングの受け皿）

w4 横展開 + 量子化方式スクリーニング波（backlog now 節）に伴う **fake-quant 側だけの拡張**。
格納の受理集合（決定 2）・実行経路の linear 限定（決定 5）・emit の適格判定は**一切変えない**。

1. **`fake_quant_int4` に `op_types` の明示 opt-in を追加**（既定 `(nn.Linear,)` のまま =
   既存呼び出しの挙動不変）。決定 5 の「追補は需要が出た op から」の**測定面の需要**が先に
   立った形 — i8 と同じ 5 op 種まで広げて「非 linear の w4 は品質がどこまで戻るか」を
   runtime 非接触で測る。group の軸は「in 軸」を一般化する（`quantize.channel_rows`:
   conv は出力チャネルごとの受容野 `Cin·K` / `Cin·Kh·Kw` を平坦化・`ConvTranspose1d` は
   転置レイアウトの軸 1・embedding は語彙エントリごとの D 軸）。conv 系の scale は受容野
   平坦化の rank 2 になり**重みと rank が合わないため emit へ構造的に渡せない**
   （誤用が沈黙しない側に倒してある）。
2. **測定専用の丸め方式群を `quant_methods.py` に新設**（FP4 e2m1 / NF4 / MXFP4 /
   k-means codebook〈per_tensor・per_channel・shared + g32 正規化の 3 粒度〉）。
   `quantize.py` と別モジュールなのは「載っている形 = 出荷できる格納形」という読み方を
   壊さないため。戻り値は計数のみで **scale / codebook の台帳を返さない**（emit へ流れる
   口を作らない）。全方式とも決定的（k-means は分位点初期化 + 固定反復）・対象選択は
   `iter_quant_targets` を格納経路と共有する。
3. **位置づけ**: 決定 6（Phase 0 sweep）の方式次元への続編。追記 1 の g 軸確定（g32）を
   前提に方式比較は g=32 固定で行い、g 軸の再評価は方式確定後に別途（2026-08-19 ユーザー
   裁定）。校正ループ系（AWQ / GPTQ）は引き続き未測定 — 実需が立てば別 ADR（追記 1 のまま）。
   codebook 系が採用に至る場合の格納の新席（表の companion 欄・view 型）は決定 2 の 3 面の
   reopen として別途起票する。
