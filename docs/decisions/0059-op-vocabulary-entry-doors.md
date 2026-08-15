# 0059 — op 語彙の入場門モデル（層番号の退役と Core ATen 1 判定への一本化）

- Status: accepted（2026-08-14・ユーザー裁定「第 1' 層が出た時点で微妙 — 現状把握からの
  再分類」→ 入場門モデル・名前制で承認）
- 対象: op 追加の判定手続き（docs のみ）。**実装・既存資産・既存 op の語彙上の席は全て
  不変** — 変わるのは今後の追加の紙のルールだけ。
- 置換: [ADR 0043](0043-op-addition-layers.md) の層定義・判定手順（同 ADR 本文は当時の
  記録として保存・冒頭に本 ADR への注記のみ）。
  [known-issues](../known-issues.md) に起票されていた層分類の 2 穴はこれでクローズ。

## Context

ADR 0043（+ 2026-08-12 追記）の層は、ADR 要否を「原子/分子 × Core ATen 内/外 × 要求元」の
合成で決めていた。原子/分子の境界（「厳密同値で書けるが容量・性能で非成立」）は桁の判断で
割れうるのに、その答えが紙の手続きを左右する。第 1' 層はその歪みを要求元軸で塞いだ
パッチで、モデル拡充波の recon で 2 穴（Core ATen 外・モデル由来の原子の行き場 /
「非成立」の線引きが 2 文書で非等価）が顕在化した。

## 決定

1. **層は番号でなく名前で呼ぶ**: export 消滅層 / Core ATen 層 / 拡張原子層 / 拡張分子層 /
   融合層。旧番号との対応: 第 0 = export 消滅層・第 1 = Core ATen 層・第 1' = 拡張原子層・
   第 2 = Core ATen 層（core の attr 変種）と拡張分子層（非 core）に分属・第 3 = 融合層。
   旧番号は過去の ADR / 台帳 NOTE の時点記録に残るが、新規記述では使わない。
2. **ADR 要否の軸は「Core ATen 帰属」の 1 判定だけ**: 入場券 = `torch.Tag.core in op.tags`
   の実測（機械判定 — 台帳に実測コマンドを記す）。core の op と、その attr 変種の別 op 名
   （`gelu_tanh` 前例 — attrs 空契約 ADR 0012 の帰結）は **Core ATen 層 = 台帳 NOTE のみ**。
   core op 由来の IR 名には overload 変種の別名（`ge.Scalar` → `ge_scalar` 等）と純改名
   （`_to_copy` → `cast`・`mm` → `matmul`・`view` → `reshape`）も含む
   （ADR 0015 の「語彙追加の残り = ADR 対象外」裁定を継承）。
   NOTE には従来どおり「手順 1/2（畳み込み・分解）を却下した実測」を書く（様式の前例 =
   `upsample_bilinear2d` の NOTE）。コミットメント上限の根拠は有限収束（Core ATen 160 +
   その attr 面）。**それ以外（非 core の aten・torchvision・karume 独自名）= ADR 必須**。
3. **原子/分子は層でなく、ADR 内の証明の型に降格する**: 拡張原子層 = 語彙内の他 op の
   合成で厳密同値に**書けない**ことの証明（`deform_conv2d` 型）。拡張分子層 = 書けるが
   保存する実測根拠 — 入場条件は旧第 2 層の①〜④（復元が脆い・意味が落ちる / 中間実体化が
   非成立 / 再出現率 / attr 変種分離）を継承（`gru_scan` 型）。**要求元（IR 機構かモデルか）
   は層を分けない**（`sym_prefix_slice` と `deform_conv2d` は同じ門 — 区別は ADR 内の
   一文で足りる）。
4. **「容量・性能で非成立」の判定は紙の手続きを左右しない**: Core ATen 内なら答えが
   どちらでも同じ門に落ちる。分解で逃げるか op を足すかの工学判断には拘束的閾値を
   置かず、**前例ガイド**を手順に添える — 合成が中間・FLOP をほぼ増やさない書き換え
   （reciprocal → 除算形・softplus 分解・RoPE 実数化型）は分解側が既定。中間が膨らむ
   場合は **1.5〜2 倍の時点で既に保存側の前例がある**（leaky_relu — ADR 0015・メモリ
   見積の前提が崩れるとして保存リスト入り）。桁で膨れる合成（upsample_bilinear2d の
   FLOP 184 倍・batch_norm の 537MB — 台帳の反例集）は**分解の却下材料**であって、
   op を足すか否かは需要側（再出現率・入場条件）で決める。
5. **記録の訂正（torch 2026-08-14 実測）**: ADR 0043 追記の「分解禁止 12 op のうち 9 本は
   Core ATen 由来」は意味面の対応（conv1d → convolution 等）で数えた値で、厳密な
   `Tag.core` は **12 中 3**（embedding / gelu / leaky_relu）。非 core の保存 9 本（conv1d /
   conv2d / conv_transpose1d / layer_norm / linear / masked_fill / rms_norm / softmax /
   attention〈SDPA・ターゲット別〉）は「エクスポータが分解を止める選択をした分子」
   そのものであり、**拡張分子層**に置く（根拠は既存 ADR・台帳 — 遡及手続きなし）。
   この帰結として機序が一貫する: **非 core の op（`Tag.core` 実測で判定 — IR 名の綴り
   ではない）が IR に現れるのは karume が意図して選んだときだけ**（保存 or 発明）—
   選択には記録（ADR）が伴う。
6. **既存 op の再配置**（席・実装・資産・ADR 要否の遡及は全てなし）:

   | 門           | 現在の在庫                                                                                                                                                                                                                                             |
   | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
   | Core ATen 層 | 実装済みの core 群（sin / upsample_bilinear2d / embedding / gelu を含む）+ core 由来の別名・overload 変種（cast / ge_scalar / gt_scalar / le_scalar / matmul / reshape 等）+ leaky_relu・gelu_tanh（旧第 2 層 — ADR 0015 / 0043 は歴史記録として残置） |
   | 拡張原子層   | sym_prefix_slice（ADR 0010）・deform_conv2d（ADR 0055）                                                                                                                                                                                                |
   | 拡張分子層   | safe_softmax（ADR 0044・`aten._safe_softmax` は非 core）・gru_scan / gru_scan_reverse（ADR 0056）・非 core の保存 9 本（決定 5）                                                                                                                       |
   | 融合層       | fusion.ts の silu / rope / upsample2x / adaln の **4 ルールちょうど**（ADR 0040）。融合 attention（ADR 0023）はここではない — IR op `attention` の**実行時カーネル変種**で、実行変種は語彙の門に関与しない                                             |

7. **不変のまま継承**: 「やらない方がよいこと」（対称性追加・160 埋めの目標化の禁止）は
   全門に優先 / 融合層の観測点必須（ADR 0040）/ 分解禁止リストは層と独立の軸
   （ADR 0043 追記 決定 3）/ Core ATen 層の契約 1 セット（OP_CONTRACTS + ops.py +
   shapes.py + fixtures + CPU 参照 + golden COVERAGE）。

## 帰結

- [op-vocabulary.md](../op-vocabulary.md) の判定手順節を本 ADR の門表・機械判定・
  前例ガイドへ全面改訂（旧番号の対応表つき）。過去の日付つき NOTE は時点記録として
  残置し、対応表で読む。
- known-issues の該当エントリは記録節へ移設（解消）。**（済 — 移設先の記録節は
  known-issues の置き場規約「解決したら該当節を削除」に従って削除済み。層分類 2 穴の履歴は
  本 ADR が持つ）**
- 次の適用対象: 波③ K-5（online attention の実行変種 = 融合層ではなく**既存 op の実行時
  変種** — 門の新設なし・opt-in 席は ADR 0058）・モデル拡充の次候補（Kokoro-82M の LSTM
  拡張は拡張分子層の ADR 1 本）。
  **Follow-up（2026-08-15）**: 波③ K-5 は**消化済み** — 行ブロック実行
  （[ADR 0060](0060-row-block-attention.md)）が既存 op `attention` の実行時変種として着地し、
  予告どおり**門の新設は無し**（ビット同一のため ADR 0058 の opt-in 席も使っていない）。
