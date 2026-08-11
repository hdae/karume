# 0044: DeBERTa text_encoder は 22 層で配る（末尾 2 層カット）

- Status: accepted（WAV sha256 門が**不変のまま緑** 2026-08-11 — 下の Consequences）
- Date: 2026-08-11
- 関連: [0026](0026-w8a8-deberta-deployment.md)（i8 資産と聴感ゲート）/
  [0039](0039-sbv2-distribution.md)（SBV2 の配布形 — 本 ADR はそのサイズ表を更新する）/
  [0013](0013-sbv2-chain-export.md)（相対位置表をグラフ入力へ昇格した前例）/
  [research/2026-08-11-deberta-size-recon.md](../research/2026-08-11-deberta-size-recon.md)（実測の正本）

## Context

SBV2 が DeBERTa から読むのは `hidden_states[-3]` **1 本だけ**である（参照実装
`nlp/japanese/bert_feature.py` の `res["hidden_states"][-3:-2]`）。`hidden_states` は
`[0]` = embedding 出力・`[i+1]` = layer i の出力という並びで長さ 25 なので、`[-3]` は
**index 22 = layer 21 の出力**を指す。つまり `encoder.layer.22` / `.23` の重みは配布形で
一度も参照されない — i8 資産 334,545,336 B のうち 25,346,048 B（7.6%）が死重みだった。

発端は「SBV2 の ONNX 推論で使われる `tsukumijima/deberta-v2-large-japanese-char-wwm-onnx` は
hidden[-3] だけを出す変換だが、サイズを縮める材料になるか」という問い（ユーザー・2026-08-11）。
実測すると **ONNX 版のグラフは確かに 22 層だがファイルは 1.03% しか小さくない**（onnxsim の
定数畳み込みが位置射影を各層 `[16,512,64]` fp32 ×2 本 = +92.3MB として焼き込み、2 層削除の
−100.8MB をほぼ相殺している）。**参考にすべきは「22 層で足りる」という事実だけ**で、成果物と
しての ONNX 版に寄せる動機は無い（karume の i8 は ONNX fp16 版の約半分）。

## Decision

### 1. 配布形の text_encoder は 22 層 variant（`sbv2-22layer`）

`export_deberta.py` の `VARIANTS` に 22 を足し、`dist.py` の `SBV2_TEXT_ENCODER_VARIANT` を
そこへ向ける。`full-24layer` は**残す** — 層別の誤差の伸びを golden から読む用途（ADR 0026
決定 2 の tolerance 導出の根拠）は 24 層でしか成立しない。

切り詰めは `load_model` が既に持っていた実装（`model.encoder.layer[:num_layers]`）で、
新規のコードは無い。

### 2. 取り出し位置は「参照側」と「配布グラフ側」を別の定数に分ける

`sbv2_demo.py` の `BERT_HIDDEN_FROM_END = 3` は**参照実装の定義そのもの**（torch の全 24 層
モデルに対する位置）なので動かさない。配布グラフから同じテンソルを引く位置は
`BERT_GRAPH_HIDDEN_FROM_END = 1`（22 層グラフの最終出力）として別に持ち、`symbols.json` へは
後者を書く。

**1 つの定数で兼ねてはならない。** 参照計算（`sbv2_demo.py` / `measure_quant_sbv2.py`）が
`symbols.json` の値を読んでいた実装のままだと、層を削った瞬間に**参照だけが別の層を指し**、
パリティ台本が「合っている / ずれている」を反転して報告する。

### 3. 組み立て門: `出力本数 − bertHiddenFromEnd == 22`（MUST）

層数は `export_deberta.py` の variant が、取り出し位置は `sbv2_demo.py` の定数が持つ**別々の
台本**なので、片方だけ動かした配布形が普通に組み上がる。ずれても shape は合ったままロードも
実行も通り、**別の層の BERT 特徴で音が出るだけで沈黙する**（スタイル表・話者表の行数門と同じ
機序）。

`dist.py` の `assert_bert_hidden` が、text_encoder の IR メタデータから `outputs` 本数を読み、
`symbols.json` の `bertHiddenFromEnd` と突き合わせて絶対位置 `SBV2_BERT_HIDDEN_INDEX = 22` を
検査する。**この絶対位置は層数に依らない不変量**なので、24 層 / 3 でも 22 層 / 1 でも通り、
片方だけ動いた組み合わせだけが落ちる。

## Consequences

### サイズ（実測）

|                              |   24 層（旧） |     22 層（新） |                    差 |
| ---------------------------- | ------------: | --------------: | --------------------: |
| `text_encoder`               |   334,545,336 | **309,167,272** | −25,378,064（−7.59%） |
| `w8` の取得量（FN4）         |   400,526,905 | **375,148,841** | −25,378,064（−6.34%） |
| 配布形 `karume-sbv2-fn` 全体 | 2,266,418,460 |   2,241,040,396 |                −1.12% |

削減が層 2 枚ぶん（25,346,048 B）より 32,016 B 多いのは、safetensors ヘッダも縮んだため
（391,616 → 359,600 B — テンソル 44 本ぶんのエントリと IR ノード記述が減る）。グラフは
1,130 ノード / 23 出力（24 層は 1,230 ノード / 25 出力）。

取得量で見ると text_encoder は依然として `w8` の **82.4%** を占める。

### 数値 — WAV は 1 bit も動かない

**移植の門 `e2e_sbv2_wav_test.ts` は参照 sha256 `a82f72e2…` のまま緑**（FN4 / w8 / seed 0 /
WAV 234,540 B）。参照値の差し替えも tolerance 化も行っていない — この門が**不変のまま通ること
自体が層カットの無害性の証明**である。

golden io でも 24 層の `output.22` と 22 層の `output.22` が 4 ケース全部でバイト完全一致
（case0 45,056 B / case1 106,496 B / case2 143,360 B / padded 65,536 B）。

torch 段では f32 / i8 fake-quant / i8+a8 の 3 構成で `max abs diff 0.0` を実測済み
（research §4）。ビット一致が壊れない根拠は 5 点あり、最大の弱点候補だった `encoder.LayerNorm`
は **hidden_states に掛からない**（`get_rel_embedding()` の中で相対位置テーブルにだけ適用される）
ことで潰れている。

### 適用範囲の限定

ビット一致は **f32 重み・f32 計算の torch 段**と、**今回の実測環境での WAV** についての命題で
ある。f16 経路や別バックエンドへ横滑りさせない（ADR 0022 の「ビット同一は実測命題」と同じ
性格）。

### 積み残し

- **HF の 2 リポ（`hdae/karume-sbv2-fn` / `hdae/karume-sbv2-jvnv`）は未アップロード。** 両方を
  同時に更新する必要がある — 片方だけ直すと、放置側は「24 層 + 旧 sha256」で整合したまま沈黙
  する（hub の検証は通ってしまう）。
- ライセンス上の性格は変わらない（cc-by-sa-4.0 の ShareAlike は i8 量子化の時点で既に改変物の
  再配布であり、層カットで新たな条件は増えない）。ADR 0039 の「公開前に裁定が要る」は未解決の
  まま持ち越し。
- DeBERTa の数値 golden 門（`e2e_deberta_test.ts`）は依然としてリポに無い。text_encoder の
  回帰を捉える網は WAV sha256 の 1 本だけで、原因の局在ができない状態が続く。
