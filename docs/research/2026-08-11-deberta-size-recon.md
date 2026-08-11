# DeBERTa text_encoder のサイズ recon — ONNX 版の実態と karume 側の削減余地

> 時点スナップショット（2026-08-11）。発端は「SBV2 の ONNX 推論で使われる
> `tsukumijima/deberta-v2-large-japanese-char-wwm-onnx` は hidden[-3] だけを出す変換らしいが、
> 配布サイズを縮める材料になるか」という問い。外部一次情報の掃引 4 レッグ + ローカル深掘り
> 3 レッグ + 敵対裁定 2 レッグ（Workflow）の統合で、**数値はすべてメインセッションで実物に
> 当て直した**。ONNX の実測はダウンロードした `model.onnx` の protobuf を直接開いたもの。

## 0. 要約

- **ONNX 版のグラフは確かに 22 層**（末尾 2 層は重みごと不在）。ただし**ファイルは小さくない** —
  1.305GB で元の 1.318GB と 1.03% しか違わない。onnxsim の定数畳み込みが位置射影を各層
  `[16,512,64]` fp32 ×2 本として実体化し、+92.3MB が 2 層削除の −100.8MB をほぼ相殺している。
- **「22 層で足りる」という事実自体は正しく、karume 側で独立に適用できる**。i8 資産では
  **−25,346,048 B（−7.58%）**。
- 22 層への切り詰めが `hidden_states[-3]` と**ビット一致すること**は torch で実測済み
  （f32 / i8 fake-quant / i8+a8 の 3 構成で `max abs diff 0.0`）。
- 層以外にも 3 件の削減余地がある（合計 −9.0%）。うち `const.*` 2 本（2MiB）は
  **ONNX が踏んだのと同種の焼き込みの軽症版**で、flow/voice 側は同じ理由で既にグラフ入力へ
  昇格済み（ADR 0013）— DeBERTa 側だけが取り残されている。

## 1. ONNX 版（tsukumijima）の実態 — 実測

`https://huggingface.co/tsukumijima/deberta-v2-large-japanese-char-wwm-onnx` の `model.onnx`
（1,304,829,184 B・sha256 `c5c880ef4bd0d3308ec6503a8728efae920bc5c5a984de4f76fc3d0ad518a2ec`）を全取得し、
`onnx.load(load_external_data=False)` で開いた実測:

| 項目                          | 実測値                                                           |
| ----------------------------- | ---------------------------------------------------------------- |
| ir_version / opset / producer | 8 / 17 / pytorch                                                 |
| nodes / initializers          | 3,618 / 432                                                      |
| graph inputs                  | `input_ids`, `attention_mask` の 2 本（`token_type_ids` は消滅） |
| graph outputs                 | **`output` 1 本**・`[dyn, 1024]`                                 |
| initializer payload 合計      | 1,303,507,188 B                                                  |

initializer の shape 分布が層数を確定させる:

| shape               | dtype |     本数 |         合計 B | 意味                                          |
| ------------------- | ----- | -------: | -------------: | --------------------------------------------- |
| `[1024,1024]`       | f32   |   **88** |    369,098,752 | q/k/v/attn-out × **22 層**                    |
| `[4096,1024]`       | f32   |   **22** |    369,098,752 | FFN intermediate                              |
| `[1024,4096]`       | f32   |   **22** |    369,098,752 | FFN output                                    |
| `[16,512,64]`       | f32   |   **44** | **92,274,688** | **onnxsim が焼き込んだ位置射影**（2 本 / 層） |
| `[22012,1024]`      | f32   |        1 |     90,161,152 | word_embeddings                               |
| `[1024,1024,3]`     | f32   |        1 |     12,582,912 | encoder.conv                                  |
| `[1024]` / `[4096]` | f32   | 203 / 22 |      1,191,936 | bias・LayerNorm                               |

`cls.*`（MLM ヘッド）・`pooler`・`rel_embeddings.weight`・`encoder.LayerNorm.*` は
initializer に存在しない（前 2 者は未使用で DCE、後 2 者は定数畳み込みで消えて `[16,512,64]` に
化けている）。

### 収支（元 safetensors 比）

元 `ku-nlp/deberta-v2-large-japanese-char-wwm` の `model.safetensors` は 1,318,453,704 B
（params 329,601,020 × 4 + header 49,624）。

```
1,318,404,080  元の生データ
 −100,769,792  末尾 2 層（12,596,224 params × 2 × 4B）
   −2,105,344  rel_embeddings + encoder.LayerNorm（畳み込みで消滅）
   −4,294,640  cls.*（MLM ヘッド）
       −2,048  position_ids
  +92,274,688  onnxsim が焼き込んだ [16,512,64] × 44
──────────────
 1,303,506,944  ≒ 実測 initializer 合計 1,303,507,188（差 244 B = スカラー定数）
```

**結論**: 「ファイルが小さいこと」は層削除の証拠になっていない（削減分の約 92% が相殺）。
ONNX 側の `config.json` は `num_hidden_layers: 24` のままで実グラフ（22 層）と食い違うため、
**config を信じると誤る**。

### 変換台本が実際にやっていること

`convert_bert_onnx.py`（tsukumijima/Style-Bert-VITS2 master）は **層を削っていない**。
`ONNXBert.forward` が `res["hidden_states"][-3:-2]` を返すだけで、`encoder.layer` のスライスは
一行も無い。末尾 2 層が消えたのは torch.onnx.export のトレース + `onnxsim.simplify` による
到達不能ノード除去の**副作用**である。量子化は fp16 変換のみ（int8 系は無し・
`model_fp16.onnx` 653,075,699 B = fp32 の 0.50053 倍）。

> 参考: karume の i8 資産 334,545,336 B は、ONNX の fp16 版 653MB の**約半分**。サイズ面で
> ONNX 版へ寄せる動機は無い（そもそも IR v1 なので読めない）。

## 2. karume 側の現状台帳 — 実測

`models/karume-sbv2-fn/shared/text_encoder/model.i8.safetensors` = **334,545,336 B**
（header 391,616 + payload 334,153,720 / 544 テンソル）。f32 系列
`outputs/series/deberta/full-24layer/model.safetensors` = 1,316,567,296 B に対し **25.41%**。

| 区分                                         | 本数 |      バイト |
| -------------------------------------------- | ---: | ----------: |
| layer00〜23（各 12,673,024 B × 22 テンソル） |  528 | 304,152,576 |
| 非層                                         |   16 |  30,001,144 |

層 1 本の内訳: i8 重み 12,582,912（`[4096,1024]`・`[1024,4096]` 各 4,194,304 + q/k/v/attn-out
`[1024,1024]` 各 1,048,576 の 6 本）+ scale 台帳 36,864 + 非圧縮 F32 53,248（bias 7 + LayerNorm
γ/β 4）。bias と LayerNorm が i8 化されないのは emit の適格判定（消費が `WEIGHT_SLOTS` の重み
スロットのみ — `karume/emit.py:105-126`）による設計上の帰結。

非層 30,001,144 B の全内訳:

| テンソル                                         | dtype / shape        |     バイト |
| ------------------------------------------------ | -------------------- | ---------: |
| `model.embeddings.word_embeddings.weight`        | I8 `[22012,1024]`    | 22,540,288 |
| `model.encoder.conv.conv.weight`                 | I8 `[1024,1024,3]`   |  3,145,728 |
| `model.encoder.rel_embeddings.weight`            | **F32** `[512,1024]` |  2,097,152 |
| `const.7051e6e79ba7d6b9`                         | I32 `[1,512,512]`    |  1,048,576 |
| `const.870ecefa531efc74`                         | I32 `[1,512,512]`    |  1,048,576 |
| word_embeddings の scale                         | F32 `[22012,1]`      |     88,048 |
| その他（conv scale・LayerNorm 7 本・スカラー 2） |                      |     32,776 |

グラフ出力は **25 本**（`[0]` = embedding 出力、`[i+1]` = layer i 出力）。SBV2 は
`symbols.json` の `bertHiddenFromEnd: 3` で `outputs[22]`（= `layer_norm_46` = layer 21 の出力）を
採る。したがって **encoder.layer.22 / 23 の重み 25,346,048 B は配布形で完全に死んでいる**。

## 3. 削減施策と実測値

| 施策                                |      削減 B |  単独適用後 | コスト | 判定               |
| ----------------------------------- | ----------: | ----------: | ------ | ------------------ |
| **A** 末尾 2 層カット               | −25,346,048 | 309,199,288 | 低     | **波 1 で実施**    |
| **B** `const.*` 2 本をグラフ入力化  |  −2,097,152 | 332,448,184 | 低〜中 | **波 3 で試行**    |
| C `rel_embeddings` を i8+scale 化   |  −1,570,816 | 332,974,520 | 高     | **却下**（下記）   |
| D 到達不能な語彙 970 行の刈り込み   |    −997,160 | 333,548,176 | 中     | 保留               |
| D2 語彙 13,938 行への攻めた刈り込み |  −8,300,072 | 326,245,264 | 中     | **非推奨**（下記） |

A+B+C+D = 304,534,160 B（−9.0%）。取得量では text_encoder が preset `w8` の **83.5%** を
占めるため、A 単独でも取得量 400.5MB → 375.2MB（−6.3%）。

> **A の実測（2026-08-11・波 1 着地 — ADR [0044](../decisions/0044-deberta-layer-trim.md)）**:
> 実サイズは **309,167,272 B**（−25,378,064 / −7.59%）で、上の予測 309,199,288 B より
> 32,016 B 小さい。差は safetensors ヘッダの縮み（391,616 → 359,600 B — テンソル 44 本ぶんの
> エントリと IR ノード記述が減る）で、テンソル実体は予測どおり。グラフは 1,130 ノード /
> 23 出力。`w8` の取得量は 375,148,841 B。**WAV sha256 は `a82f72e2…` のまま緑**、golden io の
> `output.22` も 4 ケース全部でバイト完全一致（§4 の torch 実測が実資産でも再現）。

### B の根拠 — 2MiB の表は 4KB 相当

`const.7051e6e79ba7d6b9` / `const.870ecefa531efc74` は disentangled attention の gather 添字表で、
IR 上の消費は `sym_prefix_slice → expand → gather(bmm, expand)`。実測で:

- 値は `make_log_bucket_position(rel = j − i, bucket_size = 256, max_position = 512) + 256` を
  `[0,511]` にクランプしたものと **262,144 要素すべてで一致**（不一致 0）
- t1 は Toeplitz（対角ごとに定数・違反 0・相異対角 1023 本）、t2 は t1 の**転置**（違反 0）
- 相異値は 511 個のみ ⇒ **本質は 1023 要素の i32 ベクトル（4,092 B）と等価**

同じ問題を flow/voice 側は既に解いている: O(T²) の相対位置表は「焼き込むと定数だけで 134MB」に
なるためグラフ入力へ昇格し、ホスト側 `buildRelattnTables` が生成して Python 正本とのバイト一致を
パリティテストで固定している（`packages/models/src/sbv2/relattn-tables.ts:14-23` / ADR 0013）。
**DeBERTa 側だけが Tmax=512 で焼き込まれたまま取り残されている。**

なお表を残す場合も Tmax を下げれば二次で縮む（512→2,097,152 / 256→524,288 / 128→131,072 B）が、
入力長上限の仕様変更になるので B（グラフ入力化）が本筋。B なら Tmax 制約自体が消える。

### C を却下する理由

`rel_embeddings.weight` が F32 なのは値の性質ではなく、消費 op が `WEIGHT_SLOTS` でないため
（実 IR での消費は `layer_norm` の data スロット 1 箇所 — ADR 0026 の「linear の入力スロット」は
実資産では不正確）。per-row i8 対称量子化なら 512 行すべてで**ビット完全一致で往復する**
（最大差 0.0）ことは実測したが、実現には `emit.py:109-111` の「消費が 1 つでも重みスロット以外に
あれば適格外」という**全モデル共通の不変条件**を緩める必要がある。1.6MB（全体の 0.47%）の
見返りに対し、緩め方を誤ると別モデルで f32 として読むカーネルに i8 バイト列が渡る
（ビット列の読み替え = 最悪クラスの沈黙バグ）。**blast radius が見合わない。**

### D / D2 について

トークナイザ実装上**証明可能に到達不能な語彙行が 970 行**ある（① `encode()` は 1 コードポイント
単位でしか lookup しないので複数コードポイントの行は引けない ② `tokenize()` の NFKC 正規化で
NFKC 非安定な 38 行は引けない ③ `clean_text` の removed/spaced 範囲に入る 945 行は文字列到達前に
除去される。CLS/SEP/UNK は ID 直参照なので残す）。数値的には無損失だが、重みの行刈りと
tokenizer 資産の ID 再採番を**同時に**行う必要があり、片方だけ差し替わると shape は合ったまま
全トークンが別の埋め込みを引く（沈黙誤値）。1.0MB のために取るリスクではないので保留。

D2（ハングル・非日本語文字体系・絵文字・CJK 拡張を捨てて 13,938 行）は 8.3MB 削れるが、
未知文字は `[UNK]` に落ちるだけでクラッシュしないため**劣化が観測されないまま配布に乗る**。
採るなら UNK 率をパイプラインの診断値として常設するのが前提。

### f16 系列は不利（ADR 0039 決定 2 の追認）

実資産の台帳から導出した f16 のサイズ: 層あたり 25,219,072 B（scale 不要）・非層 55,595,016 B
⇒ 24 層で 660,852,744 B、22 層でも 610,414,600 B。**末尾 2 層カット済み f16 でも無施策 i8 の
1.82 倍**で、text_encoder が配布形の 63% を占める以上、f16 追加は配布総量を約 2 倍にする。
ADR 0039 決定 2「text_encoder は i8 単体」を数値が支持する。

## 4. ビット一致の実測（波 1 の前提）

`transformers==5.14.1`・torch CPU・`attn_implementation="eager"`・eval で、24 層モデルの
`hidden_states[-3]` と 22 層に切り詰めたモデルの `last_hidden_state` を突合:

| 構成                     | 結果                                                                                          |
| ------------------------ | --------------------------------------------------------------------------------------------- |
| f32 素の重み             | `torch.equal` **True** / bytes 一致 / max abs diff **0.0**                                    |
| i8 fake-quant 後         | 同上（共有 scale 135 本が全てビット一致・full 147 − trunc 135 = 12 = 削った 2 層 × 6 linear） |
| i8 + per-token a8 フック | 同上                                                                                          |

padding mask 混じり（末尾 3 トークン mask=0）でも一致。

弱点候補として潰した論点:

1. **`encoder.LayerNorm` は hidden_states に掛からない** — `get_rel_embedding()` の中で
   相対位置テーブルにのみ適用される（`modeling_deberta_v2.py:591-601`・`norm_rel_ebd="layer_norm"`）。
   層数に依存する適用差は無い。**ここが最大の弱点候補だったが refute された。**
2. `rel_embeddings` / `relative_pos` はループ外で 1 回だけ計算され全層共有・`num_hidden_layers` を
   参照しない（同 :645, :651）。
3. `ConvLayer` は `i == 0` ハードコード（同 :665-666）。層数を変えても適用位置は動かない。
4. `z_steps` 経路（`encoded_layers[-2]` を使う分岐）は `self.z_steps = 0` がクラス内ハードコードで
   恒久的に死んでいる（同 :710, :774）。config でも上書き不可。
5. i8 の scale は per-module の重み由来（`weight.abs().amax`）で完全にローカル。層を削っても
   残る層の scale は 1 ビットも動かない。

**適用範囲の限定（事実と推測の境界）**: 上記は torch f32/i8 段までの実測。karume ランタイム
（WGSL）上でも同一の結論になるかは**未実測**であり、融合パスの掛かり方や出力 pin 集合の変化が
数値に影響しないことは波 1 で確認する。ただし前半 22 層のノード列・重み・実行順が同一である
以上、**WAV sha256 は不変であることが期待値**であり、変わった場合は融合かバッファ配置を疑う。

## 5. 沈黙する罠

- **`bertHiddenFromEnd` の取り違え（severity: high）**: 層を 22 に切るとグラフ出力は 25 → 23 本に
  なる。`bert-tile.ts:16-24` は「末尾から N 本目」を引くので、`symbols.json` の値を 3 のまま配ると
  **layer 20 の出力を静かに拾う**。ガードは長さと整数性しか見ず、shape も合い、ロードも実行も
  通る。組み立て時に「`graph.outputs` の本数」と「`bertHiddenFromEnd`」の整合を検査する門が要る
  （現状 `dist.py` が突合するのは実行時ノブ 6 本のみ）。
- **torch 参照側の切り詰め忘れ**: `sbv2_demo.py:384-395` と `measure_quant_sbv2.py:393-404` は
  HF モデルを切り詰めずに読み `hidden_states[-bertHiddenFromEnd]` を引く。配布形だけ 22 層にすると
  参照が別層になり、パリティ台本が「合っている / ずれている」を反転して報告しうる。
- **系列名の 3 点同期**: `VARIANTS`（`export_deberta.py:84`）・`dist.py:783`・
  `test_dist.py:871,1291` が variant 名を個別に持つ。片方だけ直すと組み立てが落ちるか、
  テストだけ古い名前を主張する（どちらも loud なので検出は容易）。
- **2 リポの片肺更新**: `models/karume-sbv2-fn`（10 箇所）と `models/karume-sbv2-jvnv`（4 箇所）が
  同一バイトの text_encoder を独立コピーで持つ。片方だけ更新すると、放置側は「24 層 + 旧 sha256」
  で整合したまま沈黙する（hub 検証は通ってしまう）。逆に資産だけ差し替えて manifest 未更新なら
  hub が落ちる（loud）。

## 6. 波の切り方

ユーザー裁定（2026-08-11）により 3 波に分割し、1 → 2 → 3 の順で進める。**分割の理由は
切り分け可能性**であり、まとめて入れると WAV sha256 が動いたときに原因を特定できない。

| 波 | 内容                                            | 期待される門の挙動                         |
| -- | ----------------------------------------------- | ------------------------------------------ |
| 1  | A: 末尾 2 層カット（重みが変わる）              | WAV sha256 **不変**が合格条件              |
| 2  | 出力絞り 25 → 1 本（重みは変わらない）          | sha256 が動いたら融合 / バッファ配置が原因 |
| 3  | B: `const.*` 2 本のグラフ入力化（入力が増える） | 試行して採否を判断                         |

### 波 2 の動機はサイズではない

`executor.ts:3046` は `graph.outputs` を**全部** readback する。配布形は 25 本出しのままなので、
1 本しか使わないのに毎回 25 本分の staging + mapAsync を払っている。ADR 0026 が
「壁時計が 1.16× に留まるのは全 25 層の hidden_states を出力に持ち読み戻し ~52MB が支配するから
（出力を絞る用途なら GPU 比がそのまま出る）」と記録した件そのもので、SBV2 がホスト律速である
こと（ACTIVE_DESIGN: 壁 1.08s vs GPU 0.42s）を踏まえると、**7.6% のサイズより体感に効く可能性が
ある**。重みテンソルは変わらないので、削減量は 0 B。

> **波 2 の実測（2026-08-11・着地 — ADR 0044 決定 3/4）**: 期待は**外れた**。A/B（warmup 2 +
> 8 run の中央値・順序入れ替えでも同傾向）は T=15 で 69.3 → 66.3ms（−4.5%）、T=35 で
> 71.6 → 68.7ms（−4.1%）、T=512 で 150.1 → 138.6ms（−7.7%）。readback は T=15 で
> 1,380 → 60 KiB。SBV2 の実用 T では bert 段 3〜4ms = パイプライン全体の 0.3% で、ADR 0026 が
> 期待した「GPU 比がそのまま出る」には届かない（あれは T=512 で読み戻し 52MB が支配する前提）。
> 退行が無く readback が 23 分の 1 になり golden io も 1.0MB → 45KB に縮むので採用したが、
> **ホスト固定費の本命はここではない**。WAV sha256 は不変（融合パスは DeBERTa に効くルールを
> 持たないので、出力 pin の変化が数値に触らないことも確認できた）。

### 既知の欠落

ADR 0026 決定 2 が名指しする DeBERTa の数値 golden 門 `packages/runtime/tests/e2e_deberta_test.ts`
は**リポに存在しない**（git 履歴にも無い — 移行元リポに残置のまま。ACTIVE_DESIGN の Pitfalls に
既出）。つまり text_encoder の数値回帰を捉える TS 側の網は現在無く、実質 WAV sha256 門だけが
検出器である。回帰は「WAV の 1 bit 差」としてしか現れず原因の局在ができないため、波 1 では
**torch 段のビット一致（§4）を先に取ってから**資産を差し替える手順を取る。

## 付録: 再現コマンド

```sh
# ONNX の実測（onnx パッケージは uv で一時取得）
curl -sL -o /tmp/model_full.onnx \
  https://huggingface.co/tsukumijima/deberta-v2-large-japanese-char-wwm-onnx/resolve/main/model.onnx
uv run --with onnx --python 3.12 python -c "
import onnx, collections, numpy as np
g = onnx.load('/tmp/model_full.onnx', load_external_data=False).graph
c = collections.Counter((tuple(onnx.numpy_helper.to_array(i).shape), str(onnx.numpy_helper.to_array(i).dtype)) for i in g.initializer)
print(len(g.node), len(g.initializer), [o.name for o in g.output])
print(sorted(c.items(), key=lambda kv: -np.prod(kv[0][0] or [1]) * kv[1])[:6])"

# karume 資産の層別集計（safetensors ヘッダは 先頭 8B = ヘッダ長 u64 LE + JSON）
# scratchpad の st-header.ts と同型のスクリプトで再現できる
```
