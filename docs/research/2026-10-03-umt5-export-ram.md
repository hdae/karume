> この文書は 2026-10-03 時点の計測記録で、正本ではない。

# umT5 の export のホスト RAM — 段 10b の実測（ADR 0119 決定 6）

ADR 0119 段 10b（export のホスト RAM の実測と export の形の確定）の記録。GPU は使っていない。
HEAD `9e1ae8ac`（作業ツリーに `tools/export-recipes/wan/umt5_export.py` と `wan/tests/test_umt5_export.py` を足した状態）。

- 機: AMD Ryzen 5 5600（6 コア 12 スレッド・torch のスレッド数 6）・`MemTotal` 32,775,140 kB（31.26 GiB）・
  swap は zram 8 GiB（圧縮して RAM に置く swap — RAM の外の逃げ場ではない）。計測の前の `MemAvailable` は約 28 GiB
  （他のプロセスが約 3 GiB）。
- 版: torch 2.13.0+cpu・transformers 5.14.1・safetensors 0.8.0（recipes の venv）。
- 上流: pin した `Wan-AI/Wan2.1-T2V-1.3B-Diffusers@0fad780a…` の `text_encoder`。**格納は F32**
  （5 分割・`total_size` 22,723,641,344 B・5,680,910,336 パラメータ）。bf16 は読み込み時の変換で、格納ではない。
- 計測の道具は `wan/umt5_export.py` の `MemoryMonitor`: 別スレッドで `/proc/self/status`（`VmRSS`・`RssAnon`・`RssFile`）と
  `/proc/meminfo`（`MemAvailable`・`SwapFree`）を 1 秒ごとに読み、段の始めに `/proc/self/clear_refs` へ `5` を書いて
  `VmHWM`（RSS の山）を戻す。表の「山」は段の終わりの `VmHWM`（標本の間の山も取りこぼさない）。Linux の `ru_maxrss` も同じ
  hiwater を読むので、段の始めの戻しで一緒に戻る（実測 — 段ごとの値が下がった）。外から 1 秒ごとの `/proc/meminfo` の標本も並べた。

## 要約

- **素直な形（全重みを f32 で持ち、fake-quant してから export → emit）は 31 GiB 機に収まらない見込み**。f32 の量子化対象を
  丸めた時点で匿名メモリ（`RssAnon` — ファイルに戻せないメモリ）が約 21.2 GiB になり、emit が語彙埋め込み
  （`[256384, 4096]` f32 = 3.91 GiB）を i8 にする一時が 2.25 倍（約 8.8 GiB）を上乗せする — 山は約 30.5 GiB（推計 — §2）。
  指示どおり、この形は回していない。
- **決定 6 の形は exporter core の変更なしで組めた**: 重みを持たない（meta）上流で trace し、量子化の対象は checkpoint から
  行の塊ごとに i8 にして `fixed_weights`（ADR 0097 の入口）で渡す。**実モデルで完走**: 山 6.79 GiB・69.3 s・
  容器 5,686,978,107 B（26 part）・`karume verify` 緑（§3）。
- 容器のバイトは素直な形と一致する（小模型で全 part のバイト一致 — pytest。故障注入 2 種で赤）。
- **決定 4 の実モデルでの裏付け**: f32 では「有効長だけ」と「512 + マスク」が固定 4 本で**ビット一致**。bf16 では 4 本中 3 本が
  割れる（最大絶対差 ÷ 参照の最大絶対値 = 比 2.4e-2〜1.9e-1）が、bf16 の経路そのものの誤差（同じ形の f32 に対する比
  3.0e-2〜9.4e-1）の内（§4）。「512 + マスク」の bf16 は既存の資産とビット一致（資産の経路を再現できている対照）。

## 1. 量子化の対象の本数

i8 にした重みは 169 本: linear 168 本（24 層 × 7 本 — attention の q / k / v / o と FFN の wi_0 / wi_1 / wo）と語彙埋め込み 1 本。
F32 のままの重みは 73 本: 相対位置の表 24 本・RMSNorm の重み 49 本（24 層 × 2 + 最後の 1）。ほかに exporter が足す
重み非依存の定数 2 本（`[4096]` と `[10240]` のゼロ — linear のバイアスの穴埋め）。

## 2. 素直な形の見積もり

測った部品:

| 項目                                                  | 実測                                                                    |
| ----------------------------------------------------- | ----------------------------------------------------------------------- |
| `from_pretrained(dtype=float32)`                      | 2.2 s・山 0.72 GiB（重みは checkpoint の遅延 mmap — 触るまで載らない）  |
| 同じモデルで forward を回して全重みに触れた後         | 山 18.3 GiB（`RssFile` 17.4 GiB・`RssAnon` 0.96 GiB — ファイル側）      |
| mmap の重みへの in-place 書き込み（小模型で）         | 書いた分だけ `RssAnon` が増え、checkpoint のファイルは変わらない（COW） |
| `fake_quant_int8` の一時（`[65536, 4096]` f32 1 GiB） | 対象の 2.00 倍                                                          |
| emit の i8 変換 + 逆変換の検査の一時（同じテンソル）  | 対象の 2.25 倍                                                          |

推計（測った部品の和 — 実走していない）:

- fake-quant は量子化の対象 169 本を丸めて書き戻す。書いたページは COW で匿名メモリになるので、丸め終えた時点で
  `RssAnon` ≈ 21.2 GiB（量子化の対象の f32 の和）。
- emit は格納の直前に 1 本ずつ i8 へ変換する（`karume.emit._StoredTensors`）。語彙埋め込み 3.91 GiB の番で一時が
  2.25 × 3.91 ≈ 8.8 GiB 乗り、`RssAnon` の山 ≈ 21.2 + 8.8 + Python・torch 約 0.5 ≈ **30.5 GiB**。publish の読み直し検証が
  同じ変換をもう 1 度回すので、この山は 2 度来る。
- 他のプロセスの約 3 GiB と合わせて `MemTotal` 31.26 GiB を越える。zram swap（8 GiB）は RAM の中に圧縮して置くので、
  越えた分の逃げ場としては当てにできない（推測 — 圧縮率しだい）。
- export そのもの（`torch.export` → 正規化 → 変換）は重みを複製しない（`convert` の initializer は state の参照を持つだけ —
  コードを読んだ事実。実重みでの実測はしていない）。

## 3. 決定 6 の形（実装と実測）

`wan/umt5_export.py` の `prepare`（3 段）+ recipe の外の driver の publish:

1. **trace**: `UMT5Config.from_pretrained` → `torch.device("meta")` の上で `UMT5EncoderModel` を組み、`Umt5EncoderTokens` で包んで
   `export_module`（例示入力は L = 28 — 値は trace に効かない）。小模型では、実重みの export とグラフの JSON が同一（pytest）。
2. **plain**: 量子化しない 73 本を checkpoint から f32 のまま読む（tied な語彙埋め込みは checkpoint に `shared.weight` としてしか
   無いので、同じ Parameter を指す別名から checkpoint に在る 1 つを選ぶ — 0 本・2 本以上は fail loudly）。
3. **quantize**: 量子化の対象 169 本を、safetensors の `get_slice` で 16,384 行ずつ読み、exporter の手順（`channel_scale` →
   `quantize_to_int8` → `·scale` → もう 1 度 `quantize_to_int8` → 逆変換の一致検査）で i8 の packed と行ごとの scale にする。
   scale は行の amax で閉じるので、塊に割っても全体で回した値とビット一致する（pytest — 塊の大きさ 1 / 7 / 全体）。
4. **publish**: `publish_model(..., fixed_weights=..., graph_name="text_encoder")`。量子化の対象のテンソルは meta のまま渡し、
   実体は `fixed_weights`。`fixed_weights` の回の既定の格納は f32 なので、相対位置の表の F32 の明示は要らない。
5. **verify**: `karume verify`（2 文書の構造・合流・全 block の sha256・IR の受理規則）。

| 段       | 壁時間 | 山（`VmHWM`） | `RssAnon` の標本の最大 | `MemAvailable` の最小 |
| -------- | ------ | ------------- | ---------------------- | --------------------- |
| trace    | 7.4 s  | 0.52 GiB      | 0.43 GiB               | 28.02 GiB             |
| plain    | 0.0 s  | 0.53 GiB      | 0.43 GiB               | 28.02 GiB             |
| quantize | 41.7 s | **6.79 GiB**  | 6.17 GiB               | 22.19 GiB             |
| publish  | 17.0 s | 5.87 GiB      | 5.75 GiB               | 22.52 GiB             |
| verify   | 3.0 s  | 5.85 GiB      | 5.75 GiB               | 22.71 GiB             |

合計 69.3 s。swap は使っていない（`SwapFree` は 6.6 GiB のまま）。残る山は i8 の全重み（5.30 GiB — `FixedQuantizedWeight.packed` を
全部持つ）+ 1 塊の一時。packed まで遅延にすれば 1 本ぶんまで下がるが、それは core の変更（`fixed_weights` の値を遅延の口にする）で、
今の山では要らない。

容器（`outputs/series/wan2.1-umt5-i8-dyn/text_encoder/model.krm` — git 追跡外・部品名は下見と同じ仮置き）:

| 項目      | 値                                                                                                                           |
| --------- | ---------------------------------------------------------------------------------------------------------------------------- |
| サイズ    | 5,686,978,107 B（26 part・最大の part 268,435,456 B）。i8 の重み 5,680,660,480 B + scale 4,957,696 B + f32 1,056,768 B       |
| 束縛      | i8: linear 168・語彙埋め込み 1。f32: 相対位置の表 24・RMSNorm 49・定数 2                                                     |
| 入力      | `input_ids` `[1, L]` i32・`relative_position_buckets` `[L, L]` i32。記号は `L` 1 つ                                          |
| 出力      | `[1, L, 4096]` f32                                                                                                           |
| op（914） | add 72・bmm 48・embedding 25・expand 96・gelu_tanh 24・linear 168・mul 24・permute 120・reshape 264・rms_norm 49・softmax 24 |
| 検査      | `publish_model` の `assert_runtime_support` / `assert_op_contracts`（語彙外の op なし）・読み直し検証・`karume verify` が緑  |

- RMSNorm は 49 本とも `rms_norm` に畳まれ、活性は `gelu_tanh` 24 本、`pow` は残らない（下見の結論どおり）。
- 素直な形との同値: 小模型（4 層）で、素直な形（`fake_quant_i8` → `weight_dtype="i8"` + `storage_overrides`）とこの形の容器が
  全 part のバイトで一致（`test_the_container_matches_the_whole_tensor_path_byte_for_byte`）。故障注入（相対位置の表も i8 にする・
  scale を 1e-7 ずらす）で赤。
- golden（CPU の参照）は採っていない（層逐次の参照は段 10c）。

### exporter core のどこが全重みを要るか（読んだ事実）

| 段                                         | 全重みを要るか                                                                                        |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| `export_module`（`torch.export` → 正規化） | 要らない。meta の重みで通り、グラフは同一（実測）                                                     |
| `convert`（initializer の登録）            | 要らない。state の参照を持つだけ。畳み込み定数は重み非依存の 2 本だけ                                 |
| `stored_model`（自動量子化の回）           | 計画は形だけ。変換（`_convert_for_storage`）は 1 本ずつ — ただし対象の f32 の実体が全部生きている前提 |
| `stored_model`（`fixed_weights` の回）     | 量子化の対象は meta で良い（ADR 0097）。packed は Mapping の値として全部持つ                          |
| `publish_container`                        | 実体を 2 度引く（書き出し + 読み直し検証）。書き手は 1 本ずつ                                         |

## 4. 決定 4 の実モデルでの裏付け

`check-mask` で、上流の `UMT5EncoderModel` を「有効長だけ」（`valid_output`）と「512 まで pad id で詰めてマスク」
（`padded_output` — 上流 `_get_t5_prompt_embeds` の写し方）で回し、固定 4 本で比べた。bf16 と f32 の 2 回。

| プロンプト     | L   | bf16: 有効長 vs 512 + マスク   | bf16 512 + マスク vs 資産 | f32: 有効長 vs 512 + マスク | bf16 の誤差（対 f32 の比 — 有効長 / 512 + マスク） | 同じく相対フロベニウス |
| -------------- | --- | ------------------------------ | ------------------------- | --------------------------- | -------------------------------------------------- | ---------------------- |
| boxing-cats    | 28  | ビット一致                     | ビット一致                | ビット一致                  | 3.0e-2 / 3.0e-2                                    | 3.1e-2 / 3.1e-2        |
| ferret         | 118 | 比 5.3e-2（要素の 90% が違う） | ビット一致                | ビット一致                  | 5.1e-1 / 4.8e-1                                    | 7.5e-2 / 7.2e-2        |
| cat-dog-baking | 50  | 比 1.9e-1（90%）               | ビット一致                | ビット一致                  | 9.4e-1 / 8.2e-1                                    | 1.5e-1 / 1.3e-1        |
| negative       | 126 | 比 2.4e-2（89%）               | ビット一致                | ビット一致                  | 3.1e-2 / 3.3e-2                                    | 2.5e-2 / 2.6e-2        |

- 比 = 最大絶対差 ÷ 参照の最大絶対値（`umt5_patch.max_ratio`）。bf16 の 2 形の相対フロベニウスは 0 / 2.4e-2 / 3.9e-2 / 2.1e-2。
- **f32（GPU 経路の活性の dtype — 決定 5）では 4 本ともビット一致**。マスクで隠した列の softmax の重みは厳密に 0 で、
  f32 の CPU の経路では詰めた長さで縮約の順序が変わらなかったと読める（推測 — 一致は事実、理由は未確認）。
- bf16 の割れは、bf16 の経路そのものの誤差より小さいか同じ桁（ferret: 2 形の差の比 5.3e-2 に対し、各形の f32 からの比は
  4.8e-1〜5.1e-1）。2 形はどちらも同じ精度の bf16 の近似で、割れは縮約の順序の差と読める（推測 — f64 の参照は無い。
  L = 28 だけが割れない理由も未確認）。
- 「512 + マスク」の bf16 は既存の資産（`outputs/series/wan2.1-t2v-1.3b-text-embeds/text_embeds.safetensors`）と 4 本ともビット一致。
  2 回走らせて同じ結果（決定的）。
- 所要（6 スレッド）: bf16 は有効長 3.5〜18.7 s・512 で 126〜130 s、f32 は有効長 1.6〜3.4 s・512 で 10.8〜13.5 s
  （512 で bf16 が f32 の約 10〜12 倍遅い — この CPU は bf16 の積和の命令を持たない）。

## 5. 観察（記録だけ — 判断はしていない）

- **bf16 の資産と f32 の差が大きい行がある**: 全体の相対フロベニウスは 2.6e-2〜1.3e-1 だが、行ごとのコサイン類似度の最小は
  ferret の行 64 で 0.168、cat-dog-baking の行 17 で 0.768（他の行は 0.99 台）。どちらが真値に近いかは f64 の参照が無いので
  言えない（f32 の方が近いと読むのが自然 — 推測）。決定 8 の品質の記録（GPU i8・bf16 資産・f32 参照の 3 点）で、bf16 資産を
  比較の基準にしたときの床になる。
- **bf16 の読み込み（`text_embeds` と同じ呼び方）は一時に zram swap を食う**: 山 23.9〜24.0 GiB（`RssAnon` 8.6〜9.5 GiB +
  `RssFile` 14.5〜16.2 GiB）で、`MemAvailable` は 17.9 GiB 以上を保ったのに、`SwapFree` が 1 回目は 0.02 GiB、2 回目は 4.29 GiB まで下がり、
  数秒で戻った。原因は未確認（推測: ページキャッシュが埋まった状態での回収が匿名ページを zram へ押し出した）。f32 の読み込みと
  決定 6 の形の export では起きなかった。
- f32 の `from_pretrained` は重みを遅延 mmap で持ち、in-place の書き込みは COW（ファイルは変わらない — 小模型で確認）。

## 6. 段 10c への手掛かり

- 書き手の口: `prepare` の戻りを `publish_model(..., fixed_weights=export.fixed, graph_name=<部品名>)` へ渡す。recipe の台本が
  `graph_name=` を名乗ると `tests/test_graph_names.py` が `ENTRIES` と `WAN_WEIGHTS` との一致を求めるので、部品名を決める段で
  書く関数・`ENTRIES` の行・`WAN_WEIGHTS` のキーを同時に足す。10b の容器は scratchpad の driver が書いた（下の 4 行が本体）:

  ```python
  export = ue.prepare(ue.upstream_dir(), monitor=monitor)
  with staged_publication(ue.SERIES / ue.COMPONENT_DIR) as staged:
      staged.mkdir()
      publish_model(staged / ue.MODEL_FILE, export.graph, dict(export.tensors), provenance=ue.provenance(), graph_name="text_encoder", fixed_weights=export.fixed)
  ```
- 参照の重み: i8 の fake-quant 後の f32 は `packed × scale`（行ごと）で、容器か `quantize_rows` から 1 本ずつ作れる。層逐次の参照は
  `Checkpoint.read_rows` と同じく 1 層ずつ読めば全重みを載せない。
- f32 の上流の forward は mmap のまま 512 で 10.8 s（`RssFile` 17 GiB 前後 — ファイル側なので回収できる）。fake-quant 後の重みで
  回すと書いたページが匿名になり約 21 GiB を持つ（§2 の山の大半）ので、golden を全体 1 本の forward で採る形は 31 GiB 機では
  余裕が無い（推計）。
