# 0085: PLE の配布形 — token-major + vocab レンジ shard・ホスト gather

- Status: accepted（2026-08-31 — 設計ドラフトの裁定 3 を★推奨案で確定〈ユーザー裁定〉。
  実装は未着手 = backlog now の段 1b）
- Date: 2026-08-31
- 対象: `tools/export-recipes/gemma4/`（PLE の外出しと sidecar 生成）/
  `packages/models/src/gemma/`（ホスト側 loader）。IR 仕様・ランタイム・hub は**無改変**。
- 関連: ADR [0083](0083-generation-api-surface.md)（生成 API 面 — 同じ再 export に乗る = 案 α）/
  [0084](0084-gemma-tokenizer-chat.md)（決定 5 = 同一 digest set の束ね方）/
  [0070](0070-shard-loading-admission.md)（shard ロードと admission）/
  [0081](0081-shard-spec-v2.md)（shard 仕様 v2 — 1GiB 上限）/
  [0066](0066-generation-context-state-slots.md)（Session / context の寿命）/
  [0038](0038-manifest-v1.md)（キャッシュ設計 = キーは URL・部分読み席なし）
- 根拠:
  [research/2026-08-31-generation-api-design-draft.md](../research/2026-08-31-generation-api-design-draft.md)
  §2（候補比較・実測値の正本）/
  [research/2026-08-31-freetoken-moe-over-arraybuffer.md](../research/2026-08-31-freetoken-moe-over-arraybuffer.md)
  §3・§4（スケール軸での位置づけ = **c-1 ホスト gather の初適用**）/
  [limitations](../limitations.md)（Chromium の単一 ArrayBuffer 上限）

## Context

Gemma 4 E2B の PLE（per-layer embeddings）は `input_ids` **だけ**を引数に取る純粋な行 lookup で、
recipe には切断点が既にある — `tools/export-recipes/gemma4/ple.py` の
`per_layer_inputs(tables, input_ids, scale)` が `[1,M,35,256]` を組み、decode / token の両台本が
同じ 1 本を通しているだけである。

- 常駐は **i8 35 表 × 64 MiB = 2,240 MiB**（容器ヘッダの実測）で、容器全体 3.70 GiB の **59%**。
  外に出せば **3,787 MiB → 1,547 MiB**。
- ランタイムは**全 initializer に Session 構築時の GPU 常駐席を与える契約**
  （`packages/runtime/src/runtime/executor.ts` の「席はプランナが正本 — 全 initializer を載せる
  契約」）なので、**グラフに残す限り lazy にはならない**。
- **速度には効かない**。初回実測（K-11 前・decode GPU 86.2ms のうち `linear` 系が 78.7ms）では
  embedding 35 本は「残り ≈935 本 ≈4.5ms」の内数だった。逆流するコストは
  `per_layer_inputs[1,M,35,256]` f32 のアップロードで、decode 35,840 B/token・prefill chunk 32 で
  1,146,880 B/chunk — 実測の壁（decode 32.5ms/token・prefill ≈162ms/chunk）に対して無視できる。

つまりこれは**純粋に常駐の話**であり、採否は常駐削減の効き幅だけで測れる。

## Decision

### 1. 配布形は token-major + vocab レンジ shard に固定する

`[token][layer][256] i8` + `[token][layer] scale` の **token-major** に固定し、**vocab の範囲で
shard** する。

根拠:

1. **配布形はホスト側の方式（全量常駐 / 部分読み）と独立で、先に固定して損が無い**。後から
   「キャッシュから行だけ読む」（検討した代替案 b）へ移るときに、**再 export も HF 再アップロードも
   要らない**（ホスト側の差し替えだけで済む）。
2. **token-major なら 1 token の PLE は連続 1 読み**（8,960 B + 35 scale）。table-major のまま
   だと 1 token を引くために **35 箇所の離散読み**になり、部分読みへ移った瞬間に I/O が 35 倍に
   なる（「1 token の PLE を引くために 35 個の離れた asset location を読む」形）。

### 2. 分割は「重い / 軽い」ではなく必須要件である（単一 ArrayBuffer 天井）

全量ホスト常駐の素直な形（1 本の表）は **Chromium で原理的に不可**である:

- PLE i8 全量 = 262,144 × 8,960 = **2,348,810,240 B**（= 35 × 64 MiB = 2,240 MiB）
- Chromium の単一 ArrayBuffer 上限 = **2,145,386,496 B**（limitations の恒久記載 — anima Base f16
  がロード不能だったのと同じ天井）

NOTE: ドラフト §2.3 は同じ式に `2,351,662,080 B` と書いているが、これは積の誤りである
（正 = `2,348,810,240 B`。ドラフト自身の「i8 35 表 × 64 MiB = 2,240 MiB」と一致するのは後者）。
**天井超過という結論は変わらない**（超過幅が 206,275,584 B → 203,423,744 B に変わるだけ）。

したがって「ホスト常駐案は単純だが重い」という評定は成立しない — **分割はどの案を採っても必須
要件**である。35 表に割れば 1 表 = 67,108,864 B で天井は回避できるが、それは table-major のままで
決定 1 が避けた形になる。

### 3. 初版のホスト側は「触った shard の遅延ロード + LRU」（hub に部分読み席を新設しない）

vocab レンジ shard を**ファイル単位で遅延ロードし、LRU で落とす**。hub は今日の `streamAssets` /
`prefetchAssets`（最小単位 = ファイル 1 本 = `StreamedAsset {id, bytes}`）のままで足りる。

hub に部分読み（Range）の席を新設するのは**独立の設計判断**で、この波に抱き込むと射程が膨らむ —
ADR 0038 のキャッシュ設計（キーは URL）へ踏み込むうえ、Range 並列は perf L-3 で parked のままで
ある。決定 1 のとおり配布形が同じなので、実需が出たときにホスト側だけ差し替えれば移れる。

**未実測（speculation とラベルする）**: 「実会話が触る token id が vocab のどの範囲に集中するか」は
測っていない。SentencePiece 語彙が頻度順に並ぶという一般論はあるが、この checkpoint では確認して
いない。**shard 幅は golden 3 ケース + chat コーパスで実測してから決める**。

### 4. ホスト gather の逆量子化は GPU 側 `embedding` とビット一致する MUST

さもないと token 列 parity が割れ、「機能不変であること」の証明（ADR 0066 追記 9 で sliding 容量を
変えたときに使った手）が使えなくなる。`per_layer_scale` = `256 ** 0.5` = **16.0** で 2 冪なので
f32 の乗算は厳密であり（`ple.py` の docstring が同じ理由でビット一致検査を成立させている）、
**順序さえ揃えれば成立する見込み**（実測は段 1b の合格線）。

### 5. loader で id 空間を相互照合する

tokenizer が生成し得る id / 主 embedding の vocab 行数 / PLE sidecar の行数 / special id を、
loader が突き合わせる。**ここがずれると OOB ではなく「別 token の有効な行」を引く**（例外なしで
沈黙して壊れる）ので、fail loudly の門を置く場所はここである。

### 6. 「pageable initializer」は足さない — PLE は通常のグラフ入力になるだけ

汎用ランタイムに**第五の weight lifetime**（未着荷 initializer / 動的常駐）を足す形は採らない。
PLE はホストが供給する通常のグラフ入力（`per_layer_inputs`）になるだけで、ランタイムの契約は
1 文字も変わらない。

前例 = 「flow / voice の相対位置表はグラフ入力 — 生成はホスト側の責務」（limitations）・
ADR 0079（テキスト解析は呼び手の責務）。「未着荷 initializer」席の新設は **2026-08-31 の MoE 裁定で
見送り済み**（backlog parked — IR への値依存実行選択に従属）なので、本 ADR はその裁定と整合する。

## 検討した代替案

- **b) hub のキャッシュから行だけ読む**（token ごとに 1 行）: RAM も VRAM も食わないのが利点だが、
  hub に部分読みの席が今日無く（公開面 = manifest / resolve / fetch / stream / prefetch / clear・
  最小単位はファイル 1 本）、ADR 0038 のキャッシュ設計に踏み込む。Range 並列は perf L-3 で parked。
  **配布形は決定 1 と同じ**なので、実需が出たときにホスト側だけ差し替えれば移れる — 今この波で
  買う必要が無い。
- **c) 現状維持 + PLE を i4 化**（2,240 → 1,120 MiB）: recipe だけで閉じるのが利点だが、recipe
  README が「embeddings are int8 … not int4-eligible」と明記しており、品質リスクを token 列 parity で
  潰す作業が要る（潰せること自体は利点）。常駐削減も決定 3 の形より小さい。却下（i4 化そのものは
  将来の独立候補として残る）。
- **a) 全量ホスト常駐（分割つき）**: 決定 2 のとおり分割は必須要件なので、分割してしまえば
  「触ったぶんだけ読む」（決定 3）に対する優位が残らない。ブラウザで JS heap 2.19 GiB を常時
  抱える形も現実的でない。却下。

## Consequences

- **GPU 常駐 3.70 → 1.51 GiB**（−59%）。**速度は変わらない**（Context のとおり効くのは常駐だけ）。
- **段 1b は案 α**（ドラフト §7.1 の裁定 5）: PLE 外出しと最終行 logits 出口（ADR 0083 決定 6）を
  **同じ再 export に載せ、製品グラフを 1 系列にする**。3.7GiB 系列の再 export が 1 回で済む代わりに、
  ホスト PLE loader が e2e の前提になる。既存 2 系列（logits opt-in / token-only）は検収 fixture
  として残す。
- **合格線（段 1b）**: 既存 `greedy.<case>` golden との交差 parity（`argmax(logits)` == 既存 token 列・
  3 ケース × K=16）+ PLE 逆量子化のビット一致（決定 4）。
- PLE sidecar は配布 digest set の一員になる（ADR 0084 決定 5）— 製品グラフ / weight shards /
  compiled tokenizer / chat format version と同じ束で配る。
- 本方式は freetoken 調査の **c-1（ホスト gather）の初適用**であり、スケール軸の公式スタンス
  （MoE は全 expert VRAM 常駐・総パラメータで予算 — limitations）と整合する。embedding / lm_head
  級の**行疎な表**には効くが、expert FFN には効かない（同調査 §4）ことは既に記録済み。
- 追記（2026-09-05）: `fromPretrained` は PLE 索引の `shards[].file` 集合と manifest の遅延資産
  キー集合の**対称差**を `#build` の前に見る（索引が名指す shard が manifest に無い / manifest の
  遅延資産を索引が名指さない、のどちらも構築前に落とす）。この式は `EAGER_ASSETS` が tokenizer と
  ple_index の 2 本ちょうどであることに依存する。
- ホスト側の PLE アップロードが毎 step 増えるので、将来 decode がさらに速くなった場合は
  この転送（decode 35,840 B/token）が観測対象に入りうる — 現時点の壁に対しては無視できる。

## 追記（2026-09-02 — 決定 3 の常駐上限は「本数」ではなく「バイト」）

常駐の上限は **shard の本数ではなくホスト RAM のバイト数**（`Gemma4PleOptions.maxResidentBytes` /
`Gemma4PipelineOptions.maxResidentPleBytes`）で受ける。本数のまま持つと同じ数字が資産世代ごとに
違う RAM を意味するからで、実例が既に出ている: shard 上限 1GiB 世代の E2B は 3 本（1 本 758MiB）
だったが、ADR 0090 で書き手の上限を 256MiB にした世代は 9 本（1 本 253MiB）になり、「常駐 3 本」の
意味が 2.2GiB → 759MiB へ黙って変わった。1 本ぶんのバイト数は索引（`ple.json` の `shards` と
`layers` / `dim`）だけから決まる（`(stop-start) × layers × (dim + 4)` — i8 `values` + f32
`scales`）ので、読む前に予算の検査も LRU の追い出しも判定できる。既定は**最大 shard 2 本ぶん**で、
従来の既定「2 本」の意味（どの 2 本を掴んでも収まる）を shard 幅に依らず保つ。`0` は「常駐させ
ない」という正当な指定（読み終えた shard を即座に落とす）で、それ以外で **shard 1 本すら載らない
予算は構築時に fail loudly** にする — 黙って超過すれば予算が意味を失い、黙って守れば gather が
引けないため、どちらも呼び手の指定を裏切る。
