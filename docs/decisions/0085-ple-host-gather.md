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

## 追記（2026-09-07 — 決定 3 の LRU は「1 回の gather の中では hit 先行」）

1 回の gather が触る shard は束ねて 1 本ずつ引くが、**常駐している shard を先に処理する**。初出順に
引くと、未常駐 shard の読みが起こす LRU の追い出しが「この gather がまだ触っていない常駐 shard」を
落とし、同じ gather の中で読み直しになる（実測 2026-09-06: 予算 = 最大 shard 2 本・shard 1 / 2 が常駐
の状態から `[0,1,2]` 順に引くと 3 load、`[1,2,0]` 順なら 1 load — 順序依存）。hit を先に触れば LRU の
末尾へ回るので、1 回の gather の load 数は miss 数（= 理論最小）になる。走行中に掴んだ shard の実体は
ローカル変数が持つので、途中で追い出されても値は揃う（決定 3 の「値も token 列も変わらない」は不変）。

併せて、同じ id が並ぶ列（prefill の pad 行 id 0 が典型）は最初の位置だけ逆量子化し、残りの位置へは
f32 バイト列を複写する。再計算ではなく複写なので決定 4 の 2 段丸めとビット同一である。gather を
またぐ行キャッシュ（token 行の LRU）は**入れない** — 実利用の測定（同日・自然文 2 ターン × 200 token・既定予算）で
shard の読み直しが 137 回・約 42 s / 壁 58 s と p50 の問題であり、オフラインの方針比較で行キャッシュは 71 回にしか
減らなかった（初出 token の miss は減らない）。対処は**行読み**（下の追記）。

## 追記（2026-09-07 — 代替案 b「行だけ読む」を採る: 読み口を区間読みの handle へ）

決定 3 が「実需が出たときにホスト側だけ差し替える」と予定していた代替案 b を採る。読み口を
`Gemma4Assets.openPleShard(file) → { bytes, readAll, range?: { cost: "seek" | "scan", read(offset, length) } }` に
置き換え（`readPleShard` からの**破壊的変更** — limitations）、`Gemma4Ple` は shard ごとに header を 1 度だけ小読みして行の
位置を持ち、小さい gather（decode の 1 id など）は values 8,960 B + scales 140 B の 2 区間だけを読む。方針は shard ごとに
束ねた一意行数 rows で決める: 常駐（pending 含む）→ hit / range 無し → 全量 + LRU（従来）/ cost "scan" → rows ≤ 2 なら行読み・
それ以外は全量 + LRU / cost "seek" → rows ≥ 32 かつ予算に**追い出し無しで**載るなら全量 + LRU・それ以外は行読み。range が
あるなら LRU の追い出しは起こさない（9 shard に散る自然文で回り続けるのが実測の 137 回の正体）。行の値は同じ bytes から
同じ 2 段丸めで組むので決定 4 のビット一致は不変（torch 突合門で確認）。読み口の実装: `denoDirectory` = `Deno.open` の位置読み
（seek・実測 46 µs/read）/ ブラウザの HF 取得元 = CacheStorage の `response.blob().slice()`（Chrome 152 実測 0.1〜0.3 ms・
Range 要求は 200 全量で無視される）/ Deno の HF 取得元 = 本文ストリームの読み飛ばし（`blob()` が全量を読むため・17〜76 ms・scan）。
取得層 `@hdae/fetch-cache` の `openCachedUrl` / `openHfFile`（その ADR 0012）と hub の能力 ⑧（ADR 0086 追記）がこれを支える。

## 追記（2026-09-11 — 小さい gather の値と scale を並行して読む）

行読みする一意行数が8以下の場合は、各行の値とscaleの2区間を並行して読む。
それより多い場合は従来どおり16本のworkerが各行の2区間を順に読む。
これでdecodeの1行にあった直列待ちを減らしつつ、1回のgather全体で同時読み込み数16の上限を守る。
上限はshardをまたいで掛け、ヘッダ取得後に行の読み込みを発行する。

並行に発行した2区間は、片方が拒否されても両方の決着を待つ。両方が成功してから行を書き、
`rowReads`を増やす。両方が拒否された場合は従来の順序と同じ値側の原因を返す。
中断signalの透過、常駐予算、区間位置、逆量子化、重複行の複写は変えない。
公開API・保存形式・数値契約は不変で、通常GemmaとQATの共通loaderへ適用する。

[実測](../research/2026-09-10-codex-mtp-optimization.md#ple-の行内並行読み2026-09-11)では、
HTTP Rangeを使うChromeのQAT生成が約2〜5%速くなった。Denoローカル読みはほぼ同等だった。
この差をすべての取得元やApple M2の効果として扱わない。

## 追記（2026-09-12 — 行読み後の量子化行を小さな LRU へ保持する）

9/7 に見送った行キャッシュを、行読み導入後の別の費用に対して採る。
当時は shard の全量読み直しが律速であり、行キャッシュだけでは初出 token の大きな読みを解消できなかった。
今回の HTTP Range 計測では小さな読み自体が prefill の 66〜74 ms を占め、同じ token でも毎回値と scale を読み直していた。
[費用の分解と比較](../research/2026-09-12-ple-row-cache.md)を根拠に、通常版と QAT の共通 PLE loader へ追加する。

- 行読みが成功した量子化値と f32 scale を、そのまま最大 256 行の LRU へ保持する。
  逆量子化した f32 行を別途保持せず、hit も既存の `writeRow` を使う。丸め順序・固定量子化・token 列の契約は不変。
  同じ gather 内では行 hit を先に使い、後続 miss による追い出しで未処理 hit を失わない。
- 既存 `maxResidentPleBytes` / `maxResidentBytes` の予算を共有する。全量 shard が優先で、空きに入る行数だけ保持する。
  shard の予約時と行の保存時に、`shard の予約 byte + 保持行の byte ≤ 既存予算` に収める。予算 0 は行も保持しない。
  E2B の上限 payload は通常版 2,329,600 B、QAT 1,182,720 B。新しいメモリ予算や CLI ノブは追加しない。
- 内部診断 `residentBytes` は両者の payload 合計へ拡張する。`resident` は従来どおり全量 shard 数、
  `rowReads` は実際に行を読み終えた数で、hit を読取りとして加算しない。一時読取りバッファや管理オブジェクトは従来同様に別。
- 値と scale の両方が成功してから保持する。拒否された行、読取り中に中断・dispose された行は保持しない。
  cache は pipeline の PLE 所有者と同じ寿命で、dispose で空にする。全量読みの判断基準と同時読取り上限 16 は維持する。
- 新しいプロンプトにも共通 token の hit は効くが、初出 token の読取りは残る。
  同じ入力を反復した速度は上限寄りの結果として分け、未使用の英語・日本語入力でも比較する。
  M2 の改善後の速度は利用者実機での追試が必要。

公開の関数引数・グラフ・保存資産・数値契約は変わらない。既定で既存予算の未使用部分を少量使うことと、
内部診断が行の常駐 byte を含むことが変更点である。CPU で予算・LRU・重複・並行読取り・中断・破棄と
I2/I4 の固定参照との bit 一致を検査し、実 GPU では通常/QAT・E2B/E4B の生成列と全体検証を確認する。

## 追記（2026-09-19 — GPU 常駐席: opt-in で sidecar を GPU に置き、gather も GPU 内で行う）

perf-ledger H-28。決定 3（遅延ロード + LRU）と決定 6（PLE は通常のグラフ入力）に対する
**opt-in の例外**を 1 つ置く。既定は従来どおり `"host"` で、宣言した呼び手だけが GPU 常駐になる
（利用者裁定 2026-09-19 = 案 A「単一束縛 + 束縛上限の引き上げ + opt-in 席」）。

- **席**: `Gemma4PipelineOptions.pleResidency: "host" | "gpu"`（既定 `"host"`）。通常 Gemma と
  QAT の共通 pipeline が受ける。不正値・併用できないノブは資産を読む前に fail loudly。
- **何を消すか**: run ごとのホスト逆量子化と `per_layer_inputs` の writeBuffer（decode
  35,840 B/token = 約 0.25 ms、prefill 768 行で 27.5 MB）。先行投入（H-27 段 ②）の硬い前提でもある。
- **機構**: `embedding` → `mul embed_scale` の 2 ノードだけの小さな IR を別 Session で持ち、
  target の run と**同じ batch へ先に enqueue** して出力を常駐テンソルへ書き、target はそれを
  `per_layer_inputs` の常駐入力として受ける（writeBuffer を出さない）。temperature 0 の decode は
  greedy 出力の batch に相乗りするのでフェンスは増えない。prefill と診断付き decode（通常 run）は
  gather 用の batch を 1 本余分に払う。prefill も decode も GPU gather を通る（片方だけの席にしない）。
- **重みの形**: 重みは `[tokens × layers, dim]`・添字は `id × layers + layer`。1 行 = 1 層ぶんに
  なるので、sidecar の `scales[rows, layers]` がそのまま行 scale の並びになり、**i8 / i2 の行
  scale も i4 の `group_size = dim` の group scale も同じ平坦添字**で引ける。逆量子化は
  `embedding` の `f32(q) × scale` 1 回、`embed_scale` はその後の別ノード — 決定 4 の 2 段丸めと
  同じ順序・同じ丸め点である（`ple.probe.safetensors` との u32 一致で門を張る）。
- **ロード**: 決定 2（ホストで 1 本の巨大 ArrayBuffer に連結しない）を保つため、sidecar の shard を
  1 本ずつ読んで**ランタイムの shard 逐次面**（ADR 0070 決定 3 / 0090 の piece）へ流す。ランタイムが
  piece を親 1 本ぶんの GPU バッファへ行オフセット位置に `queue.writeBuffer` する。合成 shard の器は
  1 本を使い回すので、ホスト RAM のピークは「sidecar の最大 shard 1 本 + 器 1 本」に収まる。
  companion scale は piece 1 と同じ shard に置く契約（ADR 0090 決定 1）なので、**scale だけは先に
  全量（E2B で 35 MiB）を集める** — 区間読みできる読み口なら shard あたり数 MB の小読みで済み、
  持たない読み口では sidecar を 1 度余分に読む。piece 1 は先頭 1 行に切って graph shard へ同居させる。
- **常駐量と束縛上限**: `values` は分割しない 1 本のバッファで、QAT E2B（i4）1,174,405,120 B・
  QAT E4B（i2）704,643,072 B・通常 Gemma 4 E2B（i8）2,348,810,240 B。`scales` は f32 で
  E2B 36,700,160 B。`acquireGpu` はアダプタ実測値をそのまま `requiredLimits` に要求する
  （`planRequiredLimits`）ので、device 取得側に足すものは無く、**ロード時に
  `maxStorageBufferBindingSize` / `maxBufferSize` と突き合わせて足りなければ fail loudly** する。
  黙ってホスト経路へ退避しない。参照機（RTX 3080 Ti / Vulkan）の上限は 2,147,483,644 B で、
  通常 Gemma 4 E2B の i8 sidecar はこの席を使えない（QAT の i4 / i2 は載る）。
- **見積り**: 常駐 PLE と gather Session のぶんは `estimateSessionMemory()` の `auxiliaryBytes`
  （小出力 decode の補助 Session と同じ欄）に載り、`peakAccountedBytes` にも加わる。
- **未対応の組み合わせ**（どれも fail loudly）: `maxResidentPleBytes`（GPU 常駐では引かれない）/
  投機デコード（drafter の per-layer 入力は範囲外）/ `gpuTiming`（計測中は batch を開けない）。
- **検収**: `packages/models/tests/e2e_gemma4_ple_gpu_test.ts` が ①golden（torch の 35 表経路）と
  u32 一致 ②ホスト `ple.gather` と u32 一致 ③QAT E2B の 16 token greedy 生成 id 列が `"host"` と
  一致、を見る。CPU 側は `gemma_ple_gpu_test.ts`（席の受理・gather IR の形・合成コンテナの突合）。
  **速度の採否はまだ付いていない** — Deno CLI の decode は deno_webgpu の 10 ms/token 床を含むので
  判定に使えず、M2 / Chrome の計測は未検収である。
