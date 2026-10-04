# 0119: Wan の umT5-XXL テキストエンコーダを GPU で回す — 自由なプロンプトの受理（ADR 0118 段 10）

- Status: proposed — 利用者の裁定待ち（決定 1〜11 はオーケストレータの裁定で起草した。「裁定」節の 2 点が利用者の判断待ち。
  決定 9 の `signal` は利用者承認 2026-10-03）
- Date: 2026-10-03
- 関連: research [2026-10-03-umt5-encoder-recon](../research/2026-10-03-umt5-encoder-recon.md)（以下「調査 §n」— 事実と数値の出典）/
  [2026-10-02-video-gen-recon](../research/2026-10-02-video-gen-recon.md)（前回調査）/
  [2026-10-02-b570-vram-budget](../research/2026-10-02-b570-vram-budget.md)（B570 の天井）/
  [2026-10-03-wan-w8a8-recon](../research/2026-10-03-wan-w8a8-recon.md)（exporter core の重なり）/
  ADR [0118](0118-wan21-video-generation.md)（決定 4 の「第 2 段」・決定 7 の配布形・決定 8 の数値照合・検収表・追記）/
  [0026](0026-w8a8-deberta-deployment.md)（DeBERTa i8 の golden は fake-quant 後の重み）/
  [0085](0085-ple-host-gather.md)（PLE のホスト gather）/ [0106](0106-device-keyed-references.md)（sha256 参照値の環境行と参照門）/
  [0034](0034-dit-dynamic-tokens.md) + [0013](0013-sbv2-chain-export.md)（表の入力昇格）/
  [0045](0045-deberta-layer-trim.md)（相対位置の添字表の Python / TS バイト一致）/ [0019](0019-i8-weight-execution.md)（i8 重みの実行）/
  [0009](0009-dtype-i32-bool.md)（i64 境界）/ [0092](0092-distribution-repos-and-sources.md)（配布リポの粒度と命名）/
  [0109](0109-manifest-v5-container.md)（`karume/5` と越境参照）/ [0107](0107-model-input-error.md)（入力起因の拒否）/
  [0083](0083-generation-api-surface.md) 決定 5（中断の形）/ [0112](0112-anima-transformer-residency.md)（段ごとの Session）/
  [0065](0065-exporter-core-recipe-split.md)（exporter core と recipe の境界）

## Context

### 要望と位置づけ

ADR 0118 は Wan2.1 T2V 1.3B のテキストエンコーダを段階化した（決定 4）。第 1 段は事前計算した埋め込み資産で、
`generate({ prompt })` は資産にある固定の 4 本（正 3 本 + negative 1 本）だけを受ける（docs/limitations.md の Wan2.1 節）。
本 ADR は第 2 段（ADR 0118 段 10）の設計で、umT5-XXL を GPU で回し、資産に無い自由なプロンプトを受ける。

ADR 0118 決定 4 が第 2 段の前提として決めたのは、重みは i8 per-channel・活性は f32・相対位置バイアスは分解経路・
export のホスト RAM の実測が最初の作業、の 4 点。本 ADR はこの前提を引き継ぎ、相対位置のバケット表の置き場だけを
改める（決定 3）。

### 用語（本 ADR の初出）

- **umT5-XXL**: 多言語版 T5 のエンコーダ。Wan のテキストエンコーダで、5,680,910,336 パラメータ（語彙埋め込み 10.5 億 +
  24 層 46.3 億 — 調査 §1.2）。
- **i8 per-channel**: 重みだけを 8 ビット整数で持ち、出力チャネルごと（埋め込みは行ごと）に 1 つの scale を持つ格納
  （ADR 0019）。計算は f32 へ戻して行う。活性まで i8 にする a8（per-token の i8 活性で整数内積）は本 ADR では使わない。
- **i4 g32**: 重みを 4 ビットで持ち、32 要素ごとに 1 つの scale を持つ格納。
- **bf16**: 上位 16 ビットだけの浮動小数（指数は f32 と同じ・仮数 7 ビット）。上流は umT5 を bf16 で回す。
- **fake-quant**: 重みを量子化の格子へ丸めてから f32 のまま持つ処理（glossary）。参照を「同じ丸めの重み」で採るのに使う。
- **Unigram**: SentencePiece 系のトークナイザの分割法。語彙の各ピースのスコアの和が最大になる分割を Viterbi で選ぶ。
  語彙で覆えない文字は未知ノード（unk）になる。`spiece.model` はその原本のファイル。
- **`prompt_clean`**: 上流（diffusers / 公式 Wan2.1）がトークナイザの前に掛ける前処理。ftfy の `fix_text` → HTML の
  entity の解除を 2 回 → 空白の畳み込み（調査 §3.1）。
- **mojibake（文字化け）**: UTF-8 を別の文字コードで読んだ結果の文字列（`cafÃ©` など）。ftfy は正規表現
  `BADNESS_RE`（4,050 文字）で「化けていそうか」を判定し、真なら複数の codec で読み直す（ヒューリスティック）。
- **HTML entity**: `&amp;` や `&#233;` のような文字参照。
- **NFC**: Unicode の正規化形式の 1 つ（合成済みの文字へ寄せる）。結果は Unicode の版（UCD = Unicode Character Database）に依存する。
- **C1 制御文字**: U+0080〜009F。**孤立サロゲート**: 対になっていない U+D800〜DFFF（JS の文字列と Python の str は持ちうる）。
- **相対位置バイアス / バケット**: T5 系の attention は位置の差（距離）を 32 個のバケット（区間）に落とし、層ごとの表
  `[32, 64]`（バケット × head）から引いた値をスコアに足す。umT5 は 24 層すべてが自分の表を持つ（調査 §1.1）。
- **入力昇格**: 形に依存する表をグラフの定数に焼かず、ホストが作ってグラフ入力で渡す手（ADR 0034 決定 2・SBV2 の相対位置表）。
- **有効長 L**: プロンプトのトークン列の長さ（末尾の `</s>` を含む）。上流は 512 までゼロ詰めしてマスクで隠す。
- **層逐次の参照**: 重みを 1 層ずつ safetensors から読んで回す CPU の参照計算。全重みを同時に載せない。
- **ホスト gather**: 語彙埋め込みの表を GPU に置かず、ホストが使う行だけを引いて逆量子化し、グラフ入力で渡す形（ADR 0085）。
- **case id / sha 行**: e2e の出力の sha256 を環境キーごとに持つ参照値の行（ADR 0106）。case id はその行の名前。

### 調査の結論（数値は調査の値）

- **規模と容量**（調査 §1.2・§5）: 格納ごとの重みは bf16 11,361,820,672 B（10.58 GiB）・i8 5,686,617,600 B（5.30 GiB）・
  i4 g32 3,551,412,224 B（3.31 GiB）。i8 の語彙埋め込みは 1,051,174,400 B（1,002 MiB）で、除くと 4,635,443,200 B（4.32 GiB）。
  上流の格納は F32 で `total_size` 22,723,641,344 B。
- **活性と計算量**（調査 §1.3）: L = 512 でも 1 層の中間は数百 MiB 以下（スコア `[64,L,L]` f32 は L = 512 で 64 MiB）で、
  重みが支配する。24 層の MAC は L = 126 で 0.587 T・L = 512 で 2.42 T。
- **op**（調査 §4.2）: 既存の語彙で書ける見込み。足りないのは `gelu_new` の手書き式（`pow(x, 3)` を畳む規則が無い）だけ。
  attention はスケール無しで head ごとのバイアス `[1,64,L,L]` を足すので、融合 attention（mask は `[1,1,M,N]` 限定）には
  入らず分解経路（bmm → add → softmax → bmm）で通す。
- **トークナイザの 3 つの定義が食い違う**（調査 §1.4）: 同じ `spiece.model` から、語彙外の文字が
  pin した transformers 5.14.1 では id 2（umT5 では `<s>`）、`tokenizer.json` と transformers 4.57.1 では id 3（`<unk>`）になる。
  2 経路の差は「語彙外の id」と「空白 24 文字の分割」だけで、後者は `prompt_clean` が先に U+0020 へ潰すので効かない。
  固定 4 プロンプトは 2 経路で id 列が同一・語彙外なし（長さ 28 / 118 / 50 / 126）。1 文字として引ける割り当て済みの文字は 19,848 個。
- **前処理**（調査 §3.2・§3.3）: ftfy の `fix_text` は表引きの処理と NFC に加えて、mojibake の修復（ヒューリスティック）と
  HTML entity の解除を持つ。「`&` と mojibake 判定の対象 449 文字を含まない」池（4,220 文字）の乱択 200,000 本では、
  決定的な処理だけの鏡像が `prompt_clean` と全件一致した。449 文字も混ぜた 50,000 本では 277 本（0.55%）が不一致。
  公式 negative の全角読点 `，` は `fix_character_width` で `,` に変わる。
- **有効長だけで回す**（調査 §2.3）: 小モデル（4 層・d_model 64 の乱数初期化）で「512 に詰めてマスク」と「有効長だけ」は
  f32 で最大相対 5.7e-7・bf16 では 7 本ともビット一致。実モデルでの一致は未確認。
- **バケット表**（調査 §4.3）: torch の f32 と f64 の `math.log` で、距離 −511〜511 の全件が一致。
- **トークナイザのライセンス**: google/umt5-xxl の cardData で `apache-2.0`（調査 §1.4 — ADR 0118 未解決の「ライセンス表記」は
  この事実で閉じる。配布形の NOTICE への記載は 10d）。

### 制約

- 横断の不変条件: 未対応・想定外は fail loudly（黙って近似しない）・sha 門の tolerance 化は禁止（ADR 0106）・
  ランタイム依存は Web 標準 API だけ（ftfy も Python の UCD もランタイムには持ち込めない）。
- B570 の Deno の総確保の天井は 9,600 MiB（保守側 — 調査 §5）。DiT 段の山は 33 フレーム 6.19 GiB・81 フレーム 7.31 GiB、
  VAE 段は 3.32 GiB（ADR 0118 段 6 / 8 の結果）。
- B570 の `maxStorageBufferBindingSize` は 2,147,483,644 B。ブラウザの WebGPU 既定は 128 MiB で、Chromium の単一 ArrayBuffer の上限は
  2,145,386,496 B（調査 §5 — Chrome の実際の上限は段 9 で測る）。
- 開発機の RAM は 31 GiB。gemma4 E4B 通常版の export はこの機で OOM した（調査 §5）。
- 段 9（Chrome）と並行して進める。段 9 は `packages/models/src/wan/pipeline.ts` に手を入れうる（調査 §7.2）。

## Decision

### 1. トークナイザ — anima の T5 実装を共通層へ一般化し、語彙外の文字は拒む

守るもの: 同じプロンプトが、どの上流の版で作った参照とも同じ id 列になること。

- **何を**: anima の T5 トークナイザ（`packages/models/src/anima/text/t5-tokenizer.ts`）を `packages/models/src/text/` へ
  一般化し、umT5 の `spiece.model` 由来の表（語彙 256,300・added token 304 個）を差し替えて使う。Unigram の本体は既に
  `packages/models/src/text/unigram.ts` にある。一般化で変えるのは 2 点（調査 §4.1）:
  - 正規化（`T5Assets.normalizer`）を省略可能にする。transformers 5.14.1 の umT5 の経路は正規化を持たない。
  - 表（語彙・added token・空白集合）を引数で受ける。経路は transformers 5.14.1 と同じ WhitespaceSplit → Metaspace →
    Unigram → 末尾に `</s>`。本文中の added token（`</s>` など）は上流どおり切り出す（実測 `a </s> b` → `[289, 1, 748, 1]`）。
- **語彙外の文字は fail loudly で拒む**。Unigram の分割に未知ノードが 1 つでも出たら、`ModelInputError`（ADR 0107）で落とす。
  受理される文字は、1 文字として引ける割り当て済みの 19,848 文字（CJK・仮名・Latin を含む — 調査 §1.4）。
- **なぜ**: 語彙外の id は transformers 5（id 2）と `tokenizer.json`・4.x（id 3）で版に依存し、どちらに合わせても
  「上流と同じ」と言える根拠が版で割れる。id 2 は transformers 5 の組み直しが元の T5 の語彙配置（`<unk>` = 2）を前提にした結果で、
  umT5 では `<s>` になる（`tokenization_t5.py:99`・`:112-127`）。拒めば、受理した入力ではどの経路でも未知ノードが出ないので、
  id 列は版に依らない。
- **id 列の正本**: pin した transformers 5.14.1 の実挙動。固定 4 プロンプトは、どの経路でも同じ id 列になる（調査 §1.4）。
- anima 側の parity fixture は動かさない（同じ入力で同じ id 列 — 一般化は振る舞いを変えない）。

### 2. 前処理 `prompt_clean` — 決定的な処理だけを表で移植し、残りは拒む（案 b'）

守るもの: 普通のプロンプト（欧文のアクセント・CJK・全角・曲がった引用符・改行）を上流と同じ文字列にすること。
上流のヒューリスティックな修復を黙って近似しないこと。

- **何を**: ftfy 6.3.1 の `fix_text` のうち決定的な処理（調査 §3.2 の表の表引きの処理 9 本 + NFC）を不動点まで繰り返し、
  その後 `strip` → 空白（`regex` の `\s` 25 文字）を U+0020 1 個へ畳む → `strip`。この合成が `prompt_clean` の鏡像になる。
- 表は recipe が ftfy から焼く（anima の正規化表と同じ流儀 — `tools/export-recipes/anima/text.py:1-20`）。TS は焼いた表を引くだけ。
- **拒む入力**（全て `ModelInputError`）:
  - mojibake の判定（`BADNESS_RE` を TS へ翻訳したもの）が真になる入力。
  - HTML entity の候補（`&` の後に entity になりうる並びが続く入力 — 調査 §3.4 の目安は「`&` の後に空白・`<`・`&` 以外」。
    `html.unescape` の規則との突き合わせで 10a に確定する）。
  - C1 制御文字（U+0080〜009F）と孤立サロゲート。
- **`BADNESS_RE` の JS 翻訳**は fuzz の parity で縛る。`\w` が Unicode 対応か・`.` が改行にどう当たるかが、Python と JS で
  割れうる点（調査 §3.4）。
- **NFC** は Python の UCD 16.0.0 を正本にし、JS の `normalize("NFC")` との一致を検査する。割れる文字は表で焼く
  （anima の前例 — `anima_tokenizer_test.ts:6`）。
- **なぜ**: 決定的な処理は表と NFC だけで、全件と乱択で鏡像と突き合わせられる。ヒューリスティックは「真になったら拒む」
  だけにでき、黙って近似しない（横断の不変条件）。受理集合を広く保てる（公式 negative の全角読点・曲がった引用符を受ける）。

### 3. 相対位置のバケット表 — ホストが `[L, L]` の i32 添字表を作り、グラフ入力で渡す

守るもの: L に依存する焼き込み定数をゼロに保つこと（ADR 0034 決定 2 と同じ）。バケットの境界が Python と TS で割れないこと。

- **何を**: ホストが `[L, L]` の i32 のバケット添字表を作り、グラフ入力として渡す。グラフは層ごとの表 `[32, 64]` を
  `embedding` で引き（`[L, L, 64]`）、`permute` で `[1, 64, L, L]` にしてスコアに足す（調査 §4.2）。
- **ADR 0118 決定 4 の文面を置き換える**: 0118 決定 4 は「i32 のバケット添字表 `[512,512]` = 1 MiB の定数 + 層ごとの
  `[32,64]` 表の gather」と書く。本決定がこれを置き換える（0118 側の追記はオーケストレータが書く）。
- **検査**: Python の生成器と TS の生成器のバイト一致テストを、E2E と別に置く（`sbv2_rel_pos_parity_test.ts` の前例 —
  ホストとゴールデンが同じ誤りを共有して E2E をすり抜ける形を塞ぐ）。
- **なぜ**: SBV2（DeBERTa の相対位置表）と Anima（RoPE 表）の入力昇格の前例がある。定数案は有効長の動的形（決定 4）で
  `[:L, :L]` の切り出しが要る。境界の値は、f64 と torch の f32 の一致を距離 ±511 の全域で実測済み（調査 §4.3）。

### 4. 系列長 — 有効長 L の動的形・マスク入力なし

守るもの: 計算量を L に比例させること（negative の 126 なら 512 の 24% — 調査 §2.3）。上流の黙った切り詰めを持ち込まないこと。

- **何を**: グラフの入力は token id `[1, L]` とバケット表 `[L, L]` で、L は 1 シンボルの動的次元。マスク入力は持たない。
  出力 `[L, 4096]` をホストが 512 行までゼロで詰めて DiT へ渡す（DiT のグラフは変えない — ADR 0118 決定 4）。
- **下限**: L = 1（空文字・空白だけのプロンプトで `</s>` だけになる）は拒む。`torch.export` は 0 と 1 を特殊化するので、
  下限は 2（anima の `PROMPT_MIN_TOKENS` の前例）。
- **上限**: 512 トークンを超えるプロンプトは fail loudly。上流は `truncation=True` で黙って切り詰めるが、採らない
  （今の recipe の `valid_length` と同じ方針 — `text_embeds.py:130-151`）。
- **実モデルでの同値**: 「有効長だけ」と「512 + マスク」の bf16 出力の一致は小モデルでしか確かめていない。実モデルでの一致を
  段 10c の検収に入れる。
- token id は exporter の境界で i32 へ正規化する（ADR 0009。語彙 256,384 は i32 に収まる）。

### 5. 重みの格納型と語彙埋め込みの置き場

守るもの: B570 で text 段が単独で入ること。ブラウザの上限が分かる前に、ホスト側の機構を先取りで作らないこと。

- **格納型**: i8 per-channel・活性 f32（ADR 0118 決定 4 の前提どおり）。i4 g32 は品質を測ってから別の quant 席として足す
  （本 ADR の範囲外 — 未解決）。
- **相対位置の表 `[32, 64]` × 24 は F32 のまま**にする。exporter の量子化が `embedding` を対象に含める規則なら、この小さな表まで
  i8 になりうる（調査 §1.2 の注意 — 規則は未確認・未解決）。
- **語彙埋め込み**: Deno では GPU の `embedding` op で始める（i8 で約 1,002 MiB の 1 バッファ・B570 の束縛上限 2,147,483,644 B の内）。
  Chrome（段 9）の上限の実測で、ホスト gather（ADR 0085 の前例）の要否を決める。
- **出力の dtype**: GPU 経路の出力は f32 のまま DiT へ渡す。上流が bf16 で出すのは格納型の都合で、i8 の重みで計算した
  時点で bf16 の値格子に合わせる意味は無い。資産の経路（bf16 の値）とは一致しない前提で、sha 行は別の case id にする（決定 8）。
- **なぜ**: GPU の `embedding` は既存の op だけで済む。ホスト gather は VRAM を約 1 GiB 減らし束縛上限の問題を消すが、
  PLE の機構は gemma 専用の形（索引の schema・sidecar）で、Wan へ流用できるかは未確認（調査「軸 G」）。

### 6. export のホスト RAM — 実測が先、足りなければ fake tensor の trace と重みの逐次 emit

守るもの: 31 GiB 機で export が完走すること。グラフを 1 本に保ち、Session と admission を単純に保つこと。

- **何を**: 10b で、F32 の重み 22.7 GB の全体 1 グラフの export を 31 GiB 機で回し、ピーク RSS を測る。
- 通らなければ、fake tensor（形だけで実データを持たないテンソル）で trace し、重みを逐次 emit する形を exporter core に足す
  （モデル非依存の部品なので core — ADR 0065）。
- **層の分割（複数グラフ）は採らない**。グラフと Session の本数が増え、text 段の admission が複雑になる。
- ADR 0118 決定 4 の「第 2 段の最初の作業はこの実測」をそのまま引き継ぐ。

### 7. 既定の経路と事前計算の資産

守るもの: 同じ文字列に対して経路が黙って変わらないこと。umT5 を取らない軽い使い方を残すこと。

- **何を**: GPU 経路（umT5 を回す）を既定にする。事前計算の埋め込み資産の経路は、明示の選択（umT5 を取らない軽い使い方）として
  残す。選択のノブの名前と置き場は 10d で決める。
- **自動の切り替え（資産にあるプロンプトは資産、無いものは GPU）は採らない**。同じ文字列でも資産の有無で埋め込みが変わり、
  再現性の説明がつかない。
- **GPU 経路では positive も negative も GPU で作る**。資産の bf16 の negative と GPU の i8 の positive を混ぜない
  （negative の 126 トークンは 0.587 T MAC — 調査 §1.3）。

### 8. 数値の門 — 移植の門と品質の記録の二層

守るもの: 「移植できた」を帯で言えること。i8 の品質の判断を、根拠の無い帯ではなく観測と目視で下すこと。

1. **移植の門**: 参照は、同じ i8 の fake-quant 重みで回した CPU の層逐次の参照（f32 か f64）。帯は「決定用ケースの最悪 × 5」で、
   受入れは別のケースで見る（ADR 0118 決定 8 と同じ導き方）。golden を fake-quant 後の重みで採るのは ADR 0026 決定 2 の前例。
   指標（比の形・参照の精度）は 10c の実測の前に固定する。
2. **品質の記録**（門ではなく観測）: GPU i8・bf16 資産・f32 参照の 3 点を、行ごとのコサイン類似度・相対フロベニウス誤差・
   最大絶対値で比べ、固定 4 プロンプト + 追加のプロンプトで記録する。最終判断は動画の視認 A/B（seed 4 本以上 × プロンプト 2〜3 種）で、
   DeBERTa の聴感裁定（ADR 0026 決定 4）と同じ位置づけ。

- **「GPU i8 vs bf16 資産」を門にしない**。bf16 自身の誤差（値 1〜2 で 1 ulp が 7.8e-3 で、出力の丸めだけでその半分を持つ。
  24 層を bf16 で回した累積は未測 — 調査 §6.1）が床になり、帯を決める根拠が無い。
- **sha 行**: GPU 経路の動画は新しい case id の行として足す（`KARUME_REFERENCE=write` — ADR 0106）。資産の経路の既存の行は残す。
- **前提（2026-10-03 に締めた — `7dc17309`・ADR 0106 追記）**: 参照門 `referenceGatePasses`（`packages/runtime/tests/helpers/reference.ts`）は登録した case id のどれか 1 本に行があれば緑で、
  新しい case id の書き忘れが警告付きの SKIP で通る（ADR 0106 決定 4 の字義・backlog の Wan 行の隣接の小物 2026-10-03）。
  新しい case id を足す前に、「全 case id に行があるか、無い id は明示の held」へ締める。

### 9. キャンセル — `generate` に `signal`（AbortSignal）を足す

守るもの: 30 分〜2 時間の生成を途中で止められること。止めたときに GPU の資源を残さないこと。

- **何を**: `WanGenerateRequest` に `signal?: AbortSignal` を足す（他の系列と同じ・利用者承認 2026-10-03・backlog の Wan の小物 ②）。
- **検査の位置**: 段の境界（text → DiT → VAE）と DiT の各 step の間。検査はイベントループへ 1 度譲ってから見る
  （`packages/models/src/concurrency/abort.ts` の `settleAbort` — 譲らない検査は「呼ばれた時点で既に中断済み」しか拾えない）。
- **中断時**: 開いている Session を畳んでから、`signal.reason` を包まずにそのまま投げる（既定の reason は `AbortError` の
  `DOMException`）。消費側が `error === controller.signal.reason` で自分の中断を識別できる形で、anima / irodori の構築と
  generation の前例（ADR 0083 決定 5）に合わせる。
- 1 段の中（1 回の run）は不可分で、中断は次の境界で効く（anima と同じ）。
- **構築側にも**: `fromPretrained` / `fromAssets` の構築オプションにも `signal` を足す（umT5 の取得は i8 で約 5.3 GiB —
  anima / irodori / sbv2 の構築と同じ形で、取得層へ渡す）。

### 10. 段の分割

守るもの: 段 9（Chrome）と並行して進められること。GPU を使う段を直列に保つこと。

| 段  | 中身                                                                                                                                                   | GPU  |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---- |
| 10a | トークナイザ（決定 1）と前処理（決定 2）の TS 実装・recipe の表と fixture                                                                              | 不要 |
| 10b | export のホスト RAM の実測と export の形の確定（決定 6）                                                                                               | 不要 |
| 10c | umT5 の export（i8・活性 f32・動的 L・バケット表は入力）・層逐次の参照・TS のバケット表生成器・GPU の移植の門・故障注入                                | 要る |
| 10d | パイプラインへの結線（text 段 → DiT 段の順次化・admission・manifest の部品・配布形とカード・`signal`・新しい case id の sha 行・品質の記録・視認 A/B） | 要る |
| 10e | 任意: i4 の席・語彙埋め込みのホスト gather                                                                                                             | 要る |

- 10a〜10c は `packages/models/src/wan/pipeline.ts` を触らない。10d は段 9 と調整してから入る。
- GPU を使う段は GPU ロックで直列化する（順序はオーケストレータが決める）。

### 11. 容量 — text 段を DiT 段の前に畳む

守るもの: B570 の天井 9,600 MiB の内で通しが回ること。

- **何を**: text 段の Session（positive と negative の 2 回の run）を、DiT 段の Session を張る前に畳む。DiT → VAE と同じ
  「段ごとに張って畳む」扱い（ADR 0118 決定 7・ADR 0112）。
- **なぜ**: text 段（i8 5.30 GiB）と DiT 段（81 フレーム 7.31 GiB）の和 12.6 GiB は天井を越える。33 フレームでも i8 は
  11.5 GiB で越える（調査 §5）。
- **配布形**: umT5 の i8 を足すと約 8.2 GiB になる（今は 2.92 GiB — 調査 §5）。i8 の語彙埋め込み 1,002 MiB は part の上限
  1,024 MiB の内だが余裕は 22 MiB で、テンソルの piece 分割（行範囲の分割 — ADR 0090）で part を跨げる。構成（同じリポか別リポか）は
  「裁定」節の 1。

## 採らなかった案

- **トークナイザを transformers 5.14.1 の実挙動（語彙外 → id 2）に合わせる** — 今の pin の上流と同じ id 列にはなるが、id 2 は
  元の T5 の語彙配置を前提にした組み直しの結果で、umT5 では `<s>` になる（調査 §1.4）。版を上げると変わりうる挙動を正本にする。
- **`tokenizer.json`・transformers 4.x（語彙外 → id 3 = `<unk>`）に合わせる** — 公式 Wan2.1 の当時の環境に近いが、要求は
  `transformers>=4.49.0` で上限が無く（調査 §1.4）、学習時の id を確かめる手段が無い。
- **`spiece.model` の原本どおり byte_fallback で語彙外を UTF-8 のバイトに落とす** — どの上流の経路（transformers 4.x / 5）もそうしておらず、
  上流のパイプラインと id 列が割れる（調査 §1.4 の表）。
- **前処理 (a): ftfy を全部移植する** — 受理集合は上流と同じになるが、移植量が最大（ftfy の本体 2,731 行 + codec 群）で、
  ヒューリスティックの parity は fuzz でしか縛れず、版を上げるたびに追従が要る（調査 §3.4）。
- **前処理 (b) / (c): `prompt_clean(x) == x` を満たす入力だけを受ける・正規化を利用者側に移す** — 公式 negative（全角読点）・
  曲がった引用符・全角英数・改行を含む普通の入力まで拒む。(c) はブラウザの利用者に Python を求めることになる（調査 §3.4）。
- **前処理の強い版（`&` と mojibake 判定の対象 449 文字を含めば拒む）で始め、`BADNESS_RE` の移植は後の段にする** —
  調査 §3.4 が段の分け方として挙げた案。fuzz で鏡像と全件一致した範囲そのものだが、アクセント付きの欧文と `’` を拒む。
  `BADNESS_RE` の翻訳を最初から入れ、受理集合を狭める段を作らない。
- **バケット表を `[512,512]` の定数に焼いて `[:L,:L]` を切り出す**（ADR 0118 決定 4 の元の文面）— L に依存する切り出し
  （`sym_prefix_slice`）が要り、焼き込み定数を持つ。入力昇格の前例（決定 3）を採る。
- **バケットをグラフの中で計算する**（`arange` → 差 → `abs`・`log` → `where` → `min`）— 前例はホストで作る形で（調査 §4.2）、
  GPU の `log` の丸めと境界の一致を別に縛ることになる（推測）。
- **512 固定 + マスク入力** — 計算量が常に 512 ぶんになり、マスクとバイアスの加算が要る（調査「軸 D」）。
- **512 トークン超を上流どおり黙って切り詰める** — 「未対応・想定外は fail loudly」に反する（調査「軸 E」）。
- **i4 g32 で始める / i8 と i4 の両方の席を最初から持つ** — mlx-umt5 の報告では group 量子化（MLX の group-affine）で出力が壊れた。
  karume の対称 group 32 で同じ壊れ方をするかは未測（調査「軸 F」）。品質を測ってから席を足す。
- **語彙埋め込みを最初からホスト gather にする** — VRAM を約 1 GiB 減らすが、PLE の機構は gemma 専用の形で流用できるかは未確認
  （調査「軸 G」）。Chrome の上限の実測を待つ。
- **export を層ごとの複数グラフに分ける** — グラフと Session の本数が増え、text 段の admission が複雑になる（調査「軸 H」）。
- **事前計算の資産を退役する** — umT5 の取得（i8 で約 5.3 GiB）を必須にし、今の 2.9 GiB の使い方を失う（調査「軸 I」）。
- **資産にあるプロンプトは資産、無いものは GPU（自動の切り替え）** — 決定 7。
- **GPU 経路で negative だけ資産の bf16 を使う** — 決定 7。1 本の CFG の中で精度の違う埋め込みが混ざる。
- **「GPU i8 vs bf16 資産」を数値の門にする** — 決定 8。bf16 自身の誤差が床で、帯の根拠が無い。
- **資産の経路の既存の sha 行を GPU 経路の値で焼き直す** — 資産の経路は残るので、その行も現役（決定 8）。tolerance 化は禁止。

## Consequences

- 公開面: `@karume/models/wan` の `generate` が資産の集合の外のプロンプトを受ける（GPU 経路）。受理集合は「19,848 文字の内・
  mojibake / entity の候補 / C1 / 孤立サロゲートを含まない・2〜512 トークン」に変わる。`WanGenerateRequest` に `signal` が増える。
  未リリースなので形は変わってよい。limitations の Wan2.1 節を書き換える。
- 配布形が約 8.2 GiB（同じリポに入れる場合）になる。トークナイザの資産（語彙とスコアの JSON）は 8,576,480 B（gzip 3,318,348 B）で、
  anima の T5 の同じ部分（1,059,227 B）の 8 倍（調査 §4.1）。
- 1 回の generate に text 段（重み i8 5.30 GiB の読み込みと 2 回の run）が増える。text 段の所要は未測。
- IR の語彙は増えない見込み（`gelu_new` の扱いしだいで exporter core の正規化パスが 1 本増える — 未解決）。

### 影響ファイル

| 区分             | ファイル                                                                                                                                                                    | 段       |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| トークナイザ     | `packages/models/src/text/`（T5 の一般化）・`packages/models/src/anima/text/t5-tokenizer.ts`（共通層を使う形へ）                                                            | 10a      |
| 前処理           | `packages/models/src/wan/` の新規ファイル（`prompt_clean` の鏡像・焼いた表の読み口）                                                                                        | 10a      |
| recipe（CPU）    | `tools/export-recipes/wan/` の新規台本（トークナイザの表と fixture・ftfy の表の焼き出し・NFC の表）                                                                         | 10a      |
| exporter core    | `tools/exporter/src/karume/`（fake tensor の trace と逐次 emit — 10b の実測しだい・`pow(x,3)` の正規化 — gelu_new の扱いしだい・量子化の対象から表を外す指定 — 規則しだい） | 10b・10c |
| recipe（export） | `tools/export-recipes/wan/` の新規台本（umT5 の export・層逐次の参照・バケット表の Python 生成器）                                                                          | 10c      |
| ホスト           | `packages/models/src/wan/` の新規ファイル（バケット表の TS 生成器・text 段）                                                                                                | 10c      |
| パイプライン     | `packages/models/src/wan/pipeline.ts`・`text-embeds.ts`・`config.ts`                                                                                                        | 10d      |
| 配布形           | `tools/export-recipes/wan/distribution.py`・`card.py`・`karume.json`（または別の配布リポの recipe — 裁定 1）                                                                | 10d      |
| 参照門           | `packages/runtime/tests/helpers/reference.ts`（`referenceGatePasses` を締める）                                                                                             | 10d の前 |
| テスト           | `packages/models/tests/` の新規（トークナイザ・前処理・バケット表の parity・umT5 の e2e）・`fixtures/references/wan.json`・`deno.json` のレーン                             | 10a〜10d |
| 文書             | `docs/limitations.md`・`docs/glossary.md`・`docs/quantization.md`・`docs/backlog.md`・`.claude/ACTIVE_DESIGN.md`・ADR 0118 の追記                                           | 各段     |

### 費用とリスク

- **export のホスト RAM**（決定 6）: F32 の重みだけで 22.7 GB。全体 1 グラフが通らなければ exporter core に逐次 emit を足す作業が増える。
- **層逐次の参照の RAM と所要**: f64 の全重みは 45 GB で 31 GiB 機に載らない。1 層は f64 で約 1.5 GB（調査 §6.2 — 推測）。所要は未測。
- **i8 の品質**: i8 per-channel の重みの丸めは相対 1% 前後で、bf16 の重みの丸め（相対 rms 約 2e-3）より数倍大きい見込み（調査 §6.1 — 推測）。
  視認 A/B で受けられなければ、格納型の見直しになる。
- **前処理の parity**: `BADNESS_RE` の翻訳と NFC の版差は fuzz と全件検査で縛るが、fuzz は網羅ではない。割れた入力は拒む側へ倒す。
- **段 9 との重なり**: 10d は `pipeline.ts` を触る。exporter core を触る 10b / 10c は w8a8 の席の作業（research 2026-10-03-wan-w8a8-recon）と
  重なりうる（調査 §7.2）。
- **text → DiT の切り替えの残り**: DiT を畳んだ直後に初回だけ確保が残る現象（段 6 で約 0.24 GiB・段 8 で約 0.45 GiB — ADR 0118 未解決）が、
  text 段の後にも出るかは未測。81 フレームの DiT 段（7.31 GiB）の前に残れば、天井との余裕が減る。

## 検収

| 段  | 中身                                                                                          | 検収の形（緑の条件）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | GPU  |
| --- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| 10a | トークナイザ（決定 1）と前処理（決定 2）の TS 実装・recipe の表と fixture                     | トークナイザ: recipe が transformers 5.14.1 で作る fixture（固定 4 本〈長さ 28 / 118 / 50 / 126〉+ 境界ケース + 乱択）と id 列が一致。受理した入力では `tokenizers` の `tokenizer.json` の経路とも id 列が一致（版に依らないことの検査）。anima の parity fixture は無変更で緑。前処理: ftfy 6.3.1 の `prompt_clean` との fuzz の parity（調査 §3.3 と同じ池と本数以上）が不一致 0。`BADNESS_RE` の JS 翻訳が 449 文字を混ぜた池の fuzz で `is_bad` と全件一致。NFC は Python の UCD 16.0.0 と全コードポイントで一致（割れる文字は表）。公式 negative の `，` が `,` になる。拒否ケース（語彙外・entity の候補・mojibake・C1・孤立サロゲート・空と空白だけ〈L = 1〉・512 トークン超）が全て `ModelInputError`。故障注入（表を 1 本外す・NFC を飛ばす）で parity が赤。`test:models:wan` と anima のレーン・両方の `uv run pytest` が緑 | 不要 |
| 10b | export のホスト RAM の実測と export の形の確定（決定 6）                                      | 全体 1 グラフの export のピーク RSS を記録し、31 GiB 機で完走するかを決める。通らなければ fake tensor の trace と重みの逐次 emit を exporter core に足し、その形で完走（core の pytest が緑）。決めた形を本 ADR の追記に書く                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 不要 |
| 10c | umT5 の export・層逐次の参照・TS のバケット表生成器・GPU の移植の門・故障注入（決定 3〜5・8） | IR に語彙外の op が無い。RMSNorm が `rms_norm` に畳まれたかと `gelu_new` の扱い（丸めの差）を記録。相対位置の表 24 本が F32 のまま。入力は token id `[1,L]`（i32）とバケット表 `[L,L]` で、L の受理は 2〜512。バケット表の Python / TS 生成器がバイト一致。実モデルで「有効長だけ」と「512 + マスク」の bf16 出力が固定 4 本で一致（CPU）。GPU の出力が層逐次の参照（同じ i8 の fake-quant 重み）と、決定用ケースの最悪 × 5 の帯で一致し、受入れは別ケースで帯の内。故障注入（バケット表を 1 ずらす・層の表の取り違えなど — 帯を決めた後に固定）が帯の外。VRAM の山・GPU 時間・1 submit の GPU 時間の最大（ADR 0118 決定 6 の 1 s を目安）を記録                                                                                                                                                                                       | 要る |
| 10d | パイプラインへの結線（決定 7・8・9・11）                                                      | GPU 経路が既定で、資産の経路は明示の選択でだけ動く。text 段を畳んでから DiT 段を張る（VRAM の山と切り替えの残りを fdinfo で記録）。資産の経路の既存の sha 行が 1 本も変わらない。参照門を締めた上で、GPU 経路の新しい case id の sha 行を `KARUME_REFERENCE=write` で作り、続けて回すと一致。品質の記録（3 点比較 × 固定 4 本 + 追加）を results に残す。`signal`: 段の境界と step の間の中断で `signal.reason` がそのまま投げられ、Session が畳まれ、次の generate が回る。配布形が組めて `verify_dist` が緑・カードに資源の目安。フル verify が緑。利用者の視認 A/B（seed 4 本以上 × プロンプト 2〜3 種）                                                                                                                                                                                                                            | 要る |
| 10e | 任意: i4 の席・語彙埋め込みのホスト gather（決定 5）                                          | 段 9 の Chrome の上限と 10d の品質の記録を見てから、検収の形を追記で決める                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 要る |

## 裁定

利用者に判断を仰ぐ点（推奨案つき）。

1. **配布形の構成** — umT5 をどこに置くか。
   - (a) 同じ `karume-wan2.1` に入れる（約 8.2 GiB の 1 リポ）。
   - (b) 別の配布リポに分け、`karume-wan2.1` の側から参照する。リポ名は ADR 0092 決定 2 の命名に従って 10d で決める。
   - **推奨 (b)**。umT5 を取らない軽い使い方（今の 2.9 GiB — 決定 7）を、取得の単位として保てる。Wan2.2 などの後の世代が
     同じ umT5 を使うなら共有できる（推測 — 未確認）。参照の形（manifest の越境参照〈容器単位・40 桁の commit SHA の二重 pin —
     ADR 0109 決定 3〉か `pipelineConfig` か）は 10d で決める。越境参照は参照先を先に公開して SHA を確定させる順序の制約を持つ
     （docs/release-runbook.md §0）。
2. **視認 A/B の素材** — 自由プロンプトの候補 2〜3 本は利用者が選ぶ（固定 4 本は第 1 段の資産と同じ文面で比べられる）。

## 未解決

- **実モデルでの「512 + マスク」と「有効長だけ」の同値**（小モデルでしか未確認 — 10c で確かめる）。
- **transformers 4.x の全域の unk_id**: 4.57.1 で id 3 を確認したが、要求の下限 4.49.0 自体は未確認（調査 §1.4）。決定 1 で語彙外を拒むので
  karume の id 列には効かないが、公式の当時の学習の id は分からないまま。
- **`BADNESS_RE` と NFC の parity**: fuzz と全件検査の結果しだいで、拒否の範囲や NFC の表が増える（10a）。
- **相対位置の表が i8 化されないこと**: exporter の量子化の対象選択の規則が `embedding` を含むかは未確認。含むなら F32 のままにする指定が要る（10c）。
- **`gelu_new` の扱い**: recipe で活性を `nn.GELU(approximate="tanh")` に差し替えるか、`pow(x, 3)` の正規化を exporter core に足すか。
  どちらも上流の eager と丸めが変わりうるが、差は未測（10c）。
- **export のホスト RAM**（10b の実測）。
- **i8 / i4 の誤差の実測**: i8 の重みの丸めが出力に与える差（品質の記録）と、i4 g32 が mlx-umt5 の報告のように壊れるか（10d・10e）。
- **text → DiT の切り替えの残り VRAM**: DiT の後に見えた初回だけの残りが text 段の後にも出るか（10d）。
- **Chrome の上限**: 語彙埋め込み 1,002 MiB の 1 バッファ・スコア 64 MiB（L = 512）がブラウザの束縛上限と単一 ArrayBuffer の上限に
  どう当たるか（段 9 の実測 → 10e）。
- **段 9 との重なり**: 10d が `pipeline.ts` を触る順序と、exporter core を触る作業（w8a8 の席）との調整。

## 追記（2026-10-03）: 段 10a の結果と決定 1 / 2 の改訂

- **段 10a ✅**（コミット `b5a4592f` / `d601c488` / `5e95f587`）: T5 トークナイザを共通層（`packages/models/src/text/t5-tokenizer.ts`・
  家族で割れる 3 点は `T5Policy`）へ一般化し、anima のパリティは不変。umT5 の資産は recipe（`wan/umt5_tokenizer.py`）が
  transformers 5.14.1 の backend から `outputs/series/wan2.1-umt5-tokenizer/tokenizer.json`（語彙 256,300・追加語彙 304・空白集合・
  promptClean の表を 1 本に）へ焼く。TS は `packages/models/src/wan/text/{tokenizer,prompt-clean}.ts`（公開面には未結線 — 10d）。
- **検収の結果**: fixture は固定 4 本 + 境界 30 本 + 乱択 200 本（受理したケースは transformers 5.14.1 と tokenizer.json の 2 経路の
  id 列が一致・生成時に assert）。前処理の fuzz は安全な池 200,000 本・全池 50,000 本・entity の池 20,000 本で上流 `prompt_clean` と
  不一致 0。`BADNESS_RE` の JS 翻訳は全池と Unicode 依存の構文 6 文脈 × 全コードポイントで `is_bad` と一致。NFC は Deno
  （V8 = Unicode 16）と Python UCD 16.0.0 が割り当て済み 292,531 文字 × 8 文脈で不一致 0 — **表は焼かない**。故障注入
  （表を 1 本外す・NFC を飛ばす・`</s>` を落とす・語彙外の拒否を外す・512 超を切り詰めにする）は全部赤。
- **決定 1 の改訂（実測で前提が崩れた）**: 本文中の追加語彙（`</s>`・`<extra_id_*>` など 304 個）と空白の直後の `▁` は、
  transformers 5 と tokenizer.json（transformers 4.x）で id 列が割れる（乱択 60,000 本で、割れた全件がこの 2 つか語彙外を含む）。
  決定 1 の守るもの（どの版の参照とも同じ id 列）に合わせ、この 2 つも `ModelInputError` で拒む。調査 §1.4 の「2 経路の差は
  語彙外と空白 24 文字だけ」は 1 文字の掃引による結論で不完全だった。
- **決定 2 で確定したこと**: HTML entity の候補は正規表現（各周の先頭で ftfy の `HTML_ENTITY_RE`・`fix_text` の後で
  `html.unescape` の `_charref` と同値の形）で判定し、名前表は持たない — `R&D` のように entity にならない並びも拒む
  （`R & D` と空白を挟めば通る。名前表 2,231 個を焼いて正確に絞るのは要望が出てから）。UCD 16.0.0 で未割り当てのコードポイントも
  拒む（新しいエンジンの NFC が Python と割れうるため — 語彙にある未割り当て 60 文字も拒む）。拒否の reason は
  `PromptCleanError`（`ModelInputError` の派生・unassigned / c1 / entity / mojibake）。
- **未検証**: ブラウザの NFC の一致（Chrome / Safari の ICU の版 — 段 9 で sweep のテストをブラウザでも回す）。
- **10d への手掛かり**: hub の `readAssetJson` → `parseWanTokenizerAsset` → `new WanPromptEncoder(assets)` →
  `encode(prompt, 役割)` が `Int32Array [L]`（L は 2〜512・末尾は `</s>`）。

## 追記（2026-10-03）: 段 10c の準備の結果 — 未解決 2 件を閉じる・gelu の裁定

- **準備 ✅**（`17067949` / `e6ed06c5` / `246c1d50`・GPU と実重みは未使用）: 有効長ラッパ `Umt5EncoderTokens`（入力 `input_ids [1,L]` と
  `relative_position_buckets [L,L]`・出力 `[1,L,d_model]`・マスクなし・L は 2〜512 の記号次元）で、小さな乱数 UMT5（4 層）の
  S 形 export が exporter の verify を通り、容器と golden を書ける。相対位置のバケット表は決定 3 のとおり Python（上流の
  `_relative_position_bucket` をそのまま呼ぶ）と TS（f64 の `Math.log`）で L = 2〜512 の全域がバイト一致（fixture は pin した
  config から焼く）。Session の入出力の純関数（`src/wan/umt5/session-io.ts`）まで。
- **閉じた未解決 1 — 相対位置の表の i8 化**: exporter の i8 の既定は `nn.Embedding` も丸める（`QUANT_CHANNEL_AXES`）ので、表 24 本
  （`encoder.block.N.layer.0.SelfAttention.relative_attention_bias.weight`）は `fake_quant_int8` の `include` で外し、かつ `emit` の
  `weight_dtype_overrides` で F32 を明示する（片方だけでは落ちる — 実測）。core の変更は要らない。
- **閉じた未解決 2 — `gelu_new` の扱い（裁定 (a)）**: 上流の手書き式は `aten.pow.Tensor_Scalar` で export が落ちる。recipe で
  `nn.GELU(approximate="tanh")` に差し替える（数学的に同じ関数・IR は融合済みの `gelu_tanh` 1 本）。丸めの差は要素の 0.54% が
  違い最大 4.77e-7、小模型の出力の比で 3.9e-7（表の故障は 2.6e-2 以上で 5 桁離れる）。`pow(x,3) → x·x·x` の正規化を core に
  足す案（上流 eager とビット一致する）は、活性が要素ごとの op 8 本に増えるので採らない。
- **決定 4 の裏付け（小模型）**: 「有効長だけ」と「512 + マスク」は bf16 でビット一致（L = 2〜512 の 8 本）、f32 は最悪比 7.0e-7。
  実モデルでの確認は 10c の検収のまま。
- **決定 8 への注意（コードを読んだ事実）**: 上流の UMT5 は RMSNorm の分散と attention の softmax を f32 に落とすので、`.double()`
  だけでは f64 の参照にならない。層逐次の f64 参照は DiT の `reference_dit_f64` と同じく精度を意識した書き下しが要る（10c）。
- 下見の容器は配布しないので台本の CLI からは書かない（`tests/test_graph_names.py` の門 — 部品名は 10d で決める）。

## 追記（2026-10-03）: 段 10b の結果 — 決定 6 の形の確定・決定 4 / 8 の補足

- **段 10b ✅**（`e70678b7` / `56e91b68`・[research 2026-10-03 umt5-export-ram](../research/2026-10-03-umt5-export-ram.md)）: 素直な形
  （全重みを f32 で持つ → fake-quant → export → emit）は、31 GiB 機で山が約 30.5 GiB になる見込み（推計: f32 の重み 21.2 GiB が
  COW で匿名メモリになり、語彙埋め込み 3.91 GiB の i8 変換の一時 2.25 倍が乗る）。実走はしていない。
- **決定 6 の確定（文面の改訂）**: 「fake tensor で trace して重みを逐次 emit する形を exporter core に足す」ではなく、**core の変更なし**で
  組めた形を採る — trace は重みを持たない meta の上流で回し（実重みの export とグラフの JSON が同一）、量子化の対象 169 本（linear
  168 + 語彙埋め込み）は checkpoint から行の塊（16,384 行）ごとに i8 にして `fixed_weights`（ADR 0097 の入口）で渡し、F32 の 73 本
  （相対位置の表 24・RMSNorm 49）はそのまま渡す。実モデルで山 6.79 GiB・69 s・容器 5,686,978,107 B（26 part）・`karume verify` 緑。
  影響ファイル表の「exporter core（fake tensor の trace と逐次 emit）」の行は不要になる。上流の text_encoder の格納は bf16 ではなく
  F32（22.7 GB・5 分割）だった（調査 §1.2 の bf16 10.58 GiB は計算値）。
- **決定 4 の補足（実モデル・CPU）**: 「有効長だけ」と「512 + マスク」は **f32 で固定 4 本ともビット一致**（GPU 経路の活性の dtype）。
  bf16 では 4 本中 3 本が割れる（比 2.4e-2〜1.9e-1）が、bf16 の経路そのものの f32 に対する誤差（比 3.0e-2〜9.4e-1）以下。bf16 の
  「512 + マスク」は既存の text_embeds 資産と 4 本ともビット一致（資産の再現性の確認）。**検収 10c の文面は「f32 でビット一致」に
  読み替える**（bf16 の一致は求めない）。
- **決定 8 の補足**: bf16 の資産と f32 の差は大きい（相対フロベニウス 2.6e-2〜1.3e-1・行ごとのコサイン類似度の最小は ferret の
  行 64 で 0.168）。品質の記録（②）の 3 点比較では **f32 参照を基準**にし、bf16 資産との差は「bf16 自身の誤差」として並べる
  （bf16 を基準にすると、この差が床になる）。どちらが真値に近いかは f64 の参照（10c）で決まる。
- 10c への手掛かり: 容器を書く関数と `tests/test_graph_names.py` の ENTRIES・`WAN_WEIGHTS` の行は部品名の裁定（10d）の後。参照の
  重みは packed × scale（行ごと）で容器から 1 本ずつ作れる。fake-quant 後の f32 で全体を 1 回 forward すると匿名メモリ約 21 GiB
  なので、参照は層逐次（決定 8 ①）。bf16 の読み込み（text_embeds と同じ呼び方）は zram swap を一時に使い切る観測があり、他の重い
  処理と重ねない。

## 追記（2026-10-03）: 段 10c の結果（CPU 側）— 層逐次の参照・golden・品質の記録の基準

- **参照の書き手 ✅**（`8ff54f06` / `6e325a0e`）: umT5 encoder の forward を f64 で書き下し（埋め込み → 24 × 〈RMSNorm → 相対バイアス
  つき自己 attention〈スケール無し〉→ 残差 → RMSNorm → gated GELU〈tanh〉FFN → 残差〉→ 最後の RMSNorm・マスク無し・有効長だけ）、
  層逐次で回す（1 層ぶんの重みだけを持つ — RSS の山 3.5 GiB）。重みは i8 系列の容器から 1 本ずつ読む（packed × 行の scale =
  fake-quant と同じ値・block の sha256 を検証）。上流は RMSNorm の分散と softmax を f32 に落とすので `.double()` では f64 にならず、
  書き下しは「f64 の経路で f64 以外の浮動小数を作らない」ことを `TorchDispatchMode` で縛る。小模型で f32 の書き下しは上流の eager
  （gelu 差し替え・fake-quant 済み）と L = 2〜512 でビット一致。実モデルの f32 の参照を上流の eager と直に突き合わせてはいない
  （全重みの f32 で約 18 GiB の RSS が要る）。
- **ケースと golden**: 決定用 6 本 = `parity.json` の受理した乱択から seed 20261003 で選んだ単体 3 本（L = 8 / 22 / 36）と、乱択を
  空白で連ねた合成 3 本（L = 163 / 327 / 488 — 乱択に 512 付近が無いため）。受入れ 4 本 = 固定プロンプト（L = 28 / 118 / 50 / 126）。
  golden は `reference.<case>.safetensors`（入力 `input_ids` [L]・`relative_position_buckets` [L,L]・`output.f64`〈f32 に丸めて格納
  — TS は F64 を読まない・丸めの比 3〜5e-8〉・`output.f32`・受入れだけ `output.unquantized.{f64,f32}`・メタに容器の part 0 の sha256）。
  10 本で 46 MB・i8 の参照 123 s + 量子化なし 41 s。容器を書き直したら `python -m wan.umt5_export reference` で golden も書き直す
  （ホストのテストが part 0 の sha256 の食い違いを赤で知らせる）。
- **正規化の分母（CPU f32 の参照の f64 に対する比）**: 4.1e-7〜3.6e-6（DiT の S = 14,040 と同じ桁）。TS が格納値で採る分母と真の
  分母の差は最大 2.1%。
- **決定 8 ② の基準（改訂）**: 品質の記録の基準は**量子化なしの CPU f32**（基準自身の f64 に対する誤差は 7e-7〜2.3e-6）。i8 の丸めの
  影響と bf16 の影響を分けて並べる（CPU の値・行ごとのコサインの最小 / 相対フロベニウス）: i8 = boxing-cats 0.982 / 5.0e-2・ferret
  0.573 / 7.7e-2・cat-dog-baking 0.993 / 3.4e-2・negative 0.998 / 2.3e-2。bf16 資産 = 0.997 / 3.1e-2・**0.168** / 7.2e-2・0.768 / 1.3e-1・
  0.987 / 2.6e-2。i8 の丸めは小さくないが bf16 資産と同じ桁で、最終判断は視認 A/B（②）。
- **故障注入**: 相対位置の表の 1 ずらし・1 トークンの置き換え・層 0 / 1 の表の取り違え（CPU の見積りはどれも r 5e5〜9e5）に、i8 固有の
  層 12 の `wo` の per-channel scale × 2（帯の外を判定）と × 1.0001（r を記録 — CPU の見積りでは r は δ に比例し 21〜86。q / k は
  softmax の温度として非線形に効くので避けた）。帯が決まったら DiT の SUBTLE_FAULT_MARGIN に当たる床を umT5 でも持つかを決める。
- **GPU の門**（`10614a32` / `3ba0174a`・`e2e_wan_umt5_test.ts`）: TS のトークナイザとバケット表が golden の入力とビット一致（GPU 不要）・
  正規化比 r の帯（`UMT5_NORMALIZED_BAND` は未導出 = undefined で実走が候補を出す）・品質の記録・資源の記録・計測モードの GPU 時間
  （換算表は `helpers/timestamp-unit.ts` に移した）。**GPU の実走は視認素材の生成の後**（結果は次の追記）。

## 裁定（2026-10-03・利用者）

1. **配布形の構成 = (b) 別の配布リポ**。理由: umT5 無しの軽い使い方（今の 2.9 GiB）を取得の単位で保つ。umT5-XXL は Wan2.1 専用では
   なく汎用の encoder なので、別リポで他の系列からも参照できる形にする。リポ名と参照の形（manifest の越境参照か `pipelineConfig` か）は
   10d で決める（ADR 0092 決定 2・ADR 0109 決定 3）。
2. **視認 A/B の素材**: まず固定プロンプトで（自由プロンプトの候補は後で）。
3. **10a の受理集合の裁定**（追加語彙・空白直後の `▁`・entity 候補・未割り当て文字を拒む）: 利用者は「Wan / umT5 側の制限なら異論なし・
   回避できるなら回避策を・エラー文言が分かりやすければ一旦 OK」。→ 拒否の文言は直し方（例: `R&D` は `R & D` と空白を挟む）まで言う
   （10d で確かめる）。entity は名前表（Python の `html.entities.html5`・2,231 個）を焼いて「実際に文字列が変わるときだけ拒む」正確な門
   へ狭める回避策を backlog に積む（要望が出てから — 段 10e 相当）。

## 追記（2026-10-03）: 段 10d の設計 — 別リポへの越境参照・経路の選択・signal

材料は [research 2026-10-03 umt5-wiring-recon](../research/2026-10-03-umt5-wiring-recon.md)。裁定 1（別リポ）を受けて次のとおり決める。

- **A. 参照の形 = (b) 越境参照**: Wan の manifest（`karume-wan2.1`）が `text_encoder` の容器を umT5 リポへ越境参照する（hub の
  FileRef = repo + 40 桁の commit SHA の二重 pin — ADR 0109 決定 3・実装済み・前例 anima-extra → karume-anima）。利用者は source 1 つ
  で両方を取り、Wan の revision が umT5 のバイト列を pin する（再現性）。hub の追加は要らない。費用: 全 quant 席の `requiredLimits` が
  語彙埋め込み 1,050,148,864 B の束縛 / バッファ上限を要求する（quant の weights は完全写像で、資産の経路でも宣言は同じ — 10e のホスト
  gather で下げられる）。
- **B. 経路の選択 = 構築時のオプション** `textEncoder: "gpu" | "precomputed"`（既定 `"gpu"` — 決定 7）。取得する部品が経路で変わるので
  取得の前に決める。`"precomputed"` は umT5 の部品を取らず、資産のプロンプト 4 本だけを受ける（今の挙動）。生成の要求ごとの切り替えや
  quant 席での表現は採らない（席の weights は完全写像で umT5 を持たない席を表せない）。
- **C. トークナイザ資産 = Wan リポの自前の資産**（`WAN_ASSETS`・8 MB・形式 `karume-wan-umt5-tokenizer/1` は Wan の前処理の表を束ねている）。
  資産は常に全数を取るので、umT5 リポに置くと資産の経路まで越境する。汎用のトークナイザは他の消費者が出た日に umT5 リポへ。
- **D. umT5 リポ = `karume-umt5-xxl`**（ADR 0092 決定 2 の `karume-<family>-<変種>`）・pipeline 名 `umt5-encoder/1`（読む TS の家族は
  無く、役を名乗る）・quant 席は `i8` の 1 つ・部品名 `text_encoder`。出所は Wan-AI/Wan2.1-T2V-1.3B-Diffusers の text_encoder（umT5-XXL
  encoder の bf16 の写しを F32 で格納したもの — google/umt5-xxl との重みの同一は未確認。カードと NOTICE にそう書く）。
- **E. 未公開期間の SHA = 仮の SHA（40 桁の 0）+ 機械の門**: ローカルのミラーは `dist.py --ref-*` で umT5 のローカル配布形から
  size / sha256 を採り、revision は仮の値で組む（hub のローカル取得元は crossRepo の明示 mapping で解き revision を見ない）。公開の
  手順（HF への upload・published-smoke）は仮の SHA を**拒む**門を持つ。公開の順序は umT5 リポが先（release-runbook §0）。
- **F. text 段の Session の寿命 = generate ごとに張って畳む**（決定 11・DiT 段 7.31 GiB と同居不可）。所要は 10d-4 で測る。
- **G. signal**: 構築は `WanPipelineOptions.signal`（取得層と構築の境目へ — anima / irodori と同じ）、生成は `WanGenerateRequest.signal`
  （段の境目・DiT の各 step の間・VAE のタイルの間で `settleAbort`、開いている Session を畳んでから `signal.reason` を包まず投げる —
  `generation/sequence.ts` の形）。
- **プロンプトの意味**: GPU 経路は任意の文字列を受け（prompt_clean の鏡像とトークナイザの門）、`pipeline.prompts` は資産の名前の一覧の
  まま（例示と precomputed の受理集合）。negative の既定は TS の定数（公式の sample_neg_prompt）で、資産の `negative` の原文と一致する
  ことをテストで縛る。positive も negative も GPU で作る（決定 7）。
- **拒否の文言**: prompt_clean の拒否は直し方まで言う（`R&D` → `R & D`・未割り当て / C1 は「その文字を外す」）。
- **sha 行**: GPU 経路は新しい case id（例 `gpu-text-2step-…`）。資産の経路の既存の行は `textEncoder: "precomputed"` を明示して守る。
  参照門 `referenceGatePasses` の抜け（どれか 1 本で緑）は横断の変更として別に締める（10d-4 の前）。
- **段の分割**: 10d-1 umT5 の配布 recipe（`dist.py --pipeline umt5`・LICENSE / NOTICE / カード・graph_name の登録）と Wan の配布形の
  越境参照・トークナイザ資産・NOTICE の書き換え・仮 SHA の門・ローカルミラー 2 本の再生成（CPU）/ 10d-2 WanPipeline の結線（経路の
  選択・admission の umT5 の契約・text 段・negative の定数・signal・examples / gpu-lab の追従・ホストテスト）（CPU）/ 10d-3 参照門の
  締め（横断・CPU）/ 10d-4 GPU: GPU 経路の sha 行・品質の記録・text 段の所要と VRAM の山・視認（自由プロンプト）。

## 追記（2026-10-03）: 段 10d-1 / 10d-2 の結果 — 配布リポと結線（GPU の実走は次の追記）

- **10d-1 ✅**（`e8c86279` / `1d52ced4` / `d992caf9`）: `dist.py --pipeline umt5` で `karume-umt5-xxl`（model `xxl`・pipeline
  `umt5-encoder/1`・quant `i8`・`text_encoder` 26 part）。Wan の配布形は `weights.text_encoder` を越境参照（`--ref-*`・完全写像で
  全席へ）し `assets.umt5_tokenizer` を持つ。**全席の `requiredLimits` は maxBufferSize = maxStorageBufferBindingSize =
  1,050,148,864**（語彙埋め込み 1 バッファ・precomputed の経路でも同じ宣言 — 10e のホスト gather で下げられる）。仮の SHA の門は
  recipe の `dist.py`（`--allow-placeholder-ref` の明示が無ければ書く前に落ちる）に置いた — hub の parse も core の組み立ても形しか
  見ず、仮の SHA が入る口は `--ref-revision` の 1 か所だから。公開の手順は umT5 リポ先行 → 実 SHA で Wan を焼き直す。既存の席 3 つの
  DiT / VAE / 資産と現物 31 本はバイト不変。配布形の門番に `karume-umt5-xxl` を足した。
- **決定 D の文言の訂正**: umT5 の容器の F32 のまま格納された 73 本は bf16 で表せる値ではない（249,856 要素中 249,854 で下位 16 ビットが
  非 0）。上流の text_encoder は F32 で、「bf16 の写し」ではない。カードと NOTICE は「上流は float32・config は google/umt5-xxl を名乗る・
  重みの同一は未確認」とだけ書く。
- **10d-2 ✅**（`b7d2784c` / `b935782c` / `cde8a1a0`）: `WanPipelineOptions.textEncoder: "gpu" | "precomputed"`（既定 gpu）。admission の
  `umt5Contract`（入力 2 本の名前と形・記号 L が同じ・出力 f32 `[1,L,W]` で W = DiT の文脈の幅・有効長 512 ≤ 文脈の行数・格納 f32 / i8）
  と、gpu 経路でトークナイザ資産の宣言を見る門。text 段（`#encode`）は positive → negative の順に run し、各 run の前に settleAbort、
  出力の非有限は名指しで落とし、DiT の前に disposeSteps で畳む。negative の既定は TS の定数（公式 sample_neg_prompt の逐語 — recipe の
  fixture と資産の行との一致をホストテストで縛る）。signal は構築（取得層へ・構築の境目で settleAbort）と生成（入口 → text・各 text run・
  text → DiT・DiT の各 step・DiT → VAE・VAE の各タイル）。prompt_clean の拒否文言に直し方。ホストテスト 17 本（模擬 Session・故障注入 9 件
  が赤を確認）。examples は `--text-encoder` / `--umt5-source`・任意のプロンプト、gpu-lab は経路の選択と自由プロンプト・`/models/umt5/`
  （タブの既定は precomputed のまま — 段 9 を実走中の挙動を保つ）。
- **判断**: `karume-umt5-xxl` を `*_SOURCES` / published-smoke / RELEASE_REPOSITORIES のどこに載せるか（TS の家族を持たないリポ）は
  公開の回に決める（backlog）。越境参照なしの自己完結の焼き（8.2 GiB）は core の既定どおり許す（カードは事実どおり描く）。e2e の自由
  プロンプト（"A red fox trots through fresh snow in a quiet birch forest at sunrise."・ID `gpu-text-2step-free-fox-seed42`）は仮置き —
  文面を変えたら ID も変える。
- **次（10d-4・GPU）**: 新ミラーで通し e2e（precomputed の sha 行 8 本の不変・gpu-text の 2 ケースの sha 行 write・text 段の所要と
  切り替えの VRAM の残り）。10d-3（参照門の締め・横断）はその後。

## 追記（2026-10-03）: 段 10c の GPU の門と段 10d-4 の結果 — umT5 の GPU 経路が通しで動いた

- **10c の移植の門 ✅**（`f5a63221`・B570・`e2e_wan_umt5_test.ts` 4 passed / 23 steps）: 帯 `UMT5_NORMALIZED_BAND = 79`（決定用 6 本の r は
  L = 8 / 22 / 36 / 163 / 327 / 488 で 8.25 / 5.92 / 2.20 / 4.93 / 0.904 / 15.8・最悪 × 5 を有効数字 2 桁へ切り上げ）。受入れ 4 本の r は
  2.78 / 8.87 / 4.36 / 3.74 で帯の内。故障注入（層 0 / 1 の表の取り違え・`wo.12` の scale × 2）は r 5.0e+5〜9.7e+5 で帯の外。scale × 1.0001
  は r 22.6〜80.2（記録 — 帯の縁で、1e-4 の scale のずれはこの門では区別できない。微妙な故障の床は持たない）。
- **品質の記録（受入れ 4 本・基準 量子化なし CPU f32・行ごとのコサインの最小 / 相対フロベニウス）**: i8 の丸めの影響（GPU i8）=
  boxing-cats 0.982 / 5.0e-2・ferret **0.573** / 7.7e-2・cat-dog-baking 0.993 / 3.4e-2・negative 0.998 / 2.3e-2。bf16 資産の影響 = 0.997 /
  3.1e-2・**0.168** / 7.2e-2・0.768 / 1.3e-1・0.987 / 2.6e-2。GPU i8 と i8 の CPU f32 参照はコサイン 1.000・相対フロベニウス 4e-6〜9e-6
  （GPU の誤差は i8 の丸めの 4 桁下）。i8 の丸めは bf16 資産と同じ桁で、ferret の 1 行が両方で大きく外れる（原因は未調査 — 活性の
  外れ値の疑い・観測のみ）。
- **資源（umT5 単体）**: 確保 5.30〜5.55 GiB（重み 5.30・アリーナは L に比例して最大 0.25）。1 forward の壁時間 0.14 s（L 8）〜1.73 s
  （L 488）、計測モードの GPU 時間 0.064〜1.638 s、1 submit の GPU 時間の最大 155.6 ms（目安 1 s の内）。
- **10d-4 ✅**（`b888ff00`・新ミラーで通し e2e・write と比較の再走が各 6 passed / 15 steps・29 分）: precomputed の経路の sha 行 8 本は
  不変。GPU 経路の sha 行 `gpu-text-2step-boxing-cats-seed42` と `gpu-text-2step-free-fox-seed42` を作った。**text 段の所要 10.4 s**
  （33 フレーム・positive + negative・umT5 5.3 GiB の Session 構築を含む）、transformer 71.4 s（2 ステップ）、VAE 129.3 s、非有限 0。
  **VRAM**: text 段の山 6.30 GiB → 畳んだ後 0.708 GiB（段の前 0.633 — 残り +0.08 GiB）→ DiT 段の山 5.94 GiB → VAE 段の山 3.32 GiB。
  text 段と DiT 段は重ならない（決定 11 の裏付け）。umT5 の Session は submit 52 本・窓平均の最大 30.9 ms。
- **検収表の 10c / 10d の行は ✅**（10d の残り: 視認〈自由プロンプト〉は利用者の素材の選定待ち・0120 段 6 の裁定の後にまとめて）。
- 未解決の更新: 「text → DiT の切り替えの残り VRAM」は +0.08 GiB で閉じる。「Chrome の上限」「段 9 との重なり」は段 9 の実走待ち。

## 追記（2026-10-04）: umT5 の出所は本家へ移った

- 追記「段 10d の設計」D と「段 10d-1 / 10d-2 の結果」が書く「出所は Wan の `text_encoder`・本家 `google/umt5-xxl` との重みの同一は未確認」は、
  [ADR 0122](0122-umt5-upstream-and-compatible-encoders.md) 決定 1 で置き換わった。本家の encoder は Wan 同梱の umT5 と f32 で全要素ビット一致と実測し、
  `karume-umt5-xxl` の上流は本家の commit を名乗る。系列名は `umt5-xxl-i8-dyn`。重みの part と golden のテンソルは変わっていない（ADR 0122 追記「段 a の結果」）。
