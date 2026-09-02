# 0084: Gemma の tokenizer / detokenizer / chat template — compile-to-asset と射程の切り方

- Status: accepted（2026-08-31 — 設計ドラフトの裁定 6 / 9 を★推奨案で確定〈ユーザー裁定〉。
  実装は未着手 = backlog now の段 1a / 段 4）
- Date: 2026-08-31
- 対象: `packages/models/src/text/`（家族中立の共通層）/ `packages/models/src/gemma/text/`
  （新設・ファミリ側）/ `tools/export-recipes/`（compile 台本）。IR 仕様・ランタイム・hub は無改変。
- 関連: ADR [0083](0083-generation-api-surface.md)（生成 API 面 — token イベントの粒度が
  streaming decoder の前提）/ [0085](0085-ple-host-gather.md)（PLE 配布形 — 同一 digest set の
  同居者）/ [0048](0048-irodori-host-port.md)（決定 2 = Unigram 本体は家族中立の共通層
  `src/text/`。追記の教訓 = **前処理 helper を TS と Python で共有すると parity が恒真化する**）/
  [0065](0065-exporter-core-recipe-split.md)（汎用 core と recipe の境界 — compile 台本は recipe 側）/
  [0079](0079-sbv2-two-layer-input.md)（テキスト解析は呼び手の責務・変換関数 1 本という前例）/
  [0038](0038-manifest-v1.md)（`pipelineConfig` の所有）
- 根拠:
  [research/2026-08-31-generation-api-design-draft.md](../research/2026-08-31-generation-api-design-draft.md)
  §0.2・§5・§6（以下「ドラフト §n」— 実資産の実測値と候補比較はそちらが正本）

## Context

`packages/models/src/text/` は現在 encode 側の 4 本（`unigram.ts` / `added-tokens.ts` /
`code-points.ts` / `asset-gates.ts`）だけで、**detokenize がどこにも無い**。gemma4 を
「文字列 in → 文字列 out」で渡すには Gemma の SPM-BPE encode と streaming decode、そして chat
形式が要る。

ドラフト作成時に実資産（`inputs/gemma4/gemma-4-E2B-it/` と `inputs/embeddinggemma/google-300m/`）
を直接読んで確定した事実が、レビューのレンズ記述を 3 点動かした（ドラフト §0.2・§10）:

1. **gemma4 と embeddinggemma は「BPE コアは同一・資産は別物」**。`.model.merges` は **sha256
   ビット同一**（514,906 本）だが、`.model.vocab` は 262,144 本のうち **6,206 スロットで綴りが
   違う**（予約枠の名前 — 例 id 46 = G4 `<|tool>` / EG `<unused40>`）。`added_tokens` は
   **24 本 vs 6,415 本**、post_processor は **G4 = 特殊トークン付与なし / EG = `<bos>` … `<eos>`
   を付与**。→ **実装は共用できるが資産は共用できない**。
2. **Gemma の tokenizer は Unicode 表に依存しない**。normalizer = `Replace(" " → "▁")` 1 本・
   pre_tokenizer = `Split(" ", MergedWithPrevious)` のみ・decoder =
   `Sequence[Replace("▁"→" "), ByteFallback, Fuse]`（両資産で同一構成）。Qwen2 で必要だった
   焼き表 3 種（`code-ranges` / `caseFold` / `nfcSegments`）が **1 つも要らない**。
3. **chat の綴りは Gemma 3 系ではない**。`<|turn>` / `<turn|>` / `<|channel>` / `<|think|>`
   （`tokenizer_config.json` の `sot_token` / `eot_token` / `soc_token` / `think_token` が正本）。
   `<start_of_turn>` は EG 側 vocab にはあるが **G4 vocab には無い**。

## Decision

### 1. tokenizer は compile-to-asset（recipe が実 `tokenizer.json` を検査して独自 schema へ compile）

export recipe（Python）が upstream の `tokenizer.json` を読み、repo 独自の小さな schema へ
compile する。**未知の構成は compile 時に fail loudly**（実行時に散らさない）。

家風どおりである: Qwen2 は「JS の `\p{L}` / `NFC` / `toLowerCase` を正本にしない」ために表を
焼いており（`packages/models/src/anima/text/qwen2-tokenizer.ts`）、Unigram は「正本は Rust の
`Lattice::viterbi` をコードポイント位置で写したもの」（`packages/models/src/text/unigram.ts`）。
**「実装は写す・表は焼く」が既に規約**である。

compile が吐くのは **id 順の vocab 行 / merges rank 表 / byteIds 256 本 / added tokens /
special id 集合**だけの小さな資産になる（Context 2 のとおり Gemma では Unicode 表を焼く必要が
無い）。upstream の `tokenizer.json` は **32.2 MB**（実測）で、生 JSON を配るのは無駄。

MUST: **byteIds は 256 本を明示構築し、欠落・重複を fail loudly で拒否する**。実測では
`<0x00>`…`<0xFF>` が連番（238..493・両資産とも）で `unigram.ts` の `base + byte` 前提は成立するが、
それは**この資産の実測事実であって schema の保証ではない**。compile 時に 256 本を引く費用は
ゼロに近い。

MUST: 配布破損の門は既存の `packages/models/src/text/asset-gates.ts` をそのまま効かせる
（`assertUniqueLines` = 行番号 = id の重複 / `setUnique` = merge 順の後勝ちの禁止。レビュー
CG3-6 の修正でこの門は既に家族横断へ寄せてある）。

### 2. 分割 — 共通処理は `src/text/`・受理 schema と特殊 token はファミリ側

`packages/models/src/text/`（**ファミリ非依存** — ADR 0048 決定 2 と同じ位置づけ理由）:

- BPE merge（rank 表 + tie 規則）/ byte_fallback の UTF-8 再構成 / streaming decoder の状態機械
- **decode も同じ分割**（バイト再構成は共通・特殊 token の扱いはファミリ側）

`packages/models/src/gemma/text/`（新設・ファミリ側）:

- 受理 schema（model type / flags / normalizer / pre_tokenizer / decoder sequence の
  **exact-match**）
- special id 集合・`<bos>` 方針・post_processor 相当（**G4 = 何も付けない / EG = bos…eos** —
  Context 1）

MUST: **全モジュール副作用ゼロ**（byte encoder 等をモジュールスコープで組み立てない — 遅延構築の
前例は `qwen2-tokenizer.ts`）。barrel 経由 tree-shaking の成立条件である。

### 3. BPE は最初から merge queue で書く（O(n²) の全走査 splice を写さない）

Qwen2 の `#bpe` は「全隣接ペアを走査 → 最小 rank を splice」の O(n²) で、Qwen2 は pre_tokenizer が
細かく切るので実害が出ない。だが **Gemma の pre_tokenizer は空白での分割だけ**
（`Split(" ", MergedWithPrevious)`）なので、**空白の無い言語（日本語 / 中国語）や URL・base64 では
1 pre-token が入力全長になる**。gemma4 の golden には日本語ケース（capital-ja, T=10）が既にあり、
長文の日本語入力は実需そのものである。

→ Rust の **merge queue / list 構造まで移植**し、**rank・position の tie と stale pair の無効化**を
fixture で固定する。合格線は「日本語長文で計算量が線形」（段 1a）。

### 4. detokenizer は streaming — UTF-8 の持ち越しを状態機械で持つ

`push(id) → 確定した文字列` の形にする。byte_fallback の途中（不完全な UTF-8 バイト列）は
buffer に持ち越し、確定したぶんだけ返す。

MUST: **byte run 以外を無制限に buffer しない**。`finish()` まで全 token を溜めるだけの
「偽 streaming」を通さないため、この点は直接テストで縛る。

これが ADR 0083 決定 2 の `AsyncIterable` と直結する — 1 token push ごとに確定文字列が出る形で
なければ、`for await` の 1 反復が消費者にとって意味を持たない。

### 5. chat はファミリ固有の純関数 1 本・初版の射程は「素の会話」だけ（裁定 6）

`gemma4ChatPrompt(messages, options) → token id 列`。汎用テンプレートレンダラは作らない
（検討した代替案）。

射程を絞る根拠: `chat_template.jinja` は **386 行 / 18,569 B**（実測）で tool 宣言のフォーマッタ
（`format_parameters` / `format_function_declaration`）・thinking チャネルの順序制御・tool_call
引数の直列化を含むが、**素の会話だけなら綴りは 5 個・分岐は 3 本**である（実測行 = `bos_token` /
`'<|turn>system\n'` / `'<turn|>\n'` / `'<|turn>' + role + '\n'`〈`assistant` → `model` へ写像〉/
生成プロンプト `'<|turn>model\n'`）。前例は ADR 0079 の「テキスト解析は呼び手の責務・変換関数
1 本」で、recipe README も「`<bos>` は `chat_template.jinja` が出す = **ホストの仕事**」と書いている。

**拒否する入力（fail loudly）**: `tools` / `reasoning`（thinking）/ 画像・音声パート /
`role` が `system|user|assistant|developer` 以外。**黙って無視しない** — 無視すると「tool を
渡したのに使われない」が例外なしで通る。射程を広げるのは実需が出てから。

MUST（`<bos>` の所有者）: **`encode(text)` は `<bos>` を付けない**（post_processor が何も付けない
G4 の実測と一致）。**chat 関数だけが付ける**。分けないと chat 導入時に double-BOS になる。

MUST（同一 digest set）: 製品グラフ + weight shards + PLE sidecar（ADR 0085）+ compiled tokenizer +
BOS / special-token ポリシー + prompt / chat format version を**同じ配布 digest set に束ねる**。
chat 形式と EOS 集合（ADR 0083 決定 8）はどちらもこの checkpoint の付随ファイル
（`tokenizer_config.json` / `generation_config.json`）から来るので、**別々に拾うと片方だけ古くなる**。

### 6. EmbeddingGemma の資産 compile を段 1a に同乗させる（裁定 9）

段 1a の tokenizer 実装に EG の資産 compile まで載せる（backlog later「EmbeddingGemma の完成」の
tokenizer 部分の消化）。共用できるのは `src/text/` の**実装**であって**資産ではない**（Context 1）
ので、決定 2 の分割がそのまま「1 実装 2 資産」の実証になる。EG の models pipeline / batch>1
export / runtime attention_mask 配線は本波の射程外で later に残る。

### 7. 検証（parity fixture の採り方）

- MUST: **parity fixture は Python 側で `tokenizers.Tokenizer.from_file()` を独立に呼んで採る**。
  TS と Python が前処理 helper を共有すると parity が恒真化する（ADR 0048 追記の実例 — 共有された
  前処理のせいで caption の欠陥が検出できなかった）。
- 自然文だけでは足りないので**直接叩く**: BPE rank / tie、metaspace 境界、CJK、結合文字、非 BMP、
  AddedToken 隣接、byte 0..255、複数 byte fallback、不正 / 不完全な byte run、special token decode。
- **streaming の門**: 短い token 列について**全 chunk partition**で `push()` の分け方を変え、
  連結が one-shot `decode(ids)` と一致すること + 「byte run 以外を無制限に buffer しない」直接テスト。
- **chat の門**: HF `apply_chat_template` の出力を fixture にした id 列一致。

## 検討した代替案

- **ランタイムが `tokenizer.json` を直接パースする**（compile 台本ゼロ）: **32.2 MB** の JSON を
  配布・パースすることになり、未知構成の扱いが実行時に散る。「JS の Unicode / NFC を正本に
  しない」規律（qwen2）も実装側に散る。却下。
- **既存 `unigram.ts` の拡張**（差分最小）: Unigram（Viterbi 格子）と BPE（merge rank）は別の
  アルゴリズムで、同じファイルに置く理由が無い。却下。
- **汎用テンプレートレンダラ（Jinja 風）**: 「ランタイム依存は Web 標準のみ」（CLAUDE.md 横断
  不変条件）と「投機的な一般化をしない」の両方に反する。却下（採るなら再裁定要）。
- **chat を完全に呼び手任せにする**（消費者が `<|turn>` を手書き）: 綴りが Gemma 3 系と違う
  （Context 3）ことを消費者が踏み抜く形で、EOS 集合との同期も呼び手に漏れる。却下。

## Consequences

- 段 1a は **GPU 不要で単体テストに閉じる**ので、段 1b（実 GPU）と**並行レーン**で走らせられる
  （backlog now の実行計画）。ADR 0083 のイベント形が確定していること（段 0）だけが前提で、
  段 3 の実装完了は前提にならない。
- `packages/models/src/gemma/` が新しいファミリディレクトリとして立つ（tokenizer と chat が先・
  pipeline は段 4）。
- EG 資産の compile が同乗するぶん段 1a は重くなるが、受理 schema をファミリ側へ置いた分割の
  実証（1 実装 2 資産）になる。
- compiled tokenizer と chat format version は配布 digest set の一員になる（決定 5）— dist /
  モデルカードへの反映は段 5。
- レンズ見積りの訂正 2 点が本 ADR に織り込まれている: encode 側の実装コストは焼き表が要らない
  ぶん**小さい**が、BPE の探索構造は**最初から merge queue が要る**（決定 3）。

## 追記

- 2026-08-31（公開面レビューの消化 — 裁定の原文 = `.claude/reviews/2026-08-31_182ced7/`）:
  - **`gemma4ChatTurn(tokenizer, message)` を足した**（決定 5 の chat 関数 1 本に対する **差分
    描画**の対）。`gemma4ChatPrompt` は毎ターン会話全体を描くので `GenerationSequence` で多ターンを
    回すと O(n²) の prefill になり、実走した消費者は「全体を描いて先頭の `<bos>` を剥がす」当て
    推量を書いていた（温度 0 で `chat()` と逐語一致したが**無保証**）。turn-local 契約を関数と
    doc で正本化する。MUST: **前 turn を閉じる `<turn|>` は含めない** — その綴りは生成が出して
    sequence が未 commit の frontier（`pendingToken`）として持ち、次の `generate` が prompt の
    先頭へ自動連結する（ADR 0083 決定 4）。二重に描くと turn の区切りが 2 つになり、例外は
    1 つも出ない。成立する等式 `gemma4ChatPrompt(全会話) = gemma4ChatPrompt(先頭 turn まで) ⧺
    生成本文 ⧺ [<turn|>] ⧺ gemma4ChatTurn(次の発話) ⧺ …` を任意の分割で見る門を置く。
    MUST: `assistant` は**拒否**する（model turn は生成が埋める席で、上流 template も連続
    assistant を 1 つの turn へ畳む — 差し込むなら `gemma4ChatPrompt` で全体を描き直す）。
    使えるのは生成が `<turn|>` で閉じた直後だけで、max-tokens / `break` で打ち切ったターンの
    続きは増分ではなく続きの生成（`prompt: []`）が正である。
  - **`parseGemma4PipelineConfig` を公開した**（決定 5 の「同一 digest set」を消費者側でも保つ）。
    `fromPretrained` は内部で通すので呼ぶ必要は無く、要るのは `karume.json` を自分で読んで
    `fromAssets` へ渡す側である — 手元にあるのは hub の `ModelEntry.pipelineConfig`
    （`Record<string, unknown>`）なので、この口が無いと `as` で被せるか 3 つの数を配布形と自分の
    コードに二重持ちするしかない（どちらも配布形が動いたときに黙って食い違う）。あわせて
    `fromAssets` も `fromPretrained` と**同じ門**（未知キー・値域・`chunkLength ≤ capacity ≤
    maxPosition`）をバイト列を開く前に通す。
  - **`Gemma4Pipeline.sampler` → `defaultSampler` へ改名**し、`Gemma4PipelineConfig.sampler` の型を
    `SamplerSpec` から `Gemma4DefaultSampler`（`temperature` / `topK` / `topP` の **3 欄必須**）へ
    縮小した（ADR 0083 決定 7 の「既定値は配布形が宣言する」の綴り直し）。旧名は「今この生成が
    使っている sampler」と読めるが、実際には要求が省略したときだけ使われる**既定**である。
    型を `SamplerSpec` のままにすると受理集合より広い型になり（`logitBias` /
    `seed` / `repetitionPenalty` は配布者が推奨する性質の値ではない）、「型は通るのに
    パーサが未知キーで落とす」欄が公開面に生える。部分宣言を許さないのは「温度だけ推奨・top-k は
    低層の既定」という**上流のどこにも無い**組み合わせを作らないため。
  - **破壊的変更（未リリース面）**: 上の改名・`config.sampler` の型縮小・`fromAssets` が不正宣言を
    ロード前に拒否するようになったこと。GPU golden は全て不変。
