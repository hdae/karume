> **性格**: 時点スナップショット（2026-08-31・生成 API 波の設計ドラフト逐語退避）。裁定 10 点は
> すべて★推奨案で承認済み（2026-08-31 ユーザー裁定）。正本化された決定は docs/decisions/ の
> 該当 ADR が正本 — 本書は候補比較・棄却理由・導出の記録として残す。

# 生成 API 波 — 設計案ドラフト（裁定用）

> **性格**: 設計フェーズのドラフト 1 本。コード変更なし・裁定なし。各軸で「候補 → トレードオフ →
> 推奨 1 案 + 根拠」を出し、最後に波の分割案とユーザー裁定が要る点を番号付きで並べる。
> **日付**: 2026-08-30 / **対象 HEAD**: `cc487db`（読み取り時点・ツリークリーン）
>
> 引用は全て実コード / 実資産で再確認済み（`file:line`）。**実測でない数字は「見積り」と明記**する。
> 既存裁定（sampling ホスト維持 / 二重簿記の禁止 / fail loudly / モジュール副作用ゼロ /
> ランタイム依存は Web 標準のみ）と衝突する案には §9 で印を付けた。

---

## 0. 入力と、この波の射程

読んだもの: `CLAUDE.md` / `.claude/ACTIVE_DESIGN.md` / `docs/backlog.md`（now 節「生成 API 波 =
設計フェーズとして起票」`docs/backlog.md:41-45` と later 節「生成 API 波（起票 2026-08-19）」
`docs/backlog.md:229-233`）/ `.claude/reviews/2026-08-29_9614ba9/findings/lens-llm.md`（§3 ギャップ
台帳・§4 L-2〜L-6/L-9）/ 同 `findings/CG5-verify.md`（検証記録）と**その被検証元**
`.claude/reviews/2026-08-29_chatgpt-reviews-2/5.md`（GenerationSequence / PLE sidecar token-major /
topk k63 製品グラフ / tokenizer compile-to-asset の**設計提案そのもの**はこちらにある — findings/
配下には検証記録しか無い）/ ADR 0065・0066・0067・0068 / 現行実装 5 面 /
`docs/research/2026-08-30-gemma4-decode-wallclock.md`。

**射程**: LLM（gemma4 E2B — L-11 裁定済み）を「文字列 in → 文字列 out」で消費者に渡せる面まで。
性能改善そのもの（K-11 = decode の `wi4g32` M=1 変種）は**別トラック**で、本波は
`≈85ms/token`（実測 — research §2）を所与として設計する。公開はライセンス門
（`docs/decisions/0065-exporter-core-recipe-split.md:47-50` 決定 7「upstream revision 単位の
ライセンス互換確認〈人間の interview〉はリリース gate」— gemma4 README と backlog はこれを
「ADR 0065 stage 6」と呼ぶ）後で、技術側の完成と独立に止まりうる。

### 0.1 現状の確定事実（否定の証拠つき）

| 事実                                                                               | 根拠                                                                                                                                               |
| ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 消費者が触れる面は `generateGreedy` 1 本（固定長・greedy・停止なし）               | `packages/models/mod.ts:140-141`（barrel の export はこの 2 本のみ）・`packages/models/deno.json:5-14`（サブパスは 7 ファミリで gemma / llm 無し） |
| sampling はホストにも無い                                                          | `packages/models/src/generation/greedy.ts:11-14`（「温度も top-k も RNG も置かない」MUST）                                                         |
| 停止条件が無い                                                                     | `greedy.ts:193-194`（「**EOS 停止は載せない**」NOTE）                                                                                              |
| 多ターンの面が無い                                                                 | `greedy.ts:222-225` で毎回 context を作り `:254-258` の finally で dispose                                                                         |
| detokenize がどこにも無い                                                          | `packages/models/src/text/` は `unigram.ts` / `added-tokens.ts` / `code-points.ts` / `asset-gates.ts` の 4 本（全て encode 側）                    |
| **同一 context への並行 run は既に runtime が拒否する**（CG5-2 は修正波で閉じた）  | `packages/runtime/src/runtime/generation-context.ts:337-344`（`#runs > 0` で `ExecutionError`）                                                    |
| 全 initializer は Session 構築時に GPU 常駐席を持つ（PLE 35 表は lazy にならない） | `packages/runtime/src/runtime/executor.ts:958-963`（「席はプランナが正本 — **全 initializer を載せる契約**」）                                     |
| decode は 1 token ≈85ms・GPU 86.2ms 中 `wi4g32` が 73.3ms                          | `docs/research/2026-08-30-gemma4-decode-wallclock.md` §2・§3（実測・単一リグ各 1 走）                                                              |

### 0.2 本ドラフト作成中に**実資産を読んで確定した**事実（レンズの記述を精密化するもの）

実資産 `inputs/gemma4/gemma-4-E2B-it/` と `inputs/embeddinggemma/google-300m/` を直接読んだ結果:

1. **gemma4 と embeddinggemma の tokenizer は「BPE コアは同一・資産は別物」**。
   `.model.merges` は **sha256 ビット同一**（514,906 本）だが `.model.vocab` は 262,144 本のうち
   **6,206 スロットで綴りが違う**（予約枠の名前 — 例 id 46 = G4 `<|tool>` / EG `<unused40>`）。
   `added_tokens` は **24 本 vs 6,415 本**、post_processor は **G4 = 特殊トークン付与なし /
   EG = `<bos>` … `<eos>` を付与**。→ lens-llm §1.4 の「この tokenizer は gemma4 とほぼ同じもの」は
   **実装は共用できるが資産は共用できない**と読み替える必要がある。
2. **Gemma の tokenizer は Unicode 表に依存しない**: normalizer = `Replace(" " → "▁")` 1 本・
   pre_tokenizer = `Split(" ", MergedWithPrevious)` のみ・decoder =
   `Sequence[Replace("▁"→" "), ByteFallback, Fuse]`。**両資産で同一構成**。
   → Qwen2 で必要だった焼き表 3 種（`code-ranges` / `caseFold` / `nfcSegments` —
   `packages/models/src/anima/text/qwen2-tokenizer.ts:12-23,38-45`）が **Gemma では 1 つも要らない**。
   L-5 の「実装コスト 大」はこのぶん下がる。
3. **byte_fallback の id は連番**（`<0x00>` = 238 … `<0xFF>` = 493 — 両資産とも）。
   `packages/models/src/text/unigram.ts:30-36` が置いている `base + byte` 前提は**この 2 資産では
   実測で成立**する（ただし schema 保証ではない — §6 の門の話）。
4. **added_tokens のフラグは全て `single_word/lstrip/rstrip/normalized = false`**、違うのは
   `special` だけ（G4 = 24 本すべて special / EG = 6,406 非 special + 9 special）。
   → 既存 `splitAddedTokens`（leftmost-longest）の形で足りる。
5. **chat の綴りは Gemma 3 系ではない**: `<|turn>` / `<turn|>` / `<|channel>` / `<|think|>`
   （`tokenizer_config.json` の `sot_token` / `eot_token` / `soc_token` / `think_token`）。
   lens-llm G3 の「`<bos>` + `<start_of_turn>` 系の逐語移植」は **Gemma 3 の綴りで、この
   checkpoint には当たらない**（`<start_of_turn>` は EG 側 vocab にはあるが G4 vocab には無い）。
6. **`chat_template.jinja` は 386 行 / 18,569 B**（tool 定義フォーマッタ・thinking チャネル・
   tool_call 引数直列化を含む）。素の会話だけなら小さい（§7）。
7. **停止は集合**: `generation_config.json` の `eos_token_id = [1, 106, 50]` =
   `<eos>` / `<turn|>` / `<|tool_response>`。
8. **`generation_config.json` の推奨既定は `top_k: 64`** — karume の topk 実装上限
   **k ≤ 63**（ADR 0068 追記 2 の `8·W·(k+1) ≤ 16384`・W=32 → ちょうど 16384）を **1 だけ超える**。
   軸③の決定打（§4）。

---

## 1. 軸① 生成 API の形（GenerationProgram + stateful sequence / token イベント / EOS / cancel / 多ターン）

### 1.1 出発点

backlog later 節の起票（`docs/backlog.md:229-233`）が既に方向を書いている: 静的配線とリクエストを
分離した `GenerationProgram`（setup 時に全結線を検証）+ stateful sequence API（`for await` の
token イベント・EOS 停止・cancel・多ターン継続）+ `last_row` の runner 側導出 + `generateGreedy`
の内部ヘルパ格下げ。5.md §1 の設計提案は「`GenerationContext` を製品面に露出させず
`GenerationSequence` を 1 枚置き、可変状態は `context` と `pendingToken` **だけ**」。

現行の `GreedySpec`（`greedy.ts:50-83`）は**静的配線とリクエストが 1 つの型に同居**している —
静的側 = `session` / `inputIds` / `positionIds` / `token` / `lastRow` / `chunkLength` /
`maxPosition` / `bindings`、リクエスト側 = `prompt` / `maxNewTokens`。分離の起票はここを指している。

### 1.2 候補

| 候補                                                                                 | 形                                                                                                                                                                     | 長所                                                                                                                                                            | 短所                                                                                                                                                                                       |
| ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **a) `GenerationProgram` + `GenerationSequence` + `AsyncIterable<GenerationEvent>`** | program = 静的配線の不変オブジェクト（setup 時検証）。sequence = 1 会話ぶんの寿命（`context` + `pendingToken`）。`sequence.generate(request)` が async iterable を返す | 起票と 5.md の両方に一致。token 列は「値の列」なので iterable が素直。`break` / `return()` で自然に閉じる。多ターンが sequence の寿命に乗る                     | リポ内に async generator を公開面に出した前例が無い（家風は onEvent）。放置された generator が直列化鎖の席を握る危険                                                                       |
| **b) `GenerationSequence` + anima 流 `onEvent` コールバック**                        | `sequence.generate(request, { onEvent })` が最終結果を返し、token は callback                                                                                          | 家風に完全一致（`packages/models/src/anima/pipeline.ts:145-163` — await する / 例外を握らない / **throw が step 粒度の中断手段**）。放置 generator の危険が無い | token 列という「主たる返り値」を callback に押し込む形。streaming 消費者は自分でキューを組む                                                                                               |
| **c) 関数 1 本の拡張（`generateStream(spec)`）— sequence 型を作らない**              | `generateGreedy` を async generator へ拡張し、多ターンは呼び手が context を持ち回す                                                                                    | 追加型ゼロ・最小                                                                                                                                                | `GenerationContext` を製品面に露出する = 5.md が名指しで避けた形（`pendingToken` の管理が呼び手に漏れ、「直前 assistant の最後の token が履歴から 1 個落ちる」事故を消費者側で再生産する） |

### 1.3 推奨: **a**（`AsyncIterable` を主面にし、cancel は `AbortSignal` を正の手段にする）

根拠:

1. **起票と外部レビューが独立に同じ形を指している**。backlog later 節（`docs/backlog.md:229-233`）
   と 5.md §1 は別の出所だが、`GenerationProgram` / `GenerationSequence` / `pendingToken` /
   「position counter を持たない」まで一致している。片方だけの提案ではない。
2. **token 列は進捗ではなく値**。anima の `onEvent` は「返り値が画像 1 枚で、途中経過が副次」
   （`anima/pipeline.ts:183-198` の event union は `stage` / `denoise-step` / `vae-tile` = 全部進捗）。
   LLM は**列そのものが返り値**なので、家風の理由がここでは反転する。`for await` にすると
   detokenize の部分 UTF-8 持ち越し（§6）が「push すると確定した文字列だけ出る」形と直結する。
3. **二重簿記の禁止に構造的に従える**（ADR 0066 決定 6・`greedy.ts:6-9`）。sequence は
   position / length counter を持たず、run を組む直前に `context.pastLength`
   （`generation-context.ts:481-484`）を読んで `position_ids` を作る。`totalLength` が要るなら
   `context.pastLength + (pendingToken ? 1 : 0)` をその場で導出して**保存しない**（5.md §1）。
4. **runtime 側の single-flight は既に在る**ので、sequence が自前でロックを作らなくてよい
   （`generation-context.ts:337-344` — 2 本目の発行は `ExecutionError`）。sequence 側は既存の
   `createOperationChain`（`packages/models/src/concurrency/serial.ts:24-32`）で「generate 1 回ぶん」
   を直列化すれば足り、他 3 家族と同型になる。
5. **cancel の前例が既にある**: `AnimaPipelineOptions.signal`（`anima/pipeline.ts:246-255` —
   段の境目で検査・`signal.reason` を包まずそのまま throw）。生成ループへの AbortSignal 席は
   backlog later（`docs/backlog.md:270-274`）に「需要待ち」で起票済みで、**LLM がその需要**。

### 1.4 推奨形の骨格（型は仮）

```ts
// 静的配線（setup 時に全結線を検証する不変オブジェクト）
type GenerationProgram = {
  readonly inputIds: string;
  readonly positionIds: string;
  readonly lastRow: string;
  readonly logits: string; // ← 最終行 logits 出口（§4）
  readonly chunkLength: number;
  readonly maxPosition: number;
  readonly vocabSize: number;
  readonly stopTokens: readonly number[]; // EOS 集合（§7）
  readonly bindings?: SymbolBindings;
};

type GenerationRequest = {
  readonly prompt: readonly number[]; // 多ターンでは「今ターンぶん」だけ
  readonly maxNewTokens: number;
  readonly sampler?: SamplerSpec; // 既定 = greedy（§4）
  readonly signal?: AbortSignal;
};

type GenerationEvent =
  | { readonly kind: "token"; readonly id: number; readonly position: number }
  | { readonly kind: "prefill"; readonly chunk: number; readonly chunks: number };

interface GenerationSequence {
  generate(request: GenerationRequest): AsyncIterable<GenerationEvent> & {
    /** 停止理由（`eos` / `max-tokens` / `aborted`）— iterable を汲み切った後に読む。 */
    readonly done: Promise<GenerationStop>;
  };
  dispose(): Promise<void>;
}
```

**多ターンの契約**（5.md §1 の核）: `GenerationContext` は常に**最大 1 token の未 commit
frontier**を持つ（CG5-1 で holds 追認 — `findings/CG5-verify.md:11-37`。`generateGreedy` は
`greedy.ts:245-252` で `maxNewTokens - 1` 回しか decode しないので、K token 生成後の
`pastLength = T + K − 1`）。したがって次ターンは **`pendingToken` を新 user turn の token 列の
先頭に連結して prefill** する — 余分な「commit だけの run」を増やさずに frontier を KV へ収める。

**注意（設計時に潰す穴）**:

- `break` で中断された iterable は `return()` 経由で `finally` に入る。そこで**論理長は既に進んで
  いる**（run が成功した ぶん）ので、`pendingToken` を正しく残さないと次ターンで 1 token 落ちる。
  `for await` を採る以上、この経路の門は必須。
- `rewind` は使えない（sliding スロットを含む context は**全拒否** —
  `generation-context.ts:494-497` / ADR 0066 追記 2）。編集・分岐は「新 context + token
  transcript の replay」が正（5.md §1 も同結論）。
- `enqueue` の generation 面は無い（`docs/limitations.md:806-811` — 裁定済み）。sequence は
  `run` を 1 本ずつ await する形しか採れない。
- 満杯（full スロットの `P + Q ≤ C`）は今日「汎用メッセージで fail loudly」。sequence は
  **専用の型**（`GenerationCapacityError` 相当）で落とし、「会話の切り詰めはホストの責務」を
  limitations に明文化する（lens-llm L-9 b）。

---

## 2. 軸② PLE 35 表のホスト gather 席（lens-llm L-2）

### 2.1 事実

- PLE = `input_ids` **だけ**を引数に取る純粋な行 lookup。recipe に切断点が既にある —
  `tools/export-recipes/gemma4/ple.py` の `per_layer_inputs(tables, input_ids, scale)` が
  `[1,M,35,256]` を組み、台本はそれを `per_layer_inputs=` として上流へ渡すだけ
  （`export_decode.py` / `export_token.py` の両方が同じ 1 本を通す — `ple.py` モジュール docstring）。
- 常駐は **i8 35 表 × 64 MiB = 2,240 MiB**（容器ヘッダ実測 — lens-llm §1.2）。容器全体 3.70 GiB の
  **59%**。外に出せば **3,787 MiB → 1,547 MiB**。
- **速度には効かない**: decode の GPU 86.2ms のうち `linear` 系が 78.7ms で、embedding 35 本は
  「残り ≈935 本 ≈4.5ms」の内数（research §3）。これは**純粋に常駐の話**。
- 逆流するコスト: `per_layer_inputs[1,M,35,256]` f32 のアップロードが増える —
  decode で 35,840 B/token、prefill chunk 32 で 1,146,880 B/chunk。85ms / 162ms の壁に対して無視できる。

### 2.2 候補

| 候補                            | 中身                                                        | 長所                   | 短所                                                                                                                                                                                                                                                                                                                                            |
| ------------------------------- | ----------------------------------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **a) ホスト RAM 常駐**          | 2.19 GiB の i8 表をホストに持ち、毎 step 行を引いて逆量子化 | 単純・I/O ゼロ         | **単一 ArrayBuffer では原理的に不可**（下記） / ブラウザで JS heap 2.19 GiB は現実的でない                                                                                                                                                                                                                                                      |
| **b) キャッシュから行だけ読む** | hub のキャッシュに部分読み席を新設し、token ごとに 1 行読む | RAM も VRAM も食わない | **hub に部分読みの席が今日無い**（`packages/hub/mod.ts` の公開面 = manifest / resolve / fetch / stream / prefetch / clear のみ・最小単位はファイル 1 本 = `packages/hub/src/fetch.ts:591-599` の `StreamedAsset {id, bytes}`）。Range 並列は perf L-3 で **parked**（`docs/backlog.md:326`）。ADR 0038 のキャッシュ設計（キーは URL）に踏み込む |
| **c) 現状維持 + PLE を i4 化**  | 2,240 → 1,120 MiB                                           | recipe だけで閉じる    | recipe README が「embeddings are int8 … not int4-eligible」と明記。品質リスクを token 列 parity で潰す必要（潰せるのは利点）                                                                                                                                                                                                                    |

### 2.3 レンズの評価を 1 点覆す（重要）

lens-llm L-2 は「a) ホスト RAM 常駐 — **単純だが**ブラウザでは重い」と書いているが、
**a の素直な形（1 本の表）は Chromium で原理的に不可**である:

- PLE i8 全量 = 262,144 × 8,960 = **2,351,662,080 B**
- Chromium の単一 ArrayBuffer 上限 = **2,145,386,496 B**（`docs/limitations.md` の恒久記載 —
  Base f16 がロード不能だったのと同じ天井）

→ **分割は a でも b でも必須要件**であり、「a は単純」という評価はここで崩れる。
35 表に割れば 1 表 = 67,108,864 B で天井は回避できるが、それは **table-major のまま**で、
5.md §3 が名指しで避けた形（「1 token の PLE を引くために 35 個の離れた asset location を読む」）。

### 2.4 推奨: **配布形を token-major + vocab レンジ shard に固定し、初版のホスト側は「触った shard だけ遅延ロード + LRU」（a と b の中間）**

根拠:

1. **配布形は a でも b でも同じ**（`[token][layer][256] i8` + `[token][layer] scale` の
   token-major・vocab 範囲で shard — 5.md §3）。先に固定して損が無く、後から b へ移るときに
   **再 export も HF 再アップも要らない**（ホスト側の差し替えだけ）。lens-llm の
   「b を目標に a で先に成立させる」を、配布形の観点で具体化した形。
2. **token-major なら 1 token の PLE は連続 1 読み**（8,960 B + 35 scale）。table-major のままだと
   35 箇所の離散読みで、b へ移った瞬間に I/O が 35 倍になる。
3. **hub の部分読み席の新設は独立の設計判断**で、この波に抱き込むと射程が膨らむ
   （§2.2 の b 欄）。遅延 shard ロードなら hub は今日の `streamAssets` / `prefetchAssets`
   （`packages/hub/src/fetch.ts:792-801`）のままで済む。
4. **速度には効かないので、常駐削減の効き幅だけで採否を測れる**（研究 §3）— 数値の議論が単純になる。

**未実測（speculation とラベルする）**: 「実会話が触る token id が vocab のどの範囲に集中するか」は
測っていない。SentencePiece 語彙が頻度順に並ぶという一般論はあるが、この checkpoint では確認して
いない。**shard 幅は golden 3 ケース + chat コーパスで実測してから決める**。

### 2.5 設計時に潰す穴（MUST 候補）

- **ホスト gather の逆量子化は、GPU 側 `embedding`（i8 格納）の演算とビット一致させる MUST**。
  さもないと token 列 parity が割れ、「機能不変であること」の証明（ADR 0066 追記 9 で sliding
  容量を変えたときに使った手）が使えなくなる。`per_layer_scale` = `256 ** 0.5` = **16.0**
  （2 冪なので f32 の乗算が厳密 — `ple.py` の `per_layer_scale` docstring が同じ理由でビット一致
  検査を成立させている）ので、順序さえ揃えれば成立する見込み。
- **loader で相互照合する**（5.md §4 末尾）: tokenizer が生成し得る id / 主 embedding の vocab 行数 /
  PLE sidecar の行数 / special id。ここがずれると OOB ではなく「**別 token の有効な行**」を引く。

---

## 3. 軸③ 最終行 logits 出口 + sampling ホスト維持（lens-llm L-3・ADR 0068 との整合）

### 3.1 候補

| 候補                                                             | 出口                       | 読み戻し    | 表現できる sampler                                                                          |
| ---------------------------------------------------------------- | -------------------------- | ----------- | ------------------------------------------------------------------------------------------- |
| **a) 最終行 logits**（`export_token.py` から argmax を外すだけ） | `logits[1,1,262144]`       | 1 MiB/token | 温度 / top-k（**上限なし**） / top-p（full-vocab） / repetition penalty / logit bias / seed |
| **b) topk 製品グラフ**（argmax → topk・k 固定）                  | `values[k]` + `indices[k]` | 512 B 級    | 「top-k で絞る → その中で温度 / top-p」だけ。**k ≤ 63**                                     |
| **c) 現状維持**（token-only argmax）                             | `token[1,1,1]`             | 4 B         | greedy のみ                                                                                 |

### 3.2 推奨: **a**（最終行 logits を製品グラフの出口にし、argmax 系列は検収 fixture として残す）

根拠:

1. **ADR 0068 が席を既に開けている**。決定 4 は「readback は sampled token または topk の 2 本を
   **既定**にする。全語彙 logits の readback は**グラフ出力に logits を宣言した場合の opt-in
   として残す（欄を消さない）**」（`0068-decode-exit-multi-output.md:77-79`）。却下したのは
   **GPU 側 full sampling** であって、ホストへ全 logits を返す形ではない（同 `:85-87`）。
   → **再裁定不要**。ただし「当面の既定は最終行 logits」を追記すべき（lens-llm L-3 の設計判断欄と同意見）。
2. **b は今日通れない**。exporter の停止点は handler ではなく **`operator.getitem`**（実測 —
   ADR 0068 追記 3 `:118-121`・`docs/op-vocabulary.md` の topk 節「ハンドラだけ足しても道は開かない」）。
3. **b は k ≤ 63 で、モデル自身の推奨既定 `top_k: 64` を表現できない**（§0.2-8 の実測 —
   `generation_config.json` の `top_k` = 64 に対し実装上限は ADR 0068 追記 2 の
   `8·W·(k+1) ≤ 16384`・W=32 → k ≤ 63 ちょうど）。**1 だけ足りない**という形なので、
   「実用上は 63 で十分」と言い張るのが難しい。
   補強: **ADR 0068 自身の why-not 文が「topk k 本の readback（`k ≤ 64` で 512B 級）」と
   書いている**（`:87`）— 却下の理由づけが k=64 を前提にしていたのに、実装は追記 2 で 63 に
   着地した。この 1 の差が、そのままモデル既定と噛み合わない。
4. **repetition penalty / logit bias / full-vocab nucleus は全語彙が要る**（top-63 の外の softmax
   質量をホストが知らない — 5.md §3）。chat 用途でこれらを持たない選択は現実的でない。
5. **コストが壁に対して無視できる**: 1 MiB の readback は同じ submit に相乗りするのでフェンスは
   増えず、262,144 要素の JS 走査を含めても **0.3〜0.6ms 級の見積り**。実測の壁 85ms/token
   （research §2）に対して **1% 未満**。K-11 でカーネルが直り切ってフェンス床 ≈11ms が天井に
   戻っても（research §5）、**数%**。
6. **prefill の読み戻しがむしろ減る**: 現行 logits opt-in 系列は `[1,M,262144]` を返すので
   chunk 32 で 32 MiB。最終行だけなら **1 MiB**。`last_row` による行選択 → 1 行 lm_head の配線は
   `export_token.py` の `TokenOnlyChunkWrapper.forward` に既にあり、**argmax を外すだけ**。

### 3.3 sampler の置き場と契約

- 置き場: `packages/models/src/generation/sampler.ts`（`greedy.ts` と同じ「パイプライン非依存の
  共通処理」— `greedy.ts:2-4` の位置づけ理由がそのまま当てはまる）。
- RNG: `packages/models/src/anima/random.ts` の splitmix64 の流儀を再利用。
  **MUST（前例そのまま）**: その doc が「torch の `randn` とは別物・同じ seed でも同じ列にならない」
  と書いている（`anima/random.ts:7-13`）のと同じ理由で、**HF の sampling 出力との token 列 parity は
  取れない**。parity 門は温度 0（= greedy）で採り、sampler 自体は自前 fixture で縛る。
- 既定: greedy（`sampler` 省略時）。`generateGreedy` の parity 門はこの経路で生き続ける。

---

## 4. 軸④ 最小縦割りの順序（lens-llm L-4）

### 4.1 候補

| 候補                        | 順序                                                                       | 長所                 | 短所                                                                                               |
| --------------------------- | -------------------------------------------------------------------------- | -------------------- | -------------------------------------------------------------------------------------------------- |
| **a) 面ごとの横割り**       | sequence API → sampling → tokenizer → chat → 配布                          | 各面が完成形で閉じる | 「文字列 in → 文字列 out」が最後まで存在しない = 設計の誤りが最後に出る                            |
| **b) 最短縦割り**           | 「gemma4 を Deno で 1 本、文字列 in → 文字列 out」を最短で通してから広げる | 早期に全結線が繋がる | tokenizer（大きな塊）が critical path に載る                                                       |
| **c) 配布経路先行**（L-11） | minicpm5 で dist を先に通す                                                | ライセンス門と独立   | L-11 は **gemma4 E2B に裁定済み**（`docs/backlog.md:46-48`）— tokenizer の対象が変わる話は既に決着 |

### 4.2 推奨: **b の変種 — 「イベント形の確定」を先頭に置き、tokenizer レーンを GPU レーンと並行に走らせる**

根拠:

1. **tokenizer は GPU 不要で単体テストに閉じる**（lens-llm L-5「依存: 無し」）。純関数 + fixture
   なので、GPU を使うレッグと**同時に走らせられる最大の塊**。逐次に置くと波が倍の長さになる。
2. **streaming detokenize の設計は token イベントの粒度に依存する**（部分 UTF-8 の持ち越しは
   「1 token push ごとに確定した文字列を返す」形が前提 — 5.md §4）。だから **L-4 が L-5 の前提**
   というレンズの依存関係は正しいが、必要なのは**イベント形の確定**だけで、実装完了ではない。
   → 「段 0 = 型と契約だけ確定（コード 0 行）」を切れば依存が解ける。
3. **停止は sampling 結果に対して行う**（lens-llm L-4 の依存欄）ので、EOS 集合と sampler は
   同じ段に置く。EOS 集合は chat 資産と同じ出所（`generation_config.json`）なので、
   chat 段の前に**集合の受け口だけ**を program に置く。
4. **配布とライセンスは最後**（ADR 0065 stage 6 = 人手の裁定）。技術側の完成を止めないため。

---

## 5. 軸⑤ Gemma SPM-BPE tokenizer + detokenizer（lens-llm L-5）

### 5.1 候補

| 候補                                              | 中身                                                                                                                 | 長所                                                                                 | 短所                                                                                                                                  |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| **a) compile-to-asset**（5.md §4）                | export recipe が実 `tokenizer.json` を読み、repo 独自の小さな schema へ compile。未知構成は compile 時に fail loudly | 配布物が小さい / 受理集合が機械可読 / 既存の資産門（`asset-gates.ts`）がそのまま効く | compile 台本（Python）が 1 本増える                                                                                                   |
| **b) ランタイムが `tokenizer.json` を直接パース** | JSON interpreter                                                                                                     | 台本ゼロ                                                                             | **32.2 MB** の JSON を配布・パースする。未知構成の扱いが実行時に散る。JS の Unicode / NFC を正本にしない規律（qwen2）を実装側に散らす |
| **c) 既存 unigram 経路の拡張**                    | `src/text/unigram.ts` に BPE を足す                                                                                  | 差分最小                                                                             | Unigram（Viterbi 格子）と BPE（merge rank）は別アルゴリズム。同じファイルに置く理由が無い                                             |

### 5.2 推奨: **a（compile-to-asset）**

根拠:

1. **家風に一致**。Qwen2 は「JS の `\p{L}` / `NFC` / `toLowerCase` を正本にしない」ために表を焼いて
   いる（`qwen2-tokenizer.ts:12-23,38-45`）。Unigram も「正本は Rust の `Lattice::viterbi` を
   コードポイント位置で写したもの」（`unigram.ts:9-12`）。**「実装は写す・表は焼く」が既に規約**。
2. **Gemma は幸運にも表を焼く必要が無い**（§0.2-2 の実測）: normalizer が `Replace(" " → "▁")`・
   pre_tokenizer が `Split(" ", MergedWithPrevious)` だけで、Unicode 分類にも NFC にも触らない。
   → compile が吐くのは「id 順の vocab 行 / merges rank 表 / byteIds 256 本 / added tokens /
   special id 集合」だけの**小さな資産**になる。
3. **既存の資産門がそのまま効く**: `packages/models/src/text/asset-gates.ts:59-78`
   （`assertUniqueLines` — 行番号 = id の重複は「例外にならない配布破損」）と `:80-93`
   （`setUnique` — merge 順の後勝ちを禁じる）。CG3-6（tokenizer 重複 token / merge の last-wins）の
   修正でこの門は既に家族横断へ寄せてある。
4. **配布サイズ**: `tokenizer.json` は 32.2 MB（実測）。merges 514,906 本 + vocab 262,144 本の
   生 JSON を配るのは無駄で、compile 済みなら数 MB 級（見積り）。

### 5.3 置き場と分割

`packages/models/src/text/`（**ファミリ非依存** — `unigram.ts:1-16` の位置づけ理由と同じ）に:

- BPE merge（rank 表 + tie 規則）/ byte_fallback の UTF-8 再構成 / streaming decoder の状態機械
- **decode も同じ分割**（バイト再構成は共通・特殊 token の扱いはファミリ側）

ファミリ側（`packages/models/src/gemma/text/` 新設）に:

- 受理 schema（model type / flags / normalizer / pre_tokenizer / decoder sequence の exact-match）
- special id 集合・`<bos>` 方針・post_processor 相当（**G4 = 何も付けない / EG = bos…eos** — §0.2-1）

**embeddinggemma との共用の正確な範囲**: 共用できるのは `src/text/` の**実装**であって、
**資産ではない**（merges はビット同一だが vocab は 6,206 スロット差・added_tokens は 24 vs 6,415・
post_processor が別 — §0.2-1）。backlog later の「EmbeddingGemma の完成」
（`docs/backlog.md:253-254`）はこの実装に乗るが、**資産は別に compile する**。

### 5.4 実測から出た設計要件 2 つ

1. **BPE は最初から merge queue で書く（O(n²) の全走査 splice を写さない）**。
   Qwen2 の `#bpe`（`qwen2-tokenizer.ts:233-248`）は「全隣接ペアを走査 → 最小 rank を splice」の
   O(n²)。Qwen2 は pre_tokenizer が細かく切るので実害が出ないが、**Gemma の pre_tokenizer は
   空白での分割だけ**（`Split(" ", MergedWithPrevious)`）なので、**空白の無い言語（日本語 /
   中国語）や URL・base64 では 1 pre-token が入力全長になる**。gemma4 の golden には日本語ケース
   （capital-ja, T=10）が既にあり、長文の日本語入力は実需そのもの。→ 5.md §4 の「Rust の merge
   queue / list 構造まで移植し、rank・position tie と stale pair の無効化を fixture で固定」を採る。
2. **byteIds は 256 本を明示構築して欠落・重複を拒否する**。実測では `<0x00>`…`<0xFF>` が
   連番（238..493 — §0.2-3）なので `unigram.ts:30-36` の `base + byte` 前提は成立するが、
   それは**この資産の実測事実であって schema の保証ではない**。compile 時に 256 本を引いて
   fail loudly させるコストはゼロに近い。

### 5.5 検証（5.md §4 の門をそのまま採る）

- **parity fixture は Python 側で `tokenizers.Tokenizer.from_file()` を独立に呼んで採る**
  （TS と Python が前処理 helper を共有すると恒真化する — ADR 0048 で経験済み）。
- 自然文だけでは足りない。**直接叩く**: BPE rank / tie、metaspace 境界、CJK、結合文字、非 BMP、
  AddedToken 隣接、byte 0..255、複数 byte fallback、不正 / 不完全 byte run、special token decode。
- **streaming の門**: 短い token 列について**全 chunk partition**で `push()` の分け方を変え、
  連結が one-shot `decode(ids)` と一致すること + 「byte run 以外を無制限に buffer しない」直接
  テスト（`finish()` まで全 token を溜めるだけの偽 streaming を通さない）。

---

## 6. 軸⑥ chat template（lens-llm L-6）

### 6.1 候補

| 候補                                        | 中身                                                                                                       | 判定                                                                                                               |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **a) ファミリ固有の純関数 1 本**            | `gemma4ChatPrompt(messages, options) → token id 列`・HF `apply_chat_template` 出力を fixture にした parity | 推奨                                                                                                               |
| **b) 汎用テンプレートレンダラ**（Jinja 風） | 一般化                                                                                                     | **ランタイム依存は Web 標準のみ**（CLAUDE.md 横断不変条件）と「投機的な一般化をしない」の両方に反する。§9 で衝突印 |
| **c) 完全に呼び手任せ**                     | 消費者が `<                                                                                                | turn>` を手書き                                                                                                    |

### 6.2 推奨: **a、ただし初版の射程を「素の会話」に絞り、tools / thinking は fail loudly で拒否する**

根拠:

1. **前例が ADR 0079（SBV2 二層入力）**: 「テキスト解析は呼び手の責務・変換関数 1 本」
   （`docs/backlog.md:134-135` の消化済み節）。recipe README も「`<bos>` は
   `chat_template.jinja` が出す = **ホストの仕事**」と明記している。
2. **全量移植は「小〜中」ではない**（lens-llm G3 の見積りを上方修正）: `chat_template.jinja` は
   **386 行 / 18,569 B**（実測）で、tool 宣言のフォーマッタ（`format_parameters` /
   `format_function_declaration`）・thinking チャネルの順序制御・tool_call 引数の直列化を含む。
3. **一方で素の会話の射程は本当に小さい**（実測行）:
   `chat_template.jinja:188` = `{{- bos_token -}}` / `:191` = `'<|turn>system\n'` / `:215` =
   `'<turn|>\n'` / `:234` = `'<|turn>' + role + '\n'`（`assistant` → `model` へ写像）/ `:372` =
   turn 閉じ / `:382-384` = 生成プロンプト `'<|turn>model\n'`（+ `enable_thinking` なら
   `'<|channel>thought\n'`）。**綴りは 5 個・分岐は 3 本**。
4. **綴りは Gemma 3 系ではない**（§0.2-5）。`<|turn>` / `<turn|>` / `<|channel>` / `<|think|>` で、
   `tokenizer_config.json` の `sot_token` / `eot_token` / `soc_token` / `think_token` が正本。
   lens-llm G3 の「`<start_of_turn>` 系」は**この checkpoint には当たらない**。
5. **停止条件と同じ資産から来る**: `generation_config.json` の
   `eos_token_id = [1, 106, 50]` = `<eos>` / `<turn|>` / `<|tool_response>`（§0.2-7）。
   chat 形式と EOS 集合を**別々の場所から拾うと片方だけ古くなる** → 5.md §3 の
   「product graph + weight shards + PLE sidecar + compiled tokenizer + BOS/special-token policy +
   prompt/chat format version を**同一 digest set** に束ねる」を採る。
6. **`<bos>` の所有者を分ける**（5.md §4）: `encode(text)` は `<bos>` を付けない（post_processor が
   何も付けない G4 の実測と一致）・chat 関数だけが付ける。分けないと chat 導入時に double-BOS。

### 6.3 拒否する入力（fail loudly）

`tools` / `reasoning`（thinking）/ 画像・音声パート / `role` が `system|user|assistant|developer`
以外。**黙って無視しない**（無視すると「tool を渡したのに使われない」が例外なしで通る）。
射程を広げるのは実需が出てから。

---

## 7. 波の分割案（実装順の縦割り 1 案）

> 各段の末尾に**検証の合格線**を置く。段 1a と段 1b は並行レーン。

| 段                                         | 中身                                                                                                                                                                                                                                      | 合格線                                                                                                                            |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **0. 契約固め**（コード 0 行）             | ADR 追記 2 本 = ①ADR 0068 追記 6「当面の既定出口は最終行 logits・topk 出口は k ≤ 63 と getitem 配線が要るので実需まで保留」②新 ADR「生成 API — GenerationProgram / GenerationSequence・pendingToken・イベント形・停止契約・sampler 契約」 | 型と契約が文書に載る。§8 の裁定が全て解決済み                                                                                     |
| **1a. tokenizer レーン**（GPU 不要・並行） | compile 台本（recipe 側 Python）+ `src/text/` の BPE / byteFallback / streaming decoder + `src/gemma/text/` の受理 schema + parity fixture                                                                                                | 独立採取した HF fixture と encode/decode がビット一致。全 chunk partition の streaming 一致。merge queue の性能が日本語長文で線形 |
| **1b. 製品グラフレーン**（GPU 要・並行）   | recipe 新変種 = **PLE 外出し + 最終行 logits 出口**を 1 系列に（§7.1 の裁定次第で 2 段に分割）+ ホスト PLE sidecar loader                                                                                                                 | 既存 `greedy.<case>` golden との交差 parity（`argmax(logits)` == 既存 token 列・3 ケース × K=16）。PLE 逆量子化のビット一致       |
| **2. sampling + 停止**                     | `src/generation/sampler.ts`（温度 / top-k / top-p / repetition penalty / logit bias / seed）+ EOS 集合停止 + `generateGreedy` を内部ヘルパへ格下げ（**parity 門は残す**）                                                                 | 温度 0 で既存 greedy 列と一致。sampler 自体は自前 fixture + 分布門                                                                |
| **3. sequence API**                        | `GenerationProgram` + `GenerationSequence` + `AsyncIterable` + AbortSignal + 多ターン（`pendingToken` 連結）                                                                                                                              | 多ターンで「直前 assistant の最後の token が落ちない」直接門。`break` 中断後の再開門。cancel が `signal.reason` を素通し          |
| **4. chat + パイプライン**                 | `gemma4ChatPrompt` + `Gemma4Pipeline.fromAssets` + **文字列 in → 文字列 out** の e2e                                                                                                                                                      | HF `apply_chat_template` 出力との id 列一致。e2e が実重みで完走                                                                   |
| **5. 配布形**                              | manifest 席 / dist recipe / モデルカード / shard（ADR 0081）/ pin 定数 / `fromPretrained`                                                                                                                                                 | dist 全門通過 + 実 DL 疎通。**ADR 0065 stage 6 のライセンス門が前提**                                                             |

### 7.1 段 1b の内部順序 — 裁定が要る分岐

- **案 α（推奨）**: PLE 外出しと最終行 logits を**同じ再 export**に載せ、製品グラフを 1 系列に
  する（5.md §3「product graph は次の一系列だけ」）。既存 2 系列（logits opt-in / token-only）は
  検収 fixture として残す。
  - 利点: 3.7 GiB 系列の再 export が 1 回で済む。製品グラフが最初から製品形。
  - 欠点: 段 1b が重くなり、ホスト PLE loader が e2e の前提になる。
- **案 β**: 最終行 logits を先に出し、PLE は段 5 の前に別途。
  - 利点: 段 1b が軽く、早く sampling へ進める。
  - 欠点: 再 export が 2 回。段 2〜4 の fixture を PLE 変更で採り直す。

### 7.2 この波で**やらない**こと（射程外の明示）

- decode 性能（K-11 = `wi4g32` の M=1 変種）— 別トラック（`docs/perf-ledger.md:69`）
- continuous batching / batch > 1 / 複数シーケンス — ADR 0066 決定 8 のスコープ外・**再裁定要**
- RoPE 表のホスト生成（lens-llm L-9 c）— ADR 0067 決定 4 と衝突・**再裁定要**
- KV の f16 席（L-8）・dispatch ダイエット（L-7 — 実測で利得上限が壁の数%と判明・research §5）
- topk の exporter 配線（`operator.getitem`）— 軸③ a を採る限り実需が立たない
- minicpm5 の token-only 系列 — L-11 が gemma4 に裁定済み

---

## 8. 要判断（ユーザー裁定）

1. **軸① のイベント面**: `AsyncIterable<GenerationEvent>`（推奨）か、anima 流 `onEvent`
   コールバック（家風一致）か。前者は「token 列は値であって進捗でない」を採る代わりに、
   リポ初の公開 async generator になる。
2. **軸① の `GenerationContext` 露出**: `GenerationSequence` の内側に完全に隠す（推奨・5.md）か、
   上級者向けに素の context も公開面に残すか。隠すと「1 token の未 commit frontier」が消費者から
   見えなくなる代わりに、context を直接使う既存の検収経路と面が二重になる。
3. **軸② の PLE 方式**: 「token-major 配布形 + 触った shard の遅延ロード + LRU」（推奨）か、
   全量ホスト常駐（**分割必須** — 単一 ArrayBuffer 天井 2,145,386,496 B < 2,351,662,080 B）か、
   hub に部分読み席を新設して行読み（b）か、PLE i4 化（c）か。
4. **軸③ の出口**: 最終行 logits（推奨・ADR 0068 追記で足りる）か、topk 製品グラフ
   （k ≤ 63 — モデル既定 `top_k: 64` を表現できない）か。
5. **段 1b の分割**: 案 α（PLE + logits 出口を 1 回の再 export に載せ製品グラフ 1 系列 — 推奨）か、
   案 β（logits 出口だけ先行し PLE は後）か（§7.1）。
6. **軸⑥ chat の初版射程**: 素の会話のみ + tools / thinking は fail loudly で拒否（推奨）か、
   tools まで初版に入れるか。
7. **sampler の既定値**: `generation_config.json` の推奨（temperature 1.0 / top_k 64 / top_p 0.95）を
   配布形の既定として焼くか、既定は greedy にして呼び手に選ばせるか。
   （§0.2-8 のとおり `top_k: 64` は topk 出口では表現できないが、最終行 logits 出口なら表現できる）
8. **`generateGreedy` の公開面**: 起票どおり内部ヘルパへ格下げ（barrel から外す = **breaking**）か、
   当面は公開のまま残して次の breaking 波で外すか。
9. **EmbeddingGemma の同乗**: 段 1a の tokenizer 実装に EG の資産 compile まで載せる
   （backlog later `docs/backlog.md:254` の消化）か、gemma4 だけで閉じるか。
10. **配布の対象**（段 5）: gemma4 E2B のみか、ライセンス門を待つ間に minicpm5 で配布経路だけ
    先行するか（`docs/backlog.md:47-48` が「配布経路の検証だけは minicpm5 で先行可」と書いている）。

---

## 9. 既存裁定との衝突チェック

| 案                                                                                       | 判定                                                                                                                                                                                                                                                                           |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 軸① 推奨（sequence が counter を持たない・position は run 直前に `pastLength` から作る） | **整合**（二重簿記の禁止 — ADR 0066 決定 6・`greedy.ts:5-9`）                                                                                                                                                                                                                  |
| 軸① の `rewind` 利用                                                                     | **採らない**（sliding 含む context は全拒否 — `generation-context.ts:494-497` / ADR 0066 追記 2）。緩和は compaction 実装と対で**再裁定要**                                                                                                                                    |
| 軸① の continuous batching                                                               | **射程外**（ADR 0066 決定 8 — batch>1 / 複数シーケンスはスコープ外。**再裁定要**）                                                                                                                                                                                             |
| 軸② 推奨（PLE を通常のグラフ入力へ + ホスト sidecar）                                    | **整合**。前例 = 「flow / voice の相対位置表はグラフ入力 — 生成はホスト側の責務」（`docs/limitations.md:374`）・ADR 0079（テキスト解析は呼び手の責務）。汎用ランタイムに「pageable initializer」という第五の weight lifetime を足すなら**再裁定要**（5.md §3）— 本案は足さない |
| 軸③ 推奨（最終行 logits → ホスト sampling）                                              | **整合**（ADR 0068 決定 4 の opt-in 席・決定 4 の「sampling / RNG はホスト維持」）。追記で足り、**再裁定不要**                                                                                                                                                                 |
| 軸③ の代替（GPU nucleus / full sampling）                                                | **衝突**（ADR 0068 が正面から却下・**再裁定要**）— 出さない                                                                                                                                                                                                                    |
| 軸⑤ 推奨（compile-to-asset・表を焼く・実装は逐語移植）                                   | **整合**（qwen2 / unigram の家風・`asset-gates.ts` の門）。**モジュール副作用ゼロ**は byte encoder 等をモジュールスコープ const に持たない形で守る（`qwen2-tokenizer.ts:70-72` の前例）                                                                                        |
| 軸⑥ 候補 b（汎用 Jinja 風レンダラ）                                                      | **衝突**（ランタイム依存は Web 標準のみ + 投機的な一般化をしない）— 出さない                                                                                                                                                                                                   |
| 軸⑥ 推奨（射程外の入力を fail loudly）                                                   | **整合**（fail loudly — 黙って近似しない）                                                                                                                                                                                                                                     |
| lens-llm L-9 c（RoPE 表をホスト生成）                                                    | **衝突**（ADR 0067 決定 4「RoPE は attention op の外・エクスポータがグラフに焼く」・**再裁定要**）— 本波では扱わない                                                                                                                                                           |

---

## 10. レンズ / 外部レビューの記述で、実物と突き合わせて訂正した点

1. **lens-llm §1.4「EmbeddingGemma の tokenizer は gemma4 とほぼ同じ」** → BPE コア
   （merges）はビット同一だが、vocab は 6,206 スロット差・added_tokens は 24 vs 6,415・
   post_processor は別。**実装は共用・資産は別**（§0.2-1）。
2. **lens-llm L-5「実装コスト 大」** → Gemma は Unicode 表・NFC・case folding のいずれにも触らない
   構成なので、Qwen2 で必要だった焼き表 3 種が要らない。encode 側のコストはレンズの見積りより
   小さい。逆に**BPE の探索構造は最初から merge queue が要る**（Gemma の pre_tokenizer は空白分割
   だけで、日本語のような空白の無い入力では 1 pre-token が全長になる）（§5.4-1）。
3. **lens-llm G3 / L-6「`<bos>` + `<start_of_turn>` 系」** → gemma-4-E2B-it の綴りは
   `<|turn>` / `<turn|>` / `<|channel>` / `<|think|>`。`<start_of_turn>` は G4 の vocab に無い（§0.2-5）。
4. **lens-llm L-2 の選択肢 a「単純だが重い」** → **単一 ArrayBuffer では原理的に不可**
   （2,351,662,080 B > Chromium 上限 2,145,386,496 B）。分割は a/b 共通の必須要件（§2.3）。
5. **lens-llm L-3 の「11ms の床の 5% 以下」** → 床が支配という前提自体が実測で覆っている
   （research §0）。壁は 85ms/token なので、1 MiB readback の比率は**さらに小さい**（1% 未満）。
   結論（a を推す）は変わらないが、根拠の数字は差し替えるべき。
6. **CG5-2（同一 context への並行 run）は修正波で既に閉じている**
   （`generation-context.ts:337-344`）。sequence 設計は runtime 側の single-flight を**前提として
   使える**ので、5.md §1 の「runtime 側でも `acquireRun()` で拒否するのを勧めます」は消化済み。
7. **CG5-3（`P+Q` の u32 検査が run 成功後）は今日も現状のまま**
   （`generation-context.ts:589-600` の `#advance` が唯一の和の検査点）。severity optional・
   実到達性ほぼゼロという評定どおりで、本波では触らない。
   </content>
   </invoke>
