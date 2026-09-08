# ACTIVE_DESIGN — Karume

> Short index of _current_ design focus. Keep it to a screenful. Reviewers and planners read this
> FIRST (alongside `CLAUDE.md` / `docs/`) so they don't start cold or misread an intentional
> migration as a defect. Update it whenever the current design context shifts.
> 波順・作業項目の正本は [docs/backlog.md](../docs/backlog.md)、性能候補の採否は
> [docs/perf-ledger.md](../docs/perf-ledger.md)。ここは「今この瞬間の文脈」だけを持つ —
> 履歴・完了記録は ADR / research / git へ。
>
> Last updated: 2026-09-08（MTP ④ 設計の裁定 = ADR 0096・段 1〈verify 形の準備〉実装中）

## Now

- **Codex 性能調査（2026-09-06）の消化波（2026-09-07）** — 実測と設計の正本は
  [research 2026-09-07](../docs/research/2026-09-07-codex-perf-review-followup.md)・採否は perf-ledger H-11 / H-12 / H-13 /
  H-14 / K-17 / K-18 / K-19。入ったもの: ①観測席 `onRunDiagnostics` に **phase 第 2 引数**（`Gemma4RunPhase`）+ 停止 token の
  最終 decode run も通知（`95dfe71` — opbench の decode 平均は停止 run を含むようになり過去値と厳密には比較不可）②sampler の
  top-k は**有界 heap 一本**（`86fdbe5`・`SELECTION_LIMIT` 撤去）③PLE gather の hit 先行 + 重複 id 複写（`4d39c79`）
  ④**prefill の M バケット**（ADR [0066](../docs/decisions/0066-generation-context-state-slots.md) 追記 10・runtime `chunkBuckets`
  - PreparedPlan LRU 8・models `physicalChunkRows`・既定 `GEMMA4_CHUNK_BUCKETS` = [32, 64, 128, 256] — **512 は 768 より遅い**〈GEMM
    幾何の段〉）⑤**PLE 行読み**（ADR [0085](../docs/decisions/0085-ple-host-gather.md) 追記 2026-09-07 — 自然文 400 token で shard 読み直し
    137 回・42 s が原因の p50 問題。`Gemma4Assets.readPleShard` → `openPleShard`〈破壊的・limitations〉・hub 能力 ⑧ `openAsset` / `AssetRangeReader
  {cost: seek | scan}`（ADR [0086](../docs/decisions/0086-distribution-source.md) 追記）・runtime `parseSafetensorsHeader`・取得層
    `@hdae/fetch-cache` の `openCachedUrl` / `openHfFile`〈その ADR 0012・0.8.0 公開済み・hub の HF 取得元も追従済み〉）。
    **落とし穴**: Chrome の CacheStorage は Range 要求を無視する（200 全量）— 区間は `blob().slice()` で取る。Deno の `blob()` は
    全量を読む（stream 読み飛ばし = scan）。**MTP（Gemma 4 drafter）は復活条件 ①〜③ が済み、更新した予測倍率は抽出的な長文脈で 1.8〜2.6×・自由文 0.9〜1.0×
    （目標は実用レベル・台帳 = perf-ledger K-20・実測 = research 2026-09-08）**。
    次の波 = **~~K-21~~（済・`5701262`）→ ~~H-15~~（済・`c7120f2`）→ ~~③ E-4~~（済）→ MTP ④（裁定 = ADR
    [0096](../docs/decisions/0096-speculative-decoding.md)・2026-09-08）**: drafter は別 Session で target の KV スロットを
    **読むだけ**（IR の external スロット + `sharedStates`）・埋め込み表は Session 跨ぎの**共有 initializer**・部分 commit は
    **deferred commit**（`GenerationRun.commit: "deferred"` → `context.commit(rows)`）・sliding ring は capacity = window + 8・
    出口は logits `[1,R,V]` + hidden `[1,R,H]`（`last_row [R]`）・バケット 4 / 8 + PreparedPlan LRU 12・k は固定 3 から・
    greedy 先行。**段 1（verify 形の準備）実装中** → 段 2（drafter の入口）→ 段 3（投機ループ）→ 段 4（実測・調整）。
    **落とし穴**: 1 cycle は run 2 本が下限（PLE のホスト gather があるので draft token はホストを経由する）。
    **H-15（2026-09-07）**: slot backing を容量 1 から**バイト予算つき LRU 集合**へ（ADR
    [0095](../docs/decisions/0095-plan-backing-budget.md)・`SessionOptions.planBackingBudgetBytes` 既定 256 MiB・0 = 従来・
    Gemma4Pipeline の options に透過）。勘定 = 領域 + 所有する入力バッファ・常駐は max(予算, 最大 1 本) を超えない・見積りは
    `max(予算, 最大シナリオ)` を勘定側へ・context の焼き込み束は backing の世代ごとの表。実測: prefill run 壁 80 → 46 ms・
    定常ターン 646 → 581 ms・作り直し 10 → 1 回 / 5 ターン。**落とし穴**: 予算より大きい形（gemma4 chunk 768 = capacity 16K で
    528 MiB）は 1 本だけ = 長い prompt のターンは従来どおり作り直す / 常駐入力を焼き込んだ backing が保持されている間は
    `ResidentTensor.dispose()` が fail loudly（予算 0 で従来へ）/ 見積りに既定 256 MiB の下限が載る / 他家族は予算を変える口が無い。
    **K-21（2026-09-07）**: linear の GEMV 族に**行ブロック変種**（1 スレッド = 1 列 × rows 行・y タイル）を足し、門を
    1 ≤ M ≤ 64 へ（ADR 0082 追記 5・[research 2026-09-07-gemv-rows-k21](../docs/research/2026-09-07-gemv-rows-k21.md)）。
    rows は (格納, m, n) の純関数（並列度目標 16384 スレッド・天井 256 要素/語）でキー `…c32u4r<rows>…` に載る。
    既定経路と u32 完全一致。20 token prompt の prefill 135 → 80 ms・M=8 の linear 65 → 13 ms・decode 不変。
    **落とし穴**: 行ブロックの WGSL は行数ぶん展開するので、naga の解析費がテキスト量に超線形 — 天井を 512 → 256 に
    下げて初回ターン ≈ 1.3 s → +85 ms（codegen 門に 80,000 文字の上限）。行ループを `for` に畳む形は 2〜7 倍遅い
    （private 配列がローカルメモリへ）。カーネル単体 A/B は submit 先頭にスピンアップ pass を置く（アイドル 210 MHz
    からの立ち上がりで最初の pass が 2〜15 倍遅く出る）。診断・census で linear を数えるときは `linear:` と
    `linear_gemv:` の**両方**を見る（M ≤ 64 は全て後者）。
- **0.12.0 公開完了（2026-09-06）** — lockstep bump `a24d656` → GitHub Release v0.12.0 → JSR 0.12.0 →
  `deno task smoke:published` 緑。中身 = runtime の K-16 / K-14 / K-13（下の OP / Fusion 節）+ hub の
  `evictCachedAssets` 修正（同一参照集合の兄弟席を既定の守る側から外す・`protect` / `alsoEvicted` — ADR
  [0094](../docs/decisions/0094-hub-cache-inventory-and-eviction.md) 追記）+ models の `onRetry` 透過
  （8 家族の hub オプションを `FromPretrainedHubOptions` に 1 本化）。breaking なし（公開面の差分の正本は
  リリースノート v0.12.0）。配布形は 0.10.0 のまま **HF 10 リポ**（`karume-anima` / `karume-anima-extra` /
  `karume-irodori-v4-small` / `karume-irodori-v4.1-small` / `karume-sbv2-jvnv` /
  `karume-gemma4`〈`-e2b` から改名済み〉/ `karume-siglip2`〈base + so400m 同居〉/
  `karume-depth-anything-v2` / `karume-birefnet-hr`〈1024 + 2048 同居〉/ `karume-lucida`〈同〉）で、
  取得元対応表は **7 家族 10 エントリ**（全て JSR の公開面に出ている）。未配布は vowel-detector
  だけ（[backlog](../docs/backlog.md)）。
- **キャッシュ保守面が入った（2026-09-05）** — `listCachedAssets` / `evictCachedAssets`（ADR
  [0094](../docs/decisions/0094-hub-cache-inventory-and-eviction.md)）。参照勘定は manifest 1 本の中
  だけで、全在庫の他の選択が守る・越境参照は残す・**対象と参照集合が同一の兄弟席は既定で守らない**
  （守る側の明示は `protect`・巻き添えは `alsoEvicted` — ADR 0094 追記 2026-09-06）。429 / 503 の再試行と HF 層の受信上限は取得層
  `@hdae/fetch-cache` 0.7.0 側（その ADR 0010 / 0011）。hub は 2026-09-06 に追従済み（依存 `^0.7.0`・
  `LoadManifestOptions.onRetry` の透過・`transport.ts` の撤去 — ADR 0094 決定 4。`karume.json` の
  1 MiB 上限は全量受信後の判定になった — [limitations](../docs/limitations.md)）。
- **OP / Fusion の波（2026-09-06）— 融合候補は実測で閉じ、次は GEMV**: 4 家族の実走ベースライン
  （[research](../docs/research/2026-09-06-op-fusion-baseline.md)・`opbench graph` は 4 家族対応）の上で
  K-15（`gelu_tanh`+`mul`）/ K-7（ゲート付き残差）を融合ルールとして実装 → ABBA → **判定線に届かず revert**
  （壁 −0.4% / 全 GPU −0.6% — [research](../docs/research/2026-09-06-fusion-spikes-k15-k7.md)）。要素ごと op の
  融合は「中間 1 本の往復ぶん」しか効かず、dispatch 削減は壁に出ない。P-5（`permute` 畳み込み）は
  実装せず保留。続く a 案の 1 本目 **K-16（lm_head i8 を GEMV 族へ・ビット同一）は済**（`5ddd186` —
  単体 ×5.0・decode GPU −21〜24% — [research](../docs/research/2026-09-06-gemv-i8-k16.md)・ADR 0082 追記 4）。
  **K-14（①QK の D 並列縮約 ①′）も済**（`cce129d` + M=1 門 `4182b8b` — decode 壁 P=16K −9〜15%・prefill は
  逆行するため ① のまま — [research](../docs/research/2026-09-06-state-qk-parallel-k14.md)）。席は K-12 と同じ
  `stateAttentionReduce` 1 つで、①′ は M=1 の計画だけ。**K-13 も済**（`ad8a4b9` ①ₜ + `39d5e4e` ③ₜ — prefill 計画
  M ≥ 16 は GEMM 骨格のタイル経路で ①/③ とビット同一・P=16K の prefill 壁 −64% —
  [research](../docs/research/2026-09-06-state-attention-tiled-k13.md)）。幾何表 = M=1 → ①′ / ③′（席）・M ≥ 16 → ①ₜ / ③ₜ
  （既定）。**次は未起票**（候補: prefill の linear 72%・decode の linear_gemv 57%〈split-K は席が要る〉・anima の
  attention 27% + VAE 29%・siglip2 の分解 attention）。
- **BiRefNet 2048² 工事 A / B / C は消化（2026-09-05）** — ADR
  [0093](../docs/decisions/0093-transient-liveness-packing.md) を runtime へ結線（B + C）し、recipe の
  パッチ ⑨（A）で decoder 末尾の巨大中間を消した。実測: 1024² 中間 6,283 → 749 MiB / 2048² 中間
  2,948 MiB・総確保 ≈ 4.1 GiB・run 7.5〜8.6 s。配布形は 1 リポ 2 モデル（モデル名 = 解像度・既定
  1024 — ADR 0092 決定 9）で `karume-birefnet-hr` / `karume-lucida` を **HF へ公開済み**
  （2026-09-05・pin は `BIREFNET_SOURCES`・0.10.0 で JSR の公開面へ出た）。
- **可変 capacity（[ADR 0091](../docs/decisions/0091-gemma4-host-rope-variable-capacity.md)）の
  意図的な現状 4 点** — 欠落に見えるが設計どおり: ①RoPE 表を焼いた旧配布形は読めない
  ②`GreedySpec` / `GenerationProgramSpec` は `positionIds` を持たない（位置の唯一の供給口は
  派生入力 `derive`）③states 形 attention を持つグラフの見積りは `maxStorageBufferBindingSize`
  が必須 ④K-12（③PV の KV 並列縮約）は `Gemma4Pipeline` の既定が `"parallel"` — runtime 低レベル
  面の既定は `"sequential"` のまま。
- **shard 仕様**: 受理上限は 256MiB 1 本をファイル長で検査し、超える単位はテンソル分割（piece）で
  割る（[ADR 0081](../docs/decisions/0081-shard-spec-v2.md) /
  [0090](../docs/decisions/0090-shard-spec-v3-tensor-pieces.md)）。単一ファイルの配布形は無い。
- **取得元抽象 `DistributionSource`**（[ADR 0086](../docs/decisions/0086-distribution-source.md)）:
  `denoDirectory` はローカルミラーを複製せず直読し、越境参照は明示 mapping と明示 fallback だけを
  経路に持つ。
- **GPU メモリ適合は絶対上限との決定論的比較**
  （[ADR 0089](../docs/decisions/0089-memory-limits-preflight.md)）: 重み / state の確保前検査・
  exporter の `requiredLimits` 一括導出・models は重み DL 前に検査。合計と物理空き VRAM は比較
  しない（原理的に不能 — [limitations](../docs/limitations.md)）。
- **破壊的変更の消費側 doc は [limitations](../docs/limitations.md) が索引** — 席の撤去・改名・
  throw 化はそこと各 ADR / リリースノートが正本。

## Open decisions

- MiniMax-H3（動画生成・オープンウェイト 33.1B/42.5GB 級）は遠期の関心として記録のみ —
  ブラウザ実行はメモリ規模的に現行スコープ外（レビュー DS-4）。
- 差分レビュー見送り分の中優先 2 件（正本 = `.claude/reviews/2026-09-03_7fc4ada/ROADMAP.md`）:
  W-G5-7 = `tools/opbench` / `tools/fusion-hints` の資産解決を `tools/_shared/assets.ts` へ統合
  するか / W-G4-4 = chunk 上限の出所を provenance の `sym_max` 欄へ移すか（再 export に同乗）。
  （3 件目だった ADR 0033 決定 5 の幾何突合は 2026-09-05 に「実態へ追記」で閉じた）
- Metal で `--diagnostics`（`gpuTiming: true`）を付けると device ごと落ちる件の改修投資判断 —
  切り分け実験が先（[known-issues](../docs/known-issues.md) の Metal `--diagnostics` 節）。

## Pitfalls（現役のみ）

- **フル走行の verify は VRAM 圧で稀にフレークする**（毎回別のテストが落ち、単独再走は常に緑
  — known-issues）。落ちたファイルの単独再走で切り分ける。
- **ベンチ生成先と実画像コーパスは席が別** — コーパスは `outputs/misc/corpus/` の凍結コピー
  （再実行上書き事故は構造解消済み — [assets-layout](../docs/assets-layout.md)）。凍結コピーへ
  機械が直接書く形へ戻さない。
- **`linearCompute: "a8"` は i8 常駐と i4 常駐で数値契約が別**（i8 = full-k 厳密 / i4 = group
  部分縮約 — ADR [0076](../docs/decisions/0076-w4a8-linear-execution.md)）。取り違えると atol=0 の
  主張が意味を失う。経路の識別はパイプラインキーの `:wi4g32` サフィックスと診断が担う。
- **Metal**: threadgroup `vec4` への動的インデックス書きは黙って捨てられる（`gemm.ts` の
  `storeBTransposed` の switch 展開を新しい箇所で崩さない）。attention i8a8 / conv1d /
  conv2d / gru_scan / linear GEMV の Metal 数値差は known-issues・Metal は gpuTiming 不可
  （limitations）。
- **融合 matcher は実測形 exact-match** — exporter の発行順・形が変わると黙って外れ、値は
  正しいまま性能だけ落ちる。観測 = `Diagnostics.lastRunFusions` +
  `assets_fusion_counts_test.ts`。**row-block だけは外れ方が性能でなく資源** — 128MiB 級
  device で resource-limit failure に戻る（**分解経路の matcher だけの話** — 保存 attention は
  states 形・融合 attention とも行ブロックを op 内蔵で持つ〈ADR 0067 決定 7〉）。分解形が
  matcher から外れると `bmm [H,S,S]` が**ノード出力スロット**になり原理的に分割不能 — 現状の
  該当（anima text_encoder / conditioner）は T=512 固定 16MiB で無害。
- **RoPE / SiLU 融合の丸め障壁（workgroup memory 往復）は実測依存** — バックエンド更新で
  PNG 門が割れたらまずここを疑う。
- **sim の A/B は同一リグ内でのみ有効** — 出荷リグでは GPTQ の丸め解が変わり、発話実現が
  再抽選される（最終裁定は必ず出荷バイトで）。**adaLN（modulation の scale/shift/gate）は
  量子化感度が高い**（irodori 実測 — 他 DiT へは未実測の仮説）。実測の正本 =
  [research/2026-08-24-gptq-expansion-quality.md](../docs/research/2026-08-24-gptq-expansion-quality.md)。
- **`deno task verify` はリポ内に worktree を置くと worktree 側まで test を拾う** — worktree は
  リポ外に作る（CLAUDE.md 検証コマンド節。deno.json に exclude は設けない — 2026-08-16 裁定）。
- **Session 構築の重みアップロード後 submit 1 回は瞬間ピーク +2.7GiB を抑えている** — 消さない。
- **資産の置き場**: `models/` = HF へそのまま上げる配布形のみ・系列出力は `outputs/series/`・
  入力素材は `inputs/<ファミリ>/<名前>/` — 綴りの正本は
  `tools/export-recipes/_shared/paths.py` と [assets-layout](../docs/assets-layout.md)。
  格納 dtype はヘッダが正（dist の門が検査）。旧識別子以前の資産は開けない（互換シム無し・
  席名の移行表 = ADR [0074](../docs/decisions/0074-quant-seat-naming.md) 決定 6）。
- models パッケージの tree-shaking は「全モジュール副作用ゼロ」不変条件が前提。JSR npm 互換層の
  `sideEffects: false` 出力は未検証（backlog release）。

## Stable invariants

- **公開 revision の正本は家族ごとの取得元対応表** `<FAMILY>_SOURCES`（ADR
  [0092](../docs/decisions/0092-distribution-repos-and-sources.md) 決定 3・現物 =
  `packages/models/src/*/config.ts`・re-export は `packages/models/mod.ts` と家族サブパス）。
  **キー = 公開リポ名から `karume-` を落としたもの**（`IRODORI_SOURCES["irodori-v4.1-small"]`）。
  全家族を畳んだ `KARUME_SOURCES` は barrel だけが出す。docs・モデルカード・テストに SHA を
  写さない。`fromPretrained` の `ref` は必須（既定ソースは無い）。
- **op ごとの tolerance は実測表が正本** — `packages/runtime/tests/helpers/op-tolerance.ts`。
  表に無い op は fail loudly（共通既定値での掃引は退役）。
- **manifest は `karume/4`** — それ以外の `format` は unsupported で落とす（互換シム無し）。
- **PyPI `karume` は未リリース** — exporter / recipes に移行シムは置かない。
