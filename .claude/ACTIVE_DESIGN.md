# ACTIVE_DESIGN — Karume

> Short index of _current_ design focus. Keep it to a screenful. Reviewers and planners read this
> FIRST (alongside `CLAUDE.md` / `docs/`) so they don't start cold or misread an intentional
> migration as a defect. Update it whenever the current design context shifts.
> 波順・作業項目の正本は [docs/backlog.md](../docs/backlog.md)、性能候補の採否は
> [docs/perf-ledger.md](../docs/perf-ledger.md)。ここは「今この瞬間の文脈」だけを持つ —
> 履歴・完了記録は ADR / research / git へ。
>
> Last updated: 2026-09-04

## Now

- **0.8.0 リリース済み（2026-09-04）**: JSR 3 パッケージ（runtime / hub / models）= 0.8.0・HF 6 リポ
  公開（`karume-anima` / `karume-anima-extra` / `karume-irodori-v4-small` /
  `karume-irodori-v4.1-small` / `karume-sbv2-jvnv` / `karume-gemma4-e2b`）・pin 定数 6 本を焼き込み
  済み。未配布は 4 家族（siglip2 / birefnet / depth-anything / vowel-detector）。
- **次の作業波（2026-09-04 裁定の順）** — 内容と残件の正本は [backlog](../docs/backlog.md) now:
  a) OP マイクロベンチ 2 段目 + Fusion 半自動発見 2 段目（1 段目は消化済み）
  b) 未配布 4 家族の初回公開
  c) perf K-13 / K-14（[perf-ledger](../docs/perf-ledger.md) が起票の正本）
  d) `tools/export-recipes/` の切り出し（案 A = レシピのみ・汎用 core は wheel に残す）
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
- 差分レビュー見送り分の中優先 3 件（正本 = `.claude/reviews/2026-09-03_7fc4ada/ROADMAP.md`）:
  W-G5-7 = `tools/opbench` / `tools/fusion-hints` の資産解決を `tools/_shared/assets.ts` へ統合
  するか / W-G4-4 = chunk 上限の出所を provenance の `sym_max` 欄へ移すか（再 export に同乗）/
  [ADR 0033](../docs/decisions/0033-vae-fixed-tile-decode.md) 決定 5「TS 側が幾何そのものを突合
  する」経路を作るか、決定 5 を実態へ追記するか。
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
