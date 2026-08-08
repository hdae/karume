# ACTIVE_DESIGN — Karume

> Short index of _current_ design focus. Keep it to a screenful. Reviewers and planners read this
> FIRST (alongside `CLAUDE.md` / `docs/`) so they don't start cold or misread an intentional
> migration as a defect. Update it whenever the current design context shifts.
>
> Last updated: 2026-08-08

## Active redesigns (in flight)

- **立ち上げロードマップ（ADR 0037）は P0〜P5 まで到達し一段落**。P3/P4 で `AnimaPipeline`
  （fromPretrained / fromAssets・`using` 対応）+ 共通 image 層 + 配布形 `models/anima-turbo/`
  （`karume dist`・実 hash・格納 dtype 門）+ `karume` サブコマンド CLI + **英語**モデルカード
  README 自動生成まで完了。**移植の門 = PNG sha256 ビット一致 ×4 が全緑**（ローカル /
  取得層 + integrity / example CLI）。P5 で HF 実網通しと配布形の公開まで済み、JSR publish の
  CI/Release ワークフローも設置済み（リポ直下 README は release 準備時に書き上げる WIP 表記）。
  **残る宿題**: ①波 1 積み残しの参照フィクスチャ系テスト（timestepsProj atol 突合等 —
  anima-pipeline 系列の再エミットが前提）②tokenizer parity fixture の models 側への移設。
- **models/sbv2 — 常駐**: SBV2（テキスト → 音声）は `packages/models/src/sbv2/` に移植済みで、
  サブパス面 `@karume/models/sbv2` は `Sbv2Pipeline` だけを出す（example 111 行）。実重み e2e は
  `packages/runtime/tests/e2e_sbv2_test.ts`（3 系列 × 5 ターゲット × 5 ケース）、GPU 不要の
  ホスト側は `packages/models/tests/sbv2_{relattn_parity,text,host,pipeline}_test.ts`。配布形
  `models/sbv2-FN4/` と CLI `karume export-sbv2` まで揃っている（manifest の確定は ADR 0039）。
- **参照実装ブランチからの再実装（C 波）— 完了**: `codex/kernel-quick-fixes` の triage を
  **設計から書き直して**取り込んだ。C1 = i8a8 linear の accumulator 静的展開 / PipelineCache の
  未決着共有 / exporter の隣接 permute 合成、C2 = **融合パス新設**（RoPE・SiLU・upsample2x を
  1 dispatch へ + 恒等 expand の別名化 — ADR [0040](../docs/decisions/0040-fusion-pass.md)）、
  C3 = f32/f16 linear の accumulator 静的展開。**w8a8-1024 16.1 → 13.9s（1.16×）/ f16-1024
  38.5〜38.8 → 30.1〜30.3s（1.28×）**・PNG 門 4 本は sha256 不変。採否と実測の正本は
  [research/2026-08-08-branch-adoption-perf.md](../docs/research/2026-08-08-branch-adoption-perf.md)、
  参照側の記録は [research/2026-08-06-kernel-triage/](../docs/research/2026-08-06-kernel-triage/)。
  **不採用**（実測根拠あり）: contiguous elementwise / QUANT-010 / i8a8 タイル幾何の一般化。
- **F 波（VRAM OOM の誤報告）— 完了**: 確保失敗が派生 validation に化けて「破棄後使用」に
  見えていた件を根治（errorScope の報告順を根因優先へ + Session 構築での staging 解放）。
  機序は [research/2026-08-08-vram-oom-misreport.md](../docs/research/2026-08-08-vram-oom-misreport.md)。
- **配布の次手（未着手）**: HF Xet では**レイヤー分割は DL を速くしない**（分割なし + Range
  4 並列が最速・並列は 4 で飽和）ことを実測済み。hub 側の Range 並列取得が費用対効果で勝つ。
  [research/2026-08-08-xet-split-probe.md](../docs/research/2026-08-08-xet-split-probe.md)。
- **manifest v2（ADR [0041](../docs/decisions/0041-manifest-v2.md)）— 裁定済み・実装待ち**:
  1 リポ複数モデル（ファミリー / シリーズ / 作者の緩い括り・`defaultModel` 必須）+ 語彙整理
  （presets → `quants`・variant → `dtype`・components → `weights` / `assets` 分離）。
  **v1 パーサは持たない**（実配布は anima-turbo 1 件のみで v2 再アップ）。shards は実測で
  見送り。実装順・積み残しは ADR 0041 末尾。

## Pitfalls

- **Metal（Apple GPU）は WGSL の受け取り方が Vulkan と違う** — threadgroup の `vec4` へ
  **動的インデックスで成分を書く**（`sb[i][wsl] = v`）と `wsl != 0` が黙って捨てられる。
  GEMM の B タイル充填は静的成分への `switch` 展開で回避済み（`gemm.ts` の
  `storeBTransposed`）で、**同じ形を新しく書かないこと**。残る誤値（attention i8a8 /
  conv2d の 2 経路一致）は known-issues.md、**性能**（Linux の 31〜41 倍 — 帯域は健全なのに
  GEMM のタイリングが 1.21x しか効かない = Apple GPU 向け未最適化）は
  [research/2026-08-06](../docs/research/2026-08-06-metal-silent-miscompute.md) §3。
  Metal では `gpuTiming` が使えない（dispatch 数がサンプル上限を超える）。
- **融合 matcher は Anima の実測形への決め打ち**（`[1,H,S,128]` の RoPE、headDim 128/64、
  upsample2x の 6 ノード列など — exact 一致のみで、掴めなければ素のノード列へ fallback）。
  **エクスポータのノード発行順や形が変われば黙って外れ、値は正しいまま性能だけ落ちる**
  （例外も警告も出ない）。`Diagnostics.lastRunFusions` が唯一の観測点なので、性能が戻ったら
  まずここを見る。
- **RoPE / SiLU 融合カーネルの丸め障壁（workgroup memory 往復）は WGSL 仕様の保証ではなく
  実測依存** — バックエンド更新やドライバ更新で PNG 門が割れたら、まずここを疑う。
  upsample2x は u32 ビット複製なので丸めの議論自体が無い。
- **`deno task verify` は `.claude/worktrees/` が存在する間、main 作業ツリーで素に走らない**
  （末尾の `deno test -A` がパス無しなので worktree 側のチェックアウトまで拾う）。当面は
  test 段だけ `deno test -A packages` で代替する。**恒久策は未裁定**。
- **Session 構築は重みアップロード後に submit を挟む**（`queue.writeBuffer` の staging は
  submit 完了まで解放されない）。この 1 回を消すと f16 preset の瞬間ピークが **+2.7GiB**
  （5,719 → 8,391MiB・確保天井 11,136〜11,264MiB に対し余裕が 2.7GiB まで縮む）。
- **`models/` に置くのは HF へそのまま上げられる配布形だけ**（1 ディレクトリ = 1 HF リポ）。
  エクスポータの系列出力は `outputs/series/`、実重みの**入力素材**は `inputs/<ファミリ>/<名前>/`
  — 綴りの正本は `karume/paths.py`。**ADR と docs/research 内の `models/anima-*` 表記は当時の
  記録**（時点スナップショットなので直さない）。**turbo LoRA だけ未移行**（`anima_pipeline.py`
  の `--lora` 例が配布形の親に入力素材を混ぜたまま）。
- **配布資産の格納形は series ディレクトリ名でなくヘッダが正** — dist.py の格納 dtype 門が
  組み立て時に検査する（`--dtype` 付け忘れの素 F32 が PNG 門まで沈黙した実測事故が根拠）。
  宣言外ファイル検査の例外は直下の `karume.json` / `README.md` だけ。
- **現行識別子（`karume_ir` / `karume-ir`）以前に焼かれた資産は開けない**（互換シム無し —
  fail loudly）。`models/` と `outputs/` はどちらも untracked。
- モデル e2e は anima の PNG 門 4 本が本リポに常駐（`models/anima-turbo/` 資産が前提・無ければ
  明示 SKIP）。sbv2 の実重み e2e も復帰済み（系列 `outputs/series/sbv2-FN4{,-f16,-i8}/` が前提）。
  **deberta の実重み e2e だけは移行元リポに残置のまま**（`e2e_deberta_test.ts` は未移植）。
- models パッケージの tree-shaking は「全モジュール副作用ゼロ」不変条件が前提 — 崩れると
  barrel 経由の shake が静かに死ぬ。JSR npm 互換層が `sideEffects: false` を出すかは**未検証**。
