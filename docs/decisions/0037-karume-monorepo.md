# 0037: Karume monorepo — ブランド・パッケージ構成・配布方式

- Status: accepted
- Date: 2026-08-05
- 関連: ADR [0001](0001-scope-and-non-goals.md)（フラット構成の 1 点を本 ADR で改める）/
  ADR [0003](0003-ir-v1.md)（IR コンテナ規約）/ ADR [0008](0008-public-api.md)（公開 API 面）

## Context

パッケージとして公開するにあたり、ランタイム・エクスポータ・パイプライン配線を 1 つの
傘ブランドの下に複数パッケージとして編成する。名前・パッケージ分割・モデル資産の運び方を
まとめて裁定する。

## Decision

### 1. 傘ブランドは **Karume**

- JSR: `@karume/runtime`（IR 実行）/ `@karume/hub`（モデルの resolve・DL・cache・variant
  解決）/ `@karume/models`（パイプライン・tokenizer）。
- PyPI: `karume`（エクスポータ CLI）。
- リポジトリ: `hdae/karume` の monorepo（Deno workspace）。ADR 0001 の「フラット構成・
  workspace 機構は使わない」はこの 1 点だけ改める。

空き実測（2026-08-05 時点）: **karume は JSR / PyPI / npm すべて空き**。候補比較は次のとおり。

| 候補     | 判定 | 理由                                                                     |
| -------- | ---- | ------------------------------------------------------------------------ |
| `karume` | 採用 | 3 レジストリすべて空き。特定の技術的主張を名前に負わせない               |
| `zarame` | 却下 | PyPI に先客                                                              |
| `kasoku` | 却下 | 空きは同点だが、**速度で ONNX と勝負する看板を避ける**ポジショニング判断 |
| `mokei`  | 却下 | PyPI / npm ともに先客                                                    |

`Zarame` は将来の **TS ネイティブモデル定義**プロジェクトの名前として温存する（下の 2 を参照）。

### 2. 方式は **変換アーティファクト方式**

エクスポータ（Python）で焼いた資産を `@karume/hub` がロードして `@karume/models` の
パイプラインが実行する。**モデル定義の管理は Python 側**に置き、TS 側は「焼かれたグラフを
実行する」責務に絞る。

- 目的は配線の縮退 — 素朴なホスト配線は 1,100 行超に達する（実測）。これを
  `AnimaPipeline.fromPretrained(...)` + `generate(...)` の**1 画面**に縮めるのが
  hub / models の存在意義。
- 着手は **Anima 先行**（SBV2 はモデルライセンス都合で後回し）。
- HF の素の safetensors を直接動かす **TS ネイティブなモデル定義**は本プロジェクトの
  スコープ外（将来の別プロジェクト = `Zarame`）。

### 3. 配布形は **HF 1 モデル 1 リポ・複数 safetensors**

- 1 グラフ = 1 safetensors。IR JSON は `__metadata__` 埋め込みの現行形を維持する
  （`docs/ir-v1.md` のコンテナ規約は不変）。
- **独自拡張子 `.krm` は不採用** — 素の safetensors のままにして、HF のプレビューや既存
  ツールで開ける性質を捨てない。
- エクスポータの出力ディレクトリは**そのまま HF リポとしてアップロードできる形**にする —
  変換モデル配布の習慣に従い、モデルカード `README.md`（YAML frontmatter:
  `library_name` / `base_model` / `license` / `tags` / `pipeline_tag`）を同梱する。
- 配布用 manifest（variant 解決表）は**別 ADR** を起票する。本 ADR では決めない。

### 4. `@karume/models` は barrel + サブパスの両建て / 副作用ゼロを不変条件へ昇格

- `mod.ts` の barrel と、ファミリ別サブパス export（`deno.json` の exports マップ）を両方
  提供する。使う側は「1 ファミリだけ import」も「barrel から import して tree-shaking に
  任せる」も選べる。
- その成立条件として **全モジュール副作用ゼロ**（top-level 登録・グローバル可変状態・
  import 時実行の禁止）を**横断の不変条件へ昇格**する（`CLAUDE.md` に記載）。崩れると
  barrel 経由の shake が静かに死ぬ。
- **未検証**: JSR の npm 互換層が生成する `package.json` に `sideEffects: false` が出るかは
  実測していない。無くても ESM の副作用解析でおおむね shake されるが、フラグ有りが確実。
  hub / models の整備時に実測して結果を記録する。

## Consequences

- 立ち上げは段階制: hub（manifest ADR + 実装）→ models の anima（`AnimaPipeline` 再編）→
  エクスポータ CLI 化（PyPI `karume`・サブコマンド式）→ HF 実網通し・公開。README の
  本記述は最後に書く（それまで WIP スタブ）。
- モデル e2e（anima / sbv2 / deberta の実重み系）はパイプライン整備と同時に入る — それまで
  本リポのテストカバレッジは**ランタイム核のみ**（意図的な過渡状態）。
- `models/` の大型資産は untracked（エクスポータで再生成可能）。
