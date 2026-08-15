# 0065: exporter を汎用 core（PyPI）とモデル別 recipe（repo 専用）へ分離する

- Status: accepted（2026-08-15・ユーザー裁定 — 外部レビュー〈構造・ライセンス監査〉の
  案 A を採用。「ライセンス的にも exporter は汎用モジュール + モデル別スクリプトへ再編すべき」
  はユーザー方針 2026-08-14）
- 関連: ADR [0037](0037-karume-monorepo.md)（PyPI `karume` の親決定 — 本 ADR が「エクスポータ
  CLI」の性格を「汎用 exporter / IR tooling」へ改める）/ [0052](0052-transactional-dist.md)
  （dist の transactional 契約 — engine 側に残る）/ [0063](0063-safetensors-physical-layout.md)
  （emit の物理配置契約 — core の責務）

## Context

- 公開 API（`convert` / `normalize` / `emit` / `verify` / `pipeline`）は汎用で、変換コアは
  `patch_*` へ直接依存しない（import 全数走査で確認済み）。しかし **wheel の物理境界は
  `karume/` ディレクトリ全体**で、`patch_anima.py` 等の上流モデル由来コード（多数の逐語移植を
  自己申告）・`anima_text.py`・family 知識を焼いた `dist.py`（3,557 行）/ `modelcard.py` /
  `cli.py`（10 モデルのスクリプト名簿）が同一 Python distribution に入る。
- つまり **API 上の境界と配布・ライセンス上の境界が一致していない**。`pip install karume` の
  利用者は torch.export → IR に不要な family 知識と、その provenance 義務を一緒に受け取る。
- 逆流の実測: `dist.py:69-84` が modelcard の family metadata / renderer を直接 import・
  `dist.py` に `style_bert_vits2` の遅延 import・cross-family utility（`ROPE_BUFFER_NAMES` /
  `assert_rope_lifted`）が `patch_anima` に埋まり EmbeddingGemma / Irodori が参照・
  `conftest.py` が Irodori checkout を sys.path へ注入。

## 決定

1. **PyPI `karume` = 汎用 exporter core のみ**。残るのは IR / ops / dims / extents / shapes /
   convert / normalize / quantize / act_quant / emit / verify / pipeline / goldens と、
   **generic dist engine**（artifact 配置・manifest・hash・staging/swap〈ADR 0052〉・汎用検証）
   / **generic modelcard renderer**。モデル非依存として成立する CLI（`karume verify` 等）だけ
   残し、family command・スクリプト名簿は削除する。
2. **モデル別 recipe は `tools/export-recipes/<family>/` へ**（wheel 外・repo 専用）:
   export 台本・`patch_*`・参照 pipeline・demo・quant 計測・**dist recipe**（source layout・
   モデル名・quant variants・repo 命名・カード選択）・**card template / metadata**・family
   テスト・`THIRD_PARTY_NOTICES.md`（provenance は family 単位でコードと同居させる）。
   複数 family で共有するが core に昇格できない補助は `tools/export-recipes/_shared/`。
3. **依存方向は recipe → core の一方向のみ**（core → recipe は import 禁止）。これを
   **machine gate**（pytest の境界テスト + wheel 内容検査）にする — 境界は規約でなく
   テストが守る。
4. **依存管理**: recipe 側は非公開 uv プロジェクト（`tools/export-recipes/pyproject.toml`）に
   まとめ、現行の family dependency group（anima / sbv2 / siglip2 …）をそちらへ移す。core は
   path 依存（editable）で参照する。
5. **`src/` layout**: core は再編と同時に `tools/exporter/src/karume/` へ（working tree の
   偶発 import 防止・distribution 内容の可視化）。
6. **互換 shim は作らない**（未リリース — 破壊的変更が既定）。旧 import パス・旧 CLI
   コマンドはそのまま消す。
7. **ライセンス**: core wheel には upstream モデル実装のコピーを入れない、と機械的に説明
   できる状態を目標にする。pyproject に `license`（MIT）を明示。**upstream revision 単位の
   ライセンス互換確認（人間の interview）はリリース gate**（backlog release）— 本再編は
   その前提を作る作業で、確認そのものは含まない。

## 段階（各段で main を緑に保つ）

1. **境界テスト先行**（ファイルは動かさない）: 変換 core 集合に対する「patch_* / family
   import 禁止」gate を追加。gate の対象集合は段が進むたびに広げる。
2. **横断 utility の回収**: `ROPE_BUFFER_NAMES` / `assert_rope_lifted` を `patch_anima` から
   出す（モデル非依存なら core・そうでなければ `_shared/`）。
3. **`dist.py` / `modelcard.py` の engine / recipe 分割**: family を 1 個ずつ移し、毎回
   dist テスト緑で刻む。
4. **family source + tests の移動**: `tools/export-recipes/<family>/` へ（Irodori の
   sys.path 注入は family conftest へ）。
5. **core の純化**: `src/` layout 化・CLI から family command 削除。
6. **packaging / provenance**: license metadata・wheel/sdist 検査・family 別
   `THIRD_PARTY_NOTICES.md` の骨組み。
7. **CI 分離**: core = pytest + wheel build + 内容検査 / recipe = fixture ベースのテスト。
8. **docs 同期**: ADR 0037 注記・CLAUDE.md 構成節・limitations（インストール版 CLI 制約の
   解消）・backlog クローズ。

## 検討した代替案

- **B: recipe も個別 PyPI distribution 化** — version matrix と release 責任が増える。外部へ
  recipe を安定 API として出すと決めた時点で A → B へ昇格できる（今は過剰）。
- **C: 現状維持 + extras（`karume[anima]`）** — heterogeneous な上流ライセンスを 1 wheel に
  集約し続け、モデル追加のたび core の release surface が膨らむ。汎用 exporter を出す目的に
  最も合わない。

## Consequences / 非目標

- JSR 3 パッケージ（runtime / hub / models）・ルート `models/`（配布形置き場）・`examples/`
  は**本再編の対象外**（境界は既に直交している）。
- IR 適合 fixture（`packages/runtime/tests/fixtures/`）の中立位置への移動は**保留**
  （recipe 分離に必須ではない — 需要が出たら `spec/ir-v1/` 案を再訪）。
- `karume/paths.py`（repo topology 依存）は recipe 側の関心事へ — core API は Path を
  受け取る形を保つ。
- exporter README は移動後の構成で全面改稿する（G1 では最小修正に留めた）。
