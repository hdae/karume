# GLOSSARY.ja — 対話で使う日本語の用語（草案 2026-10-15・利用者のフィードバック反映済み）

> 性格: 草案。英語化（0.15 の組み直し）後のコード・docs の用語と、日本語の対話で使う言い方の対応表。
> 意味の正本は各 ADR / docs（ここは索引）。「やめる言い方」は旧い家言葉で、対話・docs・コメントのいずれでも使わない。
> ◆ = 利用者の指示で確定。△ = 提案（要確認）。

## 用語の決め方（利用者の方針 2026-10-15）

- **意味から決める。英語の直訳はしない。** 英語の用語を見て訳語を作るのではなく、その概念が何かを
  言い表す日本語を選ぶ（例: series → 「シリーズ」ではなく「中間成果物」、spike → 「試作」、
  kill criterion → 「打ち切りの基準」）。
- 英語の用語は業界の標準語を使う。日本語は「カタカナで通じる語」か「ふつうの日本語」にする。
- 演算（numerics）の話と重み（weights / quants）の話で、同じ語（実用・参照・忠実）を使い回さない。
- 1 つの概念に 1 つの語。同じターンの中で揺らさない。
- 作業の単位は **計画 → フェーズ → ステップ** の 3 段で呼ぶ。

## 演算（numerics）と重み（weights）

| 英語（コード・docs） | 日本語（対話） | 意味 | やめる言い方 | 正本 |
| --- | --- | --- | --- | --- |
| numerics profile ◆ | 数値プロファイル | 見送り（2026-10-18・ADR 0126 は deferred — 0.15.0 には無い）。`SessionOptions` の実行形の束。`practical` / `reference` の 2 値 | 層（実用層・参照層）・既定経路 | ADR 0126（deferred・0058 / 0110 の改定は発効せず） |
| practical profile ◆ | 実用プロファイル | 見送り（同上）。実装されれば既定で、形と adapter の能力で E / C / R の変種を選ぶ。近似（Q）は含まない。今の既定の実用側は manifest の実用プリセット（`session` の束） | 実用層・実用経路 | 同上 |
| reference profile ◆ | 参照プロファイル | 見送り（同上）。opt-in の knob の値として予定していた「省略値の束（自動選択なし）」。今はその状態が runtime の既定そのもの = 参照経路（次の行） | 参照層・既定経路 | 同上 |
| reference path ◆ | 参照経路 | runtime の省略値の経路（数値ノブを全て省略した Session）。sha 行の参照行・オラクル・上流一致の根。2-8 の裁定（2026-10-18・1-1）で「やめる言い方」から戻した | 参照層・既定経路・参照席 | ADR 0058 決定 2・ADR 0110 §1 |
| original weights ◆ | オリジナルの重み（忠実な重み） | Karume 側で量子化していない重み。上流の格納のまま（f32 → f32・bf16 → bf16・上流が f16 なら f16・QAT はその整数）で演算は f32 | 忠実席・参照席・忠実な席 | ADR 0128 決定 4・0131 |
| quantized weights ◆ | 量子化された重み | i8 / i4 / i2 の格納型。`quant` の明示指定で選ぶ | 実用席の重み | 同上 |
| storage dtype | 格納型 | 容器に書いた重みの形（f32 / f16 / i8 / i4 / i2） | — | glossary.md |
| activation quantization | 活性量子化（a8） | 活性を int8 に量子化して計算する実行形。品質に効くのでプリセットの中身（明示） | — | ADR 0025 / 0030 |
| oracle | オラクル | 正しさの物差しになる自前の参照実装（TS 鏡像・f64 参照） | — | ADR 0005 / 0110 |
| tolerance band △ | 許容帯 | オラクルとの差の上限（`atol + rtol·|ref|`） | 帯 | op-tolerance.ts |
| contract class (E / C / R / Q) | 契約クラス | 変種の主張の強さ（鏡像 / 合成 / 加算順 / 近似） | — | ADR 0110 決定 2 |
| variant | 変種 | あるノブの値 1 つに対応するカーネル経路 | 入居者 | ADR 0058 |
| option / knob △ | オプション（ノブ） | `SessionOptions` の個別の設定 | 席（実行の席） | ADR 0058 決定 1 |
| fallback △ | フォールバック | 形や limit が適格でないとき参照の変種へ落ちること。診断のパイプラインキーに必ず出る | 縮退・沈黙縮退 | ADR 0058 決定 3 |
| auto-selection | 自動選択 | 見送り（2026-10-18・ADR 0126 は deferred）。実用プロファイルが形 / adapter で変種を選ぶこと。選択は確認可能で、ノブで上書きできる ◆ | auto | ADR 0126（deferred） |

## 配布と資産

| 英語（コード・docs） | 日本語（対話） | 意味 | やめる言い方 | 正本 |
| --- | --- | --- | --- | --- |
| preset ◆ | プリセット | 配布形が「重みの格納型 + 実行ノブ」を束ねた名前（`i4-fast` など）。manifest の `quants` の項目 | 席・quant 席・実用席・参照席 | ADR 0074 / 0075 |
| default preset ◆ | 既定のプリセット | manifest の `defaultQuant`。モデルごとに保守者が実測（速度 / メモリ / 品質）で選んで記録する実用のプリセット（量子化を含むことがある）。オリジナルの重みのプリセットは努力目標で opt-in | 既定席・忠実席 | ADR 0128（2026-10-17 書き換え） |
| family ◆ | ファミリー | モデルの種類（anima / wan / gemma4 …）。`@karume/models/<family>` | 系列（モデルの意味で） | CLAUDE.md |
| series ◆ | 中間成果物 | exporter が書く中間成果物のディレクトリ（`outputs/series/<name>/`）。配布形の 1 つ手前 | 系列（資産の意味で）・シリーズ | assets-layout |
| distribution △ | 配布形 | HF に置く形（manifest + `krm`）。`models/` のミラー | — | ADR 0109 / container-v1 |
| container (`krm`) | コンテナ | 配布形の実体ファイル | — | container-v1 |
| export recipe | レシピ | モデル別の export 手順（`tools/export-recipes/`） | — | ADR 0065 |
| rebuild (a distribution) △ | 作り直す | 配布形を exporter から生成し直すこと | 焼き直す・焼く | release-runbook |

## 検証

| 英語（コード・docs） | 日本語（対話） | 意味 | やめる言い方 | 正本 |
| --- | --- | --- | --- | --- |
| check ◆ | チェック | 自動テストの検証項目 1 つ | 門 | ADR 0005 |
| sha256 regression check △ | sha256 の回帰チェック | 出力の成果物の sha256 を環境ごとの参照値と突き合わせる検査。決定性（同機ビット不変）を見る | sha 門・移植の門 | ADR 0106 |
| reference value / row △ | 参照値（の行） | 環境キーごとに持つ期待 sha256。書くのは `write` / `rewrite` | 行・参照行・実用行 | ADR 0106 |
| artifact ◆ | 成果物 | sha256 を採る出力ファイル（PNG / WAV / f32 safetensors） | 実物 | ADR 0106 追記決定 6 |
| environment key | 環境キー | `<runtime>-<adapter>` の綴り（`deno-…` / `chrome-…` / `node-…`） | — | ADR 0106 決定 2 |
| all-skip guard △ | 全 SKIP ガード | 検査が 1 本も走らなかった状態を赤にする検出器（GPU / 資産 / 配布形 / 参照値）。分かりにくいが頻出ではない。良い言い方があれば差し替える | 門番・参照門 | ADR 0005 / 0106 |
| lane | レーン | verify の実行単位（core / ファミリーごと） | — | ADR 0005 追記 |
| golden | golden | 上流（torch）由来の期待値。許容差つき | — | ADR 0005 |
| fake-quant golden △ | fake-quant の golden | torch に活性量子化を掛けて採った期待値 | 鏡像 golden | glossary.md |
| census | census | 変種のパイプラインキーが実際に走った本数の検査（黙って落ちるのを防ぐ） | — | ADR 0058 決定 4 |
| self A/B △ | 自機 A/B | 同じ機で参照経路（省略値）と実用プリセットの出力を比べる検査 | 自機 A/B 門 | ADR 0110 決定 5 |
| difference floor △ | 差の下限 | 自機 A/B で「1 bit 以上違う」こと（束が丸ごと効いていない状態の検出） | 床 | ADR 0110 決定 5 |
| collapse ceiling | 崩壊上限 | 自機 A/B の相対 RMS の上限 | — | ADR 0110 決定 5 |
| fault injection | 故障注入 | チェックが空振りでないことの証明（わざと壊して赤にする） | — | ADR 0110 決定 3 |
| determinism | 決定性 | 同じ機・同じ入力・同じプリセットで出力バイトが同一 | ビット同一（主張としての） | ADR 0110 決定 6 |
| upstream parity | 上流との一致 | 数値または意図の水準で上流と合っていること。上流のバグは写さない | 100% 一致 | ADR 0005 追記 2026-10-11 |
| port △ | 移植 | 上流の処理を Karume に書き写すこと | 写す | — |
| acceptance | 検収 | ステップの完了条件を満たしたことの確認 | — | — |
| decision | 裁定 | 利用者が選んだ結論（ADR に記録） | — | docs/decisions |

## 進め方

| 英語（コード・docs） | 日本語（対話） | 意味 | やめる言い方 | 備考 |
| --- | --- | --- | --- | --- |
| plan ◆ | 計画 | 目標と検証可能なゴールを持つ作業全体 | — | 計画 → フェーズ → ステップ |
| phase ◆ | フェーズ | 計画の中の、区切りのある段階 | 波・イテレーション | 同上 |
| step ◆ | ステップ | フェーズの中の順番のある工程 | 段・ステージ | 同上 |
| backlog | バックログ | 作業項目と順序の正本 | — | docs/backlog.md |
| perf ledger | 性能台帳 | 性能候補の起票・採否・打ち切りの基準 | — | docs/perf-ledger.md |
| kill criterion ◆ | 打ち切りの基準 | 候補を打ち切る条件 | kill 基準 | perf-ledger |
| spike ◆ | 試作 | 事実を集めるための使い捨ての実装 | スパイク | — |
