# 0041: manifest v2（`karume/2`）— 1 リポ複数モデルと語彙の整理

- Status: accepted（構造・用語・shards 見送りまで**ユーザー裁定済み** 2026-08-08。
  規模上限の具体値のみ本 ADR の提案値 — 実装時に最終確認）
- Date: 2026-08-08
- 関連: ADR [0038](0038-manifest-v1.md)（manifest v1 — **本 ADR が置き換える**）/
  [0039](0039-sbv2-distribution.md)（SBV2 配布形 — 決定 7「多話者の将来設計」への回答）/
  [0037](0037-karume-monorepo.md)（配布形の親決定）/
  実測根拠: [research/2026-08-08-xet-split-probe.md](../research/2026-08-08-xet-split-probe.md)

## Context

同一アーキテクチャの別重みを扱いたい実需が 3 つ揃った: ① SBV2 の JVNV 公式 4 モデル
（F1/F2/M1/M2）を HF 公開したい ② FN1〜FN10（クローズド運用）は text_encoder 319MB が
モデルごとに複製される ③ Anima にも WAI / Copycat など第三者派生が存在する。v1 は
「manifest 1 本 = HF リポ 1 個 = アーキ実装 + 重み 1 組」で、モデル軸が構造上存在しない
（唯一の識別子がリポ名）。

リポの括りは**ファミリー / シリーズ / 作者の緩い単位**とする（ユーザー裁定 — README と
ライセンス提示が 1 本で済む範囲。例: `karume-sbv2-jvnv` に JVNV 4 モデル + 共有資産）。
第三者派生は各作者が自分のファミリーリポを立てる形で自然に成立する。

前提の実測が 2 つある。**レイヤー分割は DL を改善しない**（分割は per-stream 速度を単調に
下げ、1 リクエスト固定 TTFB ~0.67s を積む。並列化は分割不要の HTTP Range で代替でき、
Range 4 並列が全構成最速 — research/2026-08-08-xet-split-probe.md）。また実配布済み
manifest は `hdae/anima-turbo` の 1 件のみで、**v1 パーサを維持する理由が実質無い**
（ユーザー裁定「v1 は捨てて OK」）。

## Decision

### 1. ルート単一 manifest（`karume/2`）。v1 パーサは持たない

manifest はリポ直下の `karume.json` 1 本のまま（固定名維持）で、`format: "karume/2"`。
**hub は v2 だけを読む**（2 形パースはしない）。`hdae/anima-turbo` は v2 で再アップする
（副次効果: 同オブジェクトの per-stream 遅延がオブジェクト固有・永続と判明しており、
再アップで解消する公算 — research 参照）。旧クライアント（JSR v0.1.0）は
「unsupported format」で loud に落ちる。

サブ manifest 案（`<model>/karume.json`）は不採用 — path 基準のねじれ（manifest の置き場と
path の基準が乖離）と、モデル列挙が機械可読にならない点で単一形に劣る（比較の経緯は
セッション記録）。

### 2. トップレベル形: `models` + `defaultModel`（必須）

```json
{
  "format": "karume/2",
  "generator": "…",
  "defaultModel": "jvnv-F1",
  "models": {
    "jvnv-F1": {
      "pipeline": "sbv2/1",
      "weights": {
        "front": {
          "f16": { "path": "jvnv-F1/front/model.f16.safetensors", "size": 0, "sha256": "…" },
          "i8": { "path": "…", "size": 0, "sha256": "…" }
        },
        "voice": { "f16": { "…": 0 }, "i8": { "…": 0 } },
        "text_encoder": { "i8": { "path": "shared/text_encoder/model.i8.safetensors", "…": 0 } }
      },
      "assets": {
        "tokenizer": { "path": "shared/tokenizer/deberta-tokenizer.json", "…": 0 },
        "symbols": { "path": "jvnv-F1/text/symbols.json", "…": 0 },
        "style_vectors": { "path": "jvnv-F1/styles/style_vectors.safetensors", "…": 0 },
        "speaker_embeddings": { "path": "jvnv-F1/speakers/speaker_embeddings.safetensors", "…": 0 }
      },
      "quants": {
        "f16": {
          "weights": { "front": "f16", "voice": "f16", "text_encoder": "i8" },
          "session": {}
        },
        "w8": { "weights": { "front": "i8", "voice": "i8", "text_encoder": "i8" }, "session": {} },
        "w8a8": { "weights": { "…": "…" }, "session": { "linearCompute": "i8a8" } }
      },
      "defaultQuant": "w8",
      "pipelineConfig": { "…": "…" }
    }
  }
}
```

- **`defaultModel` は必須**。1 モデルリポで省略可にすると、2 個目のモデル追加が既存利用者の
  暗黙既定を壊す（ユーザー指摘）。モデル未指定の `fromPretrained` は `defaultModel` を使う。
- **`pipeline` はモデル単位**。ファミリーリポに別アーキが混ざっても壊れない。
- モデルエントリの中身は v1 の manifest から format / generator を抜いた形が基本 —
  パーサも概念もほぼ流用できる。

### 3. 語彙の整理: `quants` / `dtype`、`weights` / `assets`

- **`presets` → `quants`**（`defaultPreset` → `defaultQuant`）: 実態が量子化・精度モードの
  選択そのものであるため（GGUF の quant 表から選ぶ利用者感覚とも一致）。
- **variant → `dtype`**: 意味論は v1 の時点で「格納 dtype 語彙と 1:1」（ADR 0038 §2）。
  quant（モード）と dtype（格納形）の 2 語に分けることで「quant が variant を選ぶ」という
  量子化語の重複を解消する。ラベル語彙（`f16` / `i8` / …）は据え置き。
- **components → `weights` + `assets` の分離**: v1 の components は `{file}`（単一）と
  `{variants}`（dtype 別）の 2 形が同居していた。v2 では **weights = dtype キー必須の
  テンソル容器**（i8 単体の text_encoder も `{ "i8": … }` の統一形）、**assets = quant 選択に
  依存しない無条件ファイル**（tokenizer / symbols / style_vectors / speaker_embeddings 等）に
  分け、2 形パースを消す。quants の weights 写像が weights だけを指せることも型どおりに
  検査できる。

### 4. shards は入れない。グラフはメタデータ維持

レイヤー分割は DL 性能で**逆効果**と実測で確定した（事前承認の条件「実験がうまくいったら」を
満たさない）。分割しないので「N shard : 1 グラフ」の非対称も生じず、**グラフ JSON は従来
どおり safetensors ヘッダの `__metadata__.karume_ir`**（ir-v1.md — 334MB 中 0.085% で置き場を
動かす動機も無い）。DL の高速化は取得層の **HTTP Range 並列**（cold 3.1× / warm 84 MB/s・
manifest 無関係）が受け持つ — 別トラックで設計する。

### 5. 共有プール機構は作らない（path の一致で共有）

`shared/…` の同一 path を複数モデルの weights / assets が指すだけとする。取得・キャッシュは
URL 単位で自然に重複排除され、専用の間接参照（`"shared": …` のような席）は導出可能な構造の
二重化にしかならない。manifest が数 KB 太るだけ（上限 1MiB に対し無害）。

### 6. path はリポルート基準（v1 と同じ）・許可リスト検査も据え置き

manifest は常にルート 1 本なので、v1 の path 意味論（ルート相対・SEGMENT_RE 検査）が
そのまま成立する。

### 7. 規模上限（DoS 防波堤）の再定義 — 本 ADR の提案値

manifest 全体 1MiB・`pipelineConfig` ≤ 256KiB/モデルは据え置き。**`models` ≤ 32**、
モデルあたり **weights ≤ 32・assets ≤ 32・quants ≤ 32** とする（JVNV 4 モデル・FN 10 モデルは
余裕・v1 の components ≤ 64 / presets ≤ 32 と同じ性格の値）。

### 8. 公開 API の形

```ts
using p = await Sbv2Pipeline.fromPretrained("hdae/karume-sbv2-jvnv", {
  model: "jvnv-M1",
  quant: "w8",
});
// model 省略 = defaultModel / quant 省略 = defaultQuant
// 未知の model / quant は fail loudly で「利用可能な一覧」を添えて落とす（v2 で初めて列挙可能になる）
```

`HubRepoRef` は不変（subpath 軸は持たない）。`resolveFiles` は `(manifest, { model, quant })`
へ。JSR は破壊的変更として 0.2.0（3 パッケージ ロックステップ）。

### 9. ローカル規約は維持

`models/` の「1 ディレクトリ = 1 HF リポ」は不変（中が多段になるだけ）。`verify_dist` の
宣言外ファイル検査は「モデル別サブツリー + `shared/` + 直下 `karume.json` / `README.md`」へ
更新する。exporter は `--model` 軸（現 `SBV2_MODEL_NAME` 等の定数の引数化）とファミリー
組み立て（共有ファイルは同 sha256 なら 1 回だけ置く）を得る。

## Consequences

- 破壊的変更が 1 回で済む（単一 manifest 化 + 語彙整理 + weights/assets 分離を同時に）。
  実配布 1 件（anima-turbo）の再アップのみが移行コストで、遅延オブジェクト解消の副益つき。
- モデル列挙が機械可読になり、エラーメッセージと将来の Web UI がそのまま恩恵を受ける。
- 積み残し（本 ADR のスコープ外・別トラック）: ① HTTP Range 並列取得（hub / fetch-cache）
  ② fetchAssets の 2 相化（prefetch → 遅延 materialize — スマホ RAM の根治側）③ 検証済み
  マーカー（fetch-cache 実装済み）を hub 既定にするかの ADR 0038 §5 相当の改訂
  ④ ファミリーリポのライセンス表示（JVNV + DeBERTa の同居 — 公開波で確認）。

## 実装順（提案）

1. hub: v2 パーサ（v1 置換）+ `resolveFiles` の model / quant 軸
2. exporter: `--model` 軸・ファミリー組み立て・`verify_dist` 更新・モデルカード追随
3. models: `fromPretrained` オプション貫通（anima / sbv2 両パイプライン）
4. anima-turbo の v2 再組み立て + 再アップ、JVNV 4 モデルの変換 → `karume-sbv2-jvnv` 組み立て
   （公開波 — ライセンス確認込み）

## 追記（2026-08-09 — 実装時の追加裁定・いずれもユーザー裁定）

- **リポ名は `karume-` prefix**: HF org は作らない（現状規模では不要・後から org へ移譲可能）
  代わりに、配布リポ名で名前空間を切る。`hdae/anima-turbo` → **`hdae/karume-anima-turbo`**、
  SBV2 は **`karume-sbv2-jvnv`**（HF 公開・JVNV 4 モデル）と **`karume-sbv2-fn`**（クローズド・
  FN1〜FN10 の 10 モデル・defaultModel = FN1）。モデル名（`anima-turbo` / `FN4` / `jvnv-F1`…）は
  不変で、リポ名だけが変わる。系列名（`outputs/series/sbv2-<model>-*`）には掛からない。
- **配布形の配置はハードリンク禁止・常に独立コピー**: 系列の書き手は既存ファイルを truncate で
  上書きするため、リンク共有した配布形は系列の再 export で黙って中身が変わり、manifest の
  sha256 と現物が食い違う（`verify_dist` は sha256 を採り直さない設計なので沈黙する）。配布形は
  系列から独立した自己完結スナップショットとする。export 段が `models/` へ直接書く案は不採用 —
  1 リポ = N export 出力の合流（共有畳み込み・rope 同一検査・manifest は全部揃ってからしか
  書けない）と、re-export ゼロでの組み替え（本 v2 移行がその実例）を失うため。
- **ファミリー組み立ての defaultModel は最初の `--model`**（専用フラグは置かない）。
