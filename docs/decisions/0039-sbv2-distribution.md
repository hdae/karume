# 0039: SBV2 の配布形（`sbv2/1`）と多話者の将来設計

- Status: accepted（聴感ゲートまで**ユーザー受理** 2026-08-07「w8a8 までほぼ同じ音声に聞こえる /
  f16 は全く劣化を感じない」）
- Date: 2026-08-07
- 関連: ADR [0038](0038-manifest-v1.md)（manifest v1 — **本 ADR はその SBV2 例を実装で確定させる**）/
  [0013](0013-sbv2-chain-export.md)（チェーンの emit ターゲット）/
  [0026](0026-w8a8-deberta-deployment.md)（DeBERTa i8 の受理根拠）/
  [0027](0027-sbv2-f16-series.md)・[0029](0029-sbv2-i8-series-and-quant-quality.md)（f16 / i8 系列）/
  [0037](0037-karume-monorepo.md)（配布形の親決定）/ [0008](0008-public-api.md)（公開面の薄さ）

## Context

SBV2 の TS 側実装が本リポへ移り、実重みで e2e が全緑になった（`packages/runtime/tests/e2e_sbv2_test.ts`
= 3 系列 × 5 ターゲット × 5 ケースが実 GPU で 78 passed）。配布形を組む段になったが、ADR 0038 が
書いた SBV2 の manifest 例は自ら「本家慣例値のサンプル — export 時に実測系列から確定する」と
断っている（0038 の SBV2 例の注記）。ここで確定させる。

対象の実重みは HF `rufflet17/voice_models` の `FN/FN4/`（Style-Bert-VITS2 JP-Extra・
`version: 2.6.1-JP-Extra`・`n_speakers: 1`・**4 スタイル**・44.1kHz）。ユーザー裁定により「公開者が
改変自由としているものを引用し、Anima と同じ形で公開する」方針。

**この ckpt は先行実装で使っていた別の ckpt と state_dict がテンソル 1165 本・shape とも完全一致**
（実測）だったため、`style-bert-vits2==2.5.0` のピンを動かさずに読めた。一方で**スタイルの構成は
別物**（4 対 7）で、この差が決定 3 の動機になっている。

## Decision

### 1. components は実行に要るものだけ（3 グラフ + 4 資産）

ADR 0038 の例は emit ターゲット 6 本（`dp` / `front` / `flow` / `dec` / `voice` + text_encoder）を
並べていたが、**実行に要るのは 3 本**である。`voice` は flow + dec の融合、`front` は enc_p + dp +
sdp を含む（ADR 0013）。`dp` / `flow` / `dec` は golden 検証専用なので配布しない — 載せると
flow 155MB + dec 59MB がそのまま無駄になる。

| component            | 形                    | 実サイズ                   |
| -------------------- | --------------------- | -------------------------- |
| `text_encoder`       | `{file}`（i8 のみ）   | 334,545,336 B              |
| `front`              | `variants: {f16, i8}` | 17,956,780 / 10,326,768 B  |
| `voice`              | `variants: {f16, i8}` | 109,371,348 / 55,524,872 B |
| `tokenizer`          | `{file}`              | 122,035 B                  |
| `symbols`            | `{file}`              | 1,574 B                    |
| `style_vectors`      | `{file}`              | 4,184 B                    |
| `speaker_embeddings` | `{file}`              | 2,136 B                    |

配布形 `models/sbv2-FN4/` は 11 ファイル・504MiB。取得量は preset `f16` で 462,003,393 B、
`w8` / `w8a8` で 400,526,905 B。

### 2. `text_encoder` は i8 単体（variant を持たない）

`export_deberta.py` は f32 と i8 しか持たず、f32 は 1.32GB で配布に載らない。i8（319MiB）は
**ADR 0026 が聴感ゲート込みでユーザー受理済み**（hidden[-3] の SNR 20.7〜23.3dB）なので、f16 を
新設する動機が無い。

結果として preset `f16` は「生成ネットだけ f16・text_encoder は i8」という非対称になるが、これは
ADR 0038 が `weights` を単一文字列でなく `Record<component, label>` にした動機そのもの
（コンポーネント別の量子化混在）。`{file}` 形の component は preset の `weights` に**現れてはならない**
（hub が `ManifestReferenceError` で落とす）。

### 3. `styles` / `speakers` は資産から導出する（MUST — 表を焼き込まない）

`pipelineConfig.styles` / `speakers` は `config.json` の `data.style2id` / `data.spk2id` をそのまま
写す。既定ノブも `style_bert_vits2.constants` から引く（`DEFAULT_STYLE` / `DEFAULT_SDP_RATIO` …）。

**表を焼き込んではならない。** ckpt が変わればスタイルの名前も並びも変わり（FN4 は
Neutral / high / low / NSFW の 4、先行実装で使っていた別 ckpt は Neutral / Angry / Disgust / … の 7）、
写した表を配ると **shape は合ったまま別のスタイルの声が出る**。組み立て時に `defaults.style ∈
keys(styles)` / `defaults.speaker ∈ keys(speakers)` も検査する（存在しないスタイル名を既定に据えた
配布形は、起動して初めて落ちる）。

### 4. スタイルと話者は「表を配って実行時に行を引く」

`style_vectors`（`[スタイル数, 256]`）と `speaker_embeddings`（`[話者数, gin_channels]`）を
safetensors で配り、パイプラインが名前 → 行番号 → ベクトルと解決する。スタイルの混合式は
`mean + (picked - mean) * weight`（**`mean` は行 0** — `TTSModel.__get_style_vector` と同式）。

**話者埋め込みは最初に組んだ配布形で欠落していた。** `front` と `voice` はどちらも `g[1,512,1]` を
グラフ入力に取るのに出所が無く、デモ経路が `assets.safetensors` に焼いた `g` を渡していたため、
**配布形を hub 経由で読むまで露見しなかった**。選択済みの 1 本を焼く形（当初案）では
`pipelineConfig.speakers` が名前 → 行を持つのに話者だけ選べない非対称になるので、`style_vectors`
と同じ表配布へ揃えた。

門（組み立て時に `DistError` で止める）:

- `style_vectors` の行数 == `data.num_styles` == `len(data.style2id)`
- `speaker_embeddings` の行数 == `data.n_speakers` == `len(data.spk2id)`、列数 == `model.gin_channels`
- `styles` / `speakers` の値が `0..rows-1` の順列であること（manifest の parse 時にも検査）

**ID は行番号そのものなので、ずれてもロードも実行も通り、別のスタイル・別の話者の声が出るだけで
沈黙する。** 行数を合わせる以外に検出手段が無いため、この 3 つを門にしている。

### 5. preset は `f16` / `w8` / `w8a8`、既定は `w8`

`w8a8` だけ `session.linearCompute = "i8a8"`。SBV2 は 5 ターゲットとも conv1d が 86〜90% を占め
linear は実質 0 GFLOP（ADR 0025 決定⑤）なので、`w8a8` は速度のためではなく**選択肢として**置く。
既定を `w8` にしたのは取得量が最小（400MB）で、聴感がこれを許すため（下の Consequences）。

### 6. 日本語辞書は配布形に載せない（**暫定** — リファクタで再検討）

`@hdae/yomi` の辞書はモデル資産ではなく yomi のバージョンに結びつくため、配布形に混ぜると
「モデルは不変なのに yomi 更新で再アップ」が起きる。かつ多話者では同じ辞書が話者ぶん複製される。

一方で取得経路は**現状 `packages/models` が `fetchDictionaryBytes` を直接呼ぶ**形にした
（ユーザー裁定 2026-08-07「一旦テスト版として組んでいるだけ、詳細はリファクタリング時に検討」）。
**これは ADR 0038 が資産取得を hub に一元化した設計との緊張**であり、`packages/models` が hub を
経由せずネットワークへ出る唯一の経路である。`Sbv2PipelineOptions.dictionary` の注入席を置いてあり、
渡された場合は 1 度も取得に出ない。

**関連する未解決の課題**（ユーザー指摘・2026-08-07）: `packages/models/deno.json` に `@hdae/yomi` を
足すと、**anima 目的で `@karume/models` を入れた利用者にも yomi が降る**。全モジュール副作用ゼロに
よる tree-shaking はコードを削るが、依存宣言そのものは削れない。ファミリ別パッケージ分割
（`@karume/models-sbv2` 等）が候補で、ADR 0037 §4 の「barrel + サブパス両建て」を見直す材料になる。

### 7. 多話者は今回やらない（選択肢だけ残す）

FN4 は 1 話者なので、`speakers` map は 1 エントリで閉じている。10 話者を 1 リポジトリにまとめたい
という要求（ユーザー・2026-08-07）に対しては、**既存 variant の流用では解けない**:

- variant ラベルは「格納 dtype 語彙と 1 対 1」を優先する規約（ADR 0038）で、話者は別の軸
- 話者をラベルに載せると preset が話者 × dtype の直積になり、`presets ≤ 32` / `components ≤ 64` の
  上限に 10 話者で当たる

選択肢（どちらも未着手）:

- **(a) リポ内サブ manifest**: `FN4/karume.json` … を並べ、hub が `repo + subpath` で引く。manifest の
  形は無傷で、`text_encoder`（319MB・全話者で同一）を共有パスに置ける
- **(b) `pipelineConfig.speakers` の実体化**: 決定 4 の表配布を多話者へ広げる。ただし FN1〜FN10 は
  **別 ckpt** なので `front` / `voice` の重みも話者ぶん要り、components 分割が避けられない

`text_encoder` が配布形の 63% を占めることが、多話者を 1 リポにまとめる技術的な動機を強めている
（1 話者 1 リポだと 319MB が話者ぶん複製される）。

## Consequences

### 検証 — 段 1 経路と段 2 経路の WAV がビット一致

配布形 + `w8` preset で出した WAV と、`outputs/series/` を直接読む段 1 の経路（i8 構成）で出した
WAV が **sha256 完全一致**（`a82f72e2c18956ec725a3f692182e8c9a7dad4011e760dab9fb3d051653db2f4`）。
配布形の i8 資産 3 本が系列の実体と sha256 同一であることも確認済みなので、「同じ資産・同じ計算」が
配布形の組み立てとパイプライン化を通しても保たれている。dump の 11 テンソルもバイト一致。

torch 参照との突合（f32 構成・`sbv2_demo.py reference`）: **`w_ceil` 整数完全一致**・
`length_match: true`・波形 maxAbs **5.16e-5**（値域上端 0.396）・rmse 3.19e-6。`maxRel` は零交差で
発散するので判定に使わない（golden E2E の dec / voice tolerance と同じ事情）。

ゲート: `deno task verify` 687 passed / 0 failed / 4 ignored、pytest 2030 passed / 2 skipped。

### 量子化 — 測定値と聴感の乖離

`measure_quant_sbv2.py`（torch シミュレーション・同一発話 / 同一乱数）の実測:

| 構成  | SNR vs f32 | LSD（主指標） | 発話長（自前 `w_ceil`） |
| ----- | ---------: | ------------: | ----------------------- |
| f16   |   35.87 dB |       0.61 dB | 一致（229/229）         |
| w8    |    9.01 dB |       2.99 dB | 割れ（229/229）         |
| w8a16 |    8.98 dB |       2.99 dB | 割れ（229/229）         |
| w8a8  |    7.83 dB |       4.81 dB | **割れ（228/229）**     |

直交分解でも ADR 0029 と同じ傾向（劣化の主因は front 側 — `w8-front-only` 7.22dB /
`w8-voice-only` 15.83dB）。

**ユーザー聴感（2026-08-07）は「w8a8 までほぼ同じ音声に聞こえる」「f16 は全く劣化を感じない」。**
LSD 4.81dB でも通常再生では聞き分けられないという実測で、**数値だけでは配布 preset を決められない**
ことの記録として残す。ADR 0029 が f16 を「この表のスケールの校正点」に据えた前提は、この ckpt でも
成立した（ただし f16 の SNR は 35.87dB で、別 ckpt の 40.5dB より 4.6dB 低い）。

### ライセンス — 配布形に条件の違うものが同居する（**公開前に裁定が要る**）

実地確認（2026-08-07）:

- **声の重み** `rufflet17/voice_models`: **HF 上にライセンス宣言も README も無い**（API の `tags` は
  `region:us` のみ・`cardData` なし）。「公開者が改変自由としている」はユーザーからの情報で、
  リポジトリ上に根拠が無い。モデルカードは `license: other` + リポジトリへの `license_link` とした。
- **text_encoder** `ku-nlp/deberta-v2-large-japanese-char-wwm`: **cc-by-sa-4.0（ShareAlike）**。
  配布形の 63%（319MiB / 504MiB）を占め、i8 へ変換して**再配布**している。

モデルカード本文には両方の帰属を書いたが、**ShareAlike が配布形全体に及ぶかは法務判断**であり、
frontmatter の単一 `license` 欄でどう表すかを含めて公開前にユーザーの裁定が要る。決定 7 の (a)
（text_encoder を別リポへ分ける）を採ると、この同居自体が解消する副次効果がある。

### 積み残し

- `symbols.json` の `defaults` と `pipelineConfig.defaults` が配布形に**二重に並ぶ**。パイプラインは
  後者だけを読み（`parseJpExtraRules` の `defaults` は任意に降格済み）、組み立て時に両者の一致を
  検査して食い違えば止める門を置いてあるが、配布形から落とすのが根治。
- `symbols.json` には exporter が焼いた `style` / `speaker` の**選択結果**も残っており、こちらは門の
  対象外（決定 4 で表配布にしたため、選択結果は記録以上の意味を持たない）。
- DeBERTa の実重み e2e（`e2e_deberta_test.ts`）は未移植のまま。`export_deberta.py` の `main()` も
  `argv` を取らず、`karume` CLI から届かない。
