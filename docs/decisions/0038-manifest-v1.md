# 0038: 配布 manifest v1 と hub の取得層

- Status: superseded by [0041](0041-manifest-v2.md)（2026-08-08 — manifest v1 は v2
  〈`karume/2`〉が全面置換。v1 パーサは持たない。本文は「なぜ現在形へ移ったか」を説明する
  当時の記録として保存 — §4 の Anima 配布形〈S 形のみ + 常時 tiling〉と §5 の取得層
  〈`@hdae/fetch-cache`〉は引き続き現行の正本）
- Date: 2026-08-05（pre-mortem 3 レンズ・44 指摘を反映した改訂版）
- 関連: ADR [0037](0037-karume-monorepo.md)（配布形の親決定）/
  [0033](0033-vae-fixed-tile-decode.md)（タイル VAE）/
  [0034](0034-dit-dynamic-tokens.md)（S 形 DiT）/ [docs/ir-v1.md](../ir-v1.md)（コンテナ規約）

## Context

`@karume/hub` が HF リポジトリから資産を解決・取得するための機械可読契約を定める。設計材料は
2026-08-05 の recon（資産棚卸し・HF 配布習慣の実地確認）・同日の実測（下の実測節）・実装前の
多レンズ pre-mortem（仕様完全性 / 敵対・堅牢性 / 進化・適合。記録は
[research](../research/2026-08-05-manifest-premortem.md)）。

先行実装は「組み合わせの可否表を持たない — 資産の実在が正」を MUST にしていた（表は現物と
乖離して腐るから）。HF 配布では数 GB を撃ち終わるまで実在が分からないため表が要る。この反転を
正当化する鍵は、**manifest が手書きの表ではなく exporter が資産から導出する生成物**（CLI 化で
自動出力）であること — 資産と原子的に同期するので「表が古くて正しい組み合わせを弾く」失敗
様式は構造的に起きない。stale 表の実害（資産表と現物の食い違い）は実測済みで、手書き manifest
を禁じ生成物だけを許す根拠になっている。

**manifest は HF 上に配布されリポ外に残る最初のデータ形式**であり、「未リリースだから破壊的
変更は自由」は適用できない（他人のリポは migrate できない）。以下の互換規則はその前提で書く。

## Decision

### 1. エンベロープ — 置き場・形・検証

manifest はリポジトリ直下の固定名 **`karume.json`**。

```jsonc
{
  "format": "karume/1",
  "generator": "karume/0.1.0",
  "pipeline": "anima/1",
  "components": {/* 2 */},
  "presets": {/* 3 */},
  "defaultPreset": "w8a8-s16",
  "pipelineConfig": {/* パイプライン所有 — hub は素通し */}
}
```

- `format` は取り違え検出器 + 互換の主戦場。**繰り上げ規則**: 必須フィールドの削除・意味変更・
  必須フィールドの新設 = `karume/2`。optional 拡張点（§7 で明示列挙した席）の追加と列挙値の
  追加 = 据え置き。minor は持たない。hub は未知 major を fail loudly で拒否する。
- `pipeline` はパイプライン実装の契約名 + その major（`anima/1`）。**語彙は karume が所有**
  （transformers の `model_type` / diffusers の `_class_name` と同じ運用）。モデル作者名は
  入れない（同じ実装で動く第三者 fine-tune も同じ値を書く）。`pipelineConfig` スキーマの
  破壊的変更で major を繰り上げ、models 実装は対応 major を宣言し未知 major を fail loudly。
  第三者製パイプライン実装の名前空間は v1 では作らない（§7）。
- `generator` は焼いたツールの版（障害報告の照合用・実行意味論なし）。
- **エンベロープ内の未知キーは fail loudly で拒否**する。additive 進化が許されるのは
  `pipelineConfig` の内側と §7 の明示列挙席だけ — 「旧 hub が新 manifest を旧解釈で黙って
  実行する」経路を残さない。
- `pipelineConfig` は必須（空でも `{}` を明示 — 素通し先の実装が有無分岐を持たずに済む）。
  スキーマは `@karume/models` の各パイプライン実装が所有・検証する。パイプライン間の差
  （SBV2 に resolution が無い等）は構造的にここへ隔離される。
- **parse 時の構造検査（hub・全て fail loudly）**: JSON 不正は `ManifestFormatError` に包んで
  再送出（重複キーは `JSON.parse` の後勝ちを受理 — 検出は exporter の責務）。マップのキーに
  `__proto__` / `constructor` / `prototype` が現れたら拒否し、manifest 由来のマップは全て
  `Object.hasOwn` 経由でのみ引く（横断不変条件の適用先）。`session` 等の合成はスプレッドのみ
  （`Object.assign` 禁止）。`components` / `presets` は非空・`defaultPreset ∈ keys(presets)`。
- **規模上限（DoS 防波堤・数値で焼く）**: manifest 本体は取得中に 1MiB 超過で abort。
  `components` ≤ 64・`presets` ≤ 32・`pipelineConfig` の JSON バイト長 ≤ 256KiB。

### 2. components — ファイル表

```jsonc
"components": {
  "text_encoder": { "file": { "path": "text_encoder/model.safetensors", "size": 1194328064, "sha256": "9f…（64 桁）" } },
  "transformer": {
    "variants": {
      "f16": {
        "file": { "path": "transformer/model.f16.safetensors", "size": 3917925796, "sha256": "…" },
        "extras": { "rope_base": { "path": "transformer/rope_base.safetensors", "size": 4210688, "sha256": "…" } }
      },
      "i8": { "file": {/* … */}, "extras": { "rope_base": {/* 同一 path 可 — 下の重複規則 */} } }
    }
  }
}
```

- コンポーネントの**基本形は `{file, extras?}`**。variant を持つ場合は
  `{variants: {label: <基本形>}}`。**`file` と `variants` は排他必須**（両方あり / 両方なしは
  `ManifestFormatError`）。extras が基本形に付けられるため、付帯資産が増えても形の破壊的
  変更は不要。variant に依存しない付帯資産は独立コンポーネントとして書いてもよい
  （どちらを出すかは exporter の生成規則が一意に決める）。
- variant ラベルは**フラットな文字列**で、意味はパイプライン知識 — hub は文字列で引くだけ。
  **ラベル語彙は runtime の格納 dtype 語彙と 1 対 1 を優先**し、エコシステムの `fp16` 綴りには
  合わせない（README の variant 表に対応注記を出す — 意図的な選択）。
- ファイル参照は必ず `{path, size, sha256}` の 3 点セットで、**hub は 3 点全ての存在と形式を
  parse 時に検査**し、1 つでも欠ければ fetch を開始せず拒否する（欠落を許すと取得層の
  validate フック自体が付かず、無検証が沈黙で常態化するため）。**export 時に資産から導出**
  （手書き禁止）。
  - `sha256`: `/^[0-9a-f]{64}$/`（小文字 hex 64 桁。大文字・空白は正規化せず**拒否**）。
  - `size`: `Number.isSafeInteger(size) && 0 < size && size <= 16 * 2**30`。**Hub 上の保存形
    raw のバイト数**。取得時は `AbortController` を渡し、受信バイトが `size` を超えた時点・
    `content-length` が `size` と食い違った時点で abort する（全量読了後の判定に頼らない）。
  - 用途: キャッシュ検証（self-heal）・進捗総量の事前提示・破損検出・ディスク事前チェック。
- **path は許可リストで検査**（URL 組み立て前の生文字列に対して）: `/` 区切り・各セグメントが
  `/^[A-Za-z0-9._-]+$/` に一致・`.` / `..` そのもの及び先頭 `.` のセグメントは拒否・
  空セグメント（先頭/末尾/連続スラッシュ）拒否。禁止列挙（`..`・絶対パス・URL…）ではなく
  許可リストにする — 取得層はセグメントを percent-encode してもドットを透過するため、列挙の
  抜けがそのまま SHA ピン外への traversal になる。違反は `ManifestPathError`。
  規約は `<component>/model[.<variant>].safetensors`・付帯資産は明示エントリ。
- **同一 `path` の重複参照は合法だが、`{size, sha256}` の完全一致を parse 時に検査**
  （不一致は拒否 — 片方だけ別ハッシュの矛盾 manifest は self-heal を振動させ、正しい
  キャッシュを evict し続ける）。取得と進捗総量は **path で一意化**する。
- 「このリポに何があるか」は components 表が唯一の宣言。受理集合（解像度の刻み・上下限等）は
  manifest に**書かない** — アーキ定数はパイプライン実装、S 上限は dyn グラフの `Dim` 宣言、
  実在はこの表、と導出元を一意に保つ（二重保持の禁止）。

### 3. presets — 名前付き実行構成

```jsonc
"presets": {
  "w8a8-s16": {
    "weights": { "transformer": "i8" },
    "session": { "linearCompute": "i8a8", "attentionCompute": "i8a8", "attentionScoreStorage": "f16" }
  },
  "f16-c16": {
    "weights": { "transformer": "f16" },
    "session": { "linearCompute": "f16", "attentionCompute": "f16" },
    "gpuFeatures": { "shaderF16": true }
  }
}
```

- **`weights` は `Record<component, label>` の完全写像**: `variants` を持つ全コンポーネントを
  キーに持ち、過不足はどちらも `ManifestReferenceError`（`{file}` 形コンポーネントが現れても
  エラー）。hub は parse 時に全 preset × 全ラベルの解決可能性を検査する — 「DL 開始後に
  初めて欠けが分かる」を許さない（manifest 導入の目的そのもの）。単一文字列形は作らない
  （コンポーネント別の量子化混在は transformers.js の per-module dtype にも先行実装の
  「VAE は i8 化しない」判断にも実在する需要で、後から map 化すると配布済み manifest が
  割れる）。
- **`session` は manifest 所有の語彙**であり runtime 型の素通しではない。v1 のキーは
  `linearCompute` / `attentionCompute` / `attentionScoreStorage` の 3 つに固定し、hub が
  **キーも値も allowlist で検査**する（未知キー・未知値は fail loudly。runtime の
  `SessionOptions` への写像は hub/models が明示的に行う）。理由: ①runtime は未知オプション
  キーを黙って無視するため、綴り違いは「s16 が名前だけになる」沈黙劣化になる ②素通しは配布済み
  manifest を runtime 内部の綴りに釘付けする ③`SessionOptions` には `submitPolicy`（TDR
  予算 = ホスト政策）が含まれ、配布者に書かせてはならない。
- `gpuFeatures` は `AcquireGpuOptions` の部分集合。hub は**キーの allowlist 検査のみ**行う
  （v1 は `shaderF16` のみ・未知キーは拒否）。解釈と device 生成は models 側の責務
  （Session より前の `acquireGpu` 層に効くため）。
- `defaultPreset` は焼く —「悩まず 1 回動く」は fromPretrained の核価値で、配布者の推奨既定は
  データとして正当。
- **自動最適選択は manifest に入れない**: ①「最適」は速度/品質/VRAM のどれを取るかという
  アプリの政策で、manifest は事実だけを持つ ②WebGPU に空き VRAM を照会する API が無く、
  VRAM 閾値判定は土台が無い ③hard 要件は `gpuFeatures` 宣言で既に機械可読。将来 models 側に
  `selectPreset(capabilities)` ヘルパを置く席は空いている。
- 組み合わせ禁則（i8a8 × f16 資産等）の扱い: **禁則の組が manifest に現れないことは exporter
  の生成が保証し、最終門は runtime の fail loudly が持つ**。preset レコードが閉じているため
  利用者が組を作る余地は無いが、「JSON として書けない」わけではない — 門の所在を誤解しない
  こと。

### 4. Anima の配布形（実測に基づく）

- **transformer は S 形（可変）のみ・variant 名に "dyn" と書かない**。固定形 512/1024 は S 形
  実装前の段階的資産（ランタイム対応が段階的だった歴史的事情）であり配布しない — 開発側の
  検証アンカー（S 形 ≡ 固定形の一致門の比較対象）としてローカルにだけ残る。根拠: 実測 A/B
  （下）で per-step +0.4%（ノイズ域）・出力 PNG は sha256 完全一致。
- **VAE decoder は 1 ファイル + 常時 tiling**。512 形と 1024 形の重みペイロードは sha256 同一
  （実測）— 元々 1 モデルであり、64×64 latent タイル形 1 本で全解像度をカバーする（512 生成は
  縮退 1 タイル ≡ 非タイルのビット同一 — ADR 0033 の門）。1024 非タイル（VAE 1.5s vs 3.4s の
  +1.9s 短縮）が欲しくなったら components に additive に足せる。
- 今後の新アーキテクチャは**最初から可変（記号次元）で実装する** — 固定形を経由する理由は
  もう無い。

### 5. hub の取得層 — `@hdae/fetch-cache`（^0.3）の採用と接続契約

実行時依存ゼロ（fetch / caches / crypto.subtle のみ）の URL キー DL キャッシュ。適合点:
`./hf` 層の mutable ref → commit SHA 解決 + SHA ピン URL キャッシュ / ファイル仕様
`{path, sha256, expectedBytes}` が manifest の 3 点セットを直接消費 / validate のキャッシュ
読出し適用 + self-heal / quota 失敗の素 fetch 縮退 / single-flight / `onProgress` fan-out /
正確サイズの `Uint8Array<ArrayBuffer>` 返却（ソース確認済み）。接続契約:

- **revision はセッションあたり 1 回だけ解決**する: hub は最初に commit SHA（40 桁）を確定し、
  `karume.json` の取得も全ファイル取得も**同一 SHA を明示 revision に固定**して行う（可変 ref
  のまま複数回解決すると、途中でリポが更新された場合に manifest と重みが別コミットから来る）。
  解決済み SHA は返り値・エラー・診断に必ず載せる。利用側は `revision` に SHA を渡せる
  （その場合は解決リクエスト自体が発生せず、完全キャッシュ時は**オフラインで起動**できる。
  可変 ref 使用時の解決失敗はオフライン不可として明示的に報告する）。
- **`resolve(preset?: string)` の返り値は `Record<key, FileRef>`**（key は `"<component>"` と
  `"<component>.<extra>"`・`FileRef = {path, size, sha256}`・path で一意化済み）—
  `fetchHfFiles` の files 引数へそのまま渡せる形。preset 省略は `defaultPreset`。
- **キャッシュ名前空間は hub が必ず明示**する（`karume/1`）。ライブラリ既定名（他コードと
  共有）は使わない。`Authorization` を伴う取得は **credential ごと**の別名前空間
  （`karume/1:auth:<Authorization 値の sha256 先頭 16 hex>` — 2026-08-13 追記参照）へ隔離し、
  gated 資産が無認証経路にも**別 credential の写し**にもキャッシュヒットで供されないことを
  契約とする。`hubUrl` は manifest からは与えられない（アプリが明示指定した場合のみ有効）。
- **進捗とキャンセル**: 進捗総量は content-length ではなく manifest の `size` 合計（path
  一意化後）から算出。`AbortSignal` を全取得へ透過する。キャッシュヒットの sha256 検証中
  （3.7GB で数秒）は `verifying` フェーズとして進捗イベントに出す（無言のハングにしない）。
  同一プロセスの並行取得では single-flight の合流者に signal が効かない（キャンセル粒度が
  leader 単位）— `docs/limitations.md` に起票する既知制約。
- **同時取得数は 4 に制限**（`fetchHfFiles` の一括発火に任せず hub 側でバッチ — 数十
  コンポーネントの manifest で接続と RAM が破綻しないため）。
- `openModel` への受け渡しは `bytes.buffer` を渡す前に
  `byteOffset === 0 && byteLength === buffer.byteLength` を assert する（不成立は例外。
  slice はしない — RAM ピーク倍増の MUST を踏襲）。
- **エラーの形**: `ManifestFormatError`（JSON / 構造 / 規模）/ `ManifestReferenceError`
  （defaultPreset・weights 写像・未知キー）/ `ManifestPathError` / `IntegrityError
  {repo, revisionSha, path, expected, actual, source: "cache" | "network"}`。取得層由来
  （404・認証）は文脈（repo・SHA・path）を付けて透過。**全エラーに利用可能な preset /
  variant ラベル一覧を含める**（GGUF 利用者が README の quant 表で得ている情報の代替）。
  self-heal は 1 往復まで（再取得後の不一致は再々取得しない — 取得層の実挙動と一致）。
  `onCacheError`（quota 失敗等）は console.warn に任せず**アプリへ届く診断イベント**として
  公開する（「毎起動フル再 DL が黙って常態化」を見えるようにする —
  `navigator.storage.persist()` 案内の誘発点）。
- 設計点: sha256 検証は**キャッシュヒット毎にも全量走る**（3.7GB で起動あたり数秒）。v1 の
  既定は整合性優先で常時検証とし、opt-down（size のみ）は必要が実測されてから足す。既知の
  コスト: streaming 組み立てとハッシュ時コピーで**一時的に約 2 倍の RAM**（`openModel` が
  全量 ArrayBuffer を要求する契約に由来）。
- 依存の扱い: 横断不変条件「ランタイム依存は Web 標準 API のみ」は「**Web 標準 API のみで
  構成された依存パッケージに限る**」と読む。fetch-cache はこれを満たす（同パッケージ自身が
  同じ MUST を掲げる）。`CLAUDE.md` の文言をこの形に更新した。

### 6. 検討して落としたもの

- **minRuntime（互換域宣言）**: 未対応 op はグラフ parse が、未知の preset キー・未知 major は
  §1/§3 の allowlist 検査が、それぞれ fail loudly で受け止める（「古い実装 × 新しい
  manifest」の沈黙劣化はこの 2 つが防波堤 — 版レンジ機構は複雑さに見合わない）。
- **license / 由来を manifest に**: モデルカード（README frontmatter）の責務。機械可読側は
  最小に保つ。
- **pipelineConfig の独立版フィールド**: `pipeline` の値に major を綴る（`anima/1`）ことで
  代替 — フィールド追加ゼロで「古いリポ × 新しい models」の食い違いを検出できる。
- **VRAM 閾値の自動 preset 判定**: §3。
- **固定形グラフの配布**: §4。
- **`weights` の単一文字列形**: §3（最初から map — 配布後の形変更は他人のリポを壊す）。

### 7. 将来（optional 拡張席として明示列挙 — ここに無い追加は `karume/2`）

- ファイル参照の optional `repo` / `revision`（別リポの不変コンポーネント参照 — 第三者
  fine-tune が text_encoder 1.5GB 級を再アップロードせずに済む形。§1 の未知キー拒否と対で、
  旧 hub が別リポ参照を自リポとして黙って解決する事故は起きない）。
- preset の optional `requiredLimits`（`maxBufferSize` 等 — v1 の DL 前チェックは feature 軸
  のみで、limits 不足は DL 後の fail loudly になる制約を Consequences に記す）。
- VAE の記号 H/W 化（回収は中間解像度の非タイル +1.9s のみ — 大解像度は VRAM 壁で非タイル
  自体が不可）/ `selectPreset()` ヘルパ / 第三者パイプライン実装の名前空間 / 1024 VAE の
  additive 追加。

## 実測（2026-08-05）

1. **S 形 vs 固定形の直接 A/B**（運用形・1024²・turbo 8 step・seed 42・クールダウン規約・
   計測 off の素の時間）: per-step 1,493 vs 1,499ms（+0.4%・ノイズ域）・8 step 計 11.9 vs
   12.0s・transformer ロード 2.7 vs 3.4s（+0.7s・一回きり）・生成全体 19.0 vs 19.7s・
   **出力 PNG は sha256 完全一致**（`daaefef8…`）。
2. **VAE 512 形 / 1024 形の重みペイロード sha256 同一**（`4553582c…`・tensor 表 107 本同一・
   ファイル差分はヘッダの形状記述 440B のみ）。グラフ census: conv2d 37・attention 1
   （mid-block）・ほか elementwise/reshape 系。
3. **fetch-cache 0.3.1 のソース適合確認**（JSR 公開ソース読了）: 正確サイズ組み立て・validate
   のキャッシュ読出し適用・quota 縮退・single-flight・sha256 実装（digest 前の全量コピーあり —
   一時 RAM 2 倍の一因）。percent-encode がドットを透過することも確認（path 許可リスト検査の
   根拠）。

## Examples

`size: 0` / `"…"` は placeholder（実物は exporter が資産から導出した実値を焼く）。

### Anima（turbo 焼き込み配布リポ・`karume.json`）

```jsonc
{
  "format": "karume/1",
  "generator": "karume/0.1.0",
  "pipeline": "anima/1",
  "components": {
    "text_encoder": {
      "file": { "path": "text_encoder/model.safetensors", "size": 0, "sha256": "…" }
    },
    "text_conditioner": {
      "file": { "path": "text_conditioner/model.safetensors", "size": 0, "sha256": "…" }
    },
    "transformer": {
      "variants": {
        "f16": {
          "file": { "path": "transformer/model.f16.safetensors", "size": 0, "sha256": "…" },
          "extras": {
            "rope_base": { "path": "transformer/rope_base.safetensors", "size": 0, "sha256": "…" }
          }
        },
        "i8": {
          "file": { "path": "transformer/model.i8.safetensors", "size": 0, "sha256": "…" },
          "extras": {
            "rope_base": { "path": "transformer/rope_base.safetensors", "size": 0, "sha256": "…" }
          }
        }
      }
    },
    "vae_decoder": {
      "file": { "path": "vae_decoder/model.safetensors", "size": 0, "sha256": "…" }
    },
    "tokenizer": { "file": { "path": "tokenizer/qwen2-tokenizer.json", "size": 0, "sha256": "…" } },
    "tokenizer_2": { "file": { "path": "tokenizer_2/t5-tokenizer.json", "size": 0, "sha256": "…" } }
  },
  "presets": {
    "f16": { "weights": { "transformer": "f16" }, "session": {} },
    "i8": { "weights": { "transformer": "i8" }, "session": {} },
    "w8a8": { "weights": { "transformer": "i8" }, "session": { "linearCompute": "i8a8" } },
    "w8a8-a8": {
      "weights": { "transformer": "i8" },
      "session": { "linearCompute": "i8a8", "attentionCompute": "i8a8" }
    },
    "w8a8-s16": {
      "weights": { "transformer": "i8" },
      "session": {
        "linearCompute": "i8a8",
        "attentionCompute": "i8a8",
        "attentionScoreStorage": "f16"
      }
    },
    "f16-c16": {
      "weights": { "transformer": "f16" },
      "session": { "linearCompute": "f16", "attentionCompute": "f16" },
      "gpuFeatures": { "shaderF16": true }
    }
  },
  "defaultPreset": "w8a8-s16",
  "pipelineConfig": {
    "scheduler": { "shift": 3, "numTrainTimesteps": 1000 },
    "defaults": {
      "steps": 8,
      "guidanceScale": 1,
      "resolution": { "width": 1024, "height": 1024 },
      "negativePrompt": "low quality, worst quality, blurry, bad anatomy, jpeg artifacts"
    }
  }
}
```

### SBV2（`karume.json`）

```jsonc
{
  "format": "karume/1",
  "generator": "karume/0.1.0",
  "pipeline": "sbv2/1",
  "components": {
    "text_encoder": {
      "variants": {
        "f16": {
          "file": { "path": "text_encoder/model.f16.safetensors", "size": 0, "sha256": "…" }
        },
        "i8": { "file": { "path": "text_encoder/model.i8.safetensors", "size": 0, "sha256": "…" } }
      }
    },
    "duration_predictor": {
      "variants": { "f16": { "file": {/* … */} }, "i8": { "file": {/* … */} } }
    },
    "front": { "variants": { "f16": { "file": {/* … */} }, "i8": { "file": {/* … */} } } },
    "flow": { "variants": { "f16": { "file": {/* … */} }, "i8": { "file": {/* … */} } } },
    "decoder": { "variants": { "f16": { "file": {/* … */} }, "i8": { "file": {/* … */} } } },
    "voice": { "variants": { "f16": { "file": {/* … */} }, "i8": { "file": {/* … */} } } },
    "tokenizer": {
      "file": { "path": "tokenizer/deberta-tokenizer.json", "size": 0, "sha256": "…" }
    },
    "symbols": { "file": { "path": "text/symbols.json", "size": 0, "sha256": "…" } },
    "style_vectors": {
      "file": { "path": "styles/style_vectors.safetensors", "size": 0, "sha256": "…" }
    }
  },
  "presets": {
    "f16": {
      "weights": {
        "text_encoder": "f16",
        "duration_predictor": "f16",
        "front": "f16",
        "flow": "f16",
        "decoder": "f16",
        "voice": "f16"
      },
      "session": {}
    },
    "w8": {
      "weights": {
        "text_encoder": "i8",
        "duration_predictor": "i8",
        "front": "i8",
        "flow": "i8",
        "decoder": "i8",
        "voice": "i8"
      },
      "session": {}
    },
    "w8a8": {
      "weights": {
        "text_encoder": "i8",
        "duration_predictor": "i8",
        "front": "i8",
        "flow": "i8",
        "decoder": "i8",
        "voice": "i8"
      },
      "session": { "linearCompute": "i8a8" }
    }
  },
  "defaultPreset": "w8",
  "pipelineConfig": {
    "sampleRate": 44100,
    "language": "jp-extra",
    "speakers": { "jvnv-F1-jp": 0 },
    "styles": {
      "Neutral": 0,
      "Angry": 1,
      "Disgust": 2,
      "Fear": 3,
      "Happy": 4,
      "Sad": 5,
      "Surprise": 6
    },
    "defaults": {
      "speaker": "jvnv-F1-jp",
      "style": "Neutral",
      "styleWeight": 1.0,
      "sdpRatio": 0.2,
      "lengthScale": 1.0,
      "noise": 0.6,
      "noiseW": 0.8
    }
  }
}
```

（SBV2 の数値既定は本家慣例値のサンプル — export 時に実測系列から確定する。per-component の
`weights` map なので「decoder だけ f16 に戻す」等の非対称 preset も同じ形で書ける）

## Consequences

- hub v1 の実装範囲: `karume.json` の取得と parse（§1〜§3 の全検査・パス許可リスト・規模
  上限）/ `resolve(preset?)` / fetch-cache 接続（revision 1 回解決・SHA 固定・名前空間・
  同時 4・abort・進捗合算）/ エラー型 4 + 透過 + 診断イベント。
- exporter の CLI 化で manifest（size / sha256 / generator 込み）とモデルカード README
  （variant 表は manifest から生成）を自動出力し、path 許可リスト・重複 path 整合・禁則の
  組の不在を機械検査する。
- v1 の制約として起票するもの: DL 前チェックは feature 軸のみ（limits 不足は DL 後に fail
  loudly）/ 並行 `fromPretrained` のキャンセル粒度は single-flight leader 単位（いずれも
  `docs/limitations.md`）。
- 実装時の実測項目: ブラウザ Cache Storage への数 GB 格納可否（quota・
  `navigator.storage.persist()` 案内）/ JSR npm 互換層の `sideEffects: false` 出力 /
  **gated リポの実網取得**（HF の LFS はクロスオリジンリダイレクトで `Authorization` が
  落ちるため、署名付き URL で成立するかの確認）。

## 追記

- 2026-08-05: hub の公開面に `clearHubCache(options?)` を追加（「キャッシュを消して容量を
  空ける」の利用者ストーリー）。§5 の 2 名前空間（`karume/1` / `karume/1:auth`）を**両方**消し、
  他コードの名前空間には触らない — 認証側だけ残すと gated 資産の写しが端末に残る。1 つでも
  実在して消えたら `true`。`caches` 省略時は `globalThis.caches` を使い、Cache Storage が無い
  環境（非セキュアオリジン等）は fail loudly（「消したつもり」を作らない）。
- 2026-08-13: **認証キャッシュを credential ごとに分離**（外部レビュー HUB-004 の消化）。
  従来の隔離は `Authorization` の**有無**だけで名前空間を選んでおり、下層（fetch-cache）の
  キーが URL のみのため、token A で埋めた `karume/1:auth` に token B の同一 URL 要求が
  ヒットし得た（権限の違う 2 人が同じ端末を使う場面で gated 資産が漏れる）。現行は
  `karume/1:auth:<Authorization 値の sha256 先頭 16 hex>`（生 credential は名前に出さない —
  CacheStorage の名前は列挙可能）。`clearHubCache` は名前が事前列挙できなくなったため
  `CacheStorage.keys()` から `karume/1` 完全一致 + `karume/1:` 始まりを拾って全削除する
  （旧スキーム `karume/1:auth` の残骸も対象）。旧名前空間の写しは新コードから読まれず、
  gated 資産が 1 回だけ再ダウンロードになる。
- 2026-08-13: 進捗フェーズへ終端 `complete` を追加（外部レビュー HUB-001 の消化）。従来は
  検証（`verifying`）の後に最終イベントが `downloading` で出て phase が逆行していた。契約:
  1 ファイルの phase は `downloading`* → `verifying` → `complete` の順にだけ進み、`complete`
  はファイルごとに 1 回の終端。例外は破損キャッシュの self-heal（validate 拒否 → evict →
  network 再取得）で、この 1 巡だけ最初からやり直しになる。
- 2026-08-16: §3 の写像 `toSessionOptions` は 7 家族の `pipeline.ts` へバイト単位で複製されて
  いたが、`packages/models/src/session/options.ts` へ 1 本化した（barrel には出さない内部機構）。
  複製は「綴りの改名」こそ型検査で落ちる一方、「`SessionSpec` へのキー追加」は写像が書いて
  いないキーを黙って落とすだけで全家族の型検査を通り、追随を忘れた家族が沈黙劣化した
  （門を持つのは 2 家族だけだった — レビュー MD-1 C-1）。1 本化後は写像を
  `Required<SessionSpec>` の網羅レコードから組むため、キー追加はコンパイルエラーになる。
  門も `packages/models/tests/session_options_test.ts` の 1 本へ集約。
- 2026-08-24: 進捗イベント `AssetProgress` に**ファイル単位の 2 欄**（`fileLoaded` /
  `fileTotal`）を追加。§5 の進捗契約は集約（`loaded` / `total` = manifest `size` の合計）
  だけを定めており、消費側がファイル別の進捗バーを描く材料が無かった（`loaded` の差分を
  自前で持つと、キャッシュヒットで `downloading` が 1 度も出ないファイルや並行取得で崩れる）。
  **optional ではない必須欄**にしたのは、欄の有無を消費側が分岐すると「時々描けない進捗バー」
  ができるため。不変条件は `verifying` / `complete` で常に `fileLoaded === fileTotal`
  （全量が揃った点なので、キャッシュヒット経路でも同じ）・`fileTotal` は `FileRef.size` で
  `total` はその合計。`complete` 追加（上の 2026-08-13）と同格のスキーマ変更で、
  `FetchAssetsOptions.onProgress` として公開面に出る。
- 2026-08-25: **§7 が据え置き席として列挙していた 2 つを `karume/4` で実装した**（0.5.0 の
  breaking 波）。ADR [0075](0075-quant-presentation.md) の Context が NOTE で指摘していた
  「据え置きの約束と実装の食い違い」（`FILE_REF_KEYS` / `QUANT_KEYS` に席が無い）はこれで解消。
  - **ファイル参照の optional `repo` / `revision`（越境コンポーネント参照）**: §7 の原案より
    **狭く**入れた — `revision` は **40 桁小文字 hex の commit SHA だけ**を受ける
    （ブランチ・タグ・短縮形は fail loudly）。可変 ref を許すと §7 が言う「**不変**コンポーネント
    参照」の成立条件そのものが壊れる（参照先が動けば、こちらが宣言した `sha256` と食い違う日が
    来る）。`repo` と `revision` は**両方同時**にのみ現れ（片方だけは fail loudly）、`size` /
    `sha256` は自リポ参照と同じく必須 = **二重 pin**。取得層は宣言された (repo, revision) から
    取り、セッションの解決済み SHA は使わない。あわせて**ファイル参照の同一性キーを `path` から
    `fileRefKey`（`repo@revision/path`・自リポ参照は `path` のまま）へ切り替えた** — 別リポの
    同名 path は別のバイト列で、`path` で畳むと 3 点セット一致検査が正しい manifest を誤って
    拒否し、取得層は別ファイルのバイト列を返す。一次利用は `karume-anima-turbo` →
    `karume-anima` の text stack 共通化で、焼く側の opt-in は `tools/export-recipes/dist.py` の
    5 指定（`--ref-repo` / `--ref-revision` / `--ref-dist` / `--ref-model` / `--ref-role`・
    全部揃うか 1 つも無いかの 2 通りだけ）。**公開順序の制約**（参照先を先に上げて SHA を
    確定させないと焼けない）は [release-runbook](../release-runbook.md) §0 が正本。
  - **quant の optional `requiredLimits`**: WebGPU `GPUDeviceDescriptor.requiredLimits` と同型の
    「名前 → 満たすべき最小値」の**部分写像**（書かれていない limit は要求しない = 消費側の
    判定は `adapter.limits[name] >= value` の素の比較）。名前は 11 名の allowlist で、
    `@karume/runtime` の `REQUIRED_LIMIT_KEYS` と 1:1 に保つ — hub は runtime に依存しないので
    写しになるが、未知名は fail loudly なので綴りのずれが黙って無視されることはなく、語彙一致の
    門は両方へ依存できる唯一の位置（`packages/models/tests/limit_vocabulary_test.ts`）に置いた。
    今回の範囲は**受理と型面の露出まで**で、§7 が Consequences に書いた「DL 前チェックは feature
    軸のみ」の解消（limits 不足を DL 後の fail loudly から前へ出す結線）は**後続タスク** —
    [limitations](../limitations.md) の該当制約はまだ残っている。
- 2026-08-29: **越境参照を分割コンポーネントへ拡張**（R1 の shard 分割で共有 text_encoder が
  2 shard になり、1 参照で指せなくなった実需）。`shards` 配列の**要素ごと**に
  `{repo, revision, path, size, sha256}` の越境 FileRef を書く（`repo`/`revision` は全要素同一・
  並びは shard 番号順 = 先頭グラフ shard の規約は参照でも不変）。hub は元々 ref 単位で越境を
  解決するため**受け側は 0 行変更**（受理はテストで固定）。assets / extras の席（1 ファイル
  参照のみ）は従来どおり分割参照を fail loudly。焼く側の突合（参照先現物 = 自分で組むバイト列）
  は shard 全要素へ適用。
- 2026-08-31: **§5 の HF 固有記述は「HF アダプターの契約」へ降格する**（ADR
  [0086](0086-distribution-source.md)）。取得元は HF だけではなくなり、§5 が並べていた接続契約は
  2 つに割れる — **取得元固有**（可変 ref → commit SHA の解決・キャッシュ名前空間の所有・
  `hubUrl`・相 1 の streaming prefetch・sha256 照合）は `sources/hf.ts` の契約として読み、
  **取得元非依存**（進捗の算出と `AbortSignal` の透過・同時取得数と in-flight バイト予算・
  `openModel` へ渡す前の tight view assert・エラーの形と利用可能ラベル・`onCacheError` の
  アプリ配達）は共通層の契約として §5 のまま全取得元に掛かる。§7 の越境参照（`repo` /
  `revision`）は取得元契約の 1 つ（`originFor`）になり、ローカル取得元では明示 mapping と明示
  fallback だけが解決手段になる（ADR 0086 決定 3）。
