# 0115: GEMM 幾何の選択を「shape × adapter の静的な表（プロファイル）」へ広げる

- Status: accepted（利用者裁定 2026-09-27。M2 の quick 掃引で、勝つ幾何が GPU によって逆転した — 下の Context）
- Date: 2026-09-27
- 関連: ADR [0022](0022-gemm-register-blocking.md)（GEMM 骨格・決定 3 の MUST「縮約順は数値契約」・追記の MUST
  「実行時オートチューン禁止」「既定変更は門の再実測とセット」「バックエンド別の静的既定は明確な退行が出た場合に検討する」）/
  [0023](0023-fused-attention.md)（融合 attention ①QK・③PV）/ [0024](0024-conv2d-implicit-gemm.md)（conv の implicit GEMM）/
  [0060](0060-row-block-attention.md)（行ブロック実行の分解 attention）/ [0067](0067-autoregressive-attention-vocabulary.md)（states 形 attention）/
  [0106](0106-device-keyed-references.md)（sha256 参照値を環境キーごとの行で持つ）/
  [0058](0058-numerics-opt-in-contract.md)（i8a8 attention の変種を実走カナリアで決める）/ perf-ledger K-70・K-71・K-67・K-22。
  調査の正本は [research 2026-09-27 K-70](../research/2026-09-27-k70-metal-per-op.md) §5〜§10。
  実装 = `packages/runtime/src/kernels/geometry-profile.ts`（型・既定プロファイル・選択の純関数）・
  `packages/runtime/src/kernels/geometry-profiles/`（生成されたプロファイルと一覧 `BUILTIN_GEOMETRY_PROFILES`）。
  前提の部品 = codegen 入口の明示幾何（`9947568c`）・掃引道具 `tools/geometry-sweep`（`c562c7ae`・形式 `karume-geometry-sweep/2`）。

## Context

Apple M2（Chrome 153 / Metal）で anima の DiT が Arc B570 の 10〜30 倍遅い。per-op 実測（research §4〜§6）で、
遅さは一様ではなかった:

| 群                                                   | M2 / B570 の比 | 公称比（演算 3.2〜4.0 倍・帯域 3.8 倍）を超える倍率 |
| ---------------------------------------------------- | -------------: | --------------------------------------------------: |
| 帯域律速の小カーネル（quantize_rows・rms_norm ほか） |       3.6〜4.7 |                                            0.9〜1.2 |
| f32 中タイル linear（`reg64x32`）                    |            4.4 |                                            1.1〜1.4 |
| f32 大タイル linear（`reg128x128`）                  |           10.2 |                                            2.6〜3.2 |
| f32 attention ①QK / ③PV（大タイル）                  |     13.5 / 9.8 |                                            2.4〜4.2 |
| f32 conv2d implicit GEMM（VAE）                      |       6.0〜8.8 |                                            1.5〜2.8 |
| i8a8 linear / attention（dp4a）                      |     25.0〜33.8 |                                           6.3〜10.6 |

同じ f32 GEMM でも中タイルは公称比どおりで、大タイル（128×128・1 スレッド 64 累積・256 スレッド）だけが外れる。
実行系の差（Dawn / Metal 対 wgpu / Vulkan）が全 key に一様に効くなら中タイルも同じだけ超過するはずなので、
**主因は大タイルの幾何が M2 に合わないこと**と帰属した（research §6）。i8a8 は別の原因（Tint が `dot4I8Packed` を
展開する）で、幾何を直しても残る。

`tools/geometry-sweep` の quick 掃引（同じ shape で幾何だけを変えて GPU timestamp で測る）を M2 で回した結果
（research §9・op 5 族・33 ケース・177 行）:

| op                      | M2 の最良幾何               | M2 の対既定 | 同じ幾何の B570 の対既定 |
| ----------------------- | --------------------------- | ----------: | -----------------------: |
| f32 linear（M ≥ 1024）  | `reg64x32r4x4w8`            | ×1.58〜1.60 |              ×0.72〜0.98 |
| f32 attention ①QK       | `reg64x32r4x4w8`            | ×1.55〜1.60 |                 （未測） |
| f32 attention ③PV       | `reg64x64r8x4w16`           | ×1.66〜1.67 |                 （未測） |
| f32 conv2d              | `igemm64x64`（wg16x8 / 16） | ×1.37〜1.48 |                 （未測） |
| i8a8 linear / attention | `tile64x64r8x4w16x8k16`     | ×1.06〜1.13 |              ×0.64〜0.84 |

- **勝つ幾何が GPU によって逆転する**。B570 では大 M の f32 / i8a8 linear は既定が最速だった。
- **幾何間の出力 sha256 は全 177 行で既定と一致**（B570 の quick も全 77 行で一致）。f32 も i8a8 も幾何でビットは
  動かない。f32 の一致は理論保証ではなく実測命題（ADR 0022 追記）で、M2 がそのデータ点に加わった。
- 既定の再測定（ケースの最後に既定を測り直した比）は全ケース 0.99〜1.01 で、機の揺れは比の読みに効いていない。

ADR 0022 は「既定はバックエンド共通の 1 本（2026-08-10 裁定）。バックエンド別の静的既定は、実測で明確な退行が
出た場合にのみ検討する（その場合も実行時オートチューン禁止は不変）」と書いていた。上の表がその条件に当たる:
共通の既定は M2 で 1.4〜1.7 倍を取り逃し、M2 向けの幾何は B570 で退行する。1 本の既定ではどちらかが負ける。

## Decision

### 1. プロファイル = op × バケットの静的な表

幾何の選択を「shape の純関数」から「shape × プロファイルの純関数」へ広げる。プロファイルは次の型を持つ純データで、
`packages/runtime/src/kernels/geometry-profile.ts` に置く（副作用なし）:

```ts
export type GemmRowsRule = { readonly maxRows: number; readonly geometry: GemmGeometry };
export type GeometryProfile = {
  readonly id: string;
  readonly match: { readonly vendor?: string; readonly architecture?: string };
  readonly gemmRows: readonly GemmRowsRule[];
  readonly attention: { readonly qk: GemmGeometry; readonly pv: GemmGeometry };
  readonly conv2d: { readonly rows64: GemmGeometry; readonly rows32: GemmGeometry };
  readonly i8a8: {
    readonly linear: I8a8Geometry;
    readonly attentionQk: I8a8Geometry;
    readonly attentionPv: I8a8Geometry;
  };
  readonly provenance?: {
    readonly sweep: string;
    readonly sha256: string;
    readonly date: string;
    readonly adapter: string;
  };
};
```

- `id` はプロファイルの名前で、生成物のファイル名と一致させる（既定 = `"default"`・M2 = `"apple-metal-3"`）。
- `gemmRows` は行数バケットの表。`rows <= maxRows` で最初に当たる規則を使う。`maxRows` は昇順で、最後の規則は
  `Number.POSITIVE_INFINITY`。引くのは `gemmRowsGeometry(profile, rows)` の 1 箇所。
- `conv2d` の `rows64` / `rows32` は、implicit GEMM の m タイルのクラス（`conv2dIgemmMTile` が返す 64 行 / 32 行）ごとの
  幾何。**クラスを決めるのは今までどおり述語 `conv2dIgemmMTile`**で、実際のタイル辺を決めるのは幾何
  （`gemmTileM` / `gemmTileN` — 実タイル辺の正本は幾何という gemm-geometry.ts の MUST）。したがって 32 行クラスに
  tileM が 32 でない幾何を割り当ててもよい（`apple-metal-3` は Cout = 96 の 32 行クラスに tileM = 128 の
  `igemm128x64:wg16x16` を当てる）。
- 格納 dtype（f32 / f16 / i8 重み）は区別しない。今の既定の表も区別していない。
- `provenance` は掃引から生成したプロファイルだけが持つ（決定 4）。

### 2. 選択 = adapter の (vendor, architecture) から決定的な順で 1 本

`selectGeometryProfile(adapter, profiles = BUILTIN_GEOMETRY_PROFILES)` がプラン時に 1 回選ぶ。入力は
`GpuContext.adapterInfo` の `vendor` / `architecture`（文字列の完全一致で比べる — 正規化も前方一致もしない）。

1. `match` の vendor と architecture が両方とも一致するプロファイル。
2. 1 が無ければ、`match.architecture` が未指定で vendor だけが一致するプロファイル。
3. どちらも無ければ `DEFAULT_GEOMETRY_PROFILE`。

同じ順位に 2 本以上が当たったら `CodegenError` で落とす（fail loudly）。一覧の並び順で黙って 1 本を選ぶと、
一覧の編集だけで選択が変わる。

**類似度（GPU 名の近さ・コア数などで最近傍を選ぶ）は使わない。** 選ばれるプロファイルが機ごとに揺れると、
環境キーごとの sha256 参照値の行（ADR 0106）と「その機がどの幾何で走ったか」の対応が取れなくなる。完全一致なら、
環境キーを作るのと同じ `adapterInfo` から選択が一意に決まる。

### 3. 埋め込み = 生成された TS モジュールを commit する

プロファイルは `packages/runtime/src/kernels/geometry-profiles/<id>.ts` の `export const <ID_UPPER>: GeometryProfile`
として置き、`geometry-profiles/index.ts` の `BUILTIN_GEOMETRY_PROFILES`（既定を含まない生成物の一覧）に並べる。

- **実行時に外部から読まない**（JSON の fetch・環境変数・ファイル読み込みをしない）。ランタイム依存は Web 標準 API
  だけという不変条件と、barrel の tree-shaking（副作用ゼロ）をそのまま保つ。
- 生成物はモジュールの top-level で登録処理をしない（定数の export だけ）。一覧は `index.ts` の配列リテラル 1 本。

### 4. 生成規則 = 掃引 JSON から、確かに速い幾何だけを規則にする

生成器（`tools/geometry-sweep` の `main.ts profile`）は `karume-geometry-sweep/2` の掃引 JSON を 1 本以上
（`--from` を繰り返す）受け、合成してプロファイル 1 本を作る。

- **入力の門**: GPU timestamp で測った掃引だけを受ける（`gpuTiming.unit` が `ns` または `deno-raw-tick`、かつ
  `quantized` が false）。`gpuTiming` が欠けた掃引と壁時計（`wall`）の掃引は拒む。壁時計や 100 µs 丸めの掃引は
  比が 1 へ縮むので、生成を止める（fail loudly）。
- **合成**: 全ての掃引の adapter が同じであること（違えば落とす）。同じ sha256 の JSON を 2 度渡しても落とす。
  同じ (ケース, 幾何) を複数の掃引が測っていれば、比はその観測の幾何平均にする。出力の一致と失敗は全観測で見る。
- **既定の行の突合**: 掃引の既定の行（`isDefault`）の幾何が、今の runtime の既定（欄ごとの fallback — 決定 5 の表）と
  違えば落とす。比は既定の行に対する比なので、土台がずれた掃引では「既定より速い」が今の runtime で成り立たない。
- **候補**: `error` が無く、`identicalToDefault` が true の幾何だけ。1 度でも出力不一致か失敗した幾何は候補にしない。
  出力 sha が既定と違う幾何は、どれだけ速くても候補にしない（f32 の縮約順は数値契約 — ADR 0022 決定 3）。
- **採る条件**: プロファイルの 1 欄（`gemmRows` の 1 バケット・`attention.qk` など）に入る掃引ケースの**全部**で
  `speedupVsDefault >= 1.05`（既定より 5% 以上速い）。条件を満たす幾何のうち、そのケース群での幾何平均が最大の
  1 本を採る。幾何平均が同値なら幾何名の辞書順で先の 1 本（生成の決定性のため）。
- **満たさなければ既定のまま**。欄に入る掃引ケースが 1 本も無い場合も既定のまま（測っていない区間を補間しない —
  ADR 0022 追記の MUST と同じ規律）。
- 掃引中の揺れ（各ケースの末尾で既定を測り直した比 `defaultRepeat.driftRatio`）は、生成器は見ない。
- `gemmRows` のバケット境界は `gemm-geometry.ts` の `GEMM_ROWS_BUCKETS`（既定プロファイルと同じ 64 / 512 / ∞）から導く。
  3 段でなければ生成を止める。境界そのものを掃引から作ることはしない。
- 生成物は `provenance`（掃引 JSON の path・その sha256・掃引日・adapter の vendor / architecture / description）を
  **欄とファイル冒頭のコメントの両方**に持つ。どの実測から来た値かを、ソースだけから辿れるようにするため。
  複数の掃引から作ったときは、`sweep` / `sha256` / `date` をそれぞれ `--from` の順に `", "` で連結する。
  `--from` の順序を変えると生成物が変わるので、`--check`（再生成とバイト同一かだけを見る）は不一致になる。
  再生成のコマンドは生成物の冒頭コメントに書く。

### 5. 既定プロファイルは今の表とバイト同一

`DEFAULT_GEOMETRY_PROFILE` は今の選択と**同じ値**を持つ:

| 欄          | 値の出どころ                                                                                                                 |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `gemmRows`  | `gemmGeometryForRows` の 3 段（≤ 64 → `M16N16 r1×4 wg4×16`・≤ 512 → `M64N32 r4×4 wg8×16`・それ以上 → `defaultGemmGeometry`） |
| `attention` | ①QK / ③PV とも `defaultGemmGeometry`（`M128N128 r8×8 wg16×16`）                                                              |
| `conv2d`    | `gemmMTileGeometry(64)` / `gemmMTileGeometry(32)`                                                                            |
| `i8a8`      | `defaultI8a8Geometry`（linear / ①QK = `M128N64`・③PV = `M64N128`）                                                           |

`match` は vendor / architecture とも未指定。既定プロファイルが選ばれた機では、生成される WGSL とパイプラインキーが
本 ADR の前と 1 バイトも変わらない。既存の codegen スナップショットが無変更で緑であることが、その検出器になる。

### 6. 適用範囲

| 経路                                        | プロファイルの欄    |
| ------------------------------------------- | ------------------- |
| linear / matmul / bmm（行数バケット）       | `gemmRows`          |
| 融合 attention f32 の ①QK / ③PV（ADR 0023） | `attention.qk / pv` |
| conv2d の implicit GEMM（ADR 0024）         | `conv2d`            |
| i8a8 の linear / 融合 attention ①QK / ③PV   | `i8a8`              |

実測の裏付けの範囲:

- `gemmRows` は linear の実測で決める。同じ GEMM 骨格の matmul / bmm も同じ表を引く（違うのは B 側の充填だけ）。
  matmul / bmm 単独の速度は掃引に無い。
- `gemmRows` の ≤ 64 は M = 64 の 1 ケース、65〜512 は M = 512 の 1 ケースで決まる。M = 1〜63 の実測は無く、
  その区間への適用は外挿になる。f16 / f32 重みの linear は M = 1 だけが GEMV なので、M = 2〜64 にこの表が効く。
- `i8a8` の 3 欄は M ≥ 1024 の実測だけで決まり、全行数に効く（今の既定も行数で分けていない）。
- 掃引にケースを足して裏付けを補う案（linear の M = 16 / 32 / 128 / 256・matmul / bmm）は採用した（利用者裁定
  2026-09-29 — 追記決定 1。上の 2 点は `apple-metal-3` の時点の記述で、M2 の full 再走待ち）。

**今回は既定のままにする経路**（プロファイルを引かず、今の選択を使い続ける）:

- states 形 attention の GEMM 骨格タイル経路 ①ₜ / ③ₜ（ADR 0067）と、行ブロック実行の分解 attention（ADR 0060 —
  irodori の経路）。どちらも今は `gemmGeometryForRows` の表を通るが、掃引の対象に入っていない。別起票で測ってから広げる。
- conv1d の implicit GEMM（`gemmMTileGeometry` を conv2d と共有しているが、掃引に conv1d のケースが無い）。
- M = 1 × i4 の GEMV 族（ADR 0082 — GEMM 骨格の外）。

### 7. dp4a カナリアはプロファイルの i8a8 attention 幾何で撃つ

- i8a8 融合 attention の変種（dp4a / emu）を決める実走カナリア（`src/gpu/attention-dp4a-canary.ts`・ADR 0058 決定 2）は、
  Session が選んだプロファイルの `i8a8.attentionQk` / `i8a8.attentionPv` の幾何で撃つ。
- 理由: カナリアの MUST は「production の幾何で最低 1 タイル全域を撃つ」こと。既定の幾何のまま撃つと、プロファイルが
  i8a8 attention の幾何を変えた機で、カナリアが検証する WGSL と実走の WGSL が別物になる。
- 判定の記憶は device 単位のまま。プロファイルは同じ `GpuContext` の `adapterInfo` の純関数なので、同じ device なら
  どの Session でも幾何は同じになる。
- 固定入力の値・判定則・sanity 帯は変えない。B570 で `apple-metal-3` の幾何を渡しても判定は dp4a 厳密一致。

### 8. 実行時オートチューン禁止は不変・利用者が選ぶ口は作らない

> 2026-09-29 改定（追記決定 6）: 「利用者が選ぶ口」は `acquireGpu({ geometryProfile })` として公開した。実行時オートチューン禁止と
> `SessionOptions` に出さない点は不変。

- 選択は `adapterInfo` と静的な表だけの関数で、実行中に計測して選び直すことはしない（ADR 0022 追記の MUST のまま）。
  同じ機・同じ shape なら毎回同じ幾何・同じキー・同じ WGSL になる。
- 幾何はパイプラインキーに載ったまま（キーの綴りは不変）。プロファイルが違えばキーが違うので、パイプラインの
  キャッシュで取り違えは起きない。
- **`SessionOptions` には出さない**。利用者がプロファイルや幾何を渡す口は作らない。選ばれたプロファイルの `id` を
  診断情報（`SessionDiagnostics.geometryProfile`）で見せるだけにする。
- ADR 0022 追記の MUST「既定の変更は門の再実測とセット」は、プロファイルの追加・変更にもそのまま掛かる。プロファイルは
  その adapter にとっての既定だから。

### 採らなかった案

- **類似度で最近傍のプロファイルを選ぶ**（GPU 名の近さ・世代・コア数）— 採らない。未知の機で「近い」プロファイルが
  当たる利点はあるが、選択が名前の揺れで変わり、環境キーごとの参照 sha の行（ADR 0106）と幾何の対応が取れなくなる。
  未知の機は既定に落ちる方が、どの幾何で走ったかが一意に言える。
- **実行時に計測して選ぶ（オートチューン）** — 採らない。ADR 0022 追記の MUST。選択が実行ごとに揺れると、
  「同一キー → バイト同一 WGSL」とキーの意味が崩れる。f32 のビット同一は実測命題なので、測っていない組み合わせが
  実行時に選ばれうる形は門の外に出る。初回の計測時間も要る。
- **`SessionOptions` で幾何やプロファイルを渡す** — 採らない（2026-09-29: `SessionOptions` ではなく `acquireGpu` の GPU 単位の口として公開した — 追記決定 6）。公開面（ADR 0008 の薄い面）に内部の codegen 語彙
  （`GemmGeometry`）が漏れ、利用者が掃引の門（出力 sha の一致・5% の閾値）を通っていない幾何を持ち込める。
  掃引は `tools/geometry-sweep` が codegen の内部入口（明示幾何）で行えるので、公開の口は要らない。

## Consequences

- **参照 sha は環境キーの行のまま**（ADR 0106）。プロファイルは幾何だけを変え、幾何はビットを動かさない（実測命題）
  ので、プロファイルを足す前後で同じ機の sha は一致するはず。Apple 機の環境キーの行を足す（行が無い間は明示 SKIP）。
- **キーの綴りは不変**。既に幾何判別子（`reg64x32r4x4w8` など）が載っているので、新しい語彙は増えない。
- **M2 の DiT の見込み約 1.5 倍は推定**。research §9 の見積もり（f16 quant の DiT 段で linear が GPU 時間の約 85%、
  それが 1.6 倍 + attention）で、DiT 段の実測で確定するまでは推定として扱う。VAE（conv2d）の効果も同じく未計測。
  合成後の採用比は linear の大 M で ×1.74・attention で ×1.49〜1.73（下の表）。
- **`apple-metal-3` を生成し、`BUILTIN_GEOMETRY_PROFILES` に登録した**（下の表は 2026-09-27 時点。2026-09-29 に
  linear / matmul / bmm の full を 3 本目として足して再生成し、`gemmRows` ≤ 64 と 65〜512 は既定へ戻った — 現行の表は
  追記決定 3）。入力は M2（Chrome 153）の掃引 2 本:
  quick（2026-09-27T16-28・33 ケース・177 行・i8a8 を含む）と full（2026-09-27T18-31・linear / attention / conv2d の
  格子全体・19 ケース・927 行・出力不一致 0・失敗 0）。欄ごとの採否と退けた幾何の理由は生成物の冒頭コメントが正本:

  | 欄                 | 採用                    | 対既定（幾何平均・最小〜最大） | ケース          |
  | ------------------ | ----------------------- | -----------------------------: | --------------- |
  | `gemmRows` ≤ 64    | `reg64x64r4x4w16`       |                         ×2.261 | 1（M = 64）     |
  | `gemmRows` 65〜512 | `reg128x32r8x4w8`       |                         ×1.102 | 1（M = 512）    |
  | `gemmRows` > 512   | `reg128x32r8x4w8`       |         ×1.740（1.678〜1.773） | 6               |
  | `attention.qk`     | `reg128x32r8x4w8`       |         ×1.734（1.571〜1.830） | 4               |
  | `attention.pv`     | `reg64x64r8x4w16`       |         ×1.494（1.403〜1.671） | 4               |
  | `conv2d.rows64`    | `igemm128x64:wg16x16`   |         ×1.444（1.219〜1.711） | 2               |
  | `conv2d.rows32`    | `igemm128x64:wg16x16`   |                         ×1.648 | 1（Cout = 96）  |
  | `i8a8.linear`      | `tile64x64r8x4w16x8k16` |         ×1.112（1.061〜1.127） | 6（quick 由来） |
  | `i8a8.attentionQk` | `tile64x64r8x4w16x8k16` |         ×1.128（1.127〜1.130） | 4（quick 由来） |
  | `i8a8.attentionPv` | `tile64x64r8x4w16x8k16` |         ×1.098（1.090〜1.106） | 4（quick 由来） |

  quick 単独の見込みと違う欄: 65〜512 が既定 → `reg128x32r8x4w8`、> 512 と ①QK が `reg64x32r4x4w8` →
  `reg128x32r8x4w8`（比 ×1.59 → ×1.74・×1.58 → ×1.73）、conv2d が `igemm64x64` → `igemm128x64`。
  `attention.pv` の比は揺れを含む。full の ③PV は既定の再測定比が 0.9〜1.1 を外れたケースがある
  （pv-self-m1024 ×1.228・pv-cross-m1024 ×1.126）。合成後の比（×1.49）は quick 単独（×1.67）より低いが、幾何は同じ。
- **プロファイルは dispatch 上限も動かす**。1 workgroup = 1 出力タイルの経路が扱える n は
  `maxComputeWorkgroupsPerDimension`（65,535）× tileN まで。超えれば `DispatchLimitError` で落ち、黙って誤値は出ない。
  `apple-metal-3` の conv2d（tileN 64）では Hout·Wout ≤ 4,194,240（約 2047×2048）で、既定（tileN 128）の 8,388,480 の半分。
  linear の N 側（出力特徴数）と attention ①QK の N 側（キー数）も tileN 32 で 2,097,120 に縮むが、実用形では届かない。
- **adapter の綴り（実測）**: Chrome の M2 = vendor `apple` / architecture `metal-3`。Chrome の RTX 5070 Ti（Windows）=
  `nvidia` / `blackwell`。Deno の B570 = vendor `32902`（PCI ベンダ ID の 10 進）/ architecture 空（ADR 0106 決定 2）。
  登録済みのプロファイルは architecture を指定しているので、architecture が空の Deno では今どの機も既定プロファイルに
  落ちる（遅いが正しい）。Deno on macOS の `vendor` の綴りは未確認。
- **診断欄 `SessionDiagnostics.geometryProfile`**（公開型への欄の追加）に、選ばれたプロファイルの `id` が出る。
  名前は診断用のラベルで semver の対象外。
- **掃引の出力一致の門が実際に欠陥を捕まえた**: i8a8 融合 attention ①QK の生成器は、K 側の充填スロットが 5 以上の幾何で
  WGSL の変数名の重なり（シャドーイング）により誤値を出していた（既定と `apple-metal-3` の幾何は該当しない）。本 ADR と
  同時に修正し、宣言名の重なりの門（`tests/codegen_i8a8_shadowing_test.ts`）とスロット 8 / 16 の幾何での GPU 突合を検出器に
  した。生成器は出力不一致の幾何を候補にしないので、修正前でもプロファイルには入らなかった。
- **Intel / NVIDIA のプロファイルは本 ADR では作らない**。RTX 5070 Ti（Chrome）の full 掃引では、f32 の大 M・
  attention・conv2d は既定が最速か僅差だった（③PV の最良が ×1.02〜1.05・conv2d は c384 の ×1.092 だけ）。余地は中 M 512（`reg128x128r8x8w16` ×1.475）・小 M 64
  （`reg32x32r2x4w8` ×1.352）・i8a8 の 3 欄（`tile128x64r8x4w16x16k16` が全ケースで ×1.13〜1.31）にある。
  B570 の quick でも中 M 512 で 128×128 が ×1.41・小 M 64 で 64×32 が ×1.49。プロファイルを作るかは K-67 で
  利用者が裁定する。→ `nvidia-blackwell` は 2026-09-29 に生成・登録した（追記決定 2）。
- **「Metal では既定 quant の a8 を外す」は別 ADR**。M2 の i8a8 は幾何を直しても f32 と同じ時間で（research §9）、
  幾何の表では解けない。quant の選択（配布の席）の話なので、本 ADR の外で裁定する。
  この判断は保留中で、既定の quant 席を量子化にするか opt-in（元の重み）にするかの再検討に合流する（利用者裁定 2026-09-27）。
- states 形 attention と行ブロック attention のプロファイル化は別起票（決定 6）。

## 検収

| 項目                                                                                         | 結果                                                                                                   |
| -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| 既定プロファイルで codegen スナップショットが無変更で緑（WGSL・キーのバイト同一）            | ✅（`codegen_wgsl_test` / `codegen_dispatch_test` 79 本）                                              |
| `selectGeometryProfile` の選択順（完全一致 → vendor のみ → 既定）と同順位 2 本の fail loudly | ✅（`geometry_profile_test`）                                                                          |
| 生成器が M2 の掃引 JSON から `apple-metal-3` を作る                                          | ✅（quick + full の合成・Consequences の採用表）                                                       |
| B570 でのフル verify（既定プロファイルが選ばれ、sha256 参照門が全一致）                      | ✅（2026-09-27・3410 passed / 0 failed / 26 ignored）                                                  |
| M2（Chrome）で anima の DiT 段の GPU 時間・壁時計（適用前 / 後・f16 quant と既定 quant）     | ✅（2026-09-29・512²・f16 quant ×1.78・既定 ×1.15 — 追記決定 4）                                       |
| M2 で適用前後の PNG sha256 が一致（幾何でビットが動かないことの E2E での確認）               | ✅（f16 `041027e63559`・既定 `0a5695470e4a` とも一致）                                                 |
| 確認ページ・診断に選ばれたプロファイルの `id` が出る                                         | 実装済み（2026-09-29・段ごとの `geometryProfile` を表・環境行・JSON に出す）。実機表示は利用者確認待ち |

## 追記（2026-09-29）— 掃引ケースの追加と `nvidia-blackwell` の登録（利用者裁定 2026-09-29）

### 追記決定 1: `gemmRows` は linear / matmul / bmm のケースで決める（決定 6 の更新）

- 掃引（`tools/geometry-sweep`）に次のケースを足した。linear の M = 16 / 32（text_encoder の形 `[1,M,1024] × [3072,1024]` の
  M を置換）と M = 128 / 256（transformer の cross-attention k / v 射影の形 `[1,M,1024] × [2048,1024]` の M を置換）。
  matmul 3 本 — rank-2 の matmul は anima を含むどの系列の op census にも行が無いので、linear の各行数バケット 1 本の
  **鏡像**（同じ M / N / K・B 側は `[K,N]`）を測る。census 由来でないことは `censusCount: 0` と `mirrorOf`（鏡像元の
  linear のケース id）で表す。bmm 5 本 — anima の census の text_encoder 2 形・text_conditioner 3 形をそのまま。
- 生成器の規則: 行数バケットの欄に入るケースは linear / matmul / bmm の全部。表が 3 経路に効く以上、matmul / bmm の
  ケースで ×1.05 未満の幾何は、linear で速くても採らない。生成物の文言は「掃引にある linear / matmul / bmm のケースで
  決め」（linear だけの掃引から作った表が、測っていない op で決めたと読めないように）。
- 既存の生成物との関係: `apple-metal-3` は linear だけの掃引から作られたままで、値は不変（文言 1 行だけ再生成）。
  **M2 で再走するときは full**（op = linear / matmul / bmm）。生成器は「欄の全ケースで測った幾何」しか候補にしないので、
  新ケースを quick だけで足すと、quick 集合に無い採用幾何 `reg128x32r8x4w8` が「測っていない」で落ちて欄が後退する。
- B570 の quick 実走（新ケースだけ・判断材料で、B570 の表は作らない）: ≤ 64 の M = 16 / 32 では大タイルが ×0.30〜0.77
  と大きく負け、65〜512 では M = 128 の最良が ×1.019 に留まる一方 M = 256 と matmul M = 512 は ×1.4〜1.7 で、
  1 点で欄を決める外挿のリスクを裏付けた。
- 小さい bmm 3 本（1 dispatch が 15〜40 µs）は反復の上限 1024（`tools/opbench` と共有の `MAX_REPS` — 出力 readback の
  線形増を抑える目的で、同じバッファに重ね打ちする掃引には当てはまらない）で pass が目標 80 ms に届かず、既定の
  再測定比が 0.705〜1.552 と揺れた。→ 利用者裁定（同日）で掃引専用の上限 `SWEEP_MAX_REPS` = 16384 に分けた
  （opbench の 1024 は不変・JSON の `settings.maxReps` で見分ける）。B570 の bmm 5 ケースの再走では既定の反復が
  5,687 / 3,818 / 2,573 / 476 / 268 で、再測定比は 0.9995〜1.0049（全て範囲内）。

### 追記決定 2: `nvidia-blackwell`（Chrome の RTX 5070 Ti = `nvidia` / `blackwell`）を生成・登録する

- 入力 = RTX 5070 Ti（Windows・Chrome 153）の full 掃引 1 本（2026-09-27T18-37・5 族 33 ケース・1,599 行・失敗 0・
  出力不一致 0・既定の再測定比が範囲外のケース 0）。生成規則は決定 4 のまま（`--min-speedup 1.05`）。

  | 欄                 | 採用                      | 対既定（幾何平均・最小〜最大） | ケース       |
  | ------------------ | ------------------------- | -----------------------------: | ------------ |
  | `gemmRows` ≤ 64    | `reg32x32r2x4w8`          |                         ×1.352 | 1（M = 64）  |
  | `gemmRows` 65〜512 | `reg128x128r8x8w16`       |                         ×1.475 | 1（M = 512） |
  | `gemmRows` > 512   | 既定のまま                |                              — | 6            |
  | `attention.qk`     | 既定のまま                |                              — | 4            |
  | `attention.pv`     | 既定のまま                |                              — | 4            |
  | `conv2d.rows64`    | 既定のまま                |                              — | 2            |
  | `conv2d.rows32`    | 既定のまま                |                              — | 1            |
  | `i8a8.linear`      | `tile128x64r8x4w16x16k16` |         ×1.194（1.156〜1.256） | 6            |
  | `i8a8.attentionQk` | `tile128x64r8x4w16x16k16` |         ×1.214（1.199〜1.238） | 4            |
  | `i8a8.attentionPv` | `tile128x64r8x4w16x16k16` |         ×1.212（1.134〜1.306） | 4            |

- dispatch 上限: `gemmRows` ≤ 64 の tileN 32 で linear の N 側上限は 2,097,120（実用形では届かない）。i8a8 の tileN は
  64 で既定と同じ。conv2d は既定のまま。
- `gemmRows` の ≤ 64 と 65〜512 は、`apple-metal-3` と同じく M = 64 / M = 512 の各 1 ケースで決まっている。追記決定 1 の
  新ケースは RTX ではまだ測っていない（B570 の観察は上）。RTX で full を再走したら `--from` を 2 本にして再生成する。
- 検収: B570 の per-profile GPU テスト（`gpu_geometry_profile_test` — 既定との Uint32 一致 + 幾何判別子が実走キーに
  載ること）は緑。RTX 5070 Ti（Chrome）での anima の登録前後の PNG sha256 一致と、診断 `geometryProfile` が
  `nvidia-blackwell` になることは未計測（利用者作業）。
- 注: RTX の掃引で i8a8-attention の cross M = 1024 の 2 ケースは反復の上限 1024 に当たっている（再測定比は範囲内）。

### 検収（追記分）

| 項目                                                                                            | 結果                                                                                      |
| ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| 掃引に linear M 16 / 32 / 128 / 256・matmul 3・bmm 5 を足し、B570 で失敗 0・出力不一致 0        | ✅（2026-09-29・65 行）                                                                   |
| matmul / bmm の掃引経路が本番の recipe-builders と束縛・params・dispatch で一致（CPU 参照突合） | ✅（`harness_test` 実 GPU・故障注入で赤を確認）                                           |
| 生成器が matmul / bmm の観測を `gemmRows` の全ケース門に数える                                  | ✅（`profile_test`）                                                                      |
| `nvidia-blackwell` を登録し、B570 の per-profile GPU テストが緑                                 | ✅（2026-09-29）                                                                          |
| M2（Chrome）で full（linear / matmul / bmm）を再走し `apple-metal-3` を 3 本から再生成          | ✅（2026-09-29・追記決定 3）                                                              |
| RTX 5070 Ti（Chrome）で登録前後の PNG sha256 一致・診断 `geometryProfile` = `nvidia-blackwell`  | id は ✅（2026-09-29・4 段とも — 追記 5）。前後の sha 一致は登録前の RTX 記録が無く未確認 |

### 追記決定 3: M2 の full 再走（linear / matmul / bmm）で `apple-metal-3` を 3 本から再生成した（2026-09-29）

- 入力 = 2026-09-27 の quick + full に、2026-09-29 の full（op = linear / matmul / bmm・20 ケース・1,080 行・失敗 0・
  出力不一致 0・既定の再測定比は全ケース範囲内・checkout `cb87ab93`）を 3 本目として足した。`--from` の順は生成物の冒頭。

  | 欄                                | 2026-09-27（linear だけ） | 2026-09-29（linear / matmul / bmm）      | ケース |
  | --------------------------------- | ------------------------- | ---------------------------------------- | ------ |
  | `gemmRows` ≤ 64                   | `reg64x64r4x4w16` ×2.261  | **既定のまま**                           | 6      |
  | `gemmRows` 65〜512                | `reg128x32r8x4w8` ×1.102  | **既定のまま**                           | 7      |
  | `gemmRows` > 512                  | `reg128x32r8x4w8` ×1.740  | `reg128x32r8x4w8` ×1.647（1.171〜1.766） | 7      |
  | attention / conv2d / i8a8 の 7 欄 | 不変                      | 不変                                     | —      |

- ≤ 64 が既定へ戻った理由: linear M = 16 で全候補が既定に負ける（旧採用 `reg64x64r4x4w16` は M = 16 で ×0.648・M = 32 で
  ×1.148・M = 64 で ×2.252）。バケット ≤ 64 の中で最良の幾何が M ごとに違い（M = 16 は `reg16x32r2x4w8` ×1.21、M = 32 は
  `reg32x64r4x4w16` ×1.75、M = 64 は `reg64x64r4x4w16` ×2.25）、1 本の幾何では全ケースを ×1.05 以上にできない。
  65〜512 が戻った理由: 旧採用 `reg128x32r8x4w8` が bmm M = 512 / N 64 / K 512 で ×0.976（他のケースでは ×1.05〜1.10）。
- 効果の範囲: DiT の linear（M = 1024 / 4096）・attention・conv2d・i8a8 は幾何が不変で、K-71 の狙い（DiT 段）には影響しない。
  戻った 2 欄が効くのは text_encoder の linear（M = 64）・cross-attention の k / v 射影（M = 512）・text 段の bmm で、
  M2 ではこれらは既定の幾何に戻る（速度は 2026-09-27 以前と同じ）。
- 観察（別起票の材料・本 ADR では動かさない）: 行数バケットの境界（64 / 512 / ∞）は決定 4 のとおり掃引から作らない。
  M2 では ≤ 64 の中で最良が M = 16 / 32 / 64 で割れるので、境界を細かくすれば ×1.2〜2.25 が取れる余地がある。
  境界は runtime の既定の表（`GEMM_ROWS_BUCKETS`・ADR 0022）と共有なので、変えるなら既定プロファイルの門と一緒に別 ADR で。
- M2 でも bmm の M = 64 の 2 ケースは反復の上限 1024 に当たった（追記決定 1 の末尾・再測定比は範囲内）。
- 検収: 追記分の表の「M2（Chrome）で full を再走し 3 本から再生成」は ✅（2026-09-29）。B570 の per-profile GPU テストは緑。

### 追記決定 4: M2 の DiT 再測で `apple-metal-3` を確定する（2026-09-29・K-71 採用）

- 計測 = 利用者の M2 24 GB（Chrome 153・`apple` / `metal-3`）・確認ページ（当時の `deno task bench:anima-browser` — 同日夜に `tools/gpu-lab` の Anima タブへ統合）・512²・seed 42・
  DiT 常駐（2 回目以降の値）・checkout `cb87ab93`（DiT の幾何は現行と同じ。text_encoder の linear だけ当時の ≤ 64 の規則
  `reg64x64r4x4w16` で走っており、追記決定 3 で既定へ戻った — text 段 1.9 s のうち linear は 0.13 s で影響は無視できる）。
  記録 = `outputs/bench-browser/anima-residency-browser-f16-2026-09-29T17-15-47.294Z.json`（比較元は 2026-09-27 の 2 本）。

  | 席（512²）        | 項目             |          2026-09-27（既定幾何） | 2026-09-29（`apple-metal-3`） |            比 |
  | ----------------- | ---------------- | ------------------------------: | ----------------------------: | ------------: |
  | `f16`             | DiT 段（壁時計） |                          80.5 s |                        45.1 s |         ×1.78 |
  | `f16`             | DiT 段（GPU）    |                          77.3 s |                        44.2 s |         ×1.75 |
  | `f16`             | 内訳 linear      |                          65.4 s |                        37.1 s |         ×1.76 |
  | `f16`             | 内訳 ①QK / ③PV   |                     3.7 / 2.8 s |                   2.0 / 1.6 s | ×1.85 / ×1.75 |
  | 既定（i8a8・s16） | DiT 段（壁時計） |                          81.5 s |                        71.1 s |         ×1.15 |
  | 既定（i8a8・s16） | DiT 段（GPU）    |                          79.5 s |                        69.9 s |         ×1.14 |
  | 両席              | VAE 段           |                      3.9〜4.4 s |                         2.9 s |          ×1.4 |
  | 両席              | PNG sha256       | `041027e63559` / `0a5695470e4a` |                          同じ |          一致 |

- kill 線（f16 quant の DiT 段が 1.3 倍未満なら再検討）を超えたので、`apple-metal-3` を採用で確定する（perf-ledger K-71 ✅）。
  research §9 の見込み（linear ×1.74・attention ×1.5〜1.7 から DiT 段 約 1.5 倍）に対し、実測は ×1.78 で見込みどおり。
- M2 では **既定 quant（a8）が f16 quant より 1.58 倍遅い**（71.1 s 対 45.1 s）。09-27 の「a8 の利得ゼロ」は、幾何の直った
  f32 経路が伸びたぶん「a8 が損」に変わった。これは「既定の quant 席を量子化にするか opt-in にするか」の再検討の材料
  （backlog now・`.claude/reviews/2026-09-29_quant-default-recon/`）で、本 ADR では動かさない。

### 追記 5: RTX 5070 Ti の実走で `nvidia-blackwell` が選ばれることを確認した（2026-09-29）

- 計測 = 利用者の RTX 5070 Ti（Windows・Chrome 153・`nvidia` / `blackwell`）・確認ページをポート転送で開く・512²・seed 42・
  DiT 常駐（2 回目以降の値）・checkout `9fd39823`。記録 = `outputs/bench-browser/anima-residency-browser-f16-2026-09-29T18-11-14.451Z.json`。
  4 段（text_encoder / text_conditioner / transformer / vae_decoder）とも診断 `geometryProfile` は `nvidia-blackwell`。

  | 席（512²）        | DiT 段（壁時計） | DiT 段（GPU） | 内訳 linear                       | PNG sha256     |
  | ----------------- | ---------------: | ------------: | --------------------------------- | -------------- |
  | `f16`             |           2.86 s |        2.79 s | `reg128x128r8x8w16`（既定）2.31 s | `c3cef8d6bc64` |
  | 既定（i8a8・s16） |           1.21 s |        1.14 s | `tile128x64r8x4w16x16k16` 0.74 s  | `3b07b912c4d4` |

  text 段の 12.8 s はポート転送越しの重み取得（research K-70 §3.1 と同じ経路の問題）で、幾何とは無関係。
- 検収の「登録前後の PNG sha256 一致」は、登録前（既定幾何）の RTX のパイプライン記録が無いので未確認。幾何がビットを
  動かさないことの RTX での根拠は、RTX の full 掃引で採用幾何の全ケースが既定と出力一致だったこと（追記決定 2）と、
  B570 の per-profile GPU テスト（`nvidia-blackwell` の幾何と既定が Uint32 一致）。前後一致を取るなら、登録前のコミット
  （`cd11cdc8`）の確認ページを別ポートで立てて同じ生成を 1 回回す。
- 3 機の同条件比較（512²・DiT 常駐 2 回目以降・DiT 段の GPU 時間）: research K-70 §12。

### 追記決定 6: 利用者が静的なプロファイル 1 本を `acquireGpu` に注入できる口を公開する（決定 8 の改定・利用者裁定 2026-09-29）

- 決定 8 の「利用者がプロファイルや幾何を渡す口は作らない」は、プロファイルの出所がリポジトリの埋め込みだけだった時点の判断で、
  「未知の device で利用者が自分の掃引結果を当てる」場面を天秤に載せていなかった（利用者の指摘 2026-09-29）。その場面では既定幾何に
  落ちるしかなく、表がリリースに入るまで最適化できない。よって口を 1 つ公開する。
- 口 = `acquireGpu({ geometryProfile })`（`AcquireGpuOptions` の 1 欄・型 `GeometryProfile` と欄の型 `GemmRowsRule` / `GemmGeometry` /
  `I8a8Geometry` を公開面に出す — 型だけで、既定の表・埋め込みの一覧・選択関数は値として出さない）。プロファイルは device の性質なので
  GPU 単位に渡す。渡されたら adapter の (vendor, architecture) を見ずにその表を使い、`match` は照合しない（別の機の表を当てて A/B する
  用途）。同じ device の全 Session と dp4a カナリア（決定 7）が同じ表を使う。表は acquire 時に複製して保持し、device の寿命の間は不変。
- 据え置く不変条件: 実行中に測って選び直さない（ADR 0022 追記の MUST）・幾何は担当割りだけを変え K の縮約順は不変・`SessionOptions` と
  manifest には出さない（device の性質をモデル側の口に置かない — ADR 0111 の受理表とも混ざらない）・選択は Session 構築の 1 箇所
  （`session-build.ts`）で、診断 `geometryProfile` とカナリアはその結果を引く。
- 門: 壊れた表（id が空・`match` の形の破れ・`gemmRows` の昇順 / 末尾 Infinity の破れ・整除の破れた幾何）は device を作る前に公開の
  `GpuFeatureError` で落とす（利用者入力の失敗は公開型 — `subgroups` の検査と同じ流儀）。出力の一致は掃引の門で確かめる実測命題で、
  利用者が持ち込む任意の表について runtime が保証するものではない（生成器は出力不一致の幾何を候補にしないので、掃引の生成物なら通る）。
- 注入された表の `id` は埋め込みの id と重なってもよい（埋め込みの表そのものを別の機に当てる A/B では同じ id が真実）。診断だけでは
  注入か埋め込みかを区別しないので、記録する道具（確認ページの JSON）は「何を注入したか」を別欄に持つ。
- 検収: 注入した表の id・幾何判別子が実走キーに載ること・既定と Uint32 一致（B570・実 GPU）／壊れた表が requestDevice に届かないこと
  （偽の `navigator.gpu` で回数 0）／カナリアが注入した表の ①QK / ③PV の WGSL をコンパイルすること／注入しない経路の codegen
  スナップショットが無変更。故障注入 3 通り（門を外す・選択を adapter のみに戻す・カナリアにだけ adapter の選択を渡す）で赤を確認。
- PoC の導線（同日裁定・別コミット）: 1 ページの道具 `tools/gpu-lab` で「掃引（quick+）→ 生成 → 注入して Anima を実行」を回す。
  掃引の候補集合 quick+ = quick ∪ 登録済みプロファイルの採用幾何。

### 追記 7: PoC の道具 `tools/gpu-lab` — 掃引 → 生成 → 注入 → 実行を 1 ページで回す（2026-09-29・利用者裁定）

- 1 ページ 3 タブ（掃引 / プロファイル / Anima）・サーバ 1 本（`deno task bench:gpu-lab`・既定ポート 8790・localhost 限定・ポート転送で
  別の機の Chrome から使う）。旧 2 ページ（掃引の Chrome ページ・Anima の確認ページ）は吸収して削除した。Deno の双子 CLI
  （`tools/geometry-sweep/main.ts`・`tools/anima-residency/profile.ts`）は残る。
- 掃引の候補集合は 3 段: `quick`（既定 + 4〜5 形）・**`quick+`（既定 — quick ∪ 登録済みプロファイルの採用幾何）**・`full`（格子全体）。
  quick+ を既定にした理由: quick の固定集合には M2 の勝ち幾何（`reg128x32r8x4w8`）も RTX の i8a8 の勝ち幾何も無く、他の機で勝った幾何を
  未知の機で試すのが安い（所要は quick とほぼ同じ）。CLI は `--set <quick|quick+|full>`（省略 = quick+・`--quick` は `--set quick` の別名）、
  JSON は `settings.candidateSet`（互換の `settings.quick` は `candidateSet === "quick"` のときだけ true）。登録用の表は full で。
- プロファイルのタブは生成器の純関数（`tools/geometry-sweep/derive.ts` — CLI と共有・Deno 非依存）でその場で表を作り、TS の生成物・
  注入の JSON・アプリ用 TS（`@karume/runtime` の型を import）・登録のコマンドを出す。「適用」は GPU を取り直して注入する（Anima の
  pipeline とダミーは畳む）。
- Anima の JSON（`karume-anima-residency-browser/2`）に `geometryProfileRequested`（`auto` / `default` / `builtin:<id>` / `generated:<id>`）と
  `geometryProfileInjected`（注入した表の値・auto では無い）を足した。段ごとの診断 `geometryProfile` は使われた表の id で、注入か埋め込みかは
  この 2 欄で読む。
- 検収はページの README「What to confirm」1〜7（利用者の実機で）。
