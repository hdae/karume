# 0117: 利用者アプリの中で幾何を掃引し、表を作って保存し、次の `acquireGpu` に注入する（`@karume/runtime/tune`）

- Status: accepted（利用者裁定 2026-10-01。3 論点〈公開の形・計測の門・保存キー〉とも推奨案 a。決定 4 の「runtime の版ではなくカーネルの指紋」と
  埋め込み 2 表への適用、決定 3 の 1%・決定 2 / 6 の形も同日に推奨案で裁定）
- Date: 2026-10-01
- 関連: ADR [0115](0115-geometry-profiles.md)（幾何プロファイル・決定 1 / 4 / 8・追記決定 6 / 7 / 8・追記 7 / 9）/
  [0116](0116-geometry-profile-row-buckets.md)（行数バケット 7 段・段ごとの fallback）/
  [0022](0022-gemm-register-blocking.md)（追記の MUST「実行時オートチューン禁止」「既定変更は門の再実測とセット」）/
  [0008](0008-public-api.md)（薄い公開面・サブ面は `exports` で足す・道具は `src/` を直接 import してよい）/
  [0106](0106-device-keyed-references.md)（sha256 参照値を環境キーごとの行で持つ）/ perf-ledger K-71。
  実装（移す前）= `tools/geometry-sweep/`（harness / cases / geometries / derive / report）・`tools/gpu-lab/browser/`（掃引とプロファイルのタブ）。
  注入口 = `packages/runtime/src/gpu/acquire.ts`。表の型 = `packages/runtime/src/kernels/geometry-profile.ts`。

## Context

利用者アプリが自分の端末で掃引し、表（幾何プロファイル）を作って保存し、以後は `acquireGpu` に固定で注入する。これを成立させたい。
開発者が実機を持たない端末（例: 8 GB 級の Android）は、今は既定の幾何で走るしかない。

### 現状ある物

- 注入口 `acquireGpu({ geometryProfile })`（`acquire.ts:551`・ADR 0115 追記決定 6）。門 `assertGeometryProfile`（id・`match` の形・規則列・整除）を
  device を作る前に通し（`acquire.ts:592-594` → `640-650`）、複製して device の寿命の間は変えない。`match`・`provenance`・表を作った版は
  照合しない（`acquire.ts:537-538`）。
- 埋め込みの表の一覧 `BUILTIN_GEOMETRY_PROFILES` と型 `GeometryProfile` ほかの公開（`mod.ts:132-135`・追記決定 7）。
- 表の JSON 直列化 `profileJson`（`tools/geometry-sweep/derive.ts:1157`）。末尾の `maxRows`（Infinity）を `1e999` と書く。
- `tools/gpu-lab` での実証（ADR 0115 追記 7・追記 9）。掃引は専用の device を取って捨てる（`sweep-tab.ts:221-227` / `271-285`）。
  最後に生成した表を localStorage（キー `karume-gpu-lab/last-generated-profile/1` — `injectable-tables.ts:15`）に残す。

### 4 つの設計論点（backlog K-71 の起票）

1. 掃引の核と生成器が runtime の内部 `src/` を直接 import していて、JSR では配れない（`harness.ts:46-115`）。
2. 生成器は timestamp-query かつ非量子化の記録しか受けない（`derive.ts:262-290`）。フラグ無しの Chrome（timestamp が 100 µs に丸まる）では表を作れない。
3. 注入は `match` も版も照合しない。保存した表を当ててよいかの判断材料（保存キー）が表に無い。
4. 掃引の所要時間が記録に無い（`date` は終了時刻 — `sweep-tab.ts:186`）。

### 調査の数値

**内部 import**（機械抽出）:

| 区分                                    | 数・中身                                                                                                                                                                      |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 掃引側が import する runtime の内部の値 | 64（うち `harness.ts` が 55）                                                                                                                                                 |
| 内部の型                                | 2（`StorageRoles` / `Conv2dDims`）                                                                                                                                            |
| 掃引に要るが公開面に無いもの            | 明示幾何の codegen 入口（key / params / WGSL）・pipeline cache・error scope の規律（`RUNTIME_INTERNAL`）                                                                      |
| runtime 以外の依存                      | `tools/opbench/bench.ts` の較正定数（`calibrateReps` / `ROUNDS` / `TARGET_PASS_MS` / `WARMUP_MIN_RUNS` / `WARMUP_NS`）・`tools/anima-residency/timing.ts` の `looksQuantized` |

公開面にあるのは「表の形」と「GPU を取る口」だけで、掃引に要るものは全部内部にある。

**計測の門**（2026-09-29 の gpu-lab の full・各 45 ケース 2,247 行・当時の 3 段の生成器で 10 欄）:

| 記録                      | pass の GPU 時間 最小 / p5 / 中央値 | 床（壁時計 − GPU 時間）中央値 / 最大 | 100 µs 丸めの模擬で全欄の採否が原本と同じ試行 | 壁時計だけで選んで採否が変わった欄 |
| ------------------------- | ----------------------------------- | ------------------------------------ | --------------------------------------------: | ---------------------------------: |
| M5（Chrome 154）          | 49.6 / 71.9 / 85.9 ms               | 0.65 / 3.85 ms                       |                                     200 / 200 |                             0 / 10 |
| M2（Chrome 153）          | 20.2 / 79.1 / 85.5 ms               | 0.79 / 1.61 ms                       |                                     200 / 200 |                             0 / 10 |
| RTX 5070 Ti（Chrome 153） | 3.1 / 72.0 / 80.3 ms                | 1.25 / 13.5 ms                       |                                     200 / 200 |                             0 / 10 |

- harness は reps を壁時計で見積り、1 pass ≈ 80 ms（`TARGET_PASS_MS` — `tools/opbench/bench.ts:22`）に揃える（`harness.ts:1099-1126`）。
  100 µs 丸めの誤差は timestamp 2 つの差あたり 1 刻み未満で、80 ms の pass なら約 0.13%。採用の閾値 ×1.05 より 2 桁小さい。
- 今の門の理由（`tools/anima-residency/timing.ts:143-148`）は、Anima の 1 dispatch = 1 pass の計測から来ている。pass を 80 ms に揃える掃引には当たらない。
- 外れの実例: RTX の `i8a8-linear-m1024-n8192-k2048` の 1 行（`tile32x64r8x4w8x4k32`）は reps の見積りが外れて pass 3.1 ms だった。
  100 µs 丸めならこの行で約 3% の誤差になる。
- 丸めの模擬は、各 round の begin / end を位相が一様乱数の 100 µs 格子へ切り捨てた（同じ測定値の再利用 — Chrome の丸め方式は未確認）。

**所要時間**（記録からの積み上げ・2026-09-29 の 45 ケース時点）:

| 記録      |    行 | 積み上げ | うち空回し |
| --------- | ----: | -------: | ---------: |
| M5 quick+ |   340 |   7.3 分 |     3.2 分 |
| M2 quick+ |   340 |   8.6 分 |     3.3 分 |
| M5 full   | 2,247 |  45.9 分 |    21.8 分 |
| M2 full   | 2,247 |  60.7 分 |    24.5 分 |
| RTX full  | 2,247 |  42.4 分 |    21.0 分 |

- 目安は quick+ 7〜9 分・full 42〜61 分。空回し（1 幾何 0.5 s）が約半分を占める。
- 算入していないもの: パイプラインのコンパイル・入力の生成と書き込み（ケースごと最大約 1.1 GiB）・出力の digest の読み戻しと SHA-256
  （full で約 134 GiB・quick+ で約 20 GiB）。ADR 0116 で 57 ケースになったので、その分も伸びる（推測）。

**保存キーの材料**:

| 情報                             | フラグ無しの Chrome で | 備考                                                                                                                                                    |
| -------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| adapter の vendor / architecture | 取れる                 | Apple M2 と M5 は同じ `apple` / `metal-3`（ADR 0115 追記 9）                                                                                            |
| adapter の device                | 空                     | フラグ有りで Apple は `0x0000`・NVIDIA は PCI ID                                                                                                        |
| adapter の description           | 空                     | フラグ有りで `"Apple M2"` / `"Apple M5"`                                                                                                                |
| runtime の版                     | 取れない               | mod.ts にも src にも版の定数が無い（`packages/runtime/deno.json` の `"version"` だけ）                                                                  |
| カーネルの指紋                   | 今は無い               | codegen 決定性（同一キー → バイト同一 WGSL）で安定に導ける                                                                                              |
| ケース集合の版                   | 今は無い               | `SWEEP_CASES`（`cases.ts:351`）に版の欄が無い                                                                                                           |
| 今の `provenance`                | —                      | `sweep` / `sha256` / `date` / `adapter` の 4 文字列（`geometry-profile.ts:83-88`）。`adapter` は連結文字列（`derive.ts:937-942`）で `device` を持たない |

保存した表を黙って当てて遅くなる形は 4 つある: 別の機へ持ち出す・同じ端末で別の adapter が選ばれる・runtime を更新する（既定の幾何・
バケット境界・カーネル本体が変わる）・ブラウザやドライバを更新する。今はどれも検出しない。

**バンドル**（`deno bundle --minify` の実測）: サブパスを import したアプリだけが +36〜71 KB（gzip +11〜20 KB）。import しないアプリは 0。

## Decision

### 1. 公開の形 = `@karume/runtime` のサブパス `./tune`

- 掃引の核を `packages/runtime/src/tune/` へ移す。harness / cases / geometries / derive / report の Deno 非依存部、opbench の較正定数、
  `looksQuantized`（と量子化単位 `CHROME_TIMESTAMP_QUANTUM_NS`）。
- `packages/runtime/deno.json` の `exports` を `{ ".": "./mod.ts", "./tune": "./tune.ts" }` にする。入口は `packages/runtime/tune.ts`
  （`mod.ts` と同じく、明示的に設計した薄い面 — ADR 0008）。
- 掃引が使う codegen の内部語彙（Context の 64 値・2 型）は、同じパッケージの中から import する。`./tune` にも `.` にも出さないこと MUST。
  理由: 出すと semver の面になり、ADR 0008 追記 2026-09-03「道具のために内部 API を公開面へ出すのは禁止」と、ADR 0115 の採らなかった案
  （公開面に内部の codegen 語彙が漏れる）に当たる。
- この形を採る理由:
  - **掃引とカーネルの版が必ず揃う**。同じパッケージ版の中で、掃引が組む束縛表・params・WGSL と本番の recipe が一致する。
    別パッケージだと tune@x が runtime@y の生成器を呼びうる。束縛の入れ替わりのうち役割の同じものは `assertBindingRoles` でも見えない（`harness.ts:36-37`）。
  - **内部語彙のサブパス（`./internal`）を作らずに済む**。
  - CLI・gpu-lab・利用者アプリが同じ核を使う（実装 1 本）。
- `tools/geometry-sweep` の CLI（`main.ts` / `profile.ts`）と `tools/gpu-lab` は `./tune` の利用者になる（薄い殻）。道具だけが使う物
  （生成物の TS の描画 `renderProfileSource`・`--check`・`--min-speedup`）は道具側に残し、`src/tune/` の内部を直接 import してよい
  （ADR 0008 追記 2026-09-03 の射程）。
- `src/tune/` のモジュールは副作用ゼロ MUST（横断の不変条件）。import 時に計算しない（ケース集合の版も関数で導く — 決定 4）。
- main の面（`mod.ts`）は値を増やさない。変わるのは `AcquireGpuOptions.geometryProfile` の型（決定 6）と `GeometryProfile.provenance` の形（決定 4）。

### 2. `./tune` の面 = 高水準の入口だけ

| 入口                                                              | 役割                                                                                                                                                                                                         |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `runGeometrySweep(options)`                                       | 専用の device を取って掃引し、終わったら捨てる。戻りは記録（`karume-geometry-sweep/2`）。options = adapter の要求・候補集合（`quick` / `quick+` / `full`）・op の絞り込み・中断（`AbortSignal`）・進捗の通知 |
| `deriveGeometryProfile(reports, options)`                         | 記録 1 本以上から表を作る（生成器 — 規則は ADR 0115 決定 4・ADR 0116・本 ADR 決定 3）。戻りは表と採否の行                                                                                                    |
| `geometryProfileJson(profile)` / `parseGeometryProfileJson(text)` | 表の直列化と unknown 境界の parse（決定 7）                                                                                                                                                                  |
| `geometryProfileMismatch(profile, adapterInfo)`                   | 保存した表を当ててよいかの照合（決定 5）                                                                                                                                                                     |
| `sweepCaseSetId()`                                                | ケース集合の版（決定 4）                                                                                                                                                                                     |
| 型                                                                | 記録の型・options の型・候補集合の型・parse の失敗の型 `GeometryProfileParseError`                                                                                                                           |

- 名前は本 ADR で決める。引数と戻りの細部は段 2 / 4 で詰める（公開面スナップショット門が差分を出す）。
- `runGeometrySweep` は呼び手の `GpuContext` を受けず、device を自分で取って捨てる。掃引と推論が同じ device を共有する形を、型の上で
  作れなくするため（決定 9 の MUST を構造で守る）。gpu-lab と CLI は今も専用の device を取って捨てているので、その手順を核へ寄せるだけ。
- timestamp-query は、adapter が列挙すれば要求する（今の CLI と同じ — `derive.ts:264-266` の文言）。列挙しない adapter では壁時計で測る。
  その記録は残せるが、生成器は拒む（決定 3）。
- 掃引のケース集合は anima の op census 由来（`cases.ts:1-18`）。別の系列のアプリも anima の shape で欄の採否を決める。この性格を
  `./tune` の doc に書く。

### 3. 計測の門 = 観測ごとの丸め誤差の上界（ADR 0115 決定 4「入力の門」の改定）

量子化フラグでの一律拒否（`derive.ts:279-284`）をやめる。観測（掃引 1 本の中の 1 行）ごとに、丸めが比に与える誤差の上界を出して判定する。

- **量子化単位 q**: `gpuTiming.quantized` が true なら 100 µs（Chrome の刻み）。false なら 0（`ns` の非量子化と `deno-raw-tick`）。
- **行の上界** e(r) = q ÷ min（行 r の計測 round のうち負でないもの）。
  `perDispatch` = その min ÷ reps で、reps は整数なので誤差を持たない。timestamp 2 つの差の丸め誤差は 1 刻み未満なので、
  e(r) は `perDispatch` の相対誤差の上界になる（1 次の近似）。
- **比の上界** E(r) = e(r) + e(d)。d は同じ掃引・同じケースの既定の行。`speedupVsDefault` は 2 つの `perDispatch` の比なので、
  相対誤差は和で押さえられる（1 次の近似）。
- **しきい値**: E(r) ≤ 1%。定数 1 本で、CLI 引数にもアプリの options にもしない（規則は 1 つ）。根拠:
  - 80 ms の pass で e は約 0.13%、E は約 0.25%。
  - 2026-09-29 の 3 機の full では、既定の行の e は最大 0.15%。E が 1% を超える観測は M5 0・M2 0・RTX 1（Context の外れ行・e だけで 3.2%）。
  - 1% は採用の閾値 ×1.05 の余地（5%）の 1/5、再測定比の門の許容幅（0.9〜1.1 — 追記決定 8）の 1/10。
- **超えた観測の扱い**（追記決定 8 と同じ流儀）:
  - その掃引の比の材料から、その (ケース, 幾何) の観測を外す。出力の不一致と失敗はその観測でも判定に効く（正しさの門は丸めに依らない）。
  - 他の掃引に同じ (ケース, 幾何) の観測があればそちらで判定する。無ければ「測っていない」扱いで、その幾何はその欄の候補にならない（安全側）。
  - 既定の行の e が大きいと、そのケースの全観測の E が超える。結果として、再測定比の範囲外と同じく、そのケースの比は全部外れる。
  - 外した観測と E の値は採否の行（生成物の冒頭コメントと `deriveGeometryProfile` の戻り）に残す。除外を無効にする口は作らない。
- `gpuTiming.quantized` は記録に残す（書き手の判定 `roundsLookQuantized` — `report.ts:178-193` は変えない）。読み手は真偽値であることを
  検査し、q を決めるのに使う。
- **壁時計だけの記録**（`gpuTiming.unit` = `wall`）と `gpuTiming` の無い記録は、引き続き拒む（fail loudly）。理由:
  - 壁時計の誤差は、丸めのような既知の定数で押さえられない。submit → 完了の床が pass ごとに動く（Chrome の 3 機で中央値 0.65〜1.25 ms・
    最大 13.5 ms。Deno は約 11 ms — `report.ts:34-36`）。
  - 床は GPU 時間と並べないと記録から読めないので、行ごとの上界を記録だけから出せない。
  - 3 機で壁時計だけでも採否が同じだったのは、床が pass（約 80 ms）に比べて小さかったという観察で、モバイルの床は未知。
  - Chrome はフラグ無しでも timestamp-query を出す（100 µs に丸めて — `tools/gpu-lab/README.md`「Before measuring」）。アプリ内の掃引は
    丸めた timestamp で測れるので、壁時計を受けなくても論点 2 は解ける。
- 決定 4 の他の規則（合成・既定の行の突合・候補・×1.05・既定のまま・再測定比による除外）は変えない。

### 4. `provenance` の構造化

```ts
readonly provenance?: {
  readonly sweep: string; // 掃引の記録の path（--from の順に ", " で連結 — 今のまま）
  readonly sha256: string; // 同上の sha256
  readonly date: string; // 同上の掃引日
  readonly candidateSet: string; // 同上の候補集合（quick / quick+ / full）
  readonly adapter: {
    readonly vendor: string;
    readonly architecture: string;
    readonly device: string;
    readonly description: string;
  };
  readonly kernels: string; // カーネルの指紋
  readonly caseSet: string; // ケース集合の版
};
```

- `adapter` は連結文字列（`derive.ts:937-942`）をやめ、`GPUAdapterInfo` の 4 欄を空文字も含めてそのまま持つ。生成器は、全ての入力の記録で
  4 欄が一致することを要求する（今の 3 欄の一致の検査 — `derive.ts:819-848` — に `device` を足す）。表示用のラベルは描画のときに
  4 欄から導く（同じ情報を 2 か所に持たない）。
- `candidateSet` は記録の `settings.candidateSet`。欄の無い古い記録（quick+ の導入前）は `settings.quick` から `quick` / `full` を読む。
- 書くのは生成器 `buildGeometryProfile`（`derive.ts:1103-1145`）だけ。TS の生成物と注入の JSON は今までどおりこの 1 本から作る。
- `provenance` は公開型 `GeometryProfile` の欄なので、形の変更は公開面の変更になる（v0.x の Breaking・CHANGELOG に明記）。
- runtime（`assertGeometryProfile`・注入口）は `provenance` を見ないまま（追記決定 6）。

#### runtime の版ではなくカーネルの指紋を採る

- **指紋の定義**: 表の各欄について、その欄に入る掃引ケースの各 shape で、表の幾何と既定の幾何（欄の fallback — ADR 0116 決定 3）の
  パイプラインキー・params・WGSL・workgroups を決まった順に連結し、同期の非暗号ハッシュ（64 bit 以上）を取った 16 進文字列。
  GPU も時刻も読まない。codegen 決定性で、同じ runtime なら常に同じ値になる。導出は掃引の case plan（`harness.ts:621` の `casePlan`）と
  同じ経路を通す（掃引で測ったカーネルと、指紋が指すカーネルを別の経路で組まない）。
- 理由:
  - **情報源から導ける**。runtime には版の定数が無い。版を使うなら、src に `deno.json` の写しを置く（同期を保つ検査が要る）か、
    アプリに自分の依存の版を書かせる（書き忘れと書き違いが黙って通る）ことになる。
  - **無効になる時機が合っている**。版は GEMM のカーネルに触れないリリースでも表を捨てさせ、利用者に 7〜61 分の再掃引を強いる。
    指紋は、表が指すカーネルか既定のカーネルが変わったときだけ変わる。
  - **未リリースの checkout でも効く**（gpu-lab・開発中）。版の文字列が同じでも、カーネルが変われば指紋は変わる。
- 費用と穴:
  - 照合のたびに codegen を回してハッシュを取る（GPU なし。推測: 数 ms〜数十 ms — 段 4 で測る）。
  - キー・params・WGSL・workgroups の外の変更（recipe の束縛の並びなど）は指紋に出ない。そこは runtime 自身の検査
    （codegen スナップショット・既定の幾何での GPU テスト）が受け持つ。
  - 同期のハッシュにするのは、照合の純関数とコールバック（決定 6）を同期に保つため（`crypto.subtle` は非同期しか無い）。
    偶発の衝突だけを想定し、敵対的な偽装は射程外（表は利用者自身の保存物）。
- 埋め込みの 2 表も生成時に指紋を持つ。単体テストで「埋め込みの表の `provenance.kernels` = 今の runtime で導いた指紋」を縛る。
  GEMM のカーネルを変える変更は、2 表の再生成を同じ変更で伴う（幾何の値は変えない）。カーネルが変わったことを機械が告げ、
  per-profile の GPU テスト（ADR 0116 決定 7）が新しいカーネルで出力一致を確かめ直す。ADR 0022 追記「既定の変更は門の再実測とセット」と同じ規律。

#### ケース集合の版

- `sweepCaseSetId()` = `SWEEP_CASES`（`cases.ts:351`）と境界 `PROFILE_GEMM_ROWS_BOUNDS`（`cases.ts:135`）を正規の JSON にして、
  指紋と同じハッシュを取った文字列。手で上げる版番号は置かない（ケースを足して版を上げ忘れる形を作らない）。
- 定数ではなく関数にするのは、import 時に計算しないため（決定 1）。

### 5. 照合の純関数 `geometryProfileMismatch`

- 形: `geometryProfileMismatch(profile: GeometryProfile, adapterInfo: GPUAdapterInfo): string | undefined`。最初の不一致の文言を返し、
  一致なら `undefined`。投げない。`sessionOptionsViolation`（`mod.ts:168`）と同じ流儀で、打つ手（既定で走る・再掃引する・利用者に知らせる）は呼び手が決める。
- 照合する物（全て文字列の完全一致）:
  1. `provenance` があること（無い表 = 手書きの表 → 「照合の材料が無い」）。
  2. adapter の 4 欄（`vendor` / `architecture` / `device` / `description` — 空文字も値として比べる）。
  3. カーネルの指紋（今の runtime で表から導き直した値と `provenance.kernels`）。
  4. ケース集合の版（`sweepCaseSetId()` と `provenance.caseSet`）。
- 照合しない物: `candidateSet`（quick+ の表で足りるかはアプリの判断）・`match`（注入は `match` を見ない — 追記決定 6）・
  ブラウザやドライバの版（取れる情報が無い）。
- 置き場は `./tune`（指紋とケース集合の版が `src/tune/` にあるため）。起動時に照合するだけのアプリも `./tune` を import する。
  掃引の核はモジュールが副作用ゼロなので tree-shaking で落ちる見込み（推測 — 未測定・未解決に記録）。
- **限界**: フラグ無しの Chrome では `device` と `description` が空で、Apple M2 と M5 は同じ 4 欄になる（ADR 0115 追記 9 / 追記決定 7）。
  この関数は、表をその 2 機の間で持ち運んだことを検出できない。localStorage は端末とブラウザのプロファイルに閉じるので、アプリが自分で
  表を運ばない限りこの形は起きない（推測: Chrome の同期は localStorage を運ばない — 未確認）。

### 6. コールバック形の注入口

- `AcquireGpuOptions.geometryProfile` の型を `GeometryProfile | ((adapterInfo: GPUAdapterInfo) => GeometryProfile | undefined)` に広げる。
- 呼ぶ時機: `acquireGpu` 1 回につき 1 度。adapter を取った後（`acquire.ts:595`）で、device を作る前（`acquire.ts:606`）。
  渡す `adapterInfo` は `readAdapterInfo`（`acquire.ts:382-383` — 欠落を空値に正規化）の値で、後で `GpuContext.adapterInfo` になる物
  （`acquire.ts:627`）と同じ。
- 戻りが `undefined` なら、指定が無いときと同じ自動選択（埋め込みの表から 1 本）。戻りの表は、表を直接渡したときと同じ門と複製
  （`acquire.ts:640-650`）を通る。壊れていれば device を作る前に `GpuFeatureError`。コールバックが投げた例外はそのまま伝え、device を作らない。
- 理由: runtime が実際に選んだ adapter の情報で表を引ける。今は、アプリが生の `navigator.gpu.requestAdapter()` を自分で呼ぶ
  （runtime が選ぶ adapter と一致する保証が仕様に無い — `acquire.ts:665` の NOTE と同じ問題）か、注入なしで 1 度 acquire して捨てるしかない。
- **コールバックは同期 MUST**。理由: adapter は数秒〜数分で失効してよく、失効した adapter の `requestDevice` は例外ではなく、生まれた時点で
  lost な device を返す（`acquire.ts:660-664`）。adapter を持ったまま I/O を待つ窓を作らない。保存した表の読み込み（IndexedDB など非同期）は
  `acquireGpu` の前に済ませ、コールバックは純関数（parse・照合・選択）だけにする。
- コールバックは測らない（device がまだ無い）。adapter 情報と保存済みの表の純関数なので、ADR 0022 追記の MUST に触れない。
- runtime は注入時に照合しない（追記決定 6 の設計のまま）。照合はアプリがコールバックの中で `geometryProfileMismatch` を呼んで行う。
  `match` を見ないのも同じ（別の機の表を当てる A/B の用途は残る）。

### 7. unknown 境界の parse `parseGeometryProfileJson`

- 形: `parseGeometryProfileJson(text: string): GeometryProfile`。失敗は `GeometryProfileParseError`（`./tune` の公開型）で、文言は欄の path を名指す。
- 検査:
  - JSON として読めること。
  - 全欄の型（`id` は文字列・幾何は整数の欄だけ・`provenance` は決定 4 の形）。
  - **未知の欄の拒否**（表・`match`・幾何・`provenance` の全ての階層）。
  - `gemmRows` の末尾の `maxRows` が Infinity であること（`1e999` の復元）。末尾が `null` なら「Infinity の欠落（素の `JSON.stringify` は
    Infinity を null にする — `geometryProfileJson` で書く）」と名指す。
  - 最後に `assertGeometryProfile`（runtime の門と同じ 1 本）。
- 未知の欄を拒む理由: 表は注入して実行に効く値で、形の版が違う表を黙って読むと欄の意味の取り違えが通る。読めない表は捨てて掃引し直すのが
  アプリの正しい手（決定 10）。掃引の記録（決定 8）とは方針が逆になる。記録は欄を足していく追記型のログで、読み手は自分が読む欄だけを使うから。
- `geometryProfileJson(profile)` は今の `profileJson`（`derive.ts:1157`）。parse との往復で構造が一致することを単体テストで縛る。
- 保存した JSON の欠落が公開型でない `TypeError` で落ちる形と、末尾 `null` の文言が真因（Infinity の欠落）を名指さない形は、
  アプリの経路では parse が受け持つ。runtime の注入の門（型付きの値を受ける）は変えない。

### 8. 掃引の記録 `karume-geometry-sweep/2` に足す欄

| 欄                  | 中身                                                   | 理由                                                                                    |
| ------------------- | ------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| `startedAt`         | 掃引の開始時刻（ISO 8601）                             | 今の `date` は終了時刻（`sweep-tab.ts:186` は記録を組む時点）で、所要が記録から読めない |
| `aborted`           | 中断で終わったか（真偽値）                             | 中断した掃引と op を絞った掃引を、記録だけで区別できない（どちらもケースが欠けるだけ）  |
| `cases[].elapsedMs` | ケースごとの壁時計（ms）                               | どのケースが所要を食うかを記録から読む                                                  |
| `caseSet`           | ケース集合の版（決定 4）                               | どのケース集合で測ったか                                                                |
| `defaultKernels`    | 既定の表（`DEFAULT_GEOMETRY_PROFILE`）のカーネルの指紋 | どの runtime のカーネルを比の土台にしたか                                               |

- 形式は /2 のまま、任意の欄として足す。読み手の `parseSweepReport`（`derive.ts:251-325`）は読む欄だけを検査し、未知の欄を無視する。
  `candidateSet` を足したときと同じ流儀（`report.ts:128-131`）。
- 生成器は `caseSet` / `defaultKernels` があれば今の runtime の値と照合し、違えば止める（fail loudly）。理由は既定の行の突合（ADR 0115 決定 4）と
  同じで、土台が今の runtime と別物の記録で表を作らないため。欄の無い記録（今までの記録）は今までどおり受ける。
- `aborted` は生成器の判定に使わない。途中で切れたケースは既定の再測定が無いので、追記決定 8 の規則で比の材料から外れる。
- 書き手は `runGeometrySweep` 1 本（CLI も gpu-lab もこれを使う）。

### 9. 不変条件と責任の分界（ADR 0022 追記の MUST との整合）

ADR 0022 追記の原文（`0022-gemm-register-blocking.md:88-89`）:
「MUST: 幾何の既定変更は門の再実測とセット。**実行時に幾何が変わる形（オートチューン）は f32/f16 では取らない**」。

- **runtime の acquire / Session の経路では測らない MUST**。測るのは、利用者が明示的に呼ぶ `./tune` の `runGeometrySweep` だけ。
  `./tune` を採ると runtime パッケージが測るコードを含むので、不変条件の主語を「runtime は測らない」から「acquire / Session の経路は測らない」へ言い直す。
- **掃引は専用の device で行う MUST**（`runGeometrySweep` が取って捨てる — 決定 2）。**推論と並走させない MUST**。並走すると測定が汚れ、
  GPU も取り合う。runtime は別の device の使われ方を見られないので、これは利用者アプリへの契約として `./tune` の doc に書く。
- 表は device の寿命の間は変わらない（追記決定 6・`acquire.ts:592-594` の複製）。新しい表は次の `acquireGpu` から効く。
  同じ device の上で表を差し替える口は作らない。
- 同じ表・同じ shape なら毎回同じ幾何・同じキー・同じ WGSL（ADR 0115 決定 8）。アプリは起動時に表を 1 度決めるだけなので、キーの意味は崩れない。

| 誰が                     | 何を保証する                                                                                                                                             |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| runtime                  | 注入された表を device の寿命の間は変えない・acquire / Session の経路で測らない・注入時の構造検査（`assertGeometryProfile`）                              |
| `./tune`（掃引と生成器） | 採る幾何はクラスの全ケースで出力が既定と一致し、×1.05 以上（ADR 0115 決定 4）。証拠は掃引のケース集合（anima の shape）× その端末 × 採用幾何の範囲に限る |
| 利用者アプリ             | 照合（`geometryProfileMismatch`）で使い続けてよいかを決める・掃引を推論と並走させない・どの表で走ったかを記録する（出力の再現性）                        |

- E2E の sha 門（PNG / WAV の sha256 参照値 — ADR 0106）は、リポの埋め込みの表には掛かるが、アプリ内で作った表には掛からない。
  アプリ内の表の出力一致の証拠は、掃引の出力一致の門だけになる（追記決定 6 の「runtime は保証しない」の範囲）。
- アプリが「既定で起動 → 裏で掃引 → 表を差し替えて取り直す」を自動で回すと、アプリの水準では実行時オートチューンと同じ振る舞いになる。
  runtime の不変条件（device 単位で不変・acquire / Session の経路で測らない）は保たれる。いつどの表で走ったかの記録はアプリの責任で、
  `./tune` の doc にこの点を書く。

### 10. 表の寿命と再掃引の時機

- **使い続けてよい条件**: `geometryProfileMismatch` が `undefined` を返すこと（adapter の 4 欄・カーネルの指紋・ケース集合の版が一致）。
- **再掃引の時機**: 照合が不一致を返したとき（runtime の更新でカーネルかケース集合が変わった・別の adapter が選ばれた・フラグの切替で
  `device` / `description` が変わった）と、利用者の明示操作。不一致の間は、コールバックが `undefined` を返して既定で走れる。
- **検出できない古さ**: ブラウザ・ドライバ・OS の更新で性能特性が変わっても、照合は一致のまま（取れる情報が無い）。黙って遅くなりうる。
  出力一致の証拠は崩れない（幾何は担当割りだけを変え、カーネルは指紋で同一）。利用者の明示操作で再掃引する。
- quick+ と full のどちらを勧めるか、いつ掃引を走らせるか（初回起動・アイドル時・設定画面）は利用者アプリの判断で、本 ADR は決めない。
  目安は Context の所要時間。
- 表の保管先（localStorage・IndexedDB・ファイル）は runtime の範囲外。`geometryProfileJson` の文字列を保存し、`parseGeometryProfileJson` で戻す。

## 採らなかった案

- **別パッケージ `@karume/tune`（A）** — 採らない。JSR の別パッケージは相手の `exports` 経由でしか import できない。runtime に内部語彙の
  サブパス（harness が使う 55 値と `RUNTIME_INTERNAL` の 3 プリミティブ）が要り、codegen 語彙が semver の面になる（ADR 0008・0115 の判断に反する）。
  tune と runtime の版もずれうる。
- **公開しない（C）/ gpu-lab のページを静的に置き、利用者が自分の端末で回して JSON をアプリに貼る（C'）** — 採らない。全ての不変条件を
  そのまま保てるが、「アプリの中で」を満たさない。開発者が実機を持たない端末は既定のまま残る。
- **runtime が注入時に照合する** — 採らない。追記決定 6 が残した「別の機の表を当てて A/B する」用途が消える。照合に落ちたときの打つ手は
  アプリごとに違い、runtime が投げると選べない。照合は純関数で公開し、呼ぶかどうかをアプリに置く。
- **保存キーに runtime の版を使う** — 採らない（決定 4）。runtime に版の定数が無く、写しかアプリ任せになる。GEMM に触れないリリースでも
  表を捨てさせ、未リリースの checkout ではカーネルが変わっても同じ値のまま。
- **門を外すだけ（量子化フラグの拒否を消し、上界を見ない）** — 採らない。3 機では採否が変わらなかったが、外れ行（RTX の pass 3.1 ms・
  誤差約 3%）のような観測を材料に残す。pass の短い行は reps の見積りが外れたときに出るので、端末を選ばず起こりうる。上界の判定なら
  外れた観測だけを外し、他は今と同じに使える。
- **壁時計だけの記録も受ける** — 採らない（決定 3）。床が pass ごとに動き、記録だけから上界を出せない。
- **コールバックを非同期にする** — 採らない（決定 6）。adapter を持ったまま待つ窓を作る。
- **`runGeometrySweep` が呼び手の `GpuContext` を受ける** — 採らない（決定 2）。推論中の device を渡す形を型の上で許す。

## Consequences

- **公開面**: `@karume/runtime` に 2 面目 `./tune` が増える。main の面は値が増えない。`AcquireGpuOptions.geometryProfile` の型が広がり、
  `GeometryProfile.provenance` の形が変わる（`adapter` が文字列から 4 欄へ — v0.x の Breaking・CHANGELOG に明記）。
  公開面スナップショット門は entry を `exports` から採る（`tests/helpers/public-surface.ts:84-96`）ので、fixture に `"./tune"` のキーが 1 つ増える。
- **配布物**: runtime の publish に約 3,300 行の掃引の核と較正定数が入る。`./tune` を import しないアプリのバンドルは変わらない。
- **記録形式とケース集合に semver の義務が付く**。ケース集合を変えるとケース集合の版が変わり、全利用者の保存した表が照合で不一致になる
  （再掃引）。ケースの追加はこの費用込みで判断する。
- **フラグ無しの Chrome で表を作れるようになる**。gpu-lab のプロファイルタブも量子化した記録を受ける（README「Before measuring」の
  「the profile tab rejects it」は書き換える）。
- **GEMM のカーネルを変える変更は、埋め込み 2 表の再生成（指紋の更新）を伴う**（決定 4）。
- **小さい端末**（推測）: f32 self attention M = N = 4096 の 4 ケースは 1 本 512 MiB〜1 GiB の束縛が要る。`maxStorageBufferBindingSize` が
  小さい端末では失敗行になり、attention 系の欄は既定のまま（安全側）になる。

### 影響ファイル

| 区分       | 移動元 → 移動先（または変更先）                                                                                                                   | 中身                                                         |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| 掃引の核   | `tools/geometry-sweep/harness.ts` → `packages/runtime/src/tune/harness.ts`                                                                        | 計測の規約・case plan・digest                                |
| 〃         | `tools/geometry-sweep/cases.ts` → `src/tune/cases.ts`                                                                                             | 形状表・`PROFILE_GEMM_ROWS_BOUNDS`・ケース集合の版           |
| 〃         | `tools/geometry-sweep/geometries.ts` → `src/tune/geometries.ts`                                                                                   | 候補集合                                                     |
| 〃         | `tools/geometry-sweep/derive.ts` の記録の parse・採否・`buildGeometryProfile`・JSON → `src/tune/derive.ts`                                        | 生成器・門（決定 3）・`provenance`（決定 4）                 |
| 〃         | `tools/geometry-sweep/report.ts` → `src/tune/report.ts`                                                                                           | 記録の形・足す欄（決定 8）                                   |
| 〃         | `tools/opbench/bench.ts` の `calibrateReps` / `ROUNDS` / `TARGET_PASS_MS` / `WARMUP_MIN_RUNS` / `WARMUP_NS` → `src/tune/`                         | opbench はここから import する                               |
| 〃         | `tools/anima-residency/timing.ts` の `looksQuantized` / `CHROME_TIMESTAMP_QUANTUM_NS` → `src/tune/`                                               | anima-residency はここから import する                       |
| 新規       | `src/tune/` の指紋・ケース集合の版・照合・parse                                                                                                   | 決定 4 / 5 / 7                                               |
| 公開面     | `packages/runtime/tune.ts`（新規）・`packages/runtime/deno.json`（`exports`）                                                                     | 決定 1 / 2                                                   |
| 注入口     | `packages/runtime/src/gpu/acquire.ts`・`mod.ts` の doc                                                                                            | コールバック形（決定 6）                                     |
| 表の型     | `packages/runtime/src/kernels/geometry-profile.ts`                                                                                                | `provenance` の形（決定 4）                                  |
| 生成物     | `packages/runtime/src/kernels/geometry-profiles/apple-metal-3.ts`・`nvidia-blackwell.ts`                                                          | 再生成（幾何の値は不変・`provenance` の形と指紋）            |
| 道具（殻） | `tools/geometry-sweep/main.ts`・`profile.ts`・TS の描画（道具側に残す）・`tools/gpu-lab/browser/*`（`injectable-tables.ts` の保存キーを `/2` へ） | `./tune` の利用者                                            |
| テスト     | 掃引の核の `tools/geometry-sweep/*_test.ts`（約 1,900 行）→ `packages/runtime/tests/`                                                             | テストの置き場は検証対象と同じ層（ADR 0008 追記 2026-09-03） |
| 文書       | `tools/geometry-sweep/README.md`・`tools/gpu-lab/README.md`・glossary・backlog・ACTIVE_DESIGN・CHANGELOG                                          | 同期                                                         |

### テスト

| テスト                          | 中身                                                                                                                                                                                                                    |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 新規 `parseGeometryProfileJson` | `1e999` の復元・末尾 `null` の名指し・型の誤り・各階層の未知の欄の拒否・`provenance` の形・`geometryProfileJson` との往復                                                                                               |
| 新規 `geometryProfileMismatch`  | 一致で `undefined`・adapter の 4 欄 / 指紋 / ケース集合の版を 1 つずつ違えて文言・`provenance` 無し                                                                                                                     |
| 新規 カーネルの指紋             | 同じ表で決定的・表の 1 欄の幾何を変えると変わる・`provenance` だけ変えても変わらない・埋め込み 2 表の `provenance.kernels` が今の値と一致                                                                               |
| 新規 ケース集合の版             | ケースを 1 本足すと変わる・境界を変えると変わる                                                                                                                                                                         |
| 門（生成器）                    | 量子化した記録を受ける・E > 1% の観測だけが外れ、採否の行に E が残る・既定の行の e が大きいとケースの比が全部外れる・壁時計の記録は拒む。故障注入 3 通りで赤: 上界の判定を外す・既定の行の e を足さない・壁時計を受ける |
| 記録の欄                        | `caseSet` / `defaultKernels` が今の値と違う記録を生成器が拒む・欄の無い記録は受ける                                                                                                                                     |
| 注入口（実 GPU・B570）          | コールバックが受ける `adapterInfo` が `GpuContext.adapterInfo` と同じ・戻した表の幾何判別子が実走キーに載り既定と Uint32 一致・`undefined` で自動選択                                                                   |
| 注入口（偽の `navigator.gpu`）  | コールバックが壊れた表を返す / 投げると `requestDevice` が 0 回・コールバックは 1 度だけ呼ばれる                                                                                                                        |
| 公開面                          | fixture に `"./tune"` が増え、main の面の値は不変                                                                                                                                                                       |
| 不変                            | codegen スナップショット（WGSL・キー）無変更・埋め込み 2 表の `--check` がバイト同一・移したテストが同じ件数で緑                                                                                                        |

### 段階分解

| 段 | 中身                                                                                                                                      | 条件                                                       |
| -: | ----------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
|  1 | 本 ADR・ADR 0008 / 0115 の追記                                                                                                            | この commit                                                |
|  2 | `src/tune/` への移動と `./tune` の `exports`（決定 1 / 2）。CLI と gpu-lab を殻にする。生成の振る舞いは変えない                           | —                                                          |
|  3 | 計測の門の改定（決定 3）                                                                                                                  | 段 2 の後                                                  |
|  4 | `provenance` の構造化・カーネルの指紋・ケース集合の版・照合・parse（決定 4 / 5 / 7）。埋め込み 2 表の再生成・gpu-lab の保存キーを `/2` へ | 段 2 の後。2 表の再生成は ADR 0116 の段 4 と順序を合わせる |
|  5 | コールバック形の注入口（決定 6）                                                                                                          | 段 2 の後（段 3 / 4 と並行可）                             |
|  6 | 記録の欄（決定 8）                                                                                                                        | 段 4 の後（指紋とケース集合の版を使う）                    |
|  7 | 検収と docs の同期                                                                                                                        | 段 3〜6 の後                                               |

## 検収

| 段 | 緑の条件                                                                                                                                                                                                                         | 結果                                                                                                                                                                                                                                                                           |
| -: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
|  1 | `deno fmt --check` 緑                                                                                                                                                                                                            | ✅（2026-10-01・`cb88cb33` ADR + ADR 0008 / 0115 追記・`51724541` 決定 4 の裁定）                                                                                                                                                                                              |
|  2 | 移したテストが同じ件数で緑・codegen スナップショット無変更・2 表の `--check` バイト同一・公開面 fixture の差分が `"./tune"` の追加だけ・`deno publish --dry-run` 緑・gpu-lab で quick の掃引 → 生成 → 注入が通る（利用者の実機） | ✅（2026-10-01・`d242008f`）: 移したテストは葉 114 件で同数・codegen スナップショット無変更・2 表 `--check` バイト同一・公開面 fixture は `./tune` の 16 シンボル追加のみ・`deno publish --dry-run` 緑・フル verify 緑。gpu-lab の実機の往復（掃引 → 生成 → 注入）は利用者待ち |
|  3 | 門の単体テストと故障注入 3 通りの赤。2026-09-29 の 3 機の full を丸めの模擬（各 200 試行）で新しい門に通し、全欄の採否が原本と一致し、外れる観測は RTX の 1 行だけ。2 表の `--check` バイト同一（入力は非量子化で q = 0）        | —                                                                                                                                                                                                                                                                              |
|  4 | 新しい純関数の単体テスト緑。2 表を再生成し、幾何の値が不変・`--check` バイト同一・`provenance.kernels` の一致テスト緑。照合 1 回の所要を記録する                                                                                 | —                                                                                                                                                                                                                                                                              |
|  5 | 注入口の実 GPU テスト（B570）と偽の `navigator.gpu` のテストが緑。故障注入 2 通り（コールバックの戻りを門に通さない・device を作った後で呼ぶ）で赤                                                                               | —                                                                                                                                                                                                                                                                              |
|  6 | 記録の欄の単体テスト緑。gpu-lab の quick+ の記録に `startedAt` / `aborted` / `cases[].elapsedMs` / `caseSet` / `defaultKernels` が載る（利用者の実機）                                                                           | —                                                                                                                                                                                                                                                                              |
|  7 | フラグ無しの Chrome（M2 か M5）で、アプリの流れ（掃引 → 生成 → 保存 → 再起動 → 照合 → コールバックで注入）が通り、診断 `geometryProfile` が保存した表の id になる。フル verify 緑                                                | —                                                                                                                                                                                                                                                                              |

## 未解決

- **しきい値 1% の実測での詰め**。根拠は 2026-09-29 の 3 機の full と、同じ測定値を再利用した丸めの模擬だけ。フラグ無しで測り直した記録では、
  熱や揺れの差が丸めより大きいはず（それは再測定比の門が受け持つ）。Chrome の丸め方式（各 timestamp の切り捨てか別の方式か）も未確認。
  1 刻み未満という上界は格子への切り捨てでも最近接でも成り立つが、揺らぎ（jitter）を足す方式なら成り立たない。
- **アプリ内の表の保管先**（localStorage・IndexedDB・ファイル）と保管の失敗（容量・プライベートモード）の扱いは、runtime の範囲外。
- **M5 の表の登録**は ADR 0116 と同じく本 ADR の範囲外。M5 の利用者はアプリ内の掃引で自分の表を作れるようになる。
- **JSR 公開時の `./tune` のサイズと tree-shaking**。`deno bundle` での実測（Context）はあるが、JSR から入れた利用者のバンドラでの落ち方と、
  照合だけを import したときに掃引の核が落ちるかは未測定。
- **モバイル（8 GB 級 Android）**。床の大きさ・1 GiB 級の束縛の可否・メモリ逼迫での device lost の危険（known-issues の Pixel の項）は未観測。
  測った 3 機はどれもデスクトップ級。
- **壁時計の記録**。timestamp-query を出さない端末では表を作れないまま。床を行ごとに記録して上界を出す形は、そういう端末が実際に出てから検討する。
- **行ごとの段の時間**（compile / probe / warmup / rounds / digest）は記録に足していない。空回しが所要の約半分を占めるので、短縮を検討するときの材料になる。
- **カーネルの指紋の照合 1 回の所要**（推測: 数 ms〜数十 ms）は段 4 で測る。大きければ指紋の範囲（欄の全ケースか、代表 1 ケースか）を見直す。
