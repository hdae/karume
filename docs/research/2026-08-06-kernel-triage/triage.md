# カーネル・実行系トリアージ（2026-08-06）

> NOTE: 時点スナップショット。数値は 2026-08-06 時点の現行 IR・コード・既存実機記録に基づき、再 export や実機再計測前の値を含む。
>
> NOTE: 本ファイルは**参照実装ブランチ（`codex/kernel-quick-fixes`）での triage 記録を原文のまま持ち込んだもの**で、本リポ main の実装状態とは一致しない（実測は Apple M2 / Metal と RTX 3080 Ti / Vulkan）。
> main への採否と本リポでの再実測は [2026-08-08-branch-adoption-perf.md](../2026-08-08-branch-adoption-perf.md) が正本。

## 目的と読み方

macOS / Metal、Dawn / wgpu、NVIDIA、AMD の差を意識しつつ、カーネル、codegen、
exporter、実行計画の「遅さ・誤値・無駄な dispatch のシグナル」を優先度順に並べる。

状態:

- ✅ **対応済み**: このブランチで実装と回帰テストまで完了。
- 🧪 **実機待ち**: 候補実装と比較手段は完了したが、対象backendの実測前なのでproduction既定は維持。
- 🟨 **設計候補**: 信号と方向は整理済みだが未実装。
- ⬜ **要検証**: 怪しいが、実機プローブまたは詳細設計が先。
- 「削減」は静的な dispatch 数の試算であり、壁時計の改善率ではない。
- 誤値の可能性がある項目を速度だけの項目より優先する。

作業ブランチは `codex/kernel-quick-fixes`。対応済み項目は実装・回帰テストと同じ単位で記録する。

## 最終検証

- fmt、lint、公開modのtype checkは通過。本体packagesを明示した `deno test -A` は
  **563 passed / 0 failed**。実GPUの全runtime test、Animaの1024 / 512 golden SHA-256を含む。
- `uv run ruff check karume/normalize.py tests/test_normalize.py`: 通過。
- `uv run pytest -q`: **1,894 passed / 137 skipped**。
- 上記一括 GPU 検証はこの作業環境の Linux / wgpu-Vulkan。Apple M2 / Metalでは、
  KQF-005の直接A/BとAnima 1024既定demoのbefore / afterをユーザー環境で実測した。
  KQF-008のMN32候補まで実測し、tile縮小候補の評価を完了した。
- OP-017追加後はruntime全体を明示して **419 passed / 0 failed**。Anima w8a8-s16 / 512 / 8stepの
  PNG SHA-256も既存goldenと一致した。
- OP-018追加後はruntime全体を明示して **423 passed / 0 failed**。Anima w8a8-s16 / 512 / 8stepの
  PNG SHA-256も同じ既存goldenと一致した。
- OP-019追加後はruntime全体を明示して **424 passed / 0 failed**。Anima w8a8-s16 / 512 / 8stepの
  PNG SHA-256も既存goldenと一致した。
- OP-007のdirect-mul-first拡張後もruntime全体は **424 passed / 0 failed**。同じAnima 512 / 8stepは
  8.6秒で完走し、PNG SHA-256も既存goldenと一致した。
- direct-mul-firstのApple M2 / Metal A/BもH8 / H16の両方でbit一致し、8.630 / 5.250倍。288 query/run、
  timestamp資源10組でcounter sample buffer枯渇なく完走した。

## 今回の対応

| ID        | 状態        | 優先度 | 対象                | 症状 / 無駄                                                                                                        | 対応                                                                                            |
| --------- | ----------- | ------ | ------------------- | ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| KQF-001   | ✅ 対応済み | P1     | PipelineCache       | 同一キーを同時に要求すると、未決着結果が Map に入る前に shader module / pipeline / validation scope を重複生成する | 未決着の `Promise<GPUComputePipeline>` 自体をキャッシュ。失敗時だけ削除し再試行可能にした       |
| KQF-002   | ✅ 対応済み | P1     | elementwise WGSL    | 全入力が出力と同 shape でも、要素ごと・軸ごとに u32 の `/` と `%` で broadcast 座標を復元する                      | 同 shape の全入力を `inK[i]` で読む `:contiguous` 変種を追加。broadcast 変種とキーを分離        |
| KQF-003   | ✅ 対応済み | P1     | exporter normalize  | 恒等 `permute` と隣接鎖が各段で strided copy を実体化する                                                          | 単独恒等を除去。鎖は `inner[outer[i]]` で合成し、恒等なら除去、非恒等なら 1 permute へ短縮      |
| KQF-004   | ✅ 対応済み | P0     | linear GEMM         | `acc: array<vec4<f32>,4>` を内積・storeの両方で動的添字し、Metalでlocal memoryへ退避される疑い                     | 実行経路に名前付き `acc0..3` へcodegen展開する `:u4` 変種を追加。旧生成物は既定値で維持         |
| KQF-005   | ✅ 対応済み | P0     | i8a8 linear GEMM    | Anima既定presetはw8a8なのでKQF-004を通らず、整数GEMM側に同じ動的accumulator添字が残る                              | M2 / RTXでbit一致と全代表shapeの改善を確認し、executorをi8a8 `:u4`へ切り替え                    |
| KQF-006   | ✅ 対応済み | P0     | i8a8 linear M-tile  | M2では固定256 threadのM64×N64がoccupancyを制限する可能性。128 threadのM32×N64を分離して測る必要がある              | M2 / RTXで直接A/B。M2は加重1.036倍だが形状退行が大きいため、一律採用せずM64を維持               |
| KQF-007   | ✅ 対応済み | P0     | i8a8 linear N-tile  | M32の形状別結果が混在したため、AではなくW側の再読込を避けるM64×N32もbackendごとに評価する必要がある                | M2 / RTXとも全454本bit一致だが加重0.896 / 0.949倍。一律採用せずM64を維持                        |
| KQF-008   | ✅ 対応済み | P0     | i8a8 linear MN-tile | 128 thread候補が形状・runで混在するため、64 threadのM32×N32でoccupancy仮説の残りを分離する                         | M2 / RTXとも全454本bit一致だがpaired加重0.846 / 0.807倍。tile縮小を終えてM64を維持              |
| KQF-009   | ✅ 調査完了 | P1     | i8a8 linear K-tile  | K16はK-loop 16要素ごとにbarrierを2回通る。K32で同期頻度を半減できるが共有loadは2パスになる                         | RTXはpaired 0.974倍。M2はpaired 1.044倍だが分散が大きく、実時間比0.919倍。一律採用せずK16を維持 |
| QUANT-010 | ✅ 対応済み | P1     | i8a8 linear量子化   | 454本のlinearが同じactivationを再量子化し、195 dispatchと同じ入力走査を重複                                        | run-local・同一IR値名だけで量子化結果を共有。実グラフでlinear量子化454→259本                    |
| OP-007    | 🧪 E2E待ち  | P1     | half-split RoPE     | DiT 56鎖/predictに加えtext encoderの55鎖/runも同じ8 dispatch列を使う                                               | 両順のexact 7-node鎖だけを融合。textはM2でbit一致・H8 8.630倍 / H16 5.250倍                     |
| OP-017    | 🧪 E2E待ち  | P1     | VAE nearest x2      | 3鎖のreshape / expand列が6 dispatchと、合計468MiBのexpand出力書込みを使う                                          | exact 6-node鎖だけをu32 bit-copyで融合。M2はbit一致・4.268倍、RTXはbit一致・2.200倍             |
| OP-018    | 🧪 E2E待ち  | P2     | SiLU exact peephole | 1024 / 8stepで305鎖がsigmoid→mulの610 dispatchと、論理20.27GiBの中間write / readを使う                             | strict隣接鎖だけを2→1 dispatchへ融合。M2はfinite-bit一致・2.394倍、RTXは同2.424倍               |
| OP-019    | ✅ 対応済み | P1     | identity expand     | text / conditionerの恒等expand 160本が同じshapeをstrided copyし、論理50.58MiBをread / writeする                    | resolved入出力shape完全一致時だけreshape同様にbuffer alias化。非恒等expandは従来copyを維持      |

### KQF-001: 未決着 pipeline の共有

旧実装は `await withValidationScope(...)` の後で初めて Map へ登録していた。したがって同じ
event-loop turn で 2 回 `get(key, wgsl)` を呼ぶと、両方が生成を開始した。

修正後は次の性質を持つ。

- 同一キー・同一 WGSL の同時要求は 1 本の Promise と生成結果を共有する。
- 同一キー・異なる WGSL は、生成中でも従来どおり即座に
  `PipelineKeyConflictError` になる。
- validation failure は失敗 Promise を Map から除き、次回の再試行を妨げない。
- 回帰テストは旧実装で「異なる pipeline id」となって失敗し、修正後は 5/5 通過。

主な変更:
`packages/runtime/src/gpu/pipeline-cache.ts`,
`packages/runtime/tests/gpu_pipeline_cache_test.ts`。

### KQF-002: contiguous elementwise

現行 elementwise は出力を線形走査しながら、broadcast のために線形 index を rank 座標へ戻し、
入力 stride で再び線形 index にする。全入力が出力と同 shape なら、この往復は不要である。

静的に「全入力 shape = 出力 shape」と判定できる dispatch の候補数:

| グラフ       | contiguous 候補 / elementwise |  比率 |
| ------------ | ----------------------------: | ----: |
| Transformer  |                     398 / 764 | 52.1% |
| text encoder |                     308 / 448 | 68.8% |
| conditioner  |                      96 / 144 | 66.7% |
| VAE          |                     194 / 254 | 76.4% |
| 合計         |                   996 / 1,610 | 61.9% |

修正は数値式と params 配置を変えず、ロード index だけを `i` にする。rank 0 は従来どおり
長さ 1 の rank 1 に正規化して判定する。全 broadcast ケースは従来 WGSL と従来キーのまま。

検証:

- contiguous WGSL に座標復元の `/` / `%` が無い。
- binary も含め全入力が `inK[i]` になる。
- contiguous と broadcast のキーが全 op × dtype × rank で衝突しない。
- 既存 WGSL スナップショットはバイト単位で不変。
- codegen + pipeline cache の対象テストは 56/56 通過。
- Metal 実機での壁時計改善率は未計測。

主な変更:
`packages/runtime/src/codegen/elementwise.ts`,
`packages/runtime/src/runtime/executor.ts`,
`packages/runtime/tests/codegen_wgsl_test.ts`。

### KQF-003: permute identity removal / adjacent composition

現行 IR では permute は view ではなく strided copy 1 dispatch である。現行モデルの最大鎖を
静的に数えると、合成後の dispatch 削減候補は次のとおり。

| グラフ       | 観測した鎖                                | 合成後の削減候補 |
| ------------ | ----------------------------------------- | ---------------: |
| Transformer  | 長さ 3 の非恒等鎖 56、長さ 2 の恒等鎖 112 |              336 |
| text encoder | 同種の短縮                                |               56 |
| conditioner  | 同種の短縮                                |               36 |

FX graph 上で隣接する場合だけを対象にし、clone / reshape / 他 op をまたがない。中間 permute に
別の consumer がいても、下流 permute の入力だけを base へ付け替えるので、その consumer は
保持される。DCE は既存の正規化順序で行う。

検証:

- 単独の恒等 permute が 0 op になる。
- 逆 permute 2 段が 0 op になる。
- 非恒等 2 段が期待した軸順の 1 permute になる。
- `test_normalize.py` は 53/53 通過。
- 上表の実モデル値は修正前 IR の静的集計。修正後の実モデル再 export は未実施。

主な変更:
`tools/exporter/karume/normalize.py`,
`tools/exporter/tests/test_normalize.py`。

### KQF-004: linear GEMM の静的 accumulator

現行64×64 GEMMは、各threadの16出力を `array<vec4<f32>,4>` に置き、内積とstoreで
`acc[i]` を使う。WGSLとしては合法だが、backendがループを展開できない場合は配列が
addressableな関数ローカル領域に残り、特にMetalでregister spillになる疑いがある。

今回は実モデルで本数の多い通常のlinearだけを先行させた。

- `:u4` 変種は `acc0`〜`acc3` の4本の `vec4<f32>` を生成し、内積とstoreの両方から
  `acc[` を除く。Kタイル16、`kk`の昇順、各出力の加算順序は変えない。
- executorはf32/f16計算とf32/f16/i8重み格納の通常linearで `:u4` を選ぶ。
  w8a8の整数GEMMは別生成器なので今回の対象外。
- 既存の公開引数は既定 `false` とし、従来WGSLスナップショットをバイト単位で維持。
  キー末尾の `:u4` で旧変種とのcache衝突を防ぐ。

検証:

- codegen 52/52通過。新変種に `acc[` が無いこと、4更新の静的展開、キー分離を固定。
- Linux / wgpu-Vulkanの実GPU 34/34通過。linearのスカラ/v4、f32/f16計算、
  f16/i8重みを含み、f16計算は既存オラクルとビット単位で一致。
- Apple Silicon / Metalのbefore/afterは未計測。したがって対応済みなのは
  「動的accumulator添字の除去」であり、Metalでの速度改善率そのものではない。

主な変更:
`packages/runtime/src/kernels/gemm.ts`,
`packages/runtime/src/kernels/linear.ts`,
`packages/runtime/src/runtime/executor.ts`。

### KQF-005: Anima既定w8a8 linearの静的 accumulator A/B

`models/anima-turbo/karume.json` の既定presetは `w8a8-s16` で、linear計算は
`linearCompute: "i8a8"` を使う。そのためKQF-004の通常linear `:u4` は、報告された
728.2秒 / 729.4秒の既定demoでは実行されていなかった。

整数GEMM生成器にも同じ静的展開を追加し、直接A/Bで対象backendの数値と速度を確認してから
executorを `:u4` へ切り替えた。

- `linearI8a8Wgsl(..., staticAccumulator = false)` は既定生成物を維持する。
- 候補は `acc0`〜`acc3` の4本へ展開し、dp4a / dp4a emulation、scalar / vec4 storeの
  全組合せから `acc[` を除く。pipeline keyは末尾 `:u4` で分離する。
- `deno task bench:linear-i8a8` はpipeline compile、buffer初期化、readbackを計測外に置き、
  ABBA / BAAB順、全出力のbit比較、JSON保存を行う。timestamp-queryが無ければ複数dispatchを
  束ねたwall timeへ自動フォールバックする。
- 当初の6形状は現行Animaのi8a8 linear 449/454本を代表する。現行ベンチは特殊shape 5本も加え、
  全11形状・454/454本を重み付けする。

Linux / RTX 3080 Ti / wgpu-Vulkanの予備結果（timestamp-query、dp4a）では、全6形状が
bit一致し、node数で重み付けしたkernel時間は863.812msから604.476ms、**1.429倍**だった。
dp4a emulationとwall-time fallbackも別の短いsmokeでbit一致した。これは候補とharnessの
成立確認として使った。

Mac / Apple M2 / wgpu-Metalの本計測（timestamp-query、dp4a、8 round）では全6形状が
bit一致した。大形状4本は1.165〜1.185倍、小形状2本は1.158〜1.662倍で、退行shapeは無い。
node数で重み付けしたkernel時間は86,394.209msから73,509.902ms、**1.175倍**だった。
大形状のsample分散は大きいが、中央値の改善方向は全shapeで一致する。

事前に置いた1.25倍の単一閾値には届かなかった。一方でM2 / RTXの両方で全代表shapeが改善し、
全出力がbit一致したため、backend名による分岐を増やさずportable既定として採用した。
productionのGPU回帰はv4 / scalar store、dp4a / emulation、タイル端、K端数を含み、
TS参照とatol=0で一致する。

MacのAnima 1024既定demo実測は729.4秒から625.2秒へ短縮し、**1.167倍**、104.2秒減だった。
linear直接A/Bの重み付き1.175倍とほぼ一致し、pipeline compile、量子化、他opを含むE2Eでも
改善が反映された。

主な変更:
`packages/runtime/src/kernels/linear-i8a8.ts`,
`packages/runtime/src/runtime/executor.ts`,
`packages/runtime/tests/codegen_wgsl_test.ts`,
`packages/runtime/tests/gpu_i8a8_test.ts`,
`tools/bench/linear-i8a8.ts`。

### KQF-006: i8a8 linearのM32×N64 / 128-thread候補

M64×N64は16×16、256 invocationで1 workgroupを構成する。Apple GPUでthread数がoccupancyを
制限している可能性を分離するため、M方向だけを32へ縮めた16×8、128 invocation候補を追加した。

- 各threadの4×4出力、Nタイル64、Kタイル16、整数縮約順、store式はM64と同じ。
- x共有タイルは256語から128語へ半減する。W共有タイルは256語を維持し、128 threadで
  出力チャネル方向へ2パス充填する。M方向のworkgroup数は2倍になる。
- 公開生成関数の既定値とexecutorはM64のまま。M32はベンチからだけ選べるため、
  対象backendの実測前にproduction挙動を変えない。
- mTileは32 / 64だけを受理し、M32のpipeline keyは
  `reg32x64[v4]:wg16x8` を含む。誤った任意幾何とcache衝突を生成時に拒否する。
- `deno task bench:linear-i8a8` は同じ入力bufferとABBA / BAAB順でM64 / M32を測り、
  shapeごとに全出力のbit一致を確認する。`speedup = M64 / M32` なので1超がM32優位。

Linux / RTX 3080 Ti / wgpu-Vulkan（timestamp-query、dp4a、3 round）は全6形状bit一致。
重み付きkernel時間はM64 583.997ms、M32 690.423msで **0.846倍**となり、RTXではM64を
維持すべき結果だった。一方、crossは1.019倍、modulation-6144は1.233倍でM32が上回った。
dp4a emulationも別smokeでbit一致した。

Mac / Apple M2 / wgpu-Metal（timestamp-query、dp4a、8 round）でも全6形状がbit一致した。
attentionは1.123倍、ffn-downは1.066倍、modulation-6144は1.591倍だった一方、
ffn-upは0.858倍、crossは0.845倍まで退行し、modulation-256は同値だった。
重み付きkernel時間はM64 71,863.957ms、M32 69,368.031msで **1.036倍**。

全体では小幅に改善したが、主要shapeの大きな退行を別shapeの改善で相殺した結果である。
RTXも含め一律のportable既定には不適切なので、productionはM64のまま維持する。
このM32評価は完了とし、次は再読込する共有タイルを入れ替えるN32候補を分離して測る。

### KQF-007: i8a8 linearのM64×N32 / 128-thread候補

M32×N64はM workgroupを2倍にし、同じW領域を行タイルごとに読み直す。対になる候補として
Mを64に戻し、Nだけを32へ縮めた8×16、128 invocationのM64×N32を追加した。

- 各threadの4×4出力とKタイル16は維持する。x共有タイルは256語のまま128 threadで
  行方向へ2パス充填し、W共有タイルは128語へ半減して1パスで充填する。
- N方向のworkgroup数は2倍になるため、M32とは逆にx領域を列タイルごとに読み直す。
  AとWのどちらの再読込がbackend / shapeで軽いかを直接比較できる。
- 公開生成関数の既定値とexecutorはM64×N64のまま。N32 keyは
  `reg64x32[v4]:wg8x16` を含み、production pipelineと衝突しない。
- ベンチへ `--candidate n32` を加え、現行Animaの特殊shape 5本も含む454/454本を覆う。

Linux / RTX 3080 Ti / wgpu-Vulkan（timestamp-query、dp4a、3 round）は全11形状bit一致。
重み付きkernel時間はM64 573.080ms、N32 603.881msで **0.949倍**だった。
attention / ffn-up / ffn-downは0.950 / 0.948 / 0.939倍でRTXの主要shapeには不利だが、
crossは1.026倍、M=1の一部は1.080〜1.145倍だった。dp4a emulationのsmokeもbit一致した。

初回のMac / Apple M2 / wgpu-Metal計測は8形状・451/454本までbit一致し、暫定1.057倍だったが、
9形状目でcounter sample buffer確保に失敗した。旧harnessがroundごとにquery setを作成・破棄し、
ちょうど64組の後で停止したため、3リソースを全shapeで再利用し、部分JSONも保存するよう修正した。

修正後のM2本計測（8 round）は全11形状・454/454本bit一致。重み付きkernel時間は
M64 63,157.932ms、N32 70,471.743msで **0.896倍**だった。attention / ffn-up / ffn-down /
crossは0.890 / 0.814 / 0.993 / 0.984倍で、主要shapeに採用根拠は無い。
初回途中値ではffn-upが1.359倍だったものが完走runでは0.814倍へ反転し、p10 / p90も大きい。
独立sample中央値1回だけでshape selectorを固定するのは危険と判断し、N32評価を完了してM64を維持する。

### KQF-008: i8a8 linearのM32×N32 / 64-thread候補とpaired指標

128 thread候補でoccupancy仮説が決着しなかったため、M / Nを両方32へ縮めた8×8、
64 invocation候補を追加した。A / W共有タイルは各128語で、64 threadが各2パスで充填する。
M / N両方向のworkgroupが2倍になりA / Wをともに読み直すため、純粋に低thread数の効果を見る候補である。

- 生成器は64×64 / 32×64 / 64×32 / 32×32の4幾何だけを受理する。
  MN32 keyは `reg32x32[v4]:wg8x8` を含み、既定生成物とexecutorは変更しない。
- ベンチへ `--candidate mn32` を追加した。
- Metalの二峰性対策として、同一ABBA / BAAB round内の各変種2 sampleを平均してから比を取り、
  round比の中央値とp10 / p90を `paired` として出す。全体もroundごとに454ノード重みを付ける。
  従来の独立中央値比は比較継続のため `speedup` として残す。

Linux / RTX 3080 Ti / wgpu-Vulkan（timestamp-query、dp4a、3 round）は全11形状bit一致。
重み付きkernel時間はM64 565.931ms、MN32 700.139msで0.808倍、pairedは **0.807倍**
（p10 / p90 0.806〜0.809倍）だった。dp4a emulationもattention smokeでbit一致した。
RTXでは明確に不採用だった。

Mac / Apple M2 / wgpu-Metal（timestamp-query、dp4a、8 round）も全11形状・454/454本bit一致。
重み付きkernel時間はM64 78,650.149ms、MN32 100,450.277msで0.783倍、pairedは **0.846倍**
（p10 / p90 0.812〜0.863倍）だった。主要3形状のpairedはattention 0.884倍、ffn-up
0.843倍、ffn-down 0.859倍で全て明確に退行した。M=1の一部は改善したがnode数が少なく、
portableなshape selectorを正当化しない。64 / 128 thread候補の評価を完了し、productionはM64を維持する。
次はbackend固有tileではなく、K方向の再利用・再量子化共有・dispatch削減を優先する。

### KQF-009: i8a8 linearのK32候補

出力tile、256 thread、4×4の静的accumulatorを変えず、K tileだけを16要素（4 pack）から
32要素（8 pack）へ広げた。A/Wの共有tileは各256語から512語になり、256 threadが各2パスで
埋める。workgroup memoryは合計2KiBから4KiBへ増える一方、K-loopのbarrier epochは半分になる。

- 公開生成関数の既定K16、既存WGSL snapshot、pipeline key、executorは変更しない。
  K32は `linear:v3:i8a8:...:k32:...` の別keyで、`--candidate k32` のbenchからだけ選べる。
- K32とM/N縮小の直積は増やさず、M64×N64だけを受理する。同期頻度以外の軸を混ぜない。
- codegen testは512語の共有tile、A/W各2パス、8 pack内積、K16 key不変、不正K/幾何の拒否を固定する。

Linux / RTX 3080 Ti / wgpu-Vulkan（timestamp-query、dp4a、5 round）は全11形状・454/454本bit一致。
重み付きkernel時間はK16 584.100ms、K32 599.678msで0.974倍、pairedも **0.974倍**
（p10 / p90 0.969〜0.978倍）だった。attention / ffn-up / ffn-down / crossは
0.980 / 0.977 / 0.973 / 0.945倍で、RTXでは全代表shapeが退行した。dp4a emulationも
attentionとK端数68のsmokeでbit一致した。

Mac / Apple M2 / wgpu-Metal（timestamp-query、dp4a、8 round）も全11形状・454/454本bit一致。
重み付きkernel時間はK16 75,154.750ms、K32 81,816.391msで **0.919倍** だった。pairedは
1.044倍だがp10 / p90が0.823〜1.106倍と両側へ大きく跨ぎ、attention / crossは0.993 / 1.002倍、
ffn-up / ffn-downも1.026 / 1.073倍に留まる。RTXの安定した0.974倍と合わせると、共有memoryを
倍増してK32をbackend selectorへ加える根拠はない。K tile拡大を終了し、productionはK16を維持する。

### QUANT-010: i8a8 linearの行量子化共有

同じactivationを直接読むi8a8 linear fan-outについて、`quantize_rows`の出力
（packed i8 data + per-row scale）を1 run内で共有する。数値式とWGSLは変えず、最初のconsumerだけが
量子化をdispatchし、全consumerが同じ2 bufferを読む。

- 実行計画をrunごとに先読みし、productionと同じ `linearCompute === "i8a8"`、i8常駐重み、
  `k % 4 === 0` の条件を満たす直接consumerだけを数える。全opの入力数である `countUses` は
  非i8 consumerまで含むため使わない。
- cache keyはIR値名の完全一致。reshapeは同じGPUBufferを異なる `[m,k]` の行境界で別名化できるため、
  buffer identityでは共有しない。runを越えるcache、device generation、prepared public APIも足さない。
- `RunArena`には適格consumer数ちょうどの参照を積み、各GEMM後に1本ずつ返す。3本のlinear間に
  非i8 ReLU consumerを挟んだGPU回帰で、TS参照とのbit一致、arena drain、7→5 dispatchを固定した。

実配布transformerのsafetensors headerを再集計すると、i8a8適格linearは454本、共有対象は30値だった。
fan-outは3本×28値、56本×1値、85本×1値で、削減は **195 dispatch/predict**
（linearの行量子化454→259本）。S=4096で共有値の静的な同時保持上限は約8.52MiBだった。

Linux / RTX 3080 Ti / wgpu-Vulkanの実DiT 1 predictでも総dispatchは3,301→3,106本となり、
予測どおり195本減った。S=4096のtransient peakは712.55→713.05MiB（+0.50MiB）で、
最大保持量がそのまま実ピークへ加算される形ではなかった。単発timestampでは総GPU時間は
1,104.7→1,108.4msとノイズ内、`quantize_rows`合計は56.43→51.67msだった。
RTXで大幅な壁時計改善を主張する材料ではなく、次の判定はM2のAnima E2Eとfan-out直接A/Bで行う。

Mac / Apple M2のAnima 1024既定runは共有前625.2s、共有後626.9sで、単発差は約+0.3%の誤差圏だった。
195 dispatch削減は実グラフで確認済みだが、M2のend-to-end律速を動かす効果は観測できなかった。
数値式を変えず量子化時間そのものを減らすportableな共有として維持し、追加のcache拡張は行わない。

### OP-007: half-split RoPE の exact peephole fusion

配布中のAnima transformer（i8 / f16の両方）は28 blockあり、各blockのQ / Kに1本ずつ、
合計 **56鎖** のhalf-split RoPEがある。旧調査の112本はQ/Kを二重計上していたため訂正した。
1鎖は連続7 nodeだが、catが入力ごとにcopyを出すため実際には8 dispatchである。

    slice(x, 0:64), slice(x, 64:128), neg(second), cat(neg, first),
    mul(x, cos), mul(cat, sin), add

text encoderにも同じhalf-split式が56鎖ある。このうち **55鎖**（H=8が28本、H=16が27本）は
`mul(x, cos)`がsliceより先に置かれた次の連続7 nodeで、同じprivate kernelへ安全に畳める。

    mul(x, cos), slice(x, 0:64), slice(x, 64:128), neg(second),
    cat(neg, first), mul(cat, sin), add

残るH=16の1鎖はcatとcross mulの間に`sin`の`sym_prefix_slice`が入り、その時点では入力bufferも
未生成である。非連続scanやスケジュール変更は行わず、意図的に既存9 dispatch列へfallbackする。

public IR / op契約 / exporterは変えず、executorが次の全条件に一致する鎖だけをprivate kernelへ
畳む。1点でも外れれば従来node列をそのまま実行する。

- 7 nodeが連続し、slice-first / direct-mul-firstいずれかの結線順とslice / cat attrsが上記と完全一致。
- 全値がf32、x=[1,H,S,128]、cos / sinが[1,1,S,128]。
- 最終addより前の内部6値はuse countが1で、graph outputではない。
- Q / Kは別dispatch。偶奇RoPE、別head幅、別broadcast、Q/K同時処理へ一般化しない。

kernelは256 threadの1D grid-stride、16-byte uniform、x / cos / sin / outの4 storage binding、
2KiBのworkgroup memoryだけを使い、subgroup、atomics、f16、optional featureに依存しない。
素朴に1式へ畳むとWGSL backendの積和融合・再結合でAnima goldenが変わったため、2本の乗算結果を
vec2<u32>のworkgroup配列へ一度書き、uniform barrier後にf32へ戻して加算する。これにより
primitive間のstorage書込みが作っていたf32丸め境界を保つ。

transformer実グラフのdispatchは448→56、**392 dispatch/predict削減**。text encoderは
440→55、**385 dispatch/generate削減**。既定8 step / guidance=1では両者の対象合計が
4,024→503 dispatch、**3,521削減**となる。textの既定T=29では論理tensor traffic proxyが
111.469→46.445MiB（**65.023MiB削減**）、確実に消える中間materialized writeは41.801MiBである。
専用`deno task bench:rope`はS=4096の全出力をprimitive列とbit比較した後、readbackを小さくした
同一shapeをABBA / BAAB順で測る。Linux / RTX 3080 Ti / wgpu-Vulkan（timestamp-query、5 round）は
全8,388,608要素がbit一致し、primitive 0.589ms、融合0.166ms、独立中央値 **3.543倍**、
paired **3.627倍**（p10 / p90 3.621〜3.636倍）だった。56鎖の単純合算は
32.970→9.307msだが、これはkernel時間の試算でend-to-end改善率ではない。

Apple M2 / Metalの初回A/BはS=4096の全出力parityを完了してbit一致したが、計測中に
`MTLCounterErrorDomain: Cannot allocate sample buffer`からdevice lostとなった。これはRoPEの
数値失敗ではなく、各run / chunkで作るtimestamp query資源をDawn / Metalが遅延解放する既知の
ベンチ側制約である。

ベンチv2はparityをtimestamp無効の別deviceへ分離し、計測時はABBA / BAABの4鎖を18 dispatchの
1 graph / 1 submitへまとめた。`ROPE_KEY`の2 dispatchを融合時間、残り16 dispatchをprimitive時間
として各2鎖で割る。`--rounds 8 --warmup 2`でquery set生成は旧版の少なくとも38回から10回へ減る。
RTX 3080 Tiでv2を再検証し、全8,388,608要素bit一致、0.592→0.157ms、独立3.758倍、paired
3.756倍で完走した。

Apple M2 / Metalもv2の8 roundを完走し、同じchecksum `4833c46e`でbit一致した。primitive
12.332ms、融合2.096ms、独立 **5.885倍**、paired **5.066倍**（p10 / p90 4.492〜5.890倍）。
56鎖の単純合算は690.566→117.353ms、**573.213ms / predict削減**となる。direct A/Bの採用門は
通過し、残る確認はAnima 1024 E2Eとする。

direct-first用には既存ベンチへ実asset固定の`text-h8` / `text-h16` profileを追加した。短いT=29を
安定して測るため1 graph内で8回反復して1鎖時間へ正規化し、mixed runを144 dispatch / 288 queryに
留める。RTX 3080 Ti / Vulkan（4 round）は両shapeともbit一致し、H=8は0.026→0.003ms、独立
7.571倍・paired 7.557倍（p10 / p90 7.270〜7.832倍）、H=16は0.028→0.004ms、独立7.157倍・
paired 7.160倍（同6.909〜7.491倍）だった。55鎖のkernel時間試算は1.482→0.202msである。

Apple M2 / Metal（8 round）も両shapeでbit一致し、H=8は0.208→0.024ms、独立8.630倍・paired
8.709倍（p10 / p90 6.280〜18.040倍）、H=16は0.223→0.042ms、独立5.250倍・paired
5.295倍（同5.054〜6.567倍）だった。55鎖のkernel時間試算は11.828→1.818ms、独立 **6.506倍**、
**10.010ms / text encoder run削減**となる。2 invocationとも288 query/run、timestamp資源10組で
counter sample buffer枯渇なく完走したため、direct-firstのMetal直接A/B採用門も通過した。

特殊値GPU回帰は有限値をbit比較し、NaN payloadだけはbackendがcanonicalizeできるため
NaN分類一致を契約にした。direct-firstの先頭mulをgraph outputにした場合も既存8 dispatchへ戻る。
runtime全424 testは0 failure、Anima w8a8-s16 / 512 / 8stepは8.6秒で完走し、PNG SHA-256
`dd4506de50f346676a35919d471ff7030514992cd337077c04c0dd2ffa332756`が既存goldenと一致した。

### OP-017: VAE nearest-exact x2 の exact peephole fusion

配布中のAnima VAE decoderにはnearest-exact x2が3回あり、exporterは各回を次の連続6 nodeへ
loweringする。reshapeはaliasだが、2本のexpandはそれぞれ全出力を実体化する。

    reshape → expand(width x2) → reshape → reshape → expand(height x2) → reshape

実shapeは `[1,384,64,64]→[1,384,128,128]`、`[1,384,128,128]→[1,384,256,256]`、
`[1,192,256,256]→[1,192,512,512]`。入力は合計78MiB、expand 6本の出力書込みは合計468MiBである。

public IR / exporterは変えず、executorは6 nodeのop順、結線、全resolved shape、f32 dtype、内部5値の
use count = 1とgraph output非公開をすべて確認する。1条件でも外れれば既存primitiveへfallbackする。
専用kernelは入力1要素を1回読み、対応する2×2の4要素へ書く256 threadの1D grid-strideである。
f32 storageを`array<u32>`のbit-viewで複製するため、浮動小数点演算を通さずNaN payload、subnormal、
±0を含む全bitを保存する。subgroup、barrier、atomics、f16、optional featureには依存しない。

論理tensor trafficのproxyは、2段expandが入力サイズIあたり12I（read 6I + write 6I）、融合が5I
（read I + write 4I）。3 shape合計は936→390MiB、**546MiB削減**となる。このうち確実に消える
中間materializationの書込みだけでは156MiBであり、DRAM実効trafficとは区別する。

実GPU回帰では専用kernelが特殊値を含む全出力bitをCPU期待値どおり保存し、既存f32 primitiveとは
NaNだけ分類、それ以外をbit比較した。内部値の公開、別consumer、x3のnear-shapeはすべてfallbackする。
配布VAEでは対象3鎖だけが発火し、
全体dispatchは335→332、submitは1のまま、融合kernel 3本のRTX/Vulkan時間は0.497msだった。
Anima w8a8-s16 / 512 / 8stepも既存PNG SHA-256
`dd4506de50f346676a35919d471ff7030514992cd337077c04c0dd2ffa332756`と一致した。

専用benchmarkはproductionのstrided / upsample kernelを直接使い、実3 shapeをABBA / BAABで測る。
RTX 3080 Ti / Vulkan（timestamp-query、5 round）はshapeごとに2.401 / 2.276 / 2.135倍、合計
1.097→0.499ms、独立2.200倍、paired **2.201倍**（p10 / p90 2.144〜2.233倍）でbit一致した。
QuerySet / resolve / read bufferは全roundで1組だけ再利用し、最大dispatchもdevice limitでclampする。
Apple M2 / Metal（timestamp-query、8 round）も3 shapeすべてbit一致し、shapeごとに4.346 / 4.555 /
4.114倍、合計21.795→5.106ms、独立 **4.268倍**、paired **3.085倍**（p10 / p90 2.709〜4.187倍）で
完走した。Metalの直接A/B採用門は通過し、残る確認はRoPEとまとめたAnima 1024 E2Eだけである。

### OP-018: sigmoid→mul の exact SiLU peephole fusion

public IRやexporterへ`Silu` opを足さず、隣接するf32
`sigmoid(x) → mul(x, sigmoid)`だけをexecutor内で置換する。sigmoid出力がgraph output、別consumer、
aliasを挟む、mulの他入力がxでない、dtype / rank / resolved shapeが一致しない場合は、既存primitiveへ
fallbackする。mulの入力順はNaN伝播を含む観測形を保つため`x-sigmoid` / `sigmoid-x`を別WGSL・別keyに
固定する。配布assetの57鎖はすべて`x-sigmoid`順だった。

primitiveのsigmoid storage write/readに対応するf32 materialization点を残すため、sigmoid値をworkgroup
`u32[256]`へbitcastし、barrier後にf32へ戻してmulする。grid-stride block loopはworkgroup一様で、
subgroup、atomics、f16、optional featureを使わない。有限値、±0、subnormal、Infは実GPU A/Bでbit比較し、
WGSLがpayloadを規定しないNaNだけ分類一致を契約にする。入力xは元2 input slotぶんreleaseするが、別の
後続consumerがあるケースも実GPU回帰で固定した。

既定1024 / 8step / guidance=1では、text encoder 28鎖、DiT 2鎖×8step、VAE 29鎖×9 tileの合計
**305鎖**を610→305 dispatchへ減らす。512では28 + 16 + 29 = **73鎖**、146→73 dispatchである。
入力サイズIあたりの論理global tensor traffic proxyはprimitive 5I、融合3Iで、1024は
50.672→30.403GiB（**20.269GiB削減**）、512は5.672→3.403GiB（**2.269GiB削減**）となる。
これはDRAM実測ではなく、中間writeだけの確実な削減はそれぞれ10.134 / 1.134GiBである。

専用benchmarkはproductionのelementwise sigmoid / mulとSiLU kernelを、実7 shape、ABBA / BAAB、
全roundで再利用する1組のQuerySet / resolve / read bufferで比較する。RTX 3080 Ti / Vulkan
（timestamp-query、8 round）は全shapeでfinite-bit一致し、重み付き65.403→26.980ms、独立
**2.424倍**、paired **2.423倍**（p10 / p90 2.410〜2.426倍）だった。
Apple M2 / Metal（timestamp-query、8 round）も全7 shapeでfinite-bit一致し、重み付き
646.726→270.115ms、独立 **2.394倍**、paired **1.993倍**（p10 / p90 1.794〜2.463倍）だった。
shape単体のpaired分散はあるが、1024 profile全体のp10も1を明確に上回る。timestamp資源1組の再利用で
counter sample buffer枯渇も起こらず完走したため、Metalの直接A/B採用門は通過した。
runtime全423 testは0 failure、Anima w8a8-s16 / 512 / 8stepは8.6秒で完走し、PNG SHA-256
`dd4506de50f346676a35919d471ff7030514992cd337077c04c0dd2ffa332756`が既存goldenと一致した。
残る確認はRoPE / VAE upsampleとまとめたAnima 1024 E2Eだけである。

### OP-019: identity expand の buffer alias化

`expand`は通常stride 0の複製を実体化する必要があるが、束縛後の入力 / 出力shapeがrankを含めて
完全一致する場合は恒等写像である。executorはこの条件だけを既存`reshape`と同じbuffer aliasとして
扱い、出力確保、params / bind group生成、GPU dispatchをすべて省く。1軸でも異なる非恒等expandと
rank増加expandは従来のstrided copyへfallbackする。数値演算、WGSL、optional feature、backend分岐を
追加せず、f32 / i32 / boolの格納bit列をそのまま共有する。

配布assetの静的再集計ではidentity expandはtext encoder **112本**、text conditioner **48本**、
合計 **160本 / generate**。transformerにはexpandが無く、VAEの6本はすべて実際のx2複製なので対象外。
既定promptのT=29、Ttgt=30、Tsrc=29では、消えるcopyの論理trafficはread + writeで
**50.577MiB**、このうちmaterialized writeが25.289MiBである。

実GPU回帰はaliasをgraph outputとしてpinし、後続buffer確保と別名consumerを通しても早期reuseされず、
peak transientが増えないこととdispatch 0本を固定した。非恒等i32 / bool / rank増加expandを含む既存GPU
op testも通過し、runtime全体は424 test / 0 failure。Anima w8a8-s16 / 512 / 8stepのPNG SHA-256も
`dd4506de50f346676a35919d471ff7030514992cd337077c04c0dd2ffa332756`で既存goldenと一致した。

## 次に疑う場所

| ID         | 状態        | 優先度 | 信頼度 | シグナル                                                                                                | 次の安全な一手                                                                                     |
| ---------- | ----------- | ------ | ------ | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| COMPAT-004 | ⬜ 要検証   | P0     | 高     | Metal で attention i8a8 の dp4a / emulation が 4 ケース不一致。沈黙誤値なら速度より先に止める必要がある | device 初期化時に既知解の実走カナリアを行い、失敗時は dp4a を無効化                                |
| HOST-006   | 🟨 設計候補 | P1     | 高     | 1024 条件の既存実測で params buffer 約 139ms/step、bind group 約 52ms/step                              | executable plan 単位の params / bind group 再利用。まず生成数と寿命を再計測                        |
| OP-008     | 🟨 設計候補 | P1     | 高     | adaptive layer norm の `layer_norm→mul→add` が 85 鎖、255→85、試算 -170                                 | affine を組み込む専用 norm。broadcast と dtype 契約を先に固定                                      |
| OP-009     | 🟨 設計候補 | P2     | 中–高  | VAE channel L2が30鎖。axis-reduce済みの残りは150→30 dispatch候補だが、mul中間のf32丸め境界が消える      | exact鎖を先にcanary化し、積をu32 stagingしてprimitiveの縮約順・丸めを再現できる場合だけ融合        |
| OP-020     | 🟨 設計候補 | P2     | 高     | conditionerのD64 RoPE 24鎖中22鎖は連続だが、`[B,S,H,D]`を既存`[B,H,S,D]` kernelで読むと誤値になる       | token行strideをparams化しD64/BSHDを明示受理。22鎖を176→22へ、prefixが挟まる2鎖は別件               |
| PLAN-011   | 🟨 設計候補 | P1     | 中–高  | timestep-only 部分グラフは 856 node / 推定 774 dispatch。CFG の cond/uncond で同一 timestep を再計算    | 入力依存性で graph を分割し、1 step 内の共有出力を GPU resident のまま再利用                       |
| PLAN-012   | 🟨 設計候補 | P1     | 高     | encoder-hidden-only 部分は推定 308 dispatch。denoise step 間で不変                                      | condition ごとの prepared cross-attention K/V。量子化済みなら初回後 約392 dispatch/run の削減候補  |
| PIPE-013   | 🟨 設計候補 | P1     | 中     | 同一 Session の run は `#chain` で直列化され、pipeline の `Promise.all` は GPU 並列化にならない         | CFG batch 化、または 1 run 内で共通 subgraph を明示共有。単純な同一 Session 並列呼び出しは避ける   |
| OP-014     | 🟨 設計候補 | P2     | 高     | linear→GELU はTransformer 28、conditioner 6。SiLU分解はOP-018のstrict peepholeで対応済み                | GEMM portfolioを増やすlinear epilogue融合は、OP-018と分けて設計・実機評価する                      |
| PIPE-015   | 🟨 設計候補 | P2     | 高     | pipeline 作成は同期 API。初回コンパイルが encoding を止め、異なるキーの生成も逐次 await                 | validation の扱いを保った async pipeline 作成と、plan 全体の並列 prewarm                           |
| CONV-016   | ⬜ 要検証   | P2     | 中–高  | Metal の conv parity に exact mismatch が 2 件。WGSL 浮動小数は再結合・融合差があり得る                 | ULP / atol / rtol と実モデル誤差を分離。大誤差なら kernel bug、最終 bit 差だけならテスト契約を修正 |

大規模項目の境界、候補 API、プラットフォーム別の検証ゲートは
`large-designs.md` に分離した。
