# Metal（Deno + wgpu）の沈黙誤値と性能 — Mac 実機 recon

> NOTE: 時点スナップショット。文中のパス・行番号・実測値は記録当時（2026-08-06）のリポジトリ
> 構成と実機に基づく。

対象機: **Apple M2 / macOS 最新 / Deno 2.9.4（aarch64-apple-darwin）**。比較基準は開発機
（RTX 3080 Ti / Deno 2.9.4 / x86_64-linux）。発端は「Mac で生成すると絵が崩れ、プロンプトにも
追従しない。例外は出ない」というユーザー報告。

## 1. 沈黙誤値の根本原因（根治済み）

**threadgroup メモリ上の `vec4<f32>` への動的インデックス成分書き込み（`sb[i][wsl] = v`）が、
`wsl != 0` のとき黙って捨てられる。**

- 症状は「4 要素中 3 要素が 0 のまま内積へ入る」。相対誤差が 1 を超える（実測 maxRel 1.065 /
  11.65）のに例外は 1 つも出ない。
- 該当は `packages/runtime/src/kernels/gemm.ts` の `fillBLinear`（linear と conv2d の
  implicit GEMM が使う）と attention QK の B タイル充填の **2 関数 8 行のみ**。
  `matmul` / `bmm` は `sb[i] = bv4` と vec4 を丸ごと書くので無傷だった。
- Linux / Vulkan では同じ WGSL が正しく動く。WGSL 仕様上は合法な構文で、Karume 側の
  仕様違反ではない。

### 切り分けの経路（分布から犯人を割り出した）

実機の `deno test -A packages/runtime/tests/` が 25 本落ちた。その**落ち方の分布**が決め手。

| 通った                                                               | 落ちた              |
| -------------------------------------------------------------------- | ------------------- |
| elementwise / cast / reduce / gridstride / arena / submit / device   | linear              |
| `mlp`（matmul）・`batch_matmul`（bmm）・`conv2d_block`・`row_reduce` | 融合 attention 全般 |
| linear i8a8（`gpu_i8a8_test.ts` 全通過）                             | f16 / i8 の重み格納 |

golden の op 集合を IR から抽出すると、落ちた 4 本（`attention_block` / `fused_attention` /
`rms_norm_block` / `i8_weights`）は**すべて `linear` か `attention` を含み**、通ったものは
1 つも含まない。`matmul` / `bmm` も同じ `gemm.ts` を使うので、GEMM 本体ではなく
linear / attention 固有の層に絞られた。

### 実機プローブで確定（4 通りの突合）

リポジトリ非依存の単体スクリプトで、共有配列の型・要素数・スレッド数・バリア位置を**完全に
同一**にし、書き込み方だけを変えた 4 本を同じ期待値で突合した。

| 書き方                          | Linux | Mac(M2)                                           |
| ------------------------------- | ----- | ------------------------------------------------- |
| `sb[i][wsl] = v`（現状）        | PASS  | **FAIL**（不一致 768/1024 = 成分 1/2/3 が全て 0） |
| `sb[i] = v4`（vec4 丸ごと）     | PASS  | PASS                                              |
| `switch (wsl)` で静的成分へ展開 | PASS  | **PASS**                                          |
| `array<f32>` へスカラ化         | PASS  | **PASS**                                          |
| `(*p)[wsl] = v`（ポインタ経由） | PASS  | **FAIL**（現状と完全に同一の不一致）              |

ポインタ経由が現状と 1 ビットも変わらない壊れ方をしたので、**naga は両者を同じ MSL に落として
いる**。書き方の問題ではなくアクセスの形そのもので、**静的成分へ落とす以外に回避手段が無い**。

成分 0 の値が期待値と完全一致した点も重要で、`wsl` が定数 0 に畳まれているなら 4 スレッドが
同じ `.x` に競合して値が壊れるはず。そうなっていないので「`wsl != 0` の書き込みだけが no-op」
と読むのが唯一整合する。

### 採った回避策

`switch (wsl)` で静的成分へ展開（`storeBTransposed`）。共有タイルのレイアウトと内積ループの
読み出しを変えないので、変更が 2 関数に閉じ、Linux / Vulkan の数値は 1 ビットも動かない。
`wsl = (tid / 4) % 4` なので simdgroup 内に 4 値が揃い 4 アームとも実行されるが、コストが乗る
のは B タイル充填だけで、頻度が桁違いに高い内積ループには乗らない。

結果: **落ちるテストが 25 → 6 に減り、Mac で正しい画像が生成できるようになった**
（プロンプト追従も回復）。

### 上流の関連

- [wgpu#4460](https://github.com/gfx-rs/wgpu/issues/4460) / [naga#1702](https://github.com/gfx-rs/naga/issues/1702)
  — vector 成分へのインデックス代入。D3D は生成 HLSL を拒否するが「Vulkan と Metal では
  正しく動く」とされている。**本実測はこれを覆す**（Metal・threadgroup 上で壊れる）。
- [wgpu#4500](https://github.com/gfx-rs/wgpu/issues/4500)（open・`tag:data-corruption`）—
  naga は workgroup バッファを MSL の entry-point 引数（動的 threadgroup メモリ）として出し、
  Tint は固定長ローカル変数として出す。**Dawn では再現しないと明記**されており、
  ブラウザ（Chrome / Safari = Dawn·Tint 系）では本件が出ない可能性が高い（未検証）。

### 棄却した仮説（記録）

- **fast-math**: Metal の `fastMathEnabled` は既定 YES で wgpu-hal は無効化していないが、
  実機プローブでは denormal が保持され（`1e-40 * 1.0` → `9.99994610111476e-41`）、
  `1.0/3.0` も `0x3eaaaaab`（Linux と同一）。**痕跡なし**。maxRel 1.065 / 11.65 という差は
  そもそも数 ULP では説明できない。
- **wgpu#3181（workgroupBarrier が効かない）**: 縮約系（reduce / softmax / layer_norm /
  rms_norm）は wg=256 の `if (tid < stride)` 木縮約という #3181 と同型の構成だが、
  `gpu_reduce_axis_parity_test.ts` ほかが全通過。プローブでも wg=256 の木縮約は PASS。
- **wgpu#4500 の直撃**: `workgroupUniformLoad` はリポジトリ全体で 0 件。
- **dp4a（`dot4I8Packed`）**: `gpu_i8a8_test.ts`（linear i8a8）が全通過。`wgslLanguageFeatures`
  にも `packed_4x8_integer_dot_product` が載る。
- **バッファのオフセット / アライメント**: `setBindGroup` に dynamic offsets を渡しておらず、
  サブアロケーションもしていない（1 テンソル = 1 GPUBuffer）。

## 2. 残る誤値（未解決 — known-issues.md へ起票）

修正後も 6 本が赤のまま。**どちらも今回の機序では説明できない**。

- **attention i8a8 系 4 本** — `attention-i8a8.ts` は共有配列が `array<u32>` のスカラで動的成分
  書き込みを持たない。かつ**同じ `dot4I8Packed` を使う linear i8a8 は通る**。それでも
  `attention_qk i8a8: dot4I8Packed 版とエミュ版が atol=0 で一致する` が落ちており、整数演算に
  丸め差はないので attention の QK/PV では実際に違う値が出ている。
- **conv2d parity 2 本** — implicit GEMM ↔ 直接カーネルのビット一致。conv2d の B タイルは
  `sb[bk * 16u + bcq] = bv4` と vec4 丸ごと書きで、本件の影響を受けない構造。

なお `conv2d_block` golden（許容誤差 atol 1e-6 / rtol 1e-5）は**通っている**ので、conv2d の
値そのものは概ね正しく、2 経路のビット一致だけが崩れている。

## 3. 性能（Mac は Linux の 31〜41 倍遅い）

`AnimaPipeline` は内部 Session の `diagnostics()` を公開しないため、DiT だけを同手順で組んだ
単体ベンチで測った（`planDynDit` / `predict` の写し）。

### preset 比較（512×512・1 step の壁時計）

| preset           |  Mac(M2) | Linux(3080Ti) |  倍率 |
| ---------------- | -------: | ------------: | ----: |
| w8a8-s16（既定） | 17,241ms |         415ms | 41.5x |
| w8a8             | 18,725ms |         454ms | 41.3x |
| i8（f32 計算）   | 32,643ms |       1,039ms | 31.4x |
| f16（f32 計算）  | 32,685ms |       1,039ms | 31.5x |

1024×1024 の w8a8-s16 は Mac 79,190ms / Linux 1,461ms（54x）。

### ここから確定したこと

- **順位は両環境で同一**。整数経路は Mac でも効いている（f32 計算の 1.89 倍速い）ので、
  dp4a が遅い経路に落ちているわけではない。ただし効きは弱い（`i8 → w8a8` の改善が
  Linux 2.29x に対し Mac 1.74x）。Apple GPU の integer dot product が f32 と同レートで、
  メモリ削減分しか効いていないと読める。`w8a8 → w8a8-s16`（attention 側）は
  Linux 1.09x / Mac 1.086x と**完全に一致**。
- **メモリ圧・スワップは主因でない**。f16（3.9GB 常駐）と i8（1.96GB）で速度が同じ
  （32,685 vs 32,643ms）。`hostExpandedBytes: 0` も両方で維持。
- **ホスト側オーバーヘッドも主因でない**。512→1024px で Mac は 4.6 倍、Linux は 3.5 倍。
  dispatch 数は解像度でほぼ変わらないので、固定費律速なら Mac の増え方が緩くなるはずが、
  逆に強く増えている（= 演算・帯域律速）。
- **重みロードは正常**。Session 構築は Mac 233ms / Linux 318ms でむしろ速い。

M2 と 3080 Ti の理論比は演算・帯域とも約 9.5 倍なので、31 倍のうち **3.3 倍が未説明**。
GEMM 骨格の occupancy（1 スレッド 16 レジスタ + wg 256 = simdgroup 8 本。Metal は
レジスタ圧で `maxTotalThreadsPerThreadgroup` が実際に下がる）やタイル形状が候補。
**バグではなく最適化されていない状態**で、性質が §1 とは違う。

### Linux 側のマイクロベンチ基準（2048³ / 達成率の判定用）

| 指標                            |          実測 |          理論比 |
| ------------------------------- | ------------: | --------------: |
| 帯域（vec4 コピー read+write）  |    301.7 GB/s | 912 GB/s の 33% |
| Karume 実 GEMM（matmul v4）     | 3,464 GFLOP/s |   34,000 の 10% |
| naive GEMM（1 スレッド 1 出力） |   642 GFLOP/s |               — |
| タイリングの効き                |         5.40x |               — |

Mac 側の同ベンチ結果は未取得。**達成率が Linux と同程度なら Apple GPU 一般の姿**（最適化の
余地は小さい）、**著しく低ければ GEMM 骨格が Apple GPU に噛み合っていない**という判定になる。

## 4. 副産物 — Metal では GPU timestamp が使えない

`gpuTiming: true`（ADR 0021）で DiT を回すと、1 dispatch = 1 pass に開いた結果 3,301 dispatch
分の timestamp サンプルが必要になり、Metal が

```
Failed to create counter sample buffer: Cannot allocate sample buffer (MTLCounterErrorDomain)
```

を返して **device 消失**に至る（`GpuDeviceLostError` として正しく可視化される）。op 別の内訳は
この環境では取れないため、preset 比較と壁時計で切り分けた。limitations.md に起票。
