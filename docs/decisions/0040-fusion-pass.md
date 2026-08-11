# 0040: 実行時融合は独立した純関数パス（exact 一致 + 常設カウンタ）

- Status: accepted（PNG sha256 門 4 本が不変・SBV2 は融合 on/off で WAV sha256 一致）
- Date: 2026-08-08
- 関連: ADR [0007](0007-op-vocabulary.md)（op 語彙 allowlist — **本 ADR は公開語彙を 1 つも
  増やさない**）/ [0012](0012-attrs-and-fused-ops.md)（融合 op を IR に置く場合の契約 —
  本 ADR はその**対極**の「IR に出さない private カーネル」側）/ [0011](0011-layout-strategy.md)
  （レイアウト戦略 — 恒等 expand の別名化を追記済み）/ [0023](0023-fused-attention.md)
  （既存の融合カーネル = こちらは IR op として公開した例）/ [0021](0021-gpu-timing-diagnostics.md)
  （常設診断の流儀）/ [0004](0004-execution-model.md)（アリーナの参照計数）
- 参照実装ブランチの triage: [research/2026-08-06-kernel-triage/](../research/2026-08-06-kernel-triage/)
  （OP-007 / OP-017 / OP-018 / OP-019）・採否と実測は
  [research/2026-08-08-branch-adoption-perf.md](../research/2026-08-08-branch-adoption-perf.md)

## Context

エクスポータが出すノード列には「決まった並びで必ず現れる分解」がある。half-split RoPE の
連続 7 ノード（cat が入力ごとに copy を出すので実 dispatch は 8）、`sigmoid → mul` の SiLU、
VAE nearest-exact x2 の `reshape/expand` 6 ノード、そして shape が完全一致する恒等 `expand`。
Anima 1024 / 8step の既定経路では、これらだけで数千 dispatch と GiB 級の中間 write / read を
占める（本数は下の実測）。

一方 ADR 0007 の語彙 allowlist と ADR 0012 の契約規律により、**IR に op を足すのは重い**
（TS 契約表・Python 契約表・golden・エクスポータを 1 セットで動かす）。RoPE や SiLU は
「torch 側に対応する単一 op が無い / 分解形でしか出てこない」ので、公開語彙を増やす対価に
見合わない。実行時にだけ潰す peephole が要る。

問題は**どこに置くか**である。素直に書けば executor のノード走査ループへ `if (op === "sigmoid"
&& next.op === "mul")` を差し込む形になるが、この形は 2 つの構造的な事故を招く。

1. **判定が GPU テストでしか触れない** — 反例（use-count 2 / graph output / near-shape /
   dtype 違い / 順序違い）を網羅するのに実 GPU が要る。
2. **解放簿記が融合の本数だけ複製される** — 融合ステップは元ノード列より入力の延べ回数が
   減る（内部値を実体化しない）ので、retain / release を融合ごとに手書きすることになる。
   1 本ずれても例外は出ない（早すぎる解放 = プール再利用で値が化ける／多すぎ = peak が
   落ちない）。アリーナの参照計数（ADR 0004）が沈黙誤値の面に化ける。

## Decision

### 1. 融合は `src/runtime/fusion.ts` の純関数パス（executor 直書きの禁止）

計画済みノード列（`plan.ts` の `NodePlan[]`）→ 実行ステップ列（`ExecStep[]` = 素のノード
または融合ステップ）への変換を、**GPU に触れない 1 つの純関数** `planFusions` に閉じる。
executor はステップ列を受け取って encode するだけで、判定を持たない。

- **ルールは宣言表** `FUSION_RULES`（現行 3 本 = silu / upsample2x / rope）。各ルールは
  `match`（掴む）と `build`（binds / kernel key / params を宣言する）に分かれ、
  `defineRule` が両者の分離を型で強制する。
- **MUST: 解放簿記の根拠（外部入力の延べ列 `ins`）と走査幅（`nodeCount`）は掴んだ鎖から
  導出する** — ルール側に宣言させない。`externalIns(chain)` が「内部値を除いた元 `node.ins`
  の延べ列」を機械的に作るので、同じ事実がルールの本数だけ複製されることが構造的に起きない。
- **MUST: encode は共通簿記 1 本に合流する**。`#encodeStep` が「確保 → retain → 本体 →
  入力の release（延べ）→ 定義ぶんの release」を素のノードと融合ステップの**両方**について
  持ち、分岐するのは本体（`#encodeNode` / `#encodeFused`）だけ。融合カーネルの bind 面は
  「params, 入力…, 出力」・params は 16 バイト uniform で全ルール共通に固定する。
- 適用順は `FUSION_RULES` の宣言順。現行 3 ルールの先頭 op（`sigmoid` / `reshape` /
  `mul|slice`）は互いに素なので順序は結果に効かないが、**その互いに素性は各ルールの `heads`
  宣言からテストが機械検査する**（重なった瞬間に順序が意味を持ち始めるため）。
- 恒等 `expand`（束縛後の入出力 shape が rank を含め完全一致）は `reshape` と同じ 0 dispatch の
  バッファ別名にする（ADR 0011 追記済み）。非恒等 expand は従来の strided 実体化コピー。

### 2. matcher は exact 一致のみ（受理集合を「式が似ている」で広げない）

掴めなかった形は素のノード列にそのまま落ちる。この fallback が**常に正しい既存経路**である
ことが、融合パスの正しさの全根拠である。受理集合を「式が似ている」で広げた瞬間、
「掴めなければ必ず正しい」の外側に出て、fallback が保証にならなくなる。したがって:

- **op 列・結線・attrs・解決済み shape・dtype を全て突き合わせ、1 点でも外れたら
  `undefined` を返す**。RoPE なら `[1,H,S,D]`（D は正の偶数 — 実測は Anima 128 / Gemma 系
  256）/ table `[1,1,S,D]` / dim=3 の `0-D/2` と `D/2-D` の半分割だけ（追記 2026-08-11 —
  head 幅は分割位置から導出）、SiLU なら全スロット同 shape の f32 だけ、upsample2x なら
  f32 rank4 NCHW の各空間軸ちょうど 2 倍だけを受理する。偶奇 RoPE・broadcast SiLU・
  一般 resize へ一般化しない。
- **鎖の内部値は「消費者ちょうど 1 本・graph output でない」を全ルール共通の適格条件にする**
  （`internalsArePrivate`）。融合後は内部バッファを 1 本も作らないので、外部 consumer や
  readback が 1 つでもあれば値が消える。
- **MUST: use-count と解放簿記は実際のノード順で持つ**（RoPE は slice-first と
  direct-mul-first の 2 順序を受理するため、役割順に並べ替えた列を使うと direct-first だけ
  内部値の集合がずれる）。
- 融合は演算列を潰すが**値は変えない**。丸め位置の保存はカーネル側の責務で、RoPE / SiLU が
  使う手段（workgroup memory 往復による丸め障壁）は **WGSL 仕様の保証ではなく実測依存**である
  ことを各カーネルの docstring に明記する。upsample2x は u32 ビット複製なので丸めの議論自体が
  無い。

### 3. 適用回数は `lastRunFusions` として常設診断に出す

融合は**エクスポータのノード発行順が 1 つ変わるだけで黙って外れる**。値は正しいまま
（fallback が正しいので）性能だけが戻り、例外も警告も出ない。ここが唯一の観測点になるので、
ルール別の適用回数を `Diagnostics.lastRunFusions` として常設する（`lastRun` / `lastRunTiming`
と同じ寿命 = 直近 run・run のたびに丸ごと置き換わる）。

数えるのは融合 3 ルール + `identityExpand`。**`reshape` の別名化は数えない** — 無条件に成立し
外れようがないので、観測する意味が無い（恒等 expand は shape 条件付きで外れうるから数える）。

### 4. グラフ書き換えの役割分担基準（exporter の normalize か / runtime の融合パスか）

同じ「無駄なノードを潰す」でも置き場が 2 つある。基準は **IR の語彙の中で閉じるか**:

| 書き換え                                        | 置き場                     | 例                                                                           |
| ----------------------------------------------- | -------------------------- | ---------------------------------------------------------------------------- |
| **IR の中で閉じる**（結果も IR の op で表せる） | exporter の `normalize.py` | 隣接 permute の合成 `p∘q`・恒等 permute の除去・恒等 clone / repeat の除去   |
| **IR に無い private カーネルへ潰す**            | runtime の `fusion.ts`     | RoPE 7 ノード → 1 dispatch・SiLU 2 → 1・upsample2x 6 → 1・恒等 expand の別名 |

前者を runtime に置くと、IR を読んだだけでは分からない最適化が実行時に散る（IR の
ノード数が現実の dispatch 数と乖離する）。後者を exporter に置くと、IR へ private op を
公開することになり ADR 0007 の allowlist を壊す。

**非対称性を記録しておく**: `normalize.py` は **torch.export 経路にしか効かない**（別経路で
書かれた IR や既存の焼き済み資産には効かない）。一方 `fusion.ts` は**任意の IR に効く**
（実行時に見た形だけで判定するため）。したがって「両方で書ける」書き換えが将来出た場合、
適用範囲の広さは融合パス側が上で、IR の可読性は normalize 側が上になる — どちらを取るかは
その都度の裁定事項で、本 ADR は既定を置かない。

## 実測（2026-08-08 / RTX 3080 Ti・Deno 2.9.4・x86_64-linux）

- **融合ヒット数（anima 1024 / 8step / guidance=1）**: rope **503**（transformer 448 +
  text encoder 55）/ silu **305** / upsample2x **27** / identityExpand **160**。参照実装
  ブランチの静的集計と一致する。NOTE: 503 / 55 は**当時の matcher の実測**。2026-08-11 の
  一般化（下の追記）で text encoder の取りこぼし 1 箇所が拾えるようになり 504 / 56。
- **SBV2**: sigmoid が 0 本のため融合ルールは 1 本も発火せず（identityExpand **210** のみ）。
  融合 on / off で WAV と dump が sha256 完全一致。
- **PNG sha256 門 4 本が不変**（f32 丸め境界の保存が E2E で成立）。
- E2E の壁時計は C 波全体の効果として
  [research/2026-08-08-branch-adoption-perf.md](../research/2026-08-08-branch-adoption-perf.md)
  に記録（w8a8-1024 16.1 → 13.9s ほか）。融合単体の寄与は分離していない。
- テスト: GPU 非依存の matcher テスト（反例網羅 + 先頭 op 互いに素性の機械検査）と、
  interpose 双子グラフの bit 一致 GPU テスト 3 本 + RoPE H≥2 ケース。

## Consequences

- **IR の公開語彙は 1 つも増えない**。`op-vocabulary.md` の「`silu` が `UnaryOp` にあるが到達
  不能」という死枝疑いは、公開 op を足さずに executor 内部の融合ルールで解消された。
- **融合の追加コストが下がる**（`FUSION_RULES` に 1 本足す + カーネル 1 本 + matcher テスト）。
  裏返しに、**matcher の受理条件は実測形への決め打ち**になっているので、エクスポータ側の
  ノード発行順や shape が変われば黙って外れる。`lastRunFusions` を見る運用が前提。
- 融合カーネルは全て f32 専業・optional feature 非依存（subgroup / atomics / f16 を使わない）。
  f16 計算経路の融合は本 ADR の範囲外。
- 参照ブランチの triage が挙げた他の候補（adaptive norm の `layer_norm→mul→add` 85 鎖、
  VAE channel L2 の 30 鎖、conditioner の D64 RoPE 22 鎖）は**未実装のまま**。特に VAE
  channel L2 は mul の storage 書込みが作る f32 丸め境界が消えるため、u32 staging で縮約順を
  再現できることを先に証明する必要がある（[large-designs.md](../research/2026-08-06-kernel-triage/large-designs.md) F3）。
  → adaptive norm は 2026-08-10 の追記で実装済み（下記）。

## 追記 2026-08-10: 窓内 passthrough の導入と adaLN 融合（OP-008）

adaptive norm（DiT の変調）を 4 本目のルール `adaln` として実装した。本 ADR の Decision は
**§2 の exact 一致も §1 の解放簿記も 1 つも変えない**。増えたのは「連続窓の一部を畳まずに
通す」という走査側の概念 1 つだけである。

### 前提の訂正: 鎖は隣接していない

triage が「`layer_norm → mul → add` の 3 ノード」と書いていた形は、実 IR
（anima DiT・`model.i8` / `model.f16` 共通）では**隣接していない**。85 鎖すべてが次の窓で、
84 鎖が窓 7・末層の 1 鎖だけが窓 6（gate 無し）:

```
layer_norm(x, w, b)          -> t            [1,S,2048]  eps=1e-6
reshape × 2〜3               （変調ベクトルの unsqueeze — shift / scale / gate）
add(scale, const[1])         -> s            [1,1,2048]
mul(t, s)                    -> p            [1,S,2048]
add(p, shift)                -> y            [1,S,2048]
```

`layer_norm` の consumer はちょうど 1（mul）で、間の reshape は `layer_norm` 出力を 1 つも
消費しない（依存が無い）ので、**reshape を融合ステップより前へ動かす並べ替えは合法**。

### 決めたこと

1. **ルールは連続窓（`window`）と畳む部分列（`chain`）を宣言し、差分は passthrough として
   融合ステップの前に素のノードのまま並べる**。`planFusions` は `index += 窓幅` で進む。
   - MUST: 走査幅・passthrough・解放簿記（`ins`）・畳んだ本数はすべて `defineRule` が
     **窓と鎖から導出**する。ルール側は宣言しない（§1 の MUST をそのまま窓へ拡張した形）。
     鎖が窓の部分列でなければ `defineRule` が fail loudly（取りこぼしは未実行、はみ出しは
     二重実行で、どちらも例外の出ない沈黙誤値になる）。
   - MUST: passthrough は鎖が定義する値を 1 つも消費しないこと（`passthroughIsIndependent`）。
     現行 adaln では `internalsArePrivate` に包含されるが、鎖の**最終**ノードが passthrough
     より前に来る将来の窓では独立に効く（internalsArePrivate は最終出力を見ない）。
   - `reshape` は upsample2x の先頭 op でもあるので、**窓ごと読み飛ばすことで他ルールの機会を
     奪っていないこと**を matcher テストが機械検査する（窓の全開始位置 × 他ルール = 掴まない）。
2. **受理は実測形への決め打ち（§2 は不変）**。passthrough は「`layer_norm` の直後に並ぶ連続
   `reshape` 2〜3 本」だけ、変調は `[1,…,1,dim]` の broadcast だけ、定数は shape `[1]` だけ、
   `mul` / `add` の入力順は実測の 1 通りだけを受理する。**定数の値は仮定しない** — `1.0` を
   焼き込むと「掴めなければ必ず正しい」の外へ出るので、`one` はバッファとして束ね
   カーネルが `one[0]` を読む（storage 6 in + 1 out = 7 本・既定上限 8 の内側）。
3. **カーネル `src/kernels/adaln-norm.ts`** は素の layer_norm と同じ「1 行 = 1 workgroup(256)・
   行方向 grid-stride」。行統計（2 パス / 母分散）と affine の式は
   **素の layer_norm と同一文字列を共有**する（`LAYER_NORM_ROW_STATS_WGSL` /
   `LAYER_NORM_AFFINE_WGSL` — `SIGMOID_STABLE_WGSL` を silu と共有しているのと同じ理由）。
   共有により素の `layer_norm.wgsl` のバイト列は 1 バイトも動いていない（スナップショットで固定）。
   丸め障壁は **u32 staging + barrier の 2 段**（`1 + scale` と `t · s`）。素の列が持つ
   3 つの実体化点のうち `t` は後段が乗算なので縮約の機会が無く、障壁を要さない。
   出力ループは silu と同じ block ループへ組み替える（`o = lid` の while は workgroup 非一様で
   barrier を置けない）。
4. **`FusionCounterName` に `adaln` を追加**。`FusedDispatch.workgroupSize` は「grid-stride の
   割り数」であって `@workgroup_size` ではないことを明記した（1 行 = 1 workgroup 族は 1 を渡す）。
5. **executor は無変更**。bind 面「params, 入力…, 出力」・16 バイト uniform の共通形にそのまま乗る。

### 実測（GPU 非依存・`planFusions` を実配布グラフに適用）

`tests/assets_fusion_counts_test.ts` が **run 1 回あたり**の値を門にする（資産が無ければ SKIP）:

| グラフ                                 | silu | upsample2x | rope |  adaln | identityExpand |
| -------------------------------------- | ---: | ---------: | ---: | -----: | -------------: |
| transformer（i8 / f16 共通・S 非依存） |    2 |          0 |   56 | **85** |              0 |
| text encoder                           |   28 |          0 |   56 |      0 |            112 |
| text conditioner                       |    0 |          0 |    0 |      0 |             48 |
| VAE decoder（タイル 1 枚）             |   29 |          3 |    0 |      0 |              0 |
| EmbeddingGemma-300m（T 非依存）        |    0 |          0 |   48 |      0 |             96 |

上の実測欄が載せている predict 1 回ぶんの合計は、これをパイプラインの run 回数で畳んだもの:
rope 56×8step + 56 = 504 / silu 2×8 + 28 + 29×9tile = 305 / upsample2x 3×9tile = 27 /
identityExpand 112 + 48 = 160。**adaln は 85×8step = 680/predict**（研究ノートの「85」は
静的な鎖の本数 = DiT の 1 run ぶん）。DiT のステップ列は 2601 → 2346（**−255/run**）。
text encoder の rope 56 と EmbeddingGemma 行は 2026-08-11 の一般化後の値（下の追記）。

### 実 GPU での検証結果（2026-08-10 検収・RTX 3080 Ti / Vulkan）

- **丸め障壁は実バックエンドで効いた**: 双子グラフ門（`tests/gpu_adaln_fusion_test.ts` —
  subnormal・±0・Inf/NaN 行・分配則を誘発する scale を含む敵対的入力）で窓 6 / 7 とも
  **有限値ビット一致 + NaN 分類一致・4→1 dispatch**。`t · (scale + 1)` の分配則
  （`fma(t, scale, t)`）も staging で遮断され割れなかったので、**a2（`1 + scale` も畳む形）で
  確定**（割れた場合の退避先 = `1 + scale` を畳まない 5 in 形、は使わずに済んだ）。
- PNG sha256 門 4 本とも参照一致・全 740 テスト緑。
- 直接 A/B（gpuTiming・w8a8-1024・8 step）: 鎖 3 キー計 41.5 → 28.8 ms/step
  （`layer_norm` キーは消滅・`adaln_norm:v1` が 11.2 ms/step で出現・`ew add r3` は
  dispatch 2032 → 672）= **GPU −12.7 ms/step ≈ −102 ms/8 step**。帯域モデルの見積り
  （−135 ms）の 76% — 融合カーネルの実効帯域がモデル仮定（素の 3 op と同じ ≈650 GiB/s）より
  低めに出たぶんの差で、方向と桁は一致。加えて dispatch −255/predict のホスト費用減。

## 追記（2026-08-11）: rope ルールの一般化（head 幅の導出 + 窓内 passthrough）

EmbeddingGemma-300m（Gemma3・head 幅 256）で rope が 0 ヒットだったため、受理集合を
2 点だけ広げた。「式が似ている」への一般化ではなく、**同じ half-split rotate_half の
実測形が 2 つ増えた**という位置づけ:

1. **head 幅 D を分割位置から導出**（従来は 64 / 128 決め打ち）。受理は `[1,H,S,D]`
   （D は正の偶数）・table `[1,1,S,D]`・dim=3 の `0-D/2` / `D/2-D` の**半分割のみ**。
   カーネル（rope.ts）は head_dim / half_dim を uniform で受けるので WGSL は 1 バイトも
   変わらず、丸め障壁の議論もそのまま生きる。
2. **窓内 passthrough を rope にも導入**（adaln に次ぐ 2 例目）。cos / sin 表を実行時 T へ
   縮める `sym_prefix_slice` は θ 系統ごとに 1 度だけ作られ、その**初出 1 箇所**が `cat` と
   cross `mul` の隙間に落ちる。跨げるのは **`sym_prefix_slice` ちょうど 1 本だけ**
   （別 op・2 本以上は不受理 — 反例テストで固定）。

効果: EmbeddingGemma で rope 0 → 48（実 dispatch 1,294 → 958・−26%）、Anima text
encoder で同型の取りこぼし解消 55 → 56（sin 表の初出が隙間に落ちる 1 箇所 — 上の実測欄の
503 は当時の matcher の値で、以後は 504）。PNG sha256 門 3 本一致でビット同一を再確認。
