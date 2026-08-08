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
  `undefined` を返す**。RoPE なら `[1,H,S,128]` / table `[1,1,S,128]` / dim=3 の 0-64 と
  64-128 だけ、SiLU なら全スロット同 shape の f32 だけ、upsample2x なら f32 rank4 NCHW の
  各空間軸ちょうど 2 倍だけを受理する。偶奇 RoPE・別 head 幅・broadcast SiLU・一般 resize へ
  一般化しない。
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
  ブランチの静的集計と一致する。
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
