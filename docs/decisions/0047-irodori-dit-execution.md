# 0047: Irodori DiT の実行形 — B=1 × 選択実行・G4 は当面 G5 へ畳む・uncond はマスクで表す

- Status: accepted（ユーザー裁定 2026-08-11 — recon の推奨どおり）
- Date: 2026-08-11
- 関連: ADR [0046](0046-cat-symbolic-axis.md)（記号軸 cat）/
  [0044](0044-runtime-attention-mask.md)（実行時 bool マスク）/
  [0042](0042-prepared-execution-plan.md)（prepared — G4 切り出しの分岐点）/
  実測 = [research/2026-08-11-dit-export-recon.md](../research/2026-08-11-dit-export-recon.md)

## Context

Irodori の Euler ループは 40 step・CFG は前半 t∈[0.5,1.0] のみ・variant は
independent 3 本（text / speaker / caption の各 uncond）+ cond の計 4 本。上流実装は
バッチ 4 でまとめて forward し、uncond 用に「state を 0 にした context KV」をもう一組作る。

## Decision

### 1. uncond は「cond の context KV + マスク全 0」で表す

3 変種とも **cond の KV をそのまま使いマスクだけ 0 にした結果とビット一致**（recon §3-1 の
実測 — マスク済み要素の寄与は `exp(−inf)=0` で厳密に 0）。よって G4 相当の射影は
**cond の 1 組だけ**計算し、CFG 4 変種の違いは bool マスク 1 本に還元する。上流の
「バッチ 4 倍の context KV」（712MB 相当）は持たない。

入力契約（fail loudly）: `speaker_uncond_mode="mask"`（既定）と
`cfg_guidance_mode="independent"` のみ対応。`"noise"` / `joint` / `alternating` は
この同値が成り立たないため、パイプライン層で明示拒否する。

### 2. B=1 グラフ × 選択実行（B=4 グラフは持たない）

- CFG が要る step だけ 4 回 forward（≈100 forward）。B=4 常時（160 forward）より演算 −37%。
- 中間 `scores[1,20,S,S+1519]` は S=750 で 130MB — B=4 の 519MB は本機
  `maxStorageBufferBindingSize`（実測 2048MiB）でも重く、案 c の記号軸 cat は B=4 だと
  context KV の 4 倍展開（712MB）を強制する。
- 代償: run 固定費（~38ms/run — EmbeddingGemma での帰属値）× 100 run ≈ 3.8s。ホスト固定費の
  分解（独立将来波）が効けば B=1 の優位が確定する。B=4 は再訪しない（上記の構造的な代償）。

### 3. G4（context-KV 事前射影）は**当面 G5 へ畳んで 1 グラフ**にする

現行の実行相はグラフ入力を**毎 run `writeBuffer`** する（入力値の Session 常駐は未実装 —
タスク #7 保留の「Session 常駐 opt-in」が該当）。G4 を別グラフにすると出力 178MB
（24 本 × [1,1519,20,64]）の毎 run アップロード ≈30ms が新しい固定費として乗る
（100 run で 3.0s — 実測 recon §4-2）。畳めば入力は条件 state 3 本 = 3.6MB / run で、
再計算 59.6 GFLOP/forward ≈ 0.56s を払っても**差し引き 2.4s の得**。

**分岐点（DECIDED: 本 ADR）**: タスク #7 の「入力値の Session 常駐」が入ったら G4 を
切り出す — その時点で再計算 0.56s の節約に転じる。それまで G5' は
`x_t[1,S,32]` / `t_embed[1,512]` / `mask bool[1,1,1,S+1519]` / 条件 state 3 本
（text[1,256,512] / speaker[1,751,768] / caption[1,512,512] — Tmax 右 pad）を入力に取る。

### 4. ホスト残置

Euler 更新（`x += v·Δt`・[1,S,32] ≤ 24k 要素）と CFG 合成
（`v + Σ scale·(v_cond − v_k)`）・t_embed の生成（sin/cos 3 行 — `cos` を語彙に足さない）・
条件 state の Tmax 右 pad とセグメントマスク構築。

## Consequences

- 条件 pad により**上流の「詰めた」計算とのビット一致は成立しない**（rel 7.5e-07 — SDPA が
  縮約長でカーネルを選ぶため、方式によらず不可避。recon §1-6）。golden 門は「export した
  グラフを torch で回した値」基準なので厳格さ（atol 0 起点の実測導出）は維持できる。
  文書で「上流とビット一致」を主張しないこと。
- S=750 の scores 130MB は **WebGPU 既定の `maxStorageBufferBindingSize`（128MiB）を超える**。
  本機は 2048MiB で問題ないが、配布時のポータビリティ課題として limitations に起票する
  （解の候補 = 融合 attention の実行時マスク対応〈ADR 0023 の将来枠〉か S 上限）。
- U7（概算）: 10s 発話 ≈2s / 30s 発話 ≈6.5s の GPU 実時間 + ホスト固定費 3.8s（recon §6 —
  attention は分解経路なので楽観側。実測は G5' 着地後）。
