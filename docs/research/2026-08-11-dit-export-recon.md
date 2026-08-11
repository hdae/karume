# Irodori DiT（G4/G5）export 設計 recon — U3 の解体と実行形の実測

> 時点スナップショット（2026-08-11・実重み `inputs/irodori/v4-small` + 実装 clone に対する
> torch.export / TS 束縛の実測。GPU 実行はしていない — export 到達性と数値同値の確認）。
> 裁定の帰結は ADR [0046](../decisions/0046-cat-symbolic-axis.md)（cat 緩和）と
> [0047](../decisions/0047-irodori-dit-execution.md)（実行形）が正本。プローブ台本は
> scratchpad（揮発）。

## 1. U3（記号 4 軸 + 記号軸 cat）の解体

前提の同値: `JointAttention.forward` を「条件 KV 連結済み 1 本 + マスク連結済み」に書いた
ラッパは詰めた文脈・マスク全 1 で**上流とビット一致**（maxabs 0.0）。self マスクは推論の
3 呼び出し口すべてで `None` → `torch.ones` なので**マスクが要るのは条件側 K だけ**。

| 構成                                        | 結果                                                                                                                                                                                                                                                                  |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ③ 全軸固定（S も静的）                      | **通る**（2 層 221 ノード・新規 op ゼロ・`safe_softmax` が正規に発火）                                                                                                                                                                                                |
| ① S のみ記号 + 条件 Tmax 固定               | 2 段で落ちる: (a) 派生次元 `S+1519` が `convert.py` の range_constraints 走査で `AttributeError`（実装漏れ — sympy.Symbol 絞りで解消可能）(b) `aten.cat.default: 軸 1 が記号次元`（**層あたり K/V の 2 本だけ**。RoPE 実数対・先頭 10 head の再連結は静的軸で素通り） |
| ② 4 軸とも記号                              | **torch.export 自身が拒否**（`Only increasing linear operations with integer coefficients` — 複数シンボル和は torch の次元言語にも載らない）。原理的に不可                                                                                                            |
| 案 b: self/context 2 分岐 + flash 型 renorm | **現行語彙で通る**（12 層 1498 ノード・`amax(dim).unsqueeze` への書き換え 1 点のみ・`safe_softmax` は出ない = 自前 softmax）                                                                                                                                          |
| 案 c: cat 緩和（2 行）+ ①                   | **最後まで通る**（12 層 1223 ノード・`verify_model` 緑・TS は `validateGraphContracts` の宣言検査 1 本だけが拒み、`bindSymbols`（`S+1519` → 束縛後 1556 に正しく評価）と `planGraph` は既に正しく処理）                                                               |

数値（12 層・S=200・実文脈を Tmax pad）: 案 a/c = 上流比 rel 7.5e-07 / 案 b = 8.9e-07。
**pad するだけで上流とのビット一致は壊れる**（2 層 S=9 でも 2ulp — torch SDPA が縮約長で
別カーネルを選ぶため。意味論の差ではない）。golden 門は「export したグラフ基準」なので
厳格さは維持できる。

## 2. 案比較（要点 — 全表は ADR 0046）

案 c（採用）: 語彙変更は宣言ガード 3 本 + 適合表・ノード最小 1223・SDPA 分解そのまま・
KV キャッシュ追記と同型で E2B へ前払い。案 b（代替として実測済み）: 語彙不変だが attention
中心部に独立実装・+22% ノード。案 d（S=750 固定）: 10s 発話で 3.1× — 却下。

## 3. CFG と G4 の実測

- **uncond 3 変種 = 「cond の context KV + マスク全 0」でビット一致**（3 層・3 変種とも
  maxabs 0.0）。バッチ 4 倍の context KV（712MB 相当）は不要。`speaker_uncond_mode="noise"`
  のみ不成立（既定は `"mask"` — 既定外は fail loudly）。
- B=4 export 自体は通る（バッチ別マスク `[4,1,1,K]` は分解経路の broadcast で可）が、案 c では
  context KV の 4 倍展開（712MB）を強制。案 b の B 畳み込み（`[H,B·S,D]`）はビット一致だが
  `B·S` が複数シンボル積になり **B=4 固定の別グラフ**が要る。→ **B=1 × 選択実行**
  （100 forward vs 160・中間 scores 130MB vs 519MB）。
- G4 単体 export は素直に通る（Tmax 固定の静的 cat・記号ゼロ = prepared キー 1 本）が、
  出力 178MB の毎 run アップロードが実測 **≈30ms/run**（writeBuffer 4.9〜6.0 GiB/s）。
  **G5 へ畳むと 3.6MB/run + 再計算 59.6 GFLOP/forward** — 100 run で差し引き 2.4s の得。
  タスク #7（入力値の Session 常駐）が入ったら切り出す（ADR 0047 の分岐点）。

## 4. 第 0 層の残件（実物）

- **t_embed**: グラフ内に置くと `aten.cos.default` 未対応 → **入力昇格**（ホスト 3 行・
  `cos` は語彙に足さない）。
- **LowRankAdaLN の weightless RMS**: `aten.mean.dim / aten.rsqrt.default` が残る
  （`_fold_rms_norm` は weight rank-1 前提）→ `F.rms_norm(x,(D,),None,eps)` への
  **1 行 patch**（eager ビット一致 maxabs 0.0 実測・ones 合成の既存経路に乗る）。
- **RoPE 先頭 10 head**: `chunk(2, dim=-2)` → 静的軸 slice + 静的軸 cat で**素通り・処置不要**。
- **weight_norm**: DiT 側 714 テンソル走査で **0 件**（DACVAE 側は別）。

## 5. U7 概算（形状から机上・2 FLOP/MAC・スループットは別モデル実測値の流用）

| 条件                     |               1 forward | 100 forward の GPU 目安 |
| ------------------------ | ----------------------: | ----------------------: |
| S=250（10s）・C=1519 pad | 161.2 GFLOP（attn 17%） |                 **≈2s** |
| S=750（30s）・C=1519 pad | 506.6 GFLOP（attn 21%） |               **≈6.5s** |

Tmax pad の追加コスト +15.6%（S=250）/ +7.0%（S=750）。ホスト固定費 38ms × 100 run =
**3.8s** が別に乗り、10s 発話では GPU より支配的（固定費分解の将来波が本命）。

## 6. open

- GPU 実行は未実施 — 案 c の IR の実 GPU golden 門は W2-D 実装で立てる。
- `scores[1,20,750,2269]` = 130MB は WebGPU 既定の `maxStorageBufferBindingSize`
  128MiB 超（本機実測は 2048MiB）。配布ポータビリティは limitations 起票 + 将来の融合
  attention 実行時マスク対応 or S 上限で解く。
- 38ms/run は EmbeddingGemma での帰属値 — DiT サイズでの再実測は W2-D 着地後。
