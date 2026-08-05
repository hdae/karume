# 0035: reduce 族の軸指定（dim attr 必須 + 非最終軸のコアレス変種）

- Status: accepted（主門 = permute+行 reduce 経路との Uint32 完全一致が 9 ケース成立・
  PNG 門ビット同一 — 絵が変わらないため目視ゲート不要）
- Date: 2026-08-05
- 関連: recon = [vae-axis-reduce-recon](../research/2026-08-04-vae-axis-reduce-recon.md)
  （帰属・A3 設計・§5.2 の記号検証が正本。**R4「torch 側の縮約順序が変わる」は本実装で
  反証** — 下の知見 1）/ ADR 0011（レイアウト戦略 — 予告された「permute を消す」根治の
  最大単一例）/ 0015（attrs 宣言必須の先例）/ 0033・0034（可変解像度の両輪 — 本 ADR は
  VAE 側の帯域根治）

## Context

VAE decoder の QwenImageRMS_norm（チャネル軸 L2）はエクスポータが permute で縮約軸を
最終次元へ往復させており、その permute 対 60 dispatch が VAE 非コアレストラフィックの
99%（181ms）+ バリア律速の行 reduce 39ms = **VAE GPU の 17%** を占めていた（recon §2〜3）。
IR の `sum` が最終次元専業だったことが根本原因。

## Decision

1. **reduce 族（sum / amax / amin）に attrs `dim` を追加・宣言必須**（既定値補完なし —
   省略を許すと「チャネル軸のつもりの IR が黙って最終次元を畳む」— ADR 0015 の流儀）。
   負軸の正規化はエクスポータ境界・rank 外は両側 fail loudly。
2. **非最終軸は専用変種 `reduce:v2:<op>:…:axis:wg256`**（1 スレッド = 1 出力・縮約軸を
   inner 送りで走査 = 隣接スレッドがコアレス読み・バリアなし）。**縮約順序は行 reduce の
   「256 レーン + ビット反転二分木」を 1 スレッドの bitrev carry-stack で厳密再現**
   （recon §5.2 の記号検証 21 ケースが設計根拠・identity の +0.0 も畳まない）。
   最終次元は既存の行カーネルのまま 1 バイトも動かさない。**分岐は軸だけで決める**
   （速度で選ぶ余地を作ると既定経路のビット不変が実行時条件に依存する — executor doc）。
3. **恒久の門 = packages/runtime/tests/gpu_reduce_axis_parity_test.ts**: permute+行 reduce 経路との
   **Uint32 完全一致**（実形状 dim=96/192/384・rows 65,535 の両側と境界・巡回長 3 の
   permute・縮約長 300 > 256・amax/amin・bool sum の 9 ケース + キー踏み分け検査 +
   恒真化の門）。A1（アドレス式だけ軸対応）は縮退点のまま未使用。
4. **エクスポータ**: `_h_sum` / `_h_row_reduce`（amax/amin — 対称性のため同波で拡張）が
   1 軸を attrs へ。patch_anima の `_l2_normalize_channels` は permute 往復を捨て
   `sum(x*x, dim=1)` へ。再 emit = VAE decoder 2 系列 + **SBV2 front 3 系列**（recon §6.4
   の「sum は VAE だけ」は誤りで front が bool sum 3 本を保有 — 最終次元なので数値不変・
   IR 57 本全走査で他は 0 本と確定）+ tiny golden 5 本（io は 26 本ともバイト不変）。

## 実測（2026-08-04〜05）

- **VAE decoder IR 438 → 378 ノード（−60 = norm の permute 対ちょうど）**。
- parity 9 ケース Uint32 完全一致（bitParity: exact・A1 縮退なし）。
- **tolerance 変更 0 件**: 指示した再導出 3 対象（vae golden / 段③ / --verify）は素の
  実測が旧記録と 3 桁一致（例 --verify worst 9.336e-6 vs 旧 9.34e-6）で据え置き。
  新規導出は parity テストの CPU 参照 tolerance（atol 5e-6 / rtol 2e-5 = 実測の 7〜8.4 倍）
  のみ。
- **PNG sha256 門 2 系列とも実行前後で完全一致**（ビット同一の設計が E2E で成立）。
- VAE キー別（--gpu-timing・クロス run 比較は熱で振れるため構造変化で読む）:
  **191.25ms / 71 本 → 5.80ms / 11 本**（norm の permute 対 60 本が消滅・残 11 =
  upsample expand 6 + attention qkv 5 — recon の帰属どおり）・ 41.10ms（行）→
  **22.25ms（軸変種・−46%）**。同クロック換算で strided+reduce は −204ms 級 = recon 期待
  −208〜214ms と整合（conv2d の見かけ +54ms はクロック差）。
- 軸 reduce 22.25ms は recon の A3 楽観見積もり（6〜10ms）より高い — 1 スレッド固定
  256 反復の ALU コスト（dim=96 では 160/256 レーンぶんが identity）で、リスク R3 の
  予見どおり。それでも行 reduce から半減し、主目的（非コアレス permute の根絶）は達成。
- VAE wall（計測 off）1.6〜1.7 → **1.4〜1.5s**・運用形 1024px 生成全体 22.5 → **22.0s**。
- verify 701/0/4・pytest 1,952（メイン自己実測）。故障注入 4 件（軸 stride 固定 /
  bitrev 恒等化 / 周回落とし / carry ループ落とし — 全て赤 → 復元）。

## 検出限界・知見（本タスクの新記録）

1. **「レイアウト変更を伴わない軸移動では torch 側の参照値は 1 ulp も動かない」**
   （recon R4 の反証）: torch の permute はビューなので `permute+sum(dim=-1)` と
   `sum(dim=1)` は同じ縮約実装に落ちる（実形状 4 つで torch.equal・tiny golden io
   26 本バイト不変）。**「縮約順序が変わるはず」で tolerance を緩める判断は、必ず実測で
   確かめてから** — Pitfalls index に追加。
2. recon の census 主張（「sum を持つ実モデルは VAE decoder だけ」）は誤り — 全 IR 走査を
   仕様にした検査（57 本）で SBV2 front の 3 本を検出。**census は grep でなく IR 走査で
   機械化してから信じること**。
3. dim 宣言必須化は既存 IR の沈黙実行を fail loudly に変える（再 emit 5 系列で解消）。
   宣言必須の設計はこの「古い IR が黙って別の計算になる」事故を構造的に防いだ。

## Consequences

- VAE の非コアレス strided（126GB/s）と行 reduce（124GB/s）の非飽和が構造ごと消える
  （期待 VAE GPU −16% 級 — 実測は下表）。dispatch −60/run は非 GPU 側にも効く。
- 汎用能力（任意軸 reduce）として将来の NCHW 系（group_norm / BN 統計）にそのまま効く。
- f32 系列 Anima（models/anima/ — 資産削除済み）の tolerance 3 本は据え置き（torch 側
  ビット不変の実測より妥当・系列を再生成する場合も再導出不要の見込み）。
