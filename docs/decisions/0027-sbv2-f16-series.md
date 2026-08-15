# 0027: SBV2 の f16 格納系列

- Status: accepted
- Date: 2026-08-03
- 関連: ADR 0018（f16 格納 — 方法論の親）/ ADR 0013（SBV2 export 戦略・--verify 排他）/
  ADR 0025 決定⑤（SBV2 に i8/w8a8 を足さない理由）

## Context

ユーザー指示: SBV2 の量子化サポート（f16 なら品質劣化が少ないことを確認済み）。ランタイムの
f16 実行経路（ADR 0018）は汎用で、必要なのはエクスポータ配線・資産系列・検証網の展開のみ。

## Decision

1. `export_sbv2.py --dtype {f32,f16}`。**i8 は足さない**（SBV2 は 5 ターゲットとも conv1d
   86〜90% 支配・linear 実質 0 GFLOP — ADR 0025 決定⑤。w8 単独の価値はサイズのみで別判断）。
   **→ この「i8 は足さない」は ADR
   [0029](0029-sbv2-i8-series-and-quant-quality.md) が上書き**（2026-08-04 — i8 系列
   〈w8a32〉を正式採択）。`--dtype {f32,f16}` 以外の本決定は現行。
2. fake-quant は共有 `round_weights_to_f16` を **remove_weight_norm / パッチ適用の後・
   参照/golden 採取の前**に export するモジュールそのものへ（順序 MUST は `_fake_quant`
   docstring）。`g` / `style_vec` はグラフ入力なので丸めない。
3. **`--dtype` は emit 専用**（--verify との併用を CLI が機械拒否）。dec / voice の --verify は
   「remove 前後のビット一致」を主張し、参照は remove 前・丸めは remove 後という順序 MUST と
   構造的に両立しない（front/flow だけ効かせる例外表は ADR 0013 が排除した形）。
4. 系列 `models/sbv2-f16/`（5 ターゲット・合計 470.34 → 237.57MB = **50.5%**。front だけ
   53.8% なのは焼き込み相対位置表 2.1MB が適格外 f32 のため）。
5. E2E は系列パラメタ化（deberta の型）+ **系列×格納 dtype の一致検査**（下記・検出限界）。
   f16 系列の 5 tolerance は実測導出（f32 側は 1 ビット不変）:
   dp atol 1e-6/rtol 1e-5（rtol 主役・実測 maxRel 1.91e-6）/ front **3e-4**（3.62e-5）/
   flow 2e-5（2.15e-6）/ dec 2e-5（2.37e-6）/ voice 1.5e-5（1.71e-6）。
6. デモは `--sbv2 {f32,f16}` ノブ。聴感ゲート（WAV）はユーザー裁定。

## 実測・知見（2026-08-03）

- **5 ターゲット中 4 つは f32 系列と同桁**（flow/dec/voice はむしろ小さい）。唯一 front の
  `logw_sdp` だけ 2.1 倍（1.75e-5 → 3.62e-5）— sdp の spline 経路のみで、他 3 出力は同桁
  （機序は推測: 丸めた重みが spline 分点を動かし逆二次解の条件数が悪い領域で増幅率が上がる）。
- **「丸め掛け忘れの検出器は emit の門だけ」（Anima の実測記録 — tools/exporter/README.md の
  故障注入表）はモデル依存の性質**と判明: SBV2 の ckpt は真の f32 なので、丸めスキップは
  emit の門（EmitError）**と** E2E（atol の 31 倍超過）の両方が赤くなる。Anima で E2E に
  映らなかったのは配布重みが BF16 で丸めがほぼ恒等だったから。README の該当表へ対照実測の
  NOTE を追記済み。
- **検出限界の新記録: 系列 root の取り違えは数値網では原理的に検出できない** — 両系列の
  実測が同桁のため、互いの tolerance を素通りする（故障注入で 52 passed を実測）。唯一の
  検出器は「系列と格納宣言の一致」検査（本波で追加）。**同じ穴が e2e_anima_test.ts /
  e2e_deberta_test.ts にもある**（隣接タスク）。
- --verify（f32・4 ターゲット）は配線変更後も過去記録と一致（dec は bit_exact 全 True 維持）。
- verify 601/0/2・pytest 1,869・ruff 緑。

## Consequences

- SBV2 全チェーンが f16 資産で回る（資産 50.5%・ロード半減見込み）。聴感は
  **ユーザー受理（2026-08-04・「劣化は感じられない」）** — f16 系列は品質込みで確定。
- 音声側の量子化はこれで一区切り（i8 は必要になったら別 ADR）。
