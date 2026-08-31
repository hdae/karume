# FreeToken（MoE Cache）と ArrayBuffer 超級モデルの成立戦略 — 調査記録

> **性格**: 時点スナップショット（2026-08-31・Web 一次ソース + 現物コード照合。Sonnet 掃引 4 +
> Opus 深掘り 2 の統合）。裁定はここに書かない — 分岐点の列挙まで（裁定が出たら
> backlog / ADR / limitations が正本）。

## 1. FreeToken の同定と機構（一次ソース照合済み）

**FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution** —
arXiv:2608.16157（2026-08-17）/ 実装 [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken)
（Apache-2.0・NVIDIA RTX 30/40/50 系・MXFP4/NVFP4/FP8/BF16）。同名の別物（PyPI `freetoken` 等）とは
無関係。二次記事由来の性能主張（decode 3〜4 倍等）は本文から確認できず**未検証**。

機構（本文 §3–§4 由来）:

- **elastic full-expert cache**: 全 routed-expert の重みプールをホスト RAM に常駐（source of
  truth）し、GPU は **(layer, expert) 単位**の LRU キャッシュ。**予測しない（明示的に
  reactive）** — router 出力にキャッシュ内容を追随させる。実測: Qwen3.6 / RTX 5090・容量 37% で
  hit 84%・DeepSeek-V4-Flash 容量 11% で hit 61%。
- **bandwidth-adaptive CPU-GPU co-execution**: miss した expert を「PCIe 転送 → GPU」群と
  「CPU 直接実行」群に分割（分割比 q\* ≈ m·B_P/B_H）。pinned/DMA メモリ・コア pin した永続
  SIMD ワーカ・CUDA Graph 内 host-function ノードが前提。
- **device 側キャッシュ制御が必須**: 「host 側で制御すると MoE 層ごとに device 同期が要り
  成立しない」と本文 §4.1 が明言。residency table 判定〜victim 選定まで GPU カーネルで行う。
- 配布形 FTW: `l·E+e` 平坦化 layer-expert ID を先頭次元に取るバンク構成 +
  direct-to-final-layout ロード。
- 第二の柱 **agentic state reuse**: 会話ターン・tool 呼び出し等の**意味境界**に KV/recurrent
  state の checkpoint を張り、prompt 編集後も最深 checkpoint から復元。

## 2. karume への適用評定

| 要素                             | 評定                              | 根拠                                                                                                                                                                                                                                                                                                                                              |
| -------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| elastic expert LRU cache（中核） | **refuted**                       | WebGPU に **device 起動の host-memory 転送が存在しない**（writeBuffer / mapped / copyBufferToBuffer 全てホスト側エンコード）。reactive fill は router index の毎層・毎 token 読み戻し（mapAsync 往復）を要求 = ADR 0054 が排除した per-step ホスト同期の再導入。FreeToken 自身が不可と明言した「host 制御」に WebGPU は構造的に閉じ込められている |
| CPU-GPU co-execution             | **refuted**                       | pinned メモリ・帯域実測 API・グラフ内 host ノードのいずれもブラウザに無い。karume に CPU 実行経路も無い（reference は検証用）                                                                                                                                                                                                                     |
| FTW レイアウト + direct ロード   | holds（新規性なし）               | ADR 0063 / 0070 / 0081 が機能的等価を既に保有。持ち帰り = 将来 MoE の `l·E+e` 平坦化命名規約のみ                                                                                                                                                                                                                                                  |
| agentic state reuse              | **uncertain（起票候補）**         | ADR 0066 の state スロットと直交しない。checkpoint 本数ぶんの VRAM 複製が admission を直撃する制約は先方も同じ（§3.1）                                                                                                                                                                                                                            |
| expert 選択の時間局所性          | 事実として holds / 換金手段は無し | 局所性はモデル側の性質。ただし転送起動の制約（上）により karume はキャッシュに換金できない                                                                                                                                                                                                                                                        |

先行研究群（Mixtral-offloading 2312.17238 / MoE-Infinity 2401.14361 / SiDA-MoE 2310.18859 /
HOBBIT 2411.01433 / EdgeMoE 2308.14352 ほか）は**全て「host→device の細粒度転送をランタイムが
起動できる」前提**で、同じ一点で karume には一律に落ちる。予測型（SiDA / HOBBIT — 予測 recall
90% 級）は読み戻しを消せるが、ホスト側プール常駐（ブラウザ RAM 制約）と候補 b の 3 重衝突
（§4）は残る — 予測型の本文精読は未実施（**追検証余地**）。

**帰結**: 現時点のブラウザ WebGPU で MoE を回す唯一の成立形は**全 expert VRAM 常駐**
（= dense と同じ扱い・予算は active でなく総パラメータで組む）。karume の shard 分割配布は
この路線に既に最適化されている。

## 3. 「ArrayBuffer の壁」は既に移動している

| 壁                                        | karume での現状                                                                                                                                                     |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chromium 単一 ArrayBuffer 2,145,386,496 B | **解決済** — shard v2 の 1GiB 上限（ADR 0081）で構造的に半分以下                                                                                                    |
| ホスト RAM ピーク                         | **解決済** — 相 2 逐次ストリームで O(最大 shard)                                                                                                                    |
| **VRAM 総量**                             | **未解決 = 本当の壁**。admission は必要側しか出さず空き側と比較しない（ADR 0070 決定 5 — WebGPU が総/空きを露出しないため意図的）。「入らない」は落ちて初めて分かる |
| 単一テンソル > maxBufferSize              | export 時の手割りで回避（gemma4 PLE 35 分割が前例）。ランタイム側の分割常駐は未実装（KV は連続容量の明示裁定 — ADR 0066 追記 5）                                    |

付随の実測事実: ①requiredLimits は adapter 実測値をそのまま要求する設計（`device.ts` の
`planRequiredLimits`）なので効くのは実機 granted limit ②圧縮適格外 initializer は f32 展開
常駐（`expanded` 席）で i4→f32 は 8 倍化 ③companion scale は F32 固定
（`container.ts` `SCALE_DTYPE`）— i4 g32 でペイロード +25%（8×7B 級で ≈5.8GB）。

## 4. スケール別マトリクス（裁定材料）

|                                        | gemma4 E2B 級（3.70GiB） | 8×7B MoE 級（i4g32 ≈29GB）                                      | 更に上（70B dense / 400B MoE）    |
| -------------------------------------- | ------------------------ | --------------------------------------------------------------- | --------------------------------- |
| a. 現行 shard 逐次面                   | ◎ 成立済み               | △ VRAM 次第（32GB 級 dGPU でギリギリ）                          | × VRAM で不成立                   |
| b. MoE 動的常駐                        | 対象外                   | × 3 重衝突（下記）                                              | × 同左 — ここが唯一の道           |
| c-1. ホスト gather（PLE 方式の一般化） | ◎ 起票済み（−59%）       | ○ embedding / lm_head に効く（expert FFN は行疎でなく効かない） | △ 主役にならない                  |
| c-2. 量子化                            | ◎ 既定経路               | ○ 前提（scale F32 +25% が効く）                                 | ○ 前提だが不十分                  |
| c-3. 複数バッファ分割常駐              | △ 不要                   | △ 不要（最大テンソル 65MB 級）                                  | ○ 巨大 embedding で必要になりうる |

候補 b の 3 重衝突（file:line 裏取り済み）: ①`ShardValidator.finish()` の全 initializer
存在要求（`container.ts:261-278` — 全量/逐次 2 面で共有された唯一の門）②`weightBuffers` が
Session 構築時 1 回組みの不変 Map（退避/再ロードの席なし）③**IR v1 に値依存の実行選択が無い**
（op-vocabulary の意図的保留・topk も static-k）— MoE は dense 展開でしか書けず VRAM を
節約できない。候補 b は機能追加でなく **IR 語彙拡張 + フォーマット門改訂 + Session 寿命
モデル変更の 3 モジュール横断再設計**。

wasm64 / `WebAssembly.Memory` 迂回（ONNX Runtime Web の方式）は karume には不要かつ規約違反
（WASM 依存なし・shard で解決済み）。web-llm の shard 方式は karume shard v2 と同型で学ぶ
ものは残っていない。

## 5. 裁定を要する分岐点（裁定はここに書かない）

1. **値依存の実行選択を IR v1 の語彙に入れるか** — 候補 b の成否がこの 1 点。決定性・
   op-vocabulary の保留方針の撤回を伴う ADR 級。
2. **「未着荷 initializer」席を ShardValidator に新設するか** — 1 に従属。
3. **companion scale F32 固定を設計軸に昇格させるか**（f16 化で i4 ペイロード −10% 級）。
4. **admission に空き VRAM 比較を入れるか** — ADR 0070 の禁止理由（露出 API なし）は今も
   有効。入れるなら別の推定源が要る。

隣接（報告のみ）: `packages/models/src/hub/components.ts:17-21` のコメントが ADR 0081 以前の
記述で stale（グラフ shard は現在データ節 0）。

## 6. 限界

- FreeToken の性能主張（3〜4 倍等）は二次記事由来で未検証。著者所属も未確認。
- 予測型 offloading（SiDA / HOBBIT）の本文精読は未実施 — 候補 b を将来 reopen する際の
  最初の読み物。
- OPFS / Cache API の quota 実勢（Chrome = 総ディスク 60% 等）は MDN 記載値で、実機実測なし。
