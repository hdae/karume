# モデル拡充 recon — EmbeddingGemma / Irodori-TTS v4 / BiRefNet_HR / Gemma 4 E2B + 追加候補

> 時点スナップショット（2026-08-11）。Web 掃引 5 レッグ + コードベース棚卸し 1 レッグ +
> torch.export 実測スパイク 1 レッグの統合。EmbeddingGemma は同日中に実装着地したため
> §1 は「当時の判断材料」、§2〜§5 が生きているキュー情報。出典 URL の取得日は 2026-08-11。

## 1. EmbeddingGemma-300m — 着地済み（判断材料の記録）

- 実装の正本: `tools/exporter/export_embeddinggemma.py` / ADR 0043（gelu_tanh）/
  `packages/runtime/tests/e2e_embeddinggemma_test.ts` / limitations.md（attention_mask 非対応）。
- **公式 gated リポと unsloth ミラーは重み sha256 全一致**（model.safetensors =
  `cbf5a783…`・2_Dense / 3_Dense・config / tokenizer 系も cmp 一致）— ミラーの信頼性は
  事実として確定。公式は gated="manual"（同意で即時承認）・license は Gemma Terms
  （Apache 2.0 になったのは Gemma 4 のみ）。
- スパイク実測の要点（実装前の確定材料）:
  - 既定 preserve の未対応 op は **4 種のみ**（bitwise_or.Tensor / gt.Tensor / sin / cos）。
    sin/cos は RoPE バッファ（layer_type 接頭辞付き `sliding_attention_inv_freq` 等）が
    降格されないことが原因で、降格 + FOLDABLE 2 追加で消える。
  - **双方向 sliding マスクの実効半幅は 256（= sliding_window/2）の対称バンド**（-inf 率
    0.2490 を定数実測）。「T ≤ 512 なら恒真」ではない — 本物のマスクが残るが、定数畳み込み +
    safe-softmax ガード除去（各行有限の証明・24 層で発火）で問題なく通る。
  - SDPA 保存（WITH_ATTENTION）は bool initializer と attn_mask 契約欄なしの 2 門で不成立 —
    **既定 preserve（分解）が正解**。eager は softmax(dtype=f32) の壁が 1 枚余分で sdpa 優位。
  - fp16 活性は公式に非サポート（モデルカード明記）。f32 経路で移植した。
- ORT 比較ベースライン: onnx-community/embeddinggemma-300m-ONNX（fp32 1.23GB 〜 q4f16
  175MB・6 変種・external data・com.microsoft contrib op 使用・入力 input_ids +
  attention_mask → 出力 sentence_embedding）。transformers.js で実運用実績。公表レイテンシは
  EdgeTPU 値のみで **Web/Node の数値は自前計測が必要**。

## 2. Irodori-TTS v4 — キュー第 2 位（着手時にソース精読 recon 必須）

- 正体: Aratako/Irodori-TTS-v4-Small（HF・766M・MIT）。**SBV2 系列とは無関係**の独立系
  日本語 TTS — Rectified Flow DiT が Semantic-DACVAE-Japanese-32dim（48kHz）の連続潜在上で
  動く。テキスト（**ModernBERT-ja**）・参照音声（最大 120 秒）・キャプション（Voice Design）
  の 3 条件を単一 ckpt に統合、絵文字スタイル制御。GitHub: Aratako/Irodori-TTS。
- 構成 5 部: ModernBERT-ja エンコーダ / 条件プロジェクタ / 参照潜在エンコーダ /
  Joint-Attention DiT（RoPE・RMSNorm・LowRankAdaLN・SwiGLU）/ DurationPredictor。
- 移植観: 40 step Euler ループはホスト駆動（Anima と同型）。duration の動的出力長は SBV2 と
  同型のホスト解決で吸収できる見込み。CFG の t.item() 分岐もホスト側。ONNX 前例なし・
  config.json は HF に無く GitHub 側 YAML 管理（非標準パッケージング）。
- **確度注意**: この節の大半は WebFetch 要約経由（confidence medium/low）。file:line の
  裏取りをしていない — 着手時はリポ clone + ソース精読 recon から。
- 副産物: 移植すると ModernBERT（RoPE encoder + local/global 交互 attention）対応が
  自動的に手に入る — 単独の ModernBERT/Qwen3-Embedding 追加は限界効用が低い。

## 3. BiRefNet_HR — キュー第 3 位（blocker あり・ADR 前提）

- ZhengPeng7/BiRefNet_HR（MIT・~0.2B・2048×2048 学習）。Swin-v1-L backbone + 多段 decoder。
  RMBG-2.0 は同アーキテクチャの別重み（BRIA 独自データ・独自ライセンス）。
- **正面 blocker: decoder の ASPPDeformable が torchvision `deform_conv2d`**（カスタム op）。
  ONNX エクスポータですら非対応で、コミュニティ回避策は① grid_sample 等価書き換え
  （<1e-3 検証済みの前例）② カスタム op 登録③ opset 19 native DeformConv（BiRefNet#167）。
  karume 的には grid_sampler 系 = op 語彙台帳「難しい・遅い 15」の保留枠 — **設計裁定
  （ADR）が先**。値依存 gather + 双線形補間なので決定性は保てる見込み（scatter 系ではない）。
- backbone 側（shifted window attention の roll・patch merging・bilinear interpolate）は
  固定解像度なら静的に落ちて障害なし（roll は slice+cat へ分解可能）。
- HR 版そのものの ONNX 配布は未確認（onnx-community にあるのは標準版・lite 等）。

## 4. Gemma 4 E2B — キュー第 4 位（実行モデル設計が本体）

- **Gemma 4 は実在**（2026-04-02 発表・Gemma 3n 後継・E2B/E4B/26B-A4B MoE/31B Dense・
  arXiv:2607.02770）。E2B = google/gemma-4-E2B(-it)。**Apache 2.0 + ungated**（config.json
  匿名取得を実測 — EmbeddingGemma より条件が良い）。onnx-community の ONNX / LiteRT-LM の
  .litertlm（2.58GB・E2B 動作確認済み）が存在。
- text_config 実値: 35 層 / hidden 1536 / vocab 262144 / **MQA**（kv_heads=1）/ sliding 512
  （5 層に 1 回 full）/ **head_dim が層種別で 256(sliding)/512(full)** / 128K ctx /
  final_logit_softcapping 30 / hidden_activation = **gelu_pytorch_tanh**（今回の gelu_tanh を
  そのまま再利用）。
- 継承機構: **PLE**（per-layer embedding 262144×(35×256) + gate/projection）と **KV 共有**
  （後半 20 層が前方の KV を流用・double-wide MLP 併用）。どちらも **layer_idx の静的分岐**で
  データ依存ではない — 静的形状 IR と抵触しない見込み。AltUp / LAuReL / MatFormer /
  activation sparsity は modeling_gemma4.py（2666 行）に**痕跡なし**（廃止の公算・Gemma 3n
  側 config が gated で直接 diff は未取得）。E2B は dense（MoE コードパスを通らない）。
- karume 的な本体は op ではなく**実行モデル**: 自己回帰 decode + KV cache =「run より長い
  寿命の名前付き GPU 値」で、prepared 機構（ADR 0042）の器の上に設計する話。長さバケット化
  前提。tokenizer（Gemma SPM 262k）も EmbeddingGemma と共用。

## 5. 追加候補（アーキテクチャ検証価値順・2026-08-11 裁定なし）

| 候補                          | 規模      | 検証価値（新規面）                                                                          | ブラウザ前例                            |
| ----------------------------- | --------- | ------------------------------------------------------------------------------------------- | --------------------------------------- |
| Moonshine-tiny / Whisper-tiny | 27M / 39M | encoder-decoder + **ホスト自己回帰 + KV cache**（E2B の前哨戦）。Moonshine は RoPE + 可変長 | transformers.js 稼働・WebGPU デモあり   |
| Kokoro-82M                    | 82M       | **iSTFT vocoder**（逆 STFT / overlap-add の新 op 族）・SBV2 と別系統 TTS                    | kokoro-js が WebGPU 完走                |
| Depth-Anything-V2-Small       | 25M       | バックボーン中間層の多点取り出し + DPT 融合。静的・低リスク                                 | onnx-community 3 サイズ + pipeline 実績 |
| MobileSAM                     | ~10M      | 重い encoder 1 回 + prompt decoder N 回の **2 段実行**。固定形状マスク出力                  | SAM 系 transformers.js v2.14〜          |
| Real-ESRGAN x4plus            | 17M       | **pixel_shuffle** 族 + 純 conv（i8a8 conv 波と検証対象が重なる）                            | サードパーティ ONNX のみ（弱）          |
| SigLIP2-base (FixRes)         | 93M       | 2 塔 dual encoder。新規 op はほぼ無し                                                       | onnx-community あり                     |
| DINOv3 ViT-S/16               | 21M       | ViT への RoPE + register トークン                                                           | gated ライセンス注意・前例は二次情報    |

除外の裁定済み: Parakeet TDT（RNNT 系デコードの出力ステップ数がデータ依存 — 静的形状と
衝突の公算）・ModernBERT / Qwen3-Embedding 単独（§2 の副産物と重複 / DeBERTa と面が重なる）。

## 6. 記録の所在

- recon レッグ生データ: セッション scratchpad（揮発）。本書が恒久要約。
- EmbeddingGemma 実装の実測（IR 1681 ノード・required_ops 19・e2e maxAbs ≤ 3.9e-7 等)は
  コミット `74c0eab`〜`1cd3d2b` の本文と e2e テストのコメントに恒久化済み。
