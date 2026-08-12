# Irodori 推論ホスト経路 recon — 第 3 波（TS 移植）の対象確定

> 時点スナップショット（2026-08-11・上流実装 clone `inputs/irodori/Irodori-TTS/` の静的読解 +
> 一部は exporter の常設門による実測）。移植の裁定は ADR
> [0048](../decisions/0048-irodori-host-port.md)、DiT 実行形は [0047](../decisions/0047-irodori-dit-execution.md)
> が正本。file:line は読解時点の上流 clone のもの。

## 1. トップレベルフロー（`inference_runtime.py:1036 synthesize` — 一本道）

normalize_text → strip（空なら fail）→ text/caption tokenize（256 / 512・空 caption は
`caption_mask.zero_()`）→ 参照 latent 準備 → duration 予測 → S 決定 → Euler →
unpatchify（patch=1 で恒等）→ codec decode → 末尾トリム → 透かし。上流は encode_conditions を
duration 用と Euler 用で **2 回**回す（キャッシュしない）— karume は 1 回で共有する。

## 2. ホストが写した式（正確な形）

- **S 決定**（:1244-1334）: `frames = expm1(log_frames).float().mean()` →
  `S = clamp(round(frames), ceil(0.5×25)=13, floor(30×25)=750)`。`round` は Python の
  **偶数丸め**（TS は banker's rounding を明示実装 — `Math.round` だと .5 で 1 ずれる）。
  手動指定: `clamped=min(30,max(0.5,s))` → `target=max(1,int(clamped×48000))` →
  `S=ceil(target/1920)`（karume は frameRate 経由 — 最大 1 フレーム差・ADR 0048）。
- **Euler**（rf.py:189-208, 582）: `t_i = 0.999×(1−i/40)`（上流は
  `(1−linspace(0,1,41))×0.999` — 閉形式との差 5.96e-8）・`x += v×(t_next−t)`（dt 負・一定）。
  初期ノイズ `randn(1,S,32)`（rf.py:158-166 — generator は x_t → speaker noise の順に消費）。
- **CFG independent**（rf.py:264-341, 459-483）: scale 既定 text 3.0 / speaker 5.0 /
  caption 3.0。有効 = text（scale>0）/ speaker（scale>0 かつ参照あり — no_ref は
  `resolve_cfg_scales` が 0 に潰す）/ caption（非空 かつ scale>0）。窓は
  `cfg_min_t=0.5 ≤ t ≤ cfg_max_t=1.0` → 40 step 中**前半 20 step**。合成は
  `v = v_cond + Σ s_k(v_cond − v_k)`（text → speaker → caption の順・差の基準は常に v_cond）。
  uncond は state 0 + mask 0 — karume は「cond state + 区間マスク 0」に還元（ADR 0047 決定 1）。
- **t_embed**（model.py:45-54）: `freqs_k = 1000×exp(−ln 10000 × k/256)`・
  `[cos(t·freqs) | sin(t·freqs)]` の 512 次元（**cos 前半**）。
- **マスク**（model.py:357-411）: 連結順 self(S) → text(256) → speaker(751) → caption(512)・
  bool は **True = 参照可**。self は推論で常に全 True。
- **speaker 経路**（model.py:1783-1809, 1610-1624）: 参照 latent [T,32] → 4 行 patch
  （余り切捨・model.py:114-149）→ encoder + speaker_norm（G2 に内包）→
  **平均トークン前置はホスト**（`(state·mask).sum/max(mask.sum,1)` — B=1 全 True なら単純平均）。
  `speaker_state_override`（--ref-embed）は encoder / norm / 前置を**全て通さない**別系統。
  no_ref は出力が厳密 0（exporter `_no_reference_evidence` が毎 emit 実測）→ ホストのゼロ短絡。
- **duration 入力**（model.py:1317-1375）: text_state（norm 前・詰めた長さ）/
  `speaker_vec = speaker_state[:,0]` / has_speaker / caption_vec（**caption_norm 済み**の
  masked mean — caption-proj 第 2 出力の由来・ADR 0048 決定 1）/ has_caption。
  aux 14 特徴は `token_sum_dual_adarn_zero_no_aux` では**一切読まれない**
  （`_duration_aux_is_inert` が毎 emit 実測）→ `build_duration_features` は移植不要。
- **テキスト前処理**（text_normalization.py:196-210・tokenizer.py:81-136）: SIMPLE 10 規則 →
  REGEX 4 規則 → 外側括弧剥がし（全体を囲む間ループ）→ NFKC → `...`/`..`→`…` の順序固定。
  caption は normalize を**通さず** strip のみ。tokenize は `add_special_tokens=False` +
  body を max−1 で切って BOS(1) 手前置 + pad(3) 右詰め（静的方式のホストは pad を渡さない）。

## 3. 既定値で死んでいる上流機能（第 3 波の対象外 — limitations に固定）

LoRA 動的ロード / speaker_kv_scale 系 / truncation_factor / temporal_score_rescale /
sway スケジュール / joint・alternating CFG / speaker_uncond_mode="noise" /
num_candidates>1・decode_mode="batch" / SilentCipher 透かし（未導入なら warning のみ）。
いずれも SamplingRequest / infer.py の既定で無効（inference_runtime.py:200-246 ほか）。

## 4. codec 境界（第 4 波の入口）

Euler の戻り [1,S,32] → `unpatchify_latent`（patch≤1 で恒等）→ `z[:, :S]` →
`decode_latent`（transpose して [1,32,T] → decoder）→ [1,1,samples] → `target_samples =
S×1920` で切り → `find_flattening_point`（z 上の 20 行窓・**窓全体のスカラー** std<0.05 かつ
|mean|<0.1 — per-dim ではない）で更にトリム。denormalize は無い。参照音声側の encode には
audiotools の LUFS 正規化（-16dB）が挟まる（codec.py:133-254 — 第 4 波で要再現）。

## 5. seed 再現性（非目標の根拠）

torch CPU generator は MT19937 + Box-Muller、CUDA は Philox — device 間ですら一致せず、TS で
のビット再現は非現実的。karume は Randn（splitmix64 + Box-Muller）の自己決定論 +
`initialNoise` 注入口（ADR 0048 決定 5）。full-loop golden の噛み合わせは exporter
`irodori_pipeline.py` の常設門（S 一致・初期ノイズのビット一致・上流 `sample_euler_rf_cfg`
との z 突合 ≤1e-3・CFG 実効性）が毎 emit 実測する。
