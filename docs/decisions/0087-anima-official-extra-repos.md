# 0087: Anima の配布リポを「公式 / 追加学習」の軸で割る

- Status: accepted（2026-09-01 — ユーザー裁定。旧 karume-anima-turbo の LoRA 焼き込み配布は
  公式 Turbo checkpoint で置き換え）
- Date: 2026-09-01
- 関連: ADR [0041](0041-manifest-v2.md)（1 リポ複数モデル + `defaultModel`）/
  [0073](0073-models-source-pin.md)（pin 定数 — 1 公開リポ 1 定数）/
  [0077](0077-model-version-naming.md)（モデル名 = 上流の名乗り）/
  [0038](0038-manifest-v1.md) §7（越境参照）/ [0016](0016-anima-chain-export.md)
  （LoRA 焼き込み export — 本 ADR で配布経路からは退役）

## Context

上流 CircleStone が Anima を **Base / Aesthetic / Turbo の 3 公式変種**として配り直した
（HF `circlestone-labs/Anima` = civitai 2458426・全て単一 safetensors bf16）。特に Turbo が
公式 checkpoint になったことで、従来の「base に turbo LoRA v0.2 を焼き込む」配布
（ADR 0016・`hdae/karume-anima-turbo`）は存在理由を失った。上流 README は「まず Turbo を」
と推奨し、Base は LoRA 作成向けと位置づけている。

従来のリポ割りは「LoRA 焼き込みの有無 = NOTICE（改変告知）の違い」が軸だった。素版リポには
公式 base と第三者 fine-tune（wai / copycat）が同居しており、「公式の新変種をどこへ足すか」
「第三者 fine-tune を Civitai URL / AIR 指定で気軽に足せる置き場」（backlog N3）の両方に
答えられない。

## Decision

1. **リポの分割軸は「公式 / 追加学習」にする。**
   - `karume-anima`（公式）: `anima-turbo-v1.1`（**既定** — 上流推奨・2026-09-01 裁定）/
     `anima-v1.0` / `anima-aesthetic-v1.1` / `anima-turbo-v1.0` / `anima-aesthetic-v1.0`
     = 計 5 変種（下の追記 2026-09-01 を参照）。
   - `karume-anima-extra`（追加学習）: `anima-wai-v1.0` / `anima-copycat-20260610` +
     以後 Civitai URL / AIR 指定で足す第三者 fine-tune（N3 の受け皿）。
2. **旧 `anima-turbo`（LoRA 焼き込み）モデルと `hdae/karume-anima-turbo` リポは退役。**
   pin 定数も `ANIMA_TURBO_CURRENT` を廃止し `ANIMA_CURRENT` の 1 本へ統合（breaking —
   未リリース側の JSR 公開面。公開済み旧リポの扱い〈deprecation 掲示等〉はリリース時に裁定）。
3. **text stack（text_encoder / vae_decoder / tokenizer ×2・共有 text_conditioner）は
   extra → 公式の向きで越境参照する**（ADR 0038 §7）。従来の turbo → base と同じ運用
   （公開順序: 公式を先に上げ main SHA を焼いてから extra を組む — runbook）。
4. NOTICE は従来どおり Pipeline = リポ 1 組（改変告知の正しさが分かれ目という MUST は不変 —
   軸が「焼き込みの有無」から「公式 / 追加学習」へ変わっただけ）。

## Consequences

- 公式変種の追加は `karume-anima` の 1 リポで閉じ、利用者の既定体験
  （`fromPretrained(ANIMA_CURRENT)`）が上流推奨の Turbo になる。
- LoRA 焼き込み経路（`anima/lora.py`・`export --lora`）は配布経路から使われなくなる。
  機構自体は当面残す（将来の fine-tune 取り込みで LoRA 配布物を焼く可能性・
  `assert_lora_provenance` の「無いことの検査」は旧 fused 系列の誤挿入を落とす門として
  引き続き有効）。
- 旧系列 `outputs/series/anima-turbo-*-dyn` は参照されなくなる（`lora_provenance.json` を
  持つため新しい席には入らない）。掃除は別途裁定。
- extra リポは公式リポの公開 SHA に従属する（参照先を上げ直したら extra も焼き直し —
  runbook の公開順序 MUST）。

## 追記 2026-09-01（公式は 5 変種）

決定 1 の公式列挙へ `anima-turbo-v1.0` / `anima-aesthetic-v1.0` を足す（同日のユーザー追加裁定 —
この系はバージョン間で好みが分かれるので旧版へ戻る先を残す。ADR
[0077](0077-model-version-naming.md) の動機そのもの）。公式リポは計 **5 変種**同居になる。
Context の「上流の 3 公式変種」は上流の系列（Base / Aesthetic / Turbo）の話なので据え置き —
配布モデル数 5 = 3 系列 × 版（v1.0 / v1.1、ただし Base は v1.0 のみ）。

## 追記 2026-09-03（越境参照の対象役割の訂正と、複数モデル同居リポへの適用）

- **決定 3 の「共有 text_conditioner」は誤り**。extra の 2 モデル（`anima-wai-v1.0` /
  `anima-copycat-20260610`）はどちらも `own_text_conditioner=True`
  （`tools/export-recipes/anima/distribution.py`）で text_conditioner を自前で持つ。越境参照の
  対象は **`text_encoder` / `vae_decoder` / `tokenizer` / `tokenizer_2`** の 4 役割。
- **越境参照は複数モデル同居リポの組み立てにも掛ける。** 門は「指定役割の現物が**全モデルで
  参照先とバイト同一**」— plan ごとに突合し、全 plan の参照が一致することを確認する。旧門
  「越境参照は 1 モデルの組み立てにだけ許す」（2026-08-24 — 旧 `karume-anima-turbo` が
  1 リポ 1 モデルだった時代の綴り）は、extra が 2 モデル同居になった時点で成立しなくなった。
  曖昧さの実体は「同じ役割名が別バイトを指す」ことなので、そこを直接検査する形へ置き換える。
