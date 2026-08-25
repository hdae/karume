# 0078: anima のサンプラーは既定 Euler・DPM++ 2M は request の選択肢

- Status: accepted（2026-08-25 — ユーザー裁定「デフォルトサンプラーは Euler を維持したい。
  DPM++ 2M は選択肢の一つです」。同日実装・公開まで消化）
- Date: 2026-08-25
- 関連: ADR [0041](0041-manifest-v2.md)（`pipelineConfig` は hub 素通し — サンプラー席の
  置き場）/ [0073](0073-models-source-pin.md)（pin — 公開 revision の追随）/
  [release-runbook](../release-runbook.md) §0（越境参照リポの上げ直し順序）

## Context

0.5.0 のリリース段で、出荷バイトの視認 A/B（同 step 数 = 同計算コスト）を根拠に
`pipelineConfig.scheduler.type: "dpmpp-2m"` を base / turbo の配布 manifest へ宣言した
（`d139e50`）。A/B の観測自体は有効 — seed 42 はほぼ互角・seed 7 は構図と中景の解像で
dpmpp-2m 優位・破綻の追加なし。

一方でこの宣言は、`revision: "main"` 追従の利用者に対して **seed 固定でも 0.4.x と別の画**を
配ることになる。またこの時点の実装はサンプラーが manifest 固定で、生成要求側に上書き席が
**意図的に無かった**（`d32fa5e` — 「配布者の推奨と実際に走った更新則が黙って割れる」ことを
避ける趣旨）。つまり「DPM++ 2M を使いたい利用者」は manifest を自作する以外の手が無い。

## Decision

1. **配布既定は Euler を維持する**（上流 Anima の推奨サンプラーに合わせる・ユーザー方針）。
   base / turbo とも manifest は `scheduler.type: "euler"` を**明示**宣言する（省略でも euler に
   落ちるが、裁定済みであることが読める形を取る — `anima/distribution.py` の `ANIMA_SCHEDULER`）。
2. **DPM++ 2M は「選択肢の一つ」として request 側で選ぶ** — `AnimaGenerateRequest.sampler?:
   AnimaSamplerType` を新設し、実効サンプラー = `request.sampler ?? manifest の scheduler.type`。
   `d32fa5e` の「上書き席を置かない」は本裁定で反転（配布既定 = 配布者の推奨・更新則の選択 =
   利用者のノブ、と役割を分けることで「黙って割れる」懸念は JSDoc と既定の透明性で受ける）。
   未知の綴りは GPU に触る前に期待 / 実際を並べて fail loudly（語彙の正本は config.ts の
   `SAMPLER_TYPES` 1 本）。
3. 視認 A/B の**観測は棄却しない** — 品質序列の記録として残り（`d139e50` の本文と本 ADR）、
   既定の裁定だけが上流推奨側に立つ。

## Consequences

- 公開 revision: anima `2682441a`（karume.json のみ — scheduler.type ×3）/ turbo `88357344`
  （euler 化 + 越境参照を anima@`2682441a` へ追随 + カード Usage の repo 誤記修正）。pin は
  `7ea134d` で追随。`revision: "main"` 追従の利用者は 0.4.x と同じ画へ戻る。
- 0.5.0（pin `ebb27bc4` / `6215f965`）の利用者は dpmpp-2m 既定のまま — `sampler` 席を含む
  0.5.1 で解消する（0.5.1 = pin 更新 + `sampler` 席の 2 点）。
- 上書き経路の門は「manifest 宣言で実測した golden とのビット一致」（e2e 2 本 — turbo 512² /
  base CFG4）: 宣言経路と request 経路が同じバイトを産む = 実効サンプラーの決まり方以外に
  分岐が無いことの直接証拠。
- 副次の実害修正: 公開中だった turbo カードの Usage が `hdae/karume-anima-turbo-release` という
  存在しないリポ名だった（カードは**出力ディレクトリ名**から repo を導出するため、リリース時の
  ステージングディレクトリ名が写った）。今回は正名ディレクトリで焼いて修正。恒久の注意は
  runbook §0、導出の硬化は backlog later。
