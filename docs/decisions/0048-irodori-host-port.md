# 0048: Irodori ホスト移植の契約 — caption_norm の所在・Unigram 共有層・latent 出口

- Status: accepted（ユーザー裁定 2026-08-11 — 第 3 波計画の軸 3 つを一括承認）
- Date: 2026-08-12
- 関連: ADR [0047](0047-irodori-dit-execution.md)（DiT 実行形）/
  [0046](0046-cat-symbolic-axis.md)（記号軸 cat）/ [0044](0044-runtime-attention-mask.md)
  （実行時 bool マスク）/ [0038](0038-manifest-v1.md)（manifest — pipelineConfig の所有）/
  [0037](0037-karume-monorepo.md)（barrel 両建て）/
  recon = [research/2026-08-11-irodori-host-recon.md](../research/2026-08-11-irodori-host-recon.md)

## Context

第 3 波はホスト側の移植 — 6 グラフ（第 1〜2 波で export 済み）を TS から駆動して latent を
出すまでの全て（tokenizer・normalize・Euler + CFG・マスク構築・S 決定）。設計軸が 3 つあった:
①duration の `caption_vec` 契約（caption_norm 済み系列の masked mean）に要る `caption_norm`
の重みが dit グラフの内側にしか無い ②Irodori の Unigram Viterbi を anima の T5 実装と
共有するか ③配布形（dist）を codec の無い今の波で出すか。

## Decision

### 1. caption_norm は caption-proj の第 2 出力（重みをグラフ外へ出さない）

`caption-proj` に第 2 出力 = `caption_norm`（RMSNorm 512）適用済み系列を追加し、masked mean
だけをホストに残す。第 1 出力（dit の `caption_state` が食う生系列）は不変 — 再 export 後も
旧 golden とビット一致（実測 maxabs 0.0）。

- 却下 a) duration(G3) へ norm + masked mean を内包 — G3 の 6 入力契約と golden 全てが変わる
- 却下 b) caption_norm の重み（512 f32）を別資産で配布 — 学習済みの値がモデルファイルの外へ
  出て、正規経路がファイル 1 個で閉じなくなる（ADR 0010 と同じ理由）

caption_norm が caption-proj と dit の 2 箇所に内包されるのは、text_norm が duration / dit に
重複内包されている既存判断と同じ規律。取り違え（text_norm を 2 回読む等）は shape も dtype も
一致して golden が自己一致するため、export 台本の常設門 `_norm_divergence`（weight 最大絶対差
≥ 1e-3 — 実測 0.318）だけが守る。

### 2. Unigram 本体は家族中立の共有層 `src/text/`

T5 の Viterbi（tokenizers `Lattice::viterbi` の写し・同点規則・`fuse_unk`）を
`packages/models/src/text/unigram.ts` へ抽出し、anima / irodori の両家族が使う。byte_fallback
は `byteBaseId` オプション（未指定 = 従来の unk 1 個 — T5 の挙動は不変で既存 anima テストが
門）。`toCodePoints` / `splitAddedTokens` も同じ理由で `src/text/` へ移設した。

- 却下: 家族へのコピー（random.ts の複製前例）— random は自明な 70 行だが、Viterbi の同点
  規則はドリフトすると**無音で**トークン列が変わる。`src/audio/` / `src/image/` の
  「パイプライン非依存の共通処理」前例に載せる。
- 逆に **Randn（splitmix64 + Box-Muller）は複製のまま**（irodori/host/random.ts）— 乱数列は
  そのファミリの出力を決める入力で、共有すると片方の都合が他方の生成物を静かに動かす。

パリティ門のフィクスチャは git 追跡（`packages/models/tests/fixtures/irodori-text/`）。語彙は
ケース再現に要る部分集合（1,213 / 102,400 本）で、`minScore` / `maxTokenLength` は語彙**全体**
の値を明示的に運ぶ（部分集合から導くと未知ノードの重みと探索幅が変わる）。NFKC は自前実装
せず `String.prototype.normalize` に委ね、Python `unicodedata` との Unicode 版ずれは
**全コードポイント掃引**（単一 cp 差分表 4,964 件 + 恒等の両方向）をテストで機械照合する。

### 3. dist は第 3 波で emit・パイプラインの出口は latent

`karume dist` に irodori ターゲットを追加し、models/tests の E2E はその生成物を読む（sbv2 /
anima と同じ規約 — テスト専用 manifest の併走仕様を作らない）。codec（G6/G7）が第 4 波なので
`IrodoriPipeline` の出口は patch 済み latent `[S,32]`（`generateLatent`）。第 4 波で同じ dist
に codec を足して波形出口を追加する（未リリースなので破壊的変更は可）。

### 4. モデル固有の数は pipelineConfig が正本・対応外モードはパース時拒否

条件 state の Tmax・speaker 行数・latent 幅・S の clamp 範囲・CFG 既定は全て manifest の
`pipelineConfig`（20 欄・手書きスキーマ）から来る — TS に直書きすると重み差し替え時に
**shape は合ったまま**沈黙誤値になる。`fromAssets` はグラフ宣言と 12 点を突合してから GPU を
取る。`speakerUncondMode` / `cfgGuidanceMode` は受理集合 1 値（"mask" / "independent"）で、
対応外は値を保持して分岐するのではなく**パース時に拒否**する（ADR 0047 決定 1 の帰結）。

### 5. seed の上流互換は非目標 — torch 突合は initialNoise 注入で行う

torch の `Generator`（CPU: MT19937 + Box-Muller / CUDA: Philox）は TS で再現しない。生成は
karume Randn の seed 決定論（同 seed → 同 latent）で、上流と同じ乱数列が要る突合（full-loop
golden）は `IrodoriGenerateRequest.initialNoise`（再現・検証用と明記した公開入口）で
ノイズごと注入する。

## Consequences

- 数値の同一性は「ビット一致」ではなく **golden 突合**（各グラフの E2E 門 + full-loop latent
  門）で担保する。既知の構造差: t スケジュール（閉形式 vs 上流 linspace）5.96e-8 /
  t_embed は torch f32 exp と JS `Math.exp` の 1 ulp 差が args≈284 の列で 3.05e-5 へ拡大
  （SLEEF を写さない判断は anima の `timestepsProj` と同じ）/ full-loop の z は GPU 経路で
  1.9e-4〜7.9e-4（DiT 単体 tolerance × 40 step の累積と整合）。
- `durationSeconds`（手動指定）は frameRate 経由の `ceil` で、上流のサンプル数経由とは
  「秒 × frameRate が整数のすぐ上」の入力だけ最大 1 フレームずれる（`sampleRate` /
  `hopLength` が pipelineConfig に無いため）。codec 波で両値が配布形に載ったら上流の綴りへ
  寄せる（`host/round.ts` の MUST に記録）。
- 空 caption / 参照なしのゼロ短絡・平均トークン前置・banker 丸めなどホスト残置の全式は
  `packages/models/src/irodori/host/`（純関数・Math.fround 逐次）に集約し、値は exporter の
  golden（`irodori_tokenizer.py` / `irodori_pipeline.py`）が固定する。
