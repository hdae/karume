# 0089: GPU メモリ適合は「絶対上限との決定論的比較」へ昇格する（メモリ管理波 Phase A）

- Status: accepted（2026-09-01 — ユーザー裁定「OK、進めてください」・設計裁定 1a / 2a / 記録承認）
- Date: 2026-09-01
- 関連: ADR [0070](0070-shard-loading-admission.md)（決定 4 = errorScope 全面依存・決定 5 =
  空き側と比較しない・決定 6 = limits preflight の席予約 — 本 ADR がその席の実装）/
  [0066](0066-generation-context-state-slots.md)（追記 5 = state の binding 上限検査）/
  [0038](0038-manifest-v1.md)（§7 追記 = `requiredLimits` 欄の導入）/
  [0081](0081-shard-spec-v2.md)（shard 上限 — 本 ADR が読み手側の門を実装。値と測り方は ADR
  [0090](0090-shard-spec-v3-tensor-pieces.md) で 256MiB・ファイル長へ）

## Context

確保失敗の検出は shard 単位の out-of-memory / validation errorScope に全面依存していた
（ADR 0070 決定 4）。この網には 3 つの弱点がある: ①**実装依存** — M2 実機で out-of-memory
errorScope が沈黙する実測（known-issues・wgpu-hal metal の `check_if_oom()` は no-op）が
あり、沈黙環境では「確保失敗 = 無効バッファへの no-op writeBuffer = ゴミを読む」が通り得る
②**遅い** — 検出は数 GiB を転送した後 ③**粗い** — 名乗れるのは失敗 shard までで、どの重みが
何バイト超えたかは出ない。

一方で確保寸法は**宣言だけで確定している**（常駐計画 `planWeightResidency` は prepare 相の
純関数・state スロットも束縛で決まる）。また manifest の `requiredLimits` 欄は「exporter が
書かない（gemma4 recipe だけが手書き）・hub が parse するだけ・読み手ゼロ」の三重に空で、
shard 1GiB 上限（ADR 0081）の門は書き手側にしか無かった。

## Decision

**決定論的に比較できるものは全て、確保・取得の前の明示検査に昇格する。** 比較の相手は
デバイスの**絶対上限**（`maxBufferSize` / `maxStorageBufferBindingSize`）だけで、空き VRAM・
確保合計とは比較しない（ADR 0070 決定 5 の維持 — WebGPU は総量を露出せず、当て推量の閾値は
健全な環境で誤拒否を作る）。総量の最終検出は errorScope のままで、Metal の沈黙環境では
それが残る — [limitations](../limitations.md) の「GPU メモリの事前検査は絶対上限まで」節が
この限界の記録。

1. **重み経路の明示検査**（runtime）: 席 → 確保バッファの写像を
   `planWeightBuffers`（weight-residency.ts）へ 1 本化し（適格 = payload・適格外 = f32
   展開後・i8 / i4 は companion scale をもう 1 本・`toSizeClass` 整列後 = createBuffer 実寸）、
   `assertWeightsWithinLimits` が Session 構築の入口（shard ループ前 = 1 バイトも上げる前）で
   全エントリを両上限と突合する。超過は**全件列挙して 1 回で**落とす。見積り
   （estimate.ts）も同じ写像を消費する — 検査・見積り・実ロードが別の寸法を主張する形を
   作らない。
2. **state 検査の補完**（runtime）: 既存の binding 上限検査（ADR 0066 追記 5）に
   `maxBufferSize` も加える。`binding ≤ buffer` は device を自前計画する側の関係であって、
   外から渡された GpuContext への保証ではない。
3. **`requiredLimits` の書き手は core の組み立て一括導出**（exporter `karume/limits.py` +
   `dist.py` の `bake_required_limits`）: 需要 = quant が選ぶコンポーネントの最大テンソル
   payload と最大 state スロット（容量は `pipelineConfig` で数値化・記号 2 本以上や束縛欠落は
   fail loudly）。**常駐前提の寸法のみ** — f32 展開のワーストを要求に書くと本来動く環境を
   DL 前に誤拒否する（展開時の実寸は決定 1 の実行時検査が守る）。**焼くのは WebGPU 保証既定
   （256MiB / 128MiB）を超える席だけ**（「欄なし = 既定スペックで動く」の意味論）。
   workgroup 系はカーネル設計依存なので焼かない。**計画側の手書きは二重管理として拒否**
   （gemma4 recipe の手書き席は退役 — core 導出値と完全一致を確認済み）。

   > **意味論の限定（2026-09-05 裁定 3b）**: 「欄なし = 既定スペックで動く」は**常駐分
   > （重み・state）が既定内**という主張で、**中間テンソル（ノード出力）は含まない**。
   > 中間を静的に数えて焼く案は採らない — 融合後の実需要は granted limit に依存する
   > （行ブロック attention は S を全幅で実体化しない）ので配布形からは原理的に決まらず、
   > 融合前の最大を焼くと**動く device を DL 前に誤拒否する** = 決定 3 の MUST が禁じる
   > 失敗形になる。実例: BiRefNet 1024² は最大 initializer 38MiB で欄が空になるが、
   > 実行時は 1 binding 約 1GiB が要る。中間の上限超過は ADR
   > [0093](0093-transient-liveness-packing.md) 決定 5 の計画時 preflight（Session 構築時に
   > 全件列挙して落とす）が受け、それより前の段では「落としてから落ちる」に留まる
   > （資源の目安はモデルカードの注記で告知する）。
4. **shard 上限の読み手検査**（hub — 当時 1GiB・ADR 0090 で 256MiB のファイル長へ）: manifest parse
   時に `shards` の各 `size` を検査（閉区間・Python 正本 `shards.py` と同値）。`assets` / `extras` は
   対象外 — 上限は shard 分割の契約で、上限超の単一付帯資産は合法。
5. **席（次段 = 波 2）**: manifest `requiredLimits` の読み手（models の admission で GPU
   limits と比較して DL 前拒否・adapter 取得の先行）と、`estimateSessionMemory` のロード面
   結線。ADR 0070 決定 6 の残り半分。

## Consequences

- dist 再生成で `requiredLimits` 欄が新設される: anima 全 30 quant（296.8MiB）・irodori
  f32（300MiB 両上限）/ f16（150MiB・binding のみ — 片側だけ超過する帯の実例）。gemma4 は
  手書きと導出が同値なのでバイト不変・sbv2 は既定内で無風。再生成は models 読み手（波 2）と
  同じ回で行う。
- 検査が新たに弾く既存資産は無い: bind group は常にバッファ全体を束縛するので、binding 上限を
  超える重みが正常動作していた経路は存在しない。
- 残る同型の弱点（起票のみ・本 ADR 対象外）: `GpuContext.createResident` と run 時 transient
  の確保は依然 errorScope 頼み。gemma4 PLE sidecar は shard 連番形だが `extras` 席なので
  決定 4 の門の外（現物は上限内）。
  - **閉じ（2026-09-05・ADR [0093](0093-transient-liveness-packing.md) 決定 5）**: run 時
    transient のぶんは、計画時に「slot 実寸 > `maxStorageBufferBindingSize`」「領域 >
    `maxBufferSize`」を確保の前に全件列挙して落とす形で閉じる（`createResident` の席は残る）。

## 追記 2026-09-01（波 2 = models 読み手結線の実装）

- **limits の入手 = 案 A（アダプタを読んで捨てる）で実装**（裁定 1a）。runtime に
  `readAdapterLimits`（`requestAdapter` → `planRequiredLimits`・絞り無し・device は作らない）を
  新設し、models の `session/gpu-features.ts` に `assertRequiredLimitsSatisfied`（部分写像・
  不足は全件列挙）と `assertRequiredLimitsBeforeDownload`（共有 GPU なら `GpuContext.limits`、
  自前経路ならアダプタ実測値）を同居。席は 7 家族の admission（共有 GPU・`fromAssets` も守る）
  - 8 家族の `fromPretrained` admission 閉包（重み prefetch の前）。gemma4 は
    `gemma4ManifestConfig` が quant を返す形へ変更して閉包側に席を置き、`fromAssets`
    （manifest 無し）は対象外（limitations 記録）。
  - 追記（2026-09-05・W-M6-2）: `assertRequiredLimitsBeforeDownload` は宣言の有無に依らず、共有 GPU が
    無ければ `readAdapterLimits()` を 1 度読む（比較は宣言が無ければ no-op のまま = 「欄なし =
    既定スペックで動く」の意味論は不変）。買うのは GPU 不在の早期検出だけで、limits 検査の強化
    ではない。
- 案 B（device を DL 前に取る）/ 案 C（アダプタを持ち回る）を採らなかった理由: 8 家族の
  「資産の解析は GPU を取りに行く前」の順序 MUST と衝突する。加えて WebGPU 仕様はアダプタが
  「いつでも失効しうる」（システム変化が無くても数秒〜数分後でも可、と Note が明記）と定め、
  失効後の `requestDevice` は例外ではなく生まれた時点で lost な device を返す静かな失敗になる。
  実装者側の指針も「device 要求の直前に新しいアダプタを取れ」。A の弱点（2 回の
  `requestAdapter` が同じ物理アダプタを返す保証は無い）は、決定 1 の構築時検査が最終検査として
  残ることで吸収する（DL 前検査は事前判定）。
- **`estimateSessionMemory` のロード面結線（決定 5 の後半）は Phase B へ送る**: 見積りは
  グラフ入力の記号次元の全束縛が前提（未束縛は fail loudly）で、ロード時に束縛値を持つのは
  gemma4（capacity / chunkLength が config にある）だけ。他 7 家族は生成時の形状に依存するため、
  今結線すると家族ごとに恣意的な代表形状を焼くことになる。limitations の該当節は「呼び手に
  判断材料を渡すところまで」が目的で拒否は担わないので、遅らせても安全性は落ちない。Phase B の
  RAM ピーク実測で「どの数字が判断に効くか」を見てから席を決める。
- **dist 再生成は既存 5 ミラー**（karume-anima / irodori v4-small / irodori v4.1-small /
  sbv2-jvnv / gemma4-e2b）で実施。Consequences の見込みどおり: anima 30 quant に
  `maxBufferSize` = `maxStorageBufferBindingSize` = 311,164,928（296.75MiB）、irodori f32 に
  両上限 314,572,800（300MiB）/ f16 に binding のみ 157,286,400（150MiB）、gemma4 は manifest
  不変、sbv2 は欄なし。shard のサイズ・sha は全て不変（manifest の差分は `requiredLimits` 欄
  のみを機械比較）。「全 8 家族」は文書上の数え方で、siglip2 / birefnet / depth-anything /
  vowel-detector は未配布（ミラーも pin も無い）— ~~初回組み立てはリリース波~~ → リリース後
  （2026-09-03 裁定・backlog release 節）。
- 再生成で見えた既存ドリフト 2 点（本 ADR の対象外・記録のみ）: ①sbv2 の
  `shared/text/symbols.json` が源（`outputs/misc/sbv2-demo/`・2026-08-30 17:40 再生成）に追随して
  1,642 → 1,647 バイトへ変化（WAV 凍結 e2e は緑 = 出力に影響しないメタデータ差）②gemma4 と
  irodori v4-small の README がカード生成器の文言変更に追随（折り返し・「v4 (Small)」→
  「v4 Small」）。

## 追記（2026-09-02 — Phase B 実測と Phase C-1・見積り結線の据え置き）

- **Phase B 実測**（[research/2026-09-02-shard-size-ram-peak.md](../research/2026-09-02-shard-size-ram-peak.md)）:
  ホスト RAM ピークは「定数 + k × 最大 shard」で k ≈ 3（Linux）/ 2.4（Mac）。shard 内の完了待ちを
  刻む案は Vulkan で無効・Metal で −13%。明示 GC で shard 1 本分。効くレバーは shard サイズと、
  shard を読む器の使い回し。
- **Phase C-1 = 器の使い回し**を実装（契約変更は ADR [0070](0070-shard-loading-admission.md) 追記
  2026-09-02 が正本）: ピーク ≈ 0.45GB + 最大 shard 1 本（anima f16 1GiB shard 4,069 → 1,402 MiB・
  gemma4 2,622 → 1,116 MiB）。HF 経路は取得層側の対応待ち（→ fetch-cache 0.6.0 の `into` で 2026-09-02 夜に配線済み — ADR 0070 追記）。
- **`estimateSessionMemory` のロード面結線は据え置きを確定**: Phase B で「判断に効く数字」はホスト側
  （manifest の最大 shard から導ける）であって GPU 見積りではないと分かった。GPU 見積りは引き続き
  呼び手が生成形状を決めた後に使う面（limitations「未実装」節はそのまま）。
- 残る裁定 = 書き手の shard 目標値（512 or 256MiB — ADR 0081 側の 2 値化）。C-2 候補 = テンソル単位
  ストリーミング（ピーク → 定数 + 最大テンソル）。
