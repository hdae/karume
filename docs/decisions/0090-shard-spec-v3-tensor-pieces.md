# 0090: shard 仕様 v3 — テンソル分割（piece）・上限 256MiB 1 本・ファイル長で測る

- Status: accepted（2026-09-02 — ユーザー裁定「この方針で進めてください」。実装前の成立性は
  [research/2026-09-02-tensor-piece-split.md](../research/2026-09-02-tensor-piece-split.md)）
- Date: 2026-09-02
- 関連: ADR [0081](0081-shard-spec-v2.md)（shard 仕様 v2 — **決定 2 の上限 1GiB と追記 2026-09-02 の
  「書き手の目標 256MiB / 実効目標 = max(目標, 最大単位)」を本 ADR が置換**。決定 1 / 3 / 4 / 5 は
  据え置き）/ [0070](0070-shard-loading-admission.md)（決定 1 の co-shard を「scale は piece 1 と
  同居」へ拡張・決定 3 の逐次消費は不変・追記 2026-09-02 の器の使い回しが上限の根拠）/
  [0089](0089-memory-limits-preflight.md)（読み手側の shard 上限検査 — 値と測り方が変わる）/
  [0063](0063-safetensors-physical-layout.md)（shard 単体の物理配置は不変 — piece は普通のテンソル
  として並ぶ）/ [0069](0069-int4-storage.md)（i4 の group scale は先頭次元が行 — piece の切り方と
  一致）

## Context

ADR 0081 追記（2026-09-02）で書き手の目標を 256MiB にしたが、**テンソルは割れない**ため「目標より
大きい単位を持つコンポーネントはその単位まで持ち上げる」（実効目標 = max(目標, 最大単位)）という式が
残り、受理上限（1GiB）と目標（256MiB）の 2 値が併存していた。実資産では 3 本の埋め込み表
（gemma4 `lm_head` 384MiB / irodori `tok_embeddings` 300MiB / anima `embed_tokens` 297MiB）だけが
該当し、そのコンポーネントの shard は 385 / 300 / 297MiB に持ち上がっていた。ロード時ホスト RAM
ピークは「定数 + 最大 shard 1 本」（ADR 0070 追記）なので、この持ち上がりがそのままピークに乗る。

加えて、上限を数える量が両側で違っていた: exporter はデータ節、hub は manifest の `size`
（ヘッダ込みファイル長）。1GiB に対して 256MiB を詰める前提では余裕で吸収できたが、上限を 1 本に
戻すと「データ節ちょうど上限 + ヘッダ」が hub で落ちる。

## Decision

### 1. 読み手契約: piece（先頭次元の行範囲で割ったテンソル）

- **キー**: `<親名>#NNNNN-of-NNNNN`（5 桁・index 1 始まり・count ≥ 2）。この形に一致し、親名が
  宣言（`initializer.tensor`）に在るときだけ piece と解釈する。karume が書く通常のテンソル名に `#`
  は現れない。
- **形**: piece は親の**先頭次元（行）の連続範囲**。dtype は親と同一、shape は
  `[rows_i, *親.shape[1:]]`（`rows_i ≥ 1`）。piece 1..n は行 `[0, 親.shape[0])` を順に隙間なく覆う。
  各 shard は引き続き独立に整合な safetensors で、piece はその中の普通のテンソル（ADR 0063 の並び規約
  もそのまま掛かる）。
- **配置**: piece は**連続する shard に 1 本ずつ**、index は shard 順に増える。piece 1 の shard には
  前のテンソルが、piece n の shard には後のテンソルが同居してよい。同じ shard に同じ親の piece 2 本・
  index の飛び・逆行・count の食い違い・「丸ごと」と piece の混在は違反。
- **scale**: companion scale（`storage.scale`）は割らず、**piece 1 と同じ shard**に置く（ADR 0070
  決定 1 の co-shard の piece 版）。
- **整列**: 末尾以外の piece はバイト長が 4 の倍数 MUST（ランタイムが `queue.writeBuffer` で
  オフセット書きするため）。末尾 piece は任意長で、ランタイムの末尾詰め物（f16 / i8）が整列する。
- **完全性**: 途中までしか来なかった piece 列は読了時に欠けとして列挙する。

### 2. 読み手契約: 上限は `SHARD_BYTE_LIMIT` = 256MiB 1 本・**ファイル長**で測る

- 全 shard の**ファイル長**（ヘッダ込み）≤ 256MiB。hub は manifest の `size`、exporter の verify は
  実ファイル長で見る — 両側が同じ量を見る。
- 定数は 1 本（`karume.shards.SHARD_BYTE_LIMIT` = hub `MAX_SHARD_BYTES` = 2^28）。ADR 0081 追記の
  `SHARD_TARGET_BYTES` と実効目標の式は**廃止**。「上限超えの単位」は piece で割れるので、上限から
  派生する制約は「**1 行**（先頭次元 1 つぶんのバイト列）が容量を超えるテンソルは配布できない」だけに
  なる（実資産の行は最大数 KB）。
- 根拠: ロード時ホスト RAM ピーク = 定数 + 最大 shard 1 本（ADR 0070 追記 2026-09-02・器の使い回し）
  なので、上限がそのままピークの上乗せ分。256MiB は「ファイル数・リクエスト数の増加が hub の 4 並列
  取得と読み手上限 1024 本の内側」の下限側（ADR 0081 追記の裁定を引き継ぐ）。

### 3. 書き手ポリシー（`karume.shards`）

- 詰める容量 = `SHARD_DATA_CAPACITY` = 上限 − ヘッダ余裕 `SHARD_HEADER_ALLOWANCE`（1MiB）。
  weight shard のヘッダはテンソル 1 本あたり 100〜150 バイトで、1MiB は 7,000 本ぶん（実資産の
  最大は 1 コンポーネント全体で約 1,500 本）。余裕を超える形は verify のファイル長門が fail loudly で
  受ける — 書き手側で黙って縮めない（ヘッダ長を決めるには所属が要り、所属を決めるにはヘッダ長が
  要る循環を、固定余裕 + 読み返し検査で切る）。
- 容量を超える単位（テンソル、または weight + scale の対）の重みだけを **1MiB の行ブロック**
  （`SPLIT_BLOCK_BYTES`・4 バイト整列になる行数へ切り上げ）に砕き、既存の packer（最小本数 → 均し —
  ADR 0081 決定 4）へ通常の単位として流す。同じ shard に落ちた連続ブロックを 1 piece に畳む。
  容量以下の単位は従来どおり割らない（小さいテンソルが shard 境界で piece になる形を作らない）。
- **鎖の引き寄せ**: 対の引き寄せは「scale + 全ブロック」を連続した単位列として、対の片方に最初に
  到達した位置へ置く。scale だけを引き寄せる現行規則をブロックに素直に適用すると、F32 の scale が
  並びの先頭群に居るためブロック 1 だけが前へ移り、残りが末尾群に取り残されて piece が非連続の shard
  に散る（スパイクで実測）。
- 決定的（同入力 → 同分割 → 同バイト）。分割の判定と詰めは実データを読まずに宣言だけで行う。

### 4. ランタイム（読み手の消費）

- piece は届いた順に**親 1 本ぶんの GPU バッファ**へ行オフセット × 行バイト長の位置に
  `queue.writeBuffer` する。バッファ確保・常駐簿記・scale のアップロードは piece 1 で、末尾詰め物は
  最後の piece でだけ。GPU 上の配置・カーネル・診断の意味は変わらず、出力は分割前とビット同一。
- CPU 展開席（圧縮格納だが消費が重みスロット以外で、f32 へ展開して上げる席）も piece 単位で展開する
  （piece の shape と、先頭次元で切った scale を渡す — 展開関数は無改変）。ホストで再結合しない
  （再結合すると生バイト全体がもう 1 本ピークに乗る）。
- shard 単位の errorScope / フェンス / 参照を手放す契約（ADR 0070 決定 3）は不変。

### 5. `requiredLimits` の導出

piece 化した shard ヘッダの最大テンソル長は親の全体長より小さいので、`karume.limits.max_tensor_payload`
は piece を親名で**合算**して需要にする（1 テンソル = 1 GPU バッファ = 1 binding は不変）。

## Consequences

- 実資産で変わるのは上限超えの 4 コンポーネント（gemma4 model / anima text_encoder / irodori v4・v4.1
  backbone）。gemma4 は 6 → 7 本（最大 385 → 254MiB）、anima text_encoder は 6 本のまま（297 →
  233MiB）、irodori backbone は 7 → 6 本（300 → 252MiB）。他は既に容量以下で不変（verify の
  ファイル長門で確認する）。HF 再アップロードはリリース時（既定方針）。
- 旧ランタイムは piece を「余剰 + 不足」として fail loudly で拒否する。未リリースなので互換層は
  作らない。
- 配布形を鍵で読み返す recipes 側の読み手（irodori の参照パイプライン・vowel_detector の往復検査）は
  piece を親へ畳む必要がある。
- 「1 行が容量を超えるテンソル」は配布できない（limitations に記載）。
- ADR 0081 §5 の扉（cut 位置の選好）は据え置き。層割り・MoE 割りは引き続き書き手ポリシーの拡張で
  足せる。
