# 0073: models の既定ソース pin — 公開時点のコミット SHA に固定する

- Status: accepted（2026-08-20 — ユーザー裁定「Pin を実装してください」。実装は HF
  アップロード直後の段 — SHA が生まれてから焼く。手順の正本 =
  [release-runbook.md](../release-runbook.md) §3）
- Date: 2026-08-20
- 関連: ADR [0038](0038-manifest-v1.md)（revision 解決 — 「SHA を渡すと解決要求が発生しない」
  設計を既定経路に載せる）/ [0041](0041-manifest-v2.md)（リポ名裁定 `karume-` prefix）/
  [0071](0071-manifest-v3-shards.md)（format 断絶を pin が吸収する関係）

## Context

models パッケージはリポ名も revision も持たず、hub の既定が `revision ?? "main"`
（fetch.ts）。この形だと、配布リポ側の manifest / 資産が更新された瞬間に**公開済みの
models パッケージが読む内容が黙って変わる**（quant 席の増減・defaultQuant 変更・format
版上げで壊れる/変質する）。ブランチ・タグは HF 側で動かせるため、機械的に不変なのは
コミット SHA だけ。SHA pin + manifest 由来の sha256 検証で「公開済みパッケージが読む
バイト列」がネットワーク側の善意抜きに固定される。副次利得として、SHA 指定は revision
解決リクエスト自体が消えるので**既定経路が完全キャッシュ時にオフライン起動できる**。

## Decision

### 1. 公式リポを持つファミリに既定ソース定数を焼く

各ファミリ config に `<FAMILY>_DEFAULT_SOURCE = { repo, revision: "<40hex>" } as const
satisfies HubRepoRef` を置き、サブパス export で公開する。対象 = 公開リポが実在する
ファミリのみ（sbv2 → `hdae/karume-sbv2-jvnv`（既定）・irodori →
`hdae/karume-irodori-v4-small`・anima → `hdae/karume-anima-turbo`）。公開リポの無い
ファミリ（birefnet 等）は据え置き（ref 必須のまま）。fn の pin 定数は**作らない** —
`karume-sbv2-fn` の公開自体が保留のため（2026-08-20 再裁定・backlog parked。公開されたら
`SBV2_FN_SOURCE` として追加）。これは **models が公式既定リポを知る初の結び付き**（従来は完全リポ非依存）—
利用者向けの既定体験（`fromPretrained()` だけで動く）を優先する裁定。

### 2. `fromPretrained` の ref を optional 化（pin のあるファミリのみ）

省略時は pin 済み既定ソースを使う。追従挙動が欲しい利用者は
`fromPretrained({ ...SBV2_DEFAULT_SOURCE, revision: "main" })` で今日と同じ動きを明示的に
選べる（解除口は `HubRepoRef.revision` として型に既にある）。文字列 ref の意味
（`{ repo }` = main 追従）は変えない。

### 3. pin の更新は手書き + 手順書ゲート（stamp 自動化は不採用）

公開のたびに release-runbook §3 の手順（SHA 取得 → 定数記入 → 疎通 → verify → コミット）で
更新する。生成器 + 照合タスクによる自動 stamp は公開回数がまだ少ない時点では過剰機械
（Simplicity first）— 手作業ミスが実際に出たら再検討。タグ運用（HF の git tag を版として
使う）は不採用 — タグは付け替え可能で、不変性が運用規律依存になる。

## Consequences

- リリース順序が非可換になる: **HF アップロード → pin 焼き込み → JSR publish**
  （runbook §0）。
- 資産側の修正は models のパッチ公開までユーザーに届かない（安定優先の意識的
  トレードオフ）。逆向きの事故（公開済みパッケージが黙って壊れる）は回復不能側なので、
  回復可能側に倒す。
- hub の既定 `revision ?? "main"` 自体は据え置き（hub 単体利用の意味論は不変）。

## 追記

- 2026-08-24: **非既定の公開リポにも同じ pin を課す**（決定 1 の対象拡張）。決定 1 は
  `<FAMILY>_DEFAULT_SOURCE`（= `fromPretrained` の既定席）だけを定義していたが、波 L で
  anima の素版 3 モデルが同居する `hdae/karume-anima` を**既定ではない席**として公開し、
  `ANIMA_BASE_SOURCE` を置いた（1 リポ = 3 モデルなのでリポ参照だけでは 1 本に決まらず、
  利用者が `fromPretrained(ANIMA_BASE_SOURCE, { model })` と綴る）。pin の MUST は既定席と
  同じ — **models が repo を綴る定数はすべて commit SHA で固定する**。理由も同じで、
  「既定ではないから追従でよい」は成立しない（公開済みパッケージが読むバイト列が
  ネットワーク側の都合で黙って変わる事故は、既定かどうかと無関係に回復不能側）。決定 3 の
  更新手順（手書き + 手順書ゲート）は種別を問わず適用し、release-runbook §3 に非既定席の
  チェック行がある。命名は `<FAMILY>_<席名>_SOURCE`。
