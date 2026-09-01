# 0088: Civitai 取り込みは AIR 指定の recipe コマンド + エージェント事前確認

- Status: accepted（2026-09-01 — ユーザー裁定「1. OK / 2. モデル名+バージョンで正規化」）
- Date: 2026-09-01
- 関連: ADR [0077](0077-model-version-naming.md)（モデル名に上流バージョン —
  本 ADR が正規化規則を追加）/ [0087](0087-anima-official-extra-repos.md)（extra リポ =
  第三者 fine-tune の置き場）/ [0073](0073-models-source-pin.md)（既定ソースの pin）

## Context

extra リポ（ADR 0087）へ第三者 fine-tune を足す作業は、これまで全段が手作業だった:
ブラウザで Civitai から checkpoint を落とし、`inputs/anima/` へ手置きし、出所は
コミットメッセージと `card.py` の手書きにだけ残る。sha256 の突合も命名も人任せで、
「どの版をいつ・どの許諾表示のもとで取ったか」が機械可読に残らない。

一次確認で確定した事実（2026-09-01 実測 — wai `2544636@2983680`）:

- メタ API（`/api/v1/model-versions/{id}` / `/api/v1/models/{id}`）は無認証で読め、
  **model-versions 応答が AIR（`urn:air:anima:checkpoint:civitai:2544636@2983680`）を
  サーバ発行値として返す**。sha256・ファイル型（Model / Text Encoder / VAE）・許諾 4 欄
  （allowNoCredit / allowCommercialUse / allowDerivatives / allowDifferentLicense）・
  usageControl・description（HTML）も揃う。
- **ファイル DL だけは 401** — API トークンが要る。
- ライセンスの実態は API 欄だけでは決まらない — 本文（description）に独自条件が
  書かれることがあり、欄と本文が食い違いうる。

## Decision

**取り込みは `anima.civitai` コマンド（AIR / URL 指定）で行い、責務は
「取得 + 検証 + 記録」まで**とする。diffusers 化は既存 `anima.single_file`、export と
distribution / card への登録は従来の手順のまま（コマンドは登録を書き換えない）。

1. **入口は AIR / URL の 2 形**。版未指定の URL は版一覧の案内表示のみ（取得しない）。
   AIR はサーバ発行値を正とし、自前組み立てと食い違えばサーバ値を採って警告する。
2. **モデル名は機械正規化 `anima-<モデル名>-<版名>`**（2026-09-01 裁定 — ADR 0077 の
   「表記を正規化しない」を改訂）。小文字化・非 `[a-z0-9._-]` は `-`・丸括弧の付記は
   除去・モデル名から `anima` トークンを除去。既存実績（`anima-wai-v1.0` /
   `anima-copycat-20260610`）をそのままルール化したもので、**上流の逐語表記は
   `civitai.json` に残る**ため出所ページとの突き合わせ可能性は失われない。`--name` で
   上書き可（既定を人が裁定で覆す扉 — 0077 と同じく、既定名の自動が裁定を代替しない）。
3. **provenance は連鎖させる**: 取得先 `inputs/anima/civitai-<versionId>/` に重みと
   `civitai.json`（AIR・版名逐語・sha256・許諾欄・description・取得日時 — **機械専有**、
   人は追記しない）を並べ、`single_file` が同居する `civitai.json` を
   `source_provenance.json` へ読み継ぐ。dist まで出所が切れない。
4. **sha256 は API 記載値と突合してから完成品にする**（`.part` → 一致で rename）。
   不一致は fail loudly。既存ファイルが sha 一致なら取得をスキップ（冪等）。
5. **ライセンスはエージェント事前確認形**（2026-09-01 裁定）: コマンドは許諾欄と本文を
   記録するだけで、**判定も確認フラグの強制もしない**。取り込み作業を行うエージェントが
   取得前後に API 欄と本文を突合し、食い違い・グレーは同ディレクトリの
   `license-review.md`（人・エージェント記入 — コマンドは触らない）に記録して
   ユーザーへ報告する。Anima 派生は base ライセンス（v1.2）引き継ぎが前提で、上流が
   別ライセンスを名乗る場合は裁定に上げる。
6. **DL トークンは env `CIVITAI_API_TOKEN`、付与は `?token=` クエリ**（Authorization
   ヘッダは S3 への 307 リダイレクトで署名 URL と衝突しうる）。表示にはマスクを掛ける。

## Consequences

- `inputs/anima/civitai-<versionId>/` が取り込みの正規の置き場になる（assets-layout に
  追記）。ディレクトリ名は versionId で機械的に一意 — 人向けの名前は `civitai.json` の
  `derived_name` が持つ。
- 機械記録（civitai.json）と人の判断記録（license-review.md）をファイルで分けたので、
  再取得（civitai.json の上書き）が確認記録を消さない。
- 登録（`distribution.py` / `card.py`）は引き続き手書き — 取り込みの自動化は
  「登録の提案」まで進めない（新モデルの追加は毎回それ自体が裁定であるため）。
- 対象は primary（Model 型）ファイルのみ。非 primary（Text Encoder / VAE）は列挙表示に
  留める — checkpoint が持たない部分は base 共有で足りる（`single_file` の前提と同じ）。
- Civitai 以外のソース（HF 直・手置き）は従来どおり — `civitai.json` が無いだけで
  `single_file` は従来形の provenance を書く。
