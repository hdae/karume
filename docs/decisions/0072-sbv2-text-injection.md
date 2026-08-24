# 0072: SBV2 テキスト層の外部注入席 — overlay 辞書と given_tone

- Status: accepted（2026-08-20 — ユーザー裁定「両方でお願いします」。利用実装側の
  フィードバックで細部を再調整する前提の初版）
- Date: 2026-08-20
- 関連: ADR [0008](0008-public-api.md)（薄い公開面 — 本 ADR は追記後の実態で 1 メソッド +
  4 型 + 1 エラークラスを追加）/ [0039](0039-sbv2-distribution.md)（決定 6 — 辞書取得の
  暫定形・yomi 依存の緊張）
- 実装: `packages/models/src/sbv2/`（pipeline.ts / errors.ts / text/analyze.ts /
  text/prosody.ts / text/phone-tone.ts）

## Context

SBV2 の合成は読み・アクセントを利用者から一切修正できなかった。`@hdae/yomi` の
`analyzeWithWords(dict, text, overlay?)` は修正辞書オーバーレイを受けられるのに、karume は
2 引数で呼んでいて貫通していない。トーン列は g2p 結果から算出され `front` グラフの `tone`
入力にだけ流れる（BERT はトーンを受けない）ため、**長さを保ったトーン差し替えは既存の
整合性検査（`sum(word2ph) === P` / `inputIds.length === word2ph.length`）を壊さない**。
逆に音素列の直接上書きは word2ph の不変条件を壊す。

## Decision

### 1. 修正は 2 軸を別々の正しい席で受ける

- **読み・アクセント型の語彙単位修正 = overlay 辞書**: `Sbv2PipelineOptions.overlay?:
  readonly OverlayEntry[]` と `Sbv2GenerateRequest.overlay?: readonly OverlayEntry[]`。
  request 側があればその 1 回は request 側を**そのまま使う**（合成しない）。検証は yomi 側の
  fail-loudly（surface 正規形・モーラ分割・accentType 範囲）に委ねる。
- **発話単位のアクセント上書き = given_tone**: `Sbv2GenerateRequest.givenTone?:
  readonly number[]`。値域は **0/1 の生値のみ**・長さは解析の phones（add_blank 前・両端
  PAD 込み）と同長 MUST。不一致・値域外は期待/実際を添えて throw。適用はトーン算出後・
  モデル ID 化前の差し替え 1 点。
- **音素列の直接上書きは席にしない**（word2ph 不変条件を壊す）。読みの変更は必ず overlay
  経由で「解析からやり直す」。

### 2. 下書き API `analyzeProsody`

`analyzeProsody(text, { overlay? }) → Promise<Sbv2ProsodyDraft>`、
`Sbv2ProsodyDraft = { phones, tones }`（add_blank 前・PAD 込み・そのまま `givenTone` に
渡せる形）。GPU 不要・決定的。公開する中間表現は**この 2 欄のみ** — word2ph / bertText 等の
内部契約は公開面に固定しない（ADR 0008）。

### 3. yomi 結合は素通しに留める（将来席）

yomi 型は `import type` の素通しで公開し、変換層・抽象は作らない。**将来、SBV2 の入力を
「yomi の解析結果だけ受ける」形にして yomi を models の依存から外す方向をユーザーが表明
（2026-08-20・breaking・時期未定 — backlog later）**。本 ADR の席はその際に overlay が
呼び手側へ移る前提で、結合を最小にしておく。

## Consequences

- 追加は全て optional で非 breaking — 初版は 0.4.0（minor）に同乗し、**追記ぶん
  （`prosody` 席・`Sbv2InputError` ほか）は 0.4.1（patch）で出荷**した。
- 「効いたか」を見る口は `analyzeProsody`（+ 既存 dump 経路）が兼ねる。専用の診断欄は
  設けない。
- `givenTone` の要求長は **overlay に依存する**（overlay が読みを変えれば音素数も変わる）—
  下書きと合成は同じ text・同じ overlay で対にする。ずれは長さ検査が拾う（沈黙誤値なし）。
  → **この 1 文は誤り（2026-08-21 の追記①で撤回）**。長さ検査だけでは梱包規則のズレを拾えず、
  「長さは合うのに音だけ崩れる」が素通りする。`givenTone` は音素単位の低レベル席として、
  この沈黙誤値のリスクを承知で使う席である（構造で往復したいなら `prosody` — 決定 4）。
- examples / CLI への入口追加は席のみ（実需が言われたら）。dump 経路（torch 突合）には
  givenTone を入れない — 参照側と食い違うため。

## 追記（2026-08-21 — 利用実装のフィードバックによる再調整）

> 出荷: **0.4.1**（追加のみ・配布形は `karume/3` のまま）。初版（決定 1〜3）は 0.4.0。

初版（決定 2）の「公開する中間表現は `{ phones, tones }` の 2 欄だけ」を**改める**。VOICEVOX
ENGINE 互換サーバー（`@karume/models` の利用実装）から 0.4.0 の席を実測したフィードバックが
来て、席としては通るが**梱包規則が外部へ漏れる**ことが分かった。

### 何が問題だったか

呼び手が持つのは「アクセント句 × モーラ × 核」で、`givenTone` に渡すには両端 PAD・モーラ →
音素の展開数・記号は tone 0・句ごとに独立に立ち上がる、という **`toSbv2PhoneTone` の梱包規則を
外部で再実装する**しかない。壊れ方が悪く、①再実装が上流とズレても長さ検査は通る（長さは合う
のに音だけ崩れる）②フラット配列 → 句の割り戻しは**原理的に不可能**（`leadingPunctuations` の
個数も句境界も `{ phones, tones }` からは復元できない）。

### 決定 4: 下書きは句 / モーラ構造で返し、同じ形で受ける

`Sbv2Prosody`（`text/prosody.ts`）を karume 所有の型として置き、`analyzeProsody` は
`{ prosody, phones, tones }` を返す。`generate` は `Sbv2GenerateRequest.prosody` で受ける。

- **編集して戻す面は `prosody` 部分木だけ**。`phones` / `tones` は派生（確認用）で、同階層に
  置いて丸ごと戻す形にはしない — 核を直して戻す往復で古い `tones` が同梱され、受け側が
  「無視する（編集を黙って捨てる）」か「落とす（正当な往復が通らない）」の二択になる。
- **yomi の `FrontendResult` を素通ししない**（決定 3 の例外）。SBV2 は `normalizedText` /
  `pauseAfter` / `Mora.devoiced` を読まないので、素通しは「渡せるのに効かない欄」を公開面に
  作る。`OverlayEntry` の素通しは維持（あちらは検証の正本が yomi 側にある）。将来の yomi 依存
  分離（backlog later）でも入力型は karume 所有である必要がある。
- **門は内容一致**: `prosody` から組んだ音素列が解析の音素列と**位置ごとに**一致しなければ
  落とす。長さ検査では素通りする「梱包規則のズレ」「モーラの読み替え」「text の取り違え」が
  ここで全部止まる。音素列は常に解析由来を採る（門を通れば同一）。
  NOTE: 門は音素列だけを見るので、**句 / モーラの境界の組み替えは通る**（VOICEVOX 互換の
  アクセント句編集がここに入る — 意図的に受ける）。副作用として 1 モーラの子音と母音に別トーンが
  載る列を作れるが、不変条件は破れず影響は出力音のみ（limitations の SBV2 節）。
- **核は正規形で出す**: yomi の `moraTones` は範囲外核を尾高相当へ黙ってクランプするので、
  下書きへは**上端を `moras.length` へ**丸めて載せる（丸めてもトーン列は同じ）。載せないと
  「解析どおりの下書きを戻したのに範囲検査で落ちる」往復不能が作れてしまう。下端を丸めないのは
  負の核が実測で出ていないため — 出た場合は受理側の範囲検査（`0..moras.length`）が落とす。
- `givenTone` は音素単位の低レベル席として残す。**同時指定は落とす**（優先規則を作らない）。

### 決定 5: 解決済み `OverlayDictionary` を受ける

`overlay` の型を `readonly OverlayEntry[] | OverlayDictionary` に広げる（既定席・要求席・
`analyzeProsody` の 3 か所）。実行中に増減するユーザー辞書は「呼び手が作り直して毎回渡す」形に
なるが、初版は要求側指定のたびに `new OverlayDictionary(...)` していた（数千語で合成ごとに
効く）。

MUST: 解決済みで渡す場合、**同じ辞書に対して解決したものであること**。`OverlayDictionary` は
構築元の辞書を保持しないので karume 側では検証できない（別辞書で解決したものを渡すと文脈 ID が
ずれたまま合成が通る）。`Sbv2PipelineOptions.dictionary` で辞書を共有するのが確実。

却下: エントリ配列を弱参照キーにしたキャッシュ。API は増えないが、配列を in-place で変更されると
黙って古い辞書を使い続ける。

### 決定 6: 入力起因の失敗に型を付ける（`Sbv2InputError`）

呼び手が渡した要求が受理できないもの（text 空 / 未知の話者・スタイル / 非有限ノブ /
`lengthScale <= 0` / `givenTone` の長さ・値域 / `prosody` の門 / 運用上限超過）は
`Sbv2InputError` で飛ばす。内部不変条件の破れ（`sum(word2ph)` 不一致・tile 走査の破れ）は素の
`Error` のまま — HTTP サーバーが 400 と 500 を分けられることが目的なので、この線引き自体が仕様。
サブクラスは作らない（hub が 5 種に割っているのは分岐先が実際に違うから）。

### 決定 7: `analyzeProsody` は直列化鎖に載せない

初版は「辞書取得の 1 度きりは鎖が担保する」ため鎖に載せていたが、GPU を張らない解析が進行中の
合成（実測 0.5〜0.7s）を待つ。`ensureDictionary` が**値ではなく Promise** を持てば、await 中に
並行で入った呼び出しも同じ取得へ合流するので、鎖なしで 1 度きりが成り立つ。あわせて**失敗した
Promise は欄から捨てる**（持ち続けると 1 度のネットワーク失敗が以後の全合成へ波及し続ける —
値キャッシュには無かった失敗モード）。

`overlayFor` の既定席キャッシュは鎖の外から並行に埋まり得るが、解決は (辞書, entries) の純関数
なので二重構築が起きるだけで値は同値。

### 参照実装の調査（2026-08-21・この再調整の前提）

`clean_text_with_given_phone_tone` / `adjust_word2ph`（upstream）と AivisSpeech Engine +
その pin する fork を読んだ。要点:

- upstream には `given_phone` を**実際に組み立てて渡す呼び出し元が無い**（Gradio もエディタ用
  サーバーも `given_tone` だけ。CHANGELOG も「ライブラリとしてのみ」と限定）。実運用の編集は
  「音素数を変えないトーン編集」に閉じている。
- AivisSpeech Engine は `text` + `given_phone` + `given_tone` の 3 点セットを常に渡す。`text` は
  音素の供給源ではなく **BERT 特徴と word2ph の供給源**で、原文が無いときはモーラ列をひらがな化
  した文字列で代用する（自ら「不自然なイントネーションになる」と宣言）。karume が `text` を必須の
  第一級入力に据えているのは同じ構造。
- 音素数が変わる編集は upstream では `adjust_word2ph`（LCS 再配分）→ 調整しきれなければ
  `InvalidPhoneError`。**AivisSpeech が pin する fork はその失敗時を「均等増減で無理やり辻褄を
  合わせる」へ置き換えている**（upstream には無いコード）。後者は karume の横断不変条件
  「未対応・想定外は fail loudly（黙って近似しない）」と正面から衝突する。
- 疑問形の上げは参照実装にも AivisSpeech にも機構が無い（`enable_interrogative_upspeak` は
  「常に無視される」と docstring に明記）。`?` が音素として入るだけ — limitations に記載。

### 決定 8: 音素数が変わる編集は受け付けない（`adjust_word2ph` は移植しない）

上の調査を踏まえたユーザー裁定（2026-08-21・条件つき — 「読み編集が overlay で回るなら」）。
`prosody` の門は内容一致で落とし、`Sbv2InputError` を返す。読みの変更は overlay で解析から
やり直す。

理由:

1. fork の「無理やり辻褄合わせ」は横断不変条件（黙って近似しない）と衝突する。上流セマンティクス
   （LCS + 残差は例外）を採っても、**BERT 特徴は元テキスト由来のまま**なので「音素だけ差し替わって
   韻律の情報源は旧テキスト」という不整合は消えない。
2. karume には overlay がある。読みを overlay で変えれば **BERT 入力テキストと word2ph が正しく
   作り直される** — `adjust_word2ph` より品質が上の経路が既に開いている。あちらは「エディタが
   再解析できない」事情から来る互換ハック。
3. upstream の実運用も「音素数を変えないトーン編集」に閉じている（`given_phone` の呼び出し元が
   無い）。

カバー範囲（この裁定が成立する条件そのもの）:

- ✅ ユーザー辞書による読み・アクセント修正 → `overlay`（語単位・解析からやり直すので BERT も整合）
- ✅ アクセント核の編集 → `prosody`
- ✅ 音素単位のトーン直接指定 → `givenTone`
- ❌ 語境界に一致しない読みの差し替え（句内の一部モーラだけを別の読みへ）→ `Sbv2InputError`

復活条件つきで backlog parked（`adjust_word2ph` 移植 — 参照は**上流**セマンティクス）。
