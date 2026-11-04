# GLOSSARY.ja — 対話で使う日本語の用語（草案 2026-10-15・利用者のフィードバック反映済み）

> 性格: 草案。英語化（0.15 の組み直し）後のコード・docs の用語と、日本語の対話で使う言い方の対応表。
> 意味の正本は各 ADR / docs（ここは索引）。「やめる言い方」は旧い家言葉で、対話・docs・コメントのいずれでも使わない。
> ◆ = 利用者の指示で確定。△ = 提案（要確認）。

## 用語の決め方（利用者の方針 2026-10-15）

- **意味から決める。英語の直訳はしない。** 英語の用語を見て訳語を作るのではなく、その概念が何かを
  言い表す日本語を選ぶ（例: series → 「シリーズ」ではなく「中間成果物」、spike → 「試作」、
  kill criterion → 「打ち切りの基準」）。
- 英語の用語は業界の標準語を使う。日本語は「カタカナで通じる語」か「ふつうの日本語」にする。
- 演算（numerics）の話と重み（weights / quants）の話で、同じ語（実用・参照・忠実）を使い回さない。
- 1 つの概念に 1 つの語。同じターンの中で揺らさない。
- 作業の単位は **計画 → フェーズ → ステップ** の 3 段で呼ぶ。

## 演算（numerics）と重み（weights）

| 英語（コード・docs） | 日本語（対話） | 意味 | やめる言い方 | 正本 |
| --- | --- | --- | --- | --- |
| numerics profile ◆ | 数値プロファイル | 見送り（2026-10-18・ADR 0126 は deferred — 0.15.0 には無い）。`SessionOptions` の実行形の束。`practical` / `reference` の 2 値 | 層（実用層・参照層）・既定経路 | ADR 0126（deferred・0058 / 0110 の改定は発効せず） |
| practical profile ◆ | 実用プロファイル | 見送り（同上）。実装されれば既定で、形と adapter の能力で E / C / R の変種を選ぶ。近似（Q）は含まない。今の既定の実用側は manifest の実用プリセット（`session` の束） | 実用層・実用経路 | 同上 |
| reference profile ◆ | 参照プロファイル | 見送り（同上）。opt-in の knob の値として予定していた「省略値の束（自動選択なし）」。今はその状態が runtime の既定そのもの = 参照経路（次の行） | 参照層・既定経路 | 同上 |
| reference path ◆ | 参照経路 | runtime の省略値の経路（数値ノブを全て省略した Session）。sha 行の参照行・オラクル・上流一致の根。2-8 の裁定（2026-10-18・1-1）で「やめる言い方」から戻した | 参照層・既定経路・参照席 | ADR 0058 決定 2・ADR 0110 §1 |
| original weights ◆ | オリジナルの重み（忠実な重み） | Karume 側で量子化していない重み。上流の格納のまま（f32 → f32・bf16 → bf16・上流が f16 なら f16・QAT はその整数）で演算は f32 | 忠実席・参照席・忠実な席 | ADR 0128 決定 4・0131 |
| q4_0 QAT line △ | q4_0 QAT 系 | Gemma 4 の配布 `karume-gemma4-qat`。Google の q4_0 QAT チェックポイント（E2B / E4B / 12B）を、チェックポイント自身が持つ 4-bit の格子のまま丸め直さずに取り込み（格子の取り込み）、各モデルに QAT の MTP drafter が付く。契約は通常の `gemma4/1` | — | ADR 0097 追記 10 |
| mobile QAT line △ | モバイル QAT 系 | Gemma 4 の配布 `karume-gemma4-qat-mobile`（旧 `karume-gemma4-qat`）。公式モバイル版の固定 INT2 / INT4 / INT8 と SRQ を保つ E2B / E4B。契約は `gemma4-qat/1` | 「QAT 版」だけで呼ぶ（2 系統あるので取り違える） | ADR 0097 追記 10 |
| quantized weights ◆ | 量子化された重み | i8 / i4 / i2 の格納型。`quant` の明示指定で選ぶ | 実用席の重み | 同上 |
| storage dtype | 格納型 | 容器に書いた重みの形（f32 / f16 / i8 / i4 / i2） | — | glossary.md |
| activation quantization | 活性量子化（a8） | 活性を int8 に量子化して計算する実行形。品質に効くのでプリセットの中身（明示） | — | ADR 0025 / 0030 |
| oracle | オラクル | 正しさの物差しになる自前の参照実装（TS 鏡像・f64 参照） | — | ADR 0005 / 0110 |
| tolerance band △ | 許容帯 | オラクルとの差の上限（`atol + rtol·\|ref\|`） | 帯 | op-tolerance.ts |
| contract class (E / C / R / Q) | 契約クラス | 変種の主張の強さ（鏡像 / 合成 / 加算順 / 近似） | — | ADR 0110 決定 2 |
| variant | 変種 | あるノブの値 1 つに対応するカーネル経路 | 入居者 | ADR 0058 |
| option / knob △ | オプション（ノブ） | `SessionOptions` の個別の設定 | 席（実行の席） | ADR 0058 決定 1 |
| fallback △ | フォールバック | 形や limit が適格でないとき参照の変種へ落ちること。診断のパイプラインキーに必ず出る | 縮退・沈黙縮退 | ADR 0058 決定 3 |
| auto-selection | 自動選択 | 見送り（2026-10-18・ADR 0126 は deferred）。実用プロファイルが形 / adapter で変種を選ぶこと。選択は確認可能で、ノブで上書きできる ◆ | auto | ADR 0126（deferred） |
| packed | packed | 1 語 / 1 byte に複数の整数要素を詰めた格納（`i4` は 1 byte 2 要素・`i2` は 4 要素） | — | ADR 0069 決定 2 |
| group_size | group_size | `i4` の scale 1 個が担当する K 方向の連続要素数（exporter 側の名前・容器では `groupSize`） | グループ量子化（「group 量子化」で統一する） | ADR 0069 決定 2 |
| scale companion | scale companion | 量子化重みの復元係数を持つ別 block（束縛表の `encoding.scale`） | — | ADR 0108 決定 11 |
| split scale △ | 分割 scale | 1 block に収まらない scale companion を、データの piece と同じ行範囲で piece ごとの block に分けたもの（束縛表の `encoding.scale.pieces`・全部 piece 1 の part に置く）。読み手は 1 本につないで Session 構築へ渡す | — | ADR 0108 追記 15 |
| PLE (Per-Layer Embedding) | PLE | 層ごとに足し込む補助 embedding。グラフの外の表からホストが行を読む | — | ADR 0085 |
| SRQ (Static Range Quantization) | SRQ | 学習時に決めた固定 scale で活性を int8 の格子へ丸めて戻す処理（計算は f32 のまま） | — | ADR 0097 追記 2 |
| w8a8 | w8a8（重み 8bit × 活性 8bit 整数内積） | 重み i8 × 活性 per-token i8 の整数内積の経路（`linearCompute: "a8"`） | — | ADR 0025 |
| channel-major quantiser (per-pixel quantiser of conv3d w8a8) △ | conv3d w8a8 の画素ごとの量子化（チャネル順の形） | conv3d の入力 `x[Cin, P]` を格納の並びのまま 1 dispatch で読み、画素ごとにチャネル方向の amax を取って `quantize_rows` と同じ語と scale を書く量子化（ビット同一・2026-10-26 に転置コピー + `quantize_rows` から置き換え） | — | ADR 0133 追記 2026-10-26（channel-major quantiser の節） |
| w4a8 | w4a8 | 重み i4 × 活性 per-token i8 の整数内積の経路 | — | ADR 0076 |
| integer dot product | 整数内積 | 量子化した整数どうしを int32 累算で掛け合わせる実行形 | — | ADR 0025 / 0076 |
| integer GEMV (`integerGemv`) △ | 整数 GEMV | packed 活性の SRQ コードと重みの整数コードの積をグループごとに厳密な整数で足し、scale は最後に決まった順で 1 回だけ掛ける decode 用 GEMV（opt-in・QAT E2B の計測した形だけ） | — | ADR 0140 |
| byte-plane GEMV (`linearGemvPlanes`) △ | バイトプレーン GEMV | i2 の重みの語をバイトプレーンに分け、`unpack4x8unorm` と厳密な変換で −2..1 の値にして f32 の活性と掛けて足す decode 用 GEMV。参照との違いは足す順と、チャネルの scale を和の後で 1 回掛けることだけ（契約クラス R・opt-in・QAT E2B の lm_head の形だけ） | — | ADR 0141 |
| packed int8 activations | packed 活性 | 固定 SRQ の int8 コード 4 個を u32 1 語に詰めて並列 GEMV へ渡す形 | — | ADR 0105 |
| fake-quant | fake-quant | 重みを量子化格子へ丸めて f32 のまま持つ処理 | — | `tools/exporter/src/karume/quantize.py` |
| quality measurement (`quality.json`) | 品質計測 | export 時に torch CPU で測る量子化誤差の記録 | — | ADR 0113 決定 4・5 |
| preset derivation rule (BiRefNet) | プリセットの導出規則 | checkpoint ごとに上流の格納型からプリセットを導く規則 | 席の導出規則 | ADR 0113 追記 2026-10-11 |
| RTN (Round To Nearest) | RTN | 校正なしの最近接丸め | — | ADR 0019 / 0069 |
| GPTQ | GPTQ | 校正データで丸めを補正する方式（出荷は格子が RTN と同じ版） | — | perf-ledger Q-6 |
| fixed quantization (importing QAT) | 固定量子化 | 上流 checkpoint の整数と scale を再量子化せずに書く経路 | — | ADR 0097 追記 3 |
| grid import (QAT) △ | 格子の取り込み | QAT のチェックポイントが持つ 4-bit の格子（32 要素ごとの整数 [-8, 7] と 1 つの step）を丸め直さずに取り込むこと。q4_0 QAT 系（`karume-gemma4-qat`）の E2B / E4B / 12B の重み、E2B / E4B の PLE（索引 schema 4）、QAT の drafter で使う | 丸め直し | glossary.md・ADR 0128 追記 2026-11-03 |

## 配布と資産

| 英語（コード・docs） | 日本語（対話） | 意味 | やめる言い方 | 正本 |
| --- | --- | --- | --- | --- |
| preset ◆ | プリセット（会話では「quant の選択肢」でもよい） | 配布形が「重みの格納型 + 実行ノブ」を束ねた名前（`i4-fast` など）。manifest の `quants` の項目 | 席・quant 席・実用席・参照席 | ADR 0074 / 0075 |
| default preset ◆ | 既定のプリセット | manifest の `defaultQuant`。モデルごとに保守者が実測（速度 / メモリ / 品質）で選んで記録する実用のプリセット（量子化を含むことがある）。オリジナルの重みのプリセットは努力目標で opt-in | 既定席・忠実席 | ADR 0128（2026-10-17 書き換え） |
| family ◆ | ファミリー | モデルの種類（anima / wan / gemma4 …）。`@karume/models/<family>` | 系列（モデルの意味で） | CLAUDE.md |
| series ◆ | 中間成果物 | exporter が書く中間成果物のディレクトリ（`outputs/series/<name>/`）。配布形の 1 つ手前 | 系列（資産の意味で）・シリーズ | assets-layout |
| test assets | テスト資産 | 配布物に入らないテストの入力と参照（golden と corpus）。private dataset `hdae/karume-test-assets` を pin した版で `outputs/test-assets/` に取得する | テストアセット | ADR 0139 |
| distribution △ | 配布形 | HF に置く形（manifest + `krm`）。`models/` のミラー | — | ADR 0109 / container-v1 |
| container (`krm`) | コンテナ | 配布形の実体ファイル | — | container-v1 |
| export recipe | レシピ | モデル別の export 手順（`tools/export-recipes/`） | — | ADR 0065 |
| rebuild (a distribution) △ | 作り直す | 配布形を exporter から生成し直すこと | 焼き直す・焼く | release-runbook |
| dtype label | dtype ラベル | manifest の `quants[].weights` に書く綴り（格納型と同じ語彙） | — | ADR 0074 決定 2 |
| practical preset | 実用プリセット | `session` が空でないプリセット（実用の経路を束ねる）。自機 A/B で参照側のプリセットと比べる側 | 実用層・実用席 | ADR 0110 決定 1 |
| `krm` | krm | 重み block を持つコンテナ（model） | — | ADR 0108 決定 1 |
| `krg` | krg | 重み block を持たないコンテナ（graph）。`krm` からバイトコピーで抜ける | — | ADR 0108 決定 1 / 4 |
| descriptor (graph descriptor / model descriptor) | descriptor | コンテナ先頭に置く 2 文書の JSON（グラフ記述 / モデル記述） | ヘッダ（先頭 24 B 固定長の「ヘッダ」と混ぜない） | ADR 0108 決定 2 |
| const region | const 領域 | 重み非依存の定数を置く区画（グラフ記述が所有） | — | ADR 0108 決定 3 |
| block | block | 取得・検証・解放の単位（32 MiB 以下） | チャンク | ADR 0108 決定 5 / 7 |
| part | part | 配信粒度（ファイルの分割の単位） | — | ADR 0108 決定 6 |
| binding table | 束縛表（初出で「binding table」を併記） | `(グラフ名, initializer 名)` → 重みの供給 の対応表 | — | ADR 0108 決定 1 |
| supply form / shared declaration | 供給形 / shared 宣言 | 束縛表の値の唯一の形と、別 Session からの借用の IR 側の宣言 | — | container-v1 §5 |
| codec | codec（展開経路の名前で呼ぶときは「格納型」） | 量子化格納の詰め方・値域・scale の形・復元式をまとめた登録名 | — | ADR 0108 決定 12 / 13 |
| codec ledger | codec 台帳 | codec の格納形式をリポ内の不変データとして 1 か所に置いたもの | — | ADR 0108 決定 12・追記 12 |
| `encoding` | encoding | 束縛表の各エントリが持つ供給記述 | — | container-v1 §6 |
| `packing` | packing | 1 packing block の要素数・バイト数・先頭の整列（bit 数は派生値） | — | ADR 0108 決定 10 |
| layout (decode path) | 展開経路 / layout | codec の登録名から派生するランタイムの復元経路の種別 | — | `packages/runtime/src/format/container/codecs.ts` |
| `rowAxis` | rowAxis | scale の「行」に当たる軸（`0` / `1`） | — | ADR 0108 決定 11 |
| `groupSize` | groupSize（exporter の Python 側の `group_size` と同じ概念） | 1 個の scale が担当する行方向の連続要素数 | — | ADR 0108 決定 11 |
| ternary | 三値 | 値域 `{−1, 0, +1}` の codec（詰め方は `int2-off` と同じ） | 1.58bit | ADR 0108 決定 13 |
| scale-ledger uniqueness | 台帳の単一性 | 1 つの重みに scale 台帳は 1 本で、確定したら引き直さない規律 | — | ADR 0108 追記 9 |
| bit-plane decomposition | ビット面分解 | 奇数 bit（3 / 5 / 6 / 7）のための予約方式 | — | ADR 0108 決定 14 |
| single form / split form | 単一形 / 分割形 | 全 part を 1 ファイルへ連結した形 / part ごとに独立ファイルへ割った形 | — | ADR 0108 決定 3 |
| scan-type / seek-type sources | scan 型 / seek 型・保持枠 | hub の取得面が block を読む 2 形（part を全量読む / block ごとに位置読み） | — | ADR 0109 決定 7 |
| verified / unverified source | 検証済みの取得元 / 未検証の取得元 | 取得元が part のバイト列を既に検証したか（`BlockSource.verified`） | — | ADR 0109 決定 7 |
| merged concurrent fetches (single-flight) | 同時取得の合流（single-flight） | 同じ内容キーへの並行の全量取得をプロセス内で 1 本に束ねる仕組み | — | ADR 0080 追記 2026-10-10 |
| in-memory container | メモリ内容器 | 手元のバイト列を、容器ファイルを書かずに `krm` と同じ経路へ載せる runtime の面 | — | ADR 0108 追記 4 |
| merge layer | 合流層 | グラフ宣言と束縛表を重み取得前に合わせ、格納の規則を 1 か所で掛ける層 | — | container-v1 §13.3 |
| admission | admission | 重みを 1 バイトも取る前に、宣言だけで Session を作れるかを判定すること | — | container-v1 §13.3 |
| sequential block upload (lazy items) | block の逐次転送 | Session 構築が block を 1 本ずつ GPU へ上げて手放す形 | — | container-v1 §11 |
| supply read-ahead (Session build) △ | 供給の先読み | Session 構築が、上げている item の先の block を最大 2 本読んでおくこと。供給の順番全体で数えるので part の境界をまたぎ、今の part のフェンスを待つ間に次の part を読む | — | ADR 0108 追記 14・`READ_AHEAD_BLOCKS`（executor.ts） |
| migration CLI | 移行 CLI | 旧形式を新コンテナへ変換する Python の道具 | — | ADR 0108 決定 18 |
| three hash roles | ハッシュの役割 3 分離 | 期待ハッシュ・派生キャッシュキー・内容ハッシュを同じ欄で兼ねない規律 | — | ADR 0108 決定 8 |
| piece | piece | 上限を超えるテンソルの行範囲分割（束縛表の `pieces[]`） | — | container-v1 §5 |
| shard | shard（旧配布形の話に限る — 現行の話では part と言う） | 旧配布形のファイル分割単位 | — | ADR 0081 / 0090 |
| manifest `karume/5` | manifest v5 / `karume/5` | コンテナ配布形の入口の manifest | — | ADR 0109 |
| cross-repository reference | 越境参照 | manifest の FileRef が別の HF リポの commit を指して部品を読む形 | — | ADR 0038 §7 追記・ADR 0109 決定 3 |
| component replacement (`components`) | 部品の差し替え（`components`） | 別の `karume/5` リポの同じ役割の部品で差し替えること | 部品差し替え席 | ADR 0108 決定 19・ADR 0109 決定 10 |
| PLE index schema 3 | PLE 索引 schema 3 | コンテナの資産に入れた PLE の索引。scale は行ごとに f32 が 1 つ | — | ADR 0085 追記 2026-09-22 |
| PLE index schema 4 △ | PLE 索引 schema 4 | schema 3 に必須の `groupSize` を足し、`groupSize` 個の値ごとに f32 の scale を 1 つ持つ PLE の索引。q4_0 QAT 系の int4 の表（`groupSize` 32）の形 | — | ADR 0085 追記 2026-11-04（schema 4） |
| PLE index schema 5 △ | PLE 索引 schema 5 | bf16 の表の PLE の索引。値はチェックポイントの bf16 のまま（1 値 2 バイト）で、scale の表を持たない。通常の gemma4 の `bf16` プリセットの形 | — | ADR 0085 追記 2026-11-04（schema 5） |

## 検証

| 英語（コード・docs） | 日本語（対話） | 意味 | やめる言い方 | 正本 |
| --- | --- | --- | --- | --- |
| check ◆ | チェック | 自動テストの検証項目 1 つ | 門 | ADR 0005 |
| sha256 regression check △ | sha256 の回帰チェック | 出力の成果物の sha256 を環境ごとの参照値と突き合わせる検査。決定性（同機ビット不変）を見る | sha 門・移植の門 | ADR 0106 |
| reference value / row △ | 参照値（の行） | 環境キーごとに持つ期待 sha256。書くのは `write` / `rewrite` | 行・参照行・実用行・ゴールデン（参照値の意味で — torch 由来の golden と混ぜない） | ADR 0106 |
| artifact ◆ | 成果物 | sha256 を採る出力ファイル（PNG / WAV / f32 safetensors） | 実物 | ADR 0106 追記決定 6 |
| environment key | 環境キー | `<runtime>-<adapter>` の綴り（`deno-…` / `chrome-…` / `node-…`） | デバイス ID | ADR 0106 決定 2 |
| all-skip guard △ | 全 SKIP ガード | 検査が 1 本も走らなかった状態を赤にする検出器（GPU / 資産 / 配布形 / 参照値）。分かりにくいが頻出ではない。良い言い方があれば差し替える | 門番・参照門 | ADR 0005 / 0106 |
| lane | レーン | verify の実行単位（core / ファミリーごと） | シャード・グループ | ADR 0005 追記 |
| golden | golden | 上流（torch）由来の期待値。許容差つき | — | ADR 0005 |
| fake-quant golden △ | fake-quant の golden | torch に活性量子化を掛けて採った期待値 | 鏡像 golden | glossary.md |
| census | census | 変種のパイプラインキーが実際に走った本数の検査（黙って落ちるのを防ぐ） | — | ADR 0058 決定 4 |
| self A/B △ | 自機 A/B | 同じ機で参照経路（省略値）と実用プリセットの出力を比べる検査 | 自機 A/B 門 | ADR 0110 決定 5 |
| difference floor △ | 差の下限 | 自機 A/B で「1 bit 以上違う」こと（束が丸ごと効いていない状態の検出） | 床 | ADR 0110 決定 5 |
| collapse ceiling | 崩壊上限 | 自機 A/B の相対 RMS の上限 | — | ADR 0110 決定 5 |
| fault injection | 故障注入 | チェックが空振りでないことの証明（わざと壊して赤にする） | — | ADR 0110 決定 3 |
| determinism | 決定性 | 同じ機・同じ入力・同じプリセットで出力バイトが同一 | ビット同一（主張としての） | ADR 0110 決定 6 |
| upstream parity | 上流との一致 | 数値または意図の水準で上流と合っていること。上流のバグは写さない | 100% 一致 | ADR 0005 追記 2026-10-11 |
| port △ | 移植 | 上流の処理を Karume に書き写すこと | 写す | — |
| acceptance | 検収 | ステップの完了条件を満たしたことの確認 | — | — |
| decision | 裁定 | 利用者が選んだ結論（ADR に記録） | — | docs/decisions |
| recording mode (`KARUME_REFERENCE`) | 作るモード | 参照値の行を実測から作る指定（`write` / `rewrite`） | — | ADR 0106 決定 3 |
| held row (explicit SKIP per environment key) | held 行 | その機でプロセスごと落ちるファミリーを、環境キーの行で明示 SKIP する表 | — | ADR 0106 追記決定 8 |
| results directory (`outputs/verify/`) | 結果のディレクトリ | 走行ごとの `results.json` と成果物の置き場（追跡外・消して安全） | 結果の席 | ADR 0106 決定 5 |
| measurements | 実測 | 許容差判定が出した差の記録（`results.json` の `cases[].measurements`） | 計測値・メトリクス | ADR 0106 追記決定 2 |
| four-point kernel check | 4 点型 | R / Q の変種のカーネルのチェック（f64 との差・参照経路との A/B・構造的保証・故障注入の記録） | — | ADR 0110 決定 3 |
| full-eligibility check | 完全適格の検査 | 融合 attention の全 dispatch が宣言した変種を通ったかを比で見る検査 | — | ADR 0058 追記 2026-10-10 |
| reference-path rows / practical-preset rows | 参照経路の行 / 実用プリセットの行 | sha の参照値の行の 2 クラス（`session` が空なら参照経路の行） | 参照行・実用行 | ADR 0110 決定 7 |
| acceptance table (`FamilySessionPolicy`) | 受理表 | ファミリーごとに manifest の宣言と明示指定のどのキーを受けるかの網羅表 | — | ADR 0058 追記 2026-09-26 |
| reference partner | 参照側のプリセット | 実用プリセットと同じ重みで `session` が空のプリセット（自機 A/B の相手） | 参照席 | `packages/e2e/tests/helpers/ab-gate.ts` の `referencePartnerOf` |
| relative RMS error (relRms) | 相対 RMS 誤差 | 出力ごとの ‖x − 参照‖₂ / ‖参照‖₂ | — | ADR 0110 決定 5 |
| different sample | 別標本 | 正しい実装どうしでも深い層の出力が同じ分布の別の標本になること、またその標本 | — | ADR 0025 決定 6 |
| perturbation ensemble | 摂動アンサンブル | 埋め込みの出口に微小な乱数を入れて回した別標本の束 | — | ADR 0026 追記 2026-10-07 |
| census table (per bundle) | census 表 | 束ごとに変種の期待本数（ちょうど）を持つ表 | — | `packages/models/tests/helpers/census-table.ts` |
| timing mode (`gpuTiming`) | 計測モード | timestamp-query を要求した device での実行 | — | ADR 0021 決定 1・2 |
| split-pass timing △ | dispatch ごとの計測 | dispatch を 1 つずつ計時つきの pass に分けて測る。kernel ごとの時間が出るが、pass ごとの固定費（約 4 µs）が乗る | — | research 2026-10-27 §3 |
| grouped-pass timing △ | まとめた pass での計測 | 製品の pass を、ある種類の kernel の前後でだけ分けて測る。その種類の時間を直接測れる | — | research 2026-10-27 §3 |
| burst-free composite △ | 遅い区間を除いた合成値 | prefill の chunk の位置ごとに最速のサンプルを足したもの | — | research 2026-10-27 §3 |
| poison values (fault injection) | 毒値 | 書かれずに残った要素を見つけるため、実行前に NaN などで埋める故障注入の手法 | — | ADR 0004・0014 |
| upstream-error band △ | 上流誤差の許容帯 | f64 のオラクルに対し、同じ入力での上流自身の f32 誤差の倍数で決める許容帯（gated_delta_scan では上流 2 関数の大きい方の 2 倍） | — | ADR 0135 決定 6 |
| IR CPU reference execution △ | IR の CPU 参照実行 | 容器のグラフを、runtime の op ごとの TS 参照実装（`applyReferenceOp`）で CPU 上に 1 op ずつ回すこと（重みは容器自身の i8 などを復号して使う）。Qwen3.5 のテキストエンコーダの GPU チェックのオラクル（torch の golden ではないので既定のレーンで回す） | — | ADR 0137 決定 6 |
| band case / acceptance case △ | 帯のケース / 受け入れのケース | 実測から帯を決めるときの 2 つの入力の組。帯のケースの最悪値の定数倍が帯になり（ADR 0118 決定 8 では × 5）、受け入れのケース（別の seed・条件・境目）がその内に入ることを確かめる。帯のケースは自分の帯では赤にならないので、検査を赤にできるのは受け入れのケースと故障注入だけ | — | ADR 0118 決定 8・追記 2026-10-02 |

## op の語彙

| 英語（コード・docs） | 日本語（対話） | 意味 | やめる言い方 | 正本 |
| --- | --- | --- | --- | --- |
| entry door △ | 語彙への入口 | op を IR 語彙へ足すときに ADR が要るかを決める 1 判定（Core ATen 帰属の実測） | 入場門 | ADR 0059 決定 2・0064 |
| export-eliminated layer △ | export で消える op | export 時の畳み込み・分解で消え、ランタイムに届かない op（旧第 0 層） | export 消滅層 | ADR 0059 決定 1 |
| extended atomic layer △ | 分解できない追加 op | 語彙内の合成で厳密同値に書けない非 core の op（旧第 1' 層） | 拡張原子層 | ADR 0059 決定 1・3 |
| extended composite layer △ | 分解できるが残す追加 op | 書けるが性能・意味のために残す非 core の op（旧第 2 層の残り） | 拡張分子層 | ADR 0059 決定 1・3・5 |
| `group_norm` △ | group_norm（GroupNorm の op） | GroupNorm の Core ATen 層の op。群ごとに平均 → 偏差の二乗和の 2 パスで統計を取り（木の形は固定）、チャネルごとの affine を掛ける | — | op-vocabulary の NOTE 2026-10-27・ADR 0138 決定 7 |
| Gated DeltaNet (GDN) △ | Gated DeltaNet（GDN） | value head ごとの dk × dv 行列を状態に持ち、token ごとに exp(g) で減衰させてから beta の強さで v へ向けて rank-1 補正する線形 attention の層。Qwen3.5 系で使う（MOSS-Image のテキストエンコーダは 32 層中 24 層） | — | ADR 0135 Context |
| `gated_delta_scan` △ | gated_delta_scan（GDN の走査 op） | GDN の再帰を担う提案中の op。`(q, k, v, g, beta) → o`・arity 5・f32・attrs 空。decode はその states 形 | — | ADR 0135 決定 1・4 |
| sequential scan (GDN) △ | 逐次スキャン | token を 1 つずつ進め、状態をレジスタに置く kernel の形。(a2) は value の列 1 本を key 方向に 8 スレッドへ分ける（この分け方は数値の一部）。全ての T でこの op の kernel | — | ADR 0135 決定 3 |
| text tower (Qwen3.5) △ | テキストタワー | Qwen3.5 のチェックポイントのうち、`lm_head` と画像側（vision tower）を除いた言語モデルの部分（埋め込み・decoder 層・最後の norm）。MOSS-Image はその最後の隠れ状態を読む | — | ADR 0137 |
| blocked scan (GDN) △ | ブロック化スキャン | (a3)。逐次スキャンの状態を 8 / 16 token ずつ UT 変換で進める形。長い T で速いが丸めが逐次スキャンと違う。性能の候補に留める | — | ADR 0135 決定 3 |
| chunked scan (WY / UT form) △ | チャンク形式（WY / UT 形式） | 上流の prefill の算法（64 token の chunk・3 dispatch）。RTX 3080 Ti ではどの T でも遅く不採用 | — | ADR 0135 決定 3 |
| accurate exp △ | 高精度 exp | 組み込みの exp の代わりに使う、範囲縮小つき多項式の exp。組み込みの偏りが減衰の積で溜まるのを防ぐ。GDN の走査では MUST | — | ADR 0135 決定 3 |

## 生成と文脈

| 英語（コード・docs） | 日本語（対話） | 意味 | やめる言い方 | 正本 |
| --- | --- | --- | --- | --- |
| capacity | capacity | 1 会話が確保する KV cache の論理位置の数（使える最大の文脈長） | — | ADR 0066 |
| chunkLength | chunkLength | prefill 1 回に流す行数 | — | ADR 0066 決定 4 |
| maxChunkLength | maxChunkLength | `chunkLength` の上限（配布形の宣言） | trace 上限 | ADR 0066 |
| chunkBuckets (prefill buckets) | prefill バケット | chunk を走らせる物理行数の宣言集合 | — | limitations |
| KV cache | KV cache | 過去 token の key / value を持つ領域 | — | ADR 0066 |
| fork | 分岐（API 名は fork） | 会話の途中の状態を丸ごと複製し、独立の続きを持たせる操作 | — | ADR 0066 / 0083 追記 2026-10-11 |
| sliding layer / full layer | sliding 層 / full 層 | 直近の `window` 位置だけを見る attention の層 / 全部を見る層 | — | ADR 0066 / 0067 |
| ring | ring | sliding 層の KV を物理行へ巡回配置する環状バッファ | — | ADR 0067 |
| slack | slack | ring の行数が窓より多い余り | — | ADR 0096 |
| MTP (Multi-Token Prediction) | MTP | drafter が複数 token をまとめて提案する仕組み | — | ADR 0096 |
| speculative decoding | 投機デコード | drafter の k token を本体が 1 run で検証して一致した分を確定する実行形 | speculative（「投機デコード」と混ぜない） | ADR 0096 |
| fusion | 融合 | 隣接する複数 op を 1 回の GPU 起動へまとめる変換 | — | ADR 0040 |
| linear prologue △ | linear のプロローグ融合 | a8 の linear が入力を作る elementwise op を自分の量子化（手順 ①）に取り込み、その op の出力を格納しない融合 | — | ADR 0040 追記 2026-10-25 |
| token-major / head-major △ | トークン順 / ヘッド順（の配置） | attention の被演算子の 2 つの並び。トークン順 `[B, S, H, D]`（射影の出力をそのまま reshape した形）とヘッド順 `[B, H, S, D]`（融合 attention op の宣言形） | — | ADR 0040 追記 2026-10-25 |
| attention permute absorption △ | attention の permute 吸収 | `attentionCompute: "a8"` のとき、融合 attention が直前の q / k / v の `[0, 2, 1, 3]` permute を取り込み、トークン順のまま読む融合（直後の出力の permute はノードのまま。経路は見ずにこのオプションだけで決め、`attentionPermute` で数える） | — | ADR 0040 追記 2026-10-25 |
| key valid length / padded key axis (online attention) △ | キーの有効長 / 詰めたキー軸 | online の kernel が 4 の倍数でないキー長 N を回す形。int8 の Vᵀ の行を `ceil(N/4)` 語まで 0 で詰め、N 以降のキーは −3.0e38 で消すので詰め物は出力に入らない（③PV の i8a8 は今も N % 4 == 0 が要る） | — | ADR 0132 追記 2026-10-27 |
| online-route device facts (`attentionDeviceFacts`) △ | online の経路の device の事実 | Session が online の kernel を選ぶかとその形を決める device の事実。見積もりが Session と同じ経路を数えるために公開した | — | ADR 0132 追記 2026-10-28 |
| three-dispatch route (fused attention) △ | 3 dispatch の経路 | スコアを格納する融合 attention の形。①QK がスコア S を書き、② が行の統計（softmax）を取り、③PV が P × V を計算する — attention 1 つに kernel 3 本。online の kernel は同じ attention を 1 dispatch で回し、online が動かないところでは Session がこの経路に落ちる | — | ADR 0132・ADR 0138 Context |
| P̃ / Vᵀ (int8 attention operands) △ | P̃ / Vᵀ | P̃ = softmax の重みを 1/127 の格子で int8 にしたもの。Vᵀ = V の転置（キーの方向 N が行の中に並ぶ）。どちらも N の方向に 4 キーずつ u32 の 1 語に詰めるので、③PV の i8a8 は N % 4 == 0 が要り、online の kernel はキー軸を詰める（キーの有効長） | — | ADR 0132・ADR 0138 Context |
| rmsNormRope（rms_norm → pair RoPE fusion）△ | rms_norm と RoPE の融合 | Wan DiT の self-attention の q / k の前処理（トークン行の `rms_norm` → reshape 2 本 → ペア形の RoPE）を、正規化した行を格納しない f32 カーネル 1 本へ畳む融合（素の経路が 256 幅の workgroup の `rms_norm` を使うときだけ。`rmsNormRope` で数える） | — | ADR 0040 追記 2026-10-25 |
| deferred node △ | 後ろへ回すノード | 融合の窓の中で畳まないノードのうち、融合ステップの**後ろ**へ置いて走査し直すもの（窓内 passthrough は前に置き、走査し直さない）。q と k の鎖が入り組むときに使う | — | ADR 0040 追記 2026-10-25 |
| split-tree axis reduce △ | 部分木分割の軸 reduce | 末尾以外の軸の reduce で 1 出力を 16 スレッドが分け持ち、同じ加算木の揃った部分木を各自が畳んでから根を順に合わせる形（加算の順序が同じなのでビット同一・分割数はパイプラインキーに入る） | — | ADR 0035 追記 2026-10-26 |
| geometry profile | 幾何プロファイル（会話では「プロファイル」でもよい） | GEMM の幾何を op × 行数バケットごとに決めた静的な表 | — | ADR 0115 |
| `./tune` (runtime subpath) | tune のサブパス / 掃引の面 | 掃引・生成器・表の直列化と照合を出す `@karume/runtime/tune` | — | ADR 0117 決定 1 / 2 |
| kernel fingerprint (`provenance.kernels`) | カーネルの指紋 | 保存した表がどのカーネルで測った物かを見分けるハッシュ | — | ADR 0117 決定 4 |
| case-set version (`caseSet`) | ケース集合の版 | 掃引ケースと行数バケットの境界から導くハッシュ | — | ADR 0117 決定 4 / 8 |
| rounding-error bound (E) | 丸め誤差の上界 | 100 µs 丸めが既定比に与える相対誤差の上界 | — | ADR 0117 決定 3 |
| rounding-error bound (E of the measurement check) | 丸め誤差の上界 / 計測のチェック | E ≤ 1% の観測だけを比の材料にする計測の検査 | 計測の門 | ADR 0117 決定 3 |
| matching (`geometryProfileMismatch`) | 照合（表の） | 保存した表を今の adapter と runtime に当ててよいかを見る純関数 | — | ADR 0117 決定 5 |
| row-count bucket (gemmRows bucket) | 行数バケット（「バケット」でもよい） | GEMM の行数 M で幾何を切り替える区切り | 段 | ADR 0116 |
| decode run-ahead (`decodeAhead`) △ | 先行投入 | greedy の decode を GPU の batch 1 本に最大 d step まとめ、各 step は前の step が GPU 上で選んだ token をそのまま入力にし、d 個の選択を batch の最後のフェンス 1 回で読み戻す実行形（opt-in・Gemma で PLE を GPU に置くときだけ）。token の列は 1 step ずつの decode と同じ | — | ADR 0142・perf-ledger H-27 |
| chained deferred run (`GenerationRun.chain`) △ | 確定待ち run の連結 | 確定待ちの行が残る context に、その後ろへ続けて置く commit: "deferred" の run。位置は発行時に予約し、未決着のまま並べてよいのは同じ `BatchScope` の中だけ。範囲全体は同じ行数の deferred run 1 本と同じ上限（`slidingSlack`・`chunkLength`）に従い、`commit(rows)` 1 回でまとめて確定する | — | ADR 0142 決定 1 |
| fence floor (Deno) | Deno のフェンス床 | Deno の WebGPU 実装がフェンス 1 本ごとに 10 ms 寝る定数 | — | research 2026-09-19 §3.2 |
| in-flight submit bound | 送り込みの先行の上限 | ホストが GPU の完了より先に積んでよい `queue.submit` の本数の上限 | — | ADR 0123 |
| completion marker (in-flight) | 完了印 | 送り込みの先行の上限が置く `queue.onSubmittedWorkDone()` の 1 本（フェンスではない） | — | ADR 0123 決定 1・5 |
| residency / eviction (anima DiT) | 常駐 / 退避 / 格下げ | DiT の Session を generate を跨いで持つこと / 常駐 DiT を捨てること | — | ADR 0112 |
| headroom probe (trial allocation) | 試し確保 | 今この量が入るかを 1 度だけ確保して試す runtime の関数（`fitsHeadroom`） | 空き容量の取得 | ADR 0070 決定 5 追記 2026-09-27 |
| row-block budget | 行ブロックの予算 | attention のスコアを行ブロックに分けるときの一時 1 本ごとの上限 | — | ADR 0124 |
| non-fatal OOM | 非致命 OOM | `createBuffer` が例外を投げずに無効なバッファを返し、後の errorScope で判明する OOM | — | ADR 0095 追記 2026-10-10 |
| transient region / least-growth placement | 一時バッファの領域 / least-growth 配置 | plan が中間テンソルを offset で詰める GPU バッファ / 広がりが最も小さい領域へ置く配置（同じ大きさの並べ順 2 通りの良い方・2026-10-26 に first-fit から置き換え） | — | ADR 0093 追記 2026-10-26 |
| pre-emptive / reactive eviction | 先回りの退避 / 反応の退避 | 常駐 DiT を試し確保の結果で先に手放す線 / OOM の後に手放す線 | — | ADR 0112 追記 2026-09-27 |
| context K/V projection (`dit_context`, context K/V) | 条件側 K/V 射影（役割名は `dit_context`・観測名は `dit-context`） | irodori の DiT が条件 3 本から作る cross-attention の K / V | — | ADR 0114 |
| input error (`ModelInputError`) | 入力起因エラー | 渡した要求そのものが受理できない失敗（入力を直せば通る） | バリデーションエラー（と混ぜない） | ADR 0107 |
| folded ①′ (head fold) △ | head をまとめた ①′ | states 形の decode の ①QK で、1 workgroup が同じ kv plane を使う F = r 本の query head を受け持ち、K の各要素を 1 回だけ読む形。選んだ組では確かめた全 backend で ①′ とビット一致。計測して速く、かつビット一致を確かめた (r, D, M) の組の full attention でだけ選ぶ（E4B の F = 4 は D3D12 で不一致のため選ばない） | — | ADR 0067 追記 2026-10-27・perf-ledger K-55 / K-95 |
| unrolled KV loop △ | KV ループの展開 | ③′ PV と融合 stats + PV で、各 lane が自分の担当列 8 本分の V を先に読んでから積和する形。式と順序は v1 と同じでビット一致 | — | ADR 0067 追記 2026-10-27・perf-ledger K-91 |
| baseline WGSL (`fixtures/wgsl-baseline/`) △ | 書き換え前の WGSL の控え | ビット一致の書き換えの前の生成 WGSL を残したもの。GPU テストで新旧の kernel を同じ入力で回し、出力を 1 語ずつ比べる | — | `gpu_state_attention_fold_test.ts`・perf-ledger K-91 |
| prefetch form (rms_norm v2) △ | 先読み形（rms_norm v2） | 256 幅の rms_norm と RMS → add 融合で、weight を workgroup ごとに 1 回、行の x（と residual）を縮約の前にレジスタへ読み、出力ループの回数を定数にした形。縮約と値の式は v1 と同じ文字列でビット一致 | — | ADR 0017 追記 2・perf-ledger K-99 |
| chat dialect △ | チャットの書式 | Gemma 4 のモデルが従うチャットテンプレートの種類（`gemma4` / `gemma4-unified`） | — | glossary.md |
| suppressed tokens △ | 抑制トークン | モデルが生成しないトークン id（選択の前に logit を −∞ にする） | — | glossary.md |

## 動画生成

| 英語（コード・docs） | 日本語（対話） | 意味 | やめる言い方 | 正本 |
| --- | --- | --- | --- | --- |
| `conv3d` | conv3d | 3D 畳み込みの IR op（implicit GEMM） | — | ADR 0118 決定 1 |
| causal 3D VAE | 因果 VAE | 時間方向は過去のフレームだけを見る 3D 畳み込みの VAE | — | ADR 0118 決定 2 |
| feat_cache (causal cache) | 因果キャッシュ | CausalConv3d が次の chunk へ持ち越す末尾 2 フレーム | — | ADR 0118 決定 2 |
| chunk (VAE) | chunk | 動画 VAE のデコード 1 回分の潜在 1 フレーム | — | ADR 0118 決定 2 |
| chunk graphs (first / next) | chunk グラフ | 最初の chunk 用と以降用の 2 種の VAE グラフ | — | ADR 0118 決定 2 |
| final form (VAE rewrite) | 最終形・最終形の比 | cache の正規化に加えて全ての書き直しを当てた chunk グラフ | — | ADR 0118 検収 |
| unbatched | unbatched（バッチ軸なし） | バッチ軸を持たない形 `[C,T,H,W]` | — | ADR 0118 決定 1 |
| temporal interleave | 時間インターリーブ | `time_conv` の 2 倍チャネル出力をフレーム方向へ交互に並べ直す操作 | — | ADR 0118 決定 2 |
| interleaved form / half-split form (RoPE) | interleave 形 / half-split 形 | RoPE の回転の対の取り方（隣り合う 2 要素 / head の前半と後半） | — | ADR 0118 決定 3 |
| tiled decode (video) | タイル decode | 空間の固定タイル × 時間 chunk の 2 軸に分けてデコードする方式 | — | ADR 0118 決定 2 |
| text embedding asset | テキスト埋め込み資産 | 固定プロンプトごとに事前計算した umT5 の出力 | — | ADR 0118 決定 4 |
| valid length (L_valid) | 有効長 | トークナイザのマスクが 1 の長さ | — | ADR 0118 決定 3 / 4 |
| normalized ratio (r) | 正規化比 | GPU の誤差を CPU f32 参照自身の誤差で割った、実寸の DiT 照合の指標 | — | ADR 0118 追記 |
| `submitGpuTime` (GPU time per submit) | 1 submit の GPU 時間 | 1 submit ごとの GPU 実行の幅の観測（制御には使わない） | — | ADR 0118 決定 6 |
| umT5 (umT5-XXL) | umT5 | Wan2.1 / 2.2 のテキストエンコーダ（多言語 T5 の encoder） | — | ADR 0118 / 0119 |
| `prompt_clean` | プロンプトの前処理 | 上流 Wan のプロンプト前処理 | — | ADR 0119 決定 2 |
| layer-sequential reference | 層逐次の参照 | 重みを層ごとに読んで回す CPU 参照 | — | ADR 0119 決定 8 |
| `BADNESS_RE` (mojibake detection) | mojibake 判定 | ftfy の文字化け判定の正規表現（当たる入力は拒む） | — | ADR 0119 決定 2 |
| unquantized reference | 量子化なしの参照 | pin した checkpoint の F32 の重みをそのまま使う CPU 参照 | — | ADR 0119 追記 2026-10-03 |
| text encoder path (gpu / precomputed) | テキストエンコーダの経路 | umT5 を GPU で回すか、事前計算の埋め込み資産を使うかの選択 | — | ADR 0119 追記 |
| revision source (cross-repository reference) | revision の出どころ | レシピが宣言した越境参照の commit をビルドがどこから取るか（`--resolve-refs hf` / `--ref-revision` / `--allow-placeholder-ref` のどれか 1 つ） | — | ADR 0038 §7 追記 2026-10-28 |
| placeholder SHA (cross-repository reference) | 仮の SHA | 参照先のリポが未公開の間に越境参照へ入れる 40 桁の 0 | — | ADR 0119 追記 |
| `karume-umt5-xxl` / `umt5-encoder/1` | umT5 リポ | umT5-XXL encoder とトークナイザ資産の配布リポと pipeline 名 | — | ADR 0119 追記・ADR 0122 |
| `umt5.Encoder` (standalone umT5 encoder) | 単体の umT5 encoder | `@karume/models/umt5` の公開面（テキスト → 最終隠れ状態） | — | ADR 0122 追記 2026-10-15 |
| `karume-umt5-tokenizer/1` (umT5 tokenizer asset) | umT5 のトークナイザ資産 | `karume-umt5-xxl` のトークナイザ資産の形式 | — | ADR 0122 追記 2026-10-15 |
| TI2V (text-image-to-video) | TI2V | 1 つの重みで T2V と I2V の両方を出す動画生成 | — | ADR 0121 |
| T2V / I2V (text-to-video / image-to-video) | T2V / I2V | 文章だけから動画 / 文章 + 先頭の画像 1 枚から動画 | — | ADR 0121 |
| condition frame / condition mask (`condition_mask`) | 条件フレーム / 条件マスク | I2V で画像の潜在に置き換える先頭の潜在フレームと、そのトークンを示す DiT の入力 | — | ADR 0121 決定 3 |
| fit (`"crop"` / `"stretch"`) | fit（crop / stretch） | I2V の条件画像を出力寸法へ合わせる方法 | — | ADR 0121 裁定 3 |
| condition image / condition latent | 条件画像 / 条件の潜在 | I2V の要求の画像と、それを VAE encoder に通して正規化した値 | — | ADR 0121 決定 11 |
| encoder stage (`vae_encoder`) | encoder の処理（`vae_encoder`） | I2V で最初に回る VAE encoder の処理 | encoder の段 | ADR 0121 追記 |
| symbolic-length graphs (the 3 parts of the VAE encoder) | 記号長のグラフ | 寸法を記号のまま持ち、受理寸法ごとに作り分けないグラフ | — | ADR 0121 追記 2026-10-09 |
| release wait (`settleReleasedMemory`) | 解放待ち | 処理を畳んだ直後に空の submit を出して解放を待つこと | — | ADR 0121 追記 |
| two time inputs (`timesteps_proj_condition`) | 時刻入力 2 本 | Wan2.2 の DiT で時刻の MLP を生成側と条件側の 2 回回す形 | — | ADR 0121 決定 3 |
| patchify space (VAE 2.2) | patchify 空間 | RGB 3 ch を空間 2×2 で 12 ch に畳んだ空間 | — | ADR 0121 決定 6 |
| DupUp3D / AvgDown3D | DupUp3D / AvgDown3D | VAE 2.2 の残差の外側のショートカット | — | ADR 0121 決定 6 |
| meta trace (export) | meta trace | 重みを持たない meta の上流で trace する export の形 | — | ADR 0121 決定 7 |
| `karume-wan2.2` / `ti2v-5b` / `wan-ti2v/1` | Wan2.2 の配布リポ | Wan2.2 TI2V-5B の配布リポ・モデル名・pipeline 名 | — | ADR 0121 決定 9 / 10 |
| generation descriptor (Wan) | 世代の記述子 | Wan2.1 と 2.2 で違い、グラフ宣言から導けない値を集めた TS の定数の束 | — | ADR 0121 決定 10 |
| accepted without a development-machine sha row | 開発機の sha 行を持たない受理 | 受理集合に入れるが、開発機で sha 行もチェックも持てない寸法・フレーム数 | — | ADR 0121 決定 8 |
| 10 GB class / 12 GiB class (machine VRAM class) | 10 GB 級 / 12 GiB 級 | 動作を確かめた長さを機の VRAM の容量で分ける呼び名 | — | limitations |
| i8 weight count / linear dispatch count | i8 の重みの本数 / dispatch の本数 | 容器が i8 で持つ linear の重みの本数と、1 パスで走る linear の dispatch の本数 | — | ADR 0121 決定 2 |
| spatial compression (Wan VAE tiles) | 空間の圧縮 | グラフの入出力の空間比 × ホストの unpatchify の倍率 | — | ADR 0121 決定 6 |
| shared modules (Wan) | 共有モジュール | Wan2.1 / 2.2 の class が共有する text / DiT / VAE の処理の関数 | — | ADR 0121 決定 10 |
| `wan.series_check` (series comparison) | 中間成果物の照合 | recipe の生成物を空のディレクトリへ書き直し、既存の中間成果物と名指した項目だけ突き合わせる CLI | 系列の照合 | ADR 0121 追記 |
| original (google/umt5-xxl) | 本家 | umT5-XXL を公開した Google の HF リポ（karume-umt5-xxl の上流） | — | ADR 0122 決定 1・2 |
| compatible text encoder | 互換の text encoder | 本家と構成が同じで重みだけが違う umT5 | — | ADR 0122 決定 5・7 |
| intake / intake.json | 取り込み | 第三者の safetensors を pin した revision で取り、sha256 を照合して置く工程 | — | ADR 0122 決定 5 |
| original dtype (intake) | 元の dtype | intake.json が名乗る上流の格納（BF16 / F32） | — | ADR 0122 決定 5 |
| local experimental mirror | 実験用ミラー | 配布形と同じ形だが公開しないディレクトリ | — | ADR 0122 決定 5・6 |
| undeclared-licence marker | 未宣言の印 | 上流がライセンスを宣言しない重みの容器に書く `NOASSERTION`（SPDX） | — | ADR 0122 決定 6 |
| pre-publication licence check | 公開前のライセンスチェック | 上げる前に全容器のライセンスの印を読み、未宣言の印を拒む検査 | 公開前の門 | ADR 0122 決定 6 |
| relative-position bucket configuration (umT5) | バケットの構成 | umT5 の相対位置バイアスのバケット数・最大距離と有効長の上限 | — | ADR 0122 決定 5・7 |
| `--swap-text-encoder` (Wan example) | 差し替えのフラグ | umT5 を互換の部品へ差し替えた `examples/wan` の CLI のフラグ（0.15 で削除） | — | ADR 0122 決定 7 |
| `--generation` (Wan example) | 世代のフラグ | 世代を選んだ `examples/wan` の CLI のフラグ（0.15 で削除） | — | ADR 0121 |
| `KARUME_WAN_SWAP_TEXT_ENCODER` | 差し替えの e2e の opt-in | 差し替えの e2e に差し替え先の手元の配布形を渡す環境変数 | — | ADR 0122 追記 |
| baseline (moving umT5 provenance) | 基準線 | 出所を切り替える前に今の出所のまま作る比較の起点 | — | ADR 0122 決定 9 |
| tied alias | tied な別名 | 同じ重みを 2 つの名前で指す形 | — | ADR 0122 決定 2 |
| RNE (round to nearest even) | RNE | f32 → bf16 の標準の丸め（最近接偶数丸め） | — | ADR 0122 |
| `umt5.SOURCES` (TS — table of the umT5 public distribution repositories) | umT5 の取得元の対応表 | `@karume/models/umt5` が出す公開配布リポの対応表 | — | `packages/models/src/umt5/config.ts` |
| `wan.sources.UMT5_SOURCES` (umT5 upstream table) | umT5 の上流の表 | umT5 の書き手が読む本家の上流の表（recipe 側） | — | `tools/export-recipes/wan/sources.py` |
| two axes (`--upstream` / `--model`) | 軸の 2 本 | umT5 の台本の 2 つの選択（重みの出所 / Wan の text 側の snapshot） | — | ADR 0122 決定 3 |
| `axes` in the golden metadata | golden の axes | golden の各ケースのメタに残す、2 本の軸の上流と pin の記録 | — | ADR 0122 決定 3 |
| `export_dit --check` | `export_dit --check` | Wan2.1 の DiT の中間成果物を一時ディレクトリへ書き直し、sha256 で突き合わせるだけの口 | — | ADR 0121 追記 |
| `ti2v_export_dit eager-full` | eager-full | Wan2.2 5B の上流を f32 の全量で読み、パッチ後のグラフの eager と比べる口 | — | ADR 0121 決定 4 |
| `dit_condition_mask` (host function for the condition mask) | 条件マスクのホスト関数 | 潜在の形と patch から条件マスクを作るホストの関数 | — | ADR 0121 決定 3 |
| f32 address dependence (MKL) | f32 の番地依存 | 重みの番地の 64 バイト境界からのずれで f32 の最終ビットが変わる現象 | — | ADR 0121 追記 |
| unpatchify (VAE exit) | unpatchify | patchify 空間を RGB へ戻すホストの並べ替え | — | ADR 0121 決定 6 |
| DiT input form (`ditInputForm`) | DiT の入力の形 | 世代の記述子の欄（`t2v` = 5 入力 / `ti2v` = 7 入力） | — | ADR 0121 決定 3 |
| generation bundle (`WanFamilySpec`) | 世代の束 | class ごとに違う値をまとめた値 | — | `packages/models/src/wan/family.ts` |
| generation script (the first Wan2.2 video) | 生成スクリプト | 製品の class ができる前に Wan2.2 の動画を出した git 追跡外の台本（以後は使わない） | — | ADR 0121 追記 |
| frozen copy (GPU check) | 凍結コピー | 対象コミットを `git archive` で取り出し、資産へリンクを張ったディレクトリ | — | ADR 0121 追記 |
| relay step (test VRAM) | 中継の step | `t.step` へ中継の関数だけを渡し、GPU を掴む本体を step の終了時に手放す形 | — | ADR 0121 追記・known-issues |
| generation table (Wan distribution recipe) | 世代の表 | Wan2.1 / 2.2 で違う配布の事実を集めた frozen dataclass | — | `tools/export-recipes/wan/distribution.py` |
| seams (of the DiT pass) | 切れ目 | Wan の DiT の 1 step の中で GPU が DiT の計算をしていない部分 | — | research 2026-10-15・perf-ledger H-37 |

## 画像生成（MOSS-Image）

| 英語（コード・docs） | 日本語（対話） | 意味 | やめる言い方 | 正本 |
| --- | --- | --- | --- | --- |
| MOSS-Image (family `moss`) △ | MOSS-Image | OpenMOSS の 5B の text-to-image DiT（テキストエンコーダは Qwen3.5-9B・VAE は FLUX.2）。Karume には DiT と VAE があり、2026-10-31 からテキストエンコーダ（ADR 0137）もプロンプトの形で繋がった。2026-11-01 から `@karume/models/moss` とローカルの配布形 `karume-moss-image`（HF には未公開） | — | ADR 0138 |
| joint sequence (S = N + T) △ | 結合系列 | 画像トークン N 個とテキストトークン T 個をつないだ DiT の系列（全体で self-attention） | — | ADR 0138 決定 2 |
| text refiner (`dit_text`) △ | テキストリファイナー | テキストトークンだけに掛かる Linear + 2 block（modulation なし）。1 生成で分岐ごとに 1 回だけ回す別グラフで、どのプリセットでも f32 の計算 | — | ADR 0138 決定 3 |
| text context (`text_context`) △ | テキスト文脈 | `dit` の入力 `[1,S,3072]`。テキストの行にリファイナーの出力、画像の行に 0 を置く | — | ADR 0138 決定 2・3 |
| shared modulation △ | 共有の modulation | 時刻の埋め込みから 1 本の Linear で作り、42 block 全部が使う 6 本のベクトル | — | ADR 0138 Context |
| situ △ | situ | FFN の活性化 `(4·tanh(g/4)·sigmoid(g)) · (25·tanh(h/25))`。既存の op で書く | — | ADR 0138 決定 6・perf-ledger F-10 |
| skip 22 (`dit_skip22`) △ | skip 22（block 22 を飛ばす負側の分岐） | 上流の既定の negative の forward（block 22 を飛ばす）。Karume では `dit` の重みを全部借りる別グラフ | — | ADR 0138 決定 4 |
| DiT pair (lender / borrower) △ | DiT の組（貸し手 / 借り手） | MOSS-Image の DiT の段で同時に開く 2 つの Session。貸し手 `dit`（positive の分岐）が重みを持ち、借り手 `dit_skip22`（negative の分岐）は initializer を全て shared で宣言して重みのバイトを持たない。組の必要量は重みを 1 回だけ数える | — | ADR 0138 決定 4・5 |
| training grid (`training_grid`) △ | 学習グリッド | scheduler の 1,000 点の σ と timestep と時刻の sinusoid の表。要求の σ は必ずどれかの点に寄るので、ホストは計算せず表を引く | — | ADR 0138 決定 9・11 |
| precomputed text embeddings (MOSS-Image) △ | 事前計算の埋め込み | 上流の `encode_prompt` の出力 `[T, 4096]`（f32）を固定プロンプトについて CPU で書いたもの。リクエストの埋め込みの形（上流との一致の検証と、上流の埋め込みとの A/B の入力） | — | ADR 0138 決定 9・12 |
| prompt form / embeddings form (MOSS-Image request) △ | プロンプトの形 / 埋め込みの形 | 生成リクエストのテキストの 2 つの形。プロンプトの文字列（`prompt` / `negativePrompt` — 受理の検査でチャットテンプレートとトークナイザ、GPU でテキストエンコーダの段）か、事前計算のエンコーダ出力（`positive` / `negative`）。両方の分岐で同じ形。どちらにも negative の既定は無い（上流の既定は `""`） | — | ADR 0138 決定 9・追記 2026-10-31 |
| text-encoder stage (MOSS-Image) △ | テキストエンコーダの段 | プロンプトの形の生成の最初の段（`encodeMossPrompts`）。自分の空き容量の検査 → Qwen3.5 のテキストタワーを開く → positive、negative の順に符号化 → どの経路でも破棄して解放を待ってから `dit_text` を開く | — | ADR 0138 追記 2026-10-31・ADR 0137 決定 4 |
| latent BatchNorm restoration (`latent_batch_norm`) △ | BatchNorm の復元 | VAE の前に BatchNorm の running 統計で `z · std + mean` に戻す手順（とその資産）。その後の unpatchify（128 → 32 チャネル・2 倍）は別のホスト側の手順 | — | ADR 0138 決定 8 |
| untiled decoder (MOSS-Image VAE) △ | タイルなしの decoder | FLUX.2 の decoder を画像全体で回す形。記号長の 3 グラフに割り、resident tensor で受け渡す | — | ADR 0138 決定 8 |
| area cap (MOSS-Image) △ | 受理の面積の上限 | タイルなしの decoder の最大の中間テンソルが binding 1 本に収まる面積（2 GiB で 2,796,160 画素）。device ごとの値 | — | ADR 0138 決定 8・limitations |
| reference preset / practical preset (MOSS-Image) △ | 参照側のプリセット / 実用プリセット | `f32+dit8`（i8 の DiT の重み × f32 の計算）は `f32+dit8-a8-attn8-online`（同じ重みで `dit` / `dit_skip22` に a8 の linear と online の attention）の参照側のプリセット。既定は実用プリセット（manifest の `defaultQuant`・コードには既定なし） | — | ADR 0138 決定 13 |

## 画像の後処理

| 英語（コード・docs） | 日本語（対話） | 意味 | やめる言い方 | 正本 |
| --- | --- | --- | --- | --- |
| post-processing method (`float-bilinear` / `u8-pil-bicubic`) | 後処理の手順 | BiRefNet の logit を入力画像の寸法の 8bit α にする手順の名前 | — | ADR 0125 決定 3 |
| shared Pillow core | Pillow の共通の核 | Pillow の 8bit の分離型リサンプルの逐語の写しを 1 本に寄せたもの | — | ADR 0125 決定 7 |
| golden ε (post-processing) | golden の ε | `float-bilinear` が golden と 8bit の値で 1 ずれてよい幅 | — | ADR 0125 決定 6 |
| blind pair (three-way) | 盲検の対 | 同じ画像の 2 手順の α を、どちらがどの手順かを伏せて並べた組 | — | ADR 0125 決定 8 |
| edge band | 輪郭の帯 | どちらかの手順で α が 0 と 255 の間になる画素を 2 画素膨張した範囲 | — | ADR 0125 決定 9 |
| K columns / overall columns | K 列・総合列 | A/B の数値の列（記録だけで判定には使わない） | — | ADR 0125 決定 9 |
| pseudo ground truth (A/B) | 擬似正解 | 総合列の基準（2048² のモデルの σ） | — | ADR 0125 決定 9 |
| Gradient error | Gradient error | α の勾配の大きさの、擬似正解との差の二乗和 | — | ADR 0125 決定 9 |

## パッケージの公開面

| 英語（コード・docs） | 日本語（対話） | 意味 | やめる言い方 | 正本 |
| --- | --- | --- | --- | --- |
| bare name | 接頭辞なしの名前 | ファミリー名の接頭辞を外した公開名（`Pipeline` / `SOURCES` / `toUtterance`）。`@karume/models` の各サブパスはこれだけを出す（0.15.0 から） | — | ADR 0136 決定 1 / 2 |
| family namespace | ファミリーの名前空間 | barrel がファミリーを 1 つずつ出す `export * as <名前空間>`（サブパスのキーの camelCase）。中身はサブパスのモジュールそのもの | — | ADR 0136 決定 3 / 6 |
| shared surface (models barrel) | 共通の公開面 | どのファミリーにも属さないので barrel が平らに出す 12 個（`fromPretrained` のオプション型・`ModelInputError`・`KARUME_SOURCES`・PNG / WAV / 画像の補助） | — | ADR 0136 決定 3 |

## 進め方

| 英語（コード・docs） | 日本語（対話） | 意味 | やめる言い方 | 備考 |
| --- | --- | --- | --- | --- |
| plan ◆ | 計画 | 目標と検証可能なゴールを持つ作業全体 | — | 計画 → フェーズ → ステップ |
| phase ◆ | フェーズ | 計画の中の、区切りのある段階 | 波・イテレーション | 同上 |
| step ◆ | ステップ | フェーズの中の順番のある工程 | 段・ステージ | 同上 |
| backlog | バックログ | 作業項目と順序の正本 | — | docs/backlog.md |
| perf ledger | 性能台帳 | 性能候補の起票・採否・打ち切りの基準 | — | docs/perf-ledger.md |
| kill criterion ◆ | 打ち切りの基準 | 候補を打ち切る条件 | kill 基準 | perf-ledger |
| spike ◆ | 試作 | 事実を集めるための使い捨ての実装 | スパイク | — |
