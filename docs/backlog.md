# Backlog — 波順と作業項目の正本

> プロジェクト全体の**優先順位・波順・未消化項目**の正本はこの 1 本。
> 状態語彙: `now`（現行波）/ `next`（次の大波）/ `later` / `release` / `parked`（復活条件つき）。
> 運用契約: ①完了した項目は**削除する**（履歴は git と ADR / research が持つ）②実測値・設計論証は
> ここに**書かない** — 出典（ADR / research / 台帳）を指す ③性能候補の起票・採否・kill 基準は
> [perf-ledger](perf-ledger.md) が正本で、ここは波として参照するだけ ④by-design 制約の正本は
> [limitations](limitations.md) — 作業化が裁定された時だけここに載る。

## now — 0.5.1 リリース段（サンプラー再裁定）

**0.5.0 は 2026-08-25 にクローズ**（結果は下の消化済み節）。直後にサンプラーの再裁定
（ADR [0078](decisions/0078-anima-sampler-selection.md) — 既定 Euler 維持・DPM++ 2M は
request の `sampler` 席で選ぶ）を消化した:

- HF 側は**上げ直し済み**: anima `2682441a` / turbo `88357344`（euler 化・越境参照追随・
  カード Usage の repo 誤記修正）— `revision: "main"` 追従の利用者は復旧済み
- コード側 = `sampler` 席（`74febe7`）+ pin 更新（`7ea134d`）— **0.5.1 の中身はこの 2 点**
- 残り: push → CI 緑 → GitHub Release v0.5.1 → JSR publish（0.5.0 の pin 利用者はここで解消）

**0.5.1 が出たら次の大波はユーザー裁定待ち**。候補（詳細は各節）:

- SBV2 の yomi 依存分離（later 節 — 0.6.0 想定・breaking。ユーザー意向「早い段階で」）
- 生成 API 波（later 節 — `GenerationProgram` + sequence API。LLM 実需に直結する最大の API 波）
- R1 のロード面 API 工事 4 件 + exporter 自動分割規則（release 節）
- モデル拡充の続き・EmbeddingGemma の完成（later 節）
- anima 素版 i4 の品質改善（later 節 — 配布スキップ裁定の復活レバー）

## 消化済み（0.5.0 breaking 波 — 2026-08-25）

結果だけ残す（経緯は git / 各 ADR / [release-runbook](release-runbook.md)）:

- quant 席名の一斉改名（ADR [0074](decisions/0074-quant-seat-naming.md)）・`linearCompute` /
  `attentionCompute` の値 `"i8a8"` → `"a8"`・`karume/4` 繰り上げ + 表示欄 + `requiredLimits` +
  越境コンポーネント参照（ADR [0075](decisions/0075-quant-presentation.md) /
  [0038](decisions/0038-manifest-v1.md) 追記。`requiredLimits` の DL 前チェック結線は
  release 節に残置）・`fromPretrained` の `ref` 必須化 + `*_CURRENT` 公開 + 暗黙 main warn
  （ADR [0073](decisions/0073-models-source-pin.md) 追記）
- anima の `scheduler.type` 席 + DPM++ 2M（出荷バイトの視認 A/B で base / turbo 両採用）・
  base の i4 席 2 つは配布から除外（復活条件つき — later 節）・受理解像度 8 通り縮小（E-2）・
  estimate の恒等別名再現（レビュー R6V-2）・irodori の構築 AbortSignal
- HF 再アップロード 4 リポ: anima `ebb27bc4` / anima-turbo `6215f965`（text stack 5 役を
  anima へ越境参照 — 8.1G → 6.7G）/ jvnv `be752c63` / irodori `49b61517`（`i8+dit4` の pin
  据え置きを解消）。断片化 26〜32 MiB/term（anima の f16 transformer のみ 9.1 = 不変ファイル
  の既存水準で受理）。非公開 `karume-sbv2-fn` も焼き直しのみ実施（公開は parked のまま）
- lockstep 0.5.0（`uv.lock` 追随込み）→ CI 緑 → GitHub Release v0.5.0 → JSR publish
  （hub / runtime / models = 0.5.0）
- 事後疎通（runbook §5）: JSR 0.5.0 の消費グラフ解決と `fromPretrained(*_CURRENT)` の
  実 DL + 合成を 4 ファミリで確認

## 消化済み（波 K・リリース + 公開 — 2026-08-20〜21）

**波 K はクローズ**（K-1〜K-5 + 0.4.1 の 6 項目すべて消化）。経緯は git と ADR / runbook が
持つので、ここには結果だけ残す:

- 配布形 `karume/3`（ADR [0071](decisions/0071-manifest-v3-shards.md)）・SBV2 既定 quant =
  `w8-bert4`（ADR 0039 決定 5 の再裁定）・SBV2 トーン注入席（ADR
  [0072](decisions/0072-sbv2-text-injection.md)）・pin 焼き込み（ADR
  [0073](decisions/0073-models-source-pin.md)）
- **HF 公開 = jvnv / irodori / anima の 3 リポ**（2026-08-21・FN は parked）
- **JSR publish = 0.4.0 → 0.4.1**（2026-08-21 ユーザー確認）。0.4.1 は models の公開面が追加
  のみで配布形の作り直し不要だが、**runtime の w4a8（`c285f97` / ADR
  [0076](decisions/0076-w4a8-linear-execution.md)）を含み `linearCompute: "i8a8"` × i4 常駐の
  出力ビットが変わる破壊的変更**がこの版で初めて配られている（公開 manifest に該当席が無い
  ため patch に載せた裁定 — 0.5.0 の breaking 波とは別枠）
- 残置: R1 同席の API 工事 4 件と exporter の自動分割規則は release 節

手順の正本 = [release-runbook.md](release-runbook.md)。

autoregressive 波の**残項目（波外へ送り）**:

- **R1 と同席**: manifest の shard 欄は**波 K で消化**（ADR 0071 — `karume/3`）。exporter 側
  shard 分割規則（co-shard を吐く側の保証）は実需（LLM 級配布）まで release 節に残置。
- **MiniCPM5 の token-only 系列**（ADR 0068 追記 4 の同形展開 — models 側 `lastRow` は
  共通化済みで recipe + 門の鏡像だけ。topk の exporter 側〈多出力 aten の getitem 結線〉は
  sampling 実需まで先送りのまま）。
- L8（fake-device 注入面）は保留継続・`enqueue` の generation 面は設けない裁定で確定
  （limitations）。
- 有界論理 extent の席（R2 — IR スキーマ予約のみ・実装は最初の実需モデルまで先送り）・
  bool initializer / storage の設計・pipeline 単位の Session 常駐と device-loss lifecycle
  （perf H-4 と同体）・sampling/RNG はホスト維持（GPU 側は argmax/topk のみ）。

## later

- **カードの Usage repo 導出の硬化（起票 2026-08-25）**: `karume.dist` はカードの Usage 例の
  repo 名を**出力ディレクトリ名**から導出するため、越境参照のステージング焼き（`--out` が
  別名）で誤った repo 名がカードに載る（実害 = turbo カードに `-release` 付き誤名が公開されて
  いた — ADR [0078](decisions/0078-anima-sampler-selection.md) Consequences・runbook §0 に
  運用注意を追記済み）。恒久策 = `Pipeline.repo_name` 系の正本から導出し `--out` 名へ依存
  しない形。
- **examples/anima に `--sampler` ノブ（起票 2026-08-25）**: request 側 `sampler` 席
  （ADR 0078）を CLI デモから振れるようにする小改修。
- **anima 素版 i4 の品質改善（起票 2026-08-24 — 配布スキップ裁定の復活レバー）**: adaLN の
  i8 化（irodori の帰属で効いた知見の移植 — anima では**未実測の仮説**）と量子化感度の高い
  場所の特定。系列 + 視認物は `outputs/series-archive/2026-08-23-anima-base-i4/` に退避済みで
  校正結果は再利用可（正本 =
  [research/2026-08-24-gptq-expansion-quality.md](research/2026-08-24-gptq-expansion-quality.md) §5）。
  turbo 側の i4 席で**未検証のまま残した可能性の一覧**（専用幾何・g16・校正量・もう 1 つの
  劣化機序 — いずれも「試してダメ」ではなく「試していない」）は
  [research/2026-08-21-anima-i4-seat-speed.md](research/2026-08-21-anima-i4-seat-speed.md) §8。
- **irodori adaLN i8 の出荷リグ A/B（起票 2026-08-24）**: sim で効いた adaLN i8（+13.1 MiB）が
  出荷リグでも読み上げ方を改善するかは未検証（sim → 出荷の転移限界 — 同 research §2）。
  復活 = `i8+dit4` 席（旧 `w4`）の品質不満、またはサイズ最適化の実需。
- **生成 API 波（起票 2026-08-19 — 全体レビューの Codex 提案を採用裁定）**: 静的配線と
  リクエストを分離した `GenerationProgram`（setup 時に全結線を検証）+ stateful sequence API
  （`for await` の token イベント・EOS 停止・cancel・多ターン継続）+ `last_row` の runner 側
  導出（ADR 0068 追記 4 の所有関係のみ reopen）。`generateGreedy` は parity 検収用の内部
  ヘルパへ格下げ。LLM 実需（streaming / チャット）に直結する最大の API 波。
- **SBV2 の yomi 依存分離（0.6.0 波 — ユーザー方針 2026-08-20・breaking。0.5.0 からは
  2026-08-25 に切り離し）**: SBV2 パイプラインの入力を「yomi の解析結果だけ受ける」形へ
  再設計し、`@hdae/yomi` を models の依存から外す。ADR 0072 の注入席は結合を素通しに留めて
  あり、その際は overlay 解決が呼び手側へ移る。0.5.0 の設計検討で出た知見（延期の理由でも
  ある）:
  - 入力は **yomi の返り値が構造的に満たす karume 所有型** `{ prosody: Sbv2Prosody, words:
    Sbv2Word[] }` にできる — 変換関数を利用者に書かせる必要は無く、入口の門が核クランプ等の
    正規化を担う（今 `toSbv2Prosody` がやっていること）。
  - `moraTones` / `moraToPhones`（現状 yomi から import）は `Sbv2Mora` の `consonant` /
    `vowel` から**吸収できる** — 言語テーブルを karume 側へ持つ必要は無く、意味論の正本は
    上流 SBV2 の参照に置いたままでよい。
  - 一方で「呼び手が yomi を直接叩いて渡す」だけだと生の手数が増えるため、ヘルパの整備余地が
    残る。そこの設計が済んでいないので 0.5.0 には同乗させなかった。
- **モデルカード定型文の条件出し**（公開前レビュー minor・2026-08-21）: `shared/` パスの
  説明文が shared/ を持たないリポ（irodori / anima）でも出る — 空回りだが無害。core
  `modelcard.py` の該当文を shared/ 実在時のみ出す形へ。
- **measure_quant の配布試算の J-5b 追随**（J-3 中に発見・2026-08-22）: sbv2
  `project_distribution` が「linear の重みスロットだけ・conv / embedding の i4 は格納形も
  実行経路も無い」という pre-J-5b 前提のまま（実際は出荷済み — ADR 0069 追記 6/7）。相対
  比較には無害だが試算が過小で docstring も陳腐化。対象集合と説明の追随を 1 件で。
- **モデル拡充の続き**: Kokoro-82M（LSTM = multi-output 待ち）・MobileSAM / SAM 2
  （conv_transpose2d）・BiRefNet_HR 2048² preset・DA-V2 可変解像度（upsample_bicubic2d）。
  候補調査の時点記録は [recon-2](research/2026-08-14-model-expansion-recon-2.md)。
- **性能候補**: perf-ledger の 🚧（K-2 VAE conv2d / K-4b conv1d i8a8 / K-6 encoder tile /
  K-9 relattn / L-1 cold-load 分解 / L-2 EG 低精度）— 採否・順序は perf-ledger。
- EmbeddingGemma の完成（models pipeline / tokenizer〈Gemma SPM BPE + byte_fallback 新規実装〉/
  配布形・batch>1 export・runtime attention_mask 配線）。
- **w8a8 鏡像門の設置**: `e2e_deberta_w8a8_test.ts`（ADR
  [0026](decisions/0026-w8a8-deberta-deployment.md) 決定 3 — `e2e_deberta_test.ts` は移植済み・
  鏡像側だけ未設置。2026-08-16 裁定で起票）。
- **Anima ホスト糊 parity の常設門化**: `sigmaSchedule` / `cfgEulerStep` / `denormalizeLatents` /
  `padSequence` の「fixture と全 4 実装 bit 同一」は recipe README に実測記録として残るだけで、
  `outputs/series/anima-pipeline*` を読む常設テストは Deno / pytest のどちらにも存在しない
  （2026-08-16 判明 — fixture 4 変種は再エミット済みで前提は解消済み）。
- **ORT Web 対比ベンチ慣行**（2026-08-16 ユーザー裁定）: 両対応モデルで定期測定し、
  遅すぎないか・ボトルネックはどこかを調査する。対象 = EmbeddingGemma（models 側の完成が
  前提）と **KokoroTTS**（2026-08-16 訂正 — 当初の Irodori は打ち間違い。Kokoro は
  Transformers 系で動くため比較しやすい・karume 側は Kokoro-82M 対応が前提 = 上のモデル拡充
  候補・LSTM multi-output 待ち）。将来はブラウザ ONNX + PyTorch ネイティブ込みの比較
  マトリクスへ広げる（当面は不要の裁定）。測定条件の規範（graph capture ON /
  freeDimensionOverrides / IO binding / EP 分断確認・native EP か JSEP かの記録）は
  [runtime-landscape §4](research/2026-08-16-runtime-landscape.md) が正本。
- **生成イベントの横展開（需要待ち）**: sbv2 / birefnet / depth / siglip2 / vowel への stage
  イベント（step ループが無く提供できるのは段遷移のみ）と、**生成ループ**の AbortSignal 中断席
  （現状は onEvent の throw が step 粒度の中断手段 — 席は温存）。**構築経路の AbortSignal は
  anima で実装済み**（`AnimaPipelineOptions.signal` — 段境界での検査・取得層への透過・
  `signal.reason` 素通し）なので、流儀の先例はそこ。
- **`AssetProgress.path` が越境参照を識別できない（起票 2026-08-25・優先度低）**: 進捗イベントの
  `path` は文字列 1 本で、越境コンポーネント参照（ADR
  [0038](decisions/0038-manifest-v1.md) §7 追記）が入った以上**別リポの同名 path と区別が
  付かない**（取得層の同一性キーは `fileRefKey` へ移ったが、公開イベント側は `path` のまま）。
  消費側がファイル別の進捗を path でキーにすると 2 本が混ざる。埋め方は `repo` / `revision` を
  イベントへ足すか `fileRefKey` を出すか — 公開面の追加なので breaking 波に乗せる。
- **anima 大解像度の省 RAM タイル逐次組み立て（起票 2026-08-24 裁定）**: VAE decode のタイルを
  貯めずに順次合成できれば、E-2 の入口拒否で受理集合から外した 8 解像度
  （1456/1488/1584/1648/1680/1776/1840/1936px）を受理へ戻せる。
- **hub の sha256 同一ファイルのリポ跨ぎ重複 DL 解消（外部フィードバック提案⑥・優先度低）**:
  content addressing でキャッシュを引けば、同じバイト列を別リポから取り直さずに済む。
  ADR [0038](decisions/0038-manifest-v1.md) のキャッシュ設計（キーは URL）へ踏み込む変更。
- Metal 数値差の原因確定（known-issues）・resident 経路の診断/計測制約の解消。
- MoE の seam（fixed-k routing は静的形で表現可 — dense API に expert 非存在を焼かない）。

## release — リリース準備波（しばらく先）

- **R1 の残り: ロード面 API 工事 4 件 + exporter 自動分割規則**（manifest の shard 欄自体は
  波 K で消化 — ADR [0071](decisions/0071-manifest-v3-shards.md)。凍結が要る資産側の形は
  済み、コード API は公開後も動かせる）。
  **同席裁定（2026-08-19 全体レビュー — Codex 提案の採用）**: ①shard identity
  （`{id, bytes}`）の hub↔runtime 境界保存 ②`prepareModel(graphShard) → estimate →
  createSession(weightShards)` の 2 段境界（admission を重み DL 前へ — ADR 0070 の
  graph-first に沿う）③estimator のシナリオ別報告 + `peakAccountedBytes` 改名
  ④重み常駐の判別 union 化（`ResidentWeight` — 3 並列 map の統合）。
  exporter 側 shard 分割規則（co-shard 保証）は最初の LLM 級配布と同時。
- 実資産 CI gate（GitHub CI はローカル資産を踏まない問題）
- HF 公開: **jvnv / irodori / anima の 3 リポは波 K-4 で公開済み**（2026-08-21）。FN は parked
  （再配布の書面根拠なし）。以後の新モデルは runbook に従う
- リポ直下 README の書き上げ・JSR npm 互換層の sideEffects 検証
- ライセンス interview（export-recipes の family 別 provenance を upstream revision 単位で
  人間確認 — 再編の release gate。**公開 4 リポぶんは波 K-4 の人間ゲートで先行実施**）
- 「semantic surface と実装済み subset の分離」方針の再裁定（attention / deform_conv2d /
  gather / conv_transpose1d / upsample_bilinear2d — 観測 subset を op 意味論にしない統一規約）
- 大型 DL 前の limits preflight（DL 後にしか limits 不足が分からない問題）

## parked（復活条件つき）

- **export-recipes の別リポジトリ分離**（ユーザー意向 2026-08-15 — 現時点では何もしない）。
  ADR 0065 案 B〈別配布物化〉のリポ版。切り出し時の論点 = `_shared/paths.py` の REPO_ROOT
  導出・runtime 適合 fixture（packages/runtime/tests/fixtures/）の共有・goldens の出力先・
  uv workspace の解体。復活 = ユーザー裁定。

- **karume-sbv2-fn の HF 公開**（2026-08-20 保留裁定 — 波 K で一時「出典表記つき公開」へ
  振れたが撤回）。upstream の書面条件 = Booth 頒布ページの「商用可・クレジット不要・マージ
  自由」のみで**再配布は未言及**・配布者の素性も未確認。復活 = 配布者への再配布可否の確認、
  またはユーザーの再裁定。ローカル配布形・カード機構は維持（上げる作業だけが保留）
- **SBV2 `adjust_word2ph` の移植**（2026-08-21 不採用裁定 — ADR 0072 決定 8）。音素数が変わる
  編集（語境界に一致しない読みの差し替え）を受けるための word2ph 再配分。参照は**上流**
  Style-Bert-VITS2 のセマンティクス（LCS 差分 + 1..6 クランプ・残差は例外）で、AivisSpeech が
  pin する fork の「均等増減で無理やり辻褄を合わせる」は採らない（黙って近似しない）。
  復活 = overlay で表現できない読み編集の実需。
- **anima の g16 評価**（2026-08-23 送り — GPTQ 適用拡大を優先するユーザー裁定。SBV2 の
  g 軸裁定はモデル系統を跨いで一般化しない〈research 2026-08-22〉ため評価自体の価値は残す。
  復活 = 素版 i4 の視認で品質不満が出た場合。主作業と衝突しない裏実行での前倒しは可
  〈同日ユーザー裁定・anima 校正リグを触る J-4 ②の着地後に流すのが安全〉）
- hub Range 並列 + prefetch — 復活 = 断片化リポの再来（perf L-3）
- params / bind group キャッシュ（ADR 0032 案 2）・GPU timestamp 推定源化・全面 f16（案γ）・
  Vᵀ+列量子化融合・2048px DiT attention メモリ工事・SBV2 NFC チップ・f32 anima 系列再生成
- by-design 制約群（rank≤4 / OOB NaN / 非有限入力 / 0 要素次元 / cancellation 粒度ほか）は
  limitations.md が正本 — 実需が出た項目だけここへ昇格させる
