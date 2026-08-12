# Irodori-TTS v4 量子化 recon（2026-08-12）

> 性格: **時点スナップショット**（網羅レビュー並行の recon レッグ・opus/medium・読み取りのみ）。
> サイズは配布形 safetensors ヘッダの実測、適格判定はランタイム plan.ts と同じ規則で機械判定。
> 系列構成の裁定と実装は量子化波の ADR が正本になる（本書は判断材料の記録）。

## summary

Irodori-TTS v4 は現在 f32 単一系列（配布 3,438,182,144 B = 3.44 GB / 3.20 GiB）で、export_irodori.py には --dtype 軸が一切無い（export_sbv2.py の WEIGHT_DTYPES / _fake_quant 相当がゼロ）。一方で量子化の受け皿は既に全部揃っている: 実測した 8 グラフ全ての重みバイトの 99.9% が「重みスロットのみ消費」= f16/i8 格納の適格対象、DiT の linear 317 本は k∈{32,192,512,768,1280,3680} で全て k%4==0 かつ ≪2^17（w8a8 適格）、dist.py の IRODORI_QUANTS は 1 席空いているだけ、models 側は quant.session を dit の Session にだけ渡す配線が既に入っている。したがって不足は「エクスポータの dtype 軸 + fake-quant 済み golden の再生成 + 品質門の設計」だけで、ランタイムは 0 行。サイズ実測は f16 1.72 GB（50.1%）/ i8 0.87 GB（25.2%）、DiT 単体は 1.464 → 0.732 → 0.368 GB。品質門は WAV sha256 門が原理的に流用不可（門自身が tolerance 化・参照差し替えを MUST NOT と宣言）なので、ADR 0027/0029 の「系列パラメタ化 + 系列ごと tolerance 独立導出 + 系列×格納dtype 集合等値検査」を latent 門へ写すのが唯一の筋。</summary>
<currentState>["配布形の quants は f32 1 席のみ・全 8 グラフが f32 単一 dtype: models/karume-irodori-v4-small/karume.json（\"quants\": {\"f32\": {...\"session\": {}}} / \"defaultQuant\": \"f32\"）","export_irodori.py に dtype 軸が無い: CLI 引数は --model-dir / --source-dir / --out / --sym-max のみ（tools/exporter/export_irodori.py:1910-1919）。fake_quant / round_weights_to_f16 の import もゼロ（grep 0 件）。default_out_root(model_dir) も dtype を取らない（同 :1900）","対照: export_sbv2.py は WEIGHT_DTYPES=(\"f32\",\"f16\",\"i8\")（tools/exporter/export_sbv2.py:91）+ default_out_root(model_dir, dtype) で系列を分け（同 :94-106）、_fake_quant(dtype, module, target) を各ターゲットの参照採取直前に呼ぶ（同 :624, 716/766/845/900/949）。--dtype は emit 専用で --verify と併用不可（同 :1250-1257・ADR 0027 決定 3）","dist 側の quant 表は 1 席だけ空: tools/exporter/karume/dist.py:1444 `IRODORI_QUANTS = {\"f32\": {\"weights\": {}, \"session\": {}}}`。SBV2 は同ファイル :949-957 で f16 / w8 / w8a8 の 3 席（w8a8 は session {\"linearCompute\":\"i8a8\"}）、Anima は :671-690 で 6 席（i8a8 / attentionCompute / attentionScoreStorage 込み）","models 側の実行形ノブ配線は完成済み: packages/models/src/irodori/pipeline.ts:43「quant の `session` は `dit` の Session にだけ渡す」/ :257-263 toSessionOptions が linearCompute / attentionCompute / attentionScoreStorage を 1 キーずつ写す / :479-480 ditSessionOptions / :607 `toSessionOptions(quant.session)`。→ w8a8 を足すのに TS 側は 0 行","Session は逐次 1 本（packages/models/src/irodori/pipeline.ts:425-452 withSession が finally で dispose・fromAssets は Session を 1 本も張らない :524）。常駐ピークは「最大の 1 グラフ」= dit 1.46GB（f32）で決まる","DiT の attention は融合 attention op ではなく bmm 24 + safe_softmax 12（IR 実測）。ADR 0030 の attentionCompute:\"i8a8\" は **DiT には効かない**（融合 attention op を持つのは backbone 25 / speaker 8 で、どちらも 1 生成に 1 回しか走らない）。融合門の期待値も dit は silu 17 / identityExpand 48 のみ（packages/runtime/tests/assets_fusion_counts_test.ts:186-227）","量子化対象の型は QUANT_CHANNEL_AXES = {Linear:0, Conv1d:0, Conv2d:0, ConvTranspose1d:1, Embedding:0}（tools/exporter/karume/quantize.py:91-99）。ランタイム側 WEIGHT_SLOTS / WEIGHT_CHANNEL_AXES も同じ 5 op（packages/runtime/src/ops.ts:252-275）で、conv_transpose1d の軸 1 まで実 GPU テスト済み（packages/runtime/tests/gpu_i8_weights_test.ts:168-192 / gpu_f16_weights_test.ts:276）","patch_irodori.py が functional で呼ぶのは rms_norm / silu だけ（:199, 220-222）で、量子化対象の Linear/Conv は実モジュールのまま。DACVAE の weight_norm は convert_dacvae.py が変換時に焼き込み済み（同 :14）→ ADR 0027/0029 の「remove_weight_norm の後・参照採取の前」順序 MUST は、irodori では「load_* 直後・golden 採取前」に単純化される","golden 生成器 irodori_pipeline.py は ex.load_backbone / load_projector / load_speaker_encoder / load_duration_predictor / load_dit を呼んで full-loop latent golden を書く（:592-622, 587）。--dtype は無い（:743-747）","事実訂正: ACTIVE_DESIGN.md:90 の「f32 3.07GB」は**第 3 波（codec 前 6 グラフ）の値**（実測 3,067,373,004 B）。codec 2 本が入った現配布形は 3,438,182,144 B = 3.44 GB / 3.20 GiB"]

## sizeBreakdown

- 実測方法: 配布形 8 本の safetensors ヘッダを直接パースし、**metadata**.karume_ir の nodes から「各 initializer の消費が WEIGHT_SLOTS の重みスロット位置だけか」を判定（ランタイム plan.ts:264 の適格判定と同じ規則）。f16 は適格バイト×1/2、i8 は ×1/4 + per-channel scale（出力チャネル数×4B）で試算
- dit: f32 1,463,747,360 B（tensor 1,463,367,184）→ f16 732,149,264（50.0%）→ i8 368,085,648（25.2%）。適格 1,462,435,840 B は **全て linear**（317 ノード）。削減の絶対量が最大（i8 で −1.095 GB）
- backbone（ModernBERT-ja 25 層）: f32 1,260,268,120 → f16 630,912,000（50.1%）→ i8 317,747,200（25.2%）。適格内訳 linear 943,718,400 / embedding 314,572,800（= 語彙表も i8 適格・per-row scale）。i8 で −0.942 GB で第 2 位
- codec_decoder: f32 261,450,332 → f16 130,758,084 → i8 65,501,352。適格は conv1d 119,376,512 + conv_transpose1d 141,852,672（convT 4 本は軸 1 の per-channel）。−0.196 GB
- speaker: f32 242,524,180 → f16 121,462,836 → i8 61,259,572（適格は全て linear 241,926,144）。−0.181 GB
- codec_encoder: f32 109,358,808 → f16 54,689,536 → i8 27,428,544（適格は全て conv1d 109,184,768）。−0.082 GB
- duration: f32 87,169,416 → f16 43,623,432 → i8 21,979,148（適格は全て linear 87,035,904）。−0.065 GB
- text_proj / caption_proj: 各 6.83 MB → f16 3.42 MB → i8 1.72 MB（linear 3 本ずつ）。桁が 2 つ小さく、単独では量子化する動機にならない
- 合計: f32 3,438,182,144 B（3.44 GB / 3.20 GiB）→ **f16 1,720,435,472 B（1.60 GiB・50.1%）** → **i8 865,450,296 B（0.81 GiB・25.2%）**。i8 の scale オーバヘッドは全体で 0.16%（ADR 0019 の実測 0.4〜0.9% より小さいのは linear 支配のため）
- 適格率は全 8 グラフで 99.8〜99.9%（不適格は bias / rms_norm weight / Snake の alpha / sym_prefix_slice が食う定数表 = backbone の 1.77MB が最大）。**「f16 指定なのに適格 0MB」型の事故が起きる余地は構造上ほぼ無い**
- w8a8 の適格性（ADR 0025 決定 1 の k%4==0 かつ k≤2^17）: dit の linear k ヒストグラム = 32×1 / 192×72 / 512×49 / 768×24 / 1280×159 / 3680×12 → **317 本すべて適格**（最大 k=3680 は上限の 1/35）。backbone 768×75 / 3072×25、speaker 128×1 / 768×56 / 1996×8、duration 512×4 / 768×3 / 1024×10 も全て 4 の倍数

## gateDesign

- **WAV sha256 門（packages/models/tests/e2e_irodori_wav_test.ts）は量子化変種に流用できない** — 門の docstring 自身が「割れたら tolerance 化も参照値の差し替えも禁止」を MUST として掲げ、参照 digest は f32 / 参照環境専用と宣言している。量子化変種は 1 ビット残らず動くので、**この門は f32 quant 専用のまま据え置き、変種は別の門を持つ**のが唯一整合する形（ADR 0049 決定 5 の「latent 門と WAV 門の併存」をもう 1 軸伸ばす）
- 写す先は latent 門（packages/models/tests/e2e_irodori_latent_test.ts）: 現行は Z_ATOL 5e-3（素実測 7.9e-4 の 6.3 倍・:77）で golden の初期ノイズを注入し z を全要素突合、**S と forward 数は完全一致**、加えて実効ノブ drift 検査。ADR 0027/0029 が SBV2 で確立した型 =「①系列パラメタ化 ②系列ごとに tolerance を**素の実測から独立導出**（f32 と同桁に収まること自体が『golden が fake-quant 後の重みで採れている』の裏取り — 掛け忘れなら量子化誤差そのものが 3 桁上に出る）③系列×格納 dtype の**集合等値**検査（系列 root の取り違えは数値網では原理的に検出不能で、これが唯一の検出器）④i8 は scale の宣言と実体の両方を検査」をそのまま写せる
- **最大の設計論点 = S（フレーム数）の完全一致**。SBV2 w8 は発話長そのものが動いた（w_ceil 198→196・ADR 0029）。irodori は duration グラフの出力が S を決め、latent 門は S と forward 数を**完全一致**で縛っている。duration を i8 にすると門が「割れるのが仕様」になり、门の最強部分が失われる。取れる形は 3 つ: (i) **w8 席で duration だけ f32/f16 据え置き**（+65MB・SBV2 が text_encoder を全 quant で i8 固定にしている混成の前例あり）(ii) S 一致を外して z 突合だけにする（門の劣化 — 非推奨）(iii) 系列ごとに golden を焼き直して S も系列固有値として固定する（S が動いた事実を門が記録する形 — 最も正直だが「上流と同じ発話を作れているか」の意味は薄れる）。(i) を既定、(iii) を i8 の質を測り切った後の代替として提示するのが筋
- 品質の数値門は E2E とは別軸（ADR 0019 決定「fake-quant 方法論により E2E は実装誤差しか測らない — 量子化の質は別軸で測る」）。新設する measure_quant_irodori.py の設計材料: measure_quant_sbv2.py（815 行）の LSD 主 / SNR 従（波形 SNR は 10〜16dB 帯で構成の順序が入れ替わる = 位相ずれに弱い）+ 直交分解（w8-front-only 7.0dB vs w8-voice-only 13.6dB で主因を割った）+ 恒真化防止（f32 構成が既存 reference.wav とバイト一致することを毎回確認）。irodori 版の直交分解軸は **dit-only / backbone-only / speaker-only / duration-only / codec-only の 5 本**（グラフ境界が torch 側 load_* と 1:1 なので分解が SBV2 より素直）
- irodori 固有で足すべき指標: **S（フレーム数）と forward 数の一致率**（SBV2 の w_ceil 198→196 に相当する劣化軸で、LSD より先に効く）+ **秒数**。DACVAE 側は codec 単体で latent→WAV の往復 LSD を測れる（decode だけ差し替えれば良い）
- 活性量子化のシムを作る場合の注意（ADR 0029 の検出限界）: SBV2 では patch 層が functional.conv1d を直接呼ぶためモジュールフックが沈黙で取りこぼした。irodori の patch_irodori.py が functional で呼ぶのは rms_norm / silu だけで、量子化対象の Linear/Conv は実モジュール → **重み量子化はモジュール単位で安全**。活性シムだけは op 粒度（karume/act_quant.py の quantize_rows が数値正本）で当てるべき
- w8a8 を入れる場合の E2E は数値パリティ網にできない（ADR 0025 決定 6 / ADR 0026 検出限界 — 活性量子化の不連続性は encoder 1 forward・24 層で既に飽和し、GPU と torch 鏡像は「同じ分布の別標本」になる）。検出力は ①w8a8 非適用の上流段（backbone / text-proj）の厳密 tolerance ②判別帯（下限の床も要る — f32 への沈黙フォールバックは誤差が**小さく**なるので上限だけでは検出できない・ADR 0028 決定 6）③**走ったパイプラインキー本数検査**（i8a8 GEMM / quantize_rows の本数 = census・linear:v2 が 0 本）の 3 本に集中させる。DiT なら期待本数は linear 317 × forward 数の関数として書ける
- 最終裁定は毎回**聴感（ユーザー）**: ADR 0019（Anima 目視）/ 0025（w8a8 目視「別の絵だが受理」）/ 0026（DeBERTa 聴感「劣化は感じられない」）/ 0027（SBV2 f16 受理）/ 0029（SBV2 w8「崩壊はしてない」受理）と 5 件すべて同じ形。irodori も f32 / f16 / w8 / w8a8 の WAV を同一テキスト・同一 seed で並べて提出する形を最初から計画に入れる

## risks

- **S（フレーム数）ドリフト**が最大のリスク: duration を i8 にすると S が動きうる（SBV2 w8 の w_ceil 198→196 が同じ軸の実測）。latent 門の「S / forwards 完全一致」は irodori 移植の最強の等式なので、壊す前に混成表（duration 据え置き）で回避するか、壊す判断を明示的に取るかを先に決める必要がある
- **WAV sha256 門は量子化変種で使えない**（門自身が tolerance 化・参照差し替えを禁じている）。変種の最終検出器は latent 門 + 聴感に落ちるので、f32 で得ていた「ホストのグルー 1 行の変化まで 1 ビットで掴む」網が変種側には無い。変種ごとに WAV digest を焼く案もあるが、参照環境専用の門が 3 倍に増える（limitations の運用コスト）
- **golden 再生成のコストが重い**: irodori_pipeline.py の full-loop golden は dit を 40〜100 forward 回す torch 実行で、それを系列ごと（f16 / i8）に焼き直す。加えて export_irodori.py の per-target golden（E2E 19 + 29 + 36 件 + codec）も系列ぶん増える。outputs/series/ のディスク消費も +2.5GB 級
- **codec（conv 支配 261+109MB）の i8 は前例が無い**: Anima は VAE i8 を「−24.6MB で桁が違う」として見送り、SBV2 dec の i8 は全鎖 w8 の測定に埋もれた。irodori では decoder 261MB が単独で意味のある削減量なので、初めて conv 支配ネットの i8 品質を単独で測る必要がある。DACVAE 特有の Snake（sin の引数 22.7π 級・f32 で既に 5e-6 級誤差を観測）が量子化とどう相互作用するかも未知
- **w8a8 の速度利得が未実測の推定である**: irodori では op 別 GPU 時間内訳が取られていない（op-timing-stats は Anima / SBV2 のみ）。DiT の bmm 24 + safe_softmax 12 は量子化経路が無く、S=750 では scores が 129.8MiB（perf-ledger K-5）と大きいので、linear を 2.8× にしても GPU 合計が 1.4× 止まりの可能性がある。**波 3 に入る前に DiT の linear/bmm 比を実測すべき**
- **融合 matcher の沈黙劣化**（ACTIVE_DESIGN の Pitfall）: 格納 dtype を変えてもノード列は変わらない設計なので原理的には影響しないが、assets_fusion_counts_test.ts の期待値は f32 資産にしか掛かっていない。系列を増やすなら融合門も系列パラメタ化しないと、変種側だけ silu 17 / identityExpand 48 が外れても誰も気づかない
- **系列 root 取り違えは数値網で検出不能**（ADR 0027/0029 の検出限界が f16・i8 の両方で再現済み）。系列×格納 dtype の集合等値検査を最初から入れないと、f16 系列に f32 が混ざった配布形が全門を素通りする
- **Metal / 他バックエンド**: 量子化変種の sha256 参照門は参照環境専用（limitations 明文化済み）。加えて Metal には attention i8a8 / conv2d の 2 経路一致の既知誤値（known-issues）があり、w8a8 を既定に据えることは（そもそも既定 f32 MUST なので）しないが、配布形の defaultQuant を w8 系に動かす判断は別途要る（SBV2 は defaultQuant=w8 にしている前例あり）
- **ドキュメントの数値ずれ**: ACTIVE_DESIGN.md:90 の「f32 3.07GB」は codec 前 6 グラフの値（3,067,373,004 B）で、現配布形は 3,438,182,144 B（3.44 GB / 3.20 GiB）。量子化 ADR を書くときの基準値としてそのまま引くと 12% ずれる

## options

### a) f16 系列のみ（最小波）

内容: export_irodori.py に --dtype {f32,f16}（WEIGHT_DTYPES + default_out_root の dtype 接尾 + 8 ターゲットの参照採取直前 round_weights_to_f16）/ irodori_pipeline.py に同じ --dtype（full-loop golden を fake-quant 重みで焼き直し）/ dist.py の IRODORI_WEIGHTS を dtype 2 枝へ + IRODORI_QUANTS に f16 席 + STORAGE_REQUIREMENTS に F16 / E2E 4 本を系列パラメタ化して tolerance を素の実測から独立導出 + 系列×格納 dtype 集合等値検査。
利得: 配布 3.44 → 1.72 GB（50.1%）・DiT 常駐 1.46 → 0.73 GB・backbone 1.26 → 0.63 GB。ロードは DeBERTa 実測（197.7→62.3ms・i8 で 3.2×）から f16 で 2× 前後。**計算速度の利得はゼロ〜微**（ADR 0026「w8 は計算を速くしない」と同型）。
工数: 中（エクスポータ配線は SBV2 の型をそのまま写せる。重いのは golden 再生成 — dit だけで 40〜100 forward × ケース数を torch CPU で回す）。
リスク: 低。SBV2 f16 実測は SNR 40.5dB / LSD 0.36dB / 発話長一致（ADR 0029 の表）で事実上透明。DACVAE の Snake（sin の引数が 22.7π 級）だけは f16 丸めの前例が無い未知点。

### b) f16 + w8（a32）の 2 系列・混成 quant 表

内容: a) に --dtype i8（fake_quant_int8 + scale 台帳を emit へ）を足し、quants を f16 / w8 の 2 席にする。**w8 席は混成**にするのが肝: dit / backbone / speaker / codec_* を i8、**duration は f32（または f16）据え置き**（理由は下の gateDesign — S が動くと latent 門の「S / forwards 完全一致」が壊れる）。SBV2 が text_encoder を全 quant で i8 固定にしている混成の前例（dist.py:949-957 + complete_quant_weights）がそのまま使える。
利得: 全 i8 なら 3.44 → 0.87 GB（25.2%）・DiT 常駐 0.37 GB・ロード ~3.2×。duration を f32 据え置きにしても +65MB で 0.93 GB。
工数: 中〜大（a) + i8 の scale 宣言/実体検査 + 混成表の設計 + 品質測定台本 measure_quant_irodori.py の新設。measure_quant_sbv2.py 815 行のうち snr_db / log_spectral_distance / rel_rms / 直交分解の骨格は流用可だが共有モジュール化されていないので抽出か複製の裁定が要る）。
リスク: 中。SBV2 w8 は LSD 3.20dB・発話長 198→196 で「別の読み上げ」として知覚された（ADR 0029・聴感は受理）。irodori は duration が S を決める構造なので同じ軸の劣化が **門の等式**として顕在化する。codec（conv 支配 261+109MB）の i8 は前例が無い（Anima は VAE i8 を −24.6MB で見送り、SBV2 dec の i8 は全鎖 w8 の中に埋もれた測定）。

### c) b) + w8a8（DiT の linear 317 本を活性 i8 化）

内容: b) の w8 系列のバイトをそのまま再利用し、quants に w8a8 席（session: {"linearCompute":"i8a8"}）を 1 行足すだけ。**ランタイム 0 行・models 0 行**（pipeline.ts が quant.session を dit にだけ渡す配線が既にある）。必要なのは dist の席と、品質台本に w8a8 構成を 1 つ足すことと、判別帯 E2E（ADR 0025 決定 6 / 0026 決定 3 の型 = 走ったパイプラインキー本数検査 + 上流段の厳密 tolerance）。
利得: DiT の GPU 時間が下がる。前例の相場は DeBERTa（linear 86.8% 支配・m=T=512 固定）で linear 単体 2.80× / GPU 合計 1.74×、Anima DiT（1024px）で step 全体 2.25×。irodori DiT は linear 317 本が支配だが **bmm 24 + safe_softmax 12 が f32 のまま残る**（融合 attention ではないので ADR 0030 の a8 も効かない）ので、GPU 合計は 1.4〜1.8× の間と見るのが妥当（**未実測 — op 別内訳が irodori では取られていない**）。生成 12.5s / 14.0s のうち DiT が占める比率も未測定。
工数: 小（b) の上積みとしては最小。ただし単独では成立せず b) が前提）。
リスク: 中〜高。品質は Anima で「劣化ではなく別の絵」（PSNR 13.28dB）としてユーザー目視受理、DeBERTa では hidden[-3] SNR 20.7〜23.3dB で聴感受理。一方 SBV2 のシム実測では w8a8 が w8 より明確に悪い（LSD 4.08 vs 3.20）— ただしあれは conv1d 支配モデルで、conv には i8a8 経路が無いので irodori の linear 支配 DiT には当てはまらない。真の未知は「flow-matching 40〜100 step の軌道分岐が音声としてどう出るか」。

## recommendation

**c) を 3 波に分けて実施し、波 1（f16）を単独で着地させてから w8 / w8a8 へ進む**。波 1 = --dtype f16 + golden 再生成 + dist f16 席 + latent 門の系列パラメタ化（ここまでで配布 50.1% / DiT 常駐半減が確定利得）。波 2 = --dtype i8 + measure_quant_irodori.py（LSD / SNR / S 一致率 / グラフ別直交分解）→ 混成 w8 表の裁定（duration の据え置き可否をここで測って決める）。波 3 = w8a8 席（dist 1 行 + 判別帯 E2E + キー本数検査）。根拠: ①**適格率の実測が異常に良い** — 8 グラフ全部で 99.8〜99.9% が重みスロットのみ消費で、f16 50.1% / i8 25.2% が試算でなく実測ベースで確定している（ADR 0006 が名指しした「f16 指定なのに適格 0MB」型の事故の余地が構造的に無い）。②**受け皿が全部できている** — ランタイムは 5 op すべてで f16/i8 格納を実 GPU テスト済み、dist.py の quant 表は SBV2/Anima で 3〜6 席を回した実績があり IRODORI_QUANTS は 1 席空いているだけ、models は quant.session を dit の Session にだけ渡す配線が既にある。新規に書くのはエクスポータの dtype 軸と品質台本だけで、w8a8 に至っては dist 1 行で届く。③**DiT の 317 linear が全て w8a8 適格**（k∈{32,192,512,768,1280,3680} は全て 4 の倍数・上限 2^17 の 1/35）で、しかも DiT は 1 生成で 40〜100 forward 回る唯一の常駐 Session = 速度も VRAM も削減の集中点。ここを取らずに f16 で止めるのは、最も大きい 1 本を半分しか使わないことになる。④**段階分割が品質裁定と一致する** — f16 は SBV2 実測（SNR 40.5dB / LSD 0.36dB / 発話長一致）で事実上透明と分かっており、聴感裁定を待たずに着地できる。i8 以降は「S が動くか」という irodori 固有の未知（SBV2 の w_ceil 198→196 と同じ軸）が門の等式に直撃するので、測ってから混成表を決める順序でなければ設計が決まらない。⑤ 波 1 の時点で品質台本の f32 基準（reference WAV とのバイト一致による恒真化防止）が要るため、波 2 の台本コストは前倒しで一部償却される。
