# TurboQuant recon — karume への適用可否（2026-09-02）

> **性格**: 時点スナップショット（2026-09-02・調査のみ・実測なし）。backlog 次波④「TurboQuant recon
> スパイク」の調査段。調査 Workflow（掃引 3: 論文一次情報 / karume の量子化リグ / 二次情報 → 深掘り 2:
> 重み / KV → 反証検証 2・全 8 主張 holds）の統合。数値は論文と既存 research の引用で、本記録での新規
> 実測は無い。

## §1 論文の事実（一次情報・arXiv:2504.19874v1・ICLR 2026 採択）

- **中身**: 密なランダム回転 Π（i.i.d. 正規行列の QR・全ベクトルで共有・1 回だけ生成）→ 回転後の各座標が
  従う Beta 分布に対して Lloyd-Max で事前に解いた**固定コードブック**で座標ごとにスカラー量子化。内積を
  不偏にしたいときだけ 2 段（b−1 bit の MSE 版 + 残差への 1 bit QJL + 残差ノルム 1 スカラー）。
- **前提**: 単位球（L2 ノルムは f32 で別格納して復号後に再スケール）。キャリブレーション不要
  （data-oblivious）。2 のべき乗次元の要求は無いが、高速変換の構成も計算量解析も無く回転は密行列積
  O(d²)/ベクトル。
- **理論**: D_mse ≤ (√3π/2)·4^(−b)（Shannon 下界の高々 2.7 倍）。
- **実験**: KV キャッシュ（Llama-3.1-8B / Ministral-7B・head_dim 128・GQA・4k〜104k・LongBench-E 3.5bit で
  Full Cache と同値）と ANN 検索。**重み量子化は射程外**。
- **訂正 2 点**（二次情報の多くが誤り）: ① PolarQuant は構成要素ではなく比較ベースライン（Google 公式
  ブログの表現が論文と食い違う）② **公式実装は存在しない**（GitHub の同名 675 件は全てコミュニティ実装）。
- 二次情報の逆風: vLLM での評価（Red Hat）は全変種でスループット 73〜80%・3bit で推論系 ≈20pt 低下。
  QJL 2 段は独立実装が軒並み既定オフ（戻すと cos 0.69 まで崩れる再現報告）。

## §2 重み量子化への適用（判定: partial — 「TurboQuant」は載らないが、回転という 1 軸だけ持ち込める）

- karume の重み量子化へ移るのは**回転だけ**。ノルム → group absmax・d → g=32・2 段 → 不採用、の 3 点で
  論文とは別物になる。方式名は「群内直交回転 + 固定表」と実体で書く（TurboQuant と綴ると誤帰属）。
- **格子側の新規性はほぼゼロ**: 既存 7 方式（rtn / fp4 / nf4 / mxfp4 / kmeans 3 粒度）は全て「格子の
  張り方」の軸で、nf4 = 正規分位点・kmeans:shared = 実データ Lloyd。TurboQuant の表（Beta 分布の
  Lloyd-Max）は nf4 の等確率格子を MSE 最適格子へ差し替えるだけ。
- **回転側は既存に無い軸**: group absmax scale は 32 要素中 1 個の外れ値が値域を独占する構造で、群内の
  直交回転はその無駄を直接叩く（既知の incoherence 処理の論法）。
- **格納形は不変で済む**（検証 holds/high）: 群軸 = 重みの最終次元 = 縮約軸なので、g32 ブロック内の直交
  回転 H は活性側へ厳密に吸収できる（Σ x·W = Σ (Hx)·(HW)）。i4 g32 + F32 companion scale も GEMV の
  語ごとの scale 引き（`linear-gemv.ts`）も 1 バイト変わらない。ただし ①縮約軸なのは linear の話で、
  `+embed` 対象（nn.Embedding = 最終次元が gather の出力側）では吸収不能 → リグの「+embed」行は実現
  不能な行として出るので除外が要る ②Hx を作る op がグラフに増える（活性回転の新 op = IR 語彙 +
  WGSL + TS 参照ビット同一門。gemma4 E2B で独立 op なら +140 dispatch/token・生産側 op のエピローグへ
  融合すれば中立）。
- bpw は両アームとも 5.0（4·N + 32·G）で rtn / nf4 / kmeans:shared と同額（固定回転なら格納 0 bit）。
  QJL 2 段は 6.0 bpw で価格点を外すので採らない。
- 棄却記録との整合: Q-4（mxfp4 = E8M0 の切り下げ）とは無関係。Q-7（AWQ）の構造上の根拠（per-channel
  倍率 s が group scale へ吸収不可）は直交回転には掛からないが、品質所見（無校正の前処理 < 校正付き丸め）
  は掛かるので、**出荷比較の相手は素の rtn ではなく gptq-rtn**（MC5 teacher 41/48・EG cos mean 0.9829）。
- 既知の罠: wRMSE / NLL は下流品質の代理にならない（screening §4-2: nf4 は両方 rtn より良いのに greedy
  2/48 対 23/48）。Hadamard 単体は定数群を逆にスパイク化する（Rademacher 符号の前置が必須）。回転を
  torch matmul で書くと縮約順が処理系依存になり「同一入力 → ビット同一」MUST が割れる（固定バタフライ順）。

### スパイク案（規模 S・torch CPU fake-quant・GPU 0 行）

1. core `quant_methods.py` に回転つき fake-quant を 1 本（固定 Rademacher 符号 × 32 点 Walsh-Hadamard
   5 段バタフライ・符号は定数）。アーム 2 本: **rot-rtn**（回転後に既存 i4 ±7 格子 = 出荷格納形そのもの）/
   **rot-lloyd**（回転後に標準正規の Lloyd-Max 16 準位固定表）。scale は既存 group absmax。
2. core テスト（専用クラス — 逆回転後の重みは表×scale に載らないので `FIXED_TABLE_METHODS` へは入れない）:
   ビット同一・回転域で表×scale に載る・f32 往復誤差が量子化誤差より 2 桁小さい。
3. MiniCPM5 `sweep_w4.py` / EmbeddingGemma `measure_quant.py` へ各 2 レコード（方式数 7 → 9）。§2 と同じ
   指標（teacher /48・NLL・greedy /48・EG cos）。**linear 限定のみ**（+embed は吸収不能）。
4. 副指標: 群内 amax/RMS 比の回転前後分布（「回転が効いた量」そのもの — 品質差の帰属を格子側と分離する
   唯一の観測）。
5. 合格線: 同 5.0 bpw で ①MC5 greedy > 23/48（rtn 超え）かつ ②teacher ≥ 38/48 と NLL ≤ 2.9（kmeans:shared
   並）③EG cos mean ≥ 0.9829。既存 7 方式に 3 つ同時に満たすものは無い。**rot-rtn が満たせば格納形ゼロ
   変更で効く筋 → 最優先で ADR 起票**。
6. kill: 2 アームとも既存最良を 1 指標も超えない、または回転前後で amax/RMS 比の分布がほぼ動かない
   （g=32 では回転が構造的に効かない直接証拠 — 論文の理論は高次元漸近に乗っており d=32 の集中は弱い）。
7. 勝ち筋が出た場合のみ第 2 巡: GPTQ 併用（`quant_calib` の GridSpec 経路）で gptq-rtn と比較。

## §3 KV キャッシュへの適用（判定: partial — 前提条件が未成立・今は着手しない）

- **形（実測確認）**: gemma4 E2B の states = sliding 12 層 f32[1,1,512,256] + full 3 層 f32[1,1,C,512]・
  MQA（Hkv=1・8 query head が同一 KV を共有）。C=1024 で **KV 合計 24.0 MiB** = 配布重み 1,512 MiB の
  1.6%。4bit 化の節約は 20.9 MiB（検証 holds/high）。
- **文脈長の天井は VRAM ではなく RoPE 表**（`export_decode.py` ROPE_TABLE_POSITIONS=1024・読み手も
  capacity ≤ maxPosition を強制）。KV 圧縮は現状で文脈長を 1 トークンも伸ばさない（holds/high）。
- **壁時計に効く条件が未測定**: decode は壁 32.5 ms/token・GPU 22.63 ms で linear_gemv 9.73 ms +
  フェンス床 ≈11 ms が支配。attention_state_qk の実測は **P≈26 の 1 点のみ**（1.44 ms / 86.16 ms・K-11 前）。
  P=1023 では KV 論理読みが 7.7 MB → 201 MB（重み読み 1.47 GiB/step の 12.8%）になるが実測は無い
  （holds/high）。K-11 の教訓（真因は帯域でなく barrier のレイテンシ露出）が state-attention にも当て
  はまるなら、バイトを 1/8 にしても時間にならない。
- **難所は回転ではなく格納形**: q 回転は重み MAC の 0.54%・V の逆回転は o_proj へ焼けば 0。効くのは
  ①state dtype 語彙が f32 のみ（f16 は予約席・`ir.ts`）②ベクトル毎ノルムの companion 席が無い ③非一様表は
  runtime に前例ゼロ（Q-2/Q-3 🚧）④読み書き同式 MUST が 4 カーネル（append / QK / stats / PV）に跨り、
  ずれは「例外も NaN も出ない沈黙誤読」。門の受け皿は ADR 0058 の opt-in 3 点門（holds/medium）。
- **着手条件（順序固定）**: 段0 = 既存 census 経路で P=26 / 256 / 1023 の attention_state_* の GPU 比率を
  採る（gpuTiming ON/OFF を混ぜない）→ 段1 = 帯域律速か latency 律速か（n_live 比例性・L2 常駐形）→
  **P≈1000 で ≥25% かつ帯域律速のときだけ** Python fake-quant（MSE 単段・b∈{4,3}・full 3 層のみ / 全層・
  **f16 KV を対照に入れ、f16 を超えた分だけを価値とする**）→ 合格時のみ WGSL（full 3 層・opt-in 席）。
- kill: P≈1000 でも <10% / latency 律速 / 4bit で golden token 列が 1 ケースでも割れる / f16 KV 単独で目標
  利得の 7 割が取れる / QJL を戻さないと合格線に届かない。

## §4 結論と提案

| 対象                  | 判定                              | 次の一手                                                                                                            |
| --------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| 重み（screening rig） | partial — 回転 1 軸だけ持ち込める | **スパイク実施を推奨**（S・CPU のみ・2 アーム・合格線 §2-5）                                                        |
| KV キャッシュ         | partial — 前提未成立              | 今は着手しない。段0（P 依存の attention 比率）は capacity 拡大の P 依存実測と同じ台本で採れるので、そこへ同乗させる |

- perf-ledger への起票候補: Q 系に「群内直交回転（rot-rtn / rot-lloyd）」を 1 行（採否 = 上の合格線・
  復活条件 = g 軸拡大〈g64/g128 で回転の利得は増える見込み〉）。KV 側は L 系（decode 長文脈）に
  「KV 量子化の着手条件 = P≈1000 で attention_state_* ≥25% かつ帯域律速」を 1 行。
- 未確定点: ①g=32 で回転が構造的に効くか（amax/RMS 比で先に見える）②rot-rtn が勝った場合の出荷起票を
  Q-2/Q-3 の送り裁定とは別件にしてよいか（活性回転の新 op が要る）③w4a8 経路（活性 i8・group 境界
  flush）との非対称を許すか ④TTS / 画像系への横展開（§3 の全滅帯は格子側の所見で、scale 効率を変える
  回転は射程外の可能性）⑤capacity > 1024 の製品要求（決まらない限り KV バイトは律速にならない）。
