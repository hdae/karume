# プロトタイプ recon 統合（2026-08-01）

> NOTE: 時点スナップショット。文中のパス・コマンド・コミットハッシュは記録当時のリポジトリ構成に基づく。

Karume キックオフ時に実施した先行実験プロジェクト（以下プロトタイプ）の全体調査の統合。
調査は Workflow 8 レッグ（Opus 深読み 6 + Sonnet 掃引 2。うち 2 レッグは初回出力異常のため
制約強化で再実行）。出典は断りなければプロトタイプのリポジトリ内相対パス。本書の目的は、
以後プロトタイプ原本を再読せずに Karume の設計判断ができる状態を作ること。

## 0. 要旨

- プロトタイプ = 純 TypeScript + WebGPU(WGSL)・WASM ゼロの汎用 NN 推論ランタイム
  （ONNX Runtime Web の Web ネイティブ代替狙い）。M0〜M7 完了。SBV2 TTS E2E、
  Anima 1024px 画像生成、f16/w8 量子化、GEMM 3×系の高速化まで実証済み。
- **Karume への移行はプロトタイプ側で決定済み**（プロトタイプ決定 0013。引き継ぎ資料は
  `docs/research/2026-08-01-deno-handoff.md`）。roadmap 次タスクの 7 番目が JSR/Deno 移行。
- 移植の技術的障壁はほぼ無い: ランタイム依存 0 本・Vite 固有構文 0 件・相対 import は
  全て `.ts` 拡張子付き・WGSL はテンプレートリテラル。`deno check` 残差 7 件のみ
  （TypedArray→BufferSource 型 6 件 + `wgslLanguageFeatures` 欠落 1 件）。
- **移行の最大動機は速度でも配布でもなく「WGSL カーネルの数値検証を自動テスト(CI)に
  載せる」こと**（deno-handoff §3）。プロトタイプ最大の検証の穴（known-issues）であり、
  op 語彙 120 個の実装はこれ無しには現実的でない。

## 1. 環境前提の更新（本セッション実測 — プロトタイプの裁定を上書き）

プロトタイプの roadmap は「Linux/WSL の Deno で `requestAdapter()` が null → 検証は Windows
ネイティブへ移す」と裁定していたが、**本開発環境では否定された**:

- Deno 2.9.4 / Linux / RTX 3080 Ti で**フラグ無しで** adapter/device 取得成功（2026-08-01 実測）。
- features 実測: `shader-f16` ✅ / `timestamp-query` ✅ / subgroups ❌（列挙なし）。
- 未検証のまま残る watchlist（deno-port-recon 由来）: `wgslLanguageFeatures` の有無、
  DP4a (`dot4I8Packed`) の実効性能、limits 引き上げ上限、浮動小数の丸め差、
  wgpu での `device.destroy()` VRAM 返却挙動（実測は Dawn/D3D12 のみ）。
- 「型が通った」≠「同じ数値が出る」（Dawn≠wgpu）。数値ゴールデン突合が移植の本体。

## 2. アーキテクチャ（4 層 + IR）

- 4 層: ① Python エクスポータ（`torch.export` + decomp table キュレーション +
  モデル別パッチ層 = 一級成果物）② safetensors 1 ファイル（グラフ JSON を
  `__metadata__` の専用キーに埋め込み）③ TS ランタイム（記号束縛 → shape 式評価 →
  メモリプラン → 時間予算 submit 分割 → 必要出力のみ readback）④ WGSL カーネル
  （実行時文字列生成 + 縦融合）。量子化のスケール計算・パッキングは Python 側で完結、TS は読むだけ。
- IR: **制御フローを持たない静的 DAG**。ループ・データ依存 shape・RNG はサブグラフ分割 +
  ホスト readback で外出し。グラフ JSON はバージョン欄 + `{symbols, inputs, outputs,
  initializers, values, nodes}`、ノードは SSA 単一出力 `{op, ins, outs, attrs}`、
  全値の `{dtype, shape}` を宣言し実行時に毎ノード照合。`allow_nan=False` で
  非標準 JSON リテラルを構造排除（受理集合をブラウザの JSON.parse に揃える）。
- shape 式言語: **`coeff·sym+offset` の一次式のみ・1 次元 1 シンボル**（decisions
  0007→0008→0010 と実モデルに押されて 3 回拡張された到達点）。シンボル名の正本は
  export 呼び出し側（torch.export の `s97` は不安定）。入力 shape に現れないシンボルは拒否。
  文法の受理実装が 4 箇所（emit / verify_ir / graph.ts / executor）に分散し人間の規律で
  同時更新している — 既知の設計負債。
- op 語彙: allowlist 凍結（EMITTABLE_OPS 53 / ATEN_HANDLERS 55 / TS SUPPORTED_IR_OPS
  ≈55、突合テストあり）。融合ノード（attention）は PLAN_ONLY_OPS として IR 語彙の外
  （optimizer だけが作り executor だけが読む）→「受理集合 ⊆ 実行可能集合」が融合追加でも不変。
- dtype: **意味論 dtype は常に f32**（i64→i32 格納、bool→u32 格納）。f16/i8 は「格納のみ」の
  概念で重みスロット限定 + 不動点降格 + スロット外流入は fail loudly。bf16 は f16 に落とさず
  u32 パックのまま `bitcast<f32>(x<<16)`（DTYPE_NAMES に bf16 は無い — 格納メタは
  `<prefix>.scale.<key>` 型の companion 命名規約による暗黙結合）。
- normalize 層（exporter）: torch.export 固有形を IR 語彙を増やさず吸収
  （RMSNorm 再融合、safe-softmax ガード除去、rank≥5→≤4 降格、split→slice、select→squeeze 等）。
- 実行時資源管理: RunArena（サイズ別プール + 最終消費者解放）、SubmitScheduler
  （時間予算チャンク分割）、PipelineCache（WGSL バイト同一性キー）、device.lost 復帰。

## 3. 実測由来の不変条件（Karume でも維持する MUST 候補）

いずれもプロトタイプが実障害から得たもの。番号は本書内の参照用。

1. **codegen 決定性**: 同一キー → バイト単位同一 WGSL。ブラウザ暗黙パイプライン
   キャッシュのキーが WGSL 文字列そのもので、明示 API は存在しない。スナップショットで固定。
2. **grid-stride 前提**: elementwise + 行 reduce 族（layer_norm/rms_norm/softmax）とも。
   1D 上限 65535 は 16head×4096token=65536 行で実際に超過した。
3. **submit 時間予算分割 + device.lost 一級**: Windows TDR 既定 2 秒 / Chromium
   watchdog（origin strike でブロックリスト化）。`onSubmittedWorkDone` は累積完了で
   解決するため計測は前チャンク完了時刻からの差分で取る（素朴実装は min 張り付き）。
   チャンクサイズ 0 は flush が無限ループ。device.lost 無視は mapAsync 永久ハング。
4. **flush-before-destroy**: 未 submit のエンコードが破棄済みバッファを参照すると
   共有 scheduler 上の無関係な dispatch ごと submit が落ち、別処理の沈黙誤値になる。
5. **requiredLimits は compute 系まで明示**（載せない limit は仕様既定値に落ちる。
   workgroupSizeX/Y/Z は invocations と別 limit）。無効パイプラインは throw せず
   dispatch no-op 化 → 出力全 0 の沈黙故障。`pushErrorScope('validation')` 常設で可視化。
   maxStorageBufferBindingSize と maxBufferSize はセットで引き上げ。
6. **バッファプールの不変条件**: ① release はノード境界のみ ② 出力 alloc は dispatch
   エンコード前 ③ useCounts は node.ins の厳密延べ計数。加えて `queue.writeBuffer` で
   書くバッファはプール外・全書きしない op（pad）は noReuse・中間値 readback は拒否。
7. **格納 dtype の流出禁止**（fail loudly）+ **bias は常に f32 で読む**（プロトタイプの f16
   降格バグの根治形。§9-3 参照）。
8. **シンボル束縛は `Object.hasOwn` 経由**: `binding[sym] !== undefined` は
   Object.prototype 由来の `toString` 等が素通りし算術が黙って NaN 化する（実バグ）。
9. **受理集合 ⊆ 実行可能集合**を突合テストで機械保証。ただし op 名突合は dtype/属性差を
   見逃す（f32 sum が実行時に落ちた実バグ）→ dtype 組込みの契約テーブルで突合する。
10. **ホスト言語で正本の意味論を再実装しない**（decisions 0005/0009/0012 の共通原理）:
    shape 部分木・文字分類・トークナイザ正規化はエクスポータが正本で全数評価して表に焼き、
    emit 時に全数同値検査で門番。動機は ICU 版差・丸め差の静かな不一致の構造排除。
11. **数値判定は `atol + rtol·|ref|`**（相対誤差のみ禁止）。NaN/Inf は不合格。
    全ケース SKIP は明示 FAIL（無音の見かけ成功を防ぐ）。
12. **融合は最適化のみ**: 正しさは常に非融合経路（bmm→softmax→bmm）が担保。
    融合 attention の受理は D≤152・D%8=0・マスク無しのみ（共有メモリ 32KB 制約）。
13. **安全側の拒否**: 負 pad（u32 巻き戻しの範囲外書込が実装依存）、conv_transpose1d
    stride=0（GPU ハング）等は exporter `_expect` + executor の二重検査で拒否。
14. **VRAM は `buffer.destroy()` では 1 バイトも返らない**（Dawn の
    PooledResourceMemoryAllocator）。`device.destroy()` のみが返す（7,801→1,648MiB 実測）。
    device / PipelineCache / スケジューラは再構築可能な構造にする（値でのクロージャ捕獲禁止）。

## 4. 性能実測とボトルネック（m6-perf-recon / vram-release）

- **GEMM が演算量の 76%**（Anima DiT 1024px、17,921 GFLOP 中 linear 13,592 + bmm 4,329）。
  カーネル変種実測: 旧 16×16 タイル 3,506 GFLOP/s → **レジスタ 4×4 + vec4 で 10,754
  (3.07×)** → DP4a 16,599 (4.73×、ただし W8A8 精度ゲート前提)。8×8 タイルはレジスタ溢れで
  劣化（3,867）。f16 計算は 1.44× で本命ではない。
- WebGPU に subgroup-matrix が無くテンソルコア不可（native torch との構造差）。
- **ロードは取得律速**: 直列 19.8MB/s → 4 並列 61.4MB/s で頭打ち（8/16 本は無意味）。
  GPU アップロード自体は 2.8GB/s。3.91GB DiT 初期化: 215.8s → 4 並列 67.3s →
  Cache API ヒット 28.5s。**次の床は GraphExecutor コンストラクタの await 無し同期
  アップロードループと f16/i8 展開の要素ごとループ** → 初期化は明示 async ステージにする。
- **IR の無駄が大きい**: DiT の expand 224 本すべて恒等（37.8GB/forward の無駄コピー）、
  permute 736 本中 336 本が連鎖（畳めば 8.6GB/forward 減）。恒等 expand は他モデルでも
  共通 → グラフ最適化パス（恒等 expand 別名化・permute 連鎖畳み込み・遅延 view）は前提機能。
- attention 融合で 2,365→5,498 GFLOP/s、peak transient 7.3× 減。
- 常駐（keepResident）はロード 4.9× 高速化 vs peak VRAM 増のトレードオフ → ホスト裁量の
  明示 API にする（ランタイムに固定ポリシーを埋めない）。
- 非 GEMM 経路は帯域律速でなく strided 座標デコード（u32 除算/剰余 ≤6 回）律速の疑い
  （contiguous-identity 特殊化で検証可能、未実施）。

## 5. op 語彙（台帳は [docs/op-vocabulary.md](../op-vocabulary.md) に引き継ぎ）

- 母集団 = **Core ATen IR の 160 op**（torch.Tag.core。overload 2879 / decomp 943 に対し）。
- 5 分類: 不要 25 / 必須プリミティブ 75（プロトタイプ実装済 ≈45）/ 合成で済む 38（新カーネル 0）/
  あったら嬉しい 7 / 困難 15。実装対象 120。
- **最大の穴 = 値の max reduce 族**（amax/amin/max/min/argmax/argmin が 1 つも無い）。
  safe-softmax・max-pool・greedy デコードの前提。SUM 型行カーネル複製 ≈20 行で済む。
- 「分解できる」の反例集（敵対検証で判明）: WGSL に log1p 無し / pow の負底未定義 /
  max の NaN 伝播非保証 / var の correction=1 / tensor 比較を sub 経由にすると inf-inf=NaN。
  → 分解案は torch 突合ゲート必須。
- 困難 15 の 3 系統: scatter 系（WGSL に atomic<f32> 無し、CAS は加算順非決定 = 決定性破壊）、
  データ依存出力形状（nonzero 等 — 静的形状前提と根本衝突）、値依存並べ替え（sort/topk/fft）。
- 分解禁止（融合維持）9 op: linear, layer_norm, softmax, gelu, conv1d, conv2d,
  conv_transpose1d, embedding, masked_fill。
- 実装順序（プロトタイプ裁定）: ①行 reduce 族 ②elementwise 一括 ③tensor-tensor 比較の一般化
  ④軸の一般化 ⑤エクスポータ分解パス集中投入(≈25 op) ⑥native_group_norm ⑦conv 契約拡張
  ⑧pooling/upsample ⑨scatter_add(ADR 前提) ⑩保留。

## 6. 量子化

- 共通意味論: **「格納のみ量子化・計算 f32」**。エクスポータが fake-quant してから
  export・ゴールデン生成 → 量子化誤差と実装誤差を分離。IR 宣言 dtype は f32 のまま。
- 方式マップ: f16 採用済 / **w8（i8 + per-channel f32 scale, `unpack4xI8`, feature 依存ゼロ）
  採用**（f32 比 ≈1/4、試聴裁定 2026-07-31）/ per-tensor i8 却下（音声品質不足）/
  素朴 RTN w4 不成立（SNR −1.5〜+5.1dB）・group-wise 機構のみ低優先 backlog /
  w8a8（DP4a）は実測ベンチ待ち / 活性 f16 却下（検証の切り分けが濁る）。
- 量子化対象の突合が `id(tensor)` 依存（torch.export の実装事情に乗った時限爆弾）→
  Karume は FQN 突合 + 期待本数照合にする。

## 7. IR v0 形式の評価（exporter レッグ）

- 単一 safetensors + `__metadata__` 埋め込みは HF 配布・Range 取得と相性が良く継承価値が
  高い。ただし: メタデータサイズ上限・巨大グラフのパース時間は未検証。
- バージョン欄が整数 1 個のみ — 語彙増減の semver 的表現・capability 宣言・
  前方互換の無視可能フィールドは未設計。
- 格納メタが `<prefix>.scale.<key>` / `<prefix>.folded.<hash16>` 型の companion 命名規約による暗黙結合。
  明示フィールド化（例: `initializers[].storage = {dtype, scale_key, group_size}`）が素直。
- verify_ir = IR の torch 素朴解釈オラクル（コンバータ意味論とカーネル実装の切り分け点）。
  受理集合を TS ローダ以上に緩めない契約。`max(worst, nan)` の NaN 吸収も対処済み。
- shape 専用部分木の定数化は「2 点評価（Tmax, Tmax−1）で可換性を毎回実測」
  （sym_max ≥ 3 強制。宣言でなく実測 — allowlist だけでは守れない）。
- B=1 / rank≤4 / 最終次元限定 reduce はカーネル都合が IR 仕様へ染み出したもの。
  汎用化するなら IR 制約でなくランタイム capability + 明確な診断として表現し直す。

## 8. 検証戦略（playground レッグ）

- 4 層ハーネス: golden（実測 vs 参照 safetensors、既定 atol=1e-3, rtol=1e-2）/
  harness（GPU vs CPU 参照の合成 op 突合、matmul は atol=2e-7·K の経験則スケール）/
  bench（timestamp-query、gpuMs=0 は wall にフォールバック）/ submit（時間予算適応の
  正当性 — 冪等式だと検証が空振りするため ping-pong 依存連鎖を使う）。
- スイート: quick（日常）/ full（全回帰）。外部ネット依存と 5GB 級は手動ボタン専用。
- 本体テスト（プロトタイプ本体パッケージの tests、183 本 3,187 行）は完全 CPU 実行:
  WGSL スナップショット + CPU 参照 allclose。**実 GPU 数値一致はこの層に無い**（→ §9-2）。
- 配信層（registry.ts）: テンソル境界 1GiB チャンク（V8 ArrayBuffer 上限 2,145,386,496B
  対策）、Range 4 並列、Cache API + validator（ETag/Last-Modified + 総バイト。無ければ
  キャッシュ不使用に倒す）、206 は put 不可 → 合成 Response、二重ロード防止 Map。
- ホスト側境界: トークナイザ・文字正規化・durations 展開・乱数は「決定的純関数」として
  GPU 実行から分離（Node/Deno で Python 参照とフィクスチャ突合可能な形）。

## 9. プロトタイプの弱点 = Karume で最初から直す候補

1. **公開 API 未設計**: index.ts が内部 ≈180 シンボル素通し再輸出で既に実装と乖離。
   テストが src/* 直 import のため乖離が検出されない。JSR + SemVer とは両立しない。
2. **WGSL カーネル数値検証が CI に無い**（最大の穴）。本筋の解が Deno CI とプロトタイプ自身が
   明記 — Karume の存在理由そのもの。
3. **f16 降格バグ**: bias の f32 折り畳み定数が weight を道連れ降格（linear 454 本、
   1024px DiT で適格 0.0MB / CPU 展開 3,731MB / ロード 32 秒 / VRAM 節約ゼロ）。
   直し方確定済み（bias を常に f32、F16_WEIGHT_SLOTS を weight のみに）。
   あわせて「f16 指定なのに適格 0MB」を検出する診断ログを常設する。
4. **次元文法の 4 重実装**（emit / verify_ir / graph.ts / executor を人間の規律で同期）。
5. **初期化の同期アップロードループ**（キャッシュヒット時 28.5 秒の床）。
6. **`id(tensor)` 依存の量子化突合**（→ FQN + 期待本数照合へ）。
7. **グラフ最適化パス不在**（恒等 expand・permute 連鎖が実測で大きい）。
8. GraphExecutor 単一 2,094 行クラス・55 op の巨大 switch（分割 vs 集中は要裁定）。
9. op 突合が dtype を見ない（§3-9）。
10. 細部: silu 死枝疑い、WGSL NaN 比較・sign(NaN)・長さ 0 スライス binding の未実測。

## 10. 未決事項（Karume で裁定が要るもの）

- IR 形式: プロトタイプ IR（v0）互換 or 非互換の v1 改訂（格納メタ明示化・bf16・capability 宣言）。
- 実行モデル根幹の再確認: 全ノード出力 shape の実行前確定（静的形状）を維持するか。
  ここが nonzero/topk/動的形状の可否を決め、後戻り不能。
- KV キャッシュの IR 表現（静的最大長 vs prefill/decode 分割 + 明示 I/O）— プロトタイプ未決のまま。
- メモリプラン戦略（RunArena 踏襲 or 静的アリーナ計画）。
- 時間予算（既定 100ms・安全率 0.5 は経験値）と数値許容誤差の根拠づけ。
- バックエンド×dtype 経路マトリクスと「CI 緑」の定義（lavapipe は f16 を f32 計算）。
- W8A8 精度ゲートの定義（DP4a 4.73× の前提）。
- scatter 系の方針（CAS 回避・静的添字の gather 反転を ADR 化するか）。
- プロトタイプ playground 責務（トークナイザ・registry・キャッシュ）の Karume での置き場。
- Deno 側: Cache API 挙動（keys() 未実装報告）・quota・wgpu の device.destroy() 挙動。
