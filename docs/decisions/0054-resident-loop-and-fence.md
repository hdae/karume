# 0054 — 実行ループの GPU 常駐化とフェンス集約（resident / batch enqueue / 単一フェンス run）

- Status: accepted（2026-08-13・波②〈H-5 案 A + H-1〉のユーザー承認下でメイン裁定）
- 対象: runtime（gpu/device.ts・gpu/submit.ts・runtime/executor.ts）+ models（irodori
  pipeline）。IR 仕様・エクスポータ・配布資産は無変更（再 emit 不要）。
- 需要の実測: [research/2026-08-13-host-cost-decomposition.md](../research/2026-08-13-host-cost-decomposition.md)
  — run 境界の**フェンス待ちが壁の支配項**（待ちの床 ≈11ms/本・EG bare 54ms 中 51ms・
  irodori はフェンス関連 7.5s / run 壁 8.2s・二段待ち vs 単一待ちの素の価格差 12ms/run）。

## 決定

1. **ResidentTensor — GpuContext 所有の第 4 の寿命クラス**（既存 3 クラス = 重みアリーナ /
   slot backing / run アリーナはどれも Session に閉じ、Session を跨ぐ受け渡しは必ずホスト
   経由 = フェンス 2 本だった）。**バイト列と大きさだけを持つ**（dtype / shape なし — 持って
   いない情報を捏造しない）。構築は `GpuContext.createResident` のみ（errorScope の門 —
   確保失敗は沈黙の無効バッファになるため async で囲む）。破棄は焼き込み bind group からの
   参照 0 本が条件（fail loudly — 黙って破棄すると無関係な dispatch まで巻き添えで沈黙誤値）。
2. **BatchScope + Session.enqueue — フェンス無し実行区間**。errorScope（out-of-memory +
   validation）を区間 1 対に束ね、フェンスは `finish()` の 1 本だけ。
   **追い越し不変条件の新形（ADR 0004 不変条件④の拡張）**: enqueue は末尾で必ず eager
   submit する。`queue.writeBuffer` は issue 順で queue timeline に載るため、submit 済みの
   先行 dispatch を追い越さない — フェンス無しで不変条件が保存される。受容したトレードオフ:
   失敗の帰属は batch 単位・device 消失の検出は finish まで遅延。gpuTiming とは非両立
   （fail loudly — 1 dispatch = 1 pass の timestamp が区間ぶん未回収で溜まる）。
3. **resident への書き込みは GPU copy 経由**（`copyOutputs`: 出力 slot → resident を dispatch
   と同一コマンド列へ FIFO で積む）。出力リダイレクトを採らない理由: リダイレクトは
   ping-pong（読み書き別名の validation 回避）→ 束縛識別の交代 → backing 多面化を連鎖的に
   要求する。copy 経由なら束縛識別が安定し**単一 backing でヒットし続け**、追加機構ゼロで
   閉じる（copy の GPU 代は µs 級で無視できる）。
4. **irodori DiT ループの常駐化**: 潜在・速度場・CFG 途中結果・条件 3 本を resident に置き、
   ループ全体（40 step × 1+V forward）を 1 batch へ。CFG 合成と Euler 更新は**ホストで
   組んだ小 IR グラフ**（3 + 2 ノード・エクスポータ非経由 — models/src/irodori/host/
   sampler-graph.ts）を別 Session で enqueue。**演算の結合順・変種順・引数順は TS 正本
   （host/sampler.ts）と 1 演算ずつ同型**（f32 加算は非結合 — ずれれば WAV sha256 が割れる）。
   ループが完全に静的（CFG 窓も変種集合も step 添字だけの関数）であることが成立根拠。
   gpuTiming 有効 device はホストループへ分岐（決定 2 の非両立・出力は同一 digest を実測）。
5. **数値の席 = 既定経路**（ユーザー承認済み分岐の帰結）。parity probe（135,936 要素・
   denormal / 符号付きゼロ / fma 感応系込み）: fma 収縮 0 件（op 別 dispatch で構造的に
   収縮余地なし）・符号付きゼロ完全一致・唯一の差分は「**最終出力が denormal の要素**の
   FTZ」5,670 件（GPU シェーダ算術側・実データでは実質到達不能）。実パイプラインの
   **WAV sha256 門 2 本が digest 完全一致** = 参照ケース上のビット同一を機械証明。
   denormal caveat は limitations に記録し、WAV 門が恒久の検出器。
6. **H-1 — 通常 run の単一フェンス化**: gpuTiming OFF かつグラフ出力 ≥1 の run は、readback
   copy を run 本体のコマンド列へ積み（FIFO — mapAsync の解決 = 積んだ copy の完了 = 先行
   dispatch 完了の含意）、`mapAsync` を唯一のフェンスにする。flush の onSubmittedWorkDone と
   arena.destroy の再 flush の**フェンス 2 本が消える**。arena の後始末は submit-only
   （flush-before-destroy の実体 = 未 submit のエンコードを残さない）。gpuTiming ON
   （ADR 0021 — timestamp 回収が flush に依存）と出力 0 本のグラフ（待つ相手が無い —
   無フェンスで返さない）は二段待ちを据え置き。
7. **観測面の劣化（by-design・limitations 記載）**: 常駐経路の enqueue は run アリーナも
   計測窓も作らないため `lastRun` / `lastRunTiming` が `undefined`。op 別内訳が要るときは
   gpuTiming を有効にする = ホストループで回る（壁 ≈2 倍 — 計測モードの代価）。

## 検収（2026-08-13・RTX 3080 Ti / Vulkan・全て自己実測）

- `deno task verify` 1070 passed / 0 failed（実 GPU・PNG 門 4 + WAV 門 3 + golden/latent 込み
  — **digest 全て不変**）。フェンス数・追い越し・寿命の門 15 本を新設（故障注入 5 件で
  検出力を実証: submitPending 除去 / flush 退避 / singleFence 恒真化ほか全て赤化を確認）。
- 性能（H-2 と同一ドライバ・off モード・A/B・クールダウン規約・ベースラインは `34a3e18`
  時点 = K-4a 後）: EG bare per-run **52.54 → 28.15 / 28.60ms（×1.86）** — H-1 の kill 基準
  43.8ms を大幅超過。irodori voice-clone 素の生成壁 **8.593 → 4.896 / 4.860s（×1.76）**・
  WAV sha256 は全走 `e7846ac1…` で不変。irodori はほぼ GPU 律速へ到達（全 GPU ≈4.2s —
  露出ホストの主部だったフェンス構造が解消）。数値の正本 =
  [research/2026-08-13-host-cost-decomposition.md](../research/2026-08-13-host-cost-decomposition.md) §6。
- コミット列: R1 = `c90bd43`（runtime 基盤）→ M1 = `8ab141a`（irodori ループ）→ H-1 =
  `e339cc0`（単一フェンス run）。

## 追記（2026-08-16・レビュー修正波 B1: in-flight リース）

`BatchScope` に **in-flight リース**を導入した。`Session.enqueue` の**同期区間**（`#chain` に
積む前）でリースを 1 本取り、enqueue 本体の `finally` で返す。`finish()` は「新規 enqueue の
拒否 → 未返却リースの全返却を待つ → 未 submit を出し切る → フェンス 1 本」の順で進む。

理由: `enqueue()` はマイクロタスクを 1 段挟んでから本体が走るのに対し `finish()` は同期で決着
フラグを立てるため、**戻り Promise を await せずに finish を呼ぶと** ①まだ本体が走っていない
enqueue が全て `BatchScopeError` で reject し、区間は 0 dispatch のまま「成功」で決着する
（未 await の reject は unhandled へ流れ、観測点が無い）②走り出していた enqueue はフェンスと
errorScope pop の後に submit し、未完了の GPU 実行を残したまま finish が返る。①は実 GPU で
完全再現済み（レビュー Pass2 V-1）。

このリポは同種の順序契約を全て機構で閉じている（`Session.#chain` / errorScope 区間ロック /
焼き込み参照計数）ので、ここも散文契約にせず機構で閉じた。外形は不変・リース 0 の通常経路
（リポ内の全利用は await 済み）では待ちが 1 マイクロタスクも増えない。irodori 常駐経路の
WAV sha256 は不変を確認済み。

## 追記（2026-08-16・同修正波: onEvent はホスト経路）

生成イベント購読（`onEvent`）は本 ADR の常駐経路と構造的に非両立（1 batch + 単一フェンスの
区間は step の完了をホストから観測できない）。購読時は gpuTiming と同じ機構でホストループへ
分岐する — 出力はビット同一・壁時計の実測は生成全体で 7.2 → 8.6 秒（S 170）。
[limitations](../limitations.md) の該当節を参照。

## 追記（2026-08-30・レビュー修正波 A: 失敗帰属にホスト側の 1 件を加える）

in-flight リース（上の追記）は「本体が走っていない enqueue が区間から漏れる」ことは塞いだが、
**本体が走って落ちた**場合は塞がなかった。`enqueue` の本体（ホスト側の検査・レシピ構築）が
throw すると、リースは `finally` で返り、区間は `finish()` まで進んで**成功で決着**する —
errorScope に載るのは GPU 側の失敗だけで、ホスト側の throw は enqueue の戻り Promise にしか
出ないためである。非 await で積む用途（この ADR が想定する常駐ループそのもの）ではその
Promise を握っていないので、実態は「dispatch を 1 本落とした区間が成功を名乗り、失敗は未処理
拒否として抜ける」。

そこで `BatchScope` は区間で**最初に**起きたホスト側の失敗を 1 件だけ記録し、`finish()` が
それを投げる。errorScope 側の失敗もあるときは errorScope 側を（包み直さず）投げて、記録は
`cause` へ載せる。2 件目以降を捨てるのは、1 件目に引きずられた派生失敗が並ぶと根因が読めなく
なるため（errorScope が internal / out-of-memory を優先するのと同じ判断）。

`finish()` から `ExecutionError` 等が出るのは 0.7.0 までに対する**破壊的な挙動変更**で、
[limitations](../limitations.md) に節を持つ。同じ失敗は enqueue の戻り Promise 側にも従来どおり
出る（1 つの事実が 2 経路で見えるのは `run` と同じ）。区間の外形・待ちの増減は不変。
