# 0095: slot backing の複数保持 — バイト予算つき LRU（perf-ledger H-15）

- Status: accepted（2026-09-07 — ユーザー裁定: 既定予算 256 MiB を VRAM の上乗せとして許容・
  利用者が変更できる形で）
- Date: 2026-09-07
- 関連: ADR [0042](0042-prepared-execution-plan.md)（導出済み計画と slot backing — 本 ADR が
  「backing は容量 1」を改める）/ [0066](0066-generation-context-state-slots.md) 決定 5
  （context 側の焼き込み — 束の寿命が (context, backing 実体) の組であることは不変・
  本 ADR で backing ごとに 1 束持つ）/ [0089](0089-memory-limits-preflight.md)（見積り —
  本 ADR で `peakAccountedBytes` の勘定が変わる）/ [0093](0093-transient-liveness-packing.md)
  決定 6（footprint 不変の門 — backing 1 本の命題としてそのまま）/ [0070](0070-shard-loading-admission.md)
  CX-4.3（estimator の契約 — 「max の根拠 = 同時 1 本」を本 ADR が改める）/ [0004](0004-execution-model.md)
  （flush-before-destroy — 退役 → 破棄待ち → flush 後 destroy の規律は不変）
- 需要の実測: [research 2026-09-07](../research/2026-09-07-codex-perf-review-followup.md) §7.2
  （M が run ごとに変わると backing を作り直して +37〜42 ms/run・chat の 1 ターンで 2 回 ≈ 80 ms）/
  [research 2026-09-07-gemv-rows-k21](../research/2026-09-07-gemv-rows-k21.md) §8.2（K-21 後も切替
  run の作り直しは同数）

## Context

slot backing（導出済み計画にヒットした run が使う中間バッファ束 — ADR 0042）は Session あたり
**1 本**だった。理由は「slot は run の中間バッファそのもので、DiT では ~1 GiB 規模 — signature
ごとに抱えると VRAM が本数倍になる」。

生成（gemma4）では run の形が毎ターン切り替わる: prefill バケット形（M = 32 / 64 / 128 / 256）→
decode 形（M = 1）→ 次ターンの prefill 形。容量 1 だと切替のたびに旧 backing を退役させて新しい
形を確保し直し（bind group の焼き込みも context 側で焼き直し）、1 回 ≈ 40 ms・1 ターン 2 回で
≈ 80 ms を払う。TTFT（prefill run の壁 ≈ 80 ms — K-21 後）と同じ桁で、投機的デコード（K-20）では
draft 形 ↔ verify 形の交互がサイクルごとに起きるため復活条件 ② に挙がっていた。

一方、生成の形の backing は小さい（gemma4 E2B・io + workspace）: decode 3 MiB・バケット 32 / 64 /
128 / 256 が capacity 2K で 9 / 17 / 33 / 65 MiB・capacity 16K で 23 / 45 / 89 / 177 MiB。chunk 768 形
だけが 192〜528 MiB。「本数倍」を恐れる理由は DiT 級の大きい形にしか無い。

## Decision

### 1. backing は**バイト予算つきの LRU 集合**で持つ（`SessionOptions.planBackingBudgetBytes`）

`Session` は `Map<計画キー, ActiveBacking>`（挿入順 = 古い順・ヒットで再挿入 = LRU）を持つ。
新しい signature のヒット run は、計画の領域バイト（`planRecipes(...).totalBytes` — 確保の前に
分かる）を出し、**保持分 + 新規 ≤ 予算**になるまで古い順に退役させてから確保する。
新規 1 本だけで予算を超える形は保持中を全て退役させてその 1 本だけを持つ（= 従来の容量 1 と
同じ形）。したがって常駐は **`max(予算, 最大 1 本)` を超えない**。予算 0 は従来どおり常に 1 本。

既定は 256 MiB（`DEFAULT_PLAN_BACKING_BUDGET_BYTES`）: gemma4 E2B で decode + バケット 32 / 64 /
128 が capacity 16K でも収まり（capacity ≤ 8K なら全段）、chunk 768 形は 1 本だけ持つ側に落ちる。
利用者は `SessionOptions` と `Gemma4Pipeline` の options で変えられる（非負の安全な整数以外は
fail loudly）。

MUST: 勘定は各 backing が抱える VRAM = **領域の総和 + 所有する入力バッファ**（診断 `residentBytes +
inputBytes`）。領域だけで数えると「`max(予算, 最大 1 本)` を超えない」が領域についての命題に縮み、
保持本数（上限 = 導出済み計画の 8 本。訂正 2026-10-11: 12 本 — 翌日の MTP 段 1〈63a3d988〉で
`PREPARED_PLAN_CAPACITY` を 12 へ広げた。本数の正本はこの定数）ぶんの入力バッファが勘定の外に
積む（レビュー所見）。`residentBytes` を入力と分けて持つのは footprint 不変の門（ADR 0093 決定 6）
が領域の総和だけを見る量だから。
MUST: 退役は従来どおり「破棄待ちへ積む → その run の flush 後に destroy」（ADR 0004）。予算に
収めるための退役を**確保の前**に行うのは、確保の後だと予算に収まる形でも一時的に「予算 + 新規」
が載るため（退役分は flush まで生きる）。（訂正 2026-10-11: 確保前の退役が保証するのは保持集合の
勘定の上限だけで、物理的な重複は防がない — 退役分は確保の前後どちらで退役させても flush まで
VRAM に残り、新規の確保の時点で新旧が同時に載る。この窓は決定 4 の「非勘定側」に当たる）

### 2. 失敗経路は「この run が新規構築した 1 本」だけを退役させる

構築途中の例外・run の失敗では、新規構築した backing（キーで識別）だけを退役させ、保持中の
他の backing は残す。理由は容量 1 のときと同じ（無関係な失敗で ~GiB の再構築を強いない）で、
対象が「活性 1 本」から「新規の 1 本」へ言い直されただけ。ただし予算に収めるための退役は確保の
**前**に済んでいる（決定 1 の MUST）ので、予算超過を伴う失敗 run はその退役分も失う（次のヒットで
作り直す — 正しさは損なわれない）。
MUST: 導出済み計画の LRU 追い出し（`PREPARED_PLAN_CAPACITY`）は追い出したキーの backing を退役
させる（二度と当たらない signature の中間バッファを Session の寿命いっぱい抱えない — 不変）。

### 3. context 側の焼き込み束は backing ごとに持ち、退役と同時に捨てる

`GenerationContext` の焼き込み束（ADR 0066 決定 5）は「backing の世代識別子 → 束」の表になる。
保持中の backing ごとに 1 束なので、保持した形の間の行き来では焼き直し（診断
`stateBacking.rebindCount`）が増えない。
MUST: Session が backing を退役させるとき、生存中の全 context の該当束を `dropBakedGroups` で
捨てる（束は退役した実体を掴んでいるので、世代で照合していても参照ぶんの寿命が延びる）。
照合 → 焼き直し → dispatch の順を 1 箇所（`#generationGroups`）に閉じる規律は不変。

### 4. 見積り（ADR 0089）はシナリオの max ではなく **`max(予算, 最大シナリオ)`** を勘定側に載せる

保持集合の上限がそのまま勘定側の量になる。prefill バケット形（models 側の `chunkBuckets`）は
見積りのシナリオに列挙しないが予算の内側に入るので、予算で上から押さえる（過大側に倒す —
「勘定に入れた分のピーク」の意味論はそのまま）。非勘定側に残るのは、退役（予算超過 / LRU
追い出し）→ destroy の窓で退役分と新規が同時に載る点。`AdmissionReport` は予算を報告する。

### 5. 採らなかった案

- **generation 形だけ 2 本固定**（prefill 形 + decode 形）: バケットが 4 段あるので「prefill 形」が
  1 本に定まらず、バケット間の切替が残る。
- **現状維持 + バケット run の前に退役**（limitations「prefill バケット」の非勘定窓だけ閉じる）:
  +37〜42 ms/run は消えない。

## 検証

- 実 GPU 門 `gpu_plan_backing_test.ts`: 予算既定で交互に回しても `buildCount` 不変・保持本数と
  総和・LRU の順・新規 1 本の予算超過で全退役・故障注入（途中失敗）で新規分だけ返る・予算 0 で
  従来の挙動。`gpu_state_execution_test.ts`: 保持した形の往復で `rebindCount` が増えない・退役した
  backing の束が context から消え、再構築で 1 回焼き直す。見積りは `estimate_test.ts`。
- 実測（追記予定 — research 2026-09-07-gemv-rows-k21 §9）: 20 token ターンの prefill run 壁と
  定常ターン壁を K-21 直後（`423a1b8`）と ABBA。

## 帰結・残件

- 他家族（anima / irodori / sbv2 …）は manifest の `session` 経由で Session options を組むため
  `planBackingBudgetBytes` の透過は未（既定 256 MiB が効く・利用者が 0 に落とす口が無い）。形が予算より
  大きい家族（DiT 級）は 1 本だけ持つ側で従来と同じだが、数十 MiB 級の形を複数回す家族（siglip2 の
  base・depth-anything の小解像度）は複数保持側に入る（最大 +256 MiB・見積りに載る）。共通の
  options へ載せる件は backlog。
- **予算に収まらない形は 1 本だけ保持する**: gemma4 の chunk 768 形は capacity 16K で ≈ 528 MiB（capacity 2K で
  192 MiB）なので、既定 256 MiB では 768 token 以上のプロンプトを含むターンで 768 形 ↔ バケット形 ↔
  decode 形の作り直しが従来どおり起きる（短いターンの往復だけが消える）。長い prompt を毎ターン
  流す用途は予算を上げる（[limitations](../limitations.md)）。
- **常駐入力を焼き込んだ backing が保持されている間、その `ResidentTensor.dispose()` は fail loudly**
  （容量 1 のときは形を切り替えれば解けた窓が、予算内では Session の dispose か予算超過まで続く —
  [limitations](../limitations.md)）。
- 見積りの `peakAccountedBytes` は既定で `weights + state + max(256 MiB, 最大シナリオ)` になり、数 MiB
  しか保持しない小さなモデルでも 256 MiB の下限が載る（過大側 — 報告の `planBackingBudgetBytes` で
  引き算できる・[limitations](../limitations.md)）。
- 予算の既定は定数で device を見ない。4 GB 級端末での増分は最大 256 MiB（見積りに載る）。

## 追記（2026-10-10）— 決定 4 の改定: バケット形は見積りが計算してピークの最大に入れる

決定 4 の「prefill バケット形は見積りのシナリオに列挙しないが予算の内側に入るので、予算で上から
押さえる」を改める。

- 見積り（`estimateGraphMemory`）は `generation.chunkBuckets` の各物理行数で、prefill / decode と
  同じ計算をする。その `ioBytes + workspaceBytes` を `peakAccountedBytes` の内側の max に入れる
  （`max(予算, 最大シナリオ, 最大バケット形)`）。
- バケット形は `scenarios` に列挙しない。公開の列は prefill / decode と追加シナリオのまま。
- バケット形が予算の内側で他の形と同時に保持されるぶんは、従来どおり予算の項が上から押さえる。
- gemma4 の `estimateSessionMemory` は sequence が流すバケット（見積る `chunkLength` 未満の段）を
  見積りへ渡す。drafter の見積りには渡さない（借り手 context は M = 1 しか流さない）。

なぜ: 予算で押さえる前提は「中間は物理行数 M に単調で、ピークは最大 M の prefill 形にある」だった。
行ブロック分割（ADR [0060](0060-row-block-attention.md) の等分 — states 形 attention の
`planStateAttention` も同じ）は、枚数が 1 → 2 に変わる境界で行数の少ない形のほうが一時を多く使う
ので、この前提が成り立たない。予算がそのバケット形より小さい（予算 0 を含む）と、バケット形が
勘定から漏れる。許される物理行数を全て計算すれば、分割規則が変わっても仮定に依存しない。

## 追記（2026-10-10）: batch の GPU 側の失敗でも、その enqueue が新規構築した backing を退役させる

決定 2（失敗した実行が新規構築した 1 本だけを退役させる）を batch に揃える。非致命 OOM
（`createBuffer` が無効なバッファを返すだけで例外にならない）は enqueue を成功させたまま
`finish()` の errorScope で初めて判明するので、enqueue 本体の catch（ホスト側の失敗でその場で
退役させる — 従来どおり）では拾えない。

- enqueue は新規構築した backing をキーと世代番号 `build` で記録し、受理時に失敗決着の finalizer を
  登録する（generation の lease の有無に依らない）。
- 退役させるのは区間本体が失敗したとき（`BatchFailureOrigin` の `"gpu"` — errorScope が失敗を
  報告した・scope を検査せずに捨てた・device 消失）だけ。区間本体が通り、ある enqueue のホスト側の
  失敗だけで決着した batch では退役させない（他の enqueue が作った backing は有効。ホスト側で落ちた
  enqueue 自身の backing はその catch が退役させている）。
- MUST: 世代番号が一致するときだけ退役させる。キーで照合した場合と結果が分かれるのは、batch の
  失敗決着より先に後続の run が同じキーを作り直した競合（区間ロックの解放と finalizer の前後が
  microtask の段数で決まる）だけで、到達しにくい競合への防御である。
- MUST: batch の失敗決着では destroy しない。退役した実体は破棄待ちへ移し、破棄は Session の
  直列化鎖に積んだ 1 本で `#destroyRetired` が行う。決着の時点では後続 run が既にエンコードを
  始めていてよく、鎖の各本体は submit / discard を済ませて抜けるので、鎖の上が「未 submit の
  エンコードが無い点」になる（ADR 0004 の flush-before-destroy）。旧 backing の VRAM は鎖の
  1 本目で返るので、`finish()` の決着を受けてから発行した後続の enqueue では、作り直すより前に
  返る。決着を待たずに発行した後続の enqueue との前後は保証しない（区間ロックの解放と finalizer の
  前後が microtask の段数で決まる — 上の世代番号と同じ到達しにくい競合として扱う）。
- 門: `gpu_plan_backing_test.ts`（batch の GPU 側の失敗で新規構築の backing だけ退役し、直列化鎖の
  1 本目でちょうど 1 回返り、次の batch で作り直す・既存 backing に当たった失敗では作り直さない・
  ホスト側の失敗だけの決着では別の enqueue が作った backing を残す）。GPU 側の失敗は errorScope の
  pop に OOM を注入して作る（`tests/helpers/oom-injection.ts`）。

## 追記（2026-10-10）: GPU 側の失敗で、その実行単位が足した params と計画を巻き戻す

params（カーネル定数のバッファ）と導出済み計画も、確保が非致命 OOM なら無効なバッファを直参照した
まま登録される。巻き戻さないと、メモリが空いた後の再試行も同じ無効バッファに当たり続ける。

- 実行単位（run 1 本 / enqueue 1 本）ごとに、新しく載せた params（キャッシュのエントリ）と登録した
  計画を記録し、GPU 側の失敗が判明したときだけ外す（params はキャッシュから外すだけで破棄しない —
  weights アリーナが `Session.dispose` で回収する）。既存の正常なエントリは残す。
- GPU 側の失敗の判定: run は errorScope が失敗を報告したとき（pop の前に落ちた run も、捨てる
  scope の中身を見る — `discardFailureScopes` が捕捉の有無を返す）・pop 自体が落ちたとき・device
  消失（消失後の pop は null で resolve するので、検査の結果が証拠にならない）。enqueue は batch の
  `finish()` で区間本体が失敗したとき（上の追記の `"gpu"`）。
- ホスト側の失敗では外さない（run / enqueue とも）。導出の throw・入力の検査・`copyOutputs` の
  検査・読み戻しの上限の検査・generation の検査（ミス run の `assertGenerationRun` 等）が該当する。
  作れた params と計画は有効で、外すと何度でも同じく落ちる失敗（入力の誤り等）のたびに params を
  新しく確保し、dispose まで積もる。
- run の読み戻しの失敗は経路で扱いが分かれる。単一フェンス経路では staging の確保と copy が run 本体の
  errorScope に相乗りするので、その失敗は GPU 側として外す。run 本体の scope を失敗なしで通った
  後の失敗は外さない — 読み戻しの完了待ち（`mapAsync`）の失敗と、二段待ち経路の `#readOutputs` が
  自前の scope で報告する失敗（staging の確保・copy）が該当する（device 消失なら上の判定で外す）。
- backing は従来どおり（決定 2 と上の追記 — 失敗した実行単位が新規構築したものを退役させる）。
- MUST: 照合は同一性で行う（計画はオブジェクト・params は実体）。上の追記の世代番号と同じく、
  到達しにくい競合への防御である。
- MUST: 計画を外すときはそのキーの backing も退役させる（LRU 追い出しと同じ — backing は計画の
  レシピ〈= 外す params〉を焼き込んでいる）。通常の経路では、そのキーの backing は同じ失敗した
  実行単位が作ったもので既に退役している。効くのは batch の決着より先に後続の run がその計画に
  当たって backing を作った競合のときだけで、これも到達しにくい競合への防御である。その競合では
  計画の無いキーに backing が残りうるが無害（焼き込んだ params が無効ならその run 自身が落ちて
  自分の backing を退役させ、有効なら実体は dispose まで生きていて内容も同じ）。
- 門: `gpu_params_cache_test.ts`（GPU 側の失敗の run / batch が足した params と計画だけを外し、
  既存の計画は残す・入力の誤りで落ちる enqueue を繰り返しても params の確保数
  〈`diagnostics().weights.allocCount`〉が増えず、計画も残る）・`gpu_state_execution_test.ts`
  （ミス run が `assertGenerationRun` で落ちる run を繰り返しても params の確保数が増えない・同じ
  run で捨てた errorScope が非致命 OOM を捕捉していれば計画と params を外し、次の run で作り直す）。
