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
保持本数（上限 = 導出済み計画の 8 本）ぶんの入力バッファが勘定の外に積む（レビュー所見）。`residentBytes`
を入力と分けて持つのは footprint 不変の門（ADR 0093 決定 6）が領域の総和だけを見る量だから。
MUST: 退役は従来どおり「破棄待ちへ積む → その run の flush 後に destroy」（ADR 0004）。予算に
収めるための退役を**確保の前**に行うのは、確保の後だと予算に収まる形でも一時的に「予算 + 新規」
が載るため（退役分は flush まで生きる）。

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
