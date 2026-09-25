# 0005 — 検証戦略（Deno GPU CI を一級成果物に）

- Status: accepted（2026-08-01）
- 根拠資料: recon §1/§8/§9-2。先行実験プロジェクト（以下プロトタイプ）最大の検証の穴は
  「WGSL カーネルの数値検証が自動テストに無い」ことで、プロトタイプ自身が Deno CI を
  本筋の解と明記。本環境（Deno 2.9.4 + RTX 3080 Ti、
  フラグ無しで WebGPU 動作、shader-f16 / timestamp-query 有効）で実行可能なことを実測済み。

## 決定

検証は三段 + 受け入れの構成とし、**実 GPU の数値検証を CI（`deno test`）に含める**。

1. **WGSL スナップショット** — codegen 決定性（バイト単位同一）を固定する。
2. **CPU 参照実装との allclose** — 判定は常に `|x−y| ≤ atol + rtol·|ref|`
   （相対誤差のみは禁止）。NaN / Inf は不合格。GPU 不要でどこでも走る。
3. **実 GPU golden 突合**（Deno + 実ハードウェア）— op 単位・グラフ単位で torch 由来の
   ゴールデンと突合する。op を 1 個足すたびにここへテストを足すことを実装契約にする。
4. **実ブラウザ受け入れレーン** — examples/ 側のハーネスで手動実行（自動化しない）。
   Dawn ≠ wgpu の実装差（丸め・features）はここで受け止める。

### 規律（プロトタイプから継承）

- 全ケース SKIP は明示 FAIL（無音の見かけ成功を防ぐ）。
- エクスポータの emit 集合 ⊆ ランタイム実行可能集合を**dtype 組込みの契約テーブル**で
  突合する（op 名のみの突合は f32 sum 事故を見逃した）。
- 「型が通った」≠「同じ数値が出る」— 移植・カーネル変更は必ず段 3 を通す。
- GPU テストは環境により縮退する（アダプタ無し環境では段 3 を SKIP と明示表示するが、
  リリース判定は段 3 緑を必須とする）。

## 帰結

- CI マトリクスに「バックエンド × dtype 経路」の緑条件を定義する必要がある
  （lavapipe 等ソフトウェアアダプタは f16 を f32 計算するため、実 HW レーンと区別する）。
  具体値は M0 実測後に追記する。
- 追記（2026-09-05）: リリース判定機は `shader-f16` と `timestamp-query` を列挙するアダプタで
  あること。列挙しない機で verify を通すには `KARUME_ALLOW_NO_SHADER_F16=1` /
  `KARUME_ALLOW_NO_TIMESTAMP_QUERY=1` を意図表明として設定する（既定は
  `packages/runtime/tests/gpu_gate_test.ts` が全 SKIP を FAIL にする）。実資産不在も同形で
  `assets_gate_test.ts` が FAIL にし、opt-out は `KARUME_ALLOW_NO_ASSETS=1`。これらを設定した
  環境の緑はリリース判定に使わない（[release-runbook](../release-runbook.md) §1）。

## 追記（2026-09-20）— テストレーン（実行単位の分割）

フル `deno test -A` は 30 分を超え、その大半は**モデル系列ごとの e2e**（`e2e_*_test.ts` 39 本 —
実重み・実 GPU）に集中する。1 系列に閉じる変更のたびに他系列の e2e まで払うのは、時間の 9 割を
関係の無い検査に使うことなので、`deno.json` の task を実行単位に分けた。**分けたのは実行単位
だけで、検証戦略（上の 4 段と規律）は変えていない** — フル verify は横断変更とリリース前の
最終確認として今のまま残る。

- **粒度**: `test:core`（runtime / hub / 共通層 + tools + examples）と `test:models:<系列>`
  （anima / sbv2 / irodori / gemma4 / gemma4-qat / birefnet / depth-anything / siglip2 /
  vowel-detector / minicpm5 / embeddinggemma / deberta / dacvae）。系列名は配布形の綴りに揃える
  （テストファイル名もそれに追随させた — `gemma_*_test.ts` → `gemma4_*_test.ts` 他）。
- **門番は core だけ**: `gpu_gate` / `assets_gate` / `distribution_gate` は全 SKIP を FAIL に
  する門で、系列とは無関係に配布形を要求する（`distribution_gate` の射程は、2026-09-25 裁定で**公開済み 10 リポの
  全ミラー** + `karume-gemma4-qat`〈未公開だが QAT の公開入口 e2e と融合ヒット数の検査が根にする〉へ広げた。
  配布形ミラーを根にする実資産 e2e と、`assets_fusion_counts_test.ts` の融合ヒット数の配布形節〈anima /
  anima-extra / irodori / gemma4 / gemma4-qat〉の無音 SKIP を塞ぐため。e2e がまだ読まないミラーも載せる —
  e2e を足した日に門番の更新を忘れても無音 SKIP が戻らないように。未公開の vowel-detector は配布形を作ってから載せる。
  見るのは manifest が `karume/5` であることと既定選択の part の実在・長さで、失敗文言は各ミラーの作り方
  〈`tools/export-recipes` の `dist.py`・越境参照で焼く anima-extra は release-runbook〉と opt-out
  `KARUME_ALLOW_NO_DISTRIBUTION=1` を案内する。2026-09-24 までの射程は gemma4 系 2 本だけだった）。
  系列レーンに同梱すると、その系列と関係の無い資産の不在でレーンが赤くなる。
- **被覆の門**: レーン分割の唯一の危険は「テストを足したのにどのレーンにも入らない」形で、
  これは無音で通る（レーン実行では一度も走らず、フル verify でしか現れない）。
  `packages/runtime/tests/verify_lanes_test.ts` が `deno.json` の task 文字列を真実源として
  ①core ∪ 全レーン = リポの全 `*_test.ts` ②core と系列レーンは互いに素 ③各レーンは 1 本以上、
  を見る。task 文字列が想定の形（`deno test -A <対象…> [--ignore=<glob,…>]`）から外れたら
  推測せず throw する。
- **系列レーンどうしの重複は許す**: 通常配布形と QAT 配布形の両方を読むファイル
  （`e2e_gemma4_ple_gpu_test.ts`）は gemma4 と gemma4-qat の両レーンに入れる。被覆漏れは
  許さないが、重複は 2 度払うだけで結論を変えない。
- **リリース判定は従来どおりフル verify の緑**（レーンの緑の寄せ集めでは代えない — レーンは
  互いに素だが、同一プロセスでの相互作用と実行順はフルでしか見ていない）。

## 追記（2026-09-20）— 参照門（sha256 参照値が無いままの全 SKIP を FAIL にする）

sha256 参照値は**環境キーごとの行**になった（ADR [0106](0106-device-keyed-references.md)）ので、
「この機の行がまだ 1 件も無い」状態が新しく生まれる。そのとき sha 門は全ケース明示 SKIP になるが、
これを無音の緑にすると「検証していないもの」を「検証済み」と誤読させる — 上の規律「全ケース SKIP は
明示 FAIL」そのものの形なので、**門番の並びに参照門を加える**。

- **参照門**（`registerReferenceGate` — sha256 参照値を持つ e2e 3 本の末尾に 1 本ずつ）: 現環境の行が
  1 件も無く、かつ作るモード（`KARUME_REFERENCE`）でもないなら FAIL。opt-out は
  `KARUME_ALLOW_NO_REFERENCE=1` で、`KARUME_ALLOW_NO_SHADER_F16` などと同じ意図表明の席。
  GPU も実資産も無くて sha 門自体が走らない環境では、この門番も鳴らさない。
- **レーンとの関係**: 門番 3 本（`gpu_gate` / `assets_gate` / `distribution_gate`）は core にしか無いので
  **系列レーンを単独で回すと走らない**。一方、参照門は系列ごとの e2e ファイルに同梱されるので、
  `deno task test:models:<系列>` でも必ず一緒に走る（その系列の参照値の有無はその系列のレーンが見る）。
