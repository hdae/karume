# shard サイズ × 転送の刻み × GC — ロード時ホスト RAM ピークの実測（メモリ管理波 Phase B）

> **性格**: 時点スナップショット（2026-09-02・RTX 3080 Ti 12GB / Linux / Vulkan (wgpu) /
> Deno 2.9.6・ホスト RAM 31GB）。ADR [0089](../decisions/0089-memory-admission-limits.md) の
> Phase B。2026-08-19 の実測（[shard-load-ram-peak](2026-08-19-shard-load-ram-peak.md) —
> f32 単一モデル・512MB 分割）を、**実配布形**（anima transformer f16 3.7GB / gemma4 i4 1.6GB）と
> 3 つの軸（shard 上限・shard 内の完了待ちの刻み・shard 境界の明示 GC）へ広げたもの。
> 数値は本機のドライバ・V8 の GC 挙動に依存する。Mac（Metal）/ Chrome（Dawn）は未測。

## 方法

- 資産: `models/karume-anima`（anima-turbo-v1.1 / f16 — transformer 5 shard・最大 948MiB）と
  `models/karume-gemma4-e2b`（e2b / i4 — 3 shard・最大 756MiB）。shard 上限の変種は
  `karume.repack.repack_component`（上限の差し込み）でテンソルのバイト列を変えずに詰め替え、
  manifest の shards 表だけを書き換えた写しミラー（`outputs/bench/<model>/2026-09-02_shard-size/`）。
  gemma4 は最大テンソル `lm_head` 384MiB のため 256MiB 変種は作れない。
- 計測: `tools/ram-peak/measure.ts`。**1 構成 = 1 プロセス × 3 回**。ピークは `/proc/self/status`
  の **VmHWM**（高水位標）をプロセス終端で読む。2 面ある:
  - **パイプライン面** = `fromPretrained` → 最小生成 1 回（anima は 512² / 2 step・Session は
    生成時に張るので生成まで回す）→ dispose。利用者が見る数字。
  - **コンポーネント面** = `createSessionFromShards` 直叩き（対象コンポーネントだけ・生成なし）。
    軸の切り分け用。
- 刻み = 実験専用ノブ `UPLOAD_FENCE_BYTES`（`SessionOptions` の非公開鍵）: shard 内で宣言バイト
  数の累計が N に達するごとに空 submit + `onSubmittedWorkDone` を入れる（既定は shard 末尾で 1 回）。
- 明示 GC = `deno run --v8-flags=--expose-gc` + shard 境界で `globalThis.gc()`（診断のみ —
  製品経路にもブラウザにも無い）。

## 結果 1 — パイプライン面（利用者が見る数字）

| 構成                   | VmHWM MiB（平均 / 3 回の範囲） | 対 1GiB |  load+生成 s | 主コンポ shard 数 | フェンス待ち ms |
| ---------------------- | -----------------------------: | ------: | -----------: | ----------------: | --------------: |
| anima f16 1GiB（現行） |          4,069（4,066〜4,073） |       — |         11.2 |                 5 |             338 |
| anima f16 512MiB       |          2,667（2,666〜2,668） |    −34% |  12.1（+8%） |                 9 |             400 |
| anima f16 256MiB       |          2,074（2,070〜2,081） |    −49% | 12.3（+10%） |                16 |             461 |
| gemma4 i4 1GiB（現行） |          2,622（2,621〜2,623） |       — |          3.2 |                 3 |               — |
| gemma4 i4 512MiB       |          2,203（1,870〜2,372） |    −16% |          3.3 |                 4 |               — |

## 結果 2 — コンポーネント面: shard 上限と刻み（anima transformer 単体・gemma4 model 単体）

| 構成                |       刻み | VmHWM MiB（平均 / 範囲） | 時間 s | shard 数 | フェンス待ち ms |
| ------------------- | ---------: | -----------------------: | -----: | -------: | --------------: |
| transformer 1GiB    | shard 末尾 |    3,882（3,882〜3,883） |    6.0 |        5 |             224 |
| transformer 1GiB    |    256 MiB |    3,882（3,881〜3,883） |    6.1 |        5 |             370 |
| transformer 1GiB    |    128 MiB |    3,875（3,862〜3,882） |    6.3 |        5 |             518 |
| transformer 1GiB    |     64 MiB |    3,882（3,881〜3,882） |    6.4 |        5 |             786 |
| transformer 512MiB  | shard 末尾 |    2,645（2,482〜2,910） |    7.0 |        9 |             286 |
| transformer 256MiB  | shard 末尾 |    1,884（1,878〜1,895） |    6.9 |       16 |             340 |
| gemma4 model 1GiB   | shard 末尾 |    2,417（2,416〜2,417） |    2.5 |        3 |              70 |
| gemma4 model 1GiB   |    128 MiB |    2,417（2,416〜2,417） |    2.4 |        3 |             141 |
| gemma4 model 512MiB | shard 末尾 |    2,163（2,163〜2,163） |    2.6 |        4 |              77 |

## 結果 3 — コンポーネント面: shard 境界の明示 GC（anima transformer 1GiB）

| 構成    |       刻み | VmHWM MiB（平均 / 範囲） | フェンス待ち ms |
| ------- | ---------: | -----------------------: | --------------: |
| 明示 GC | shard 末尾 |    2,955（2,954〜2,956） |             249 |
| 明示 GC |    128 MiB |    2,954（2,953〜2,955） |             527 |

## 所見

1. **ピーク ≈ 定数 + k × 最大 shard、k ≈ 3**（transformer 単体: 1GiB 3,882 / 512MiB 2,645 /
   256MiB 1,884 MiB — 傾き 2.7〜3.2 MiB/MiB・定数 ≈ 1.05GB）。2026-08-19 の「係数 3〜4」を
   実配布形で再確認。ばらつきは 3 回で ±4 MiB（512MiB 変種の単体だけ 2,482〜2,910 と揺れる —
   GC のタイミング依存）。
2. **刻み（shard 内の完了待ち）はピークを 1 MiB も下げない**（256 / 128 / 64 MiB とも誤差内）。
   費用だけ増える（フェンス待ち 224 → 786 ms・1 回 ≈ 11ms の床 × 回数）。**「writeBuffer の
   staging が完了まで RSS に残る」という仮説は Linux / Vulkan で棄却**。staging はこの環境では
   RSS の高水位に現れない（device 側のホスト可視メモリとして別勘定か、区間内で再利用されている）。
3. **明示 GC で −927 MiB ≈ shard 1 本分**（3,882 → 2,955）。刻みとの組み合わせでも同じ。
   よって k = 3 の内訳は「今の shard（1）+ GC 待ちの前 shard（1）+ **GC しても消えない 1**」。
   最後の 1 本は、消費側（`for await` の shard ループ）が次 shard の到着まで前 shard への参照を
   保つ構造由来と見立てる（speculation — 生成器側で GC を打つ時点では前 shard がまだ到達可能）。
4. **効くレバーは shard サイズだけ**で、刻みは無効。ただし shard サイズは配布形の再分割 =
   HF 再アップを伴う。GC は Deno / ブラウザから制御できない。
5. **構造的な候補（Phase C）= ホスト側の器の再利用**: shard を毎回新しい `ArrayBuffer` へ読むの
   ではなく、最大 shard 長の器 1 本へ読み込んで流せば、原理上 k = 1（−2 × shard ≈ 1GiB shard で
   −1.9GB）。512MiB 化（−1.2GB）より大きく、再配布も要らない。代償は「view は buffer 全体を
   占める MUST」（`ModelShard.bytes` の契約）の見直しと、取得層（Deno `readFile` / ブラウザ
   `Response`）を「与えられた器へ読む」形へ変えること。ブラウザ側の成立性は未検証。
6. 時間: 512MiB 化で load+生成 +8%（shard 待ち = 供給側の読み時間が 4.9 → 5.7 s に伸びる分が
   主で、フェンス床 ≈ 11ms × 追加 4 回は 60 ms しか無い）。256MiB は +10%。
7. パイプライン面 256MiB 変種（2,074）と単体（1,884）の差 190 MiB は、他コンポーネント
   （text_encoder 575MiB shard × 2 など）と生成時 transient の寄与。shard を 256MiB まで下げると
   ピークの主役が transformer から text_encoder 側へ移りうる（未分離 — speculation）。

## 判定の材料（裁定は ADR 0089 追記に記録）

- 512MiB 既定化は Phase B の採否線（ピーク −15% 以上・時間 +10% 未満）を満たす（−34% / +8%）。
- 刻みノブは無効 → 製品に入れない。実験ノブは Mac / Chrome の追試まで残し、追試後に削除する。
- 器の再利用（Phase C）が成立すれば 512MiB 化を上回り、再配布を要さない。

## 台本（再現用）

- 計測: `deno run -A tools/ram-peak/measure.ts …`（ファイル冒頭の使い方）。Mac は `--v8-flags` 無し・
  `/usr/bin/time -l` の最大 RSS で読む。
- 変種: `karume.repack.repack_component(<一時 dir の代表名>, out_dir=…, _shard_byte_limit=N)`。
  コンポーネントの shard は 2 dir に割れる（グラフ shard = `shared/…`・重み = モデル dir）ので、
  manifest の shards 表から全 shard を一時 dir へ symlink で集め、**番号なしの代表名**
  （`model.f16.safetensors`）を渡す。manifest は対象 (model, component, quant) の shards 表
  （path / size / sha256）だけを書き換える。
- 生データ: `outputs/bench/karume-anima/2026-09-02_shard-size/{results,fence,gc}.jsonl`。

## 結果 4 — Mac（Apple M2 24GB / Metal / Deno 2.9.4）コンポーネント面・ユーザー実測

ピークは `/usr/bin/time -l` の maximum resident set size（バイト → MiB）。各 3 回、1 回目は
ファイルキャッシュが冷えていて時間だけ長い（ピークは同水準）。

| 構成               |       刻み | 最大 RSS MiB（平均 / 範囲） | 対 1GiB | 構築 s（2〜3 回目） | フェンス待ち ms |
| ------------------ | ---------: | --------------------------: | ------: | ------------------: | --------------: |
| transformer 1GiB   | shard 末尾 |       4,727（4,622〜4,780） |       — |            1.4〜1.6 |        316〜409 |
| transformer 512MiB | shard 末尾 |       3,608（3,472〜3,676） |    −24% |                 1.4 |        334〜349 |
| transformer 1GiB   |    128 MiB |       4,121（4,116〜4,124） |    −13% |            1.9〜2.3 |        685〜846 |

所見の追記:

8. **Metal では刻みが効く**（−606 MiB・−13%。Linux では 0）。ユニファイドメモリでは writeBuffer
   の staging がプロセスの RSS に載り、完了待ちの間隔がそのまま滞留量になる — 所見 2 の棄却は
   **Vulkan / Linux 限定**に狭める。代償は構築 +0.5〜0.9 s（フェンス 1 回 ≈ 13 ms × 追加 ~25 回 +
   転送の直列化）。
9. **512MiB 化は Mac でも −1.1GB（−24%）で、時間は増えない**（1.4 s のまま — SSD とユニファイド
   メモリで shard 待ちが Linux より短い）。
10. Mac の傾きは 2.4 MiB / MiB・定数 ≈ 2.4GB と、Linux（3.0 / 1.05GB）より定数が大きい。GPU 側の
    常駐バッファがユニファイドメモリ上でプロセスの RSS に混ざっている可能性（speculation —
    分離するには Metal 側の計測が要る）。

## 結果 5 — 器の使い回し（Phase C-1・Linux パイプライン面・各 3 回）

hub の逐次面がコンポーネントの最大 shard 長の buffer を 1 本だけ確保し、毎回の shard を先頭から
読む（`FileReadOptions.into` / `DirectoryAdapter.readFileInto`・Deno のディレクトリ取得元が実装）。
runtime の shard 受け口は「buffer の先頭からの view」を受ける契約へ（`parseSafetensors` がファイル長を
別に受ける）。数値は結果 1 と同じ台本（`fromPretrained` → 最小生成 1 回）。

| 構成             | 器の使い回し   |  VmHWM MiB（平均 / 範囲） |  対 従来 | load+生成 s |
| ---------------- | -------------- | ------------------------: | -------: | ----------: |
| anima f16 1GiB   | なし（結果 1） |                     4,069 |        — |        11.2 |
| anima f16 1GiB   | **あり**       | **1,402（1,401〜1,403）** | **−66%** |     **5.7** |
| anima f16 256MiB | なし（結果 1） |                     2,074 |        — |        12.3 |
| anima f16 256MiB | **あり**       |       **911（909〜913）** | **−56%** |         6.7 |
| gemma4 i4 1GiB   | なし（結果 1） |                     2,622 |        — |         3.2 |
| gemma4 i4 1GiB   | **あり**       | **1,116（1,114〜1,119）** | **−57%** |         2.0 |

所見の追記:

11. **ピーク ≈ 450 MiB + 1 × 最大 shard** になった（anima: 1,402 − 948 = 454 / gemma4: 1,116 − 756 =
    360）。係数 3 → 1 だけでなく、所見 1 の「定数 1.05GB」も半分以下に落ちた — 定数の多くは
    shard ごとの ArrayBuffer 確保と GC の往復が残す断片・未回収分だった。
12. **ロードも速い**（anima 11.2 → 5.7 s・shard 待ち 4.87 → 1.66 s）。同じバイト数を読んでいるので、
    差は確保（ページのゼロ埋め・mmap）と GC の費用。「器の再利用は RAM のため・時間は据え置き」
    という見込みは外れで、時間も得る。
13. shard サイズの効きは残る（1GiB 1,402 → 256MiB 911）が、傾きは 1 × shard に落ちた。**512MiB か
    256MiB のどちらでも 1.0〜1.2 GB に収まる**ので、shard サイズの裁定は配信粒度（リクエスト数・
    キャッシュ本数）の側で決めてよい。
14. 効くのはローカル取得元（Deno / ディレクトリアダプター）だけ。HF（ブラウザ）経路は取得層
    `@hdae/fetch-cache` がキャッシュから読むたびに buffer を確保するので、`into` 相当の口を取得層へ
    足すまで従来の係数のまま（起票）。

## 結果 6 — 器の使い回し・Mac（Apple M2 / Metal / Deno 2.9.4・ユーザー実測・各 3 回）

| 面                                           | 器の使い回し     | 最大 RSS MiB（`time -l`・平均 / 範囲） | サンプル最大 MiB |        構築 s |
| -------------------------------------------- | ---------------- | -------------------------------------: | ---------------: | ------------: |
| transformer 単体                             | なし             |                  4,471（3,852〜4,780） |     3,818〜4,761 |      1.9〜3.2 |
| transformer 単体                             | **あり**         |              **2,060（1,995〜2,171）** |     1,957〜1,982 |    1.15〜1.33 |
| パイプライン（fromPretrained → 512² 2 step） | あり（hub 経路） |                  3,068（2,974〜3,158） |     2,936〜2,940 | 生成込み 25.5 |

所見の追記:

15. Mac でも係数は 1 へ（transformer 単体 −2.4GB・−54%）。構築時間も 1.9〜3.2 → 1.2 s。
16. パイプライン面は 2,936 MiB で単体より約 1GB 大きい。ユニファイドメモリでは生成中の GPU 側
    バッファ（text_encoder の常駐・attention の一時領域・VAE）がプロセスの RSS に混ざるため
    （speculation — Linux では GPU 側は RSS に出ないので同じ差は出ない）。

## 結果 7 — 目標 256MiB の配布形 + 器の使い回し（Linux パイプライン面・各 3 回・最終形）

書き手の目標 256MiB（ADR 0081 追記 2026-09-02・実効目標 = max(目標, 最大単位)）で系列 146 コンポーネントを
repack し、5 ミラーを再生成した後の値（結果 1 / 5 と同じ台本）。

| 構成                                            | shard         | 従来（結果 1） | 器のみ（結果 5） |                **最終** |  対 従来 |
| ----------------------------------------------- | ------------- | -------------: | ---------------: | ----------------------: | -------: |
| anima f16（transformer 16 shard ≤ 256MiB）      | 1GiB → 256MiB |      4,069 MiB |        1,402 MiB | **766 MiB（765〜767）** | **−81%** |
| gemma4 i4（model 6 shard ≤ 384MiB = `lm_head`） | 1GiB → 384MiB |      2,622 MiB |        1,116 MiB | **745 MiB（743〜747）** | **−72%** |

load+生成は anima 5.9 s（結果 5 と同等）・gemma4 1.8 s。

所見の追記:

17. 最終形のピーク ≈ 0.45〜0.5GB + 最大 shard 1 本。anima は 766 − 256 ≈ 510 MiB、gemma4 は
    745 − 384 ≈ 360 MiB が定数側。
18. 実効目標の規則が残す小さな歪み: 目標より大きい単位（割れないテンソル）の**直前**にある小さい
    テンソル群が単独の shard になる（順序を変えないので、大きい単位を足すと実効目標を超える位置で
    cut が入る）。実資産では gemma4 model（1.2MB）・irodori backbone（1.6MB）・anima text_encoder
    （1.9MB）の 3 コンポーネントに各 1 本。リクエスト 1 回ぶんの費用で、テンソル分割（C-2）が入れば
    規則ごと消える。
