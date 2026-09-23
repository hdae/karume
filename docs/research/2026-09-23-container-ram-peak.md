# コンテナ経路のホスト RAM ピーク（段 2 の検収②③・2026-09-23）

> 時点スナップショット（Arc B570 / Deno 2.9.6 / Linux・HEAD `750b78b1` の直後）。道具は
> `tools/ram-peak/matrix.ts`（1 構成 1 プロセス・cold → warm → local を各 3 回・中央値）。生の記録は
> `outputs/ram-peak/2026-09-23_<系列>/results.jsonl`（git 追跡外）。判定の正本は
> [ADR 0108](../decisions/0108-container-format.md) 段階分解表 段 2 の検収②③と
> [container-v1](../container-v1.md) §7 / §11。

## 何を測ったか

- **取得元 3 通り**: cold = 疑似 HF（`serveLocalDist`）+ 空のディレクトリ固定キャッシュ / warm = 直前の
  cold が温めた同じキャッシュ / local = `denoDirectory`（位置読みあり = seek 型）。Deno の HF 経由は
  取得層の既定戦略が "stream" なので hub の取得面は **scan 型**（part を 1 度に読んで block に切る —
  ADR 0109 決定 7）。
- **数値**: `external`（V8 が抱える ArrayBuffer の実勢 — 50 ms 標本化の最大値・load = `fromPretrained`
  の決着まで / run = 最小の生成 1 回）、VmHWM、`crypto.subtle.digest` の回数（descriptor 2 文書の突合を
  除いた payload 側を括弧に）、キャッシュ書込の本数。
- **宣言だけで閉じる見積り**（container-v1 §11）: 最大 part 256 MiB / 最大 block 32 MiB。

## 結果（中央値・MiB）

| 構成                                                   | 取得元 | external 最大 load | external 最大 run | VmHWM | load ms | digest 全体 (payload) | cache 書込 本 (MiB) |
| ------------------------------------------------------ | ------ | -----------------: | ----------------: | ----: | ------: | --------------------: | ------------------: |
| gemma4 e2b `i4-fast`（重み 1,505 + PLE 資産 2,275）    | cold   |              1,287 |             1,050 | 1,676 |  24,079 |                 3 (1) |        85 (3,790.6) |
|                                                        | warm   |              1,287 |             1,082 | 1,752 |   3,098 |                 2 (0) |                   0 |
|                                                        | local  |                775 |               746 | 1,375 |   2,981 |             874 (872) |                   0 |
| irodori v4.1-small `i8-a8`（重み 823）                 | cold   |                 68 |               777 | 1,223 |   4,756 |                17 (1) |          26 (829.4) |
|                                                        | warm   |                  6 |               777 | 1,217 |     180 |                16 (0) |                   0 |
|                                                        | local  |                  4 |               393 |   931 |     175 |         1,238 (1,222) |                   0 |
| anima turbo-v1.1 `f16+dit8-a8-attn8-s16`（重み 3,313） | cold   |                 69 |             1,261 | 1,659 |  18,840 |                10 (2) |        28 (3,320.7) |
|                                                        | warm   |                  7 |             1,261 | 1,652 |     318 |                 8 (0) |                   0 |
|                                                        | local  |                  6 |               676 |   956 |     294 |         1,594 (1,586) |                   0 |

irodori / anima は Session を生成時に張る（load は descriptor と資産だけ）ので、重みのピークは run 側に
出る。gemma4 は `fromPretrained` で常駐 Session を組むので load 側に出る。

## 読み取り

1. **検収③（warm で digest 0 回）は成立**: 3 構成とも warm の payload 側 digest は 0・キャッシュ書込 0 本。
   全体の 2 / 16 / 8 回は descriptor 2 文書の突合（容器 1 本につき 2 回・開くたびに掛かる — §7 の①）。
   local の 872 / 1,222 / 1,586 回は未検証の取得元に対する block ごとの digest（§7 の表の 3 行目）。
2. **HF 経由（scan 型）のピークは local（seek 型）の約 2 倍**: run のピークで gemma4 1,050 / 746、irodori
   777 / 393、anima 1,261 / 676。差はおよそ part 2 本ぶん（512 MiB）。scan 型では取得層の `readFile` が
   part 全量の新しいバッファを返し、hub がそれを 1 枠保持し、runtime の `containerBatches` が同じ part の
   block を `slice()` で写して items に溜めてから yield するので、瞬間的に part 2〜3 本が生きる。seek 型は
   block ごとに読むので part 1 本 + block 1 本に収まる（宣言からの見積りどおり）。
3. **ブラウザ**は取得層の戦略が "blob"（seek 型）なので local 列の形になる見込み（**推測** — Chrome での
   実測は未）。
4. cold の load / run 時間は疑似 HF からの取得（ディスク → ディスク）を含むので、帯域の数値としては
   読まない。

## 段 3 への持ち越し（RAM ピークの改善候補 — 実測で採否を付ける）

- hub の scan 型: 取得層の `readFile` に `into`（器）を渡して part 長 1 本の器を使い回す（今は取得層が毎回
  新しいバッファを確保する）。
- runtime の `containerBatches`: part の全 block を items に溜めてから yield する形を、block ごと（または
  同時処理バイトの予算ごと）の yield にする（errorScope は既に block ごと・フェンスは part ごと）。
- `slice()` の写しを view（`subarray`）にする — hub の 1 枠が次の part を読むまで前の part を保持する契約と
  `WeightBatch` を part 単位で使い切る契約は噛み合う。
- 「持越し scale」（piece 分割で part を跨いで持つ scale）の欄は runtime の診断に席が無い（`missingFromRuntime`）。
