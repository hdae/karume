# gemma4 投機デコードの M=1 / M=4 同一性 — 既定席の「報告」を門へ上げた根拠の実測（2026-09-26）

> 2026-09-26 時点の実測スナップショット（Intel Arc B570〈BMG G21〉/ Deno・features: shader-f16, timestamp-query〈subgroups 無し〉・資産 `models/karume-gemma4` の e2b〈既定 quant `i4-fast`〉・HEAD `8bd655d0` の `e2e_gemma4_speculative_test.ts`）。門へ上げる**前**のテストで採った値で、数値はログの逐語から写した。

## 1. 何を確かめたか

gemma4 の投機デコードは、自己採算ゲート（`speculative: true` の既定）が壁時計で decode 形（M=1）と verify 形
（M=4）を切り替える。実用層でもデバイス内決定性を MUST とする裁定の下では、この切替が出力を変えないことが
要る。ADR 0058 追記 2026-08-29 の一般則 3（ビット同一を根拠にする自動選択は機械検証つきに限る）に従い、
選べる 2 経路のビット同一を機械検証できればゲートはこの MUST に適合する。

測定前の docs（limitations の投機節・`pipeline.ts` の `speculative` の NOTE・ADR 0096 追記 2026-09-09）は
「既定席〈`stateAttentionReduce: "parallel"`〉では同じ seed でも稀に出力が変わりうる」としており、既定席の
相違は e2e が**報告するだけ**（門ではない）だった。

## 2. 実測

`deno test -A packages/models/tests/e2e_gemma4_speculative_test.ts` の結果: `ok | 3 passed (16 steps) | 0 failed (6m1s)`。

### 投機① — `stateAttentionReduce: "sequential"` 席（既存の門）

| ケース          |    T | 生成 token | cycle | 受理 / draft | token/cycle | 受理数の分布  | 投機 ms | 非投機 ms | torch 継続列との先頭一致 |
| --------------- | ---: | ---------: | ----: | ------------ | ----------: | ------------- | ------: | --------: | -----------------------: |
| short-en        |   25 |        200 |   132 | 67/390       |       1.508 | [80,40,9,3]   |  27,400 |     6,268 |                      200 |
| readme-recipes  | 2335 |        200 |    99 | 100/291      |       2.010 | [50,16,15,18] |  20,485 |    18,839 |                      200 |
| readme-exporter | 4844 |        200 |    93 | 106/277      |       2.140 | [33,28,18,14] |  29,172 |    29,359 |                      200 |

ゲート付き（readme-recipes）: 200 token / 74 cycle / plain step 52 / 切替 2 / ゲート 19,093 ms vs always 19,791 ms。

### 投機② — 既定席（`i4-fast` = GEMV parallel + RMS→add 融合・attention `"parallel"`）

| ケース          | 相違（個） | 最初の不一致 | token/cycle | 投機 ms | 非投機 ms |
| --------------- | ---------: | ------------ | ----------: | ------: | --------: |
| short-en        |      0/200 | 無し         |       1.508 |  28,045 |     5,495 |
| readme-recipes  |      0/200 | 無し         |       2.010 |  19,849 |    16,949 |
| readme-exporter |      0/200 | 無し         |       2.140 |  28,833 |    26,727 |

### 投機⑤ — verify 行 0 と decode 1 行の logits（u32）

| 席                                              | u32 相違     | 最初の不一致 | 最大絶対差 |    ms |
| ----------------------------------------------- | ------------ | ------------ | ---------: | ----: |
| attention sequential × GEMV sequential          | 0/262,144 語 | —            |          — | 3,906 |
| attention parallel × GEMV sequential（`i4` 席） | 0/262,144 語 | @-1（無し）  |          0 | 2,433 |

ログ行を持たない ⑤ の門 step 6 本（attention 3 変種 × GEMV parallel・attention sequential × RMS→add 融合 × GEMV 2 種・
attention parallel × GEMV parallel × RMS 融合 × linear→SRQ 指定）も緑である（16 step 全緑から）。

## 3. 読み取り

- 既定席でも投機 / 非投機の token 列は 3 ケースとも完全一致し、⑤ の u32 突合は門にしたすべての組で一致した。
  ゲートが選ぶ 2 経路（M=1 / M=4）はこの device でビット同一である。
- ② の相違 0 と ⑤ の attention parallel × GEMV sequential の一致は、測定前には「報告」で門ではなかった。
  値が 0 なので、門へ上げても現状の緑は変わらない。
- token/cycle は ① と ② で 3 ケースとも同じ値になった（② のログは受理数の分布を出さないので、分布の比較は無い）。

## 4. 何が門になったか（同日のテスト変更）

- 投機② の 3 ケース: 相違 0 と共通接頭辞 = 200 を `assertEquals` で門にした（ログ行は門より先に出す）。
- 投機②③④ の test に、既定席のゲート付き（`speculative: true`）の列が always と同一である step を足した。
  plain step 数・切替回数は壁時計依存なのでログのみ。同日の再走で緑: 既定席のゲート付き = 200 token / 97 cycle / plain step 7 / 切替 0 / ゲート 16,794 ms vs always 17,075 ms（列は always と同一）。
- 投機⑤ を attention 3 変種 × GEMV 2 種 × RMS→add 融合の全 12 組（① の sequential × sequential × 融合なし + 11 組）の u32 門にした。同日の再走（22 step 全緑・5 分 55 秒）で 11 組とも相違 0/262,144 語・最大絶対差 0（各 2.0〜2.1 秒）。
- docs は「既定席では稀に割れうる」を撤回し、「保証は device ごと（門が赤の機では相違が実在し、その機では
  `"always"` か `stateAttentionReduce: "sequential"` を選ぶ）」へ改めた（limitations の投機節・ADR 0096 追記
  2026-09-26）。

## 5. 残る範囲

- 保証はレーンを回した device に閉じる。一般則 3 が例に挙げる実走カナリア（利用者の device で実行時に同一を
  確かめる口）は無い。
- u32 の門は全 12 組を覆う（同日の再走で確認）。門に無いのは linear→SRQ 指定を他の組と併せた形だけである。
