# 0114: Irodori DiT の条件側 K/V 射影を別グラフ `dit_context` に割る

- Status: accepted（利用者裁定 2026-09-27・案 A。B570 の ABBA で voice-clone −12.7% が kill 線 −5% を超えた — 下の「検収」節）
- Date: 2026-09-27
- 関連: ADR [0047](0047-irodori-dit-execution.md)（決定 3「G4 は当面 G5 へ畳む」を本 ADR で置き換える）/
  [0054](0054-resident-loop-and-fence.md)（ResidentTensor・batch enqueue・`copyOutputs` — 本 ADR が使う 3 部品）/
  [0050](0050-irodori-quant-series.md)（irodori の格納系列）/ [0069](0069-packed-w4-storage.md)（i4 格納）/
  [0074](0074-quant-seat-naming.md)（席名の文法）/ perf-ledger H-30・H-7・K-54。
  実装 = `tools/export-recipes/irodori/export.py`（`DitContextGraph` / `DitGraph` / `name_boundary`）・
  `tools/export-recipes/irodori/distribution.py`（`irodori_context_kv_names`）・
  `packages/models/src/irodori/dit-loop.ts`（段 6'）・`packages/models/src/irodori/admission.ts`（`admitDitContext`）。
  調査の正本は `.claude/reviews/2026-09-25_perf-recon/deep/B4-hoist-cfg-h30.md`（git 追跡外）。

## Context

irodori の DiT は、各ブロックの cross-attention で条件 3 本（text 256 + speaker 751 + caption 512 = 1519 token）から
context K / V を射影する（上流 `project_context_kv`）。この条件側は可変入力（`x_t` / `t_embed` / `mask`）のどれにも
依存しない 244 ノード（linear 72）で、1 回の生成の全 forward で同じ値を出す。ADR 0047 決定 3 はこれを `dit` に
畳み、forward ごとに再計算していた。畳んだ理由は「別グラフにすると出力 178 MiB を毎 run アップロードする」ことだった。

その理由は ADR 0054 で消えた（ADR 0047 の 2026-09-25 追記）: ResidentTensor と `copyOutputs` があれば、別グラフの
出力を GPU の中で常駐テンソルへ写し、`dit` の run に常駐入力として渡せる。

現行の実行形で測った条件側の GPU 時間（B570・`opbench single`・i8-a8）: 条件側 72 linear + rms_norm 38 + cat 24 の
単発合計が forward あたり 14.5 ms @S=750 / 14.2 ms @S=170。S=750 の `dit` 1 forward の GPU 60.9 ms の **24%** に当たる。

## Decision

1. **境界 = ブロックごとの連結済み context K / V の 24 本**。新しい役割 `dit_context`（系列ディレクトリ `dit-context`）が
   各ブロックの条件側射影と 1 段目の連結（text / speaker / caption → 1519 行）までを持つ。
   - 入力: `text_state [1,256,512]` / `speaker_state [1,751,768]` / `caption_state [1,512,512]`（projector の生の出力を
     Tmax 右 pad したもの）。`text_norm` / `caption_norm` はこのグラフの内側で掛かる。
   - 出力: `context_k_0, context_v_0, …, context_k_11, context_v_11`（各 `[1,1519,20,64]` f32・K は `k_norm` 済み）。
     24 本で 178.0 MiB。
   - `dit` の入力は `x_t` / `t_embed` / `mask` + この 24 本。条件 state 3 本は `dit` から消える。
   - 72 本の射影を個別の出力にしないのは、連結まで条件側に含めると境界が半分の本数で済み、`dit` 側の連結も消えるため。
2. **境界名の綴りは 1 箇所**: `irodori.distribution.irodori_context_kv_names(blocks)`（ブロック b ごとに
   `context_k_<b>` → `context_v_<b>`・b の昇順）。export の書き手（IR の境界名の付け替え）と dist の読み手が同じ関数から組む。
   **ブロック数は焼かない** — exporter は `len(model.blocks)`、models は `dit_context` の宣言（出力本数）から導く。
3. **常駐経路の順序**（`dit-loop.ts` の `projectContextOnGpu`）:
   ① 常駐テンソル 24 本を確保する ② `dit_context` の Session を開き、前段の batch で 1 回 enqueue + `copyOutputs` →
   finish ③ **Session を dispose する** ④ `dit` と combine / euler の小グラフを開いてループの batch を回す。
   - ループと同じ batch に積まない理由: batch の中では Session を dispose できない（ADR 0054）ので、`dit_context` の
     出力スロット 178 MiB が finish まで残り、VRAM の増分が倍になる。前段 batch に割るとフェンスが 1 本増えるが、
     増分は常駐 K / V の 178 MiB に留まる。
4. **ホスト経路**（`gpuTiming` 有効の device / `onEvent` 購読 — ADR 0054 の既存の分岐）: `dit_context` を `run` →
   出力をホストで受けて常駐テンソルへ `write` → `dit` の `run` に常駐入力で渡す。毎 forward の 178 MiB アップロードは
   しない。Session は常駐テンソルを確保する前に畳む。2 経路の違いは写し方（GPU コピーかホスト経由の `write` か）だけで、
   `dit` が読む K / V のバイトは同じ。
5. **同じ Session 設定**: `dit_context` にも `ditSessionOptions`（quant 席の `linearCompute`）を渡す。分割前は条件側の
   linear が `dit` の中で同じ席の実行形で走っていたので、別の席で走らせると数値が動く。
6. **観測名**: `IrodoriRunComponent` と `stage` イベントの段名に `"dit-context"` を足す（役割名の `_` → `-` の既存規則）。
   `dit_context` は生成 1 回に run 1 回。
7. **admission**（`admission.ts` の `admitDitContext` — 重みを取る前）が 4 点を見る:
   - `dit_context` は記号次元を持たない（常駐テンソルは確保時に大きさが要る）。
   - 出力は 2 本以上の偶数本（ブロックごとの K / V の対）。
   - `dit` の入力名の集合 = {`x_t`, `t_embed`, `mask`} ∪ `dit_context` の出力名。
   - 各出力は静的な形で、`dit` の同名入力と形・dtype が一致する。
     常駐入力は runtime では大きさしか検査されないので、形や dtype の食い違いは admission で落とさないと別の並びとして
     読まれた K / V で沈黙のまま回る。条件 state 3 本の静的次元の突合（`maxTextLen` ほか）も `dit` から `dit_context` へ移る。
8. **i4 席は DiT の 2 本**: `IRODORI_DTYPE_ROLES["i4"] = ("dit", "dit_context")`。`i8+dit4` は両方を i4 系列から採る。
   分割前は条件側の 72 本が `dit` の中で i4 格納だったので、i8 へ落とすと席の数値が動く。席名 `i8+dit4` のトークン
   `dit` は DiT の 2 本を指す（席名を変えるのは配布の breaking なので変えない）。席の説明文「the other seven graphs」は
   DiT を 2 本と数えてもそのまま正しい。
9. **runtime は無改修**（常駐テンソル・`copyOutputs`・`run` の常駐入力の既存 3 部品で閉じる）。

### 採らなかった案

- **案 B: models 側でロード時にグラフを割る**（読み込んだ宣言を TS で依存解析して 2 本に合成）— 採らない。
  再 export は要らないが、汎用のグラフ分割器が models に入る（H-7 の機構が場所を変えて戻る）。合成した宣言が
  合流層（`format/container/bind.ts`）の検証を迂回し、走るグラフが export で検証したグラフ（eager 同値・golden）と
  一致しなくなる。
- **案 C: runtime の staged execution（perf-ledger H-7）** — 採らない。H-7 は「機構の複雑さに見合わない」で棄却済みで、
  条件キャッシュの鍵（内容の世代）とアリーナ寿命の拡張が要る。効く相手が irodori 1 本で、案 A は既存部品だけで閉じる。

## Consequences

- **数値はビット同一**。根拠と検証:
  - 分割ペアの合成（`dit_context` → `dit`）と上流（パッチ前）の eager が atol 0 で一致（全 golden ケース・f32 / f16 / i8 / i4 の
    4 系列）。
  - 格納バイトの突合: 旧 `dit` 430 本 = 新 `dit` 356 本 ∪ 新 `dit_context` 88 本（重なりは `k_norm` 14 本で両側一致）で、
    4 dtype とも 430 / 430 がバイト一致。
  - i4 は GPTQ を同じ校正入力で再実行し、条件側 72 本が旧系列とビット一致（校正の再現性が実測で成立）。
  - full-loop golden 8 本（4 dtype × 2 ケース）の sha256 は不変。WAV の参照 sha（f32・B570 の行）も不変で、`onEvent`
    経路の WAV も同一。
  - w8a8 の census は `dit` 245 / `dit_context` 72（分割前 `dit` 317）。
  - 故障注入: `context_k_0` と `context_v_0` を入れ替えると WAV sha が変わる（配線の検出力）。
  - runtime の系列 e2e に `dit-context` を足した（代表 1 ケース・atol 5e-4 / rtol 1e-6・実測 maxAbs 6.48e-5）。
- **速度**: B570 の i8-a8 で voice-clone −1.17 s（−12.7%）・30 s 発話 −0.67 s（−5.0%）（下の「検収」節）。
- **VRAM**: 常駐 K / V は 178.0 MiB。GPU バッファの観測ピークは分割の前後で不変（DiT 段は生成全体のピークの位置ではない）。
- **配布形の breaking**: manifest の weights に役割 `dit_context` が増える（f32 / f16 / i8 / i4 の 4 系列）。`dit_context` を
  持たない配布形は、重みを取る前に部品の欠落で落ちる。公開済み 2 リポ（`irodori-v4-small` / `irodori-v4.1-small`）の
  HF 再アップロードと pin 更新はリリース時（backlog release 節 — later の「irodori 重複 5,605 MiB を越境参照で消す」の
  breaking 波と同じ回）。
- `dit` の io golden は入力に K / V 24 本を含む（条件 state 3 本の代わり）。
- ADR 0054 決定 4 の「条件 3 本を resident に置く」は、本 ADR 以後は「条件側 K / V 24 本を resident に置く」と読む。
- 本 ADR の外: adaLN の timestep 共有（仮称 H-30b — `t_embed` だけに依存する M=1 linear を CFG 変種で共有する）と
  K-54（a8 の M=1 linear の小 M 経路）。材料として adaLN の M=1 linear 144 本が forward あたり 11.6〜12.0 ms
  （`dit` GPU の ≈19%）と測れた（perf-ledger K-54）。

## 検収（2026-09-27・Arc B570 / Linux xe / Deno 2.9.6）

ABBA（i8-a8・A = 分割前 / B = 分割後・A B B A の 4 ブロック。各ブロックは初回を捨てて 3 回を採る = 側ごとに n = 6・
中央値は中央 2 値の平均）。WAV の sha は A と B で同一。

| ケース                                                  | A（分割前） | B（分割後） | 差                | GPU バッファの観測ピーク（A / B） |
| ------------------------------------------------------- | ----------: | ----------: | ----------------- | --------------------------------- |
| voice-clone（S=170・forward 100）                       |      9.18 s |      8.01 s | −1.17 s（−12.7%） | 811 / 811 MiB                     |
| representative（S=750・forward 60・durationSeconds 30） |     13.25 s |     12.58 s | −0.67 s（−5.0%）  | 864 / 864 MiB                     |

- 観測ピーク = createBuffer / destroy を数えた生成 1 回の最大。
- kill 線（voice-clone で −5% 未満なら見送り）を超えた → 採用。
- 条件側の単発計測（Context の 24%）は i8a8 linear のパイプラインキーに形が載らないので、`opbench graph` の per-key では
  条件側 72 本を分けられない。形ごとの時間は `opbench single` で採った。
- 実測の置き場（git 追跡外）: ABBA の JSON = `outputs/bench/karume-irodori/2026-09-27_h30/abba/`・単発計測 =
  `outputs/bench/karume-irodori/2026-09-27_op-single*`・export のログ = `outputs/bench/karume-irodori/2026-09-27_h30/`。
