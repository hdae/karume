# 0060 — 分解 attention の行ブロック実行（K-5 案 B・既定経路・ビット同一）

- Status: accepted（2026-08-14・ユーザー裁定「B 先行 — ビット同一で先にチェック」→ 実装承認。
  スパイクで成立性・ビット同一・代価を先に実測してから本実装）
- 対象: runtime（fusion.ts / executor.ts / kernels/bmm.ts / gemm.ts / gpu/device.ts）。
  IR 仕様・エクスポータ・配布資産は無変更（再 emit 不要）。
- 需要: irodori DiT の分解 attention（bmm QKᵀ → add mask → safe_softmax → bmm PV）は
  S=750 で中間 `scores[1,20,750,2269]` = 136,140,000 B を実体化し、WebGPU core 既定の
  `maxStorageBufferBindingSize`（128MiB）の機で確保・束縛に失敗する
  （一次出典 = [dit-export-recon](../research/2026-08-11-dit-export-recon.md)・台帳 K-5）。
- 位置づけ: **K-5 の 2 段構成の前段**。本 ADR（案 B）はビット同一のままポータビリティを
  既定経路で解決し、案 A（online attention — 加算順が変わる速度層）は同じシャーシの上の
  opt-in 席（[ADR 0058](0058-numerics-opt-in-contract.md)）として後続。

## 決定

1. **FusedStep を「dispatch 列 + ステップ内一時」へ拡張**（融合層の機構拡張）。従来の
   4 ルールは 1 dispatch のまま（記述も実質不変）。一時の寿命は dispatch 境界の添字で宣言し、
   executor が `StepRecipeBuilder` へ replay する — 解放簿記の合流点（`executeStepRecipe`）と
   useCounts の根拠（`FusedStep.ins` の延べ列）は 1 本のまま。workgroup 数は判別共用体
   （grid-stride = 縮退可 / tiled = GEMM 族・上限超過 fail loudly）。
2. **`rowBlockAttention` 融合ルール**: エクスポータが出す連続 9 ノード窓（bmm → reshape →
   add → safe_softmax → 恒等 expand/reshape ×2 → bmm・全結線と形を exact-match）を、
   クエリ行ブロックごとの同型 4 dispatch 列へ置換する。掴めた窓は**常時融合**し、枚数 n は
   Session 構築時の純関数 `planRowBlocks`（入力 = granted limit・H・C・解決済み S のみ —
   実行時オートチューン禁止 ADR 0022）で「1 枚が束縛上限に収まる最小枚数の等分」。
   **n=1 の機では素の 4 dispatch 列と完全同一**（既存キー・既存 params — 追加コストゼロ）。
   1 行でも上限に入らない形は fail loudly（黙って素の列へ戻すと、確保失敗が「融合が
   外れただけ」に見える）。
3. **bmm の行窓変種**（`:rwa` = QK 側が q を全 M ストライド + 行オフセットで読む /
   `:rwc` = PV 側が出力を同様に書く）。キーに載るのはこの 1 ビットだけで、オフセットと
   全 M は uniform 値（bmm 専用 5 語 — 共有 `gemmParams` には 1 語も足さない = 他 op の
   スナップショット総取っ替え回避）。生成物の差は `batchPrologue` の base 算術 2 行に閉じ、
   **1 出力要素の K 縮約順・積和の字面は 1 文字も動かない** — 行ブロックが 1 枚実行と
   ビット同一になる根拠（gemm-geometry.ts の数値契約の帰結）。add（mask `[1,1,1,C]`
   broadcast）と safe_softmax は行内で閉じるためブロックバッファ相手に既存カーネルのまま。
4. **数値の席 = 既定経路**: ビット同一なので ADR 0058 の opt-in 席は不要（席の対象は
   「数値を変える最適化」— 案 A のみ）。門は ①既存 WAV/PNG/golden digest 不変 ②強制分割
   parity（`ROW_BLOCK_SPLIT` 内部面で 2/3/端数割を強制し 1 枚実行と Uint32 完全一致 —
   出力が定数でないことも検査し恒真化を防ぐ）。
5. **ポータビリティ門**: `LIMIT_CAPS` 内部面（acquireGpu の requiredLimits を**絞る向き
   のみ** — 引き上げには使えない）で `maxStorageBufferBindingSize` = 128MiB の device を
   作り、scores > 128MiB の合成グラフ（実資産不要）が行ブロックで緑・窓を崩すと
   `GpuValidationError`（束縛上限超過）で落ちることを実 GPU で検査。「絞った device 上で
   実走する」のが core 既定機での動作を確かめる唯一の手段（列挙は証拠にならない）。
6. **観測面**: `lastRunFusions.rowBlockAttention`（dit = 12 — assets census 門に追加）。
   `identityExpand` 48 → 24 は窓内の恒等 expand が融合に飲まれた**移動**（退行ではない）。
   gpuTiming の帰属キーは **n ≥ 2 のときだけ** `…:rwa` / `…:rwc` に分かれる（n=1 は従来
   キーのまま — restats 系ドキュメントとの突合は無風）。

## 検収（2026-08-14・RTX 3080 Ti / Vulkan・レッグ実測 + メイン再実測）

- `deno task verify` 全緑（1275 passed / 0 failed / 27 ignored）。WAV 門 2 本
  （`05f82a9c…` / `5cc43d7f…`）・PNG 門・golden 27 本 digest 不変。
- 強制分割 parity: 3 形状 ×（1/2/3/5 枚・端数割・幾何バケット跨ぎ）で素の 9 ノード列と
  Uint32 完全一致。
- ポータビリティ門: 128MiB device + `H4 M2048 N4200 D8`（scores 137,625,600 B）が 2 枚で
  緑・窓を崩すと束縛上限の validation で落ちる（効力証明）。
- 性能: S=170 voice-clone A/B（ABBA・各 2 走）で −0.21% = 誤差内の非退行（n=1 経路）。
  S=750 実測形の 12 層合成グラフでは 2 枚 46.8ms vs 1 枚 48.9ms — スパイク時の +1.4%
  （slice/cat コピー方式）は行窓化で解消。中間ピーク 267.0 → 137.2MiB（2 枚）。
- スパイクの正本記録: 全構成 WAV sha 一致（B_r 17〜750）・S=750 で scores 129.83 → 44.3MiB
  （3 枚）・resident/batch（ADR 0054）経路と両立（フェンス・readback ゼロ増）。

## 残余・接続

- 案 A（online attention・opt-in 席）は本ルールの matcher・門・シャーシの上にループ本体
  差し替えで載せる（台帳 K-5 の後段）。
- 隣接（対象外・記録）: この窓を IR op `attention`（ADR 0023）へ寄せて S 自体を非実体化する
  案は、mask 契約（`[1,1,M,N]` vs `[1,1,1,N]`）と safe_softmax の空行契約（ADR 0044）の
  2 点が要る別トラック。`FusedDispatch.operands` 省略時の並び規約は doc の MUST のみ
  （型で縛る案は既存 4 ルールの記述量とのトレードで見送り）。
