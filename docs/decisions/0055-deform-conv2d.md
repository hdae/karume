# 0055 — 語彙拡張: deform_conv2d（DCNv2・第 1' 層）

- Status: accepted（2026-08-13）
- 関連: ADR [0043](0043-op-addition-layers.md)（層の定義と判定手順 — 本 ADR は第 1' 層の
  2 例目）/ [0023](0023-fused-attention.md)（契約の絞り = 決定 4「欄の不存在が語彙に無いことを
  構造で表す」）/ [0024](0024-conv2d-implicit-gemm.md)（conv 系カーネルの縮約順と bias-first・
  im2col 実体化の容量却下）/ [0017](0017-rms-norm-conv2d-clamp-min.md)（conv2d 契約）/
  [0020](0020-nan-propagation-bitwise.md)（NaN をビット列で判定する規律）。
  層分類の穴と暫定運用は [known-issues.md](../known-issues.md) の
  「op 追加の層分類が一部の op で自明にならない」節。
- 需要の実測: BiRefNet 一族（公式 8 チェックポイント + サードパーティ FT の Lucida）が
  全て `dec_att = 'ASPPDeformable'` で、**1 forward あたり `torchvision.ops.deform_conv2d`
  を 20 回**通る（`ASPPDeformable` 5 インスタンス × 4 分岐・全て `64 → 256`）。
  BiRefNet_HR / lite / base / Lucida の正面 blocker はこの 1 op。

## Context

`torchvision::deform_conv2d` は `TORCH_LIBRARY_FRAGMENT(torchvision, …)` で登録される
カスタム op で、**Core ATen の外**にある。実測（本ワークツリーで `torch.export` →
`run_decompositions(curated_decompositions())` を実走）でも
`torchvision.deform_conv2d.default` の**1 ノードのまま残る**（分解表に登録が無い）。

BiRefNet の `DeformableConv2d.forward` は `offset = offset_conv(x)`（学習済み conv の
実行時出力・**クランプはコメントアウト済みで無効**）と
`modulator = 2·sigmoid(modulator_conv(x))` を渡す DCNv2 形。offset は入力平面の外を指しうる
無制限の実数で、**export 時定数ではない**（第 0 層は成立しない）。

語彙内の合成でも書けない: `floor` が語彙に無く（`cast` は trunc なので負値の floor に
ならない）、`gather` は最終次元固定で 2 軸の値依存 gather を表せない。→ **原子**。
原子 かつ Core ATen 外なので、known-issues の暫定運用（2026-08-13 ユーザー裁定 —
第 1 層 = Core ATen 内の原子 / **第 1' 層 = それ以外の原子**、要求元が IR かモデルかを
問わない）に従って **第 1' 層 = 恒久の公開コミットメント = ADR 必須**。

## 決定

### 1. 契約は「実測形だけ」— 欄の不存在が絞りを表す（ADR 0023 決定 4）

`deform_conv2d`・**アリティ 5 固定**（`x` / `weight` / `offset` / `mask` / `bias`）・
uniform f32・attrs は **`padding`（`[H, W]` の 2 成分）1 本だけ**。

| 形       | 契約                                                             |
| -------- | ---------------------------------------------------------------- |
| `x`      | `[B, Cin, H, W]`                                                 |
| `weight` | `[Cout, Cin, Kh, Kw]`（groups が無いので第 2 軸は Cin そのもの） |
| `offset` | `[B, 2·Kh·Kw, Hout, Wout]`（**偶数チャネル = y / 奇数 = x**）    |
| `mask`   | `[B, Kh·Kw, Hout, Wout]`（modulator — 補間の**後**に掛かる）     |
| `bias`   | `[Cout]`                                                         |
| 出力     | `[B, Cout, Hout, Wout]`                                          |

- **`stride` / `dilation` / `groups` / `offset_groups` の欄を作らない = 1 固定**。
  実測 20 箇所が全て 1 で、欄を持たないことが「その形は語彙に無い」の宣言になる
  （conv_transpose1d の `output_padding` / `dilation` / `groups` と同じ手筋 — ADR 0015）。
  「対称性のための追加をしない」（op-vocabulary.md）が stride にも等しく効く。
- **`mask` は入力スロットで必須 = DCNv2 専業**。`use_mask=False`（DCNv1）は attrs の
  bool ではなく**スロットの有無**で表される形なので、アリティ固定がそのまま fail loudly に
  なる。torchvision は `mask=None` のとき `[1,1]` のダミーテンソルと `use_mask=False` を
  渡す（実測）ので、エクスポータ側は `use_mask` フラグを直接見て落とす。
- `bias` はアリティ 3 の conv 族と違い**合成しない**。`bias=None` のとき torchvision の
  Python ラッパが `aten.full([Cout], 0)` を挿し、それが第 0 層の定数畳み込みで initializer に
  なる（実測）— エクスポータに二重の合成経路を作らない。
- `padding` だけが実測で動く（BiRefNet の 4 分岐は `k//2` = 0 / 0 / 1 / 3）。出力空間は
  `Hout = H + 2·padH − (Kh − 1)` / `Wout = W + 2·padW − (Kw − 1)` で **x と weight と attrs から
  導き**、`offset` / `mask` の空間 2 軸が一致することを契約検査で突き合わせる
  （torchvision の meta カーネルは `offset.shape[-2:]` を出力形にするが、同じ事実を 2 か所から
  取ると「offset だけ形が違う IR」が素通りする）。
- **低精度重み格納の適格外**（`WEIGHT_SLOTS` / `WEIGHT_CHANNEL_AXES` に載せない）。
  BiRefNet の量子化は現時点で需要が無く、i8 の scale 軸取り違えは ADR 0024 決定 6 が
  記録した沈黙誤値を再生産する。需要が出たら 1 行ずつ足す。

### 2. カーネルは直接畳み込み 1 本（implicit GEMM は書かない）

1 invocation = 出力 1 要素の grid-stride（full-write — ADR 0014）。ADR 0024 の
implicit GEMM 骨格は B タイルの gather を 1 読み → 4 読みに替えるだけで流用できるが、
**本 ADR では書かない**: ①新規原子には対になる既存経路が無く、igemm を先に書いても
A/B オラクルにならない（オラクルは決定 4 が別に立てる）②性能は未計測で、2 本目を
入れる根拠が無い（性能の候補は perf-ledger 起票が先）。

- 縮約順は `(ic, kh, kw)` **昇順の逐次**で、conv2d 直接カーネルと厳密に同じ入れ子。
- **MUST: `offset` / `mask` の読みを `ic` ループの外へ巻き上げない。** `offset_groups == 1`
  なら巻き上げは意味論的に等価だが、そのためにループを `(kh, kw)` 外・`ic` 内へ組み替えると
  **縮約順が変わり、決定 4 の退化ビット一致が失われる**。巻き上げは性能候補であって
  この ADR の範囲ではない（アドレスが `ic` 非依存なので、シェーダコンパイラ側の巻き上げは
  順序を変えないため無害）。
- bias は `acc` の初期値（bias-first — ADR 0024 決定 3）。
- 双線形の 4 隅は torchvision の逐語形: 中心が `(−1, in)` の外なら**タップ全体が 0**、
  内側でも範囲外の隅は**その隅だけ 0**（border clamp ではない）。重みと加算順は
  `w1·v1 + w2·v2 + w3·v3 + w4·v4`（`w1 = λy0·λx0` … の順）で固定。
- mask は**補間の後・重みの前**に掛ける（`(m · v) · w`）。torchvision が im2col バッファへ
  `mask · bilinear(…)` を書き、GEMM が `w × col` を取る形と積の括り方まで一致する。

### 3. 決定性は保てる（forward に atomics は無い）

1. 出力要素 `(b, oc, oy, ox)` は 1 スレッドが排他所有する純 gather。torchvision の forward
   （`deformable_im2col_kernel`）も atomics 0 件で、非決定なのは backward の
   `deformable_col2im_kernel` だけ。`scatter_add` を避ける理由（WGSL に `atomic<f32>` が
   無い）は当たらない。
2. offset は実行時値だが、それが決めるのは**どのアドレスを読むか**だけ。縮約順
   `(ic, kh, kw)` 昇順と 4 隅の加算順はプログラム順で静的に固定されており、丸め列は
   入力にのみ依存する。
3. 非決定になる 2 条件（同一出力アドレスへの複数スレッド加算 / 実行時に変わる縮約順）を
   どちらも踏まない。

### 4. ビット同一のオラクルは「退化 = conv2d」に置く

新規原子なので A/B の相手がいない。代わりに**退化ケースを既存 op に落とす**:

- **`offset` 全 0・`mask` 全 1 なら、素の `conv2d`（同じ weight / bias / padding・stride 1・
  dilation 1・groups 1）と Uint32 でビット一致する。** 根拠: ①λ が厳密に 0 なので
  `w1 = 1, w2 = w3 = w4 = 0` となり `1·v + 0 + 0 + 0 = v`（f32 で厳密）②`m = 1.0` の積は
  厳密③範囲外タップは deform が `0` を**加算**し conv2d は**読み飛ばす**が
  `fl(s + 0) = s`（差は符号付きゼロのみ — ADR 0024 決定 3 が記録した唯一の数値差分と同型）
  ④縮約順が決定 2 のとおり同一。
  実 GPU 門: `packages/runtime/tests/gpu_deform_conv2d_test.ts`。
- **主門は torch（torchvision）出力を焼いた golden** `deform_conv2d_block`
  （非対称形 Cin ≠ Cout / Kh ≠ Kw / padding の H≠W）。オフセットは正負・非整数・
  境界外を跨ぐ範囲で生成し、ゼロ埋めの 2 段（中心の早期 0 と 4 隅個別の 0）を必ず踏ませる。
- CPU 参照は**カーネルと別形**で書く（4 重ループ + タップを座標で直に回す）。平面添字の
  畳み方を共有しないのは、軸の取り違えを両側で相殺させないため。

### 5. offset の NaN / 巨大値 — 範囲判定は正の形・NaN は伝播させる

`i32(floor(y))` は値域外・NaN で未定義（[limitations.md](../limitations.md)）。torch の C++ も
`(int)floor(NaN)` の UB を持つので「torch と一致」は逃げにならない。

1. **範囲判定は正の形で書く**: `y > −1.0 && y < f32(H) && x > −1.0 && x < f32(W)`。
   有限値では torchvision の否定形（`h <= −1 || height <= h` で早期 0）と厳密に同値で、
   巨大値（`±1e30`）も torch と同じく 0 タップになる。この判定を通った後の
   `floor(y) ∈ [−1, H−1]` は i32 で必ず表現できるので、cast の未定義は**構造的に**消える。
2. **NaN は 0 に落とさず出力へ伝播させる**（ADR 0020 の系譜）。上の正の形は NaN で
   false に落ちるため、そのままだと「NaN の offset が黙って 0 寄与になる」= 沈黙誤値に
   なる。範囲外と NaN をビット列判定（`(bitcast<u32>(v) & 0x7fffffff) > 0x7f800000`）で
   区別し、NaN なら出力要素を NaN にする。`clamp` に流す案は ADR 0020 が根治した
   「ドライバの max が NaN を飲む」を再演するので採らない。
3. **範囲外の添字は契約違反ではない**。gather / embedding の「範囲外 = NaN 汚染」
   （limitations.md）とは別で、deform の境界外サンプルは**正常な意味論（0）**。語彙の中で
   2 つの規約が併存するので、op の doc で明示的に切り分ける。

### 6. 却下: `grid_sampler_2d` 経由の分解

`offset` を grid へ整形 → `grid_sampler_2d` → mask 乗算 → `conv2d(stride=(Kh,Kw))` は
厳密に等価で、しかも `aten.grid_sampler_2d` は **Core ATen 内**（= 第 1 層・ADR 不要）という
手続き上の利点がある。それでも採らない:

- **中間が im2col そのもの**（`[B, Cin, Hout·Kh, Wout·Kw]`）。BiRefNet_HR の
  `decoder_block1`（512²・Cin=64）で **k=7 → 3.29GB / k=3 → 604MB**。ADR 0024 が im2col
  実体化を却下した 3.62GB と同水準で、却下理由がそのまま当たる（1024² 運用でも
  k=7 で 822MB）。
- ノード数が 20 → 約 280（1 箇所あたり整形 permute 2 + reshape 3 + 正規化 4 + 本体 2）。
  permute は実体化コピーなので offset の整形だけで k=7 の 102.8MB × 2 が別途乗る。
- **混成案でもコミットメントは減らない**: 4 分岐のうち k=1 の 2 本は分解でも容量が問題に
  ならないが、残る k=3 / k=7 は結局 deform 本体を要求する。
- 正規化座標（`[−1,1]` ↔ ピクセル）の往復で丸めが 1 段増える（torch 突合門で差が出うる —
  未実測）。

`grid_sampler_2d` 自体は将来 STN / 光流 warp の需要が出た時に**独立に**判断する
（op-vocabulary.md の系統 3 の保留のまま — 本 ADR は解除しない）。

### 7. torchvision をエクスポータの基本依存へ上げる

op が `torchvision` 名前空間に登録されるため、①ハンドラのキー
（`torch.ops.torchvision.deform_conv2d.default`）を書くには import が要る
②golden 生成（= `uv run pytest` が毎回再生成する）にも要る。条件付き import で
soft 依存にすると「表に無い op として落ちる env」と「通る env」が分岐するので、
`pyproject.toml` の base deps へ移す（`anima` / `siglip2` グループの記載はそのまま —
グループ内の重複は害が無く、消すと「なぜ要るか」の記録が消える）。

## Consequences

- 契約 1 セット（`OP_CONTRACTS` + `karume/ops.py` + `shapes.py` +
  `fixtures/op-contracts.json` + CPU 参照 + golden COVERAGE）+ カーネル 1 本 +
  エクスポータハンドラ 1 本。既存 IR・既存資産への影響はゼロ（新しい op 名が増えるだけ）。
- **回収範囲は BiRefNet 一族 1 アーキテクチャ**（公式 8 + Lucida = `BiRefNet_HR` と
  アーキテクチャ完全同一の重み差し替え）。恒久コミットメントの対価としては狭いが、
  この一族に他の逃げ道は無い（`lite` も `dec_att` は同じ）。
- BiRefNet_HR を 2048² で実際に動かすには、本 op とは別に ①`conv2d` の dispatch 上限
  （limitations.md の 65,536 > 65,535）②モデル自身の 3.22GB 中間 ③BatchNorm2d の
  畳み込みパス が要る。本 ADR はどれにも触れない。
- 性能は未計測。`decoder_block1`（512²・k=7・Cin=64・Cout=256）は 1 出力要素あたり
  offset 2 + mask 1 + 入力 4 読みで、素の conv2d の約 3.5 倍のロードになる。実測が要るなら
  perf-ledger へ起票する（implicit GEMM 化と offset 読みの巻き上げが候補 — どちらも
  決定 2 / 決定 4 の縮約順を動かすので、退化ビット一致門の再設計とセット）。
- known-issues の「層分類が自明にならない」節の穴 1（Core ATen 外・モデル由来の原子）は
  本 ADR が**暫定運用のまま 1 例を通した**記録になる。ADR 0043 本文への反映（層定義の
  改訂）は同節が予告するとおり整理タイミングで行う。
