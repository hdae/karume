# 0010 — 記号依存定数の Tmax 畳み込みと記号 prefix スライス

- Status: accepted（2026-08-02）
- 根拠資料: [../research/2026-08-02-deberta-front-recon.md](../research/2026-08-02-deberta-front-recon.md)
  （相対位置バケット表が未対応 op 27 種中 14 種の発生源）

## 決定

DeBERTa の相対位置バケット表（arange / sign / log / ceil / clamp / 比較 / bitwise_and …の
T 依存部分木）は、**エクスポータが Tmax（モデルの最大系列長）で実評価して i32 定数に焼き、
実行時は記号 prefix スライスで切り出す**。

- IR v1 を改訂し、**initializer の意味論 i32 と格納 dtype `i32`（生の int32）**を追加する
  （現行の「格納語彙は f32 値の符号化」の明示的な例外）。
- IR op **`sym_prefix_slice`** を追加: attrs `{ sym, slices: [{dim, coeff, offset}] }`。
  次元言語 `coeff·sym+offset`（ir-v1.md）をそのまま流用する。
- 畳み込みの適格判定は「Tmax 実評価が prefix と可換」であること。**allowlist 掲載だけでは
  担保しない** — エクスポータが **2 点評価（異なる 2 つの T で評価して prefix 一致を実測）**で
  検査し、不一致は fail loudly（プロトタイプで実証済みの機構）。

## 検討した代替案

1. **GPU で整数演算をそのまま実行** — IR 改訂は不要だが、i32 elementwise・比較・sign・
   ceil・log・clamp の実装に加え、台帳の未決 3 件（WGSL の NaN 比較 / sign(NaN) /
   max の NaN 非伝播）を全部 load-bearing にする。さらに**バケット境界の 1 ulp 差が
   gather 添字の 1 ずれになる**バグクラスが恒久的に残る。
2. **ホスト側で事前計算して追加入力にする** — 実装最軽量だが「モデルファイル 1 個で完結」と
   薄い公開面（ADR 0008）に反し、モデル固有ロジックが全呼び出し側へ漏れる。
3. **Tmax 畳み込み + prefix スライス（採択）**。

## 採択理由

1. バケット境界バグクラスの**構造的排除** — 表は torch 自身が計算するので golden と同一
   意味論。GPU 再実装の丸め差が添字に混入する経路が存在しない。
2. 未対応 op が 27 種 → 実質 13 種前後に半減し、WGSL NaN 系の未決を P2 の経路に載せない。
3. プロトタイプが同方式で DeBERTa E2E を実証済み（可換性の 2 点実測検査込み）。
4. 未リリースにつき IR v1 非互換改訂のコストは仕様書と実装の同時更新のみ（ADR 0003 の
   改訂手順どおり。シム・移行は作らない）。

## 帰結

- ir-v1.md の改訂（i32 initializer / storage `i32` / `sym_prefix_slice`）は実装波と同一
  コミットで行う。
- Tmax はモデル属性（DeBERTa は 512）。Tmax を超える実行時系列長は束縛検査で fail loudly。
- prefix 非可換な部分木は畳まず、未対応 op として全件列挙に出す（黙って残さない）。

## 追記

- 2026-08-13: **可換性の担保機構を 2 段へ改訂**（外部レビュー P0-1 の消化）。従来の「2 点実測
  （Tmax / Tmax−1）が担保する」は不健全だった — `scalar_tensor(T)` でシンボルがテンソル
  **データ**へ昇格した部分木（例 `(T−Tmax)(T−(Tmax−1))` — 両評価点で 0）は 2 点で偶然一致し、
  export・verify_model とも緑のまま他の T で沈黙誤値になる（反例を実行して確認済み）。現行は
  ① シンボルの**消費位置**で拒否（`SYMBOL_EXTENT_ARGS` — extent = 長さ・形状の引数位置だけを
  ホワイトリストで許し、値位置へ届いた部分木は畳まず通常 lowering へ落とす = 語彙外なら
  export ごと拒否）② 2 点実測は防波堤として残す（比較は `torch.equal` から**ビット厳密**へ
  強化 — `-0.0 == 0.0` の素通りを塞ぐ）。本文の「2 点実測検査込み」はこの改訂で読み替える。
