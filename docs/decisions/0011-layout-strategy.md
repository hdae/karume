# 0011 — レイアウト実行戦略: エイリアス reshape + strided 実体化コピー

- Status: accepted（2026-08-02）
- 根拠資料: [../research/2026-08-02-deberta-front-recon.md](../research/2026-08-02-deberta-front-recon.md)
  （2 層で view 28 / permute 22 / clone 24 / expand 5 / squeeze 5 / unsqueeze 13 — M0 には
  レイアウトの概念が皆無）

## 決定

- **要素順を変えないもの（view / squeeze / unsqueeze）**は IR op `reshape` 1 本に正規化し、
  実行は**バッファ別名**（コピーなし）。アリーナの retain/release は別名参照を通常の
  uses として計上する（エイリアス対応を ADR 0004 の不変条件に追記）。
- **要素順が変わる・実体が要るもの（permute / expand / slice）**は
  **strided 実体化コピー 1 カーネル族**で実行する: 入力を `(offset, strides[STRIDED_RANK])`
  で読み、出力は常に連続。expand は stride 0。**STRIDED_RANK = 4**
  （DeBERTa は全値 rank ≤ 4。rank ≥ 5 はエクスポータの rank 下げ正規化で先に潰す —
  プロトタイプ前例）。
- **elementwise / reduce / matmul の codegen は変更しない**（入力は連続前提のまま）。
  消費側カーネルへ strided 読みを融合してコピーを消す最適化は、**IR 語彙を変えずに実行
  戦略の差し替えだけで到達できる**ため、perf マイルストーンの upgrade path として明記して
  先送りする。
- 追記（2026-08-02 ユーザー裁定・同日更新）: 案 B は正しさ優先の**暫定戦略**として受理
  された（IR 語彙が同一のため B の実装が無駄になることはない）。perf マイルストーンの
  優先目標は**「正しくパフォーマンスを出すこと」**であり、案 A（strided 融合読み）は
  有力候補だが唯一の選択肢ではない — 実測でより良い別方針があればそちらを採ってよい。
  検証しつつ段階的に改善する。いずれの方針でも同一 IR のまま golden E2E を回帰の網として
  使う（帰結の項）。

## 検討した代替案

- **案 A: strided 融合読みを最初から全 codegen に入れる**（プロトタイプ方式）。
  コピーが出ず SBV2 以降の性能に有利だが、M0.1 で固めた elementwise / reduce codegen と
  決定性スナップショットを全面書き換えする（P2 最大の blast radius）。
- **案 B: メタ reshape + 実体化コピー（採択）**。

## 採択理由

1. **不可逆性の低い側** — IR 語彙は両案で同一。B から A へは実行戦略の差し替えで移行でき、
   逆方向の学び直しコストが無い。まず可逆な最小を取り、実測（perf マイルストーン）で
   A の採否を裁定する。
2. **blast radius 最小** — M0.1 でレビュー 2 巡を通した codegen と snapshot 基線を温存し、
   P2 の新規リスクをレイアウトカーネル 1 族に閉じ込める。
3. P2 の目標は正しさの E2E。B=1・T ≤ 512 でのコピーは MB 級（gather 添字の expand は
   CSE 後 2 本 ≈ 16MB×2）で、正しさにも VRAM にも実害が無い。

開示: recon は「SBV2 front（rank ≥ 5）まで見ると A の方が負債が小さい」と評価しており、
本 ADR はそこからの**意識的な逸脱**（rank ≥ 5 は両案とも rank 下げ正規化で処理する前提の
ため、A の優位は融合による性能のみ — それは perf マイルストーンの裁定事項）。

## 帰結

- 新 IR op: `reshape` / `permute` / `expand` / `slice`（+ 恒等 clone・恒等 repeat は
  エクスポータの正規化で除去 — IR に持ち込まない）。
  - 追記（2026-08-02 実装時）: **`slice` は見送り**。実測グラフの slice は相対位置バケット表の
    部分木 1 本のみで、ADR 0010 の Tmax 畳み込みで消える側（語彙 allowlist — ADR 0007）。
    strided カーネルは (offset, strides[4]) の形を保っており、将来 slice を族に足す場合の
    可変点は params の offset 1 語のみ。
- エイリアス値の破棄規律（flush-before-destroy / discard）は実体バッファ単位で判定する。
- perf マイルストーンで融合を入れる場合、golden E2E は同一 IR のまま回帰の網になる。
