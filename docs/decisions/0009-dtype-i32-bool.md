# 0009 — 意味論 i32 / bool の実行解禁と i64 境界正規化

- Status: accepted（2026-08-02）
- 根拠資料: [../research/2026-08-02-deberta-front-recon.md](../research/2026-08-02-deberta-front-recon.md)
  （DeBERTa front の実測 — 入力 i64・bool マスク経路は入力値依存で畳み込み不能）

## 決定

- **意味論 i32 / bool の実行を解禁する**（M1-P2）。IR の宣言語彙（ir-v1.md）には両者とも
  既在 — 変えるのは実行系: 入力転送（`RUNTIME_SUPPORT.io`）、op 契約の dtype 集合、
  elementwise codegen の要素型パラメタ化。**要素型は WGSL 正準化キーに含める**
  （codegen 決定性の不変条件はそのまま）。
- **bool の GPU 格納は u32 の 0 / 1**（ストレージバッファに 1bit 型は無い。
  プロトタイプで実証済みの規約）。
- **IR に i64 は導入しない。** torch の既定整数 i64 は**エクスポータ境界で i32 へ正規化**し、
  値域検査（|x| > 2^31−1 で fail loudly）を必ず伴う。64bit の無い WebGPU 世界への変換点を
  エクスポータ 1 箇所に固定する。
- 新 op は **cast**（f32/i32/bool 間。f32→i32 は torch 準拠の truncate を契約に明記）と
  **bitwise_not**（bool 否定 — attention mask 反転の実行に必須）。
  `where` は現時点で実測グラフに実行対象として現れないため**追加しない**
  （語彙 allowlist 凍結 — ADR 0007。必要になった実測グラフが追加の根拠）。
  **→ この 1 点は ADR [0015](0015-conv-family-extension.md)「語彙追加の残り」で撤回**
  （実測グラフが出て `where` を追加 — 現行は `WHERE_OP` として実装済み）。
- **公開 `Tensor` を dtype 判別ユニオンへ改訂**（ADR 0008 の部分改訂）:
  `data` は f32 = Float32Array / i32 = Int32Array / bool = Uint8Array ではなく
  **Uint32Array**（GPU 格納と同じ 0/1）。入出力で対称に扱う。

## 検討した代替案

1. **IR に i64 を追加**（プロトタイプ方式）— WebGPU に 64bit 整数バッファが無く、転送時に
   必ず i32 変換が入る。語彙が増えるだけで実行表現力は増えない。
2. **添字を f32 で運ぶ** — 2^24 超の添字で静かに精度が落ち、fail loudly の原則に反する。
   意味論の濁りが契約表の検査可能性を壊す。
3. **i32 境界正規化（採択）** — 語彙最小・検査可能・変換点 1 箇所。DeBERTa の実値域
   （T ≤ 512、vocab 22012）は i32 に十分収まる。

## 帰結

- TS（`packages/runtime/src/ops.ts`）と Python（`karume/ops.py`）の契約表は 1 セットで拡張し、
  op ごとに許容 dtype 集合を宣言する（f32 専用 op は f32 のまま — 一括解禁はしない）。
- readback / アップロードは dtype ディスパッチになる（要素 4 バイトは全型共通）。
- i32 / bool の演算を新設するたび、CPU 参照と golden を同時に足す（ADR 0005）。
