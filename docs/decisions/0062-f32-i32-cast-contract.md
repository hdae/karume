# 0062: f32 → i32 cast は値域内 truncate を保証し、値域外・NaN は未定義とする

- Status: accepted（2026-08-15 — 既存裁定の正本化。これまで正本が `docs/limitations.md` に
  あった）
- 関連: ADR [0009](0009-dtype-i32-bool.md)（意味論 i32 の解禁 — 「torch 準拠の truncate」を
  契約に明記したが、値域外・NaN の扱いは裁定していなかった）

## Context

- WGSL の `i32(f32)` も torch の `.to(int)` も、値域外・NaN 入力の結果は実装依存で、両者の
  一致は仕様上保証されない。
- 「未対応・想定外は fail loudly」の不変条件に素直に従うなら要素ごとの値域検査だが、cast は
  本来メモリ律速の 1 パス — 検査を足すと演算律速へ性質が変わり、全 cast 利用点が恒常コストを
  払う。

## Decision

- **値域内は torch 準拠の truncate（0 方向切り捨て）で一致を保証する**（ADR 0009 の契約を
  再確認）。
- **値域外・NaN は未定義とする**（GPU と CPU 参照・torch の間で一致を主張しない）。
- **要素ごとの値域検査は意図的に入れない**（性能特性を変えるため — fail loudly 不変条件の
  明示的な例外）。範囲外になりうる値は**呼び出し側・モデル側で先に clamp する**ことを契約に
  する。

## Consequences

- 「fail loudly の例外」を明文で持つ唯一の cast 経路になる — 新しい cast 変種（i32 → f32 は
  対象外・全値表現可能）を足すときは本 ADR と同じ軸（律速の性質が変わるか）で個別裁定する。
- `docs/limitations.md` の該当節は本 ADR を指す要約になる。
