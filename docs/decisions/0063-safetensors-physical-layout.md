# 0063: safetensors 物理配置の契約 — 隙間なし・要素整列・固定書き出し順

- Status: accepted（2026-08-15 — 既存裁定の正本化。ランタイム・エクスポータ双方に実装済みで、
  これまで cross-package 不変条件の正本が `docs/limitations.md` にあった）
- 関連: ADR [0003](0003-ir-v1.md)（コンテナ形式）/ [0018](0018-f16-weight-execution.md)（f16
  格納 — 偶奇問題の発生元）/ [0019](0019-i8-weight-execution.md)（i8 格納 — 1 バイト要素）/
  実装 = `packages/runtime/src/format/safetensors.ts`（リーダ）・
  `tools/exporter/karume/emit.py`（ライタ）・`karume/verify.py` の `assert_reader_layout`

## Context

- safetensors 仕様自体はテンソルの並び順・整列を規定しない。しかし Karume のリーダは
  TypedArray ビューをファイル buffer 上に直接張るため、**各テンソルの絶対 offset が要素
  サイズに整列していること**が必要になる。要素数が奇数の F16（バイト長 ≡ 2 mod 4）の直後に
  F32 / I32 を置くと offset が 4 の倍数から外れる。
- HF の `safe_open` は整列違反のファイルを**読めてしまう**ので、上流ツールを通すだけでは
  この違反を検出できない — 検出器は自前で持つしかない。

## Decision

- **リーダ（runtime）**: データ節を「隙間なく・要素サイズに整列して」覆うことを要求し、
  違反は `SafetensorsError` で fail loudly（gap・整列・宣言バイト数の厳密一致）。
- **ライタ（exporter）**: 書き出し順を「**F32 → I32 → 偶数要素 F16 → 奇数要素 F16 → I8**」に
  固定する（`emit.py` — `safetensors.torch.save_file` は使わない）。奇数要素の F16・要素数が
  4 の倍数でない I8 は末尾側へ寄る配置で、後続テンソルの整列を崩さない。
- **emit 直後の自己検査**: `verify.assert_reader_layout` がリーダ規則を写した検査を通す
  （書けたのに読めないファイルを配布形に混入させない）。
- GPU 常駐時のゼロ詰め（ADR 0018 / 0019）は**バッファ内**の話で、ファイル上のバイト列は
  詰めない（本契約と独立）。

## Consequences

- リーダ規則とライタ規則は**対**で変える（cross-package 不変条件 — 片側だけの変更は
  `assert_reader_layout` が落とす）。
- 新しい格納 dtype（例: 将来の packed 4bit）を足すときは、整列・並び順・端数の契約を
  **本 ADR の改訂として**先に決める — 1 要素 = 1 payload 要素の前提を破る格納形は、shape と
  バイト数の対応も含めて別 ADR 級（backlog next の packed weight storage）。
- `docs/limitations.md` の該当節は本 ADR を指す要約になる。
