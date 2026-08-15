# 0052: exporter 成果物の transactional な公開（temp→verify→replace / staging→swap）

- Status: accepted
- Date: 2026-08-13
- 関連: ADR [0005](0005-verification.md)（fail loudly / 検証門）/
  [0041](0041-manifest-v2.md)（dist の組み立てと検証）。起票 = 外部レビュー EXPORTER-004 /
  EXPORTER-002 / B2-stale（`.claude/reviews/2026-08-13_chatgpt-reviews/TRIAGE.md` —
  git 追跡外）。

## Context

`export_to_file` は final path を直接 truncate してから書き、検証（`verify_model`）は書いた
**後**だった — 書き込み途中の故障・検証失敗のどちらでも、手元の正常な成果物が失われ、不正な
新ファイルが final path に残る（docstring の「書けたが読めないファイルを配布物として残さない」
契約が commit の形で担保されていなかった）。`karume dist` の組み立ても既存 destination を
in-place 更新（unlink → copy・`karume.json` は最後）しており、I/O 故障で「旧 manifest +
新旧混在ツリー」が残り得た。さらに A+B を組んだ出力へ A だけを再組み立てすると B の残骸が
残り、GB 級のコピーを終えた後の `verify_dist` で宣言外ファイルとしてようやく落ちていた。

## Decision

1. **単体 export**: `export_to_file` は同一ディレクトリの一時ファイル
   （`<final 名>.<uuid>.partial`）へ `write_model` → `verify_model(temp)` →
   `os.replace(temp, final)`。失敗時は temp を捨て、既存 final は 1 バイトも変えない。
   原子化はこの層のもの — `emit.write_model` 直呼びは「与えられた path へ書く」下層のまま。
2. **dist 組み立て**: `assemble_family` は staging（`<出力名>.staging`・同じ親 = 同一 FS で
   rename が原子的）へ全て生成（配置 → 共有畳み込み → `karume.json` → `verify_dist` →
   `README.md`）し、通ってから rename 2 回（既存を `.old` へ退避 → staging を据える →
   `.old` 破棄）で差し替える。失敗（Ctrl-C 含む）は staging だけを消し、既存配布形は不変。
   前回の中断が残した staging / `.old` は黙って捨てて作り直す。
3. **再組み立ては丸ごと置換**: `out_dir` の元の中身は 1 つも引き継がない — A+B → A の
   再組み立ては B が消える（宣言外ファイルで遅く落ちる従来挙動は削除）。モデルカードも
   staging の中で「検証を通った manifest」から描くので、据わった配布形はカードまで揃う。

## Consequences

- dist のディスクピークは swap の瞬間まで新旧ツリーが併存するぶん**約 2 倍**
  （docs/limitations.md に記載 — by-design のトレードオフ）。
- 出力先の親に一時的に `<名前>.staging` / `<名前>.old` が現れる（models/ は git 追跡外）。
- `place_file` 等の `dest.unlink(missing_ok=True)` は staging が常に空から作られるため
  実質到達しない防御になった（撤去は別コミット候補）（2026-08-16 撤去済み）。
