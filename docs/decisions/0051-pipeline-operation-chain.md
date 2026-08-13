# 0051: pipeline の公開 API 直列化と非同期 dispose

- Status: accepted
- Date: 2026-08-13
- 関連: ADR [0004](0004-execution-model.md)（flush-before-destroy）/
  [0048](0048-irodori-host-port.md)。起票 = 外部レビュー
  P1-1（`.claude/reviews/2026-08-13_chatgpt-reviews/TRIAGE.md` — git 追跡外）と自前網羅
  レビュー M3-W-5（dispose の in-flight 待ち）の統合。

## Context

3 家族の pipeline（anima / sbv2 / irodori）は「重いグラフは 1 本ずつ張っては畳む」を
モジュール不変条件に持つが、これは 1 回の `generate()` の**内側**でしか成立していなかった。
runtime の直列化は Session 単位（`Session#chain`）と run 区間の device ロックだけで、
`Promise.all([generate(a), generate(b)])` は heavyweight Session を同時常駐させ（anima なら
text 1.4GB + DiT 3.7GB 級）、VRAM 設計の前提を公開 API から破れた。同一 GpuContext では
run が最終的に直列化されるため並行 generate に throughput 利得は無く、常駐だけが重複する。
また同期 `dispose()` は in-flight の生成の下から owned GPU を即 destroy し、
flush-before-destroy（ADR 0004）の前提を破った。

## Decision

1. **models 内共有の直列化鎖 `createOperationChain`**（`src/concurrency/serial.ts` —
   barrel 非公開）を新設し、各 pipeline が 1 本ずつ持つ。`generate()`（irodori は
   `generateLatent` も同じ鎖）は鎖に載せ、並行呼び出しは**待たされて順に**走る。鎖自体は
   決して reject しない（決着だけを次へ渡す — 1 回の失敗が以後を道連れにしない）。
2. **`dispose()` は async 化**（`(): Promise<void>`・`Symbol.dispose` →
   `Symbol.asyncDispose`）。破棄操作も同じ鎖に積むことで「dispose 済みマーク → 先行操作の
   決着待ち → owned GPU destroy」の順序を 1 箇所で決める。2 度目以降は同じ完了を返す。
   dispose 済みの判定は `#disposal !== undefined` の 1 本（派生真偽値を持たない）。
3. dispose 済み後の `generate()` は従来どおり即例外。判定は**呼び出し時点**で行う
   （鎖の中で見ると dispose より前に受理した生成まで巻き添えで落ちる）。

## Consequences

- 公開型の破壊的変更（unreleased につき可）: `dispose(): void` → `Promise<void>`。`using` は
  `await using` へ。examples / e2e テストは追従済み。
- 並行 `generate()` の意味論は「拒否」ではなく「直列実行」— 呼び出し側のリトライ分岐が不要で、
  once-at-a-time の資源保証だけが目的（順序は呼び出し順）。
- runtime の `Session` は `dispose(): Promise<void>` のまま `Symbol.asyncDispose` を持たない
  非対称が残る（別件候補）。
