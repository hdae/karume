# 0020 — GPU の NaN 伝播はビット列判定で保証する

- Status: accepted（2026-08-03・ユーザー裁定 = 5 op 一括の a 案）
- 対象: `clamp` / `clamp_min` / `relu`（elementwise）+ `amax` / `amin`（行 reduce）
- 根拠実測: 2026-08-02（M1-P4 波 1・実 GPU）。`clamp(NaN,-1,1) = -1` /
  `clamp_min(NaN, 0) = 0` / `relu(NaN) = 0` — CPU 参照（NaN を伝播）と乖離し、
  limitations.md の「GPU と CPU 参照で伝播が一致することのみ保証対象」が破れていた。

## 問題の機序

WGSL の比較単体（`select(0.0, 1.0, NaN < m)`）は仕様どおり false になるのに、
`select(x, m, x < m)` **全体**はシェーダコンパイラが `max` イディオムへ畳み、ドライバの
`max` が NaN を飲む。つまり「select 形なら NaN が伝播する」という ADR 0015 / 0017 採択時の
見立ては**この環境では成立しない**（当該 ADR は履歴としてそのまま残し、本 ADR が上書きする）。
`leaky_relu` だけが伝播していたのは**両方の枝に x が現れる**ためで、select 形の効能ではない。

## 決定

1. NaN 判定は**浮動小数の比較を使わず、ビット列**で行う:
   `(bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u`（符号を落として指数部全 1 + 仮数部非 0）。
   整数の `&` と `>` は min/max イディオムへの畳み込みの対象にならない。
2. elementwise は既存の値式を外殻 `select(finite, x, is_nan_bits(x))` で**そのまま包む**
   （非 NaN の生成物は文字列レベルで従来と同一 — 数値回帰ゼロ）。reduce は畳み込み関数を
   `nan_max` / `nan_min` に置換し、**1 スレッド走査と workgroup 木の両段**が同じ関数を通す
   （片段だけでは NaN が identity ±F32_MAX に飲まれる）。
3. パイプラインキーは**族ごと**に改版（`ew:v1→v2` / `reduce:v1→v2`）。op 別の例外表は
   「どの op が今どの版か」の二重管理になるため持たない（WGSL が不変の op はキーが変わる
   だけで、実害はプロセス内キャッシュの初回ミス 1 回）。
4. MUST: 今後の新カーネルでも、**データ値**に `max` / `min` / 比較 select を使う経路で
   torch が NaN を伝播する op は、同じビット列判定で伝播を保証する。
5. 対象外（意図的）: `leaky_relu` は select 形のまま（両枝に x があり伝播する — 実 GPU
   テストで固定済み。生成物を動かさない方を選ぶ）。safe-softmax 内部の max 縮約も素の
   `max` のまま（NaN は後段の `exp(x − amax)` と総和が必ず運ぶため、伝播の有無は torch /
   CPU 参照と一致する）。gather / embedding の範囲外 NaN 汚染（limitations.md）は別裁定。

## 検証

- 生成物の形: packages/runtime/tests/codegen_wgsl_test.ts が「比較ではなくビット列で判定」を 5 op で固定し、
  補助関数が他 op に漏れないことも固定（スナップショット同梱）。
- 実 GPU: packages/runtime/tests/gpu_ops_test.ts が NaN の位置（先頭/中間/末尾・縮約は走査ループ 2 周目
  含む）を変えて CPU 参照との **isNaN パターン一致** + 非 NaN 要素の厳密一致 + 対照要素の
  存在（常時 NaN 実装の除外）を assert。故障注入（relu を素の max へ / amax の畳み込みを
  素の max へ）で赤くなることを実測済み — 修正前の非伝播が本機で再現する検出器である。
- NaN の**ビットパターン**一致は要求しない（GPU の quiet NaN と CPU の NaN のビットが
  一致する保証は無い）。

## 帰結

- limitations.md の「伝播が一致することのみ保証対象」が全 op で再び成立する。
- 上流バグ由来の NaN が clamp 系で有限値に化けて検出が遅れる、という故障モードが消える。
- 残リスク: 外殻の `select` 自体が畳まれて NaN が飲まれる可能性は WGSL 仕様上ゼロでは
  ないが、条件が整数比較になったため現実的には低い（実 GPU テストが恒常の検出器）。
