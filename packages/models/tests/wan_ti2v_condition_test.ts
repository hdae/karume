// Wan2.2 の I2V の DiT の条件づけのホスト関数（`src/wan/dit-loop.ts` の `wanConditionMask` と
// `withWanFirstFrameCondition` — ADR 0121 決定 11）の門。純関数なので GPU も資産も要らない。
//
// 縛るのは 3 点:
//
// ① 条件マスクの防御の枝（fail loudly）— 入力の形 "t2v" に条件を渡す・時間の patch ≠ 1 で I2V を組む。
//    今の公開 API からは届かない（計画が 2.1 の image を先に拒む・`wanDitPatch` の時間の patch は 1 固定）
//    ので、配線を変えたときに黙って通らないことをここで固定する。
// ② 条件マスクの真の範囲が先頭の潜在フレームのトークン `[0, rows·cols)` だけであること。
// ③ 置き換えが上流 diffusers の式 `(1 − m)·condition + m·latents` を逐語に計算すること — 符号付きゼロ
//    まで上流と同じ値になる。「先頭フレームを写す」への簡約は ±0 の要素で上流と割れる（下の期待値は
//    f32 の上流の式の値を手で引いたもの）。I2V の sha 行も ±0 の差を拾うが、遠くで赤になると帰属に
//    手間がかかるので、ここで直接縛る。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { wanConditionMask, withWanFirstFrameCondition } from "../src/wan/dit-loop.ts";
import { type WanPatchGeometry, wanTokenGrid } from "../src/wan/dit-tokens.ts";

const OWNER = "test";

/** I2V の patch（時間 1・空間 2×2 — `wanDitPatch` と同じ刻み）。 */
const PATCH: WanPatchGeometry = { channels: 4, patchFrames: 1, patchHeight: 2, patchWidth: 2 };

/** 潜在 `[4, 3, 4, 6]` → トークン格子 3 × 2 × 3（先頭の潜在フレームは 6 トークン）。 */
const GRID = wanTokenGrid([4, 3, 4, 6], PATCH);

/** 符号付きゼロを区別して表示する（`assertEquals` の差分表示は −0 と +0 を同じ "0" と書く）。 */
const signed = (value: number): string => (Object.is(value, -0) ? "-0" : String(value));

/** `Object.is` で符号付きゼロまで一致することを確かめる。 */
const assertSameValue = (actual: number, expected: number, label: string): void =>
  assert(Object.is(actual, expected), `${label}: ${signed(actual)}（期待 ${signed(expected)}）`);

describe("wanConditionMask（I2V の条件マスク）", () => {
  it("入力の形 't2v' に条件の潜在を渡すと、条件マスクの入力が無いと投げる", () => {
    assertThrows(
      () => wanConditionMask("t2v", GRID, PATCH, true, OWNER),
      Error,
      `${OWNER}: 入力の形 't2v' の DiT は条件の潜在を受けない`,
    );
  });

  it("時間の patch が 1 でない I2V は、先頭の潜在フレームがトークンのフレームに揃わないと投げる", () => {
    const patch: WanPatchGeometry = { ...PATCH, patchFrames: 2 };
    // 潜在の 4 フレームを時間の patch 2 で割った格子（patch と格子の整合は保ったまま patch だけを変える）。
    const grid = wanTokenGrid([4, 4, 4, 6], patch);
    assertThrows(
      () => wanConditionMask("ti2v", grid, patch, true, OWNER),
      Error,
      `${OWNER}: 時間の patch 2 では先頭の潜在フレームがトークンのフレームに揃わない`,
    );
  });

  it("I2V は先頭の潜在フレームのトークン [0, rows·cols) だけが真で、他は全て偽", () => {
    const mask = wanConditionMask("ti2v", GRID, PATCH, true, OWNER);
    assertEquals(GRID.rows * GRID.cols, 6, "前提: 先頭の潜在フレームは 2 × 3 トークン");
    const expected = new Uint32Array(GRID.count);
    expected.fill(1, 0, 6);
    assertEquals(mask, expected);
  });
});

describe("withWanFirstFrameCondition（モデル入力の先頭フレームの置き換え）", () => {
  it("潜在の要素数が潜在の形と合わなければ投げる", () => {
    assertThrows(
      () => withWanFirstFrameCondition(new Float32Array(7), new Float32Array(4), [1, 2, 1, 4]),
      Error,
      "条件の置き換え: 潜在 7 要素・条件 4 要素が潜在の形 [1, 2, 1, 4]（条件は 1 フレーム）と合わない",
    );
  });

  it("条件が 1 フレームの要素数でなければ投げる（2 フレームぶんの条件を渡す）", () => {
    assertThrows(
      () => withWanFirstFrameCondition(new Float32Array(8), new Float32Array(8), [1, 2, 1, 4]),
      Error,
      "条件の置き換え: 潜在 8 要素・条件 8 要素",
    );
  });

  it("符号付きゼロまで上流の式の値になる（先頭フレームを写すだけの値とは割れる）", () => {
    // 潜在 `[C=1, F=2, H=1, W=4]`・条件 `[1, 1, 1, 4]`。条件は全フレームへ broadcast される。
    const condition = Float32Array.of(-0, -0, -2, 3);
    const latents = Float32Array.of(
      // 先頭フレーム（m = 0 → 1·cond + 0·lat）
      +0,
      1,
      5,
      7,
      // 2 フレーム目（m = 1 → 0·cond + 1·lat）
      11,
      13,
      -0,
      -0,
    );
    const out = withWanFirstFrameCondition(latents, condition, [1, 2, 1, 4]);
    // cond = −0・lat = +0 → −0 + 0·(+0) = −0 + +0 = +0（写すと −0）。
    assertSameValue(out[0], +0, "先頭フレーム [0]");
    // cond = −0・lat = 1 → −0 + +0 = +0（写すと −0）。
    assertSameValue(out[1], +0, "先頭フレーム [1]");
    // 非ゼロの要素は、先頭フレームが条件・2 フレーム目が潜在そのもの。
    assertSameValue(out[2], -2, "先頭フレーム [2]");
    assertSameValue(out[3], 3, "先頭フレーム [3]");
    assertSameValue(out[4], 11, "2 フレーム目 [0]");
    assertSameValue(out[5], 13, "2 フレーム目 [1]");
    // cond = −2 < 0・lat = −0 → 0·(−2) + −0 = −0 + −0 = −0。
    assertSameValue(out[6], -0, "2 フレーム目 [2]");
    // cond = 3 > 0・lat = −0 → 0·3 + −0 = +0 + −0 = +0（潜在をそのまま残すと −0）。
    assertSameValue(out[7], +0, "2 フレーム目 [3]");
  });
});
