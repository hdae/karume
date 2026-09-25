// 実行形ノブの受理集合検査（{@link assertExecutionKnobs}）の門 — GPU に触れない純関数なので、
// アダプタ無し環境でも回帰を撃てる位置に置く。`createSessionFromContainer` 経由の GPU テストは「実構築でも
// 拒否される」1 本だけを残し、値の総当たりはここが正本。

import { assertEquals, assertThrows } from "@std/assert";
import { assertExecutionKnobs } from "../src/runtime/session-build.ts";
import { ExecutionError } from "../src/runtime/plan.ts";

/**
 * **型の外から来る呼び手**（JS の消費者・改名前の綴りを残した古いビルド）を再現する。TS の型は
 * この経路を 1 つも守らないので、受理集合の検査はここでしか撃てない。
 */
const jsCallerAssert = assertExecutionKnobs as unknown as (
  ...knobs: readonly unknown[]
) => void;

/** 受理集合の中の綴り（この並びが {@link assertExecutionKnobs} の引数順）。 */
const VALID: readonly [string, string, string, string, string] = [
  "f32",
  "f32",
  "f32",
  "parallel-fused",
  "parallel",
];
const NAMES: readonly [string, string, string, string, string] = [
  "linearCompute",
  "attentionCompute",
  "attentionScoreStorage",
  "stateAttentionReduce",
  "linearGemvReduce",
];

Deno.test("受理集合の中の綴りは 5 ノブとも通る（下が「何を渡しても落ちる」ではない証明）", () => {
  assertExecutionKnobs("f32", "f32", "f32", "parallel-fused", "parallel");
});

Deno.test("実行形ノブの非文字列は受理集合を引く前に落ち、診断で利用者の変換を呼ばない", () => {
  for (const [index, name] of NAMES.entries()) {
    let conversions = 0;
    // 受理集合の中の綴りへ**変換できる**オブジェクト。`Object.hasOwn` のキー変換や診断の
    // 文字列化が走れば通ってしまうので、呼ばれた回数そのものを検査する。
    const object = {
      [Symbol.toPrimitive](): string {
        conversions++;
        return VALID[index];
      },
      toJSON(): string {
        conversions++;
        throw new Error("診断で利用者の変換を呼ばない");
      },
    };
    for (const value of [[VALID[index]], object, false, 0, 1n, Symbol(VALID[index])]) {
      const knobs: unknown[] = [...VALID];
      knobs[index] = value;
      const error = assertThrows(
        () => jsCallerAssert(...knobs),
        ExecutionError,
        name,
      );
      // 型名だけを出す（利用者の `toString` の戻り値を載せない）。
      assertEquals(error.message.includes(`${name}: ${typeof value}`), true, error.message);
      assertEquals(error.message.includes("実行形ノブ 1 本"), true, error.message);
    }
    assertEquals(conversions, 0, `${name}: 利用者の変換が呼ばれた`);
  }
});

Deno.test("union 外の綴りは全件列挙して 1 回で落ちる", () => {
  // `"i8a8"` は 0.5.0 で `"a8"` へ改名した綴り（ADR 0074 決定 3・互換シム無し）。
  const error = assertThrows(
    () => jsCallerAssert("i8a8", "f32", "s16", "parallel-fused", "parallel"),
    ExecutionError,
    "実行形ノブ 2 本",
  );
  assertEquals(error.message.includes(`linearCompute: "i8a8"`), true, error.message);
  assertEquals(error.message.includes(`attentionScoreStorage: "s16"`), true, error.message);
  assertEquals(error.message.includes("'f32' / 'a8' / 'f16'"), true, error.message);
});
