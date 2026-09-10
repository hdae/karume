/**
 * 固定 prompt の組み立ての単体検証（GPU 不要・素材はリポ内の文書）。
 *
 * 見る 4 点:
 *
 * 1. **切るのは段落境界** — 上限手前の最後の空行で切れること・境界が無ければ落ちること
 *    （文字数で切ると生成が退化列になり、prompt 側の理由で受理率が動く）
 * 2. **4 種の形** — role の並びと末尾の指示文（`extract` と `summarize` の差は依頼文だけ）
 * 3. **効かないノブは拒否** — 文書を読まないワークロードに `documentChars` を渡したら落ちること
 * 4. **warm の追記列** — 12 本・空文字なし・互いに違う・最初の発話とも違う（同じ発話を追記すると
 *    model が前の答えを写して受理率が跳ねる）・文書系は落ちること
 *
 * NOTE: リポの慣習に合わせて `Deno.test`（文脈）+ `t.step`（振る舞い）で書く
 * （`@std/testing/bdd` はこのリポの依存に無い）。
 */

import { assert, assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import {
  buildWorkload,
  readWorkloadDocument,
  truncateAtParagraph,
  warmFollowUps,
  WORKLOAD_NAMES,
} from "./workloads.ts";

const PARAGRAPHS = "first paragraph\n\nsecond paragraph\n\nthird paragraph\n";

Deno.test("文書の切り詰め", async (t) => {
  await t.step("上限を超える文書は上限の手前の最後の段落境界で切れる", () => {
    // "first paragraph" = 15 文字・境界は 15..16、次の境界は 33..34。上限 33 では手前の境界だけが
    // 丸ごと内側に入るので、切り口は最初の段落の末尾になる。
    assertEquals(truncateAtParagraph(PARAGRAPHS, 33), "first paragraph");
  });

  await t.step("上限以内なら末尾の空白だけ落として丸ごと返す", () => {
    assertEquals(truncateAtParagraph(PARAGRAPHS, 1000), PARAGRAPHS.trimEnd());
  });

  await t.step("上限の手前に段落境界が無ければ落ちる（黙って途中で切らない）", () => {
    assertThrows(
      () => truncateAtParagraph(PARAGRAPHS, 10),
      Error,
      "段落境界",
    );
  });

  await t.step("上限が 1 以上の整数でなければ落ちる", () => {
    assertThrows(() => truncateAtParagraph(PARAGRAPHS, 0), Error, "1 以上の整数");
  });
});

Deno.test("4 種のワークロード", async (t) => {
  await t.step("extract: 前置き + 文書 + 抽出の依頼文の user 発話 1 本", () => {
    const messages = buildWorkload("extract", { documentChars: 4000 });
    assertEquals(messages.length, 1);
    assertEquals(messages[0].role, "user");
    assert(messages[0].content.startsWith("Read the following project document."));
    assert(messages[0].content.endsWith("one per line."));
    assertStringIncludes(messages[0].content, "List every shell command");
  });

  await t.step("summarize: 同じ前置きと同じ文書で、依頼文だけが違う", () => {
    const extract = buildWorkload("extract", { documentChars: 4000 });
    const summarize = buildWorkload("summarize", { documentChars: 4000 });
    const document = truncateAtParagraph(readWorkloadDocument(), 4000);
    assertStringIncludes(extract[0].content, document);
    assertStringIncludes(summarize[0].content, document);
    assert(summarize[0].content.endsWith("Summarize this document in 10 bullet points."));
  });

  await t.step("dialogue: user / assistant / user の 3 発話で、末尾が続きの依頼", () => {
    const messages = buildWorkload("dialogue");
    assertEquals(messages.map((message) => message.role), ["user", "assistant", "user"]);
    assertEquals(messages[2].content, "Expand on the second point with a concrete example.");
    // 2 発話目は固定の返答（3 発話目が指す「2 点目」がそこに固定されていることが再現性の要）。
    assertStringIncludes(messages[1].content, "2. Buffer re-allocation.");
  });

  await t.step("freeform: 短い依頼文の user 発話 1 本", () => {
    const messages = buildWorkload("freeform");
    assertEquals(messages.length, 1);
    assertEquals(messages[0].role, "user");
    assertStringIncludes(messages[0].content, "lighthouse keeper");
  });

  await t.step("system は 1 種も付けない（素の会話だけを測る）", () => {
    for (const name of WORKLOAD_NAMES) {
      const messages = buildWorkload(
        name,
        name === "extract" || name === "summarize" ? { documentChars: 4000 } : {},
      );
      assert(messages.every((message) => message.role !== "system"), name);
    }
  });
});

Deno.test("documentChars の受け付け", async (t) => {
  await t.step("文書を読まないワークロードに渡すと落ちる（効かないノブを黙って受けない）", () => {
    assertThrows(
      () => buildWorkload("dialogue", { documentChars: 4000 }),
      Error,
      "documentChars を受けない",
    );
    assertThrows(() => buildWorkload("freeform", { documentChars: 4000 }), Error);
  });

  await t.step("文書系で省略すると落ちる（既定を持つのは CLI 側）", () => {
    assertThrows(() => buildWorkload("extract"), Error, "documentChars が必須");
  });
});

Deno.test("warm の追記列（ターンごとに違う発話）", async (t) => {
  /** warm を受けるワークロード（文書系は列を持たない）。 */
  const WARM_WORKLOADS = ["dialogue", "freeform"] as const;

  await t.step("自由文 / 対話にそれぞれ 12 本（既定 rounds 3 の plain 13 本ぶん）", () => {
    for (const name of WARM_WORKLOADS) assertEquals(warmFollowUps(name).length, 12, name);
  });

  await t.step("空の発話は無い", () => {
    for (const name of WARM_WORKLOADS) {
      for (const [at, followUp] of warmFollowUps(name).entries()) {
        assert(followUp.trim().length > 0, `${name}[${at}] が空`);
      }
    }
  });

  await t.step("互いに違う（同じ発話を 2 度流すと答えの写しで受理率が跳ねる）", () => {
    for (const name of WARM_WORKLOADS) {
      const followUps = warmFollowUps(name);
      assertEquals(new Set(followUps).size, followUps.length, name);
    }
  });

  await t.step("会話に既に在る発話とも違う（1 本目の繰り返しにならない）", () => {
    for (const name of WARM_WORKLOADS) {
      const conversation = buildWorkload(name).map((message) => message.content);
      for (const followUp of warmFollowUps(name)) {
        assert(!conversation.includes(followUp), `${name}: ${followUp}`);
      }
    }
  });

  await t.step("文書系は落ちる（発話列を持たないことを名指す）", () => {
    for (const name of ["extract", "summarize"] as const) {
      assertThrows(() => warmFollowUps(name), Error, "--warm 非対応");
      assertThrows(() => warmFollowUps(name), Error, name);
    }
  });
});
