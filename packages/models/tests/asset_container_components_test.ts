/**
 * The check of the component opener of the whole-bytes surface (`from*Assets`)
 * (`assetComponentOpener` in `src/hub/components.ts`).
 * ①② open a Session on a real GPU (each case is explicitly skipped via `ignore` when no adapter
 * is present); ③〜⑥ are the acceptance-side gates and need no GPU.
 *
 * ① **A Session opens from a split-form Record (a key sequence `<role>[i]`)** (X2-101 — "a
 *   distribution form `fromPretrained` can read, `fromAssets` can read too"). The bytes are not
 *   concatenated but opened as the part sequence, so the run's output shows that the part holding
 *   the weight block reached the GPU.
 * ② **Two Sessions in a row open from the same opener** (blocks are re-read on every call — it is
 *   not a sequence that is exhausted after one read).
 * ③ **A gap in the indexes fails loudly** (a Record with only `[0]` and `[2]` is not silently read
 *   as one). Shapes **without** `[0]` (`dit[1]` alone / a single-form key + `dit[1]`) fail at the
 *   same check — a sequence that has at least one indexed key but no `[0]` could skip both the
 *   mixing check and the gap check and become a single-form one.
 * ④ **Mixing a single-form key with `[i]` fails loudly** (which one is authoritative cannot be
 *   decided).
 * ⑤ **A shape with no container of the component at all fails with the spellings of both forms** (a
 *   misspelling does not turn into the distant diagnostic "asset missing").
 * ⑥ **A sequence missing a weight part fails at the opener** (no silent partial Session). What
 *   rejects it is the container reader (the comparison with the declared part count), and it needs
 *   no GPU.
 *
 * For ③④⑤ the checks go as far as the diagnostic containing the word `part` and **a listing of the
 * keys present** (the convention of the existing asset diagnostics) — the failure itself is a sign
 * that the way the keys are built is broken, and it is useless unless the reader can compare it
 * with what is there.
 *
 * NOTE: `assetComponentOpener` **opens every component before opening one** (the only way to return
 * a synchronous opener is to open first and hand out after), so these fail **when the opener is
 * built**, not at `open(key)`.
 */

import { test } from "vite-plus/test";
import { assertEquals, assertRejects, assertStringIncludes } from "@std/assert";
import { acquireGpu, ContainerFormatError } from "@karume/runtime";
import { openerOf, part, parts, single } from "./helpers/asset-container-fixture.ts";
import { LINEAR_PROBE_Y, runLinearProbe } from "./helpers/container-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

test.skipIf(!GPU_AVAILABLE)(
  "assetComponentOpener: a Session opens from a split-form Record (real GPU)",
  async () => {
    const open = await openerOf(parts("dit"));
    const component = open("dit");
    // The graph declaration is read from part 0 (the descriptor).
    assertEquals(component.graph.outputs, ["y"]);

    const gpu = await acquireGpu();
    try {
      // The weight block is only in the last part, so getting the known output is itself the
      // evidence that "the part sequence was read and reached the GPU". Opening two in a row is
      // check ② — a shape where the second one received an empty block sequence and opened
      // zero-filled also breaks the output here.
      for (let index = 0; index < 2; index += 1) {
        // MUST: do not call `open` again — that rebuilds the whole opener, so the target ("one
        // component's `createSession` re-reads the blocks every time") would never be exercised.
        const session = await component.createSession(gpu);
        assertEquals(await runLinearProbe(session), LINEAR_PROBE_Y, `Session number ${index + 1}`);
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "assetComponentOpener: a Session opens from a single-form key too (real GPU)",
  async () => {
    const open = await openerOf({ dit: single() });
    assertEquals(open("dit").graph.outputs, ["y"]);

    const gpu = await acquireGpu();
    try {
      const session = await open("dit").createSession(gpu);
      assertEquals(await runLinearProbe(session), LINEAR_PROBE_Y);
      await session.dispose();
    } finally {
      gpu.destroy();
    }
  },
);

test("assetComponentOpener: a gap in the indexes fails naming part", async () => {
  const error = await assertRejects(
    () => openerOf({ "dit[0]": part(0), "dit[2]": part(2) }),
    Error,
  );
  assertStringIncludes(
    error.message,
    "part indexes of component 'dit' are not contiguous from [0]",
  );
  // Lists the keys present (a form the reader can compare with what is there — the convention of
  // the existing asset diagnostics).
  assertStringIncludes(error.message, "dit[0] / dit[2]");
});

test("assetComponentOpener: mixing single-form and split-form keys fails", async () => {
  const error = await assertRejects(() => openerOf({ dit: single(), ...parts("dit") }), Error);
  assertStringIncludes(error.message, "both the single-form key and split-form keys");
  assertStringIncludes(error.message, "dit / dit[0]");
});

test("assetComponentOpener: mixing a single-form key with split-form keys other than [0] fails too (a missing [0] does not slip past the check)", async () => {
  // An implementation that picks up only the run contiguous from `[0]` would, when `[0]` is
  // missing, skip the mixing check and build "a single-form one" (silently ignoring `dit[1]` = a
  // Record built as a split distribution fails from a distant layer with "not enough parts").
  const error = await assertRejects(() => openerOf({ dit: single(), "dit[1]": part(1) }), Error);
  assertStringIncludes(error.message, "both the single-form key and split-form keys");
  assertStringIncludes(error.message, "dit / dit[1]");
});

test("assetComponentOpener: a sequence whose indexes do not start at [0] fails naming the start", async () => {
  const error = await assertRejects(
    () => openerOf({ "dit[1]": part(1), "dit[2]": part(2) }),
    Error,
  );
  assertStringIncludes(error.message, "part indexes of component 'dit' do not start at [0]");
  assertStringIncludes(error.message, "dit[1] / dit[2]");
});

test("assetComponentOpener: a sequence contiguous from [0] opens as the part sequence", async () => {
  const open = await openerOf(parts("dit"));
  assertEquals(open("dit").graph.outputs, ["y"]);
});

test("assetComponentOpener: a single-form key opens from the same opener too", async () => {
  const open = await openerOf({ dit: single() });
  assertEquals(open("dit").graph.outputs, ["y"]);
});

test("assetComponentOpener: when the key exists in neither form it fails with the spellings of both forms", async () => {
  const error = await assertRejects(() => openerOf(parts("dit"), ["dit", "vae"]), Error);
  assertStringIncludes(error.message, "component 'vae' has no container");
  assertStringIncludes(error.message, "split form: 'vae[0]' onwards in index order");
});

test("assetComponentOpener: a sequence missing a weight part fails naming the mismatch between the part count and the declaration", async () => {
  // A sequence with the last part (holding the weight block) dropped — it disagrees with the part
  // count the descriptor declares.
  const error = await assertRejects(
    () => openerOf({ "dit[0]": part(0), "dit[1]": part(1) }),
    ContainerFormatError,
  );
  assertStringIncludes(error.message, "part count is 2 but 3 are declared");
});
