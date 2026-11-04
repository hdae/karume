// The two axes (model / quant) of `resolveSelection`, and `selectionRefs`, which derives the flat
// fetch list from them. The fetch layer is not involved; only building a selection from a manifest
// is looked at.
//
// Six points are pinned here:
//  ① When omitted, the selection falls back to `defaultModel` / `defaultQuant` and returns the
//     real names.
//  ② A selection is a **structured type** (component name → container / asset name → FileRef) and
//     has no spelling convention for fetch keys.
//  ③ The order is declaration order (the order of `weights`, part index order inside a container,
//     assets last).
//  ④ `selectionRefs` **drops zero-length parts** and dedupes the same file by `fileRefKey`.
//  ⑤ Passing a subset of `weights` puts only those roles into the selection (assets are
//     always all).
//  ⑥ An unknown model / quant / weights fails with **the list of available ones** (ADR 0041 §8).

import { test } from "vite-plus/test";
import { assertEquals, assertNotStrictEquals, assertThrows } from "@std/assert";
import {
  type FileRef,
  ManifestReferenceError,
  parseManifest,
  resolveSelection,
  selectionRefs,
} from "../mod.ts";
import {
  fetchManifest as manifest,
  STYLE_VECTORS,
  TEXT_CONDITIONER_PARTS,
  TEXT_ENCODER_PARTS,
  TOKENIZER,
  TRANSFORMER_F16_PARTS,
  TRANSFORMER_I8_EMPTY,
  TRANSFORMER_I8_FETCHED,
  TRANSFORMER_I8_PARTS,
  VAE_DECODER_PARTS,
} from "./helpers/fixture.ts";

const paths = (refs: readonly FileRef[]): string[] => refs.map((ref) => ref.path);

test("resolveSelection: when omitted, returns the defaultModel / defaultQuant pair by real name", () => {
  const selection = resolveSelection(manifest);
  assertEquals(selection.model, "anima-turbo");
  assertEquals(selection.quant, "w8a8-s16");
  // Components in the declaration order of weights (no spelling convention for fetch keys — the
  // entry is the component name itself).
  assertEquals(Object.keys(selection.containers), [
    "text_encoder",
    "text_conditioner",
    "transformer",
    "vae_decoder",
  ]);
  assertEquals(Object.keys(selection.assets), ["tokenizer", "style_vectors", "style_alias"]);
  // defaultQuant = w8a8-s16 (the transformer is the i8 container).
  assertEquals(paths(selection.containers["transformer"].parts), [...TRANSFORMER_I8_PARTS]);
});

test("resolveSelection: specifying a quant switches the dtype selection", () => {
  const selection = resolveSelection(manifest, { quant: "f16" });
  assertEquals(selection.quant, "f16");
  assertEquals(paths(selection.containers["transformer"].parts), [...TRANSFORMER_F16_PARTS]);
  // Only the container switches, and its descriptor comes along (the expected values of the 2
  // documents belong to the container). The expected values are spelled from the fixture's actual
  // values — looking them up again from the same manifest would move both sides together and stay
  // green even for an implementation that drops the descriptor (the i8 graph is 22, so a dtype
  // mix-up fails too).
  assertEquals(selection.containers["transformer"].descriptor.graph.length, 23);
  assertEquals(selection.containers["transformer"].descriptor.model.length, 5);
});

test("resolveSelection: assets do not change when the quant is switched", () => {
  const defaults = resolveSelection(manifest);
  const f16 = resolveSelection(manifest, { quant: "f16" });
  assertEquals(f16.assets, defaults.assets);
  // Same content, but **the table itself is rebuilt every time** — exposing the parsed manifest's
  // table would let an assignment to the returned assets pollute every later selection, the
  // inventory accounting and evict (the lifetime is handled the same way as containers).
  assertNotStrictEquals(defaults.assets, manifest.models["anima-turbo"].assets);
  assertNotStrictEquals(f16.assets, defaults.assets);
});

test("resolveSelection: specifying a model switches to that model's selection", () => {
  const selection = resolveSelection(manifest, { model: "anima-lite" });
  assertEquals(selection.model, "anima-lite");
  assertEquals(selection.quant, "w8");
  assertEquals(Object.keys(selection.containers), ["text_encoder", "transformer"]);
  assertEquals(Object.keys(selection.assets), ["tokenizer", "style_vectors"]);
  // A shared path is the same set of 3 across models (ADR 0041 §5, "shared by matching path").
  assertEquals(
    selection.containers["text_encoder"].parts[0],
    resolveSelection(manifest).containers["text_encoder"].parts[0],
  );
});

test("selectionRefs: lists every part of every container + assets in declaration order", () => {
  const refs = selectionRefs(resolveSelection(manifest));
  assertEquals(paths(refs), [
    ...TEXT_ENCODER_PARTS,
    ...TEXT_CONDITIONER_PARTS,
    ...TRANSFORMER_I8_FETCHED,
    ...VAE_DECODER_PARTS,
    TOKENIZER,
    STYLE_VECTORS,
  ]);
});

test("selectionRefs: a zero-length part is not listed (there is no content to fetch)", () => {
  const refs = selectionRefs(resolveSelection(manifest));
  assertEquals(
    refs.filter((ref) => ref.path === TRANSFORMER_I8_EMPTY),
    [],
    "a zero-length part remains in the fetch list",
  );
  // It stays in the declaration itself (the index is the part id, so dropping it would shift the
  // following indices).
  assertEquals(resolveSelection(manifest).containers["transformer"].parts.length, 3);
  assertEquals(
    refs.filter((ref) => ref.size === 0),
    [],
  );
});

test("selectionRefs: two asset names that point at the same file fold into one", () => {
  const selection = resolveSelection(manifest);
  // The table has 3 entries (tokenizer / style_vectors / style_alias) but only 2 files.
  assertEquals(Object.keys(selection.assets).length, 3);
  const refs = selectionRefs(selection);
  assertEquals(refs.filter((ref) => ref.path === STYLE_VECTORS).length, 1);
  assertEquals(new Set(paths(refs)).size, refs.length, "duplicates remain in the list");
});

test("selectionRefs: a part shared across containers also folds into one", () => {
  // The 2 containers of anima-lite point at the same files as anima-turbo, so the list is a subset
  // of the turbo default.
  const lite = new Set(paths(selectionRefs(resolveSelection(manifest, { model: "anima-lite" }))));
  const turbo = new Set(paths(selectionRefs(resolveSelection(manifest))));
  for (const path of lite) {
    assertEquals(turbo.has(path), true, `${path} is not in the subset of the default selection`);
  }
  assertEquals(lite.size, TEXT_ENCODER_PARTS.length + TRANSFORMER_I8_FETCHED.length + 2);
});

// ---- ⑤ A subset of weights (`ResolveOptions.weights`). The axis that handles a model that "runs
// with the main weights alone and can also add extra roles" (gemma4's model + drafter) without
// splitting the distribution form.

test("resolveSelection: narrowing weights puts only the named roles into the selection (assets are all)", () => {
  const selection = resolveSelection(manifest, { weights: ["transformer"] });
  assertEquals(Object.keys(selection.containers), ["transformer"]);
  assertEquals(Object.keys(selection.assets), ["tokenizer", "style_vectors", "style_alias"]);
  // Narrowing does not change how the dtype is chosen (i8 of the default quant).
  assertEquals(paths(selection.containers["transformer"].parts), [...TRANSFORMER_I8_PARTS]);
  assertEquals(paths(selectionRefs(selection)), [
    ...TRANSFORMER_I8_FETCHED,
    TOKENIZER,
    STYLE_VECTORS,
  ]);
});

test("resolveSelection: the order of weights is declaration order (not the order given)", () => {
  const selection = resolveSelection(manifest, { weights: ["vae_decoder", "text_encoder"] });
  // Even when the caller lists them in reverse, the selection keeps the manifest's declaration
  // order (the caller does not move the dispatch order of fetches).
  assertEquals(Object.keys(selection.containers), ["text_encoder", "vae_decoder"]);
});

test("resolveSelection: an empty weights array fetches nothing (assets only)", () => {
  const selection = resolveSelection(manifest, { weights: [] });
  assertEquals(Object.keys(selection.containers), []);
  assertEquals(paths(selectionRefs(selection)), [TOKENIZER, STYLE_VECTORS]);
});

test("resolveSelection: rejects a weights name that does not exist, with the list of available ones", () => {
  const error = assertThrows(
    () => resolveSelection(manifest, { weights: ["transformer", "drafter"] }),
    ManifestReferenceError,
    "weights 'drafter' is not in the manifest",
  );
  assertEquals(error.available.models, ["anima-turbo", "anima-lite"]);
});

test("resolveSelection: rejects duplicate weights", () => {
  assertThrows(
    () => resolveSelection(manifest, { weights: ["transformer", "transformer"] }),
    ManifestReferenceError,
    "weights 'transformer' was given twice",
  );
});

test("resolveSelection: whether a weights name exists is checked per model", () => {
  // anima-lite has no text_conditioner (anima-turbo does).
  assertThrows(
    () => resolveSelection(manifest, { model: "anima-lite", weights: ["text_conditioner"] }),
    ManifestReferenceError,
    "available: text_encoder / transformer",
  );
});

test("resolveSelection: rejects a model that does not exist, with the list of available ones", () => {
  const error = assertThrows(
    () => resolveSelection(manifest, { model: "anima-xl" }),
    ManifestReferenceError,
  );
  assertEquals(error.available.models, ["anima-turbo", "anima-lite"]);
});

test("resolveSelection: rejects a quant that does not exist, with the list of available ones", () => {
  const error = assertThrows(
    () => resolveSelection(manifest, { quant: "q4" }),
    ManifestReferenceError,
  );
  assertEquals(error.available.quants, ["f16", "w8a8-s16", "f16-c16"]);
  assertEquals(error.available.dtypes, {
    text_encoder: ["f16"],
    text_conditioner: ["f16"],
    transformer: ["f16", "i8"],
    vae_decoder: ["f16"],
  });
});

test("resolveSelection: the list of quants is the one of the specified model", () => {
  const error = assertThrows(
    () => resolveSelection(manifest, { model: "anima-lite", quant: "w8a8-s16" }),
    ManifestReferenceError,
  );
  // It does not suggest a quant that only another model (anima-turbo) has.
  assertEquals(error.available.quants, ["w8"]);
});

// ---- Component names and asset names go into **separate fields**, so the failure shape of
// `karume/4`, "fetch-key collision", no longer exists (folding the 3 namespaces into one table was
// the reason for that check).

test("resolveSelection: a component name and an asset name with the same name do not collide (separate fields)", () => {
  const collide = parseManifest(
    JSON.stringify({
      format: "karume/5",
      generator: "karume/0.1.0",
      defaultModel: "m",
      models: {
        m: {
          pipeline: "anima/1",
          weights: {
            tokenizer: {
              f16: {
                container: {
                  descriptor: {
                    graph: { length: 11, sha256: "f2".repeat(32) },
                    model: { length: 5, sha256: "a3".repeat(32) },
                  },
                  parts: [
                    {
                      path: "tokenizer/model-00001-of-00002.krm",
                      size: 40,
                      sha256: "b2".repeat(32),
                    },
                    {
                      path: "tokenizer/model-00002-of-00002.krm",
                      size: 64,
                      sha256: "c3".repeat(32),
                    },
                  ],
                },
              },
            },
          },
          assets: {
            tokenizer: { path: "tokenizer/tokenizer.json", size: 12, sha256: "a1".repeat(32) },
          },
          quants: { q: { weights: { tokenizer: "f16" }, session: {} } },
          defaultQuant: "q",
          pipelineConfig: {},
        },
      },
    }),
  );
  const selection = resolveSelection(collide);
  assertEquals(Object.keys(selection.containers), ["tokenizer"]);
  assertEquals(Object.keys(selection.assets), ["tokenizer"]);
  assertEquals(paths(selectionRefs(selection)), [
    "tokenizer/model-00001-of-00002.krm",
    "tokenizer/model-00002-of-00002.krm",
    "tokenizer/tokenizer.json",
  ]);
});
