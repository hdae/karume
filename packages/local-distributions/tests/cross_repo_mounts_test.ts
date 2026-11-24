import { type Manifest, parseManifest } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import { crossRepoMounts, type RepoHolders } from "../browser.ts";
import { MOUNTS } from "../mounts.ts";

const hex = (character: string): string => character.repeat(64);
const V4 = "hdae/karume-irodori-v4-small";
const V41 = "hdae/karume-irodori-v4.1-small";
const REVISION = "4".repeat(40);
const HOLDERS: RepoHolders<"irodori-v4"> = { [V4]: "irodori-v4" };
const MF_HOLDERS: RepoHolders<"irodori-v4" | "irodori-v4.1"> = {
  [V4]: "irodori-v4",
  [V41]: "irodori-v4.1",
};

const container = (prefix: string, borrowed: boolean | string) => ({
  container: {
    descriptor: {
      graph: { length: 8, sha256: hex("d") },
      model: { length: 8, sha256: hex("e") },
    },
    parts: [1, 2].map((part) => ({
      path: `${prefix}-0000${part}-of-00002.krm`,
      size: part === 1 ? 40 : 8,
      sha256: hex(part === 1 ? "a" : "b"),
      ...(borrowed === false
        ? {}
        : { repo: borrowed === true ? V4 : borrowed, revision: REVISION }),
    })),
  },
});

/**
 * A `karume/6` manifest shaped like Irodori v4.1-small-mf (ADR 0150): only `dit` is its own; the
 * duration predictor points at v4.1-small's repository, the backbone and the tokenizer at v4-small's.
 */
const mfManifest = (): Manifest =>
  parseManifest(
    JSON.stringify({
      format: "karume/6",
      generator: "test",
      defaultModel: "v4.1-small-mf",
      models: {
        "v4.1-small-mf": {
          pipeline: "irodori/1",
          assets: {
            tokenizer: {
              path: "v4-small/tokenizer/tokenizer.json",
              size: 16,
              sha256: hex("c"),
              repo: V4,
              revision: REVISION,
            },
          },
          weights: {
            backbone: { i8: container("v4-small/backbone/model.i8", V4) },
            dit: { i8: container("v4.1-small-mf/dit/model.i8", false) },
            duration: { i8: container("v4.1-small/duration/model.i8", V41) },
          },
          quants: { i8: { weights: { backbone: "i8", dit: "i8", duration: "i8" }, session: {} } },
          defaultQuant: "i8",
          pipelineConfig: {},
        },
      },
    }),
  );

/**
 * A `karume/6` manifest shaped like Irodori v4-large: its own graphs and tokenizer asset, and the
 * codec pair pointing at v4-small's repository (the published containers).
 */
const largeManifest = (): Manifest =>
  parseManifest(
    JSON.stringify({
      format: "karume/6",
      generator: "test",
      defaultModel: "v4-large",
      models: {
        "v4-large": {
          pipeline: "irodori/1",
          assets: {
            tokenizer: { path: "v4-large/tokenizer/tokenizer.json", size: 16, sha256: hex("c") },
          },
          weights: {
            backbone: { i8: container("v4-large/backbone/model.i8", false) },
            dit: { i8: container("v4-large/dit/model.i8", false) },
            codec_encoder: { i8: container("v4-small/codec_encoder/model.i8", V4) },
            codec_decoder: { i8: container("v4-small/codec_decoder/model.i8", V4) },
          },
          quants: {
            i8: {
              weights: { backbone: "i8", dit: "i8", codec_encoder: "i8", codec_decoder: "i8" },
              session: {},
            },
          },
          defaultQuant: "i8",
          pipelineConfig: {},
        },
      },
    }),
  );

/**
 * A `karume/6` manifest shaped like Irodori v4.1-small: with `borrowing`, the components it shares
 * with v4-small (`backbone`, `dit` in i8, the tokenizer) point at v4-small's repository while
 * `duration` and the i4 `dit` stay its own — the per-container rule. Without it, the distribution
 * built before the references (everything its own).
 */
const v41Manifest = (borrowing: boolean): Manifest =>
  parseManifest(
    JSON.stringify({
      format: "karume/6",
      generator: "test",
      defaultModel: "v4.1-small",
      models: {
        "v4.1-small": {
          pipeline: "irodori/1",
          assets: {
            tokenizer: {
              path: `${borrowing ? "v4-small" : "v4.1-small"}/tokenizer/tokenizer.json`,
              size: 16,
              sha256: hex("c"),
              ...(borrowing ? { repo: V4, revision: REVISION } : {}),
            },
          },
          weights: {
            backbone: {
              i8: container(
                `${borrowing ? "v4-small" : "v4.1-small"}/backbone/model.i8`,
                borrowing,
              ),
            },
            dit: {
              i8: container(`${borrowing ? "v4-small" : "v4.1-small"}/dit/model.i8`, borrowing),
              i4: container("v4.1-small/dit/model.i4", false),
            },
            duration: { i8: container("v4.1-small/duration/model.i8", false) },
          },
          quants: {
            i8: { weights: { backbone: "i8", dit: "i8", duration: "i8" }, session: {} },
            "i8+dit4": { weights: { backbone: "i8", dit: "i4", duration: "i8" }, session: {} },
          },
          defaultQuant: "i8",
          pipelineConfig: {},
        },
      },
    }),
  );

describe("crossRepoMounts", () => {
  it("maps the repository a selection borrows from to the mount that holds it", () => {
    expect(crossRepoMounts(v41Manifest(true), "v4.1-small", "i8", HOLDERS)).toEqual({
      [V4]: "irodori-v4",
    });
  });

  it("still maps it when only part of the selection is borrowed", () => {
    // `i8+dit4` reads its own i4 DiT and duration, and the rest from v4-small.
    expect(crossRepoMounts(v41Manifest(true), "v4.1-small", "i8+dit4", HOLDERS)).toEqual({
      [V4]: "irodori-v4",
    });
  });

  it("maps each of two borrowed repositories to its own mount (the MeanFlow model)", () => {
    expect(crossRepoMounts(mfManifest(), "v4.1-small-mf", "i8", MF_HOLDERS)).toEqual({
      [V4]: "irodori-v4",
      [V41]: "irodori-v4.1",
    });
  });

  it("fails loudly when only one of the two borrowed repositories has a mount", () => {
    expect(() => crossRepoMounts(mfManifest(), "v4.1-small-mf", "i8", HOLDERS)).toThrow(
      /references hdae\/karume-irodori-v4\.1-small, which no mount/,
    );
  });

  it("maps v4-large's borrowed codec pair to the v4-small mount", () => {
    expect(crossRepoMounts(largeManifest(), "v4-large", "i8", HOLDERS)).toEqual({
      [V4]: "irodori-v4",
    });
  });

  it("is empty for a distribution that references no other repository", () => {
    expect(crossRepoMounts(v41Manifest(false), "v4.1-small", "i8", HOLDERS)).toEqual({});
  });

  it("fails loudly for a referenced repository no mount holds", () => {
    expect(() => crossRepoMounts(v41Manifest(true), "v4.1-small", "i8", {})).toThrow(
      /'v4\.1-small' \(i8\) references hdae\/karume-irodori-v4-small, which no mount/,
    );
  });
});

describe("the mounts other distributions reference", () => {
  // karume-anima-extra and karume-anima-2.9b borrow from karume-anima, the rebuilt
  // karume-irodori-v4.1-small from karume-irodori-v4-small, karume-irodori-v4.1-small-mf from both
  // small Irodori repositories, and karume-irodori-v4-large its codec pair from
  // karume-irodori-v4-small, each at the revision its manifest pins — not the revision these mounts are served at. Without the opt-in
  // the serving table throws, and /models/index.json builds it, so every page would fail
  // (plugin.ts).
  it.each(["anima", "irodori-v4", "irodori-v4.1"] as const)(
    "%s aliases the referenced revisions",
    (name) => {
      expect(MOUNTS[name].aliasReferencedRevisions).toBe(true);
    },
  );
});
