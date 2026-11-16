import { type Manifest, parseManifest } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import { crossRepoHolders, type DistributionIndex } from "../browser.ts";

const hex = (character: string): string => character.repeat(64);
const REVISION = "4".repeat(40);

/** A `karume/5` manifest with one model per entry of `models`, each asset referencing its repo. */
const manifestOf = (models: Readonly<Record<string, readonly (string | undefined)[]>>): Manifest =>
  parseManifest(
    JSON.stringify({
      format: "karume/5",
      generator: "test",
      defaultModel: Object.keys(models)[0],
      models: Object.fromEntries(
        Object.entries(models).map(([model, repos]) => [
          model,
          {
            pipeline: "test/1",
            assets: Object.fromEntries(
              repos.map((repo, index) => [
                `asset${index}`,
                {
                  path: `${model}/asset${index}.bin`,
                  size: 4,
                  sha256: hex("c"),
                  ...(repo === undefined ? {} : { repo, revision: REVISION }),
                },
              ]),
            ),
            weights: {
              net: {
                f16: {
                  container: {
                    descriptor: {
                      graph: { length: 8, sha256: hex("d") },
                      model: { length: 8, sha256: hex("e") },
                    },
                    parts: [
                      { path: `${model}/net-00001-of-00002.krm`, size: 40, sha256: hex("a") },
                      { path: `${model}/net-00002-of-00002.krm`, size: 8, sha256: hex("b") },
                    ],
                  },
                },
              },
            },
            quants: { f16: { weights: { net: "f16" }, session: {} } },
            defaultQuant: "f16",
            pipelineConfig: {},
          },
        ]),
      ),
    }),
  );

const INDEX: DistributionIndex = {
  wan22: {
    directory: "models/karume-wan2.2",
    available: true,
    hf: { repo: "hdae/karume-wan2.2", revision: "1".repeat(40) },
  },
  umt5: { directory: "models/karume-umt5-xxl", available: false },
  anima: {
    directory: "models/anima-elsewhere",
    available: true,
    hf: { repo: "hdae/karume-anima", revision: "2".repeat(40) },
  },
  "anima-copy": { directory: "models/karume-anima", available: true },
};

describe("crossRepoHolders", () => {
  it("maps a repository to the mount served under its name, before a directory of that name", () => {
    expect(crossRepoHolders(manifestOf({ a: ["hdae/karume-anima"] }), INDEX)).toEqual({
      "hdae/karume-anima": "anima",
    });
  });

  it("falls back to the mount whose directory is named after the repository, present or not", () => {
    expect(crossRepoHolders(manifestOf({ a: ["hdae/karume-umt5-xxl"] }), INDEX)).toEqual({
      "hdae/karume-umt5-xxl": "umt5",
    });
  });

  it("covers every model of the manifest and leaves out a repository no mount holds", () => {
    const manifest = manifestOf({
      a: ["hdae/karume-umt5-xxl", undefined],
      b: ["hdae/karume-anima", "someone/elsewhere"],
    });
    expect(crossRepoHolders(manifest, INDEX)).toEqual({
      "hdae/karume-umt5-xxl": "umt5",
      "hdae/karume-anima": "anima",
    });
  });

  it("is empty for a distribution that references no other repository", () => {
    expect(crossRepoHolders(manifestOf({ a: [undefined] }), INDEX)).toEqual({});
  });
});
