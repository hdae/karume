/** A minimal `karume/6` manifest for the distribution openers' tests (it parses; its bytes are hashed). */
export const DISTRIBUTION_MANIFEST = JSON.stringify({
  format: "karume/6",
  generator: "test",
  defaultModel: "a",
  models: {
    a: {
      pipeline: "test/1",
      assets: {},
      weights: {
        net: {
          f16: {
            container: {
              descriptor: {
                graph: { length: 8, sha256: "d".repeat(64) },
                model: { length: 8, sha256: "e".repeat(64) },
              },
              parts: [
                { path: "a/model-00001-of-00002.krm", size: 40, sha256: "a".repeat(64) },
                { path: "a/model-00002-of-00002.krm", size: 8, sha256: "b".repeat(64) },
              ],
            },
          },
        },
      },
      quants: { f16: { weights: { net: "f16" }, session: {} } },
      defaultQuant: "f16",
      pipelineConfig: {},
    },
  },
});
