import { assert, assertEquals, assertThrows } from "@std/assert";
import {
  HubError,
  ManifestFormatError,
  ManifestPathError,
  ManifestReferenceError,
  parseManifest,
} from "../mod.ts";

// 悪意 / 破損 manifest の受理集合は tests/fixtures/manifest-invalid.json が正本。ここは表を
// 全件回すだけで、TS 側に第 2 の定義を作らない（規模上限だけは手書きが非現実的なので組み立てる）。

type InvalidCase = {
  readonly name: string;
  readonly error: string;
  readonly manifest: unknown;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** 宣言されたエラー名が実際の型と一致するか（クラス同一性まで見る）。 */
const matchesDeclaredClass = (error: HubError, declared: string): boolean => {
  switch (declared) {
    case "ManifestFormatError":
      return error instanceof ManifestFormatError;
    case "ManifestReferenceError":
      return error instanceof ManifestReferenceError;
    case "ManifestPathError":
      return error instanceof ManifestPathError;
    default:
      return false;
  }
};

const loadInvalidCases = async (): Promise<readonly InvalidCase[]> => {
  const raw: unknown = JSON.parse(
    await Deno.readTextFile(new URL("./fixtures/manifest-invalid.json", import.meta.url)),
  );
  if (!isRecord(raw) || !Array.isArray(raw["cases"])) throw new Error("fixture: cases が無い");
  const cases = raw["cases"].map((entry: unknown): InvalidCase => {
    if (!isRecord(entry) || typeof entry["name"] !== "string") {
      throw new Error(`fixture: ケースの形が不正 ${JSON.stringify(entry)}`);
    }
    const error = entry["error"];
    if (typeof error !== "string") {
      throw new Error(`fixture: エラー名が文字列でない ${JSON.stringify(error)}`);
    }
    return { name: entry["name"], error, manifest: entry["manifest"] };
  });
  if (cases.length === 0) throw new Error("fixture: ケースが空");
  return cases;
};

const invalidCases = await loadInvalidCases();

const validManifestText = await Deno.readTextFile(
  new URL("./fixtures/manifest-fetch.json", import.meta.url),
);

const withEnvelope = (body: Record<string, unknown>): string =>
  JSON.stringify({
    format: "karume/1",
    generator: "karume/0.1.0",
    pipeline: "anima/1",
    components: {
      c: {
        file: {
          path: "c/model.safetensors",
          size: 4,
          sha256: "a1".repeat(32),
        },
      },
    },
    presets: { p: { weights: {}, session: {} } },
    defaultPreset: "p",
    pipelineConfig: {},
    ...body,
  });

Deno.test("parseManifest: fixture の全違反ケースが宣言どおりのエラー型で赤くなる", async (t) => {
  for (const testCase of invalidCases) {
    await t.step(`${testCase.name} → ${testCase.error}`, () => {
      const thrown = assertThrows(
        () => parseManifest(JSON.stringify(testCase.manifest)),
        HubError,
      );
      assertEquals(thrown.name, testCase.error);
      assert(
        matchesDeclaredClass(thrown, testCase.error),
        `${testCase.name}: ${thrown.name} は宣言された ${testCase.error} でない`,
      );
    });
  }
});

Deno.test("parseManifest: JSON として壊れていれば ManifestFormatError に包んで再送出する", () => {
  const error = assertThrows(
    () => parseManifest('{"format": "karume/1",}'),
    ManifestFormatError,
  );
  assert(error.cause instanceof SyntaxError, "元の SyntaxError を cause に残す");
});

Deno.test("parseManifest: 規模上限を数値で弾く", async (t) => {
  const fileRef = (name: string) => ({
    path: `${name}/model.safetensors`,
    size: 4,
    sha256: "a1".repeat(32),
  });

  await t.step("components 65 件", () => {
    let components: Record<string, unknown> = {};
    for (let index = 0; index < 65; index += 1) {
      components = { ...components, [`c${index}`]: { file: fileRef(`c${index}`) } };
    }
    assertThrows(() => parseManifest(withEnvelope({ components })), ManifestFormatError);
  });

  await t.step("presets 33 件", () => {
    let presets: Record<string, unknown> = {};
    for (let index = 0; index < 33; index += 1) {
      presets = { ...presets, [`p${index}`]: { weights: {}, session: {} } };
    }
    assertThrows(
      () => parseManifest(withEnvelope({ presets, defaultPreset: "p0" })),
      ManifestFormatError,
    );
  });

  await t.step("pipelineConfig 256KiB 超", () => {
    const pipelineConfig = { blob: "x".repeat(256 * 1024) };
    assertThrows(() => parseManifest(withEnvelope({ pipelineConfig })), ManifestFormatError);
  });

  await t.step("manifest 本体 1MiB 超", () => {
    // pipelineConfig 単体の上限より先に本体の上限へ当たる形（同じ 1 本の門を別経路から踏む）。
    const text = `${" ".repeat(1024 * 1024)}${withEnvelope({})}`;
    assertThrows(() => parseManifest(text), ManifestFormatError);
  });
});

Deno.test("parseManifest: 全エラーに利用可能な preset / variant ラベルが載る", () => {
  const error = assertThrows(
    () =>
      parseManifest(JSON.stringify({
        format: "karume/1",
        generator: "karume/0.1.0",
        pipeline: "anima/1",
        components: {
          net: {
            variants: {
              f16: { file: { path: "net/f16.safetensors", size: 8, sha256: "b2".repeat(32) } },
              i8: { file: { path: "net/i8.safetensors", size: 4, sha256: "c3".repeat(32) } },
            },
          },
        },
        presets: { fast: { weights: { net: "q4" }, session: {} } },
        defaultPreset: "fast",
        pipelineConfig: {},
      })),
    ManifestReferenceError,
  );
  assertEquals(error.available.presets, ["fast"]);
  assertEquals(error.available.variants, { net: ["f16", "i8"] });
  assert(error instanceof HubError, "全て HubError で一括して捌ける");
});

Deno.test("parseManifest: 正常な manifest を宣言どおりに読む", () => {
  const manifest = parseManifest(validManifestText);
  assertEquals(manifest.format, "karume/1");
  assertEquals(manifest.pipeline, { name: "anima", major: 1 });
  assertEquals(manifest.generator, "karume/0.1.0");
  assertEquals(manifest.defaultPreset, "w8a8-s16");
  assertEquals(manifest.available.presets, ["f16", "w8a8-s16", "f16-c16"]);
  assertEquals(manifest.available.variants, { transformer: ["f16", "i8"] });
  assertEquals(manifest.pipelineConfig, { defaults: { steps: 8, guidanceScale: 1 } });

  const transformer = manifest.components["transformer"];
  assert(transformer.kind === "variants");
  assertEquals(Object.keys(transformer.variants["i8"].extras), ["rope_base"]);
  assertEquals(
    transformer.variants["i8"].file.path,
    "transformer/model.i8.safetensors",
  );

  const tokenizer = manifest.components["tokenizer"];
  assert(tokenizer.kind === "single");
  assertEquals(tokenizer.files.extras, {});

  assertEquals(manifest.presets["w8a8-s16"].session, {
    linearCompute: "i8a8",
    attentionCompute: "i8a8",
    attentionScoreStorage: "f16",
  });
  assertEquals(manifest.presets["f16-c16"].gpuFeatures, { shaderF16: true });
  assertEquals(manifest.presets["f16"].gpuFeatures, undefined);
});

Deno.test("parseManifest: pipeline の major は形だけ検査し、裁定は models 側へ渡す", () => {
  // ADR 0038 §1: 未知 pipeline major を弾けるのは対応 major を宣言する models 実装だけ。
  // hub は綴りを検査して major を型で取り出すところまでを持つ。
  const manifest = parseManifest(withEnvelope({ pipeline: "sbv2/7" }));
  assertEquals(manifest.pipeline, { name: "sbv2", major: 7 });
});
