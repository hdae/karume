import { assert, assertEquals, assertThrows } from "@std/assert";
import {
  HubError,
  ManifestFormatError,
  ManifestPathError,
  ManifestReferenceError,
  parseManifest,
} from "../mod.ts";

// 悪意 / 破損 manifest の受理集合は tests/fixtures/manifest-invalid.json が正本。ここは表を
// 全件回すだけで、TS 側に第 2 の定義を作らない（境界値だけは手書きが非現実的なので組み立てる）。

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

/** 資産 1 本（コンテナの外側にある quant 非依存のファイル）。 */
const FILE = { path: "tokenizer/tokenizer.json", size: 12, sha256: "a1".repeat(32) };

/** コンテナのヘッダ長（container-v1 §1）。part 0 は「ヘッダ + 2 文書ちょうど」。 */
const HEADER_BYTES = 24;
const GRAPH_LENGTH = 11;
const MODEL_LENGTH = 5;
/** `24 + 11 + 5` — part 0 の size はこの値ちょうどでなければならない。 */
const PART0_SIZE = HEADER_BYTES + GRAPH_LENGTH + MODEL_LENGTH;

const DESCRIPTOR = {
  graph: { length: GRAPH_LENGTH, sha256: "f2".repeat(32) },
  model: { length: MODEL_LENGTH, sha256: "a3".repeat(32) },
};

/**
 * part 1 本の宣言。`index` は**添字**（0 始まり）で、ファイル名の連番は書き手の綴り
 * （1 始まり — `tools/exporter/src/karume/container.py` の `container_paths`）に合わせる。
 * hub はファイル名を検査しないが、実ミラーと突き合わせる人の材料になる。
 */
const part = (index: number, total: number, size: number, mark: string) => ({
  path: `net/model-${String(index + 1).padStart(5, "0")}-of-${String(total).padStart(5, "0")}.krm`,
  size,
  sha256: mark.repeat(32),
});

const PART0 = part(0, 2, PART0_SIZE, "b2");
const PART1 = part(1, 2, 64, "c3");

/** 最小の合法なコンテナ（part 0 + part 1）。 */
const CONTAINER = { descriptor: DESCRIPTOR, parts: [PART0, PART1] };

/** `container` を部分的に差し替えた weights エントリ。 */
const withContainer = (patch: Record<string, unknown>): Record<string, unknown> => ({
  net: { f16: { container: { ...CONTAINER, ...patch } } },
});

/**
 * 検査に要る欄だけを持つ最小の v5 manifest。`patch` は `models.m` の中身を、`envelope` は
 * トップレベルを上書きする。
 */
const withModel = (
  patch: Record<string, unknown> = {},
  envelope: Record<string, unknown> = {},
): string =>
  JSON.stringify({
    format: "karume/5",
    generator: "karume/0.1.0",
    defaultModel: "m",
    models: {
      m: {
        pipeline: "anima/1",
        weights: { net: { f16: { container: CONTAINER } } },
        assets: {},
        quants: { q: { weights: { net: "f16" }, session: {} } },
        defaultQuant: "q",
        pipelineConfig: {},
        ...patch,
      },
    },
    ...envelope,
  });

/** `models.m.weights.net.f16.container` を読む近道。 */
const containerOf = (text: string) =>
  parseManifest(text).models["m"].weights["net"]["f16"].container;

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
    () => parseManifest('{"format": "karume/5",}'),
    ManifestFormatError,
  );
  assert(error.cause instanceof SyntaxError, "元の SyntaxError を cause に残す");
});

Deno.test("parseManifest: v1（karume/1）は読まずに未対応 major として落とす", () => {
  // ADR 0109 決定 1: hub は現行版だけを読む（2 形パースを持たない）。旧クライアントの裏返しで、
  // 新クライアントが旧 manifest を**旧解釈で黙って実行する**経路も作らない。
  const error = assertThrows(
    () =>
      parseManifest(JSON.stringify({
        format: "karume/1",
        generator: "karume/0.1.0",
        pipeline: "anima/1",
        components: { net: { file: FILE } },
        presets: { p: { weights: {}, session: {} } },
        defaultPreset: "p",
        pipelineConfig: {},
      })),
    ManifestFormatError,
  );
  assert(
    error.message.includes("karume/5"),
    `${error.message} が「読めるのは karume/5」を名指ししていない`,
  );
});

Deno.test("parseManifest: 直前版（karume/4）も読まず、現行が karume/5 であることを名指しする", () => {
  // v4 と v5 の差は dtype エントリの中身（`shards` + `extras` → `container`）だけで、他の席は
  // そのまま。構造から入ると「未知キー 'shards'」という枝葉の診断になり、本当の理由
  // （この版は karume/5 のみ読む）が隠れる。断絶は format 文字列だけが宣言する（ADR 0109 決定 1）。
  const error = assertThrows(
    () =>
      parseManifest(JSON.stringify({
        format: "karume/4",
        generator: "karume/0.1.0",
        defaultModel: "m",
        models: {
          m: {
            pipeline: "anima/1",
            weights: {
              net: { f16: { shards: [{ ...PART1, path: "net/model.f16.safetensors" }] } },
            },
            assets: {},
            quants: { q: { weights: { net: "f16" }, session: {} } },
            defaultQuant: "q",
            pipelineConfig: {},
          },
        },
      })),
    ManifestFormatError,
  );
  assert(error.message.includes("karume/4"), `${error.message} が拒否した版を名指ししていない`);
  assert(
    error.message.includes("karume/5"),
    `${error.message} が「読めるのは karume/5」を名指ししていない`,
  );
  assert(
    error.message.includes("旧版"),
    `${error.message} が「旧版のパーサを持たない」を伝えていない`,
  );
  assert(!error.message.includes("'shards'"), `${error.message} が枝葉の未知キーを主因にしている`);
});

Deno.test("parseManifest: karume/2 の綴り（dtype エントリが {file}）も版で落とす", () => {
  // format を未知キー検査より先に見ていることの観測点（上の v4 と同じ理由を旧い綴りで踏む）。
  const error = assertThrows(
    () =>
      parseManifest(JSON.stringify({
        format: "karume/2",
        generator: "karume/0.1.0",
        defaultModel: "m",
        models: {
          m: {
            pipeline: "anima/1",
            weights: { net: { f16: { file: FILE } } },
            assets: {},
            quants: { q: { weights: { net: "f16" }, session: {} } },
            defaultQuant: "q",
            pipelineConfig: {},
          },
        },
      })),
    ManifestFormatError,
  );
  assert(error.message.includes("karume/2"), `${error.message} が拒否した版を名指ししていない`);
  assert(!error.message.includes("'file'"), `${error.message} が枝葉の未知キーを主因にしている`);
});

Deno.test("parseManifest: 規模上限を数値で弾く", async (t) => {
  const modelEntry = (quantName: string) => ({
    pipeline: "anima/1",
    weights: { net: { f16: { container: CONTAINER } } },
    assets: {},
    quants: { [quantName]: { weights: { net: "f16" }, session: {} } },
    defaultQuant: quantName,
    pipelineConfig: {},
  });

  await t.step("models 33 件", () => {
    let models: Record<string, unknown> = {};
    for (let index = 0; index < 33; index += 1) {
      models = { ...models, [`m${index}`]: modelEntry("q") };
    }
    const text = JSON.stringify({
      format: "karume/5",
      generator: "karume/0.1.0",
      defaultModel: "m0",
      models,
    });
    assertThrows(() => parseManifest(text), ManifestFormatError);
  });

  await t.step("weights 33 件", () => {
    let weights: Record<string, unknown> = {};
    let mapping: Record<string, string> = {};
    for (let index = 0; index < 33; index += 1) {
      weights = { ...weights, [`w${index}`]: { f16: { container: CONTAINER } } };
      mapping = { ...mapping, [`w${index}`]: "f16" };
    }
    assertThrows(
      () =>
        parseManifest(withModel({
          weights,
          quants: { q: { weights: mapping, session: {} } },
        })),
      ManifestFormatError,
    );
  });

  await t.step("parts 1025 件（1024 件は通る）", () => {
    const parts = (count: number): unknown[] => [
      part(0, count, PART0_SIZE, "b2"),
      ...Array.from({ length: count - 1 }, (_unused, index) => part(index + 1, count, 8, "c3")),
    ];
    // 上限ちょうどが通ることまで見る（片側だけだと「常に落ちる」実装でも緑になる）。
    const accepted = parseManifest(withModel({ weights: withContainer({ parts: parts(1024) }) }));
    assertEquals(accepted.models["m"].weights["net"]["f16"].container.parts.length, 1024);
    assertThrows(
      () => parseManifest(withModel({ weights: withContainer({ parts: parts(1025) }) })),
      ManifestFormatError,
    );
  });

  await t.step("assets 33 件", () => {
    let assets: Record<string, unknown> = {};
    for (let index = 0; index < 33; index += 1) {
      assets = { ...assets, [`a${index}`]: { ...FILE, path: `assets/a${index}.json` } };
    }
    assertThrows(() => parseManifest(withModel({ assets })), ManifestFormatError);
  });

  await t.step("quants 33 件", () => {
    let quants: Record<string, unknown> = {};
    for (let index = 0; index < 33; index += 1) {
      quants = { ...quants, [`q${index}`]: { weights: { net: "f16" }, session: {} } };
    }
    assertThrows(
      () => parseManifest(withModel({ quants, defaultQuant: "q0" })),
      ManifestFormatError,
    );
  });

  await t.step("pipelineConfig 256KiB 超（モデルあたり）", () => {
    const pipelineConfig = { blob: "x".repeat(256 * 1024) };
    assertThrows(() => parseManifest(withModel({ pipelineConfig })), ManifestFormatError);
  });

  await t.step("manifest 本体 1MiB 超", () => {
    // pipelineConfig 単体の上限より先に本体の上限へ当たる形（同じ 1 本の門を別経路から踏む）。
    const text = `${" ".repeat(1024 * 1024)}${withModel()}`;
    assertThrows(() => parseManifest(text), ManifestFormatError);
  });
});

Deno.test("parseManifest: 1MiB 未満でも深すぎる入れ子は型付きエラーで落とす", async (t) => {
  // 深さは**バイト数と独立**に伸びる（開き括弧の連続だけで数千段が数 KB）。深さ検査が無いと
  // 全域走査の再帰がスタックを食い潰し、`ManifestFormatError` ではなく素の `RangeError` が
  // 抜けて `instanceof HubError` の分岐から漏れる。ここはその型を観測値として固定する。
  const nested = (depth: number): string => `${"[".repeat(depth)}${"]".repeat(depth)}`;
  /** JSON.stringify の再帰を経由せずに深い配列を埋め込む（組み立て側でスタックを使わない）。 */
  const withDeepConfig = (depth: number): string =>
    withModel({ pipelineConfig: { deep: "@" } }).replace('"@"', nested(depth));

  await t.step("素の再帰なら RangeError になる深さでも ManifestFormatError になる", () => {
    const text = withDeepConfig(3000);
    assert(new TextEncoder().encode(text).length < 1024 * 1024, "1MiB 未満で組めていない");
    const error = assertThrows(() => parseManifest(text), ManifestFormatError);
    assert(
      error.message.includes("入れ子"),
      `${error.message} が深さ超過を名指ししていない`,
    );
  });

  await t.step("実用の入れ子（pipelineConfig 数段）は通る", () => {
    // 上限を実用要求より下に置いてしまう退行の検出器（上限値そのものの下限を縛る）。
    const manifest = parseManifest(
      withModel({ pipelineConfig: { a: { b: { c: [[[1, 2], [3]]] } } } }),
    );
    assertEquals(manifest.models["m"].defaultQuant, "q");
  });
});

Deno.test("parseManifest: エラーに利用可能な model / quant / dtype ラベルが載る", () => {
  const container = (mark: string) => ({
    descriptor: DESCRIPTOR,
    parts: [
      { ...PART0, path: `net/${mark}-00001-of-00002.krm` },
      { ...PART1, path: `net/${mark}-00002-of-00002.krm`, sha256: mark.repeat(32) },
    ],
  });
  const error = assertThrows(
    () =>
      parseManifest(JSON.stringify({
        format: "karume/5",
        generator: "karume/0.1.0",
        defaultModel: "fast",
        models: {
          fast: {
            pipeline: "anima/1",
            weights: {
              net: { f16: { container: container("b2") }, i8: { container: container("c3") } },
            },
            assets: {},
            quants: { w8: { weights: { net: "q4" }, session: {} } },
            defaultQuant: "w8",
            pipelineConfig: {},
          },
          slim: {
            pipeline: "anima/1",
            weights: { net: { i8: { container: container("d4") } } },
            assets: {},
            quants: { w8: { weights: { net: "i8" }, session: {} } },
            defaultQuant: "w8",
            pipelineConfig: {},
          },
        },
      })),
    ManifestReferenceError,
  );
  assertEquals(error.available.models, ["fast", "slim"]);
  // 文脈のモデル（fast）のものだけが載る — 別モデルの quant 名を勧めるのは誤誘導になる。
  assertEquals(error.available.quants, ["w8"]);
  assertEquals(error.available.dtypes, { net: ["f16", "i8"] });
  assert(error instanceof HubError, "全て HubError で一括して捌ける");
});

Deno.test("parseManifest: トップレベルの違反にはモデル一覧だけが載る（quant 文脈が無い）", () => {
  const error = assertThrows(
    () => parseManifest(withModel({}, { defaultModel: "absent" })),
    ManifestReferenceError,
  );
  assertEquals(error.available.models, ["m"]);
  assertEquals(error.available.quants, []);
  assertEquals(error.available.dtypes, {});
});

Deno.test("parseManifest: 正常な manifest を宣言どおりに読む", () => {
  const manifest = parseManifest(validManifestText);
  assertEquals(manifest.format, "karume/5");
  assertEquals(manifest.generator, "karume/0.1.0");
  assertEquals(manifest.defaultModel, "anima-turbo");
  assertEquals(Object.keys(manifest.models), ["anima-turbo", "anima-lite"]);
  assertEquals(manifest.available.models, ["anima-turbo", "anima-lite"]);

  const turbo = manifest.models["anima-turbo"];
  assertEquals(turbo.pipeline, { name: "anima", major: 1 });
  assertEquals(turbo.defaultQuant, "w8a8-s16");
  assertEquals(turbo.available.quants, ["f16", "w8a8-s16", "f16-c16"]);
  assertEquals(turbo.pipelineConfig, { defaults: { steps: 8, guidanceScale: 1 } });

  // weights は dtype キー必須の統一形で、中身はコンテナ 1 本（ADR 0109 決定 3）。
  assertEquals(Object.keys(turbo.weights["transformer"]), ["f16", "i8"]);
  assertEquals(Object.keys(turbo.weights["vae_decoder"]), ["f16"]);
  const i8 = turbo.weights["transformer"]["i8"].container;
  assertEquals(i8.parts.length, 3, "part 0 + part 1 + part 2 の 3 本");
  assertEquals(i8.parts[0].size, 24 + i8.descriptor.graph.length + i8.descriptor.model.length);
  // 長さ 0 の part（const が空のコンテナ）は宣言に残り、sha256 は空列の値を名乗る。
  assertEquals(i8.parts[1].size, 0);
  assertEquals(
    i8.parts[1].sha256,
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  );

  // assets は quant 選択に依存しない無条件ファイル（dtype の階層を持たない）。
  assertEquals(Object.keys(turbo.assets), ["tokenizer", "style_vectors", "style_alias"]);
  assertEquals(turbo.assets["tokenizer"].path, "tokenizer/qwen2-tokenizer.json");
  // 同じ実体を 2 つの名前が指す形（表の席は落とさない — 一意化は選択の列を作るときだけ）。
  assert(turbo.assets["style_alias"] === turbo.assets["style_vectors"]);

  assertEquals(turbo.quants["w8a8-s16"].session, {
    linearCompute: "a8",
    attentionCompute: "a8",
    attentionScoreStorage: "f16",
  });
  assertEquals(turbo.quants["f16-c16"].gpuFeatures, { shaderF16: true });
  assertEquals(turbo.quants["f16"].gpuFeatures, undefined);

  // 2 個目のモデルは自分の quant / dtype 面だけを持つ。
  const lite = manifest.models["anima-lite"];
  assertEquals(lite.defaultQuant, "w8");
  assertEquals(lite.available.quants, ["w8"]);
  assertEquals(lite.available.dtypes, { text_encoder: ["f16"], transformer: ["i8"] });
});

Deno.test("parseManifest: モデル間で同一 path を指す共有は成立する（ADR 0041 §5）", () => {
  const manifest = parseManifest(validManifestText);
  const shared = manifest.models["anima-turbo"].weights["text_encoder"]["f16"].container.parts[0];
  const same = manifest.models["anima-lite"].weights["text_encoder"]["f16"].container.parts[0];
  assertEquals(same.path, shared.path);
  assertEquals(same.sha256, shared.sha256);
  // 表は 1 本なので、同じ path は同じ FileRef インスタンスに畳まれる（取得も 1 回になる）。
  assert(same === shared, "同一 path の参照が畳まれていない");
});

Deno.test("parseManifest: weights の container 欄（ADR 0109 決定 3）", async (t) => {
  await t.step("descriptor は 2 文書それぞれの長さと sha256 を持つ", () => {
    const container = containerOf(withModel());
    assertEquals(container.descriptor.graph, DESCRIPTOR.graph);
    assertEquals(container.descriptor.model, DESCRIPTOR.model);
  });

  await t.step("parts は宣言順のまま保たれる（添字が part の id）", () => {
    const parts = [PART0, part(1, 3, 8, "c3"), part(2, 3, 16, "d4")];
    const container = containerOf(withModel({ weights: withContainer({ parts }) }));
    assertEquals(
      container.parts.map((entry) => entry.path),
      parts.map((entry) => entry.path),
    );
  });

  await t.step("2 件ちょうどは通り、1 件で落ちる（part 0 + part 1 が最小）", () => {
    assertEquals(containerOf(withModel()).parts.length, 2);
    assertThrows(
      () => parseManifest(withModel({ weights: withContainer({ parts: [PART0] }) })),
      ManifestFormatError,
      "2 件以上",
    );
  });

  await t.step("parts の中でも同一 path の 3 点セット不一致は拒否する", () => {
    const error = assertThrows(
      () =>
        parseManifest(withModel({
          weights: withContainer({ parts: [PART0, PART1, { ...PART1, sha256: "d4".repeat(32) }] }),
        })),
      ManifestReferenceError,
    );
    assert(
      error.message.includes(PART1.path),
      `${error.message} が食い違った path を名指ししていない`,
    );
  });

  await t.step("退役した shards / extras のキーは未知キーとして落ちる", () => {
    for (const retired of ["shards", "extras"]) {
      const error = assertThrows(
        () =>
          parseManifest(withModel({
            weights: { net: { f16: { container: CONTAINER, [retired]: {} } } },
          })),
        ManifestReferenceError,
        undefined,
        `dtype エントリの '${retired}' が通ってしまった`,
      );
      assert(
        error.message.includes(retired) && error.message.includes("container"),
        `${error.message} が未知キーと許可キーを出していない`,
      );
    }
  });

  await t.step("container / descriptor / 文書の未知キーも落ちる", () => {
    const patches: Record<string, unknown>[] = [
      { weights: { net: { f16: { container: { ...CONTAINER, graph: FILE } } } } },
      {
        weights: withContainer({
          descriptor: { ...DESCRIPTOR, codecs: [] },
        }),
      },
      {
        weights: withContainer({
          descriptor: { ...DESCRIPTOR, graph: { ...DESCRIPTOR.graph, offset: 0 } },
        }),
      },
    ];
    for (const patch of patches) {
      assertThrows(
        () => parseManifest(withModel(patch)),
        ManifestReferenceError,
        undefined,
        `${JSON.stringify(patch)} が通ってしまった`,
      );
    }
  });

  await t.step("descriptor の model は省略できない（krm は必ずモデル記述を持つ）", () => {
    assertThrows(
      () =>
        parseManifest(withModel({
          weights: withContainer({ descriptor: { graph: DESCRIPTOR.graph } }),
        })),
      ManifestFormatError,
      "descriptor.model: 無い",
    );
  });
});

Deno.test("parseManifest: part 0 の size は「ヘッダ + 2 文書」ちょうど（container-v1 §8）", async (t) => {
  // 宣言どうしの整合なので、1 バイトも取らずに見られる。ここが緩いと「取得してから長さ違いに
  // 気づく」形に戻る。
  await t.step("ちょうどは通る", () => {
    assertEquals(containerOf(withModel()).parts[0].size, PART0_SIZE);
  });

  await t.step("1 バイトでもずれれば両側とも落ちる", () => {
    for (const delta of [-1, 1]) {
      const error = assertThrows(
        () =>
          parseManifest(withModel({
            weights: withContainer({ parts: [{ ...PART0, size: PART0_SIZE + delta }, PART1] }),
          })),
        ManifestFormatError,
        undefined,
        `part 0 の size ${PART0_SIZE + delta} が通ってしまった`,
      );
      assert(
        error.message.includes(String(PART0_SIZE)),
        `${error.message} が期待した part 0 の長さを出していない`,
      );
    }
  });

  await t.step("descriptor の長さを動かせば part 0 の要求もその分動く", () => {
    const graph = { ...DESCRIPTOR.graph, length: GRAPH_LENGTH + 100 };
    const container = containerOf(withModel({
      weights: withContainer({
        descriptor: { ...DESCRIPTOR, graph },
        parts: [{ ...PART0, size: PART0_SIZE + 100 }, PART1],
      }),
    }));
    assertEquals(container.parts[0].size, PART0_SIZE + 100);
  });
});

Deno.test("parseManifest: descriptor の 1 文書は 32MiB まで（container-v1 §10）", async (t) => {
  // 綴りは Python 正本 `tools/exporter/src/karume/container.py` の `MAX_DESCRIPTOR_BYTES` と同値。
  const LIMIT = 32 * 2 ** 20;
  const sized = (length: number) => ({
    descriptor: { ...DESCRIPTOR, graph: { ...DESCRIPTOR.graph, length } },
    parts: [{ ...PART0, size: HEADER_BYTES + length + MODEL_LENGTH }, PART1],
  });

  await t.step("ちょうど 32MiB は通る（書き手と同じ閉区間）", () => {
    const container = containerOf(withModel({ weights: withContainer(sized(LIMIT)) }));
    assertEquals(container.descriptor.graph.length, LIMIT);
  });

  await t.step("1 バイト超は落ちる", () => {
    const error = assertThrows(
      () => parseManifest(withModel({ weights: withContainer(sized(LIMIT + 1)) })),
      ManifestFormatError,
    );
    assert(error.message.includes(String(LIMIT)), `${error.message} が上限を名乗っていない`);
  });

  await t.step("長さ 0 の文書は拒否する（krm は必ず 2 文書を持つ）", () => {
    assertThrows(
      () => parseManifest(withModel({ weights: withContainer(sized(0)) })),
      ManifestFormatError,
    );
  });

  await t.step("sha256 は小文字 hex 64 桁", () => {
    for (const sha256 of ["F2".repeat(32), "f2".repeat(31), 12]) {
      assertThrows(
        () =>
          parseManifest(withModel({
            weights: withContainer({
              descriptor: { ...DESCRIPTOR, model: { length: MODEL_LENGTH, sha256 } },
            }),
          })),
        ManifestFormatError,
        undefined,
        `descriptor の sha256 '${sha256}' が通ってしまった`,
      );
    }
  });
});

Deno.test("parseManifest: part のバイト上限 1024MiB（container-v1 §10 — ファイル長で測る）", async (t) => {
  // 読み手は「器の寸法を宣言から見積る」（ADR 0089）前提で組まれているので、上限違反の part を
  // parse が黙って通すとブラウザで初めて破綻する。上限の綴りは Python 正本
  // `tools/exporter/src/karume/container.py` の `PART_MAX_BYTES` と同値。
  const LIMIT = 1024 * 2 ** 20;
  const big = (size: number, mark: string) => ({
    path: "net/model-00002-of-00002.krm",
    size,
    sha256: mark.repeat(32),
  });

  await t.step("上限超過は落ち、エラーが part の path・寸法・上限を名指しする", () => {
    const error = assertThrows(
      () =>
        parseManifest(withModel({
          weights: withContainer({ parts: [PART0, big(LIMIT + 1, "c3")] }),
        })),
      ManifestFormatError,
    );
    for (const expected of ["net/model-00002-of-00002.krm", String(LIMIT + 1), String(LIMIT)]) {
      assert(error.message.includes(expected), `${error.message} に '${expected}' が無い`);
    }
  });

  await t.step("ちょうど 1024MiB は通る（書き手と同じ閉区間）", () => {
    // 片側だけだと「常に落ちる」実装でも緑になる。
    const container = containerOf(
      withModel({ weights: withContainer({ parts: [PART0, big(LIMIT, "c3")] }) }),
    );
    assertEquals(container.parts[1].size, LIMIT);
  });

  await t.step("非 part の FileRef は対象外 — assets は上限超でも通る", () => {
    // MUST: 上限は**part 分割の契約**であって全 FileRef の天井ではない（それは 16GiB の
    // `MAX_FILE_BYTES`）。ここを取り違えると上限超の実在資産（例: PLE sidecar）が読めなくなる。
    const manifest = parseManifest(withModel({
      assets: { style_vectors: { ...FILE, path: "style/vectors.safetensors", size: LIMIT + 1 } },
    }));
    assertEquals(manifest.models["m"].assets["style_vectors"].size, LIMIT + 1);
  });
});

Deno.test("parseManifest: 長さ 0 の part は添字 1 以上だけ（ADR 0109 決定 3）", async (t) => {
  const EMPTY_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
  const emptyPart = { path: "net/model-00002-of-00003.krm", size: 0, sha256: EMPTY_SHA };

  await t.step("添字 1 の長さ 0 は通り、宣言にそのまま残る", () => {
    const container = containerOf(withModel({
      weights: withContainer({ parts: [PART0, emptyPart, part(2, 3, 16, "d4")] }),
    }));
    assertEquals(container.parts[1].size, 0);
    assertEquals(container.parts[1].sha256, EMPTY_SHA);
  });

  await t.step("part 0 の長さ 0 は落ちる（ヘッダ + 2 文書を必ず持つ）", () => {
    assertThrows(
      () =>
        parseManifest(withModel({
          weights: withContainer({ parts: [{ ...PART0, size: 0, sha256: EMPTY_SHA }, PART1] }),
        })),
      ManifestFormatError,
    );
  });

  await t.step("長さ 0 なのに sha256 が空列の値でなければ落ちる", () => {
    const error = assertThrows(
      () =>
        parseManifest(withModel({
          weights: withContainer({
            parts: [PART0, { ...emptyPart, sha256: "c3".repeat(32) }, part(2, 3, 16, "d4")],
          }),
        })),
      ManifestFormatError,
    );
    assert(error.message.includes(EMPTY_SHA), `${error.message} が空列の sha256 を出していない`);
  });

  await t.step("assets の長さ 0 は従来どおり拒否する（空を許すのは parts だけ）", () => {
    assertThrows(
      () =>
        parseManifest(withModel({
          assets: { tokenizer: { ...FILE, size: 0, sha256: EMPTY_SHA } },
        })),
      ManifestFormatError,
    );
  });
});

Deno.test("parseManifest: pipeline の major は形だけ検査し、裁定は models 側へ渡す", () => {
  // ADR 0038 §1: 未知 pipeline major を弾けるのは対応 major を宣言する models 実装だけ。
  // hub は綴りを検査して major を型で取り出すところまでを持つ。
  const manifest = parseManifest(withModel({ pipeline: "sbv2/7" }));
  assertEquals(manifest.models["m"].pipeline, { name: "sbv2", major: 7 });
});

/** quant 席を 1 つだけ差し替えた manifest（新席の観測用）。 */
const withQuant = (patch: Record<string, unknown>): string =>
  withModel({ quants: { q: { weights: { net: "f16" }, session: {}, ...patch } } });

Deno.test("parseManifest: quant の表示欄 label / description（ADR 0075）", async (t) => {
  await t.step("設定した文字列がそのまま型面へ出る", () => {
    const manifest = parseManifest(withQuant({
      label: "Balanced (int8)",
      description: "Half the download of f16 with no visible difference.",
    }));
    const quant = manifest.models["m"].quants["q"];
    assertEquals(quant.label, "Balanced (int8)");
    assertEquals(quant.description, "Half the download of f16 with no visible difference.");
  });

  await t.step("未設定の席は欄を持たない（呼び手が id をそのまま出す）", () => {
    const quant = parseManifest(withModel()).models["m"].quants["q"];
    assertEquals(quant.label, undefined);
    assertEquals(quant.description, undefined);
  });

  await t.step("上限ちょうどは通り、1 文字超で落ちる", () => {
    // 片側だけだと「常に落ちる」実装でも緑になるので、境界の両側を観測する。
    assertEquals(
      parseManifest(withQuant({ label: "x".repeat(64) })).models["m"].quants["q"].label
        ?.length,
      64,
    );
    assertThrows(() => parseManifest(withQuant({ label: "x".repeat(65) })), ManifestFormatError);
    assertEquals(
      parseManifest(withQuant({ description: "x".repeat(200) })).models["m"].quants["q"]
        .description?.length,
      200,
    );
    assertThrows(
      () => parseManifest(withQuant({ description: "x".repeat(201) })),
      ManifestFormatError,
    );
  });

  await t.step("上限超過は期待と実際を添えて落ちる", () => {
    const error = assertThrows(
      () => parseManifest(withQuant({ label: "x".repeat(70) })),
      ManifestFormatError,
    );
    assert(error.message.includes("期待 64 文字以内"), `${error.message} が期待値を出していない`);
    assert(error.message.includes("実際 70 文字"), `${error.message} が実際の長さを出していない`);
  });

  await t.step("非文字列は期待と実際を添えて落ちる", () => {
    const error = assertThrows(() => parseManifest(withQuant({ label: 12 })), ManifestFormatError);
    assert(error.message.includes("期待 文字列"), `${error.message} が期待の型を出していない`);
    assert(error.message.includes("実際 number"), `${error.message} が実際の型を出していない`);
  });

  await t.step("内容の意味は解釈しない（上限内なら実態と食い違う説明も通る）", () => {
    // ADR 0075 決定 2: 長さは境界検査、内容の妥当性は hub には判定できないし、しない。
    const quant =
      parseManifest(withQuant({ label: "f32 (lossless)", description: "🌀".repeat(64) }))
        .models["m"].quants["q"];
    assertEquals(quant.label, "f32 (lossless)");
    // サロゲートペアはコードポイント 1 つとして数える（同じ見た目が綴りで通ったり落ちたりしない）。
    assertEquals(quant.description, "🌀".repeat(64));
  });
});

Deno.test("parseManifest: quant の requiredLimits（ADR 0038 §7 の据え置き席）", async (t) => {
  await t.step("limit 名 → 最小値の部分写像として型面へ出る", () => {
    const manifest = parseManifest(withQuant({
      requiredLimits: { maxBufferSize: 2147483648, maxStorageBufferBindingSize: 1073741824 },
    }));
    assertEquals(manifest.models["m"].quants["q"].requiredLimits, {
      maxBufferSize: 2147483648,
      maxStorageBufferBindingSize: 1073741824,
    });
  });

  await t.step("未設定の席は欄を持たず、空の宣言は空のまま通る", () => {
    assertEquals(parseManifest(withModel()).models["m"].quants["q"].requiredLimits, undefined);
    assertEquals(
      parseManifest(withQuant({ requiredLimits: {} })).models["m"].quants["q"]
        .requiredLimits,
      {},
    );
  });

  await t.step("runtime の requiredLimits 語彙の名前を受ける", () => {
    // 綴りが runtime（`REQUIRED_LIMIT_KEYS`）と 1 対 1 であることの観測点。compute 系まで
    // 明示する語彙なので、workgroup 系が拒否されないことまで見る。
    const manifest = parseManifest(withQuant({
      requiredLimits: {
        maxUniformBufferBindingSize: 65536,
        maxStorageBuffersPerShaderStage: 10,
        maxUniformBuffersPerShaderStage: 12,
        maxComputeWorkgroupStorageSize: 32768,
        maxComputeInvocationsPerWorkgroup: 1024,
        maxComputeWorkgroupSizeX: 1024,
        maxComputeWorkgroupSizeY: 1024,
        maxComputeWorkgroupSizeZ: 64,
        maxComputeWorkgroupsPerDimension: 65535,
      },
    }));
    assertEquals(
      Object.keys(manifest.models["m"].quants["q"].requiredLimits ?? {}).length,
      9,
    );
  });

  await t.step("未知の limit 名は許可一覧つきで落ちる（綴り違いを黙って無視しない）", () => {
    const error = assertThrows(
      () => parseManifest(withQuant({ requiredLimits: { maxBufferSizes: 1024 } })),
      ManifestReferenceError,
    );
    assert(
      error.message.includes("maxBufferSize"),
      `${error.message} が許可される名前を出していない`,
    );
  });

  await t.step("非正整数は期待と実際を添えて落ちる", () => {
    for (const value of [0, -1, 1.5, "1024", null]) {
      const error = assertThrows(
        () => parseManifest(withQuant({ requiredLimits: { maxBufferSize: value } })),
        ManifestFormatError,
        undefined,
        `requiredLimits.maxBufferSize = ${JSON.stringify(value)} が通ってしまった`,
      );
      assert(
        error.message.includes("期待 正の安全整数"),
        `${error.message} が期待を出していない`,
      );
    }
  });
});

const COMMIT = "0123456789abcdef0123456789abcdef01234567";

Deno.test("parseManifest: ファイル参照の越境席 repo / revision（ADR 0038 §7）", async (t) => {
  const cross = { repo: "other/stack", revision: COMMIT };
  /** 容器まるごとを越境させた形（ADR 0109 決定 3 — 越境は容器単位）。 */
  const foreignContainer = (patch: Record<string, unknown> = {}) => ({
    descriptor: DESCRIPTOR,
    parts: [{ ...PART0, ...cross, ...patch }, { ...PART1, ...cross, ...patch }],
  });

  await t.step("容器の全 part に同じ座標が載る", () => {
    const container = containerOf(
      withModel({ weights: { net: { f16: { container: foreignContainer() } } } }),
    );
    for (const ref of container.parts) {
      assertEquals(ref.repo, "other/stack");
      assertEquals(ref.revision, COMMIT);
    }
  });

  await t.step("assets の参照にも同じ席が載る", () => {
    const manifest = parseManifest(withModel({
      assets: { tokenizer: { ...FILE, ...cross } },
    }));
    assertEquals(manifest.models["m"].assets["tokenizer"].repo, "other/stack");
    assertEquals(manifest.models["m"].assets["tokenizer"].revision, COMMIT);
  });

  await t.step("越境は容器単位 — 混在も片方だけも落ちる", () => {
    const mixed: unknown[][] = [
      [PART0, { ...PART1, ...cross }],
      [{ ...PART0, ...cross }, PART1],
      [
        { ...PART0, ...cross },
        { ...PART1, repo: "another/stack", revision: COMMIT },
      ],
    ];
    for (const parts of mixed) {
      const error = assertThrows(
        () => parseManifest(withModel({ weights: withContainer({ parts }) })),
        ManifestFormatError,
        undefined,
        `${JSON.stringify(parts)} が通ってしまった`,
      );
      assert(
        error.message.includes("容器単位 — ADR 0109 決定 3"),
        `${error.message} が容器単位の規則を名乗っていない`,
      );
    }
  });

  await t.step("片方だけの宣言は両方向とも落ちる", () => {
    for (const half of [{ repo: "other/stack" }, { revision: COMMIT }]) {
      const error = assertThrows(
        () =>
          parseManifest(withModel({
            weights: withContainer({ parts: [{ ...PART0, ...half }, { ...PART1, ...half }] }),
          })),
        ManifestFormatError,
        undefined,
        `${JSON.stringify(half)} だけの宣言が通ってしまった`,
      );
      assert(
        error.message.includes("両方同時"),
        `${error.message} が「両方同時」の要求を出していない`,
      );
    }
  });

  await t.step("revision はブランチ・タグ・短縮形・大文字を拒否する", () => {
    for (const revision of ["main", "v1.0", COMMIT.slice(0, 7), COMMIT.toUpperCase()]) {
      assertThrows(
        () =>
          parseManifest(withModel({
            weights: { net: { f16: { container: foreignContainer({ revision }) } } },
          })),
        ManifestFormatError,
        undefined,
        `revision '${revision}' が通ってしまった`,
      );
    }
  });

  await t.step("repo は owner/name の 2 セグメント許可リスト", () => {
    for (const repo of ["stack", "other/stack/extra", "other/..", "other/.hidden", "other/re po"]) {
      assertThrows(
        () =>
          parseManifest(withModel({
            weights: { net: { f16: { container: foreignContainer({ repo }) } } },
          })),
        ManifestFormatError,
        undefined,
        `repo '${repo}' が通ってしまった`,
      );
    }
  });

  await t.step("同じ path でもリポが違えば別のファイル（3 点セット一致を要求しない）", () => {
    // path だけで畳むと、正しい manifest が「重複 path の食い違い」で拒否され、取得層では
    // 片方のバイト列がもう片方に配られる。同一性は (repo, revision, path) の 3 つ。
    const manifest = parseManifest(withModel({
      weights: {
        net: { f16: { container: CONTAINER } },
        text: {
          f16: {
            container: {
              descriptor: DESCRIPTOR,
              parts: [
                { ...PART0, ...cross, size: PART0_SIZE },
                { ...PART1, ...cross, size: 128, sha256: "e5".repeat(32) },
              ],
            },
          },
        },
      },
      quants: { q: { weights: { net: "f16", text: "f16" }, session: {} } },
    }));
    assertEquals(manifest.models["m"].weights["net"]["f16"].container.parts[1].size, 64);
    assertEquals(manifest.models["m"].weights["text"]["f16"].container.parts[1].size, 128);
  });

  await t.step("同一の (repo, revision, path) は 1 本に畳まれる", () => {
    const manifest = parseManifest(withModel({
      weights: {
        net: { f16: { container: foreignContainer() } },
        text: { f16: { container: foreignContainer() } },
      },
      quants: { q: { weights: { net: "f16", text: "f16" }, session: {} } },
    }));
    const net = manifest.models["m"].weights["net"]["f16"].container;
    const text = manifest.models["m"].weights["text"]["f16"].container;
    assert(net.parts[1] === text.parts[1], "同一参照が畳まれていない");
  });

  await t.step("同一の (repo, revision, path) で 3 点セットが食い違えば拒否する", () => {
    assertThrows(
      () =>
        parseManifest(withModel({
          weights: {
            net: {
              f16: {
                container: {
                  descriptor: DESCRIPTOR,
                  parts: [
                    { ...PART0, ...cross },
                    { ...PART1, ...cross },
                    { ...PART1, ...cross, size: 99 },
                  ],
                },
              },
            },
          },
        })),
      ManifestReferenceError,
    );
  });
});

Deno.test("parseManifest: quantのGEMV加算指定を保持し、未指定と不正値を区別する", async (t) => {
  for (const linearGemvReduce of ["sequential", "parallel"] as const) {
    await t.step(linearGemvReduce, () => {
      const manifest = parseManifest(withModel({
        quants: { q: { weights: { net: "f16" }, session: { linearGemvReduce } } },
      }));
      assertEquals(manifest.models.m.quants.q.session, { linearGemvReduce });
    });
  }
  assertEquals(parseManifest(withModel()).models.m.quants.q.session, {});
  for (const value of ["auto", "Parallel", 1, true, null]) {
    assertThrows(() =>
      parseManifest(withModel({
        quants: { q: { weights: { net: "f16" }, session: { linearGemvReduce: value } } },
      })), HubError);
  }
});

Deno.test("parseManifest: 融合の真偽値を保持し、未指定・不正値と区別する", async (t) => {
  for (const key of ["fuseRmsNormAdd", "fuseLinearStaticQuantize", "packedStaticQuantize"]) {
    for (const value of [false, true]) {
      await t.step(`${key}=${value}`, () => {
        const session = { [key]: value };
        const manifest = parseManifest(withModel({
          quants: { q: { weights: { net: "f16" }, session } },
        }));
        assertEquals(manifest.models.m.quants.q.session, session);
      });
    }
    for (const value of [null, 0, 1, "true", "false", [], {}]) {
      // 段名に値を綴るのは、assertThrowsの既定メッセージが対象キー・値を出さないため。
      await t.step(`${key}=${JSON.stringify(value)}は拒否`, () => {
        assertThrows(
          () =>
            parseManifest(withModel({
              quants: { q: { weights: { net: "f16" }, session: { [key]: value } } },
            })),
          HubError,
        );
      });
    }
  }
  // 実配布のi4-fastと同じ形（GEMV指定と真偽値3欄の同時宣言）を1件固定する。
  await t.step("linearGemvReduceと同時に宣言できる", () => {
    const session = {
      linearGemvReduce: "parallel",
      fuseRmsNormAdd: true,
      fuseLinearStaticQuantize: true,
      packedStaticQuantize: true,
    } as const;
    const manifest = parseManifest(withModel({
      quants: { q: { weights: { net: "f16" }, session } },
    }));
    assertEquals(manifest.models.m.quants.q.session, session);
  });
  await t.step("未宣言なら欄ごと無い", () => {
    assertEquals(parseManifest(withModel()).models.m.quants.q.session, {});
  });
});
