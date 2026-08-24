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

const FILE = { path: "net/model.f16.safetensors", size: 4, sha256: "a1".repeat(32) };

/**
 * 検査に要る欄だけを持つ最小の v4 manifest。`patch` は `models.m` の中身を、`envelope` は
 * トップレベルを上書きする。
 */
const withModel = (
  patch: Record<string, unknown> = {},
  envelope: Record<string, unknown> = {},
): string =>
  JSON.stringify({
    format: "karume/4",
    generator: "karume/0.1.0",
    defaultModel: "m",
    models: {
      m: {
        pipeline: "anima/1",
        weights: { net: { f16: { shards: [FILE] } } },
        assets: {},
        quants: { q: { weights: { net: "f16" }, session: {} } },
        defaultQuant: "q",
        pipelineConfig: {},
        ...patch,
      },
    },
    ...envelope,
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
    () => parseManifest('{"format": "karume/4",}'),
    ManifestFormatError,
  );
  assert(error.cause instanceof SyntaxError, "元の SyntaxError を cause に残す");
});

Deno.test("parseManifest: v1（karume/1）は読まずに未対応 major として落とす", () => {
  // ADR 0041 §1: hub は現行版だけを読む（2 形パースを持たない）。旧クライアントの裏返しで、
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
    error.message.includes("karume/4"),
    `${error.message} が「読めるのは karume/4」を名指ししていない`,
  );
});

Deno.test("parseManifest: 直前版（karume/3）も読まず、現行が karume/4 であることを名指しする", () => {
  // v3 と v4 の差は optional な新席（label / description / requiredLimits / 越境参照）だけで、
  // v3 の manifest は**構造としては v4 のパーサを素通りしてしまう**。だから断絶は format 文字列
  // だけが宣言する（ADR 0075 決定 4）。診断が「拒否した版」と「この版が読む版」の両方を
  // 名指しすることまでを観測値として固定する。
  const error = assertThrows(
    () =>
      parseManifest(JSON.stringify({
        format: "karume/3",
        generator: "karume/0.1.0",
        defaultModel: "m",
        models: {
          m: {
            pipeline: "anima/1",
            weights: { net: { f16: { shards: [FILE] } } },
            assets: {},
            quants: { q: { weights: { net: "f16" }, session: {} } },
            defaultQuant: "q",
            pipelineConfig: {},
          },
        },
      })),
    ManifestFormatError,
  );
  assert(
    error.message.includes("karume/3"),
    `${error.message} が拒否した版を名指ししていない`,
  );
  assert(
    error.message.includes("karume/4"),
    `${error.message} が「読めるのは karume/4」を名指ししていない`,
  );
  assert(
    error.message.includes("旧版"),
    `${error.message} が「旧版のパーサを持たない」ことを伝えていない`,
  );
});

Deno.test("parseManifest: karume/2 の綴り（dtype エントリが {file}）も版で落とす", () => {
  // 構造から入ると「未知キー 'file'」という枝葉の診断になり、本当の理由（この版は karume/4 のみ
  // 読む）が隠れる — format を未知キー検査より先に見ていることの観測点。
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
    weights: { net: { f16: { shards: [FILE] } } },
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
      format: "karume/4",
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
      weights = { ...weights, [`w${index}`]: { f16: { shards: [FILE] } } };
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

  await t.step("shards 1025 件（1024 件は通る）", () => {
    const shards = (count: number): unknown[] =>
      Array.from({ length: count }, (_, index) => ({
        path: `net/model.f16.shard${index}.safetensors`,
        size: 8,
        sha256: "b2".repeat(32),
      }));
    // 上限ちょうどが通ることまで見る（片側だけだと「常に落ちる」実装でも緑になる）。
    const accepted = parseManifest(
      withModel({ weights: { net: { f16: { shards: shards(1024) } } } }),
    );
    assertEquals(accepted.models["m"].weights["net"]["f16"].shards.length, 1024);
    assertThrows(
      () => parseManifest(withModel({ weights: { net: { f16: { shards: shards(1025) } } } })),
      ManifestFormatError,
    );
  });

  await t.step("assets 33 件", () => {
    let assets: Record<string, unknown> = {};
    for (let index = 0; index < 33; index += 1) assets = { ...assets, [`a${index}`]: FILE };
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
  const error = assertThrows(
    () =>
      parseManifest(JSON.stringify({
        format: "karume/4",
        generator: "karume/0.1.0",
        defaultModel: "fast",
        models: {
          fast: {
            pipeline: "anima/1",
            weights: {
              net: {
                f16: {
                  shards: [{ path: "net/f16.safetensors", size: 8, sha256: "b2".repeat(32) }],
                },
                i8: { shards: [{ path: "net/i8.safetensors", size: 4, sha256: "c3".repeat(32) }] },
              },
            },
            assets: {},
            quants: { w8: { weights: { net: "q4" }, session: {} } },
            defaultQuant: "w8",
            pipelineConfig: {},
          },
          slim: {
            pipeline: "anima/1",
            weights: {
              net: { i8: { shards: [{ path: "slim/i8.st", size: 2, sha256: "d4".repeat(32) }] } },
            },
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
  assertEquals(manifest.format, "karume/4");
  assertEquals(manifest.generator, "karume/0.1.0");
  assertEquals(manifest.defaultModel, "anima-turbo");
  assertEquals(Object.keys(manifest.models), ["anima-turbo", "anima-lite"]);
  assertEquals(manifest.available.models, ["anima-turbo", "anima-lite"]);

  const turbo = manifest.models["anima-turbo"];
  assertEquals(turbo.pipeline, { name: "anima", major: 1 });
  assertEquals(turbo.defaultQuant, "w8a8-s16");
  assertEquals(turbo.available.quants, ["f16", "w8a8-s16", "f16-c16"]);
  assertEquals(turbo.pipelineConfig, { defaults: { steps: 8, guidanceScale: 1 } });

  // weights は dtype キー必須の統一形（v1 の {file} / {variants} 2 形は無い）。
  assertEquals(Object.keys(turbo.weights["transformer"]), ["f16", "i8"]);
  assertEquals(Object.keys(turbo.weights["vae_decoder"]), ["f16"]);
  assertEquals(
    turbo.weights["transformer"]["i8"].shards.map((shard) => shard.path),
    ["transformer/model.i8.safetensors"],
  );
  // extras は dtype エントリの内側（ADR 0041 §3 は席を動かしていない）。
  assertEquals(Object.keys(turbo.weights["transformer"]["i8"].extras), ["rope_base"]);
  assertEquals(turbo.weights["vae_decoder"]["f16"].extras, {});

  // assets は quant 選択に依存しない無条件ファイル（dtype の階層を持たない）。
  assertEquals(Object.keys(turbo.assets), ["tokenizer", "rope_alias"]);
  assertEquals(turbo.assets["tokenizer"].path, "tokenizer/qwen2-tokenizer.json");

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
  const shared = manifest.models["anima-turbo"].weights["text_encoder"]["f16"].shards[0];
  const same = manifest.models["anima-lite"].weights["text_encoder"]["f16"].shards[0];
  assertEquals(same.path, shared.path);
  assertEquals(same.sha256, shared.sha256);
  // 表は 1 本なので、同じ path は同じ FileRef インスタンスに畳まれる（取得も 1 回になる）。
  assert(same === shared, "同一 path の参照が畳まれていない");
});

Deno.test("parseManifest: weights の shards 欄（ADR 0070 決定 1）", async (t) => {
  const shard = (name: string, size: number, mark: string) => ({
    path: `net/${name}`,
    size,
    sha256: mark.repeat(32),
  });

  await t.step("1 要素の shards は単一ファイル配布として読める", () => {
    const manifest = parseManifest(withModel());
    assertEquals(manifest.models["m"].weights["net"]["f16"].shards, [FILE]);
  });

  await t.step("複数要素は宣言順のまま保たれる（配列位置が shard id）", () => {
    const graph = shard("graph.safetensors", 6, "a1");
    const first = shard("weights-0.safetensors", 8, "b2");
    const second = shard("weights-1.safetensors", 4, "c3");
    const manifest = parseManifest(
      withModel({ weights: { net: { f16: { shards: [graph, first, second] } } } }),
    );
    // 並べ替えも重複畳み込みもしない — 順序は先頭 = グラフ shard という意味を持つ（検査は runtime）。
    assertEquals(
      manifest.models["m"].weights["net"]["f16"].shards.map((entry) => entry.path),
      ["net/graph.safetensors", "net/weights-0.safetensors", "net/weights-1.safetensors"],
    );
  });

  await t.step("shards の中でも同一 path の 3 点セット不一致は拒否する", () => {
    const error = assertThrows(
      () =>
        parseManifest(withModel({
          weights: {
            net: {
              f16: {
                shards: [
                  shard("weights-0.safetensors", 8, "b2"),
                  shard(
                    "weights-0.safetensors",
                    8,
                    "c3",
                  ),
                ],
              },
            },
          },
        })),
      ManifestReferenceError,
    );
    assert(
      error.message.includes("weights-0.safetensors"),
      `${error.message} が食い違った path を名指ししていない`,
    );
  });

  await t.step("extras は shards と独立の席のまま", () => {
    const manifest = parseManifest(withModel({
      weights: {
        net: {
          f16: {
            shards: [shard("weights-0.safetensors", 8, "b2")],
            extras: { rope_base: shard("rope_base.safetensors", 6, "d4") },
          },
        },
      },
    }));
    const files = manifest.models["m"].weights["net"]["f16"];
    assertEquals(files.shards.length, 1);
    assertEquals(Object.keys(files.extras), ["rope_base"]);
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
  const foreign = (patch: Record<string, unknown> = {}) => ({
    path: "text_encoder/model.safetensors",
    size: 12,
    sha256: "e5".repeat(32),
    repo: "other/stack",
    revision: COMMIT,
    ...patch,
  });

  await t.step("shard 列の要素に載る（席は 3 点セットと同じ位置）", () => {
    const manifest = parseManifest(
      withModel({ weights: { net: { f16: { shards: [FILE, foreign()] } } } }),
    );
    const shards = manifest.models["m"].weights["net"]["f16"].shards;
    assertEquals(shards[0].repo, undefined, "自リポ参照は席を持たない");
    assertEquals(shards[1].repo, "other/stack");
    assertEquals(shards[1].revision, COMMIT);
  });

  await t.step("extras / assets の参照にも同じ席が載る", () => {
    const manifest = parseManifest(withModel({
      weights: { net: { f16: { shards: [FILE], extras: { rope_base: foreign() } } } },
      assets: { tokenizer: foreign({ path: "tokenizer/tokenizer.json", size: 9 }) },
    }));
    assertEquals(
      manifest.models["m"].weights["net"]["f16"].extras["rope_base"].repo,
      "other/stack",
    );
    assertEquals(manifest.models["m"].assets["tokenizer"].revision, COMMIT);
  });

  await t.step("片方だけの宣言は両方向とも落ちる", () => {
    for (const half of [{ repo: "other/stack" }, { revision: COMMIT }]) {
      const error = assertThrows(
        () =>
          parseManifest(withModel({
            weights: { net: { f16: { shards: [{ ...FILE, ...half }] } } },
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
            weights: { net: { f16: { shards: [foreign({ revision })] } } },
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
          parseManifest(withModel({ weights: { net: { f16: { shards: [foreign({ repo })] } } } })),
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
        net: {
          f16: { shards: [{ path: "shared/model.safetensors", size: 4, sha256: "a1".repeat(32) }] },
        },
        text: {
          f16: {
            shards: [
              {
                path: "shared/model.safetensors",
                size: 12,
                sha256: "b2".repeat(32),
                repo: "other/stack",
                revision: COMMIT,
              },
            ],
          },
        },
      },
      quants: { q: { weights: { net: "f16", text: "f16" }, session: {} } },
    }));
    assertEquals(manifest.models["m"].weights["net"]["f16"].shards[0].size, 4);
    assertEquals(manifest.models["m"].weights["text"]["f16"].shards[0].size, 12);
  });

  await t.step("同一の (repo, revision, path) は 1 本に畳まれる", () => {
    const manifest = parseManifest(withModel({
      weights: {
        net: { f16: { shards: [foreign()], extras: { alias: foreign() } } },
      },
    }));
    const entry = manifest.models["m"].weights["net"]["f16"];
    assert(entry.shards[0] === entry.extras["alias"], "同一参照が畳まれていない");
  });

  await t.step("同一の (repo, revision, path) で 3 点セットが食い違えば拒否する", () => {
    assertThrows(
      () =>
        parseManifest(withModel({
          weights: { net: { f16: { shards: [foreign(), foreign({ size: 99 })] } } },
        })),
      ManifestReferenceError,
    );
  });
});
