import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { admitGemma4Qat, assertGemma4QatModel, assertGemma4QatPle } from "../src/gemma/qat.ts";
import { gemma4QatRopeInputs, gemma4RopeInputs, type Gemma4RopeSpec } from "../src/gemma/rope.ts";
import { Gemma4QatPipeline } from "../gemma4-qat.ts";
import type { Gemma4PleIndex } from "../src/gemma/ple-index.ts";
import { pleFixture } from "./helpers/ple-fixture.ts";

type Graph = Parameters<typeof admitGemma4Qat>[0];

/**
 * 構造門を 1 分岐ずつ踏むための改変（1 つの fault で 1 つの throw だけを起こす）。
 *
 * `undefined` は正常系。`head-without-srq` だけは**通る**改変で、共有 head の前後に SRQ が
 * 無い形（公式 checkpoint の lm_head は SRQ scale が 0 = 未較正）を表す。
 */
type Fault =
  | "no-per-layer-inputs"
  | "layers"
  | "ple-dim"
  | "hidden"
  | "two-embeddings"
  | "embedding-i8"
  | "no-head"
  | "no-projection"
  | "two-projections"
  | "linear-not-initializer"
  | "linear-shared"
  | "linear-f32"
  | "only-i4"
  | "srq-before-missing"
  | "srq-after-missing"
  | "extra-consumer"
  | "head-without-srq";

/** f32 linear が 1 本だけ引く重み（門はこの initializer 名で projection を名指す）。 */
const PROJECTION = "model.model.per_layer_model_projection.weight";

const graphOf = (model: "e2b" | "e4b", fault?: Fault): Graph => {
  const hidden = model === "e2b" ? 1536 : 2560,
    layers = model === "e2b" ? 35 : 42;
  let raw: { op: string; ins: string[]; outs: string[] }[] = [
    { op: "embedding", ins: ["head", "input_ids"], outs: ["embedded"] },
    { op: "linear", ins: ["embedded", PROJECTION], outs: ["projected"] },
    { op: "static_quantize", ins: ["projected"], outs: ["input4"] },
    { op: "linear", ins: ["input4", "w4"], outs: ["linear4"] },
    { op: "static_quantize", ins: ["linear4"], outs: ["input8"] },
    { op: "linear", ins: ["input8", "w8"], outs: ["linear8"] },
    { op: "static_quantize", ins: ["linear8"], outs: ["input2"] },
    { op: "linear", ins: ["input2", "head"], outs: ["linear2"] },
    { op: "static_quantize", ins: ["linear2"], outs: ["logits"] },
  ];
  // 合流後の initializer は**名前が実体の鍵**（docs/ir-v2.md）。門が f32 projection を
  // 名指すのもこの名前なので、宣言名は FQN そのものにする。
  const initializers: Record<string, Graph["initializers"][string]> = {
    head: { storage: { codec: "int2-off" } },
    [PROJECTION]: { storage: { codec: "f32" } },
    w4: { storage: { codec: "int4-sym-g", groupSize: 32, rowAxis: 0 } },
    w8: { storage: { codec: "int8-sym" } },
  };
  let pleShape: (string | number)[] = [1, "M", layers, 256];
  let inputs: Graph["inputs"] = [
    { name: "input_ids", dtype: "i32", shape: [1, "M"] },
    { name: "per_layer_inputs", dtype: "f32", shape: pleShape },
  ];
  let hiddenWidth: string | number = hidden;
  switch (fault) {
    case "no-per-layer-inputs":
      inputs = [{ name: "input_ids", dtype: "i32", shape: [1, "M"] }];
      break;
    case "layers":
      pleShape = [1, "M", 7, 256];
      inputs = [inputs[0], { name: "per_layer_inputs", dtype: "f32", shape: pleShape }];
      break;
    case "ple-dim":
      pleShape = [1, "M", layers, 128];
      inputs = [inputs[0], { name: "per_layer_inputs", dtype: "f32", shape: pleShape }];
      break;
    case "hidden":
      hiddenWidth = hidden + 1;
      break;
    case "two-embeddings":
      raw = [...raw, { op: "embedding", ins: ["head", "input_ids"], outs: ["embedded2"] }];
      break;
    case "embedding-i8":
      initializers.head = { storage: { codec: "int8-sym" } };
      break;
    case "no-head":
      // head の linear が token embedding と別の重みを引く（共有 head が 0 本になる）。
      raw = raw.map((node, i) => i === 7 ? { ...node, ins: ["input2", "w4"] } : node);
      break;
    case "no-projection":
      raw = raw.filter((_, i) => i !== 1);
      break;
    case "two-projections":
      raw = [...raw, { op: "linear", ins: ["embedded", PROJECTION], outs: ["projected2"] }];
      break;
    case "linear-not-initializer":
      raw = raw.map((node, i) => i === 3 ? { ...node, ins: ["input4", "absent"] } : node);
      break;
    case "linear-shared":
      initializers.w8 = { shared: true };
      break;
    case "linear-f32":
      initializers.w4 = { storage: { codec: "f32" } };
      break;
    case "only-i4":
      initializers.w8 = { storage: { codec: "int4-sym-g", groupSize: 32, rowAxis: 0 } };
      break;
    case "srq-before-missing":
      raw = raw.map((node, i) => i === 2 ? { ...node, op: "reshape" } : node);
      break;
    case "srq-after-missing":
      raw = raw.map((node, i) => i === 4 ? { ...node, op: "reshape" } : node);
      break;
    case "extra-consumer":
      raw = [...raw, { op: "reshape", ins: ["linear4"], outs: ["spare"] }];
      break;
    case "head-without-srq":
      // 共有 head の前後から SRQ を外す（前は reshape・後ろは直接グラフ出口）。
      raw = [
        ...raw.slice(0, 7),
        { op: "reshape", ins: ["input2"], outs: ["normed"] },
        { op: "linear", ins: ["normed", "head"], outs: ["logits"] },
      ];
      break;
  }
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["embedding", "linear", "static_quantize"] },
    symbols: ["M", "R", "C"],
    inputs,
    outputs: ["logits", "hidden"],
    values: {
      logits: { dtype: "f32", shape: [1, "R", 262144] },
      hidden: { dtype: "f32", shape: [1, "R", hiddenWidth] },
    },
    initializers,
    states: {},
    nodes: raw.map((node) => ({
      ...node,
      attrs: node.op === "static_quantize" ? { scale: 0.125 } : {},
      states: {},
    })),
  };
};

/** 索引 1 本（block 1 本の合成 — 寸法だけが門の対象なので桁は小さくてよい）。 */
const pleIndexOf = (model: "e2b" | "e4b"): Gemma4PleIndex =>
  pleFixture({
    tokens: 64,
    layers: model === "e2b" ? 35 : 42,
    dim: 256,
    embedScale: 16,
    storage: model === "e2b" ? "i4" : "i2",
  }).index;

Deno.test("固定 QAT の family admission", async (t) => {
  for (const model of ["e2b", "e4b"] as const) {
    await t.step(`${model} の固定混成と共有 head を受ける`, () => {
      const graph = graphOf(model);
      assertEquals(admitGemma4Qat(graph, model), model);
      assertEquals(admitGemma4Qat(graph), model);
      assertThrows(
        () => admitGemma4Qat(graph, model === "e2b" ? "e4b" : "e2b"),
        Error,
        "構成が違う",
      );
    });
    await t.step(`${model} の共有 head は前後の SRQ が無くても受ける`, () => {
      // 公式 checkpoint の lm_head は SRQ の scale が入出力とも 0（未較正 = 恒等）なので、
      // recipe は恒等 SRQ を IR に挟まない。SRQ 有りの形（上の step）も受ける。
      assertEquals(admitGemma4Qat(graphOf(model, "head-without-srq"), model), model);
    });
  }

  // 拒否の 1 分岐 = 1 fault。文言まで縛るのは、条件を 1 つ書き換えたときに「別の理由で
  // たまたま落ちる」形を緑にしないため（門の 3 条件同居を解いたのがこの表の前提）。
  const rejections: readonly [Fault, string][] = [
    ["no-per-layer-inputs", "per_layer_inputs が無い"],
    ["layers", "E2B 35 / E4B 42 でない"],
    ["ple-dim", "PLE 次元"],
    ["hidden", "hidden 幅"],
    ["two-embeddings", "token embedding が1本でない"],
    ["embedding-i8", "token embedding は固定 INT2"],
    ["no-head", "token embedding を共有する head が 0 本"],
    ["no-projection", "per_layer_model_projection の f32 linear が 0 本"],
    ["two-projections", "per_layer_model_projection の f32 linear が 2 本"],
    ["linear-not-initializer", "が initializer でない"],
    ["linear-shared", "共有 initializer"],
    ["linear-f32", "固定 INT2/INT4/INT8 が必要"],
    ["only-i4", "固定 INT4 と INT8 を揃えていない"],
    ["srq-before-missing", "前後に固定 SRQ"],
    ["srq-after-missing", "前後に固定 SRQ"],
    ["extra-consumer", "前後に固定 SRQ"],
  ];
  for (const [fault, message] of rejections) {
    await t.step(`改変 ${fault} を拒否する`, () => {
      assertThrows(() => admitGemma4Qat(graphOf("e2b", fault)), Error, message);
    });
  }

  await t.step("未対応の model 名を拒否する", () => {
    assertThrows(() => assertGemma4QatModel("12b"), Error, "未対応 model '12b'");
    assertEquals(assertGemma4QatModel("e2b"), "e2b");
    assertEquals(assertGemma4QatModel("e4b"), "e4b");
    assertThrows(() => admitGemma4Qat(graphOf("e2b"), "12b"), Error, "未対応 model '12b'");
  });

  await t.step("PLE の格納・層数・次元をグラフへ突合する", () => {
    for (const model of ["e2b", "e4b"] as const) {
      assertGemma4QatPle(model, graphOf(model), pleIndexOf(model));
    }
    const index = pleIndexOf("e2b");
    assertThrows(
      () => assertGemma4QatPle("e2b", graphOf("e2b"), { ...index, storage: "i2" }),
      Error,
      "PLE の格納 'i2'",
    );
    assertThrows(
      () => assertGemma4QatPle("e4b", graphOf("e4b"), { ...pleIndexOf("e4b"), storage: "i4" }),
      Error,
      "PLE の格納 'i4'",
    );
    assertThrows(
      () => assertGemma4QatPle("e2b", graphOf("e2b"), { ...index, layers: 42 }),
      Error,
      "PLE の層数 42",
    );
    assertThrows(
      () => assertGemma4QatPle("e2b", graphOf("e2b"), { ...index, dim: 128 }),
      Error,
      "PLE の次元 128",
    );
  });
});

Deno.test("固定 QAT は MTP 指定を資産取得より前に拒否する", async (t) => {
  // 型側は `speculative?: never` で塞いであるが、JS の呼び手・`as` 経由・動的に組んだ option には
  // 効かない。実行時の拒否が**資産を 1 byte も取る前**に出ることを、存在しない取得元で確かめる。
  await t.step("fromPretrained", async () => {
    await assertRejects(
      () => Gemma4QatPipeline.fromPretrained("does-not-exist", { speculative: true } as never),
      Error,
      "QAT の MTP は未対応",
    );
  });
  await t.step("fromAssets", async () => {
    await assertRejects(
      () => Gemma4QatPipeline.fromAssets({} as never, { speculative: true } as never),
      Error,
      "speculative は受けられない",
    );
  });
});

Deno.test("QAT のホスト RoPE", async (t) => {
  const spec = {
    sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
    full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
  } satisfies Gemma4RopeSpec;
  await t.step("pad と回転しない次元を厳密に保存し、入力4本を返す", () => {
    const input = gemma4QatRopeInputs(spec, [0, 17, 127]);
    assertEquals(Object.keys(input).length, 4);
    for (const [key, tensor] of Object.entries(input)) {
      const width = key.includes("full_attention") ? 512 : 256;
      assertEquals(tensor.shape, [1, 3, width]);
      assertEquals(tensor.dtype, "f32");
      assertEquals(
        Array.from(tensor.data.slice(0, width)),
        Array(width).fill(key.endsWith("cos") ? 1 : 0),
      );
      for (let i = 0; i < tensor.data.length; i++) {
        assert(Number.isFinite(tensor.data[i]));
      }
    }
    for (const part of ["cos", "sin"]) {
      const t = input[`rope_full_attention_${part}`];
      for (let row = 0; row < 3; row++) {
        for (let i = 64; i < 256; i++) {
          assertEquals(t.data[row * 512 + i], part === "cos" ? 1 : 0);
          assertEquals(t.data[row * 512 + i + 256], part === "cos" ? 1 : 0);
        }
      }
    }
  });
  await t.step("角度の f32 丸めを保ち、通常 Gemma の f64 契約へ戻らない", () => {
    // Torch f32: sin(tensor(127) * (1 / tensor(10))) = 0x3e086dee。
    // theta=10000、次元32/256は逆周波数1/10。べき乗そのものは厳密なので角度の丸めだけを検査する。
    const key = "rope_sliding_attention_sin";
    assertEquals(gemma4QatRopeInputs(spec, [127])[key].data[32], 0.13323184847831726);
    assertEquals(gemma4RopeInputs(spec, [127])[key].data[32], 0.13323204219341278);
  });
  await t.step("両層種別 × 位置 5 点の cos / sin を golden とビット単位で突合する", async () => {
    // 1 要素だけの検査では、逆周波数が厳密に表せる次元を選んだときに「べき乗段の丸めが
    // 抜けても緑」になる。golden は表全体を u32 で凍結してその抜けを捕まえる。
    // MUST: これは **karume 自身の出力の凍結**で、上流 Torch とのビット一致ではない
    // （ADR 0097 追記 5 が全ビット一致を保証していない）。丸め契約を意図して変える
    // ときだけ焼き直す。
    const golden = JSON.parse(
      await Deno.readTextFile(
        new URL("./fixtures/gemma4-qat-rope-golden.json", import.meta.url),
      ),
    ) as {
      readonly positions: readonly number[];
      readonly spec: Gemma4RopeSpec;
      readonly tables: Readonly<Record<string, string>>;
    };
    assertEquals(golden.spec, spec);
    const inputs = gemma4QatRopeInputs(golden.spec, golden.positions);
    assertEquals(Object.keys(inputs).sort(), Object.keys(golden.tables).sort());
    for (const [name, base64] of Object.entries(golden.tables)) {
      const bytes = Uint8Array.from(atob(base64), (char) => char.charCodeAt(0));
      const expected = new Uint32Array(bytes.buffer as ArrayBuffer);
      const data = inputs[name].data as Float32Array<ArrayBuffer>;
      const actual = new Uint32Array(data.buffer, data.byteOffset, data.length);
      assertEquals(actual.length, expected.length, `${name}: 要素数`);
      for (let i = 0; i < expected.length; i++) {
        if (actual[i] === expected[i]) continue;
        throw new Error(
          `${name}[${i}]: 0x${actual[i].toString(16)} が golden 0x${
            expected[i].toString(16)
          } と違う`,
        );
      }
    }
  });
  await t.step("不正な位置と f32 で表せない逆周波数を拒否する", () => {
    assertThrows(() => gemma4QatRopeInputs(spec, [-1]), Error, "非負整数");
    assertThrows(
      () =>
        gemma4QatRopeInputs({
          ...spec,
          full_attention: { theta: 1e300, headDim: 4, rotaryDim: 4 },
        }, [1]),
      Error,
      "f32 逆周波数",
    );
  });
});
