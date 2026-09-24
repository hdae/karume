// Session のライフサイクル系のうち、GPU を取らずに回る門（実 GPU 側は
// gpu_runtime_executor_test.ts）。

import { assertRejects } from "@std/assert";
import { ContainerFormatError, writeHeader } from "../src/format/container/header.ts";
import { HEADER_BYTES } from "../src/format/container/limits.ts";
import { openContainer } from "../src/format/container/open.ts";
import { parseIrDeclarationValue } from "../src/format/ir.ts";
import { chainGraph, chainTensors } from "./helpers/chain-graph.ts";
import { GRAPH_NAME } from "./helpers/model-fixture.ts";
import { writeModelContainer } from "./helpers/container-write.ts";

// deno-lint-ignore no-explicit-any
const anyOf = (value: unknown): any => value;

Deno.test({
  name: "束縛が取れないシンボルは容器の読み手が受理しない（入力 shape に素の形で現れない）",
  fn: async () => {
    // 書き手は宣言をパースしてから書くので、壊れた宣言は容器に入らない。読み手の側の門
    // （`descriptor.ts` の宣言パース）を撃つには、**正しい krm を書いてからグラフ記述の
    // バイト列を壊す**しかない。
    const written = await writeModelContainer({
      graphs: { [GRAPH_NAME]: parseIrDeclarationValue(chainGraph()) },
      consts: [],
      weights: chainTensors(),
      assets: [],
      provenance: { license: "test" },
    }, {});
    const descriptorDoc = JSON.parse(new TextDecoder().decode(written.graphDescriptorBytes));
    const declaration = anyOf(descriptorDoc).graphs[GRAPH_NAME];
    // `S` は入力 shape に素の形で現れないので、どの実行時値からも束縛が取れない。
    declaration.symbols = ["S", "T"];
    declaration.values.h = { dtype: "f32", shape: ["S", 3] };
    const graphBytes = new TextEncoder().encode(JSON.stringify(descriptorDoc)) as Uint8Array<
      ArrayBuffer
    >;
    // 単一形は block の offset が part 0 の長さから決まるので、長さの変わる書き換えは
    // part 列で渡す（part 0 = ヘッダ + 2 文書）。
    const parts = [...written.parts];
    const model = written.modelDescriptorBytes;
    const part0 = new Uint8Array(
      new ArrayBuffer(HEADER_BYTES + graphBytes.byteLength + model.byteLength),
    );
    part0.set(
      writeHeader({
        kind: "model",
        version: 1,
        graphDescriptorLength: graphBytes.byteLength,
        modelDescriptorLength: model.byteLength,
      }),
      0,
    );
    part0.set(graphBytes, HEADER_BYTES);
    part0.set(model, HEADER_BYTES + graphBytes.byteLength);
    parts[0] = part0;
    await assertRejects(
      () => openContainer({ kind: "parts", parts }),
      ContainerFormatError,
      "束縛が取れない",
    );
  },
});
