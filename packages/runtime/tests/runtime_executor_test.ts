// Session のライフサイクル系のうち、GPU を取らずに回る門（実 GPU 側は
// gpu_runtime_executor_test.ts）。

import { assertThrows } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { IrError } from "../src/format/ir.ts";
import { chainGraph, chainModelBuffer } from "./helpers/chain-graph.ts";

Deno.test({
  name: "束縛が取れないシンボルは IR パーサが受理しない（入力 shape に素の形で現れない）",
  fn: () => {
    const unbindable = chainGraph();
    unbindable.symbols = ["T", "S"];
    unbindable.values.h = { dtype: "f32", shape: ["S", 3] };
    assertThrows(() => openModel(chainModelBuffer(unbindable)), IrError, "束縛が取れない");
  },
});
