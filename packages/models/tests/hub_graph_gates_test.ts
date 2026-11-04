// Checks a built graph's declarations against the declarations of `pipelineConfig`
// (`src/hub/graph-gates.ts`). 4 families (birefnet / depth-anything / irodori / siglip2) go through
// this one path.
//
// This check **has only failure paths**: a mismatch that gets through passes the host side all the
// way and fails only at the Session's shape check = a place where the reader cannot tell "which is
// right, the declaration or the graph". Three points are checked:
//
// ① A match passes, and a mismatch on even one axis fails (the comparison is `===` — the string of
//    a symbolic dimension does not equal a number).
// ② When the input name itself is missing, it fails naming it before reading the axis.
// ③ The message carries the family, which declaration was being checked (`where`), the input name,
//    the axis, the actual value and the expected value.

import { test } from "vite-plus/test";
import { assert, assertThrows } from "@std/assert";
import { stubModel } from "./helpers/stub-model.ts";
import { assertGraphInputDim } from "../src/hub/graph-gates.ts";

/** A 224² vision graph (`patch` breaks exactly one point). */
const visionGraph = (shape: readonly (number | string)[] = [1, 3, 224, 224]) =>
  stubModel({
    inputs: [{ name: "pixel_values", shape }],
    outputs: ["pooler_output"],
    values: { pooler_output: [1, 768] },
  });

test("assertGraphInputDim: an axis matching the declaration passes", () => {
  assertGraphInputDim("siglip2", visionGraph(), "pixel_values", 2, 224, "imageHeight");
  assertGraphInputDim("siglip2", visionGraph(), "pixel_values", 1, 3, "3 RGB channels");
});

test("assertGraphInputDim: an axis mismatch fails naming the family, where, the actual value and the expected value", () => {
  const error = assertThrows(
    () =>
      assertGraphInputDim(
        "siglip2",
        visionGraph([1, 3, 384, 384]),
        "pixel_values",
        2,
        224,
        "imageHeight",
      ),
    Error,
    "siglip2: imageHeight — graph input 'pixel_values' axis 2 is 384",
  );
  assert(error.message.includes("pipelineConfig has 224"), error.message);
});

test("assertGraphInputDim: the family label is the caller's as is", () => {
  assertThrows(
    () =>
      assertGraphInputDim("depth-anything", visionGraph(), "pixel_values", 3, 518, "imageWidth"),
    Error,
    "depth-anything: imageWidth — graph input 'pixel_values' axis 3 is 224, pipelineConfig has 518",
  );
});

test("assertGraphInputDim: a missing input name fails before the axis is read", () => {
  assertThrows(
    () => assertGraphInputDim("irodori", visionGraph(), "pixel", 2, 224, "imageHeight"),
    Error,
    "irodori: graph input 'pixel' is missing (imageHeight)",
  );
});

test("assertGraphInputDim: a symbolic-dimension axis fails because it does not equal a static value", () => {
  // A symbolic dimension (a data-dependent axis) required as a static declaration. With `===` the
  // string does not equal a number, and the message shows the symbol name as is (readable as "not
  // built statically").
  assertThrows(
    () =>
      assertGraphInputDim(
        "birefnet",
        visionGraph([1, 3, "H", 224]),
        "pixel_values",
        2,
        224,
        "imageHeight",
      ),
    Error,
    "birefnet: imageHeight — graph input 'pixel_values' axis 2 is H, pipelineConfig has 224",
  );
});

test("assertGraphInputDim: an axis beyond the declared rank fails as undefined", () => {
  // A mixed-up axis number (axis 4 required of a 4-dimensional declaration). Passing silently would
  // add "axes nobody looks at".
  assertThrows(
    () => assertGraphInputDim("birefnet", visionGraph(), "pixel_values", 4, 224, "imageHeight"),
    Error,
    "birefnet: imageHeight — graph input 'pixel_values' axis 4 is undefined, pipelineConfig has 224",
  );
});
