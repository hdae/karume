import { test } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import {
  type IrDeclaration,
  IrError,
  parseIrDeclaration,
  parseIrDeclarationValue,
} from "../src/format/ir.ts";
import {
  baseDeclaration,
  type DeclarationJson,
  withStateReaders,
} from "./helpers/model-fixture.ts";

const parseMutated = (mutate: (graph: DeclarationJson) => void): IrDeclaration => {
  const graph = baseDeclaration();
  mutate(graph);
  return parseIrDeclaration(JSON.stringify(graph));
};

/**
 * Checks that each validation rule actually works by "breaking exactly one point of a valid case".
 *
 * MUST: check the expected message too (the diagnostic specific to that rule) — with the `IrError`
 * type alone, the test would stay green after the targeted rule died if the same input failed on
 * another rule.
 */
const assertRejects = (
  hint: string,
  expected: string,
  mutate: (graph: DeclarationJson) => void,
): void => {
  assertThrows(() => parseMutated(mutate), IrError, expected, hint);
};

// An IR v2 declaration carries only the **computation contract** (storage is the container's
// binding table — docs/ir-v2.md). The initializer name is itself the key to the data, so v1's
// "renaming to the data's tensor key" no longer happens.
test("parseIrDeclaration: accepts the smallest valid graph", () => {
  const graph = parseIrDeclaration(JSON.stringify(baseDeclaration()));
  assertEquals(graph.format, "karume-ir");
  assertEquals(graph.version, 2);
  assertEquals(graph.symbols, ["T"]);
  assertEquals(graph.inputs, [{ name: "x", dtype: "f32", shape: ["T", 4] }]);
  assertEquals(graph.outputs, ["y"]);
  assertEquals(graph.requires.ops, ["add", "matmul"]);
  assertEquals(
    graph.nodes.map((node) => node.op),
    ["matmul", "add"],
  );
  assertEquals(Object.keys(graph.initializers), ["w", "b"]);
  assertEquals(graph.initializers["w"], { shared: false });
  assertEquals(graph.values["w"], { dtype: "f32", shape: [4, 3] });
  assertEquals(graph.nodes[0].ins, ["x", "w"]);
  assertEquals(graph.values["h"], { dtype: "f32", shape: ["T", 3] });
});

test("parseIrDeclaration: fixed values of format / version", () => {
  assertRejects("format mismatch", "graph.format is not 'karume-ir'", (g) => {
    g.format = "karume";
  });
  assertRejects("version mismatch", "graph.version is not 2", (g) => {
    g.version = 1;
  });
});

test("parseIrDeclaration: the top-level key set", () => {
  assertThrows(
    () => parseIrDeclaration(JSON.stringify({ ...baseDeclaration(), extra: 1 })),
    IrError,
    "unknown key",
  );
  const { requires: _dropped, ...withoutRequires } = baseDeclaration();
  assertThrows(() => parseIrDeclaration(JSON.stringify(withoutRequires)), IrError, "required key");
  assertThrows(
    () => parseIrDeclaration(JSON.stringify({ ...baseDeclaration(), nodes: {} })),
    IrError,
    "is not an array",
  );
});

test("parseIrDeclaration: requires.ops equals the set of ops used", () => {
  assertRejects("missing declaration", "undeclared [add]", (g) => {
    g.requires.ops = ["matmul"];
  });
  assertRejects("surplus", "surplus [gelu]", (g) => {
    g.requires.ops = ["matmul", "add", "gelu"];
  });
  assertRejects("duplicate", "graph.requires.ops: 'add' is duplicated", (g) => {
    g.requires.ops = ["matmul", "add", "add"];
  });
});

test("parseIrDeclaration: SSA single assignment", () => {
  assertRejects(
    "node output defined twice",
    "graph.nodes[1] (add): value 'h' is defined twice",
    (g) => {
      g.nodes[1].outs = ["h"];
    },
  );
  assertRejects("duplicate input name", "graph.inputs: value 'x' is defined twice", (g) => {
    g.inputs.push({ name: "x", dtype: "f32", shape: ["T", 4] });
  });
  assertRejects(
    "initializer collides with an input",
    "graph.initializers: value 'x' is defined twice",
    (g) => {
      g.initializers["x"] = {};
      g.values["x"] = { dtype: "f32", shape: [3] };
    },
  );
});

test("parseIrDeclaration: topological order (forward references rejected)", () => {
  assertRejects(
    "nodes out of order",
    "graph.nodes[0] (add): input 'h' is undefined (forward reference",
    (g) => {
      g.nodes.reverse();
    },
  );
  assertRejects("undefined ins", "input 'unknown' is undefined", (g) => {
    g.nodes[0].ins = ["x", "unknown"];
  });
  assertRejects("undefined outputs", "graph.outputs: 'z' is undefined", (g) => {
    g.outputs = ["z"];
  });
  assertRejects("duplicate outputs", "graph.outputs: 'y' is duplicated", (g) => {
    g.outputs = ["y", "y"];
  });
});

test("parseIrDeclaration: completeness of the declaration", () => {
  assertRejects(
    "node output missing from values",
    "no dtype/shape declaration for node output 'y'",
    (g) => {
      delete g.values["y"];
    },
  );
  assertRejects(
    "initializer missing from values",
    "graph.initializers['w']: no dtype/shape declaration in values",
    (g) => {
      delete g.values["w"];
    },
  );
  assertRejects(
    "input declared twice",
    "inputs are declared in inputs[] (double declaration)",
    (g) => {
      g.values["x"] = { dtype: "f32", shape: ["T", 4] };
    },
  );
  assertRejects(
    "orphan values declaration",
    "graph.values['ghost']: declaration not defined by any node",
    (g) => {
      g.values["ghost"] = { dtype: "f32", shape: [1] };
    },
  );
  assertRejects(
    "symbolic dimension in an initializer",
    "symbolic dimensions cannot be used in an initializer shape",
    (g) => {
      g.values["w"] = { dtype: "f32", shape: ["T", 3] };
    },
  );
  assertRejects(
    "initializer with semantic dtype bool",
    "initializer semantic dtype 'bool' is outside the vocabulary",
    (g) => {
      g.values["w"] = { dtype: "bool", shape: [4, 3] };
    },
  );
});

/**
 * The only thing an IR v2 initializer declaration may contain is `shared` (the mapping to the data
 * is held by the container's binding table, keyed by `(graph name, initializer name)` —
 * docs/ir-v2.md). Writing the old distribution's `tensor` / `storage` as is fails as an unknown
 * key.
 */
test("parseIrDeclaration: the key set of an initializer declaration is `shared` only", () => {
  // Only two forms: an empty object for a declaration with data, `{ shared: true }` for a borrowed
  // one.
  assertEquals(parseMutated(() => {}).initializers["w"], { shared: false });
  const borrowed = parseMutated((g) => {
    g.initializers["w"] = { shared: true };
  });
  assertEquals(borrowed.initializers["w"], { shared: true });

  for (const key of ["tensor", "storage", "scale", "group_size"]) {
    assertThrows(
      () =>
        parseMutated((g) => {
          Object.assign(g.initializers["w"], { [key]: "enc.w" });
        }),
      IrError,
      `unknown key '${key}'`,
    );
  }
  // MUST: `false` is the same declaration as an absent field, so it cannot be spelled (canonical
  // serialization would not be unique).
  assertThrows(
    () =>
      parseMutated((g) => {
        Object.assign(g.initializers["w"], { shared: false });
      }),
    IrError,
    "write true only when it is borrowed",
  );
});

/**
 * The intersection of semantic dtype and storage (codec) (i32 is raw int32 only — ADR 0010) became
 * the job of the **merge layer** (`allowedLayouts` in `format/container/bind.ts`). This layer,
 * which looks only at the declaration, accepts a `values` semantic dtype of i32 or f32 alike.
 */
test("parseIrDeclaration: accepts an initializer with semantic i32 as a declaration", () => {
  const graph = parseMutated((g) => {
    g.values["w"] = { dtype: "i32", shape: [4, 3] };
  });
  assertEquals(graph.values["w"].dtype, "i32");
  assertEquals(graph.initializers["w"], { shared: false });
});

test("parseIrDeclaration: dtype vocabulary", () => {
  assertRejects(
    "f16 as a semantic dtype",
    "graph.values['h'].dtype: semantic dtype 'f16' is outside the vocabulary",
    (g) => {
      g.values["h"].dtype = "f16";
    },
  );
  assertRejects(
    "i64 as a semantic dtype",
    "graph.inputs[0].dtype: semantic dtype 'i64' is outside the vocabulary",
    (g) => {
      g.inputs[0].dtype = "i64";
    },
  );
});
test("parseIrDeclaration: symbol declarations and bindability", () => {
  assertRejects(
    "use of an undeclared symbol",
    "symbol 'S' is not declared in graph.symbols",
    (g) => {
      g.inputs[0].shape = ["S", 4];
    },
  );
  assertRejects(
    "never appears at a dimension position of an input shape",
    "'T' does not appear at a dimension position of an input shape / states shape",
    (g) => {
      g.inputs[0].shape = [8, 4];
    },
  );
  assertRejects("duplicate symbols", "graph.symbols: 'T' is duplicated", (g) => {
    g.symbols = ["T", "T"];
  });
  assertRejects("invalid symbol name", "invalid symbol name '2T'", (g) => {
    g.symbols = ["T", "2T"];
  });
  // An input with only a derived form is **accepted** (the solution from the actual size is
  // unique — ADR 0057). The vowel-detection CRNN can declare its length axis only as `2T` because
  // of the stride 2 of its first conv.
  const derived = parseMutated((g) => {
    g.inputs[0].shape = ["2T", 4];
  });
  assertEquals(derived.inputs[0].shape, ["2T", 4]);
});

test("parseIrDeclaration: shape elements", () => {
  assertRejects(
    "non-canonical dimension expression",
    "dimension expression '1T' does not match the canonical grammar",
    (g) => {
      g.inputs[0].shape = ["1T", 4];
    },
  );
  assertRejects("negative dimension", "dimension -3 is not a non-negative integer", (g) => {
    g.values["b"].shape = [-3];
  });
  assertRejects("fractional dimension", "dimension 1.5 is not a non-negative integer", (g) => {
    g.values["b"].shape = [1.5];
  });
  const boolDim = JSON.stringify(baseDeclaration()).replace('"shape":[3]', '"shape":[true]');
  assertThrows(() => parseIrDeclaration(boolDim), IrError, "neither a number nor a string");
});

test("parseIrDeclarationValue: rejects holes in host arrays with their path", () => {
  // JSON.parse does not create holes, but declarations built by the host (new Array(n) or delete)
  // can. map / some / reduce skip holes, so a declaration with a missing dimension would slip
  // through the checks and become an undefined dimension further on.
  const holeAt = (index: number, length: number): number[] => {
    const shape = new Array<number>(length);
    for (let i = 0; i < length; i += 1) if (i !== index) shape[i] = 4;
    return shape;
  };
  for (const [hint, index] of [
    ["first", 0],
    ["middle", 1],
    ["last", 2],
  ] as const) {
    const graph = baseDeclaration();
    graph.values["w"].shape = holeAt(index, 3);
    assertThrows(
      () => parseIrDeclarationValue(graph),
      IrError,
      `graph.values['w'].shape[${index}] is missing (a hole in a sparse array)`,
      hint,
    );
  }
  const graph = baseDeclaration();
  const ins = new Array<string>(3);
  ins[0] = "x";
  ins[2] = "w";
  graph.nodes[0].ins = ins;
  assertThrows(
    () => parseIrDeclarationValue(graph),
    IrError,
    "graph.nodes[0].ins[1] is missing (a hole in a sparse array)",
  );
});

test("parseIrDeclaration: node structure", () => {
  assertRejects("attrs is not an object", "graph.nodes[0].attrs: is not an object", (g) => {
    g.nodes[0].attrs = 5;
  });
  assertRejects("op is an empty string", "graph.nodes[0].op: is not a non-empty string", (g) => {
    g.nodes[0].op = "";
  });
  assertRejects(
    "an outs element is an empty string",
    "graph.nodes[0].outs[0]: is not a non-empty string",
    (g) => {
      g.nodes[0].outs = [""];
    },
  );
});

// ADR 0067 decision 5: the number of `outs` is given meaning by the **contract layer** (zero is
// allowed only for ops whose contract declares an effect). The parser does not look at the count —
// what this checks is that very "not looking", and `outs: []` on a non-effect op fails through
// **another rule** (the output declaration becomes an orphan).
test("parseIrDeclaration: the number of outs is not the parser's job", () => {
  // Removing the output declaration too lets it through the parser (the count is enforced and
  // pinned by tests/ops_conformance_test.ts)
  const graph = parseMutated((g) => {
    g.nodes[1].outs = [];
    delete g.values["y"];
    g.outputs = ["h"];
  });
  assertEquals(graph.nodes[1].outs, []);
  // Keeping only the declaration fails as an "orphan values declaration" (not as a wrong output
  // count)
  assertThrows(
    () =>
      parseMutated((g) => {
        g.nodes[1].outs = [];
        g.outputs = ["h"];
      }),
    IrError,
    "declaration not defined by any node",
  );
});

// A graph with "__proto__" as a name. The fixture MUST be built from a raw JSON string — writing
// `"__proto__":` in a JS object literal sets [[Prototype]] instead of an own key, which differs
// from the own property JSON.parse creates, and the test would miss its target.
//
// The initializer name is itself the key to the data, so no renaming happens, but the merge
// (`mergedGraph`) **repacks the initializers into a new container** after storage is settled, so
// **both** the declaration-side container (the parser) and the merged container must have a null
// prototype, or the own property silently disappears.
const protoNameGraph = `{
  "format": "karume-ir",
  "version": 2,
  "requires": { "ops": ["matmul"] },
  "symbols": ["T"],
  "inputs": [{ "name": "x", "dtype": "f32", "shape": ["T", 4] }],
  "outputs": ["y"],
  "initializers": { "__proto__": {} },
  "values": {
    "__proto__": { "dtype": "f32", "shape": [4, 3] },
    "y": { "dtype": "f32", "shape": ["T", 3] }
  },
  "nodes": [{ "op": "matmul", "ins": ["x", "__proto__"], "outs": ["y"], "attrs": {} }]
}`;

test("parseIrDeclaration: keeps a declaration named '__proto__' as an own property", () => {
  const graph = parseIrDeclaration(protoNameGraph);

  // Some engines create an own property even with `o["__proto__"] = v`, so hasOwn alone cannot
  // detect a regression to a plain `{}`. It also pins that the container has a null prototype
  // (= no engine differences enter the accepted set).
  assertEquals(Object.getPrototypeOf(graph.values), null);
  assertEquals(Object.getPrototypeOf(graph.initializers), null);
  assertEquals(Object.hasOwn(graph.values, "__proto__"), true);
  assertEquals(Object.hasOwn(graph.initializers, "__proto__"), true);
  assertEquals(graph.values["__proto__"], { dtype: "f32", shape: [4, 3] });
  assertEquals(graph.initializers["__proto__"], { shared: false });
  // It did not just slip past the declaration check; it passes including the references from nodes.
  assertEquals(graph.nodes[0].ins, ["x", "__proto__"]);
});

test("parseIrDeclaration: an orphan declaration named '__proto__' is rejected, not silently dropped", () => {
  const orphan = `{
    "format": "karume-ir",
    "version": 2,
    "requires": { "ops": ["matmul"] },
    "symbols": ["T"],
    "inputs": [{ "name": "x", "dtype": "f32", "shape": ["T", 4] }],
    "outputs": ["y"],
    "initializers": { "w": {} },
    "values": {
      "w": { "dtype": "f32", "shape": [4, 3] },
      "y": { "dtype": "f32", "shape": ["T", 3] },
      "__proto__": { "dtype": "f32", "shape": [1] }
    },
    "nodes": [{ "op": "matmul", "ins": ["x", "w"], "outs": ["y"], "attrs": {} }]
  }`;

  assertThrows(() => parseIrDeclaration(orphan), IrError, "declaration not defined by any node");
});

// ADR 0066 decision 2: named slots that hold the mutable state of one generation. Now that the
// referencing fields exist (ADR 0067's states field / state_append), **a graph with declarations
// only fails reference completeness** — tests that look at the declaration itself set up the
// referencing side with withStateReaders.
test("parseIrDeclaration: a graph without a states section has an empty slot set", () => {
  const graph = parseIrDeclaration(JSON.stringify(baseDeclaration()));

  assertEquals(Object.keys(graph.states), []);
  // A container whose names come from the parse input has a null prototype (the same MUST as
  // values / initializers).
  assertEquals(Object.getPrototypeOf(graph.states), null);
});

test("parseIrDeclaration: accepts state slot declarations", () => {
  const empty = parseMutated((g) => {
    g.states = {};
  });
  assertEquals(Object.keys(empty.states), []);

  // A mix of slots with different capacities (sliding layers / full layers are separate slots —
  // no assumption of layers × uniform KV).
  const slots = parseMutated((g) => {
    g.states = {
      "layer0.k": { dtype: "f32", shape: [1, 2, 512, 128] },
      "layer0.v": { dtype: "f32", shape: [1, 2, 512, 128] },
      "layer1.k": { dtype: "f32", shape: [1, 2, 131072, 128] },
    };
    withStateReaders(g);
  });
  assertEquals(Object.keys(slots.states), ["layer0.k", "layer0.v", "layer1.k"]);
  assertEquals(slots.states["layer0.k"], {
    dtype: "f32",
    shape: [1, 2, 512, 128],
    external: false,
  });
  assertEquals(slots.states["layer1.k"].shape, [1, 2, 131072, 128]);

  // Symbolic dimensions are resolved by the Session's symbols (the symbol also appears in an input,
  // so the binding point is an input shape's dimension position as before — states-only symbols
  // have their own test).
  const symbolic = parseMutated((g) => {
    g.states = { cache: { dtype: "f32", shape: ["T", 4] } };
    withStateReaders(g);
  });
  assertEquals(symbolic.states["cache"].shape, ["T", 4]);

  // Both ends of rank 1..4 (a generic slot with a single capacity axis is valid too).
  const ranks = parseMutated((g) => {
    g.states = {
      flat: { dtype: "f32", shape: [8] },
      full: { dtype: "f32", shape: [1, 2, 3, 4] },
    };
    withStateReaders(g);
  });
  assertEquals(ranks.states["flat"].shape, [8]);
  assertEquals(ranks.states["full"].shape, [1, 2, 3, 4]);
});

/**
 * Replaces only the states section and checks the rejection. **The diagnostic text MUST be
 * checked too** — if the section itself regressed to unsupported, it would fail with "unknown key
 * 'states'", so a check on `IrError` alone would not prove that the rule works.
 */
const assertStatesReject = (states: unknown, includes: string): void => {
  assertThrows(
    () =>
      parseMutated((graph) => {
        (graph as { states: unknown }).states = states;
      }),
    IrError,
    includes,
  );
};

test("parseIrDeclaration: state slot dtype is f32 only", () => {
  // f16 only has its place reserved (ADR 0066 addendum 5) — it fails as "not supported", not as
  // "outside the vocabulary".
  assertStatesReject({ cache: { dtype: "f16", shape: [4] } }, "is not supported");
  for (const dtype of ["i32", "bool", "bf16", "i8"]) {
    assertStatesReject({ cache: { dtype, shape: [4] } }, "is outside the vocabulary (f32)");
  }
});

test("parseIrDeclaration: a state slot shape is a concrete shape including capacity", () => {
  assertStatesReject({ cache: { dtype: "f32", shape: [1, 1, 1, 1, 1] } }, "rank 5 is outside 1..4");
  assertStatesReject({ cache: { dtype: "f32", shape: [] } }, "rank 0 is outside 1..4");
  assertStatesReject({ cache: { dtype: "f32", shape: [1, 0, 4] } }, "is not a positive integer");
  assertStatesReject({ cache: { dtype: "f32", shape: [-4] } }, "is not a non-negative integer");
  assertStatesReject({ cache: { dtype: "f32", shape: ["S", 4] } }, "is not declared");
  assertStatesReject({ cache: { dtype: "f32", shape: ["1T", 4] } }, "canonical grammar");
});

/**
 * Injects exactly one dimension as **raw JSON text** (`JSON.stringify` folds `1.0` into `1`, so
 * accepting integral floats cannot be observed through objects).
 */
const parseRawDim = (
  raw: string,
  mutate: (graph: DeclarationJson, dim: string) => void,
): IrDeclaration => {
  const graph = baseDeclaration();
  mutate(graph, "__DIM__");
  return parseIrDeclaration(JSON.stringify(graph).replaceAll('"__DIM__"', raw));
};

// JSON `4.0` / `4e0` become a single number in JSON.parse, so they pass as integers. On the Python
// side (_parse_shape in tools/exporter/src/karume/verify.py) int and float are separate types, so
// unless this form is pinned as **the same accepted set on both sides** it would drift into "the
// runtime can read it but the exporter's validation fails" (the mirror is
// TestIntegralFloatDimensions in tools/exporter/tests/test_verify.py).
test("parseIrDeclaration: accepts integral float dimensions in JSON and rejects non-integers", () => {
  for (const raw of ["4.0", "4e0", "0.4e1"]) {
    const values = parseRawDim(raw, (g, dim) => {
      g.values["w"].shape = [dim];
    });
    assertEquals(values.values["w"].shape, [4], raw);
    const states = parseRawDim(raw, (g, dim) => {
      g.states = { cache: { dtype: "f32", shape: [1, dim] } };
      withStateReaders(g);
    });
    assertEquals(states.states["cache"].shape, [1, 4], raw);
  }
  for (const raw of ["1.5", "15e-1"]) {
    assertThrows(
      () =>
        parseRawDim(raw, (g, dim) => {
          g.values["w"].shape = [dim];
        }),
      IrError,
      "is not a non-negative integer",
    );
    assertThrows(
      () =>
        parseRawDim(raw, (g, dim) => {
          g.states = { cache: { dtype: "f32", shape: [1, dim] } };
          withStateReaders(g);
        }),
      IrError,
      "is not a non-negative integer",
    );
  }
  // Exceeding the safe range fails at the same point as on the Python side (> 2^53−1).
  assertThrows(
    () =>
      parseRawDim("1e300", (g, dim) => {
        g.values["w"].shape = [dim];
      }),
    IrError,
    "is not a non-negative integer",
  );
});

/**
 * Empty value / initializer names are rejected for the same reason as states slot names (the
 * referencing fields — inputs[].name / ins / outs / outputs — all accept only non-empty strings, so
 * letting them through would make a declaration that "can be declared but can never be
 * referenced").
 *
 * MUST: an empty-named value **alone** is blocked by the orphan-declaration check, but written **as
 * a pair** with an empty-named initializer it enters the defined set and satisfies both the
 * orphan-declaration check and the reverse-reference check — one unreferenceable real tensor would
 * sit in the distribution. That pair is the only thing this check detects.
 */
test("parseIrDeclaration: rejects empty value / initializer names", () => {
  assertThrows(
    () =>
      parseMutated((g) => {
        g.values[""] = { dtype: "f32", shape: [4, 3] };
        g.initializers[""] = {};
      }),
    IrError,
    "value name in graph.values: is not a non-empty string",
  );
  // The form where only the initializer side is empty fails too (the name check applies to both
  // sides of the declaration)
  assertThrows(
    () =>
      parseMutated((g) => {
        g.initializers[""] = {};
      }),
    IrError,
    "initializer name in graph.initializers: is not a non-empty string",
  );
  // Counterpart: with a non-empty name the same pair is accepted (= only the empty name is
  // rejected, and both declaration-completeness checks pass this pair — which is exactly why this
  // check is needed)
  const graph = parseMutated((g) => {
    g.values["ghost"] = { dtype: "f32", shape: [4, 3] };
    g.initializers["ghost"] = {};
  });
  assertEquals(graph.initializers["ghost"], { shared: false });
});

test("parseIrDeclaration: states structure", () => {
  assertStatesReject([], "graph.states: is not an object");
  assertStatesReject({ cache: 4 }, "graph.states['cache']: is not an object");
  assertStatesReject({ cache: { dtype: "f32", shape: 4 } }, "shape: is not an array");
  assertStatesReject({ cache: { shape: [4] } }, "required key 'dtype' is missing");
  assertStatesReject({ cache: { dtype: "f32", shape: [4], window: 512 } }, "unknown key 'window'");
  // An empty slot name is not accepted by the referencing fields (ADR 0067) = a declaration that
  // cannot be referenced.
  assertStatesReject(
    { "": { dtype: "f32", shape: [4] } },
    "slot name in graph.states: is not a non-empty string",
  );
});

// Slot names are a separate namespace from value names (not referenced by ins / outs), but a shared
// name would only make "a value name written in a slot-name field / or the reverse" undetectable,
// so it is rejected.
test("parseIrDeclaration: a state slot name cannot collide with a value name", () => {
  // All four roles: input / initializer / intermediate value / graph output.
  for (const name of ["x", "w", "h", "y"]) {
    assertStatesReject(
      { [name]: { dtype: "f32", shape: [4] } },
      `graph.states['${name}']: same name as a value`,
    );
  }
});

test("parseIrDeclaration: keeps a state slot named '__proto__' as an own property", () => {
  const graph = parseIrDeclaration(`{
    "format": "karume-ir",
    "version": 2,
    "requires": { "ops": ["matmul", "state_append"] },
    "symbols": ["T"],
    "inputs": [{ "name": "x", "dtype": "f32", "shape": ["T", 4] }],
    "outputs": ["y"],
    "initializers": { "w": {} },
    "values": {
      "w": { "dtype": "f32", "shape": [4, 3] },
      "y": { "dtype": "f32", "shape": ["T", 3] }
    },
    "states": { "__proto__": { "dtype": "f32", "shape": [4] } },
    "nodes": [
      { "op": "matmul", "ins": ["x", "w"], "outs": ["y"], "attrs": {} },
      {
        "op": "state_append",
        "ins": ["x"],
        "outs": [],
        "attrs": {},
        "states": { "slot": "__proto__" }
      }
    ]
  }`);

  assertEquals(Object.getPrototypeOf(graph.states), null);
  assertEquals(Object.hasOwn(graph.states, "__proto__"), true);
  assertEquals(graph.states["__proto__"], { dtype: "f32", shape: [4], external: false });
  // The referencing fields have a null prototype for the same reason (with a plain `{}` the
  // assignment would turn into setting [[Prototype]], the field would silently vanish, and a form
  // that slips past both the reference-completeness check and the contract's key-set check could be
  // built).
  const referring = graph.nodes[1].states;
  assertEquals(Object.getPrototypeOf(referring), null);
  assertEquals(referring["slot"], "__proto__");
});

/**
 * The smallest form with one layer that reads state (states-form attention) and one that writes it
 * (`state_append`) (ADR 0067 decisions 4 / 5; emission order is decision 5b's "readers → append").
 * This is a **parser-layer** test, so shapes and contracts cannot be checked — what is pinned here
 * is only the field structure and reference completeness.
 */
const statefulGraph = (): DeclarationJson => {
  const graph = baseDeclaration();
  graph.states = {
    "kv.k": { dtype: "f32", shape: [1, 2, 512, 8] },
    "kv.v": { dtype: "f32", shape: [1, 2, 512, 8] },
  };
  graph.requires.ops.push("attention", "state_append");
  graph.values["att"] = { dtype: "f32", shape: ["T", 3] };
  graph.nodes.push(
    {
      op: "attention",
      ins: ["h", "h", "h"],
      outs: ["att"],
      attrs: { scale: 0.5, window: 512 },
      states: { k: "kv.k", v: "kv.v" },
    },
    { op: "state_append", ins: ["h"], outs: [], attrs: { window: 512 }, states: { slot: "kv.k" } },
    { op: "state_append", ins: ["h"], outs: [], attrs: { window: 512 }, states: { slot: "kv.v" } },
  );
  return graph;
};

const parseStateful = (mutate: (graph: DeclarationJson) => void = () => {}): IrDeclaration => {
  const graph = statefulGraph();
  mutate(graph);
  return parseIrDeclaration(JSON.stringify(graph));
};

// ADR 0067 decision 4: a node refers to slots by name in a field **separate** from `ins` / `outs`.
test("parseIrDeclaration: reads a node's states field", () => {
  const graph = parseStateful();

  assertEquals(
    graph.nodes.map((node) => node.op),
    ["matmul", "add", "attention", "state_append", "state_append"],
  );
  // A node without the field gets an empty table (an always-present field — consumers need not
  // write `?? {}`)
  assertEquals(Object.keys(graph.nodes[0].states), []);
  assertEquals(Object.entries(graph.nodes[2].states), [
    ["k", "kv.k"],
    ["v", "kv.v"],
  ]);
  assertEquals(graph.nodes[3].states["slot"], "kv.k");
  // A zero-output node passes the parser (the count is enforced in the contract layer — ADR 0067
  // decision 5)
  assertEquals(graph.nodes[3].outs, []);
});

test("parseIrDeclaration: the states field can refer only to declared slots", () => {
  assertThrows(
    () =>
      parseStateful((g) => {
        g.nodes[3].states = { slot: "kv.missing" };
      }),
    IrError,
    "state slot 'kv.missing' is not declared in graph.states",
  );
  // The mix-up of writing a value name (the slot namespace and the value namespace are separate —
  // value names cannot be referenced)
  assertThrows(
    () =>
      parseStateful((g) => {
        g.nodes[3].states = { slot: "h" };
      }),
    IrError,
    "is not declared in graph.states",
  );
});

test("parseIrDeclaration: rejects a state slot that nobody references", () => {
  assertThrows(
    () =>
      parseStateful((g) => {
        g.states!["kv.orphan"] = { dtype: "f32", shape: [1, 2, 512, 8] };
      }),
    IrError,
    "graph.states['kv.orphan']: declaration not referenced by any node",
  );
  // A single reference is enough (layers with only a reader or only a writer both exist — KV-shared
  // layers have no append)
  const readerOnly = parseStateful((g) => {
    g.nodes.splice(3, 2);
    g.requires.ops = ["matmul", "add", "attention"];
  });
  assertEquals(Object.keys(readerOnly.states), ["kv.k", "kv.v"]);
});

test("parseIrDeclaration: states field structure", () => {
  const reject = (states: unknown, includes: string): void => {
    assertThrows(
      () =>
        parseStateful((g) => {
          (g.nodes[3] as { states: unknown }).states = states;
        }),
      IrError,
      includes,
    );
  };
  reject([], "graph.nodes[3].states: is not an object");
  reject({ slot: 4 }, "graph.nodes[3].states['slot']: is not a non-empty string");
  reject({ slot: "" }, "graph.nodes[3].states['slot']: is not a non-empty string");
  reject({ "": "kv.k" }, "graph.nodes[3].states key: is not a non-empty string");
});

// ADR 0066 addendum 7: there are two binding points (input shapes / states shapes), but **their
// reach is asymmetric**.
test("parseIrDeclaration: states-only symbols can be declared but do not appear in value shapes", () => {
  // The capacity symbol C appears nowhere in the inputs (it is decided when the context is
  // created) — it can still be declared
  const graph = parseStateful((g) => {
    g.symbols.push("C");
    g.states!["kv.k"].shape = [1, 2, "C", 8];
    g.states!["kv.v"].shape = [1, 2, "C", 8];
  });
  assertEquals(graph.states["kv.k"].shape, [1, 2, "C", 8]);

  // Appearing in a value shape fails loudly (only bindings from inputs resolve ordinary values, so
  // it would always be unbindable at run time — it fails at declaration time)
  assertThrows(
    () =>
      parseStateful((g) => {
        g.symbols.push("C");
        g.states!["kv.k"].shape = [1, 2, "C", 8];
        g.states!["kv.v"].shape = [1, 2, "C", 8];
        g.values["att"].shape = ["C", 3];
      }),
    IrError,
    "states-only symbol 'C' appears in a value shape",
  );

  // A symbol that appears in neither inputs nor states cannot be bound, as before
  assertThrows(
    () =>
      parseStateful((g) => {
        g.symbols.push("C");
      }),
    IrError,
    "'C' does not appear at a dimension position of an input shape / states shape",
  );

  // A symbol that appears in **both** inputs and states is not "states-only" (it may appear in
  // value shapes)
  const shared = parseStateful((g) => {
    g.states!["kv.k"].shape = [1, 2, "T", 8];
    g.states!["kv.v"].shape = [1, 2, "T", 8];
  });
  assertEquals(shared.values["att"].shape, ["T", 3]);
});

test("parseIrDeclaration: the JSON accepted set", () => {
  assertThrows(() => parseIrDeclaration("{"), IrError, "Cannot parse");
  assertThrows(
    () => parseIrDeclaration('{"format":"karume-ir","version":NaN}'),
    IrError,
    "Cannot parse",
  );
  // 1e999 is valid syntax but rounds to Infinity — it is rejected under the contract that aligns
  // the accepted set with the browser's JSON.parse.
  const infinite = JSON.stringify(baseDeclaration()).replace('"shape":[4,3]', '"shape":[4,1e999]');
  assertThrows(() => parseIrDeclaration(infinite), IrError, "finite");
});
