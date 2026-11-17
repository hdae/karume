// The name map's interpreter on small synthetic maps: each rule kind, the order of the rules, the
// routing between components, and each refusal. The Anima map itself is exercised on the upstream
// files in `packages/e2e/tests/e2e_anima_single_file_test.ts`.

import { describe, it } from "vite-plus/test";
import { assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import {
  ConvertError,
  NAME_MAP_FORMAT,
  type NameMap,
  type NameMapComponent,
  type NameMapLayout,
  type NameMapRule,
  placeTensors,
  type TensorPlacement,
} from "../mod.ts";

const component = (
  role: string,
  fields: Partial<Omit<NameMapComponent, "role">> = {},
): NameMapComponent => ({
  role,
  source: "the test",
  route: "",
  rules: [],
  initializerPrefix: "",
  ...fields,
});

const layout = (
  name: string,
  components: readonly NameMapComponent[],
  fields: Partial<Omit<NameMapLayout, "name" | "components">> = {},
): NameMapLayout => ({
  name,
  source: "the test",
  stripPrefix: [],
  markers: [],
  components,
  ...fields,
});

const mapOf = (...layouts: readonly NameMapLayout[]): NameMap => ({
  format: NAME_MAP_FORMAT,
  layouts,
});

/** One layout of one component that applies `rules` to every key. */
const rulesMap = (rules: readonly NameMapRule[]): NameMap =>
  mapOf(layout("only", [component("c", { rules })]));

/** The placement as plain data: per role, `[initializer, key]` pairs in file order. */
const plain = (placement: TensorPlacement): unknown => ({
  layout: placement.layout,
  components: Object.fromEntries(
    [...placement.components].map(([role, names]) => [role, [...names]]),
  ),
  excluded: placement.excluded,
  unplaced: placement.unplaced,
});

const refusal = (map: NameMap, keys: readonly string[]): string => {
  const error = assertThrows(() => placeTensors(map, keys), ConvertError);
  return error.message;
};

describe("placing a file's tensors with a name map", () => {
  it("removes the prefix every key carries, whichever of the layout's prefixes it is, and puts the initializer prefix before the result", () => {
    const map = mapOf(
      layout("single-file", [component("transformer", { initializerPrefix: "model." })], {
        stripPrefix: ["model.diffusion_model.", "net."],
        markers: ["a.weight"],
      }),
    );
    for (const prefix of ["net.", "model.diffusion_model."]) {
      assertEquals(plain(placeTensors(map, [`${prefix}a.weight`, `${prefix}b.weight`])), {
        layout: "single-file",
        components: {
          transformer: [
            ["model.a.weight", `${prefix}a.weight`],
            ["model.b.weight", `${prefix}b.weight`],
          ],
        },
        excluded: [],
        unplaced: [],
      });
    }
  });

  it("routes a key to the component whose route is the longest prefix it starts with, and lists the keys no route takes", () => {
    const map = mapOf(
      layout(
        "two",
        [
          component("transformer", { route: "" }),
          component("text_conditioner", {
            route: "llm_adapter.",
            rules: [{ stripPrefix: "llm_adapter." }],
          }),
        ],
        { markers: ["llm_adapter.embed.weight"] },
      ),
      layout(
        "routed",
        [component("transformer", { route: "blocks." }), component("head", { route: "final." })],
        { stripPrefix: ["x."] },
      ),
    );
    assertEquals(plain(placeTensors(map, ["llm_adapter.embed.weight", "blocks.0.weight"])), {
      layout: "two",
      components: {
        transformer: [["blocks.0.weight", "blocks.0.weight"]],
        text_conditioner: [["embed.weight", "llm_adapter.embed.weight"]],
      },
      excluded: [],
      unplaced: [],
    });
    assertEquals(plain(placeTensors(map, ["x.blocks.0.weight", "x.other.weight"])), {
      layout: "routed",
      components: { transformer: [["blocks.0.weight", "x.blocks.0.weight"]], head: [] },
      excluded: [],
      unplaced: ["x.other.weight"],
    });
  });

  it("removes a component's prefix only from a key that starts with it", () => {
    const placement = placeTensors(rulesMap([{ stripPrefix: "llm_adapter." }]), [
      "llm_adapter.a",
      "b.llm_adapter.c",
    ]);
    assertEquals(
      [...(placement.components.get("c") ?? [])],
      [
        ["a", "llm_adapter.a"],
        ["b.llm_adapter.c", "b.llm_adapter.c"],
      ],
    );
  });

  it("replaces every occurrence of a text, left to right, and takes the replacement literally", () => {
    const placement = placeTensors(rulesMap([{ replace: "aa", with: "$&b" }]), ["aaa.aa"]);
    assertEquals([...(placement.components.get("c")?.keys() ?? [])], ["$&ba.$&b"]);
  });

  it("applies the rules in their order: Anima's adaln_modulation_self_attn.1 is rewritten before self_attn, and the list reordered gives another name", () => {
    const adaln = { replace: "adaln_modulation_self_attn.1", with: "norm1.linear_1" };
    const selfAttn = { replace: "self_attn", with: "attn1" };
    const keys = ["blocks.0.adaln_modulation_self_attn.1.weight", "blocks.0.self_attn.q.weight"];
    assertEquals(
      [
        ...(placeTensors(rulesMap([adaln, selfAttn]), keys)
          .components.get("c")
          ?.keys() ?? []),
      ],
      ["blocks.0.norm1.linear_1.weight", "blocks.0.attn1.q.weight"],
    );
    assertEquals(
      [
        ...(placeTensors(rulesMap([selfAttn, adaln]), keys)
          .components.get("c")
          ?.keys() ?? []),
      ],
      ["blocks.0.adaln_modulation_attn1.1.weight", "blocks.0.attn1.q.weight"],
    );
  });

  it("excludes a key that contains an exclusion's text where the exclusion stands in the list", () => {
    const keys = ["net.pos_embedder.seq", "a.weight", "b._extra_state"];
    const after = placeTensors(
      rulesMap([{ replace: "pos_embedder", with: "pos" }, { exclude: "pos_embedder.seq" }]),
      keys,
    );
    assertEquals(after.excluded, []);
    const before = placeTensors(
      rulesMap([
        { exclude: "pos_embedder.seq" },
        { replace: "pos_embedder", with: "pos" },
        { exclude: "_extra_state" },
      ]),
      keys,
    );
    assertEquals(before.excluded, ["net.pos_embedder.seq", "b._extra_state"]);
    assertEquals([...(before.components.get("c")?.keys() ?? [])], ["a.weight"]);
  });

  it("tells layouts apart by their markers", () => {
    const map = mapOf(
      layout("transformer-file", [component("transformer")], { markers: ["patch_embed.w"] }),
      layout("conditioner-file", [component("text_conditioner")], { markers: ["embed.w"] }),
    );
    assertEquals(placeTensors(map, ["embed.w", "blocks.0.w"]).layout, "conditioner-file");
    assertEquals(placeTensors(map, ["patch_embed.w"]).layout, "transformer-file");
  });
});

describe("what the interpreter refuses", () => {
  it("a file no layout recognises, with every layout's reason", () => {
    const map = mapOf(
      layout("single-file", [component("t")], { stripPrefix: ["net.", "model."] }),
      layout("diffusers", [component("t")], { markers: ["patch_embed.w", "proj_out.w"] }),
    );
    const message = refusal(map, ["net.a", "model.b", "proj_out.w"]);
    assertStringIncludes(message, "no layout of the name map recognises the file (3 keys)");
    assertStringIncludes(
      message,
      "layout 'single-file': not every key starts with the same one of 'net.', 'model.'",
    );
    assertStringIncludes(message, "layout 'diffusers': the file has no key 'patch_embed.w'");
  });

  it("a file more than one layout recognises", () => {
    const map = mapOf(
      layout("a", [component("t")], { markers: ["w"] }),
      layout("b", [component("t")], { markers: ["w"] }),
    );
    assertStringIncludes(
      refusal(map, ["w"]),
      "more than one layout of the name map recognises the file: 'a', 'b'",
    );
  });

  it("two keys that map to one initializer and a key the routes leave between two components, all listed together", () => {
    const map = mapOf(
      layout("l", [
        component("transformer", { rules: [{ replace: "old.", with: "" }] }),
        component("first", { route: "dup." }),
        component("second", { route: "dup." }),
      ]),
    );
    const message = refusal(map, ["a", "old.a", "b", "old.b", "dup.x"]);
    assertStringIncludes(message, "the name map cannot place the file's tensors (layout 'l')");
    assertStringIncludes(
      message,
      "keys 'a' and 'old.a' both map to initializer 'a' of component 'transformer'",
    );
    assertStringIncludes(
      message,
      "keys 'b' and 'old.b' both map to initializer 'b' of component 'transformer'",
    );
    assertStringIncludes(
      message,
      "key 'dup.x' is routed to more than one component ('first', 'second')",
    );
  });

  it("a map of another format", () => {
    assertStringIncludes(
      refusal({ format: "karume-name-map/0", layouts: [] }, ["a"]),
      `the name map's format is "karume-name-map/0"`,
    );
  });
});
