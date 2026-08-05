import { assertEquals, assertThrows } from "@std/assert";
import { ManifestReferenceError, parseManifest, resolveFiles } from "../mod.ts";

const manifest = parseManifest(
  await Deno.readTextFile(new URL("./fixtures/manifest-fetch.json", import.meta.url)),
);

Deno.test("resolveFiles: preset 省略で defaultPreset の組を返す", () => {
  const files = resolveFiles(manifest);
  assertEquals(Object.keys(files), [
    "text_encoder",
    "text_conditioner",
    "transformer",
    "transformer.rope_base",
    "rope_alias",
    "vae_decoder",
    "tokenizer",
  ]);
  // defaultPreset = w8a8-s16（transformer は i8）。
  assertEquals(files["transformer"].path, "transformer/model.i8.safetensors");
});

Deno.test("resolveFiles: preset 指定で variant の選択が切り替わる", () => {
  const files = resolveFiles(manifest, "f16");
  assertEquals(files["transformer"].path, "transformer/model.f16.safetensors");
  // extras は variant 側にぶら下がるが、この manifest では f16 / i8 で同じ実体を指す。
  assertEquals(files["transformer.rope_base"].path, "transformer/rope_base.safetensors");
});

Deno.test("resolveFiles: 同一 path を指すキーは落とさず、同じ 3 点セットを返す", () => {
  const files = resolveFiles(manifest);
  assertEquals(files["rope_alias"], files["transformer.rope_base"]);
  const paths = Object.values(files).map((ref) => ref.path);
  assertEquals(new Set(paths).size, 6, "7 キー / 6 パス（取得と進捗は path で一意化される）");
});

Deno.test("resolveFiles: 実在しない preset は利用可能一覧つきで拒否する", () => {
  const error = assertThrows(() => resolveFiles(manifest, "q4"), ManifestReferenceError);
  assertEquals(error.available.presets, ["f16", "w8a8-s16", "f16-c16"]);
  assertEquals(error.available.variants, { transformer: ["f16", "i8"] });
});
