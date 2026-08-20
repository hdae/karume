// `resolveFiles` の 2 軸（model / quant）。取得層は通さず、manifest から取得キー表を作るところ
// だけを見る。
//
// ここで押さえるのは 4 つ:
//  ① 省略時は `defaultModel` / `defaultQuant` に落ちる（v2 で model 軸が増えた）。
//  ② weights は選んだ dtype、assets は quant に依らず常に同じ実体が入る。
//  ③ 未知の model / quant は**利用可能な一覧**を添えて落ちる（ADR 0041 §8）。
//  ④ 複数 shard は宣言順のまま `<weights>[i]` へ展開される（v3 の shards 欄）。

import { assertEquals, assertThrows } from "@std/assert";
import { ManifestReferenceError, parseManifest, resolveFiles } from "../mod.ts";

const manifest = parseManifest(
  await Deno.readTextFile(new URL("./fixtures/manifest-fetch.json", import.meta.url)),
);

Deno.test("resolveFiles: 省略時は defaultModel / defaultQuant の組を返す", () => {
  const files = resolveFiles(manifest);
  // weights（宣言順）→ assets（宣言順）の順に並ぶ。
  assertEquals(Object.keys(files), [
    "text_encoder",
    "text_conditioner",
    "transformer",
    "transformer.rope_base",
    "vae_decoder",
    "tokenizer",
    "rope_alias",
  ]);
  // defaultQuant = w8a8-s16（transformer は i8）。
  assertEquals(files["transformer"].path, "transformer/model.i8.safetensors");
});

Deno.test("resolveFiles: quant 指定で dtype の選択が切り替わる", () => {
  const files = resolveFiles(manifest, { quant: "f16" });
  assertEquals(files["transformer"].path, "transformer/model.f16.safetensors");
  // extras は dtype 側にぶら下がるが、この manifest では f16 / i8 で同じ実体を指す。
  assertEquals(files["transformer.rope_base"].path, "transformer/rope_base.safetensors");
});

Deno.test("resolveFiles: assets は quant を切り替えても動かない", () => {
  const defaults = resolveFiles(manifest);
  const f16 = resolveFiles(manifest, { quant: "f16" });
  assertEquals(f16["tokenizer"], defaults["tokenizer"]);
  assertEquals(f16["rope_alias"], defaults["rope_alias"]);
});

Deno.test("resolveFiles: model 指定でそのモデルの表に切り替わる", () => {
  const files = resolveFiles(manifest, { model: "anima-lite" });
  assertEquals(Object.keys(files), [
    "text_encoder",
    "transformer",
    "transformer.rope_base",
    "tokenizer",
  ]);
  // 共有 path はモデルを跨いでも同じ 3 点セット（ADR 0041 §5 の「path の一致で共有」）。
  assertEquals(files["text_encoder"], resolveFiles(manifest)["text_encoder"]);
});

Deno.test("resolveFiles: 同一 path を指すキーは落とさず、同じ 3 点セットを返す", () => {
  const files = resolveFiles(manifest);
  assertEquals(files["rope_alias"], files["transformer.rope_base"]);
  const paths = Object.values(files).map((ref) => ref.path);
  assertEquals(new Set(paths).size, 6, "7 キー / 6 パス（取得と進捗は path で一意化される）");
});

Deno.test("resolveFiles: 複数 shard は宣言順のまま別々の取得キーへ展開する", () => {
  const shard = (name: string, size: number, mark: string) => ({
    path: `net/${name}`,
    size,
    sha256: mark.repeat(32),
  });
  const graph = shard("graph.safetensors", 6, "a1");
  const first = shard("weights-0.safetensors", 8, "b2");
  const second = shard("weights-1.safetensors", 4, "c3");
  const sharded = parseManifest(JSON.stringify({
    format: "karume/3",
    generator: "karume/0.1.0",
    defaultModel: "m",
    models: {
      m: {
        pipeline: "anima/1",
        weights: {
          net: {
            f16: {
              shards: [graph, first, second],
              extras: { rope_base: shard("rope_base.safetensors", 6, "d4") },
            },
          },
        },
        assets: {},
        quants: { q: { weights: { net: "f16" }, session: {} } },
        defaultQuant: "q",
        pipelineConfig: {},
      },
    },
  }));
  const files = resolveFiles(sharded);
  // shard は `[i]`、extras は `.` — 名前空間が交わらないので取り違えが起きない。
  assertEquals(Object.keys(files), ["net[0]", "net[1]", "net[2]", "net.rope_base"]);
  assertEquals([files["net[0]"], files["net[1]"], files["net[2]"]], [graph, first, second]);
});

Deno.test("resolveFiles: 実在しない model は利用可能一覧つきで拒否する", () => {
  const error = assertThrows(
    () => resolveFiles(manifest, { model: "anima-xl" }),
    ManifestReferenceError,
  );
  assertEquals(error.available.models, ["anima-turbo", "anima-lite"]);
});

Deno.test("resolveFiles: 実在しない quant は利用可能一覧つきで拒否する", () => {
  const error = assertThrows(
    () => resolveFiles(manifest, { quant: "q4" }),
    ManifestReferenceError,
  );
  assertEquals(error.available.quants, ["f16", "w8a8-s16", "f16-c16"]);
  assertEquals(error.available.dtypes, {
    text_encoder: ["f16"],
    text_conditioner: ["f16"],
    transformer: ["f16", "i8"],
    vae_decoder: ["f16"],
  });
});

Deno.test("resolveFiles: quant の一覧は指定したモデルのものになる", () => {
  const error = assertThrows(
    () => resolveFiles(manifest, { model: "anima-lite", quant: "w8a8-s16" }),
    ManifestReferenceError,
  );
  // 別モデル（anima-turbo）にしか無い quant を勧めない。
  assertEquals(error.available.quants, ["w8"]);
});
