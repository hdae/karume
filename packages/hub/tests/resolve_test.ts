// `resolveSelection` の 2 軸（model / quant）と、そこから平坦な取得列を導く `selectionRefs`。
// 取得層は通さず、manifest から選択結果を作るところだけを見る。
//
// ここで押さえるのは 6 つ:
//  ① 省略時は `defaultModel` / `defaultQuant` に落ち、実名が返る。
//  ② 選択は**構造型**（部品名 → 容器 / 資産名 → FileRef）で、取得キーの綴り規約を持たない。
//  ③ 並びは宣言順（`weights` の順・容器の中は part の添字順・最後に assets）。
//  ④ `selectionRefs` は**長さ 0 の part を落とし**、同一実体を `fileRefKey` で一意化する。
//  ⑤ `weights` の部分集合を渡すとその役割だけが選択に出る（assets は常に全数）。
//  ⑥ 未知の model / quant / weights は**利用可能な一覧**を添えて落ちる（ADR 0041 §8）。

import { assertEquals, assertNotStrictEquals, assertThrows } from "@std/assert";
import {
  type FileRef,
  ManifestReferenceError,
  parseManifest,
  resolveSelection,
  selectionRefs,
} from "../mod.ts";
import {
  fetchManifest as manifest,
  STYLE_VECTORS,
  TEXT_CONDITIONER_PARTS,
  TEXT_ENCODER_PARTS,
  TOKENIZER,
  TRANSFORMER_F16_PARTS,
  TRANSFORMER_I8_EMPTY,
  TRANSFORMER_I8_FETCHED,
  TRANSFORMER_I8_PARTS,
  VAE_DECODER_PARTS,
} from "./helpers/fixture.ts";

const paths = (refs: readonly FileRef[]): string[] => refs.map((ref) => ref.path);

Deno.test("resolveSelection: 省略時は defaultModel / defaultQuant の組を実名で返す", () => {
  const selection = resolveSelection(manifest);
  assertEquals(selection.model, "anima-turbo");
  assertEquals(selection.quant, "w8a8-s16");
  // 部品は weights の宣言順（取得キーの綴り規約は持たない — 席は部品名そのもの）。
  assertEquals(Object.keys(selection.containers), [
    "text_encoder",
    "text_conditioner",
    "transformer",
    "vae_decoder",
  ]);
  assertEquals(Object.keys(selection.assets), ["tokenizer", "style_vectors", "style_alias"]);
  // defaultQuant = w8a8-s16（transformer は i8 の容器）。
  assertEquals(paths(selection.containers["transformer"].parts), [...TRANSFORMER_I8_PARTS]);
});

Deno.test("resolveSelection: quant 指定で dtype の選択が切り替わる", () => {
  const selection = resolveSelection(manifest, { quant: "f16" });
  assertEquals(selection.quant, "f16");
  assertEquals(paths(selection.containers["transformer"].parts), [...TRANSFORMER_F16_PARTS]);
  // 切り替わるのは容器だけで、descriptor も一緒に付いてくる（2 文書の期待値は容器の持ち物）。
  // 期待値は fixture の実値で綴る — 同じ manifest から引き直すと、descriptor を落とす実装でも
  // 両辺が同時に動いて緑のままになる（i8 の graph は 22 なので dtype の取り違えも落ちる）。
  assertEquals(selection.containers["transformer"].descriptor.graph.length, 23);
  assertEquals(selection.containers["transformer"].descriptor.model.length, 5);
});

Deno.test("resolveSelection: assets は quant を切り替えても動かない", () => {
  const defaults = resolveSelection(manifest);
  const f16 = resolveSelection(manifest, { quant: "f16" });
  assertEquals(f16.assets, defaults.assets);
  // 中身は同じでも**表そのものは毎回組み直す** — parse 済み manifest の表を露出すると、返り値の
  // assets への代入が以後の全選択・在庫勘定・evict を汚染する（containers と寿命の扱いを揃える）。
  assertNotStrictEquals(defaults.assets, manifest.models["anima-turbo"].assets);
  assertNotStrictEquals(f16.assets, defaults.assets);
});

Deno.test("resolveSelection: model 指定でそのモデルの選択に切り替わる", () => {
  const selection = resolveSelection(manifest, { model: "anima-lite" });
  assertEquals(selection.model, "anima-lite");
  assertEquals(selection.quant, "w8");
  assertEquals(Object.keys(selection.containers), ["text_encoder", "transformer"]);
  assertEquals(Object.keys(selection.assets), ["tokenizer", "style_vectors"]);
  // 共有 path はモデルを跨いでも同じ 3 点セット（ADR 0041 §5 の「path の一致で共有」）。
  assertEquals(
    selection.containers["text_encoder"].parts[0],
    resolveSelection(manifest).containers["text_encoder"].parts[0],
  );
});

Deno.test("selectionRefs: 全容器の全 part + assets を宣言順で並べる", () => {
  const refs = selectionRefs(resolveSelection(manifest));
  assertEquals(paths(refs), [
    ...TEXT_ENCODER_PARTS,
    ...TEXT_CONDITIONER_PARTS,
    ...TRANSFORMER_I8_FETCHED,
    ...VAE_DECODER_PARTS,
    TOKENIZER,
    STYLE_VECTORS,
  ]);
});

Deno.test("selectionRefs: 長さ 0 の part は列に載らない（取りに行く中身が無い）", () => {
  const refs = selectionRefs(resolveSelection(manifest));
  assertEquals(
    refs.filter((ref) => ref.path === TRANSFORMER_I8_EMPTY),
    [],
    "長さ 0 の part が取得列に残っている",
  );
  // 宣言そのものには残る（添字が part の id なので、落とすと後続の添字がずれる）。
  assertEquals(resolveSelection(manifest).containers["transformer"].parts.length, 3);
  assertEquals(refs.filter((ref) => ref.size === 0), []);
});

Deno.test("selectionRefs: 同じ実体を指す 2 つの資産名は 1 本に畳まれる", () => {
  const selection = resolveSelection(manifest);
  // 表の席は 3 つ（tokenizer / style_vectors / style_alias）だが、実体は 2 本。
  assertEquals(Object.keys(selection.assets).length, 3);
  const refs = selectionRefs(selection);
  assertEquals(refs.filter((ref) => ref.path === STYLE_VECTORS).length, 1);
  assertEquals(new Set(paths(refs)).size, refs.length, "列に重複が残っている");
});

Deno.test("selectionRefs: 容器を跨いで同じ part を共有しても 1 本に畳まれる", () => {
  // anima-lite の 2 容器は anima-turbo と同じ実体を指すので、列は turbo 既定の部分集合になる。
  const lite = new Set(paths(selectionRefs(resolveSelection(manifest, { model: "anima-lite" }))));
  const turbo = new Set(paths(selectionRefs(resolveSelection(manifest))));
  for (const path of lite) {
    assertEquals(turbo.has(path), true, `${path} が既定選択の部分集合になっていない`);
  }
  assertEquals(lite.size, TEXT_ENCODER_PARTS.length + TRANSFORMER_I8_FETCHED.length + 2);
});

// ---- ⑤ weights の部分集合（`ResolveOptions.weights`）。1 つのモデルが「本体だけでも動き、
// 追加の役割を足すこともできる」形（gemma4 の model + drafter）を、配布形を割らずに扱う軸。

Deno.test("resolveSelection: weights を絞ると指定した役割だけが選択に出る（assets は全数）", () => {
  const selection = resolveSelection(manifest, { weights: ["transformer"] });
  assertEquals(Object.keys(selection.containers), ["transformer"]);
  assertEquals(Object.keys(selection.assets), ["tokenizer", "style_vectors", "style_alias"]);
  // 絞っても dtype の選び方は変わらない（既定 quant の i8）。
  assertEquals(paths(selection.containers["transformer"].parts), [...TRANSFORMER_I8_PARTS]);
  assertEquals(paths(selectionRefs(selection)), [
    ...TRANSFORMER_I8_FETCHED,
    TOKENIZER,
    STYLE_VECTORS,
  ]);
});

Deno.test("resolveSelection: weights の並びは宣言順（指定した順ではない）", () => {
  const selection = resolveSelection(manifest, { weights: ["vae_decoder", "text_encoder"] });
  // 呼び手が逆順に並べても、選択は manifest の宣言順のまま（取得の送出順を呼び手が動かさない）。
  assertEquals(Object.keys(selection.containers), ["text_encoder", "vae_decoder"]);
});

Deno.test("resolveSelection: weights の空配列は 1 本も取らない（assets だけ）", () => {
  const selection = resolveSelection(manifest, { weights: [] });
  assertEquals(Object.keys(selection.containers), []);
  assertEquals(paths(selectionRefs(selection)), [TOKENIZER, STYLE_VECTORS]);
});

Deno.test("resolveSelection: 実在しない weights 名は利用可能一覧つきで拒否する", () => {
  const error = assertThrows(
    () => resolveSelection(manifest, { weights: ["transformer", "drafter"] }),
    ManifestReferenceError,
    "weights 'drafter' は manifest に無い",
  );
  assertEquals(error.available.models, ["anima-turbo", "anima-lite"]);
});

Deno.test("resolveSelection: weights の重複は拒否する", () => {
  assertThrows(
    () => resolveSelection(manifest, { weights: ["transformer", "transformer"] }),
    ManifestReferenceError,
    "weights 'transformer' が 2 度指定された",
  );
});

Deno.test("resolveSelection: weights の実在はモデルごとに見る", () => {
  // anima-lite に text_conditioner は無い（anima-turbo にはある）。
  assertThrows(
    () => resolveSelection(manifest, { model: "anima-lite", weights: ["text_conditioner"] }),
    ManifestReferenceError,
    "利用可能: text_encoder / transformer",
  );
});

Deno.test("resolveSelection: 実在しない model は利用可能一覧つきで拒否する", () => {
  const error = assertThrows(
    () => resolveSelection(manifest, { model: "anima-xl" }),
    ManifestReferenceError,
  );
  assertEquals(error.available.models, ["anima-turbo", "anima-lite"]);
});

Deno.test("resolveSelection: 実在しない quant は利用可能一覧つきで拒否する", () => {
  const error = assertThrows(
    () => resolveSelection(manifest, { quant: "q4" }),
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

Deno.test("resolveSelection: quant の一覧は指定したモデルのものになる", () => {
  const error = assertThrows(
    () => resolveSelection(manifest, { model: "anima-lite", quant: "w8a8-s16" }),
    ManifestReferenceError,
  );
  // 別モデル（anima-turbo）にしか無い quant を勧めない。
  assertEquals(error.available.quants, ["w8"]);
});

// ---- 部品名と資産名は**別の欄**に入るので、`karume/4` にあった「取得キーの衝突」という
// 失敗形はもう存在しない（3 つの名前空間を 1 枚の表へ畳んでいたのがその門の理由だった）。

Deno.test("resolveSelection: 部品名と資産名が同名でも衝突しない（欄が別）", () => {
  const collide = parseManifest(JSON.stringify({
    format: "karume/5",
    generator: "karume/0.1.0",
    defaultModel: "m",
    models: {
      m: {
        pipeline: "anima/1",
        weights: {
          tokenizer: {
            f16: {
              container: {
                descriptor: {
                  graph: { length: 11, sha256: "f2".repeat(32) },
                  model: { length: 5, sha256: "a3".repeat(32) },
                },
                parts: [
                  { path: "tokenizer/model-00001-of-00002.krm", size: 40, sha256: "b2".repeat(32) },
                  { path: "tokenizer/model-00002-of-00002.krm", size: 64, sha256: "c3".repeat(32) },
                ],
              },
            },
          },
        },
        assets: {
          tokenizer: { path: "tokenizer/tokenizer.json", size: 12, sha256: "a1".repeat(32) },
        },
        quants: { q: { weights: { tokenizer: "f16" }, session: {} } },
        defaultQuant: "q",
        pipelineConfig: {},
      },
    },
  }));
  const selection = resolveSelection(collide);
  assertEquals(Object.keys(selection.containers), ["tokenizer"]);
  assertEquals(Object.keys(selection.assets), ["tokenizer"]);
  assertEquals(paths(selectionRefs(selection)), [
    "tokenizer/model-00001-of-00002.krm",
    "tokenizer/model-00002-of-00002.krm",
    "tokenizer/tokenizer.json",
  ]);
});
