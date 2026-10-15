/**
 * `Umt5Encoder`（`@karume/models/umt5`）の通しの e2e（実 GPU）— 配布形ミラー `models/karume-umt5-xxl` を
 * `fromPretrained(denoDirectory(…))` で組み、Wan の umT5 の移植の門（`e2e_wan_umt5_test.ts`）と**同じ golden 10 本**を
 * **同じ判定**（正規化比 r が帯の内 — `helpers/umt5-golden.ts`）で見る。
 *
 * 入力は golden のメタの `cleaned`（上流の前処理 `prompt_clean` の後の文字列）: golden の `input_ids` はこれを
 * transformers 5.14.1 に通した列で、単体の encoder は前処理をしない（Wan の前処理の後の文字列は単体のトークナイザの
 * 受理集合に入る — `umt5_tokenizer_test.ts`）。
 *
 * ## 門
 *
 * 1. **ホスト（GPU 不要）**: ミラーのトークナイザ資産（manifest の `tokenizer` — hub が sha256 を照合して渡す）で組んだ
 *    トークナイザが、`cleaned` から golden の `input_ids` と**ビット一致**する列を出す。golden を採った容器の part 0 の
 *    sha256 が、ミラーの manifest が宣言する `text_encoder` の part 0 と同じ（= golden は配る重みから採った）。
 * 2. **実 GPU**: `tokenize(cleaned)` と `encode(cleaned).inputIds` が golden の `input_ids` と一致し、`hiddenStates`
 *    （`[L, 4096]`）の正規化比 r が帯の内・非有限 0（`judgeBand` — 故障注入は Wan の門が持つので、ここは渡さない）。
 * 3. **実 GPU**: 中断済みの signal は run の前に `signal.reason` を包まずに投げ、同じ encoder で次の `encode` が回る。
 *    `dispose` の後の `encode` は落ちる。
 *
 * ミラー・golden・GPU のどれかが無い環境は明示 SKIP（ADR 0005）— ミラーごと無い形は門番 `distribution_gate_test.ts`、
 * golden の欠けは `e2e_wan_umt5_test.ts` の完全性テストが FAIL にする。
 */

import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { fetchAssets, loadManifest, resolveSelection } from "@karume/hub";
import { denoDirectory } from "@karume/hub/deno";
import { Umt5Encoder } from "../umt5.ts";
import { parseUmt5TokenizerAsset, Umt5Tokenizer } from "../src/umt5/tokenizer.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "./helpers/gpu.ts";
import { settleReleases } from "./helpers/settle-releases.ts";
import {
  CASES,
  difference,
  filePresent,
  firstMismatch,
  formatRatio,
  GENERATE,
  type Golden,
  goldenUrl,
  judgeBand,
  loadGolden,
  ratioOf,
  referenceRatioOf,
  UMT5_NORMALIZED_BAND,
} from "./helpers/umt5-golden.ts";

/** 配布形ミラー（`dist.py --pipeline umt5` の既定の出力先）。 */
const DIST_ROOT = new URL("../../../models/karume-umt5-xxl/", import.meta.url);
const ASSEMBLE = "cd tools/export-recipes && uv run python dist.py --pipeline umt5";
/** 席（明示 — 省略すると manifest の既定へ黙って移る。golden は i8 の重みから採った）。 */
const MODEL = "xxl";
const QUANT = "i8";
/** manifest の部品名とトークナイザ資産のキー（`src/umt5/encoder.ts` と同じ綴り）。 */
const TEXT_ENCODER = "text_encoder";
const TOKENIZER = "tokenizer";

const DIST_PRESENT = filePresent(new URL("karume.json", DIST_ROOT));
const GOLDENS_PRESENT = CASES.every(({ name }) => filePresent(goldenUrl(name)));

if (!DIST_PRESENT) {
  console.warn(
    `[karume] 配布形ミラー ${DIST_ROOT.pathname} が無いため umT5 の encoder の e2e を SKIP する。組み立て: ` +
      `${ASSEMBLE}（全 SKIP は門番 distribution_gate_test.ts が FAIL にする）`,
  );
} else if (!GOLDENS_PRESENT) {
  console.warn(
    `[karume] umT5 の golden（${
      goldenUrl(CASES[0].name).pathname
    } など）が揃っていないため umT5 の encoder の ` +
      `e2e を SKIP する。生成: ${GENERATE}`,
  );
}

/** ミラーの選択（manifest の宣言 — 席は明示）。 */
const loadMirror = async () => {
  const loaded = await loadManifest(denoDirectory(DIST_ROOT));
  return { loaded, selection: resolveSelection(loaded.manifest, { model: MODEL, quant: QUANT }) };
};

/** 1 ケースの入力（golden のメタの前処理の後の文字列）と、その golden の id 列との最初の食い違い。 */
const idsMismatch = (ids: Int32Array, golden: Golden): number => firstMismatch(ids, golden.ids);

Deno.test({
  name:
    "umT5 encoder ホスト（GPU 不要）: ミラーのトークナイザが golden の前処理の後の文字列から golden の id 列を出し、" +
    "golden はミラーの text_encoder の part 0 から採られている",
  ignore: !DIST_PRESENT || !GOLDENS_PRESENT,
  fn: async () => {
    const { loaded, selection } = await loadMirror();
    assert(Object.hasOwn(selection.assets, TOKENIZER), `ミラーに資産 '${TOKENIZER}' が無い`);
    const assets = await fetchAssets(loaded, { [TOKENIZER]: selection.assets[TOKENIZER] });
    const tokenizer = new Umt5Tokenizer(
      parseUmt5TokenizerAsset(JSON.parse(new TextDecoder().decode(assets[TOKENIZER]))),
    );
    const part0 = selection.containers[TEXT_ENCODER].parts[0].sha256;
    for (const { name } of CASES) {
      const golden = await loadGolden(name);
      assertEquals(
        golden.meta.part0Sha256,
        part0,
        `${name}: golden を採った容器（part 0 の sha256）がミラーの text_encoder と違う`,
      );
      const ids = tokenizer.tokenize(golden.meta.cleaned, name);
      const at = idsMismatch(ids, golden);
      assertEquals(at, -1, `${name}: id 列が golden と ${at} 番目で割れる（${ids.length} 本）`);
    }
  },
});

Deno.test({
  name:
    "umT5 encoder（実 GPU）: fromPretrained(denoDirectory) で組み、golden 10 本の hiddenStates の正規化比 r が帯の内" +
    "（Wan の umT5 の移植の門と同じ golden・同じ帯）・tokenize と inputIds が golden の id 列",
  ignore: !DIST_PRESENT || !GOLDENS_PRESENT || !GPU_AVAILABLE,
  fn: async (t) => {
    let deviceLost: string | undefined;
    const gpu = await acquireTestGpu({
      onDeviceLost: (info) => {
        deviceLost = `${info.reason}: ${info.message}`;
      },
    });
    try {
      const encoder = await Umt5Encoder.fromPretrained(denoDirectory(DIST_ROOT), {
        gpu,
        model: MODEL,
        quant: QUANT,
      });
      try {
        for (const { name } of CASES) {
          await t.step(name, async () => {
            const golden = await loadGolden(name);
            const text = golden.meta.cleaned;
            const tokenized = encoder.tokenize(text);
            assertEquals(idsMismatch(tokenized, golden), -1, `${name}: tokenize の id 列`);

            const encoding = await encoder.encode(text);
            assertEquals(idsMismatch(encoding.inputIds, golden), -1, `${name}: inputIds`);
            assertEquals(encoding.length, golden.ids.length, `${name}: length`);
            const [, rows, width] = golden.outputShape;
            assertEquals([encoding.length, encoding.hiddenSize], [rows, width], `${name}: 形`);
            const diff = difference(encoding.hiddenStates, golden.expected);
            const referenceRatio = referenceRatioOf(golden, name);
            const ratio = ratioOf(diff) / referenceRatio;
            console.log(
              `[umt5-encoder] ${name}: L ${encoding.length} / ${
                formatRatio("GPU / f64 参照", diff, referenceRatio)
              }`,
            );
            assertEquals(diff.nonFinite, 0, `${name}: 非有限`);
            assertEquals(deviceLost, undefined, "device lost");
            judgeBand(UMT5_NORMALIZED_BAND, name, ratio, []);
            await settleReleases(gpu);
          });
        }

        await t.step(
          "中断済みの signal は run の前に reason をそのまま投げ、次の encode は回る",
          async () => {
            const controller = new AbortController();
            const reason = new Error("umt5-encoder: テストの中断");
            controller.abort(reason);
            const error = await assertRejects(() =>
              encoder.encode("a cat", { signal: controller.signal })
            );
            assertStrictEquals(error, reason);
            const after = await encoder.encode("a cat");
            assertEquals(after.length, encoder.tokenize("a cat").length);
          },
        );
      } finally {
        await encoder.dispose();
        await settleReleases(gpu);
      }
      await assertRejects(() => encoder.encode("a cat"), Error, "dispose 済み");
      assertEquals(deviceLost, undefined, "device lost");
    } finally {
      gpu.destroy();
    }
  },
});
