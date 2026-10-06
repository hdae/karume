// 配布形ミラー（`models/*/karume.json`）の**全 model × 全 quant** について、共通の合成
// `resolveSessionOptions(家族の受理表, quant.session, {})` が 1 本化前の写像と同じ
// `SessionOptions` を返すことの門（GPU も重みも要らない — manifest を読むだけ）。
//
// 1 本化前は、gemma 以外の 7 家族が `toSessionOptions(quant.session)` を直接 Session へ渡し、
// gemma は明示指定 `{}` のとき宣言の 4 欄をそのまま写していた（= 受理される宣言については
// `toSessionOptions` と同じ値）。ここで両者が一致すれば、Session へ渡る設定は 1 バイトも
// 動いていない = 既存の参照値（sha 行）が動かないことの根拠になる。受理表が宣言を拒否する
// quant が 1 つでもあれば、それは配布済みの資産を読めなくする退行としてここで落ちる。
//
// 併せて、全家族が GPU へ要求する feature（`sessionGpuFeatures` — 宣言 ∪ 実効設定が要る feature）も
// 宣言そのままであることを見る（要求が増えると、同じ資産を読む device の条件が黙って厳しくなる）。
//
// ミラーが無い機では理由を出して**明示 SKIP** する（`runtime/tests/assets_fusion_counts_test.ts`
// と同じ規律 — ミラーごと無い形は `runtime/tests/distribution_gate_test.ts` が FAIL にする）。

import { assertEquals } from "@std/assert";
import { type Manifest, parseManifest } from "@karume/hub";
import { ANIMA_SESSION_POLICY } from "../src/anima/pipeline.ts";
import { BIREFNET_SESSION_POLICY } from "../src/birefnet/pipeline.ts";
import { DEPTH_ANYTHING_SESSION_POLICY } from "../src/depth-anything/pipeline.ts";
import { GEMMA_SESSION_POLICY, resolveGemmaSessionOptions } from "../src/gemma/session-options.ts";
import { IRODORI_SESSION_POLICY } from "../src/irodori/admission.ts";
import { SBV2_SESSION_POLICY } from "../src/sbv2/pipeline.ts";
import { sessionGpuFeatures } from "../src/session/gpu-features.ts";
import {
  type FamilySessionPolicy,
  resolveSessionOptions,
  toSessionOptions,
} from "../src/session/options.ts";
import { SIGLIP2_SESSION_POLICY } from "../src/siglip2/pipeline.ts";
import { VOWEL_DETECTOR_SESSION_POLICY } from "../src/vowel-detector/pipeline.ts";
import { WAN_SESSION_POLICY } from "../src/wan/family.ts";
import { UMT5_ENCODER_SESSION_POLICY } from "./helpers/census-table.ts";

const MODELS_ROOT = new URL("../../../models/", import.meta.url);

/**
 * manifest の pipeline 名 → 家族の受理表。
 *
 * MUST: 未知の pipeline 名は SKIP せず落とす — 新しい家族のミラーを足した日に、この門が
 * 黙ってその家族を外すと、受理表の追随漏れ（配布済みの宣言を拒否する表）が見えなくなる。
 */
const POLICIES: Readonly<Record<string, FamilySessionPolicy>> = {
  anima: ANIMA_SESSION_POLICY,
  birefnet: BIREFNET_SESSION_POLICY,
  "depth-anything": DEPTH_ANYTHING_SESSION_POLICY,
  gemma4: GEMMA_SESSION_POLICY,
  "gemma4-qat": GEMMA_SESSION_POLICY,
  irodori: IRODORI_SESSION_POLICY,
  sbv2: SBV2_SESSION_POLICY,
  siglip2: SIGLIP2_SESSION_POLICY,
  "umt5-encoder": UMT5_ENCODER_SESSION_POLICY,
  "vowel-detector": VOWEL_DETECTOR_SESSION_POLICY,
  wan: WAN_SESSION_POLICY,
  // Wan2.2 TI2V は 2.1 と同じ受理表を共有する（共通の admission — `family.ts`）。
  "wan-ti2v": WAN_SESSION_POLICY,
};

/**
 * ミラー 1 本の manifest を読む（無ければ `undefined`）。
 *
 * MUST: NotFound 以外は伝播させる — I/O 異常を「ミラーが無い」に丸めると、実行されていない
 * 検証が SKIP の顔で緑になる。
 */
const readMirror = async (dir: URL): Promise<Manifest | undefined> => {
  try {
    return parseManifest(await Deno.readTextFile(new URL("karume.json", dir)));
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return undefined;
    throw cause;
  }
};

const listMirrors = async (): Promise<readonly (readonly [string, Manifest])[]> => {
  const mirrors: (readonly [string, Manifest])[] = [];
  try {
    for await (const entry of Deno.readDir(MODELS_ROOT)) {
      if (!entry.isDirectory) continue;
      const manifest = await readMirror(new URL(`${entry.name}/`, MODELS_ROOT));
      if (manifest !== undefined) mirrors.push([entry.name, manifest]);
    }
  } catch (cause) {
    if (!(cause instanceof Deno.errors.NotFound)) throw cause;
  }
  return mirrors.sort(([a], [b]) => a.localeCompare(b));
};

const MIRRORS = await listMirrors();
if (MIRRORS.length === 0) {
  console.warn(
    `[karume] ${MODELS_ROOT.pathname} に配布形ミラーの karume.json が 1 本も無いため、` +
      "SessionOptions の合成と 1 本化前の写像の同値検査を SKIP する",
  );
}

Deno.test({
  name: "配布形ミラーの全 quant で、共通の合成が 1 本化前の写像と同じ SessionOptions を返す",
  ignore: MIRRORS.length === 0,
  fn: () => {
    let checked = 0;
    for (const [mirror, manifest] of MIRRORS) {
      for (const [modelName, entry] of Object.entries(manifest.models)) {
        const family = entry.pipeline.name;
        if (!Object.hasOwn(POLICIES, family)) {
          throw new Error(
            `${mirror}: pipeline '${family}' の受理表がこの門に無い（POLICIES へ足す）`,
          );
        }
        const policy = POLICIES[family];
        for (const [quantName, quant] of Object.entries(entry.quants)) {
          const where = `${mirror} ${modelName} '${quantName}'`;
          const resolved = resolveSessionOptions(policy, quant.session, {}, where);
          assertEquals(resolved, toSessionOptions(quant.session), where);
          if (policy === GEMMA_SESSION_POLICY) {
            // gemma の入口が実際に呼ぶ包みも同じ値（resolveGemmaSessionOptions の呼び方は変えていない）。
            assertEquals(resolveGemmaSessionOptions(quant.session, {}, where), resolved, where);
          }
          assertEquals(sessionGpuFeatures(quant.gpuFeatures, resolved), quant.gpuFeatures, where);
          checked++;
        }
      }
    }
    // 0 件で緑になる形（manifest はあるが quant を 1 つも読めていない）を通さない。
    if (checked === 0) throw new Error("ミラーはあるが quant を 1 つも検査していない");
  },
});
