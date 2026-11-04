// A check that, for **every model × every quant** of the distribution mirrors
// (`models/*/karume.json`), the shared composition `resolveSessionOptions(family policy,
// quant.session, {})` returns the same `SessionOptions` as the mapping before the unification (no
// GPU and no weights needed — it only reads manifests).
//
// Before the unification, the 7 families other than gemma passed `toSessionOptions(quant.session)`
// straight to the Session, and gemma, given an explicit `{}`, copied the declaration's 4 fields as
// they were (= the same values as `toSessionOptions` for an accepted declaration). If the two agree
// here, the settings that reach the Session have not moved by a single byte = the grounds that the
// existing reference values (sha rows) do not move. If the policy refuses the declaration of even
// one quant, that fails here as a regression that makes already-distributed assets unreadable.
//
// It also checks that the features every family requests from the GPU (`sessionGpuFeatures` —
// the declared ones ∪ the features the effective settings need) are exactly the declared ones (more
// requests would silently tighten the device requirements for reading the same asset).
//
// On a machine without the mirrors it **SKIPs explicitly** with a reason (the same rule as
// `runtime/tests/e2e_*_fusion_counts_test.ts` — the shape with no mirror at all is FAILed by
// `runtime/tests/distribution_gate_test.ts`).

import { isNotFound, readDir, readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { type Manifest, parseManifest } from "@karume/hub";
import { ANIMA_SESSION_POLICY } from "../../models/src/anima/pipeline.ts";
import { BIREFNET_SESSION_POLICY } from "../../models/src/birefnet/pipeline.ts";
import { DEPTH_ANYTHING_SESSION_POLICY } from "../../models/src/depth-anything/pipeline.ts";
import {
  GEMMA_SESSION_POLICY,
  resolveGemmaSessionOptions,
} from "../../models/src/gemma/session-options.ts";
import { IRODORI_SESSION_POLICY } from "../../models/src/irodori/admission.ts";
import { MOSS_SESSION_POLICY } from "../../models/src/moss/preset.ts";
import { SBV2_SESSION_POLICY } from "../../models/src/sbv2/pipeline.ts";
import { sessionGpuFeatures } from "../../models/src/session/gpu-features.ts";
import {
  type FamilySessionPolicy,
  resolveSessionOptions,
  toSessionOptions,
} from "../../models/src/session/options.ts";
import { SIGLIP2_SESSION_POLICY } from "../../models/src/siglip2/pipeline.ts";
import { VOWEL_DETECTOR_SESSION_POLICY } from "../../models/src/vowel-detector/pipeline.ts";
import { UMT5_ENCODER_SESSION_POLICY } from "../../models/src/umt5/encoder.ts";
import { WAN_SESSION_POLICY } from "../../models/src/wan/family.ts";

const MODELS_ROOT = new URL("../../../models/", import.meta.url);

/**
 * The manifest's pipeline name → the family's policy.
 *
 * MUST: an unknown pipeline name fails rather than SKIPs — if this check silently left out a new
 * family on the day its mirror is added, a policy that has not caught up (one that refuses
 * already-distributed declarations) would go unseen.
 */
const POLICIES: Readonly<Record<string, FamilySessionPolicy>> = {
  anima: ANIMA_SESSION_POLICY,
  birefnet: BIREFNET_SESSION_POLICY,
  "depth-anything": DEPTH_ANYTHING_SESSION_POLICY,
  gemma4: GEMMA_SESSION_POLICY,
  "gemma4-qat": GEMMA_SESSION_POLICY,
  irodori: IRODORI_SESSION_POLICY,
  "moss-image": MOSS_SESSION_POLICY,
  sbv2: SBV2_SESSION_POLICY,
  siglip2: SIGLIP2_SESSION_POLICY,
  "umt5-encoder": UMT5_ENCODER_SESSION_POLICY,
  "vowel-detector": VOWEL_DETECTOR_SESSION_POLICY,
  wan: WAN_SESSION_POLICY,
  // Wan2.2 TI2V shares the policy of 2.1 (the shared admission — `family.ts`).
  "wan-ti2v": WAN_SESSION_POLICY,
};

/**
 * Reads one mirror's manifest (`undefined` when there is none).
 *
 * MUST: anything other than NotFound propagates — folding an I/O failure into "no mirror" would
 * turn a check that never ran green under the guise of a SKIP.
 */
const readMirror = async (dir: URL): Promise<Manifest | undefined> => {
  try {
    return parseManifest(await readTextFile(new URL("karume.json", dir)));
  } catch (cause) {
    if (isNotFound(cause)) return undefined;
    throw cause;
  }
};

const listMirrors = async (): Promise<readonly (readonly [string, Manifest])[]> => {
  const mirrors: (readonly [string, Manifest])[] = [];
  try {
    for await (const entry of readDir(MODELS_ROOT)) {
      if (!entry.isDirectory) continue;
      const manifest = await readMirror(new URL(`${entry.name}/`, MODELS_ROOT));
      if (manifest !== undefined) mirrors.push([entry.name, manifest]);
    }
  } catch (cause) {
    if (!isNotFound(cause)) throw cause;
  }
  return mirrors.sort(([a], [b]) => a.localeCompare(b));
};

const MIRRORS = await listMirrors();
if (MIRRORS.length === 0) {
  console.warn(
    `[karume] ${MODELS_ROOT.pathname} has no distribution-mirror karume.json, so the equivalence` +
      " check of the SessionOptions composition against the pre-unification mapping is SKIPped",
  );
}

test.skipIf(MIRRORS.length === 0)(
  "for every quant of the distribution mirrors, the shared composition returns the same SessionOptions as the pre-unification mapping",
  () => {
    let checked = 0;
    for (const [mirror, manifest] of MIRRORS) {
      for (const [modelName, entry] of Object.entries(manifest.models)) {
        const family = entry.pipeline.name;
        if (!Object.hasOwn(POLICIES, family)) {
          throw new Error(
            `${mirror}: this check has no policy for pipeline '${family}' (add it to POLICIES)`,
          );
        }
        const policy = POLICIES[family];
        for (const [quantName, quant] of Object.entries(entry.quants)) {
          const where = `${mirror} ${modelName} '${quantName}'`;
          const resolved = resolveSessionOptions(policy, quant.session, {}, where);
          assertEquals(resolved, toSessionOptions(quant.session), where);
          if (policy === GEMMA_SESSION_POLICY) {
            // The wrapper gemma's entry point actually calls gives the same value (the way
            // resolveGemmaSessionOptions is called is unchanged).
            assertEquals(resolveGemmaSessionOptions(quant.session, {}, where), resolved, where);
          }
          assertEquals(sessionGpuFeatures(quant.gpuFeatures, resolved), quant.gpuFeatures, where);
          checked++;
        }
      }
    }
    // Do not let a zero-count run go green (manifests exist but not one quant was read).
    if (checked === 0) throw new Error("mirrors exist but not one quant was checked");
  },
);
