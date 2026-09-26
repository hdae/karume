/**
 * SBV2 の**自機 A/B 門**（実 GPU・ADR 0110 決定 5）。実用席 `i8-a8`（`front` / `voice` の linear を
 * 活性まで整数内積で回す `linearCompute: "a8"` — 既定席ではない）を、**同じ機・同じ i8 重み**で
 * `session` が空の参照席（manifest から導く — {@link referencePartnerOf}。今の配布形では `i8`）と
 * 比べる。この席にはこれまで門が 1 本も無かった。
 *
 * ## 観測点 = front の 4 出力（席が最初に効いた直後 — 決定 5-4）
 *
 * SBV2 に DiT の step は無い。quant の `session` が届くのは `front` と `voice` の Session だけで
 * （`text_encoder` は `{}` — `src/sbv2/pipeline.ts` のモジュール doc）、先に走るのは `front`
 * （enc_p + dp + sdp reverse）なので、その**グラフ出力 4 本**（logw_sdp / logw_dp / m_p /
 * logs_p — 位置で引く）を観測点にする。
 *
 * - 最終出力（波形）を観測点にしない理由: front の logw は継続長の `ceil` を通ってフレーム数を
 *   決めるので、a8 の差で 1 音素でも長さが割れると波形の長さが変わり、要素ごとの比較が成り立た
 *   ない（離散化の前で比べる）。
 * - 出力を 4 本とも帯に掛ける理由: front の適格 linear は 2 本しかなく
 *   （docs/research/2026-08-03-dp4a-w8a8-design.md）、どの出力にどれだけ効くかは実測まで分からない。
 *   1 本に絞ると、効かない出力を選んだときに床が恒常的に赤になる。
 *
 * 公開面（`Sbv2Pipeline.generate`）は front の出力を返さないので、この門は `Sbv2Pipeline` と
 * **同じ合成本体**（`openSbv2State` → `synthesizeSbv2` — dump 経路 `examples/sbv2/dump.ts` と
 * 同じ入口）を回し、`front` の部品だけを「Session の出力を写して渡す」包みに差し替える
 * （{@link capturingOutputs} — 数値には触らない）。quant → SessionOptions → front / voice の
 * Session という配線は包みの外（admission と `synthesizeSbv2`）なので、門はその配線ごと見る。
 *
 * 門と故障注入の対応は `e2e_anima_ab_test.ts` のモジュール doc と同じ。census は front と voice の
 * Session を**別々に**見る（片方の Session にだけ席が届かない配線を掴むため）。束のノブは
 * `linearCompute` の 1 つだけなので故障注入は ①（`session` を空にした manifest → 床）と
 * ②（2 回目だけ別 seed → 決定性）の 2 本。
 *
 * MUST: 資産は `models/karume-sbv2-jvnv/`（untracked・実 GPU 機のローカル資産）。無い環境と GPU
 * 無し環境は理由を出して**明示 SKIP** する（ADR 0005）。発話の解析は呼び手の責務なので、日本語
 * 辞書はこのテストが `@hdae/yomi` で取る（初回だけネットワーク — `e2e_sbv2_wav_test.ts` と同じ）。
 */

import { assertThrows } from "@std/assert";
import { type FileRef, type Manifest, parseManifest, resolveSelection } from "@karume/hub";
import type { Session, SessionDiagnostics } from "@karume/runtime";
import { analyzeWithWords } from "@hdae/yomi";
import { getDictionary } from "@hdae/yomi/loader";
import { type Sbv2Utterance, toSbv2Utterance } from "../mod.ts";
import {
  assetOpener,
  closeSbv2State,
  openSbv2State,
  type Sbv2RunComponent,
  synthesizeSbv2,
} from "../src/sbv2/pipeline.ts";
import type { ComponentOpener, ModelComponent } from "../src/hub/components.ts";
import {
  assertSeatsApplied,
  type CensusRow,
  mergeCensus,
} from "../../runtime/tests/helpers/pipeline-census.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import {
  type AbBand,
  type AbMeasurement,
  assertBitIdentical,
  comparisonOf,
  describeAb,
  describeBundleCensus,
  judgeAb,
  measureAb,
  overrideQuantSession,
  quantOf,
  referencePartnerOf,
} from "./helpers/ab-gate.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { readTextIfPresent } from "./helpers/read-if-present.ts";

/** 資産の置き場（リポ直下 `models/karume-sbv2-jvnv/`）。 */
const ASSETS_DIR = new URL("../../../models/karume-sbv2-jvnv/", import.meta.url);
/** 実測と決着の置き場（`outputs/verify/<環境キー>/<日付>_sbv2-ab/` — 消して安全）。 */
const results = openResults("sbv2-ab");

const MODEL = "F1";
const PRACTICAL = "i8-a8";
/** 入力（`e2e_sbv2_wav_test.ts` の sha 門と同じ文・seed — ノブは配布形の既定）。 */
const TEXT = "こんにちは、これはテストです。";
const SEED = 0;

/**
 * front の出力（グラフ出力の**位置**順 — IR の出力名は torch のノード名で意味を持たない）と帯。
 * `band: undefined` = 未導出 — 門は実測を出して赤で止まる。
 *
 * 帯の導出規則は `e2e_anima_ab_test.ts` の `CASES` の doc と同じ（上限 = 実測 × 2 程度・床 =
 * 1 bit 以上違う MUST + 示せたときだけ係数）。導出表（2026-09-26・Intel Arc B570 / Deno・参照席 `i8`・
 * 上限 = 実測 × 2 を有効数字 2 桁へ）:
 *
 * | 出力            | 実測 relRMS | 実測 maxAbs | floor | ceiling | 採った日・機 |
 * | --------------- | ----------- | ----------- | ----- | ------- | ------------ |
 * | front.logw_sdp  | 5.1713e-3   | 2.3147e-2   | 0     | 1.0e-2  | 2026-09-26 B570 |
 * | front.logw_dp   | 1.7008e-3   | 7.4958e-3   | 0     | 3.4e-3  | 2026-09-26 B570 |
 * | front.m_p       | 3.9264e-3   | 2.5642e-2   | 0     | 7.9e-3  | 2026-09-26 B570 |
 * | front.logs_p    | 1.4743e-4   | 3.3269e-3   | 0     | 3.0e-4  | 2026-09-26 B570 |
 */
const FRONT_OUTPUTS: readonly { readonly name: string; readonly band: AbBand | undefined }[] = [
  { name: "front.logw_sdp", band: { floor: 0, ceiling: 1.0e-2 } },
  { name: "front.logw_dp", band: { floor: 0, ceiling: 3.4e-3 } },
  { name: "front.m_p", band: { floor: 0, ceiling: 7.9e-3 } },
  { name: "front.logs_p", band: { floor: 0, ceiling: 3.0e-4 } },
];

const manifestText = await readTextIfPresent(new URL("karume.json", ASSETS_DIR));
if (manifestText === undefined) {
  console.warn(
    `[karume] ${ASSETS_DIR.pathname} に karume.json が無いため SBV2 の自機 A/B 門を SKIP する` +
      "（exporter の dist.py で焼く）",
  );
}
const RUNNABLE = GPU_AVAILABLE && manifestText !== undefined;

const readManifest = (): Manifest => parseManifest(manifestText as string);

/** 発話（解析は 1 度きり — 辞書の取得を門ごとに繰り返さない）。 */
const utteranceOnce = (() => {
  let cached: Promise<Sbv2Utterance> | undefined;
  return (): Promise<Sbv2Utterance> => {
    cached ??= getDictionary().then((dictionary) =>
      toSbv2Utterance(analyzeWithWords(dictionary, TEXT))
    );
    return cached;
  };
})();

/**
 * quant が要求する資産をローカルから読む（`fetchAssets` のローカル版 — `e2e_sbv2_wav_test.ts` の
 * 読み口と同じ形）。MUST: 長さ 0 の part も並べる（添字が容器の中の id）。
 */
const loadLocalAssets = async (
  manifest: Manifest,
  quant: string,
): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const selection = resolveSelection(manifest, { model: MODEL, quant });
  const byPath = new Map<string, Uint8Array<ArrayBuffer>>();
  let assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  const read = async (key: string, ref: FileRef): Promise<void> => {
    const cached = byPath.get(ref.path);
    const bytes = cached ?? await Deno.readFile(new URL(ref.path, ASSETS_DIR));
    if (cached === undefined) byPath.set(ref.path, bytes);
    assets = { ...assets, [key]: bytes };
  };
  for (const name of Object.keys(selection.containers)) {
    const { parts } = selection.containers[name];
    for (const [index, ref] of parts.entries()) await read(`${name}[${index}]`, ref);
  }
  for (const name of Object.keys(selection.assets)) await read(name, selection.assets[name]);
  return assets;
};

/**
 * Session の `run` の出力を（グラフ出力の位置順に）写して `sink` へ渡す部品。数値には触らない —
 * 呼び手へは元の出力をそのまま返し、`sink` には写しを渡す。
 *
 * NOTE: `Session` は private フィールドを持つので、`run` 以外のメンバは元の Session に束縛して
 * 返す（Proxy の受け手のまま呼ぶと private フィールドの読みが落ちる）。
 */
const capturingOutputs = (
  component: ModelComponent,
  sink: (outputs: readonly Float32Array<ArrayBuffer>[]) => void,
): ModelComponent => ({
  ...component,
  createSession: async (gpu, options) => {
    const session = await component.createSession(gpu, options);
    const run = async (
      ...args: Parameters<Session["run"]>
    ): Promise<Awaited<ReturnType<Session["run"]>>> => {
      const outputs = await session.run(...args);
      sink(component.graph.outputs.map((name) => {
        const tensor = outputs[name];
        if (tensor === undefined || tensor.dtype !== "f32") {
          throw new Error(`front の出力 '${name}' が f32 で無い / 無い`);
        }
        return new Float32Array(tensor.data);
      }));
      return outputs;
    };
    return new Proxy(session, {
      get: (target, property) => {
        if (property === "run") return run;
        const value: unknown = Reflect.get(target, property, target);
        return typeof value === "function" ? value.bind(target) : value;
      },
    });
  },
});

/** 1 合成ぶんの観測（front の出力 4 本・波形・front / voice の census）。 */
type Observation = {
  readonly front: readonly Float32Array<ArrayBuffer>[];
  /** 波形（帯には掛けない — 決定性の検査だけに使う。voice の Session まで含めた決定性）。 */
  readonly audio: Float32Array<ArrayBuffer>;
  readonly census: Readonly<Record<"front" | "voice", readonly CensusRow[]>>;
};

/** `quant` で 1 回合成し、front の出力と census を採る（GPU は状態が自前で取って返す）。 */
const observeFront = async (
  manifest: Manifest,
  quant: string,
  seed: number,
): Promise<Observation> => {
  const where = `${quant} / seed ${seed}`;
  const assets = await loadLocalAssets(manifest, quant);
  const base = await assetOpener(assets);
  const captured: (readonly Float32Array<ArrayBuffer>[])[] = [];
  const open: ComponentOpener = (key) =>
    key === "front" ? capturingOutputs(base(key), (outputs) => captured.push(outputs)) : base(key);
  const runs: Record<Sbv2RunComponent, SessionDiagnostics["lastRunPipelines"][]> = {
    text_encoder: [],
    front: [],
    voice: [],
  };
  const state = await openSbv2State({ manifest, assets }, open, {
    model: MODEL,
    quant,
    onRunDiagnostics: (component, diagnostics) => {
      runs[component].push(diagnostics.lastRunPipelines);
    },
  });
  let audio: Float32Array<ArrayBuffer>;
  try {
    ({ audio } = await synthesizeSbv2(state, await utteranceOnce(), { seed }));
  } finally {
    closeSbv2State(state);
  }
  if (captured.length !== 1) {
    throw new Error(`${where}: front の run が ${captured.length} 回（1 合成 = 1 回のはず）`);
  }
  return {
    front: captured[0],
    audio,
    census: {
      front: mergeCensus(runs.front, `${where} の front`),
      voice: mergeCensus(runs.voice, `${where} の voice`),
    },
  };
};

/** front の出力ごとの実測（位置で {@link FRONT_OUTPUTS} と対にする）。 */
const measureFront = (
  practical: Observation,
  reference: Observation,
): readonly AbMeasurement[] => {
  if (practical.front.length !== FRONT_OUTPUTS.length) {
    throw new Error(
      `front の出力が ${practical.front.length} 本（期待 ${FRONT_OUTPUTS.length} 本 — 位置の対応が崩れた）`,
    );
  }
  return FRONT_OUTPUTS.map((_, index) => measureAb(practical.front[index], reference.front[index]));
};

const errorText = (cause: unknown): string =>
  cause instanceof Error ? cause.message : String(cause);

const ID = `${MODEL}-${PRACTICAL}`;

Deno.test({
  name: `e2e(実GPU): ${ID} の front 出力が参照席と決定性・census・帯で釣り合う（自機 A/B）`,
  ignore: !RUNNABLE,
  fn: async () => {
    await runRecordedCase(results, { id: ID, failureNote: errorText }, async (recorded) => {
      const manifest = readManifest();
      const reference = referencePartnerOf(manifest, MODEL, PRACTICAL);
      const declared = quantOf(manifest, MODEL, PRACTICAL).session;
      const referenceRun = await observeFront(manifest, reference, SEED);
      const first = await observeFront(manifest, PRACTICAL, SEED);
      const second = await observeFront(manifest, PRACTICAL, SEED);
      const measured = measureFront(first, referenceRun);
      FRONT_OUTPUTS.forEach(({ name, band }, index) => {
        recorded.comparisons.push(
          comparisonOf({ output: name, reference, practical: PRACTICAL }, measured[index], band),
        );
        console.log(`[e2e] sbv2 A/B ${ID} ${name}: ${describeAb(measured[index])}`);
      });
      const census = `front ${describeBundleCensus(first.census.front, declared)} / voice ${
        describeBundleCensus(first.census.voice, declared)
      }`;
      console.log(`[e2e] sbv2 A/B ${ID}: ${reference} → ${PRACTICAL} / census ${census}`);
      FRONT_OUTPUTS.forEach(({ name }, index) => {
        assertBitIdentical(first.front[index], second.front[index], `${ID}: 実用席の 2 回 ${name}`);
      });
      assertBitIdentical(first.audio, second.audio, `${ID}: 実用席の 2 回の波形`);
      assertSeatsApplied(first.census.front, declared, `${ID}: 実用席の front`);
      assertSeatsApplied(first.census.voice, declared, `${ID}: 実用席の voice`);
      FRONT_OUTPUTS.forEach(({ name, band }, index) => {
        judgeAb(measured[index], band, `${ID} ${name}`);
      });
      return { status: "pass", note: `census ${census}` };
    });
  },
});

// --- 故障注入（ADR 0110 決定 5-5 — 門が赤になることを assert する）-------------------

Deno.test({
  name: "e2e(実GPU): 故障注入 ① 実用席の session を空にした manifest は床で赤になる",
  ignore: !RUNNABLE,
  fn: async () => {
    const id = `fault1-empty-session-${MODEL}`;
    await runRecordedCase(results, { id, failureNote: errorText }, async (recorded) => {
      const manifest = readManifest();
      const reference = referencePartnerOf(manifest, MODEL, PRACTICAL);
      const injected = parseManifest(
        JSON.stringify(
          overrideQuantSession(JSON.parse(manifestText as string), MODEL, PRACTICAL, {}),
        ),
      );
      const referenceRun = await observeFront(manifest, reference, SEED);
      const faulty = await observeFront(injected, PRACTICAL, SEED);
      const measured = measureFront(faulty, referenceRun);
      FRONT_OUTPUTS.forEach(({ name, band }, index) => {
        recorded.comparisons.push(
          comparisonOf(
            { output: name, reference, practical: `${PRACTICAL}（session 空）` },
            measured[index],
            band,
          ),
        );
        console.log(`[e2e] sbv2 故障注入 ① ${name}: ${describeAb(measured[index])}`);
        assertThrows(() => judgeAb(measured[index], band, `${id} ${name}`), Error, "床の失敗");
      });
    });
  },
});

Deno.test({
  name: "e2e(実GPU): 故障注入 ② 実用席の 2 回目だけ別 seed にすると決定性の門が赤になる",
  ignore: !RUNNABLE,
  fn: async () => {
    const manifest = readManifest();
    const first = await observeFront(manifest, PRACTICAL, SEED);
    const second = await observeFront(manifest, PRACTICAL, SEED + 1);
    // seed が効くのは sdp reverse のノイズ（z_noise）なので、確実に割れるのは logw_sdp（位置 0）。
    assertThrows(
      () => assertBitIdentical(first.front[0], second.front[0], "故障注入 ②"),
      Error,
      "ビット一致しない",
    );
  },
});
