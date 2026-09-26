// 束の census 表（`helpers/census-table.ts`）の門（GPU も重みも要らない — manifest と表だけを読む）。
//
// ① 表そのものの整合: 行の束が家族の合成を通り、書いた席が束の非参照値の席だけであること。
// ② 鍵の出どころ: 表の鍵が manifest の `session` を家族の合成に通した**実効** SessionOptions で
//    決まること（stateAttentionReduce 昇格前の gemma は models の既定を合成の外で足していたため、
//    manifest の字面で引く表は実際に走る束を取り違えた — その穴の回帰）。
// ③ 配布ミラーの束の網羅: 数値を変える非参照値を 1 つでも持つ quant 席は、表に行があること
//    （census の無い束が配布に増えるのを機械で防ぐ — ADR 0110 決定 5 ①）。
//
// ミラーが無い機では③だけを理由つきで**明示 SKIP** する（`session_options_mirror_test.ts` と同じ
// 規律 — ミラーごと無い形は `runtime/tests/distribution_gate_test.ts` が FAIL にする）。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { type Manifest, parseManifest, type SessionSpec } from "@karume/hub";
import { resolveGemmaSessionOptions } from "../src/gemma/session-options.ts";
import {
  type BundleCensusRow,
  bundleKey,
  CENSUS_TABLE,
  censusRowOf,
  effectiveSessionOptions,
  phaseExpectations,
} from "./helpers/census-table.ts";

const MODELS_ROOT = new URL("../../../models/", import.meta.url);

/** 行の (部品, 相) を全て並べる。 */
const phasesOf = (row: BundleCensusRow): readonly (readonly [string, string])[] =>
  Object.entries(row.census).flatMap(([component, phases]) =>
    Object.keys(phases ?? {}).map((phase) => [component, phase] as const)
  );

describe("census 表の行の整合（GPU・資産不要）", () => {
  it("行の束は家族の合成をそのまま通り、鍵は参照の束（空）でない", () => {
    for (const row of CENSUS_TABLE) {
      const where = `${row.family} [${bundleKey(row.session)}]`;
      // 明示指定として合成へ通す — 家族が受けないキー・組合せ違反はここで落ちる。
      const effective = effectiveSessionOptions(row.family, {}, row.session, where);
      assertEquals(bundleKey(effective), bundleKey(row.session), where);
      assert(bundleKey(row.session) !== "", `${where}: 参照の束に census の行は要らない`);
    }
  });

  it("(系列, モデル, 束) は重複しない", () => {
    const seen = new Set<string>();
    for (const row of CENSUS_TABLE) {
      for (const model of row.models) {
        const key = `${row.family} / ${model} / ${bundleKey(row.session)}`;
        assert(!seen.has(key), `census 表の行が重複している: ${key}`);
        seen.add(key);
      }
    }
  });

  it("各行は相を 1 つ以上持ち、期待は束の非参照値の席だけで、本数は非負の整数", () => {
    for (const row of CENSUS_TABLE) {
      const where = `${row.family} [${bundleKey(row.session)}]`;
      const phases = phasesOf(row);
      assert(phases.length > 0, `${where}: 相が 1 つも無い（未導出なら行ごと書かない）`);
      for (const [component, phase] of phases) {
        const expected = phaseExpectations(row, component, phase);
        assert(expected !== undefined && expected.length > 0, `${where} ${component}/${phase}`);
        for (const one of expected) {
          assert(
            "count" in one && Number.isSafeInteger(one.count) && one.count >= 0,
            `${where} ${component}/${phase}: ${one.label} の本数`,
          );
        }
      }
    }
  });

  it("束に無い席の本数を書いた行は、期待を組む時点で落ちる", () => {
    const broken: BundleCensusRow = {
      family: "sbv2",
      models: ["F1"],
      session: { linearCompute: "a8" },
      census: { front: { run: { linearGemvReduce: { variant: 1, reference: 0 } } } },
    };
    assertThrows(() => phaseExpectations(broken, "front", "run"), Error, "linearGemvReduce");
  });

  it("未記入の相は期待を返さない（0 本の断言に化けない）", () => {
    const row = censusRowOf("gemma4", "e2b", { linearGemvReduce: "parallel" });
    assert(row !== undefined);
    assertEquals(phaseExpectations(row, "target", "prefill"), undefined);
  });
});

describe("表の鍵は manifest の session を家族の合成に通した実効設定で決まる", () => {
  it("gemma の宣言は入口が呼ぶ合成と同じ値になり、宣言の無い席は参照の束になる", () => {
    const where = "census: gemma4 e2b";
    const declared: SessionSpec = {
      linearGemvReduce: "parallel",
      stateAttentionReduce: "parallel",
    };
    const effective = effectiveSessionOptions("gemma4", declared, {}, where);
    // 表が使う合成と、gemma の入口（`resolveGemmaSessionOptions`）が Session へ渡す値は同じ 1 本。
    assertEquals(effective, resolveGemmaSessionOptions(declared, {}, where));
    assertEquals(bundleKey(effective), "linearGemvReduce=parallel,stateAttentionReduce=parallel");
    // 宣言の無い席（`i4`）は参照の束 — 家族のコード既定が合成へ紛れ込むとここが空でなくなる。
    assertEquals(bundleKey(effectiveSessionOptions("gemma4", {}, {}, where)), "");
    // 明示の参照値は宣言に勝ち、鍵から落ちる（③ を sequential へ戻した i4-gemvpar は GEMV だけの束）。
    const reverted = effectiveSessionOptions("gemma4", declared, {
      stateAttentionReduce: "sequential",
    }, where);
    assertEquals(bundleKey(reverted), "linearGemvReduce=parallel");
    assertEquals(
      censusRowOf("gemma4", "e2b", reverted)?.session,
      { linearGemvReduce: "parallel" },
    );
  });

  it("行の無い束・未知の系列は引けない", () => {
    assertEquals(censusRowOf("irodori", "v4-small", { linearCompute: "f16" }), undefined);
    assertEquals(censusRowOf("gemma4", "e4b", { linearGemvReduce: "parallel" }), undefined);
    assertThrows(() => effectiveSessionOptions("unknown-family", {}), Error, "unknown-family");
  });
});

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
  return mirrors.toSorted(([a], [b]) => a.localeCompare(b));
};

const MIRRORS = await listMirrors();
if (MIRRORS.length === 0) {
  console.warn(
    `[karume] ${MODELS_ROOT.pathname} に配布形ミラーの karume.json が 1 本も無いため、` +
      "配布の束が census 表に行を持つことの検査を SKIP する",
  );
}

describe({
  name: "配布ミラーの束は census 表に行を持つ",
  ignore: MIRRORS.length === 0,
  fn: () => {
    it("非参照値を持つ全 quant 席の (系列, モデル, 実効 SessionOptions) に行がある", () => {
      const missing: string[] = [];
      let bundles = 0;
      for (const [mirror, manifest] of MIRRORS) {
        for (const [modelName, entry] of Object.entries(manifest.models)) {
          const family = entry.pipeline.name;
          for (const [quantName, quant] of Object.entries(entry.quants)) {
            const where = `${mirror} ${modelName} '${quantName}'`;
            const effective = effectiveSessionOptions(family, quant.session, {}, where);
            if (bundleKey(effective) === "") continue;
            bundles++;
            if (censusRowOf(family, modelName, effective) === undefined) {
              missing.push(`${where}: ${family} [${bundleKey(effective)}]`);
            }
          }
        }
      }
      assertEquals(missing, [], "census 表（helpers/census-table.ts）に行の無い束");
      // 0 件で緑になる形（ミラーはあるが非参照値の束を 1 つも読めていない）を通さない。
      assert(bundles > 0, "ミラーはあるが非参照値の束を 1 つも検査していない");
    });
  },
});
