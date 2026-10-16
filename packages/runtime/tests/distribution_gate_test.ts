// ADR 0005「全ケース SKIP は明示 FAIL」の**配布形ミラー**版の門番（実重み系列版は
// tests/assets_gate_test.ts・GPU 版は tests/gpu_gate_test.ts）。
//
// 配布形ミラー（`models/karume-<family>/`）を根にする検査 — anima / sbv2 / irodori / gemma4 /
// QAT / wan の実資産 e2e（packages/models/tests/e2e_*_test.ts と helpers/irodori-assets.ts）・融合ヒット数の
// 配布形節（anima / anima-extra / irodori / gemma4 / gemma4-qat / wan — tests/assets_fusion_counts_test.ts）
// — は、いずれも `karume.json` の有無で `ignore` し `console.warn` だけを残して緑になる。
// `outputs/series/` だけ持つ機（新しい作業機・worktree を別ホストへ持ち出した場合）では
// assets_gate_test.ts が緑のまま実資産検査が丸ごと消えるので、ここで 1 本落とす。
//
// 欠如の検出（ミラーごと無い形）の射程は**公開済みの全ミラー**（2026-09-25 裁定）。e2e がまだ読まないミラー（siglip2 など）も
// 載せる — 門番の射程を「今 e2e が読むもの」に合わせると、e2e を足した日に門番の更新を忘れて
// 無音 SKIP が戻る。未公開の vowel-detector は載せない（配布形を作ってから載せる）。未公開でも
// e2e（と融合ヒット数の門）が根にするミラー（gemma4-qat / wan2.1）は載せる。
//
// 見るのは**有無だけではない**（ADR 0108 段 3 検収②）。`karume.json` はあるが中身が旧 major の
// まま、あるいは宣言された part が 1 本足りない・長さが宣言と違う、という形は上の e2e を
// 「開けない資産」で落とすか、悪くすると**前回の書き出しの残骸**を今回の期待値で読ませる。
// どちらも「資産が無い」とは別の壊れ方なので、門番が manifest を parse して part の実在と
// 長さまで突き合わせる。中身（block の sha256）は読まない — 実バイトの突合は各 e2e が
// `openContainer` の経路で必ず通る（§7 のハッシュ 3 分離）。
//
// manifest の parse と既定選択の検査の射程は名簿ではなく**`models/` の配置**から導く（2026-10-11
// 裁定）。`models/` は配布形だけを置く根（docs/assets-layout.md）なので、子ディレクトリは
// どれも hub の parser に通す。名簿に頼ると、`karume dist` が焼いたのに名簿へ足していない新しい
// 配布形（未公開の vowel-detector など）が「焼けたが hub が読めない」まま素通りする。名簿
// {@link DISTRIBUTIONS} が担うのは「在るべきものが無い」の検出だけで、これは配置からは導けない。
//
// この門番自身は資産がある環境では**通る（緑の 1 件として見える）**。ignore にすると
// 「門番が効いているのか、門番ごと消えているのか」が区別できなくなるため（他の 2 門番と同じ理由）。

import {
  assert,
  assertEquals,
  AssertionError,
  assertRejects,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { parseManifest, resolveSelection } from "@karume/hub";

/**
 * 「配布形ミラー無しでの全 SKIP」を明示的に許可する opt-out（`KARUME_ALLOW_NO_ASSETS` と同形）。
 *
 * MUST: 既定は fail loudly。配布形を持たない作業機や CI で通すには、意図表明としてこの環境変数を
 * 要求する（実重み系列側の opt-out とは別に持つ — 片方だけ持つ機が実在する）。
 */
const ALLOW_NO_DISTRIBUTION = Deno.env.get("KARUME_ALLOW_NO_DISTRIBUTION") === "1";

/** 配布形ミラーを置く根（上記の検査が `karume.json` を読む先と同じ URL の組み立て方）。 */
const MODELS_ROOT = new URL("../../../models/", import.meta.url);

/**
 * 門番が**在ることを要求する**配布形ミラー（欠如の検出の名簿 — manifest の検査の射程は
 * {@link discoverDistributions} が配置から導く）= 公開済みの 12 リポ（0.14.0 で初公開の `karume-umt5-xxl` と
 * `karume-wan2.2` を含む）+ 未公開の `karume-gemma4-qat` + `karume-wan2.1`。
 *
 * 公開済みの 2 本は公開前から門番に載っていた: `karume-umt5-xxl` は Wan の GPU 経路の e2e が越境参照の根にし
 * （ADR 0119 段 10d・無いと欠如が無音の SKIP になる）、`karume-wan2.2` は Wan2.2 の通しの e2e
 * （`e2e_wan_ti2v_pipeline_test.ts`）と DiT の e2e の実用席の記録（`e2e_wan_ti2v_dit_test.ts` — 席の束を manifest から
 * 読む）が根にする（ADR 0121 段 8）。
 *
 * `karume-gemma4-qat` は未公開だが、QAT の公開入口 e2e と融合ヒット数の門が根にするので、
 * 射程を広げる前から門番に載っていた（外すと既存の門を緩めることになる）。`karume-wan2.1` も未公開
 * （HF に公開しない — ADR 0118 追記（2026-10-15））だが、Wan の通しの e2e（`e2e_wan_pipeline_test.ts`）と
 * 融合ヒット数の門が根にする（ADR 0118 段 7）。
 */
const DISTRIBUTIONS = [
  "karume-anima",
  "karume-anima-extra",
  "karume-birefnet-hr",
  "karume-depth-anything-v2",
  "karume-gemma4",
  "karume-gemma4-qat",
  "karume-irodori-v4-small",
  "karume-irodori-v4.1-small",
  "karume-lucida",
  "karume-sbv2-jvnv",
  "karume-siglip2",
  "karume-wan2.1",
  "karume-wan2.2",
  "karume-umt5-xxl",
] as const;

/** この版の読み手が受け付ける配布 manifest の major（旧版は読まない — ADR 0109 決定 1）。 */
const MANIFEST_FORMAT = "karume/5";

/** 欠如の検出の名簿（per-mirror の検査で「名簿にあるか」を引く形）。 */
const REQUIRED: ReadonlySet<string> = new Set(DISTRIBUTIONS);

/**
 * `karume dist` の作業席の接尾辞（`karume.artifacts.staged_publication` の `.staging` = 書きかけ・
 * `.old` = 据え替え前の退避）。どちらもミラーではないので manifest の検査に掛けない — 書きかけを
 * 読めば焼いている最中に門が赤くなり、退避は次の焼きで回収される。
 */
const WORK_SEAT_SUFFIXES = [".staging", ".old"] as const;

const distributionRoot = (models: URL, series: string): URL => new URL(`${series}/`, models);

/**
 * `models` 直下の配布形ミラーを**配置から**列挙する（名前順・作業席とファイルは除く）。
 *
 * NOTE: 子はシンボリックリンクでもよい（stat はリンクを辿る）— 資産を別の場所に置いてリンクで
 * 見せる機がある。`models` 自体が無い機は空を返す（欠如は名簿の検査が名指しで落とす）。
 * `models` の一覧での NotFound 以外の I/O 異常は伝播させる（{@link manifestPresent} と同じ理由）。
 * 子の stat の異常は {@link isMirrorCandidate} が一覧に残し、per-mirror の test の中で名指しで落とす。
 */
const discoverDistributions = (models: URL): readonly string[] => {
  let entries: Deno.DirEntry[];
  try {
    entries = [...Deno.readDirSync(models)];
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return [];
    throw cause;
  }
  return entries
    .map((entry) => entry.name)
    .filter((name) => !WORK_SEAT_SUFFIXES.some((suffix) => name.endsWith(suffix)))
    .filter((name) => isMirrorCandidate(new URL(name, models)))
    .sort();
};

/**
 * `models` 直下の子を manifest の検査に掛けるか（ディレクトリなら掛ける・ファイルなら掛けない）。
 *
 * MUST: stat が投げた子（壊れたシンボリックリンク・辿れないリンク・読めない子）は**掛ける側に
 * 残す**。ここはモジュールのトップレベル（test の登録時）で走るので、投げればファイル全体が
 * どのミラーとも名指しの無いまま落ち、欠如の門も一緒に消える。残せば同じ異常が per-mirror の
 * test の中で、そのミラーの名前の付いた FAIL として出る（例外は握り潰さず、出る場所を移すだけ）。
 */
const isMirrorCandidate = (child: URL): boolean => {
  try {
    return Deno.statSync(child).isDirectory;
  } catch {
    return true;
  }
};

/**
 * その系列の manifest が置かれているか。
 *
 * MUST: NotFound 以外は伝播させる — 権限エラー等を「資産が無い」と読み替えると、門番自身が
 * 環境の壊れを資産の不在として報告する。ここが見るのは `karume.json` がファイルであることだけで、
 * 中身は {@link assertDistributionIntact} が見る。
 */
const manifestPresent = (root: URL): boolean => {
  try {
    return Deno.statSync(new URL("karume.json", root)).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

/**
 * 登録時の `ignore` の判定用: manifest が**無いと分かっている**か。
 *
 * MUST: NotFound 以外の I/O 異常は「無いとは分からない」（`false`）に倒して test を走らせる —
 * 登録時に投げるとファイル全体が名指し無しで落ちる（{@link isMirrorCandidate} と同じ理由）。
 * 走った test の中の {@link manifestPresent} が同じ異常を、そのミラーの名前の付いた FAIL として投げる。
 */
const manifestKnownAbsent = (root: URL): boolean => {
  try {
    return !manifestPresent(root);
  } catch {
    return false;
  }
};

/** ファイル長（不在は `undefined`・それ以外の I/O 異常は伝播させる）。 */
const fileBytes = (url: URL): number | undefined => {
  try {
    const stat = Deno.statSync(url);
    return stat.isFile ? stat.size : undefined;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return undefined;
    throw cause;
  }
};

/**
 * `models` 直下に、`series` の全ミラーの `karume.json` が在ることを断言する（無いものを名指しする）。
 */
const assertNoDistributionMissing = (models: URL, series: readonly string[]): void => {
  const missing = series.filter((name) => !manifestPresent(distributionRoot(models, name)));
  assert(
    missing.length === 0,
    `models/ に配布形ミラーの karume.json が無い系列がある: ${missing.join(", ")}。` +
      "配布形ミラーが無い機では、それを根にする実資産 e2e と融合ヒット数の検査が無音 SKIP するので、" +
      "ADR 0005 によりこれは FAIL として扱う。" +
      "tools/export-recipes の dist.py で作成すること（--pipeline と引数は dist.py 冒頭の用例 — " +
      "越境参照で焼く karume-anima-extra は docs/release-runbook.md の手順に従う）。" +
      "配布形の無い環境で意図的に通すには KARUME_ALLOW_NO_DISTRIBUTION=1 を設定すること。",
  );
};

/**
 * `root` のミラーの manifest が `karume/5` で、既定選択（既定 model × 既定 quant）の part が
 * 宣言どおりの長さで在ることを断言する。
 *
 * NOTE: 既定でない model / quant の part までは見ない。それらを読む e2e は part が欠ければ
 * `openContainer` / NotFound で赤くなる（無音 SKIP にはならない）ので、門番の担当は
 * 「ミラーごと無い」形と、既定選択が旧 major・残骸のまま据わっている形に絞る。
 */
const assertDistributionIntact = async (root: URL, series: string): Promise<void> => {
  // MUST: 名簿に無いミラーの欠如はここでしか見えない — `models/` は配布形だけの根なので、
  // `karume.json` の無い子ディレクトリは焼きかけか前回の残骸で、e2e が読めば別の壊れ方をする。
  assert(
    manifestPresent(root),
    `models/${series}/ に karume.json が無い。models/ は配布形だけを置く根なので、` +
      "これは焼きかけか残骸のディレクトリ。tools/export-recipes の dist.py で作り直すか、取り除くこと。",
  );
  const manifest = parseManifest(await Deno.readTextFile(new URL("karume.json", root)));
  // parse 自身も major を見るが、`karume/5` を**名指しで**断言しておく — hub が受ける major を
  // 増やした日に、この門番が「旧 major のミラーを据えたまま」を素通しするのを防ぐ。
  assertEquals(manifest.format, MANIFEST_FORMAT, `${series}: manifest の format`);
  const selection = resolveSelection(manifest);
  const broken: string[] = [];
  for (const [component, container] of Object.entries(selection.containers)) {
    for (const [index, part] of container.parts.entries()) {
      const where = `${component}[${index}] ${part.path}`;
      if (part.repo !== undefined) {
        // 越境参照はローカルミラーに実体を持たない（ADR 0038 §7）— 取得層の担当で、
        // ここで「無い」と数えると門番が別の系統の話を報告することになる。
        continue;
      }
      const actual = fileBytes(new URL(part.path, root));
      if (actual === undefined) {
        broken.push(`${where}: 宣言された part が無い`);
      } else if (actual !== part.size) {
        broken.push(`${where}: 長さ ${actual} が宣言の ${part.size} と違う`);
      }
    }
  }
  assertEquals(
    broken,
    [],
    `models/${series}/ の配布形が manifest の宣言と食い違っている` +
      `（model '${selection.model}' / quant '${selection.quant}'）。` +
      "焼き直しの途中や前回の残骸を e2e が読むのを避けるため、ADR 0005 によりこれは FAIL。" +
      "tools/export-recipes の dist.py で作り直すこと。",
  );
};

Deno.test({
  name: "配布門番: 配布形ミラーが欠けた全 SKIP は明示 FAIL（ADR 0005）",
  // opt-out は「配布形ミラー無しを承知で通す」意図表明のときだけ。既定では ignore しない。
  ignore: ALLOW_NO_DISTRIBUTION,
  fn: () => assertNoDistributionMissing(MODELS_ROOT, DISTRIBUTIONS),
});

for (const series of discoverDistributions(MODELS_ROOT)) {
  const root = distributionRoot(MODELS_ROOT, series);
  Deno.test({
    name:
      `配布門番: ${series} の manifest が hub の parser を通り（karume/5）、既定 quant の part が宣言どおり在る`,
    // 名簿にあるミラーの不在は上の 1 本が名指しで落とす（同じ壊れで 2 本赤くしても読み手の仕事は
    // 増えない）。名簿に無いものの不在はここでしか見えないので走らせる。
    ignore: REQUIRED.has(series) && manifestKnownAbsent(root),
    fn: () => assertDistributionIntact(root, series),
  });
}

// 以下は門番の判定そのものの故障注入。一時ディレクトリだけで完結する（実資産も GPU も要らない）
// ので、opt-out の有無に関わらず走る — 門番が ignore される CI でも判定の退行は見える。

/** 一時ディレクトリを根として `run` を引き、後始末する。 */
const withTemporaryRoot = async (run: (root: URL) => Promise<void> | void): Promise<void> => {
  const root = new URL(`file://${Deno.makeTempDirSync({ prefix: "karume-distribution-" })}/`);
  try {
    await run(root);
  } finally {
    Deno.removeSync(root, { recursive: true });
  }
};

describe("配布門番の判定（表のミラーの karume.json が揃っているか）", () => {
  /** 表の全ミラーに（中身を問わない）`karume.json` を置いた根を作る。 */
  const arrangeAll = (models: URL): void => {
    for (const series of DISTRIBUTIONS) {
      const root = distributionRoot(models, series);
      Deno.mkdirSync(root, { recursive: true });
      Deno.writeTextFileSync(new URL("karume.json", root), "{}");
    }
  };

  it("表の全ミラーが揃っていれば通る", async () => {
    await withTemporaryRoot((models) => {
      arrangeAll(models);
      assertNoDistributionMissing(models, DISTRIBUTIONS);
    });
  });

  it("表のどのミラーが 1 本欠けても、そのミラーを名指しして FAIL する", async () => {
    await withTemporaryRoot((models) => {
      arrangeAll(models);
      for (const series of DISTRIBUTIONS) {
        const manifest = new URL("karume.json", distributionRoot(models, series));
        Deno.removeSync(manifest);
        const error = assertThrows(
          () => assertNoDistributionMissing(models, DISTRIBUTIONS),
          AssertionError,
        );
        assertStringIncludes(error.message, `無い系列がある: ${series}。`);
        Deno.writeTextFileSync(manifest, "{}");
      }
    });
  });

  it("karume.json がディレクトリならミラーは無いと判定する", async () => {
    await withTemporaryRoot((models) => {
      arrangeAll(models);
      const manifest = new URL("karume.json", distributionRoot(models, "karume-lucida"));
      Deno.removeSync(manifest);
      Deno.mkdirSync(manifest);
      const error = assertThrows(
        () => assertNoDistributionMissing(models, DISTRIBUTIONS),
        AssertionError,
      );
      assertStringIncludes(error.message, "無い系列がある: karume-lucida。");
    });
  });
});

describe("配布門番の射程（models/ の配置から導くミラーの一覧）", () => {
  it("子ディレクトリを名前順に全て拾い、dist の作業席とファイルは拾わない", async () => {
    await withTemporaryRoot((models) => {
      for (const name of ["karume-b", "karume-a", "karume-a.staging", "karume-b.old"]) {
        Deno.mkdirSync(new URL(`${name}/`, models));
      }
      Deno.writeTextFileSync(new URL("README.md", models), "");
      assertEquals(discoverDistributions(models), ["karume-a", "karume-b"]);
    });
  });

  it("名簿に無い新しいミラーも拾う（名簿の更新忘れで manifest の検査から漏れない）", async () => {
    await withTemporaryRoot((models) => {
      const unlisted = "karume-vowel-detector";
      assert(!REQUIRED.has(unlisted), "被験体の前提: 名簿に無い名前");
      Deno.mkdirSync(new URL(`${unlisted}/`, models));
      assertEquals(discoverDistributions(models), [unlisted]);
    });
  });

  it("ディレクトリを指すシンボリックリンクもミラーとして拾う", async () => {
    await withTemporaryRoot((root) => {
      const models = new URL("models/", root);
      const target = new URL("elsewhere/", root);
      Deno.mkdirSync(models);
      Deno.mkdirSync(target);
      Deno.symlinkSync(target.pathname, new URL("karume-linked", models).pathname);
      assertEquals(discoverDistributions(models), ["karume-linked"]);
    });
  });

  it("壊れたリンクと辿れないリンクも投げずに拾う（登録時に落ちてファイルごと消えない）", async () => {
    await withTemporaryRoot((models) => {
      Deno.symlinkSync(
        new URL("nowhere", models).pathname,
        new URL("karume-broken", models).pathname,
      );
      Deno.symlinkSync("karume-loop", new URL("karume-loop", models).pathname);
      assertEquals(discoverDistributions(models), ["karume-broken", "karume-loop"]);
    });
  });

  it("壊れたリンクのミラーは、そのディレクトリを名指しして FAIL する", async () => {
    await withTemporaryRoot(async (models) => {
      Deno.symlinkSync(
        new URL("nowhere", models).pathname,
        new URL("karume-broken", models).pathname,
      );
      await assertRejects(
        () => assertDistributionIntact(distributionRoot(models, "karume-broken"), "karume-broken"),
        AssertionError,
        "models/karume-broken/ に karume.json が無い",
      );
    });
  });

  it("辿れないリンクは ignore の判定で投げずに test を走らせ、test の中でそのパスを名指しして投げる", async () => {
    await withTemporaryRoot(async (models) => {
      Deno.symlinkSync("karume-loop", new URL("karume-loop", models).pathname);
      const root = distributionRoot(models, "karume-loop");
      assertEquals(manifestKnownAbsent(root), false);
      await assertRejects(
        () => assertDistributionIntact(root, "karume-loop"),
        Deno.errors.FilesystemLoop,
        "karume-loop/karume.json",
      );
    });
  });

  it("models/ 自体が無ければ空（欠如は名簿の検査が名指しで落とす）", async () => {
    await withTemporaryRoot((root) => {
      assertEquals(discoverDistributions(new URL("missing/", root)), []);
    });
  });

  it("karume.json の無いミラーは、そのディレクトリを名指しして FAIL する", async () => {
    await withTemporaryRoot(async (root) => {
      await assertRejects(
        () => assertDistributionIntact(root, "karume-half-built"),
        AssertionError,
        "models/karume-half-built/ に karume.json が無い",
      );
    });
  });
});

describe("配布門番の判定（manifest の宣言と配布形の食い違い）", () => {
  const sha256 = "0".repeat(64);
  /** part 0 はヘッダ 24 B + グラフ記述 10 B + モデル記述 5 B ちょうど（container-v1 §8）。 */
  const parts = [
    { path: "m/depth/model.f32-00001-of-00002.krm", size: 39, sha256 },
    { path: "m/depth/model.f32-00002-of-00002.krm", size: 8, sha256 },
  ] as const;
  const manifestText = (format: string): string =>
    JSON.stringify({
      format,
      generator: "karume/0.13.0",
      defaultModel: "m",
      models: {
        m: {
          pipeline: "depth-anything/1",
          weights: {
            depth: {
              f32: {
                container: {
                  descriptor: {
                    graph: { length: 10, sha256 },
                    model: { length: 5, sha256 },
                  },
                  parts,
                },
              },
            },
          },
          assets: {},
          quants: { f32: { weights: { depth: "f32" }, session: {} } },
          defaultQuant: "f32",
          pipelineConfig: {},
        },
      },
    });

  /** 宣言どおりのミラー（format だけ差し替えられる）を根に置く。 */
  const arrange = (root: URL, format = MANIFEST_FORMAT): void => {
    Deno.writeTextFileSync(new URL("karume.json", root), manifestText(format));
    Deno.mkdirSync(new URL("m/depth/", root), { recursive: true });
    for (const part of parts) {
      Deno.writeFileSync(new URL(part.path, root), new Uint8Array(part.size));
    }
  };

  it("宣言どおりのミラーは通る", async () => {
    await withTemporaryRoot(async (root) => {
      arrange(root);
      await assertDistributionIntact(root, "synthetic");
    });
  });

  it("宣言された part が 1 本無ければ、その part を名指しして FAIL する", async () => {
    await withTemporaryRoot(async (root) => {
      arrange(root);
      Deno.removeSync(new URL(parts[1].path, root));
      await assertRejects(
        () => assertDistributionIntact(root, "synthetic"),
        AssertionError,
        `depth[1] ${parts[1].path}: 宣言された part が無い`,
      );
    });
  });

  it("part の長さが宣言と違えば、実長と宣言長を並べて FAIL する", async () => {
    await withTemporaryRoot(async (root) => {
      arrange(root);
      Deno.writeFileSync(new URL(parts[1].path, root), new Uint8Array(parts[1].size - 1));
      await assertRejects(
        () => assertDistributionIntact(root, "synthetic"),
        AssertionError,
        `depth[1] ${parts[1].path}: 長さ 7 が宣言の 8 と違う`,
      );
    });
  });

  it("manifest が karume/5 でなければ FAIL する", async () => {
    await withTemporaryRoot(async (root) => {
      arrange(root, "karume/4");
      await assertRejects(() => assertDistributionIntact(root, "synthetic"), Error, "karume/4");
    });
  });
});
