// ADR 0005「全ケース SKIP は明示 FAIL」の**実資産**版の門番（GPU 版は tests/gpu_gate_test.ts）。
//
// 実重み e2e（birefnet / dacvae / deberta / depth_anything / embeddinggemma / gemma4 / irodori /
// minicpm5 / sbv2 / siglip2 / vowel_detector）は `ignore: !available || !GPU_AVAILABLE` で、
// `available` は系列 root の列挙が空なら偽になる。GPU はあるが実重み資産が 1 つも無い機
// （新しい作業機・worktree を別ホストへ持ち出した場合）では、これらが丸ごと SKIP したまま
// `deno task verify` が緑になる — 「検証していない」を「検証済み」と誤読させる無音の見かけ
// 成功なので、ここで 1 本だけ落とす。
//
// 見るのは**有無だけではない**（ADR 0108 段 3 検収②）。系列は `krm` コンテナ（`model.krm` の
// part 列）になったので、ディレクトリだけ在って中身が旧 shard のまま、という形が実在しうる。
// それは「資産が無い」とは別の壊れ方で、e2e 側の `modelPresent` が偽になって**無音 SKIP** に
// 化ける。もう 1 本の門番が系列ごとに (a) 容器が part 列として開けること（part の本数と
// 長さの突合まで `openContainer` が掛ける）と (b) 容器が名乗るグラフ名が期待の weights キー
// であることを見る。block の実バイトは読まない（§7 のハッシュ 3 分離 — 実バイトの突合は各
// e2e が通る）。
//
// グラフ名の表の正本は `packages/models/tests/helpers/series-graphs.ts` の 1 本で、**各 e2e も同じ 1 本から引く**。
// 門番と e2e が別々に表を持つと、片方だけ書き換えても誰も落ちない（門番は「期待どおり」と
// 言い、e2e は別のグラフを読む）。
//
// この門番自身は資産がある環境では**通る（緑の 1 件として見える）**。ignore にすると
// 「門番が効いているのか、門番ごと消えているのか」が区別できなくなるため（GPU 版と同じ理由）。
//
// NOTE: 見るのは実重み e2e 11 本が共有する `outputs/series/` の 1 根だけ。配布形ミラー
// （`models/`）は distribution_gate_test.ts の担当で、そちらの SKIP は本門番の射程外。
//
// ADR 0139 decision 5 moves this gate to the test-asset root `outputs/test-assets/` (the fetched
// cache of the pinned private dataset): it is red while no fetch completed there (no `.revision`)
// or the fetched revision differs from the pin (`TEST_ASSETS` in
// `@karume/test-support/test-assets`), with the same opt-out `KARUME_ALLOW_NO_ASSETS=1`. The series
// checks below stay until the families have moved off `outputs/series/` (ADR 0139 staging 4);
// they are retired with `series-graphs.ts` then.

import {
  isNotFound,
  makeTempDirSync,
  mkdirSync,
  removeSync,
  statSync,
  writeFileSync,
} from "@karume/test-support/fs";
import { assert, assertEquals, assertFalse, assertStringIncludes } from "@std/assert";
import {
  readFetchedRevision,
  REVISION_FILE,
  TEST_ASSETS,
  TEST_ASSETS_ROOT,
  testAssetsFetched,
  type TestAssetsPin,
} from "@karume/test-support/test-assets";
import { describe, it, test } from "vite-plus/test";
import { modelPresent, openSeriesContainer } from "../../runtime/tests/helpers/container-files.ts";
import { SERIES_GRAPHS } from "../../models/tests/helpers/series-graphs.ts";

/**
 * 「実資産無しでの全 SKIP」を明示的に許可する opt-out（`KARUME_ALLOW_NO_GPU` と同形）。
 *
 * MUST: 既定は fail loudly。models パッケージだけを触る作業機や資産を持たない CI で通すには、
 * 意図表明としてこの環境変数を要求する。
 */
const ALLOW_NO_ASSETS = process.env.KARUME_ALLOW_NO_ASSETS === "1";

/** 実重み e2e が資産を探す唯一の根（各 e2e の `SERIES_ROOT` / `SERIES_PARENT` の親）。 */
const SERIES_PARENT = new URL("../../../outputs/series/", import.meta.url);

/** 系列の容器の代表 path（実体は `model-NNNNN-of-NNNNN.krm` の part 列）。 */
const MODEL_FILE = "model.krm";

/** 系列 + 部品ディレクトリ（重みが根直下なら空文字）の容器の代表 path。 */
const representativeOf = (parent: URL, series: string, component: string): URL =>
  new URL(
    component === "" ? MODEL_FILE : `${component}/${MODEL_FILE}`,
    new URL(`${series}/`, parent),
  );

/**
 * 表（`SERIES_GRAPHS`）の系列の容器がひとつでも実在するか。
 *
 * MUST: 見るのは表の系列の容器だけで、述語は e2e が SKIP を決める `modelPresent` と同じ 1 本を
 * 使う。直下のディレクトリを数えると、表に無い系列（`anima-*` 等）や中身の無いディレクトリ
 * だけの機で門が緑になり、実重み e2e は全部 SKIP する。NotFound 以外の I/O 異常は
 * `modelPresent` が伝播させる — 権限エラー等を「資産が無い」と読み替えると、門番自身が環境の
 * 壊れを資産の不在として報告する。
 */
const seriesAssetsPresent = (parent: URL): boolean =>
  Object.entries(SERIES_GRAPHS).some(([series, components]) =>
    Object.keys(components).some((component) =>
      modelPresent(representativeOf(parent, series, component)),
    ),
  );

/** ディレクトリが在るか（不在は偽・それ以外の I/O 異常は伝播させる）。 */
const directoryPresent = (url: URL): boolean => {
  try {
    const stat = statSync(url);
    return stat.isDirectory;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

/** The command that fetches the pinned test assets (named in the gate's messages). */
const FETCH_COMMAND = "node tools/test-assets/fetch.ts";

/**
 * The verdict of the test-asset check (ADR 0139 decision 5) — a pure function of the revision the
 * last completed fetch recorded and the pin: `undefined` when they agree, otherwise why the gate is
 * red. An absent root reads as "no completed fetch" (`readFetchedRevision` returns `undefined`).
 *
 * MUST: a fetched revision that differs from the pin is red, not a warning — the goldens of one
 * revision are the expectations of the tests of that commit, and an older set would compare the
 * new tests against stale references (or skip on files the older set does not have).
 */
const testAssetsVerdict = (
  fetched: string | undefined,
  pin: TestAssetsPin,
  root: URL,
): string | undefined => {
  const remedy =
    `Run \`${FETCH_COMMAND}\` from the repository root (ADR 0005: a run whose real-weight tests ` +
    "lack their goldens is a FAIL). To pass deliberately on a machine without test assets, set " +
    "KARUME_ALLOW_NO_ASSETS=1.";
  if (fetched === undefined) {
    return (
      `${root.pathname} holds no completed fetch of ${pin.repo} (no ${REVISION_FILE}); the pinned ` +
      `revision is ${pin.revision}. ${remedy}`
    );
  }
  if (fetched !== pin.revision) {
    return (
      `${root.pathname} was fetched at revision ${fetched} of ${pin.repo}, but the pinned revision ` +
      `is ${pin.revision}. ${remedy}`
    );
  }
  return undefined;
};

test.skipIf(ALLOW_NO_ASSETS)(
  "assets gate: the test-asset root holds a completed fetch of the pinned revision (ADR 0139 decision 5)",
  () => {
    const verdict = testAssetsVerdict(readFetchedRevision(), TEST_ASSETS, TEST_ASSETS_ROOT);
    assert(verdict === undefined, verdict);
  },
);

test.skipIf(ALLOW_NO_ASSETS)(
  "資産門番: 実重み系列が 1 つも無い全 SKIP は明示 FAIL（ADR 0005）",
  () => {
    assert(
      seriesAssetsPresent(SERIES_PARENT),
      `${SERIES_PARENT.pathname} に表（packages/models/tests/helpers/series-graphs.ts）の系列の容器が 1 つも無く、` +
        "実重み e2e が全て SKIP された。" +
        "ADR 0005 によりこれは FAIL として扱う（リリース判定は実資産の golden 突合が必須）。" +
        "資産の無い環境で意図的に通すには KARUME_ALLOW_NO_ASSETS=1 を設定すること。",
    );
  },
);

test.skipIf(ALLOW_NO_ASSETS)(
  "資産門番: 在る系列の容器が開けて、期待の weights キーを名乗る（ADR 0108 段 3）",
  async () => {
    const broken: string[] = [];
    for (const [series, components] of Object.entries(SERIES_GRAPHS)) {
      const root = new URL(`${series}/`, SERIES_PARENT);
      if (!directoryPresent(root)) continue;
      for (const [component, expected] of Object.entries(components)) {
        const where = component === "" ? series : `${series}/${component}`;
        const representative = representativeOf(SERIES_PARENT, series, component);
        try {
          // part 列の解決・part の本数と長さの突合・2 文書の parse・束縛表との合流まで掛かる。
          const opened = await openSeriesContainer(representative);
          const declared = Object.keys(opened.graphs);
          if (declared.length !== 1 || declared[0] !== expected) {
            broken.push(`${where}: グラフ名が ${declared.join(" / ")}（期待は ${expected}）`);
          }
        } catch (cause) {
          broken.push(
            `${where}: 容器を開けない — ${cause instanceof Error ? cause.message : cause}`,
          );
        }
      }
    }
    assertEquals(
      broken,
      [],
      `${SERIES_PARENT.pathname} の系列が容器として読めない、または期待の weights キーを` +
        "名乗っていない。ディレクトリだけ在って中身が旧 shard のままだと、e2e は無音 SKIP する" +
        "（ADR 0005 によりこれは FAIL）。tools/export-recipes の各 export で焼き直すか、" +
        "退役した系列なら上の表から行を消すこと。",
    );
  },
);

describe("資産門番の判定（表の系列の容器が 1 つでも在るか）", () => {
  /** 一時ディレクトリを系列の根として判定を引き、後始末する。 */
  const judge = (arrange: (parent: URL) => void): boolean => {
    const parent = new URL(`file://${makeTempDirSync()}/`);
    try {
      arrange(parent);
      return seriesAssetsPresent(parent);
    } finally {
      removeSync(parent, { recursive: true });
    }
  };

  it("表に無い系列のディレクトリだけなら、容器が在っても資産無しと判定する", () => {
    assertFalse(
      judge((parent) => {
        const root = new URL("anima-f16/", parent);
        mkdirSync(root);
        writeFileSync(new URL("model-00001-of-00001.krm", root), new Uint8Array());
      }),
    );
  });

  it("表の系列のディレクトリが在っても、容器が無ければ資産無しと判定する", () => {
    assertFalse(judge((parent) => mkdirSync(new URL("birefnet-hr-1024-f16/", parent))));
  });

  it("表の系列に容器の part が在れば資産ありと判定する", () => {
    assert(
      judge((parent) => {
        const root = new URL("birefnet-hr-1024-f16/", parent);
        mkdirSync(root);
        writeFileSync(new URL("model-00001-of-00001.krm", root), new Uint8Array());
      }),
    );
  });

  it("表の系列の部品ディレクトリに容器が在れば資産ありと判定する", () => {
    assert(
      judge((parent) => {
        const component = new URL("dacvae-32dim/decoder/", parent);
        mkdirSync(component, { recursive: true });
        writeFileSync(new URL("model-00001-of-00001.krm", component), new Uint8Array());
      }),
    );
  });
});

describe("assets gate verdict (the fetched revision of the test-asset root against the pin)", () => {
  const pin: TestAssetsPin = {
    repo: "hdae/karume-test-assets",
    revision: "0123456789abcdef0123456789abcdef01234567",
  };

  /**
   * Reads the verdict for the root `<temporary dir>/<child>` arranged by `arrange`, then removes the
   * temporary directory.
   */
  const verdictOf = (arrange: (root: URL) => void, child = ""): string | undefined => {
    const dir = new URL(`file://${makeTempDirSync({ prefix: "karume-test-assets-" })}/`);
    const root = new URL(child, dir);
    try {
      arrange(root);
      return testAssetsVerdict(readFetchedRevision(root), pin, root);
    } finally {
      removeSync(dir, { recursive: true });
    }
  };

  it("is red when no fetch completed (no .revision), naming the pin and the fetch command", () => {
    const verdict = verdictOf(() => {});
    assert(verdict !== undefined);
    assertStringIncludes(verdict, `no ${REVISION_FILE}`);
    assertStringIncludes(verdict, pin.revision);
    assertStringIncludes(verdict, FETCH_COMMAND);
  });

  it("is red when the root itself is absent", () => {
    const verdict = verdictOf(() => {}, "missing/");
    assert(verdict !== undefined);
    assertStringIncludes(verdict, `no ${REVISION_FILE}`);
  });

  it("is red when the fetched revision differs from the pin, naming both", () => {
    const stale = "f".repeat(40);
    const verdict = verdictOf((root) =>
      writeFileSync(new URL(REVISION_FILE, root), new TextEncoder().encode(`${stale}\n`)),
    );
    assert(verdict !== undefined);
    assertStringIncludes(verdict, `fetched at revision ${stale}`);
    assertStringIncludes(verdict, `pinned revision is ${pin.revision}`);
    assertStringIncludes(verdict, FETCH_COMMAND);
  });

  it("is green when the fetched revision is the pin (the trailing newline is not part of it)", () => {
    assertEquals(
      verdictOf((root) =>
        writeFileSync(new URL(REVISION_FILE, root), new TextEncoder().encode(`${pin.revision}\n`)),
      ),
      undefined,
    );
  });
});

describe("testAssetsFetched (a complete fetch of the pin, which makes a missing asset a FAIL)", () => {
  const pin: TestAssetsPin = {
    repo: "hdae/karume-test-assets",
    revision: "0123456789abcdef0123456789abcdef01234567",
  };

  const fetchedWith = (record: string | undefined): boolean => {
    const root = new URL(`file://${makeTempDirSync({ prefix: "karume-test-assets-" })}/`);
    try {
      if (record !== undefined) {
        writeFileSync(new URL(REVISION_FILE, root), new TextEncoder().encode(record));
      }
      return testAssetsFetched(root, pin);
    } finally {
      removeSync(root, { recursive: true });
    }
  };

  it("is true only when the record is the pin", () => {
    assert(fetchedWith(`${pin.revision}\n`));
    assertFalse(fetchedWith(`${"f".repeat(40)}\n`));
  });

  it("is false when no complete fetch left a record (none, or a partial fetch)", () => {
    assertFalse(fetchedWith(undefined));
  });
});
