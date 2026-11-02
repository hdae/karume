// ADR 0005「全ケース SKIP は明示 FAIL」の**配布形ミラー**版の門番（実重み系列版は
// tests/e2e_assets_gate_test.ts・GPU 版は tests/gpu_gate_test.ts）。
//
// 配布形ミラー（`models/karume-<family>/`）を根にする検査 — anima / sbv2 / irodori / gemma4 /
// QAT / wan の実資産 e2e（packages/models/tests/e2e_*_test.ts と helpers/irodori-assets.ts）・融合ヒット数の
// 配布形節（anima / anima-extra / irodori / gemma4 / gemma4-qat / wan — tests/e2e_*_fusion_counts_test.ts）
// — は、いずれも `karume.json` の有無で `ignore` し `console.warn` だけを残して緑になる。
// `outputs/series/` だけ持つ機（新しい作業機・worktree を別ホストへ持ち出した場合）では
// e2e_assets_gate_test.ts が緑のまま実資産検査が丸ごと消えるので、ここで 1 本落とす。
//
// 欠如の検出（ミラーごと無い形）の射程は**公開済みの全ミラー**（2026-09-25 裁定）。e2e がまだ読まないミラー（siglip2 など）も
// 載せる — 門番の射程を「今 e2e が読むもの」に合わせると、e2e を足した日に門番の更新を忘れて
// 無音 SKIP が戻る。未公開の vowel-detector は載せない（配布形を作ってから載せる）。未公開でも
// e2e（と融合ヒット数の門）が根にするミラー（wan2.1）は載せる。
//
// 見るのは**有無だけではない**（ADR 0108 段 3 検収②）。`karume.json` はあるが中身が旧 major の
// まま、あるいは宣言された part が 1 本足りない・長さが宣言と違う、という形は上の e2e を
// 「開けない資産」で落とすか、悪くすると**前回の書き出しの残骸**を今回の期待値で読ませる。
// どちらも「資産が無い」とは別の壊れ方なので、門番が manifest を parse して part の実在と
// 長さまで突き合わせる。中身（block の sha256）は読まない — 実バイトの突合は各 e2e が
// `openContainer` の経路で必ず通る（§7 のハッシュ 3 分離）。
//
// Since ADR 0139 decision 5 the check covers every model × quant × weights key a manifest
// declares, not only the default selection, and it also opens each container's two documents
// through the manifest and checks that its graph is named by the weights key — the graph-name
// check that the assets gate ran on the series (`series-graphs.ts`) moves here.
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
  type DirEntry,
  isFilesystemLoop,
  isNotFound,
  makeTempDirSync,
  mkdirSync,
  readDirSync,
  readTextFile,
  removeSync,
  statSync,
  symlinkSync,
  writeFileSync,
  writeTextFileSync,
} from "@karume/test-support/fs";
import {
  assert,
  assertEquals,
  AssertionError,
  assertRejects,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import { describe, it, test } from "vite-plus/test";
import {
  type ContainerRef,
  loadManifest,
  type Manifest,
  type ModelEntry,
  openContainerSource,
  parseManifest,
} from "@karume/hub";
import { openContainer, parseIrDeclarationValue } from "../mod.ts";
import { type WrittenContainer, writeModelContainer } from "./helpers/container-write.ts";
import { mirrorDirectory } from "../../models/tests/helpers/mirror.ts";

/**
 * 「配布形ミラー無しでの全 SKIP」を明示的に許可する opt-out（`KARUME_ALLOW_NO_ASSETS` と同形）。
 *
 * MUST: 既定は fail loudly。配布形を持たない作業機や CI で通すには、意図表明としてこの環境変数を
 * 要求する（実重み系列側の opt-out とは別に持つ — 片方だけ持つ機が実在する）。
 */
const ALLOW_NO_DISTRIBUTION = process.env.KARUME_ALLOW_NO_DISTRIBUTION === "1";

/** 配布形ミラーを置く根（上記の検査が `karume.json` を読む先と同じ URL の組み立て方）。 */
const MODELS_ROOT = new URL("../../../models/", import.meta.url);

/**
 * The distribution mirrors the guard **requires to exist** (the roster of the absence check — the
 * reach of the manifest checks is derived from the layout by {@link discoverDistributions}) = the
 * 14 published repositories (including `karume-umt5-xxl` and `karume-wan2.2`, first published in
 * 0.14.0, and `karume-moss-image` and `karume-gemma4-qat`, first published in 0.15.1) + the
 * unpublished `karume-wan2.1`.
 *
 * 公開済みの 2 本は公開前から門番に載っていた: `karume-umt5-xxl` は Wan の GPU 経路の e2e が越境参照の根にし
 * （ADR 0119 段 10d・無いと欠如が無音の SKIP になる）、`karume-wan2.2` は Wan2.2 の通しの e2e
 * （`e2e_wan_ti2v_pipeline_test.ts`）と DiT の e2e の実用席の記録（`e2e_wan_ti2v_dit_test.ts` — 席の束を manifest から
 * 読む）が根にする（ADR 0121 段 8）。
 *
 * `karume-gemma4-qat` was on the roster long before its publication: the QAT public-entry e2e and
 * the fusion hit-count check root on it, so it was listed before the guard's reach was widened
 * (dropping it would have loosened an existing check).
 *
 * `karume-wan2.1` は未公開
 * （HF に公開しない — ADR 0118 追記（2026-10-15））だが、Wan の通しの e2e（`e2e_wan_pipeline_test.ts`）と
 * 融合ヒット数の門が根にする（ADR 0118 段 7）。
 *
 * `karume-moss-image` was on the roster before its publication as well (ADR 0138 addendum of
 * 2026-11-01): the pipeline e2e (`packages/e2e/tests/e2e_moss_pipeline_test.ts`) roots on it, so its
 * absence must not be a silent skip.
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
  "karume-moss-image",
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
  let entries: DirEntry[];
  try {
    entries = [...readDirSync(models)];
  } catch (cause) {
    if (isNotFound(cause)) return [];
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
    return statSync(child).isDirectory;
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
    return statSync(new URL("karume.json", root)).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
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
    const stat = statSync(url);
    return stat.isFile ? stat.size : undefined;
  } catch (cause) {
    if (isNotFound(cause)) return undefined;
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
 * Asserts that the mirror at `root` has a `karume/5` manifest and that **every container it
 * declares** — every model × weights key × dtype label, which covers every quant's selection —
 * is intact (ADR 0139 decision 5):
 *
 * - each part is present with its declared size (cross-repo parts are skipped: they live in their
 *   own mirror, which this gate checks under its own name — ADR 0038 §7);
 * - the container opens through the manifest (header, the two documents checked against the
 *   declared length and sha256, the declared part lengths) and its one graph is named by the
 *   weights key (container-v1 §2.1). A cross-repo container is opened from its sibling mirror.
 *
 * Block bytes are never read (the block sha256 is checked by the tests that build Sessions — §7
 * keeps the three hashes apart), so the cost is one stat per part plus part 0's documents per
 * container: about two hundred containers over the fifteen mirrors open in about a second on the
 * development machine (2026-11-02), GPU-free.
 */
const assertDistributionIntact = async (root: URL, series: string): Promise<void> => {
  // MUST: 名簿に無いミラーの欠如はここでしか見えない — `models/` は配布形だけの根なので、
  // `karume.json` の無い子ディレクトリは焼きかけか前回の残骸で、e2e が読めば別の壊れ方をする。
  assert(
    manifestPresent(root),
    `models/${series}/ に karume.json が無い。models/ は配布形だけを置く根なので、` +
      "これは焼きかけか残骸のディレクトリ。tools/export-recipes の dist.py で作り直すか、取り除くこと。",
  );
  const manifest = parseManifest(await readTextFile(new URL("karume.json", root)));
  // parse 自身も major を見るが、`karume/5` を**名指しで**断言しておく — hub が受ける major を
  // 増やした日に、この門番が「旧 major のミラーを据えたまま」を素通しするのを防ぐ。
  assertEquals(manifest.format, MANIFEST_FORMAT, `${series}: manifest の format`);
  const broken: string[] = [];
  const intact: {
    readonly where: string;
    readonly key: string;
    readonly container: ContainerRef;
  }[] = [];
  for (const [model, entry] of Object.entries(manifest.models)) {
    for (const [key, dtypes] of Object.entries(entry.weights)) {
      for (const [dtype, { container }] of Object.entries(dtypes)) {
        const where = `${model}/${key}/${dtype}`;
        const missing: string[] = [];
        for (const [index, part] of container.parts.entries()) {
          if (part.repo !== undefined) continue;
          const actual = fileBytes(new URL(part.path, root));
          if (actual === undefined) {
            missing.push(`${where}[${index}] ${part.path}: the declared part is absent`);
          } else if (actual !== part.size) {
            missing.push(`${where}[${index}] ${part.path}: ${actual} bytes, declared ${part.size}`);
          }
        }
        broken.push(...missing);
        // A container with a missing or short part cannot be opened; it is reported once, above.
        if (missing.length === 0) intact.push({ where, key, container });
      }
    }
  }
  if (intact.length > 0) {
    const loaded = await loadManifest(mirrorDirectory(root));
    for (const { where, key, container } of intact) {
      try {
        const opened = await openContainer(
          { kind: "source", source: openContainerSource(loaded, container) },
          container.descriptor,
        );
        const graphs = Object.keys(opened.graphs);
        if (graphs.length !== 1 || graphs[0] !== key) {
          broken.push(
            `${where}: the container's graphs are ${graphs.join(" / ")}, expected ${key}`,
          );
        }
      } catch (cause) {
        broken.push(
          `${where}: the container does not open — ${cause instanceof Error ? cause.message : cause}`,
        );
      }
    }
  }
  assertEquals(
    broken,
    [],
    `models/${series}/ does not match its manifest. A tool or test reading it would hit a ` +
      "half-built or stale container, so ADR 0005 counts this as a FAIL. Rebuild the mirror with " +
      "tools/export-recipes dist.py.",
  );
};

/** label / description が在るか（空白だけは無いと数える — モデルカードの表示と同じ扱い）。 */
const hasText = (value: string | undefined): boolean => (value ?? "").trim() !== "";

/** 2 つの weights 写像が同じか（同じ部品の集合に、部品ごとに同じ dtype ラベル）。 */
const sameWeights = (
  left: Readonly<Record<string, string>>,
  right: Readonly<Record<string, string>>,
): boolean => {
  const roles = Object.keys(left);
  return (
    roles.length === Object.keys(right).length &&
    roles.every((role) => Object.hasOwn(right, role) && right[role] === left[role])
  );
};

/**
 * 1 モデルの既定プリセットとプリセット一覧が、配布形の満たすべき 3 制約を破っている箇所を返す
 * （違反ごとに 1 文・プリセットを名指し。満たしていれば空）。DECIDED: ADR 0128 決定 2。
 *
 * - C1: 既定プリセットは素の WebGPU で走る — `quants[defaultQuant].gpuFeatures` が無いか、
 *   true のメンバーを持たない（今日の語彙では `shaderF16` を要求しない）。
 * - C2: `linearCompute` か `attentionCompute` が `"a8"` のプリセットには、どちらも `"a8"` でなく
 *   weights の写像が同一（同じ部品・同じラベル = 同じコンテナ）の別名の兄弟プリセットがある。
 *   この兄弟が、dp4a の無い device で runtime が出す警告
 *   （packages/runtime/src/runtime/a8-emulation-warning.ts）の指す逃げ先で、追加の取得が要らない。
 * - C3: プリセットが 2 つ以上あるモデルでは、全プリセットが trim して空でない label と description を
 *   持つ（選択 UI とモデルカードが人に見せる面）。プリセットが 1 つだけのモデルは選ぶものが無いので除く。
 */
const defaultPresetViolations = (entry: ModelEntry): string[] => {
  const violations: string[] = [];
  const requested = Object.entries(entry.quants[entry.defaultQuant]?.gpuFeatures ?? {})
    .filter(([, value]) => value === true)
    .map(([feature]) => feature);
  if (requested.length > 0) {
    violations.push(
      `C1: 既定プリセット '${entry.defaultQuant}' が GPU feature を要求している` +
        `（${requested.join(", ")}）— 既定は素の WebGPU で走らなければならない`,
    );
  }
  const presets = Object.entries(entry.quants);
  for (const [name, quant] of presets) {
    const knobs = (["linearCompute", "attentionCompute"] as const)
      .filter((knob) => quant.session[knob] === "a8")
      .map((knob) => `${knob}: "a8"`);
    if (knobs.length === 0) continue;
    const sibling = presets.some(
      ([other, candidate]) =>
        other !== name &&
        candidate.session.linearCompute !== "a8" &&
        candidate.session.attentionCompute !== "a8" &&
        sameWeights(quant.weights, candidate.weights),
    );
    if (!sibling) {
      violations.push(
        `C2: プリセット '${name}' は int8 活性（${knobs.join(", ")}）を持つが、` +
          "同じ weights で活性量子化を持たない兄弟プリセットが無い",
      );
    }
  }
  if (presets.length >= 2) {
    for (const [name, quant] of presets) {
      for (const field of ["label", "description"] as const) {
        if (!hasText(quant[field])) {
          violations.push(`C3: プリセット '${name}' に ${field} が無い（空白だけも無いと数える）`);
        }
      }
    }
  }
  return violations;
};

/** manifest の全モデルが ADR 0128 決定 2 の制約（C1〜C3）を満たすことを断言する（系列とモデルを名指し）。 */
const assertDefaultPresetPolicy = (manifest: Manifest, series: string): void => {
  for (const [model, entry] of Object.entries(manifest.models)) {
    assertEquals(
      defaultPresetViolations(entry),
      [],
      `models/${series}/ の model '${model}' の既定プリセットとプリセット一覧が ` +
        "ADR 0128 決定 2 の制約を破っている。karume dist の recipe（tools/export-recipes）の" +
        "プリセットの宣言を直して焼き直すこと。",
    );
  }
};

test.skipIf(ALLOW_NO_DISTRIBUTION)(
  "配布門番: 配布形ミラーが欠けた全 SKIP は明示 FAIL（ADR 0005）",
  () => assertNoDistributionMissing(MODELS_ROOT, DISTRIBUTIONS),
);

for (const series of discoverDistributions(MODELS_ROOT)) {
  const root = distributionRoot(MODELS_ROOT, series);
  test.skipIf(REQUIRED.has(series) && manifestKnownAbsent(root))(
    `distribution gate: ${series} parses as karume/5 and every declared container is intact (parts, documents, graph name)`,
    () => assertDistributionIntact(root, series),
  );
  test.skipIf(REQUIRED.has(series) && manifestKnownAbsent(root))(
    `配布門番: ${series} の既定プリセットとプリセット一覧が ADR 0128 決定 2 の制約（C1〜C3）を満たす`,
    async () => {
      const manifest = parseManifest(await readTextFile(new URL("karume.json", root)));
      assertDefaultPresetPolicy(manifest, series);
    },
  );
}

// 以下は門番の判定そのものの故障注入。一時ディレクトリだけで完結する（実資産も GPU も要らない）
// ので、opt-out の有無に関わらず走る — 門番が ignore される CI でも判定の退行は見える。

/** 一時ディレクトリを根として `run` を引き、後始末する。 */
const withTemporaryRoot = async (run: (root: URL) => Promise<void> | void): Promise<void> => {
  const root = new URL(`file://${makeTempDirSync({ prefix: "karume-distribution-" })}/`);
  try {
    await run(root);
  } finally {
    removeSync(root, { recursive: true });
  }
};

describe("配布門番の判定（表のミラーの karume.json が揃っているか）", () => {
  /** 表の全ミラーに（中身を問わない）`karume.json` を置いた根を作る。 */
  const arrangeAll = (models: URL): void => {
    for (const series of DISTRIBUTIONS) {
      const root = distributionRoot(models, series);
      mkdirSync(root, { recursive: true });
      writeTextFileSync(new URL("karume.json", root), "{}");
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
        removeSync(manifest);
        const error = assertThrows(
          () => assertNoDistributionMissing(models, DISTRIBUTIONS),
          AssertionError,
        );
        assertStringIncludes(error.message, `無い系列がある: ${series}。`);
        writeTextFileSync(manifest, "{}");
      }
    });
  });

  it("karume.json がディレクトリならミラーは無いと判定する", async () => {
    await withTemporaryRoot((models) => {
      arrangeAll(models);
      const manifest = new URL("karume.json", distributionRoot(models, "karume-lucida"));
      removeSync(manifest);
      mkdirSync(manifest);
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
        mkdirSync(new URL(`${name}/`, models));
      }
      writeTextFileSync(new URL("README.md", models), "");
      assertEquals(discoverDistributions(models), ["karume-a", "karume-b"]);
    });
  });

  it("名簿に無い新しいミラーも拾う（名簿の更新忘れで manifest の検査から漏れない）", async () => {
    await withTemporaryRoot((models) => {
      const unlisted = "karume-vowel-detector";
      assert(!REQUIRED.has(unlisted), "被験体の前提: 名簿に無い名前");
      mkdirSync(new URL(`${unlisted}/`, models));
      assertEquals(discoverDistributions(models), [unlisted]);
    });
  });

  it("ディレクトリを指すシンボリックリンクもミラーとして拾う", async () => {
    await withTemporaryRoot((root) => {
      const models = new URL("models/", root);
      const target = new URL("elsewhere/", root);
      mkdirSync(models);
      mkdirSync(target);
      symlinkSync(target.pathname, new URL("karume-linked", models).pathname);
      assertEquals(discoverDistributions(models), ["karume-linked"]);
    });
  });

  it("壊れたリンクと辿れないリンクも投げずに拾う（登録時に落ちてファイルごと消えない）", async () => {
    await withTemporaryRoot((models) => {
      symlinkSync(new URL("nowhere", models).pathname, new URL("karume-broken", models).pathname);
      symlinkSync("karume-loop", new URL("karume-loop", models).pathname);
      assertEquals(discoverDistributions(models), ["karume-broken", "karume-loop"]);
    });
  });

  it("壊れたリンクのミラーは、そのディレクトリを名指しして FAIL する", async () => {
    await withTemporaryRoot(async (models) => {
      symlinkSync(new URL("nowhere", models).pathname, new URL("karume-broken", models).pathname);
      await assertRejects(
        () => assertDistributionIntact(distributionRoot(models, "karume-broken"), "karume-broken"),
        AssertionError,
        "models/karume-broken/ に karume.json が無い",
      );
    });
  });

  it("辿れないリンクは ignore の判定で投げずに test を走らせ、test の中でそのパスを名指しして投げる", async () => {
    await withTemporaryRoot(async (models) => {
      symlinkSync("karume-loop", new URL("karume-loop", models).pathname);
      const root = distributionRoot(models, "karume-loop");
      assertEquals(manifestKnownAbsent(root), false);
      const error = await assertRejects(
        () => assertDistributionIntact(root, "karume-loop"),
        Error,
        "karume-loop/karume.json",
      );
      assert(isFilesystemLoop(error), `expected ELOOP, got ${String(error)}`);
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

describe("distribution gate verdict (every declared container against the mirror's files)", () => {
  const revision = "0123456789abcdef0123456789abcdef01234567";

  /** One real container: the graph `graph` (`linear` with one f32 weight), split into small parts. */
  const writeContainer = (graph: string): Promise<WrittenContainer> =>
    writeModelContainer(
      {
        graphs: {
          [graph]: parseIrDeclarationValue({
            format: "karume-ir",
            version: 2,
            requires: { ops: ["linear"] },
            symbols: [],
            inputs: [{ name: "x", dtype: "f32", shape: [2, 2] }],
            outputs: ["y"],
            initializers: { w: {} },
            values: { w: { dtype: "f32", shape: [2, 2] }, y: { dtype: "f32", shape: [2, 2] } },
            states: {},
            nodes: [{ op: "linear", ins: ["x", "w"], outs: ["y"], attrs: {} }],
          }),
        },
        consts: [],
        weights: [
          {
            graph,
            initializer: "w",
            bytes: new Uint8Array(Float32Array.from([1, 2, 3, 4]).buffer),
            encoding: { codec: "f32" },
          },
        ],
        assets: [],
        provenance: { license: "test" },
      },
      { partBytes: 4096, blockBytes: 512 },
    );

  const hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
    [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("");

  /** A container written to disk under `stem` and the manifest entry that declares it. */
  type Placed = {
    readonly paths: readonly string[];
    readonly entry: (options?: { readonly repo?: string }) => Record<string, unknown>;
  };

  /** Writes `written` under `root` as `<stem>-NNNNN-of-NNNNN.krm` (zero-length parts too). */
  const place = async (root: URL, stem: string, written: WrittenContainer): Promise<Placed> => {
    const total = String(written.parts.length).padStart(5, "0");
    const paths = written.parts.map(
      (_, index) => `${stem}-${String(index + 1).padStart(5, "0")}-of-${total}.krm`,
    );
    mkdirSync(new URL(stem.slice(0, stem.lastIndexOf("/") + 1), root), { recursive: true });
    const parts = await Promise.all(
      written.parts.map(async (bytes, index) => {
        writeFileSync(new URL(paths[index]!, root), bytes);
        return { path: paths[index]!, size: bytes.byteLength, sha256: await hex(bytes) };
      }),
    );
    const descriptor = {
      graph: {
        length: written.graphDescriptorBytes.byteLength,
        sha256: await hex(written.graphDescriptorBytes),
      },
      model: {
        length: written.modelDescriptorBytes.byteLength,
        sha256: await hex(written.modelDescriptorBytes),
      },
    };
    return {
      paths,
      entry: ({ repo } = {}) => ({
        container: {
          descriptor,
          parts: repo === undefined ? parts : parts.map((part) => ({ ...part, repo, revision })),
        },
      }),
    };
  };

  /** The manifest of one model `m` whose component `depth` has the quants f32 (default) and i8. */
  const manifestText = (
    f32: Record<string, unknown>,
    i8: Record<string, unknown>,
    format = MANIFEST_FORMAT,
  ): string =>
    JSON.stringify({
      format,
      generator: "karume/0.15.1",
      defaultModel: "m",
      models: {
        m: {
          pipeline: "depth-anything/1",
          weights: { depth: { f32, i8 } },
          assets: {},
          quants: {
            f32: { weights: { depth: "f32" }, session: {}, label: "F32", description: "Full." },
            i8: { weights: { depth: "i8" }, session: {}, label: "I8", description: "Small." },
          },
          defaultQuant: "f32",
          pipelineConfig: {},
        },
      },
    });

  /** An intact mirror; the i8 container's graph is `i8Graph` (the weights key when omitted). */
  const arrange = async (
    root: URL,
    options: { readonly format?: string; readonly i8Graph?: string } = {},
  ): Promise<{ readonly f32: Placed; readonly i8: Placed }> => {
    const f32 = await place(root, "m/depth/model.f32", await writeContainer("depth"));
    const i8 = await place(
      root,
      "m/depth/model.i8",
      await writeContainer(options.i8Graph ?? "depth"),
    );
    writeTextFileSync(
      new URL("karume.json", root),
      manifestText(f32.entry(), i8.entry(), options.format),
    );
    return { f32, i8 };
  };

  it("an intact mirror passes", async () => {
    await withTemporaryRoot(async (root) => {
      await arrange(root);
      await assertDistributionIntact(root, "synthetic");
    });
  });

  it("a part of the default quant is absent: FAIL naming the container and the part", async () => {
    await withTemporaryRoot(async (root) => {
      const { f32 } = await arrange(root);
      const last = f32.paths.length - 1;
      removeSync(new URL(f32.paths[last]!, root));
      await assertRejects(
        () => assertDistributionIntact(root, "synthetic"),
        AssertionError,
        `m/depth/f32[${last}] ${f32.paths[last]}: the declared part is absent`,
      );
    });
  });

  it("a part of a non-default quant is absent: FAIL too (every quant is checked, not only the default)", async () => {
    await withTemporaryRoot(async (root) => {
      const { i8 } = await arrange(root);
      const last = i8.paths.length - 1;
      removeSync(new URL(i8.paths[last]!, root));
      await assertRejects(
        () => assertDistributionIntact(root, "synthetic"),
        AssertionError,
        `m/depth/i8[${last}] ${i8.paths[last]}: the declared part is absent`,
      );
    });
  });

  it("a part's length differs from its declaration: FAIL with both lengths", async () => {
    await withTemporaryRoot(async (root) => {
      const { f32 } = await arrange(root);
      const last = f32.paths.length - 1;
      const at = new URL(f32.paths[last]!, root);
      const size = statSync(at).size;
      writeFileSync(at, new Uint8Array(size - 1));
      await assertRejects(
        () => assertDistributionIntact(root, "synthetic"),
        AssertionError,
        `m/depth/f32[${last}] ${f32.paths[last]}: ${size - 1} bytes, declared ${size}`,
      );
    });
  });

  it("the container's graph is not named by the weights key: FAIL naming both", async () => {
    await withTemporaryRoot(async (root) => {
      await arrange(root, { i8Graph: "other" });
      await assertRejects(
        () => assertDistributionIntact(root, "synthetic"),
        AssertionError,
        "m/depth/i8: the container's graphs are other, expected depth",
      );
    });
  });

  it("part 0's documents differ from the declared sha256: FAIL, the container does not open", async () => {
    await withTemporaryRoot(async (root) => {
      const { f32 } = await arrange(root);
      // Another container of the same shape (a graph name of the same length) over the declared
      // part 0: the part lengths agree, the documents' sha256 does not.
      const other = await writeContainer("deptx");
      writeFileSync(new URL(f32.paths[0]!, root), other.parts[0]!);
      await assertRejects(
        () => assertDistributionIntact(root, "synthetic"),
        AssertionError,
        "m/depth/f32: the container does not open",
      );
    });
  });

  it("the manifest is not karume/5: FAIL", async () => {
    await withTemporaryRoot(async (root) => {
      await arrange(root, { format: "karume/4" });
      await assertRejects(() => assertDistributionIntact(root, "synthetic"), Error, "karume/4");
    });
  });

  it("a cross-repo container is opened from its sibling mirror (its parts are not stat'ed here)", async () => {
    await withTemporaryRoot(async (models) => {
      const base = new URL("karume-base/", models);
      const derived = new URL("karume-derived/", models);
      mkdirSync(base);
      mkdirSync(derived);
      const { f32, i8 } = await arrange(base);
      const repo = "hdae/karume-base";
      writeTextFileSync(
        new URL("karume.json", derived),
        manifestText(f32.entry({ repo }), i8.entry({ repo })),
      );
      await assertDistributionIntact(derived, "karume-derived");
      removeSync(base, { recursive: true });
      await assertRejects(
        () => assertDistributionIntact(derived, "karume-derived"),
        Error,
        "'hdae/karume-base'",
      );
    });
  });
});

describe("配布門番の判定（既定プリセットとプリセット一覧 — ADR 0128 決定 2）", () => {
  const sha256 = "0".repeat(64);
  /** manifest に書くプリセットの宣言（JSON のまま — 型と値の検査は hub の parser に任せる）。 */
  type FabricatedQuant = {
    readonly weights: Readonly<Record<string, string>>;
    readonly [key: string]: unknown;
  };
  const labelled = { label: "Preset", description: "A preset for the test." };
  /** 素の WebGPU で走る、活性量子化を持たない i8 のプリセット（C2 の兄弟の雛形）。 */
  const plainI8 = { weights: { text: "i8", dit: "i8" }, session: {}, ...labelled };
  /** i8 の重みに int8 活性を載せたプリセット。 */
  const a8I8 = {
    weights: { text: "i8", dit: "i8" },
    session: { linearCompute: "a8", attentionCompute: "a8" },
    ...labelled,
  };

  /**
   * `quants` を宣言した 1 モデルの manifest を hub の parser に通し、そのモデルの宣言を返す
   * （weights の容器は quants が参照する部品 × ラベルから組む — parser の完全写像の検査を通すため）。
   */
  const entryOf = (
    quants: Readonly<Record<string, FabricatedQuant>>,
    defaultQuant: string,
  ): ModelEntry => {
    const weights: Record<string, Record<string, unknown>> = {};
    for (const quant of Object.values(quants)) {
      for (const [role, label] of Object.entries(quant.weights)) {
        weights[role] = {
          ...weights[role],
          [label]: {
            container: {
              descriptor: { graph: { length: 10, sha256 }, model: { length: 5, sha256 } },
              parts: [
                { path: `m/${role}/model.${label}-00001-of-00002.krm`, size: 39, sha256 },
                { path: `m/${role}/model.${label}-00002-of-00002.krm`, size: 8, sha256 },
              ],
            },
          },
        };
      }
    }
    const manifest = parseManifest(
      JSON.stringify({
        format: MANIFEST_FORMAT,
        generator: "karume/0.15.0",
        defaultModel: "m",
        models: {
          m: {
            pipeline: "anima/1",
            weights,
            assets: {},
            quants,
            defaultQuant,
            pipelineConfig: {},
          },
        },
      }),
    );
    return manifest.models["m"]!;
  };

  /** 違反が 1 件だけで、その文がプリセットを名指しし、制約の記号を含むことを断言する。 */
  const assertSingleViolation = (entry: ModelEntry, constraint: string, preset: string): void => {
    const violations = defaultPresetViolations(entry);
    assertEquals(violations.length, 1, violations.join("\n"));
    assertStringIncludes(violations[0]!, `${constraint}:`);
    assertStringIncludes(violations[0]!, `'${preset}'`);
  };

  it("label と description を持つ 2 プリセットで、a8 に兄弟があれば違反は無い", () => {
    assertEquals(defaultPresetViolations(entryOf({ a8: a8I8, i8: plainI8 }, "a8")), []);
  });

  it("既定が shaderF16 を要求すれば C1 違反として既定プリセットを名指しする", () => {
    const f16 = {
      weights: { text: "f16", dit: "f16" },
      session: { linearCompute: "f16" },
      gpuFeatures: { shaderF16: true },
      ...labelled,
    };
    const entry = entryOf({ f16, i8: plainI8 }, "f16");
    assertSingleViolation(entry, "C1", "f16");
    assertStringIncludes(defaultPresetViolations(entry)[0]!, "shaderF16");
  });

  it("既定でないプリセットの shaderF16 は C1 違反にしない", () => {
    const f16 = {
      weights: { text: "f16", dit: "f16" },
      session: {},
      gpuFeatures: { shaderF16: true },
      ...labelled,
    };
    assertEquals(defaultPresetViolations(entryOf({ f16, i8: plainI8 }, "i8")), []);
  });

  it("a8 のプリセットに兄弟が無ければ C2 違反として、そのプリセットと a8 のノブを名指しする", () => {
    const f32 = { weights: { text: "f32", dit: "f32" }, session: {}, ...labelled };
    const entry = entryOf({ a8: a8I8, f32 }, "a8");
    assertSingleViolation(entry, "C2", "a8");
    const [violation] = defaultPresetViolations(entry);
    assertStringIncludes(violation!, `linearCompute: "a8"`);
    assertStringIncludes(violation!, `attentionCompute: "a8"`);
  });

  it("兄弟の候補の weights がラベル 1 つでも違えば C2 違反のまま（追加の取得が要る）", () => {
    const nearly = { weights: { text: "f32", dit: "i8" }, session: {}, ...labelled };
    assertSingleViolation(entryOf({ a8: a8I8, nearly }, "a8"), "C2", "a8");
  });

  it("同じ weights でも、兄弟の候補が片方の a8 を残していれば C2 違反のまま", () => {
    const attentionOnly = { ...a8I8, session: { attentionCompute: "a8" } };
    const entry = entryOf({ a8: a8I8, attentionOnly }, "a8");
    const violations = defaultPresetViolations(entry);
    assertEquals(violations.length, 2, violations.join("\n"));
    assertStringIncludes(violations[0]!, "C2: プリセット 'a8'");
    assertStringIncludes(violations[1]!, "C2: プリセット 'attentionOnly'");
  });

  it("description の無いプリセットは C3 違反として、そのプリセットを名指しする", () => {
    const withoutDescription = { weights: plainI8.weights, session: {}, label: "Preset" };
    const entry = entryOf({ a8: a8I8, i8: withoutDescription }, "a8");
    assertSingleViolation(entry, "C3", "i8");
    assertStringIncludes(defaultPresetViolations(entry)[0]!, "description");
  });

  it("空白だけの label は無いと数え、C3 違反として、そのプリセットを名指しする", () => {
    const entry = entryOf({ a8: { ...a8I8, label: " \t " }, i8: plainI8 }, "a8");
    assertSingleViolation(entry, "C3", "a8");
    assertStringIncludes(defaultPresetViolations(entry)[0]!, "label");
  });

  it("プリセットが 1 つだけのモデルは label と description が無くても C3 の対象外", () => {
    const f32 = { weights: { text: "f32", dit: "f32" }, session: {} };
    assertEquals(defaultPresetViolations(entryOf({ f32 }, "f32")), []);
  });
});
