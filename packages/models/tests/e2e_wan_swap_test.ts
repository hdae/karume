/**
 * Wan2.1 の umT5（`text_encoder`）の**部品差し替え**（`components` — ADR 0122 決定 7・ADR 0108 決定 19）を、
 * 手元の配布形ミラーで実 GPU の生成まで通す e2e。
 *
 * ホストテスト（`wan_components_test.ts`）は疑似 HF とメモリ上のディレクトリで admission と取得を押さえるが、
 * 差し替え先が**手元のディレクトリ**のとき、重みの block は Session の構築（`generate` の text 段）で区間読み
 * されるので、ホストテストでは観測できない（ADR 0122 追記「段 d の結果」の残り）。ここで押さえるのは 4 点:
 *
 * ① **構築**: Wan の配布形の取得元に umT5 の越境先の `crossRepo` mapping を 1 本も渡さずに、`text_encoder` を
 *    差し替えた `WanPipeline.fromPretrained` が通る。
 * ② **出所**: 差し替え先の取得元を、読んだ path と区間を記録する読み口で包む。差し替え先の `text_encoder` の
 *    グラフが束縛する block（実体・scale・zero point — 差し替え先の descriptor から引く）の区間が全部、text 段の
 *    中の差し替え先からの読みで隙間なく覆われ、構築中に差し替え先から読まれたのは manifest と part 0
 *    （descriptor）だけで、text 段の後には 1 度も読まれない。Wan の取得元は text encoder の part の path を
 *    1 度も読まない（= 差し替え先のほかに text encoder のバイト列を供給したものが無い — 越境先は mapping が
 *    無いので、そもそも開けない）。前提: 差し替え先の実体を持つ重みの part（part 1 以降）は全部、グラフの
 *    束縛する block を持つ（資産だけの part を持つ差し替え先は対象外 — GPU に触る前に見て落とす）。
 * ③ **生成**: `e2e_wan_pipeline_test.ts` の GPU 経路の 1 本目と同じ要求（固定プロンプト `boxing-cats` の原文・
 *    seed 42・2 ステップ・832×480・33 フレーム・既定の negative・`f16` 席）が完走し、非有限 0・宣言どおりの形。
 * ④ **対照**（GPU 不要 — 模擬 GPU）: 同じ取得元で差し替えず mapping も無ければ、umT5 の part 0 の事前取得で
 *    hub の「越境先が無い」の案内を cause に持つ失敗で落ちる（① が mapping の残りで通ったのではないことの確認）。
 *
 * 差し替え先は利用者が選ぶ手元の配布形（実験用ミラー）で、ディレクトリを環境変数 {@link SWAP_ENV} で受ける
 * （追跡されるコードに特定のミラーを書かない — ADR 0122「影響ファイル」の TS のテスト）。未設定なら ①〜③ は
 * 明示 SKIP（opt-in）。④ は差し替え先に依らず構築だけで済むので、配布形があれば既定のレーンで回す。
 *
 * ## 観測（門ではない）
 *
 * 出力の sha256 は門にしない（差し替え先は利用者が選ぶので、環境ごとの行を持てない）。sha256・構築と段の
 * 所要・差し替え先から読んだバイト数を `outputs/verify/<環境キー>/<日付>_wan-swap/` に残し、同じ要求を元の
 * umT5 で回した行（`fixtures/references/wan.json` の {@link ORIGINAL_CASE_ID}）との一致 / 不一致も記録する
 * （行は引くだけで、書かない・突き合わせの決着にもしない）。
 *
 * 配布形が無い環境は明示 SKIP（全 SKIP を FAIL にするのは門番 `distribution_gate_test.ts`）。
 */

import {
  assert,
  assertEquals,
  assertInstanceOf,
  assertRejects,
  assertStringIncludes,
} from "@std/assert";
import {
  type DistributionSource,
  type FileRef,
  loadManifest,
  localDirectory,
  openContainerSource,
  parseManifest,
  resolveSelection,
} from "@karume/hub";
import { denoDirectory } from "@karume/hub/deno";
import { openContainer, type SessionDiagnostics } from "@karume/runtime";
import {
  type GeneratedVideo,
  wanFrameToRgba,
  type WanGenerateEvent,
  WanPipeline,
  type WanRunComponent,
} from "../wan.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "./helpers/gpu.ts";
import { settleReleases } from "./helpers/settle-releases.ts";
import { assertRunningAdapter } from "../../runtime/tests/helpers/environment.ts";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import { openReferences, sha256Hex } from "../../runtime/tests/helpers/reference.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";

/** 差し替え先（手元の配布形 — `karume.json` を持つディレクトリ）を受ける環境変数（opt-in）。 */
const SWAP_ENV = "KARUME_WAN_SWAP_TEXT_ENCODER";
const SWAP_RAW = Deno.env.get(SWAP_ENV);
/** 差し替え先のディレクトリ（未設定・空文字なら undefined = ①〜③ を SKIP）。 */
const SWAP_DIRECTORY = SWAP_RAW === undefined || SWAP_RAW === "" ? undefined : SWAP_RAW;

/** 配布形ミラー（`dist.py --pipeline wan` の既定の出力先）。 */
const DIST_ROOT = new URL("../../../models/karume-wan2.1/", import.meta.url);
const ASSEMBLE_COMMAND = "cd tools/export-recipes && uv run python dist.py --pipeline wan";
/** manifest の部品名（`src/wan/pipeline.ts` の `TEXT_ENCODER`）。 */
const TEXT_ENCODER = "text_encoder";
/**
 * 席（明示 — 省略すると manifest の `defaultQuant` へ黙って移る。`e2e_wan_pipeline_test.ts` の GPU 経路と同じ
 * `f16` 席にして、元の umT5 の行と同じ要求にする）。
 */
const F16_QUANT = "f16";
const STEPS = 2;
/** `frames` / 寸法を省いたときの既定（`src/wan/pipeline.ts`）— 形の検査にだけ使う。 */
const DEFAULT_FRAMES = 33;
const DEFAULT_WIDTH = 832;
const DEFAULT_HEIGHT = 480;

/**
 * 差し替えの生成 1 本。要求は `e2e_wan_pipeline_test.ts` の `GPU_TEXT_CASES` の 1 本目と同じ（固定プロンプトの
 * 原文・seed 42・2 ステップ・既定の negative / 寸法 / フレーム数）— 違うのは text encoder の重みだけ。
 *
 * MUST: 文面・seed・step 数を変えたら ID も変え、{@link ORIGINAL_CASE_ID} との比較もやめる（別の条件の sha を
 * 比べることになる）。
 */
const SWAP_CASE = { id: "swap-text-2step-boxing-cats-seed42", prompt: "boxing-cats", seed: 42 };
/** 同じ要求を元の umT5 で回した sha 行（比べるだけ — 観測）。 */
const ORIGINAL_CASE_ID = "gpu-text-2step-boxing-cats-seed42";

const fileExists = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

const DIST_PRESENT = fileExists(new URL("karume.json", DIST_ROOT));
if (!DIST_PRESENT) {
  console.warn(
    `[karume] 配布形ミラー ${DIST_ROOT.pathname} が無いため Wan の text_encoder の差し替えの e2e を SKIP する。` +
      `組み立て: ${ASSEMBLE_COMMAND}（全 SKIP は門番 distribution_gate_test.ts が FAIL にする）`,
  );
}
if (SWAP_DIRECTORY === undefined) {
  console.warn(
    `[karume] Wan の text_encoder の差し替えの生成（実 GPU）は opt-in のため SKIP する` +
      `（${SWAP_ENV}=<差し替え先の手元の配布形のディレクトリ> で回す）`,
  );
}

const references = openReferences(new URL("fixtures/references/wan.json", import.meta.url));
const results = openResults("wan-swap");

/** 実体を持つ part（長さ 0 の part は取得の対象外 — hub も取らない）。 */
const nonEmpty = (parts: readonly FileRef[]): readonly FileRef[] =>
  parts.filter((part) => part.size > 0);

/** manifest の `text_encoder` の part 列（席は `quant`・model は既定）。 */
const textEncoderParts = async (root: URL, quant?: string): Promise<readonly FileRef[]> => {
  const manifest = parseManifest(await Deno.readTextFile(new URL("karume.json", root)));
  const container = resolveSelection(manifest, {
    ...(quant === undefined ? {} : { quant }),
    weights: [TEXT_ENCODER],
  }).containers[TEXT_ENCODER];
  assert(container !== undefined, `${root.pathname} の manifest に '${TEXT_ENCODER}' の容器が無い`);
  return container.parts;
};

/**
 * Wan の配布形の `text_encoder` の part 列。前提（越境参照であること）もここで見る — 自リポの容器だと「mapping
 * 無しで通る」が自明になり、① と ④ が何も確かめない。
 */
const wanTextEncoderParts = async (): Promise<readonly FileRef[]> => {
  const parts = await textEncoderParts(DIST_ROOT, F16_QUANT);
  assert(
    parts.length > 0 && parts.every((part) => part.repo !== undefined),
    `配布形 ${DIST_ROOT.pathname} の '${TEXT_ENCODER}' が umT5 の配布リポへの越境参照でない` +
      `（この e2e の前提 — 組み直し: ${ASSEMBLE_COMMAND}）`,
  );
  return parts;
};

/** グラフが束縛する block 1 本の所在（容器の part 番号 = manifest の part 列の添字と、part 内の区間）。 */
type BoundBlock = {
  readonly id: string;
  readonly part: number;
  readonly offset: number;
  readonly length: number;
};

/**
 * 差し替え先の `text_encoder` の容器を区間読みで開き（読むのは part 0 の 2 文書だけ — 重みは読まない）、
 * グラフが束縛する block（実体・scale・zero point）を全部返す。選択は {@link textEncoderParts} の差し替え先の
 * 呼び方と同じ（model / quant は既定）。グラフ名 = manifest の weights キー（書き手の規約 — container-v1 §2.1）。
 */
const textEncoderBlocks = async (root: URL): Promise<readonly BoundBlock[]> => {
  const loaded = await loadManifest(denoDirectory(root.pathname));
  const container = resolveSelection(loaded.manifest, { weights: [TEXT_ENCODER] })
    .containers[TEXT_ENCODER];
  assert(container !== undefined, `${root.pathname} の manifest に '${TEXT_ENCODER}' の容器が無い`);
  const opened = await openContainer(
    { kind: "source", source: openContainerSource(loaded, container) },
    container.descriptor,
  );
  const graph = opened.graphs[TEXT_ENCODER];
  assert(
    graph !== undefined,
    `${root.pathname} の '${TEXT_ENCODER}' の容器にグラフ '${TEXT_ENCODER}' が無い`,
  );
  return [...graph.supplies.values()].flatMap((supply) => [
    ...supply.blocks,
    ...(supply.scale === undefined ? [] : [supply.scale]),
    ...(supply.zeroPoint === undefined ? [] : [supply.zeroPoint]),
  ]);
};

/** 読みが起きた区間（構築中 / text 段の中 / text 段の後）。 */
type Phase = "construction" | "text_encoder" | "after_text_encoder";

/** 取得元の読み口が受けた読み 1 回（全量読みは offset 0・長さ = 返したバイト数）。 */
type Read = {
  readonly path: string;
  readonly offset: number;
  readonly length: number;
  readonly phase: Phase;
};

/** `reads`（同じ path の読み — 順不同・重なりを許す）が `[offset, offset + length)` を隙間なく覆うか。 */
const covers = (reads: readonly Read[], offset: number, length: number): boolean => {
  const end = offset + length;
  let reached = offset;
  for (const read of [...reads].sort((a, b) => a.offset - b.offset)) {
    if (read.offset > reached) break;
    reached = Math.max(reached, read.offset + read.length);
    if (reached >= end) return true;
  }
  return reached >= end;
};

/**
 * 手元のディレクトリを取得元にし、読んだ path と区間を `reads` に積む。読み口は `@karume/hub/deno` の
 * `denoDirectory` と同じ 2 本（全量読み + 位置読み）— 位置読みを落とすと Session の構築が区間読み（seek）から
 * part の全量読み（scan）へ倒れ、利用者の経路と別の読み方を観測することになる。
 *
 * WHY 取得元ハンドル（`DistributionSource`）ではなく読み口（`DirectoryAdapter`）を包む: ハンドルは公開メンバを
 * 持たない不透明な値で（hub の `src/source.ts` の MUST）、外から委譲の包みを組めない。読み口を包んで
 * `localDirectory` に渡すのが、hub の公開面で組める唯一の委譲（`denoDirectory` 自身も同じ形）。
 * NOTE: 読み口の中身は `denoDirectory` の 2 本の写しで、読めないときの包みの文言・`readFile` の中断の扱い・
 * label は写していない（取得元の意味論 — 区間読み・size の境界・越境の解決 — は本物の `localDirectory` を
 * 通る）。hub が取得元ハンドルの委譲口を公開したら、ハンドルを包む形へ戻す。
 */
const recordingDirectory = (
  root: URL,
  label: string,
  reads: Read[],
  phase: () => Phase,
): DistributionSource =>
  localDirectory({
    readFile: async (path, { signal }) => {
      const abort = signal === undefined ? {} : { signal };
      const bytes = await Deno.readFile(new URL(path, root), abort);
      reads.push({ path, offset: 0, length: bytes.byteLength, phase: phase() });
      return bytes;
    },
    // MUST: `length` ちょうどを buffer 全体を占める view で返す（`DirectoryAdapter.readFileRange` の契約）。
    readFileRange: async (path, offset, length, { signal }) => {
      signal?.throwIfAborted();
      reads.push({ path, offset, length, phase: phase() });
      const at = new URL(path, root);
      const file = await Deno.open(at);
      try {
        await file.seek(offset, Deno.SeekMode.Start);
        const target = new Uint8Array(new ArrayBuffer(length));
        let filled = 0;
        while (filled < length) {
          signal?.throwIfAborted();
          const read = await file.read(target.subarray(filled));
          if (read === null) {
            throw new Error(`${at.pathname} が offset ${offset} からの ${length} バイトに足りない`);
          }
          filled += read;
        }
        return target;
      } finally {
        file.close();
      }
    },
  }, { label });

/** 環境変数のパス → ディレクトリの `file:` URL（末尾 `/` つき — 無いと `new URL(path, root)` が兄弟を指す）。 */
const directoryUrl = (path: string): URL => {
  let resolved: string;
  try {
    resolved = Deno.realPathSync(path);
  } catch (cause) {
    throw new Error(`${SWAP_ENV}='${path}' のディレクトリを解決できない`, { cause });
  }
  const url = new URL("file:///");
  url.pathname = `${resolved}/`;
  return url;
};

/** 出力フレームの uint8 の RGB を全フレーム連結したバイト列（sha256 の実物 — `wanFrameToRgba` の規則）。 */
const rgbBytes = (video: GeneratedVideo): Uint8Array<ArrayBuffer> => {
  const plane = video.width * video.height;
  const out = new Uint8Array(video.frames * plane * 3);
  for (let frame = 0; frame < video.frames; frame += 1) {
    const rgba = wanFrameToRgba(video, frame);
    for (let index = 0; index < plane; index += 1) {
      out.set(rgba.subarray(index * 4, index * 4 + 3), (frame * plane + index) * 3);
    }
  }
  return out;
};

const textOf = (pipeline: WanPipeline, name: string): string => {
  const entry = pipeline.prompts.find((candidate) => candidate.name === name);
  assert(entry !== undefined, `埋め込み資産に '${name}' が無い`);
  return entry.prompt;
};

Deno.test({
  name:
    "Wan の text_encoder の差し替えの対照（GPU 不要・模擬 GPU）: 差さず umT5 の越境先の mapping も無ければ、" +
    "umT5 の part 0 の事前取得で「越境先が無い」の案内を cause に持って落ち、Wan の取得元から umT5 の part を読まない",
  ignore: !DIST_PRESENT,
  fn: async () => {
    const wanParts = await wanTextEncoderParts();
    const [part0] = wanParts;
    const reads: Read[] = [];
    const error = await assertRejects(
      () =>
        WanPipeline.fromPretrained(
          recordingDirectory(DIST_ROOT, "wan-swap-control", reads, () => "construction"),
          // 落ちるのは descriptor の事前取得（admission の前）— GPU に触る段まで進まない。
          { gpu: fakeGpuContext(fakeDevice()), textEncoder: "gpu", quant: F16_QUANT },
        ),
      Error,
      `${part0.path} の事前取得に失敗した（repo ${part0.repo} @ ${part0.revision}）`,
    );
    // 取得層は取得元の案内を cause に残す（hub の local.ts — 隣のディレクトリを推測しない）。
    assertInstanceOf(error.cause, Error);
    assertStringIncludes(error.cause.message, `repo '${part0.repo}' の越境先が無い`);
    assertStringIncludes(error.cause.message, "crossRepo");
    // NOTE: 保険（独立の検出力はほぼ無い）— 越境の解決は読みより先に投げるので、上の失敗文言の検査が通る限り
    // 読みは起きない。独立に赤になるのは、hub が Wan の配布形のディレクトリに越境先を推測しに行く退行だけ。
    const textEncoderPaths = new Set(wanParts.map((part) => part.path));
    assertEquals(
      reads.filter((read) => textEncoderPaths.has(read.path)).map((read) => read.path),
      [],
      "umT5 の part を Wan の取得元から読もうとした",
    );
  },
});

Deno.test({
  name:
    `Wan の text_encoder の差し替え（実 GPU・opt-in ${SWAP_ENV}）: umT5 の越境先の mapping 無しで構築が通り、` +
    "差し替え先の束縛された block が全部 text 段で差し替え先から読み切られ（ほかに text encoder のバイト列の出所が無い）、" +
    "2 ステップの生成が非有限 0・宣言どおりの形",
  ignore: SWAP_DIRECTORY === undefined || !DIST_PRESENT || !GPU_AVAILABLE,
  fn: async () => {
    assert(SWAP_DIRECTORY !== undefined);
    const swapRoot = directoryUrl(SWAP_DIRECTORY);
    assert(
      fileExists(new URL("karume.json", swapRoot)),
      `${SWAP_ENV} のディレクトリ ${swapRoot.pathname} に karume.json が無い（手元の配布形を指すこと）`,
    );
    const wanParts = await wanTextEncoderParts();
    // 差し替えは `{ source }` だけ（model / quant は差し替え先の既定）— 同じ選択で part 列を引く。
    const swapParts = await textEncoderParts(swapRoot);
    assert(
      swapParts.every((part) => part.repo === undefined),
      `差し替え先 ${swapRoot.pathname} の '${TEXT_ENCODER}' が越境参照を持つ（この e2e は差し替え先のディレクトリが` +
        "実体を持つ形だけを見る）",
    );
    const swapWeightParts = nonEmpty(swapParts.slice(1));
    assert(swapWeightParts.length > 0, `差し替え先の '${TEXT_ENCODER}' に重みの part が無い`);
    // 前提（GPU に触る前に見る）: 重みの part は全部グラフの束縛する block を持つ。これが崩れると、下の
    // block の被覆は「その part を読まなかった」を見逃す（資産だけの part は Session の構築で読まれない）。
    const swapBlocks = await textEncoderBlocks(swapRoot);
    const boundPaths = new Set(swapBlocks.map((block) => swapParts[block.part].path));
    assertEquals(
      swapWeightParts.map((part) => part.path).filter((path) => !boundPaths.has(path)),
      [],
      `差し替え先の '${TEXT_ENCODER}' に、グラフの束縛する block を持たない重みの part がある` +
        "（資産だけの part を持つ差し替え先はこの e2e の対象外）",
    );

    await assertRunningAdapter();
    let deviceLost: string | undefined;
    const gpu = await acquireTestGpu({
      onDeviceLost: (info) => {
        deviceLost = `${info.reason}: ${info.message}`;
      },
    });
    try {
      await runRecordedCase(results, { id: SWAP_CASE.id }, async () => {
        let phase: Phase = "construction";
        const currentPhase = (): Phase => phase;
        const wanReads: Read[] = [];
        const swapReads: Read[] = [];
        const diagnostics = new Map<WanRunComponent, SessionDiagnostics>();

        // ① Wan の取得元に crossRepo を 1 本も渡さない（umT5 の越境先はどこからも開けない）。
        const constructionStarted = performance.now();
        await using pipeline = await WanPipeline.fromPretrained(
          recordingDirectory(DIST_ROOT, "wan-swap", wanReads, currentPhase),
          {
            gpu,
            textEncoder: "gpu",
            quant: F16_QUANT,
            components: {
              [TEXT_ENCODER]: {
                source: recordingDirectory(swapRoot, swapRoot.pathname, swapReads, currentPhase),
              },
            },
            onRunDiagnostics: (component, diagnosed) => diagnostics.set(component, diagnosed),
          },
        );
        const constructionMs = performance.now() - constructionStarted;

        const stageStarted = new Map<string, number>();
        const stageMs: Record<string, number> = {};
        const onEvent = (event: WanGenerateEvent): void => {
          if (event.kind !== "stage") return;
          if (event.component === TEXT_ENCODER) {
            phase = event.at === "start" ? "text_encoder" : "after_text_encoder";
          }
          const now = performance.now();
          if (event.at === "start") stageStarted.set(event.component, now);
          else stageMs[event.component] = now - (stageStarted.get(event.component) ?? now);
        };
        const generateStarted = performance.now();
        const video = await pipeline.generate({
          prompt: textOf(pipeline, SWAP_CASE.prompt),
          seed: SWAP_CASE.seed,
          steps: STEPS,
          onEvent,
        });
        const generateMs = performance.now() - generateStarted;
        assertEquals(phase, "after_text_encoder", "text 段の start / end が届いていない");
        assert(diagnostics.has(TEXT_ENCODER), "umT5 の run の診断が届いていない");

        // ② 差し替え先から読んだもの。
        const swapPathSet = new Set(swapParts.map((part) => part.path));
        assertEquals(
          [...new Set(swapReads.map((read) => read.path))].filter(
            (path) => path !== "karume.json" && !swapPathSet.has(path),
          ),
          [],
          "差し替え先から text_encoder の part と manifest 以外を読んだ",
        );
        assertEquals(
          [
            ...new Set(
              swapReads.filter((read) => read.phase === "construction").map((read) => read.path),
            ),
          ].filter((path) => path !== "karume.json" && path !== swapParts[0].path),
          [],
          "構築中に差し替え先の part 0（descriptor）より先を読んだ（重みは Session の構築で読む契約）",
        );
        // バイトの被覆: 束縛された block の区間が全部、text 段の差し替え先からの読みで覆われる（上の前提と
        // 合わせて、重みの part は全部 text 段で読まれる）。
        const textStageReads = new Map<string, Read[]>();
        for (const read of swapReads.filter((candidate) => candidate.phase === "text_encoder")) {
          textStageReads.set(read.path, [...(textStageReads.get(read.path) ?? []), read]);
        }
        assertEquals(
          swapBlocks.filter((block) =>
            !covers(
              textStageReads.get(swapParts[block.part].path) ?? [],
              block.offset,
              block.length,
            )
          ).map((block) => `${block.id}（part ${block.part}）`),
          [],
          "差し替え先の block のうち、text 段で差し替え先から読み切られなかったもの",
        );
        assertEquals(
          swapReads.filter((read) => read.phase === "after_text_encoder").map((read) => read.path),
          [],
          "text 段の後に差し替え先を読んだ",
        );
        // Wan の取得元は text encoder の part の path（元の越境先の宣言・差し替え先の宣言のどちらの綴りでも）を
        // 1 度も読んでいない — 差し替え先のほかに text encoder のバイト列を供給したものが無い。
        // NOTE: 保険（独立の検出力はほぼ無い）— 元の席が開かれる退行は、mapping が無いので越境の解決で ① の
        // 構築が先に落とす。独立に赤になるのは、hub が Wan の配布形のディレクトリに越境先を推測しに行く退行だけ。
        const textEncoderPaths = new Set([...wanParts, ...swapParts].map((part) => part.path));
        assertEquals(
          wanReads.filter((read) => textEncoderPaths.has(read.path)).map((read) => read.path),
          [],
          "Wan の取得元から text encoder の part を読んだ",
        );

        // ③ 生成の形と有限性。
        assertEquals(
          [video.frames, video.height, video.width],
          [DEFAULT_FRAMES, DEFAULT_HEIGHT, DEFAULT_WIDTH],
        );
        assertEquals(video.data.length, 3 * video.frames * video.height * video.width, "要素数");
        const nonFinite = video.data.reduce(
          (count, value) => count + (Number.isFinite(value) ? 0 : 1),
          0,
        );
        assertEquals(nonFinite, 0, `${SWAP_CASE.id}: 非有限`);
        assertEquals(deviceLost, undefined, "device lost");

        // 観測: sha256（門にしない）・元の umT5 の行との一致・所要・差し替え先から読んだ量。
        const bytes = rgbBytes(video);
        const sha256 = await sha256Hex(bytes);
        const artifact = `${SWAP_CASE.id}.rgb`;
        await Deno.writeFile(results.artifact(artifact), bytes);
        const original = references.lookup(ORIGINAL_CASE_ID);
        const swapBytes = swapReads.reduce((total, read) => total + read.length, 0);
        const weightBytes = swapWeightParts.reduce((total, part) => total + part.size, 0);
        const gib = (value: number) => `${(value / 2 ** 30).toFixed(3)} GiB`;
        const seconds = (ms: number) => `${(ms / 1000).toFixed(1)} s`;
        const verdict = original === sha256 ? "一致" : "不一致";
        const stages = Object.entries(stageMs).map(([stage, ms]) => `${stage} ${seconds(ms)}`);
        const notes = [
          `差し替え先: ${swapRoot.pathname}`,
          `重みの part ${swapWeightParts.length} 本・${gib(weightBytes)}`,
          `差し替え先から読んだ量 ${gib(swapBytes)}（${swapReads.length} 回）`,
          original === undefined
            ? `元の umT5 の行 ${ORIGINAL_CASE_ID} はこの環境に無い`
            : `元の umT5 の行 ${ORIGINAL_CASE_ID} と${verdict}（${original}）`,
          `構築 ${seconds(constructionMs)}・生成 ${seconds(generateMs)}・段 ${stages.join("・")}`,
          `非有限 ${nonFinite}`,
          `sha256 ${sha256}（観測 — 門にしない）`,
        ];
        console.log(`[wan-swap] ${SWAP_CASE.id}:\n  ${notes.join("\n  ")}`);
        return { status: "pass", actual: sha256, artifact, note: notes.join(" / ") };
      });
    } finally {
      // MUST: device を捨てる前に解放を待つ（B570 の destroy の遅れ — 後続のテストの予算を残す）。
      await settleReleases(gpu);
      gpu.destroy();
    }
  },
});
