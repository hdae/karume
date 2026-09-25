/**
 * コンテナ（`krm`）1 本の**取得面**（ADR 0109 決定 7）。manifest の `container` 欄を受けて、
 * 読み手（`@karume/runtime` の `openContainer`）がそのまま食える「part → バイト区間」の口を返す。
 *
 * この面が持つのは**区間読みだけ**である: 温めた永続キャッシュ（または手元の実体）に対して
 * `[offset, offset+length)` を引き、費用の型（`AssetRangeReader.cost`）で経路を分ける。
 *
 * MUST: **温め（相 1）はこの面に入れない**。実行できないモデルの重みは 1 バイトも落とさない
 * （ADR 0108 決定 19 / ADR 0070 決定 5）ので、呼び手は
 *
 * 1. descriptor を持つ part 0 だけを取って admission を通す
 * 2. 通ったら重みの part を温める（{@link ../mod.ts prefetchAssets} — 進捗・中断・4 並列つき）
 * 3. この面を開いて読む
 *
 * の順を守る。面の中で全 part を温めると、この順序が面の内側から壊れる。温めずに開いても
 * 読めはする（HF 取得元の区間読み口は在庫が無ければ相 1 と同じ温めを 1 度だけ挟む — `source.ts` ⑧）
 * が、その温めは進捗にも中断にも現れない。
 *
 * MUST: hub は `@karume/runtime` を import しない（依存は一方向: models → hub / runtime）。
 * {@link ContainerBlockSource} は runtime の `BlockSource` と**構造互換**の型をこちら側で綴った
 * もので、`verified` 欄だけがこの面の上乗せ（ADR 0109 決定 7 の「検証済みを運ぶ」）。
 *
 * MUST: descriptor の中身は**開かない**。2 文書の検証も block 目次の突合も `openContainer` の
 * 責務で、hub が持つのは宣言（`parts[].size`）だけで閉じる検査に限る（ADR 0109 決定 6）。
 */

import { createFetchContext } from "./context.ts";
import { HubError } from "./errors.ts";
import { assertTightView } from "./fetch.ts";
import type { ContainerRef, FileRef } from "./manifest.ts";
import { type LoadedManifest, type LoadManifestOptions, pinnedSourceOf } from "./session.ts";
import { type AssetRangeReader, type PinnedSource, sourceForRef } from "./source.ts";

/**
 * part → バイト区間の引き出し口。`@karume/runtime` の `BlockSource` と構造互換で、
 * `openContainer({ kind: "source", source })` にそのまま渡せる。
 *
 * MUST: 返るバイト列は**呼び手が書き換えない**（取得元の実装によっては、取得元が抱え続ける
 * buffer そのものであり得る）。scan 経路の返りは part の器全体を握る view なので、区間より長く
 * 持つ呼び手は写す（握り続けると区間ではなく part 全体が生き残る）。
 */
export type ContainerBlockSource = {
  /** 宣言された part の本数（長さ 0 の part も数える — 添字が part の id）。 */
  readonly partCount: number;
  /**
   * 取得元がバイト列を**全量検証済み**か（ADR 0109 決定 7）。HF 取得元は取得層が part 全量を
   * 流しながら sha256 を照合するので `true`、ローカルディレクトリは照合しないので `false`。
   * 読み手はこれを見て block ごとの digest を掛けるかどうかを決める（cold の 2 重 digest を
   * 避ける — container-v1 §7）。
   */
  readonly verified: boolean;
  /** 宣言された part 長（同期・1 バイトも取らずに答える）。範囲外の添字は fail loudly。 */
  partLength(index: number): number;
  /**
   * `[offset, offset + length)` を返す。seek 経路は区間ぶんの tight view、scan 経路は保持枠の
   * part の器の view（`byteOffset` = 区間の開始位置・`buffer` = part 全体）。
   */
  read(part: number, offset: number, length: number): Promise<Uint8Array<ArrayBuffer>>;
};

/** 区間読みの経路。part ごとに 1 度だけ決める（取得元の能力と費用の型で分かれる）。 */
type ReadPlan =
  /** 位置読みが安い取得元（ブラウザの Blob・ローカルの区間読み）— block ごとに引く。 */
  | { readonly kind: "seek"; readonly reader: AssetRangeReader }
  /**
   * 読み飛ばしが offset に比例する取得元（Deno の既定）と、区間読みを持たない取得元 —
   * part を 1 度だけ全量読んで切り出す（同じ part の block を順に読むと読み飛ばしが二次に
   * なるため）。
   */
  | { readonly kind: "scan" };

/** scan 経路の保持枠 1 つぶん（in-flight の全量読みも同じ席で表す）。 */
type HeldPart = {
  readonly part: number;
  readonly bytes: Promise<Uint8Array<ArrayBuffer>>;
};

/**
 * コンテナ 1 本の取得面を開く。**何も取りに行かない**ので同期で返る — 実体に触れるのは
 * {@link ContainerBlockSource.read} を呼んだときだけ（part ごとに初回の読みで区間読み口を開く）。
 *
 * 進捗も `signal` の席もこの面には無い（温めを持たないので出すものが無い）。中断は口を配る前に
 * 1 度だけ見て、開いた後の読みには効かせない — 読み口は消費側が寿命ぶん掴み続ける前提なので、
 * 開いたときの signal を握ると以後の読みが全部それに道連れになる（`source.ts` ⑧）。
 *
 * 失敗の扱いは 2 つに分かれる: 取得元の解決（未 mapping の越境）は取得の設定不足なので
 * `HubFetchError` に包み、`read` の失敗は取得元の素の `Error` をそのまま通す（`openAsset` と
 * 同じ — 読みは取得ではない）。
 */
export const openContainerSource = (
  loaded: LoadedManifest,
  container: ContainerRef,
  options: LoadManifestOptions = {},
): ContainerBlockSource => {
  // 中断は口の組み立てより先に見る（取得元の能力差で中断の見え方が変わらない — 共通層の作法）。
  options.signal?.throwIfAborted();
  const session = pinnedSourceOf(loaded, options);
  const context = createFetchContext(loaded, session.origin);
  const { parts } = container;

  const partAt = (index: number): FileRef => {
    if (!Number.isInteger(index) || index < 0 || index >= parts.length) {
      throw new Error(
        `hub: part ${index} はこの容器に無い（宣言は ${parts.length} 本 — 添字が part の id）`,
      );
    }
    return parts[index];
  };

  // 越境は**容器単位**（parse が一様性を門にしている）ので、取得元は容器に 1 つに決まる。
  // 開くときに 1 度だけ解決して束縛する — read ごとに引くと、越境容器では読みのたびに越境先の
  // 取得元が組み直される。
  let origin: PinnedSource;
  try {
    origin = sourceForRef(session, parts[0]);
  } catch (error) {
    // 未 mapping の越境（`sources/local.ts` の素の Error）は呼び手の設定不足。同じ manifest の
    // `fetchAssets` と同じ見え方（path / repo / revisionSha / available つき）で落とす。
    if (error instanceof HubError) throw error;
    throw context.fetchFailure(parts[0], "取得", error);
  }
  const verified = origin.origin.integrity === "network";

  const plans = new Map<number, Promise<ReadPlan>>();
  // MUST: 引くのは添字だけ（ref は `partAt` で導く）— 添字と ref の組を呼び手が渡せる形にすると、
  // 食い違った組で「別の part の読み口」を掴める。
  const planFor = (index: number): Promise<ReadPlan> => {
    const existing = plans.get(index);
    if (existing !== undefined) return existing;
    const { openFile } = origin;
    const opened: Promise<ReadPlan> = openFile === undefined
      ? Promise.resolve({ kind: "scan" })
      : openFile(partAt(index), {}).then((reader): ReadPlan =>
        reader.cost === "seek" ? { kind: "seek", reader } : { kind: "scan" }
      );
    // MUST: 失敗した決定は覚えない — 覚えると、一過性の失敗で開けなかった part が以後の読みで
    // すべて同じ reject を返し続ける（開き直す手段が消える）。
    const decided = opened.catch((error: unknown) => {
      plans.delete(index);
      throw error;
    });
    plans.set(index, decided);
    return decided;
  };

  /**
   * scan 経路の保持枠。**同時に持つのは 1 part ぶんだけ**（別の part を読んだら前を手放す）。
   * 切り出しは器の view なので、枠が握る part の器のほかにホスト RAM は乗らない — 重みの読み手
   * （runtime の `containerBatches`）は block を 1 本ずつ引いて、上げた反復で手放す（引いた view も
   * この器の中を指す）。part の境界では、手放した器が GC されるまで一時的に 2 part ぶんが生きる。
   *
   * MUST: **in-flight の全量読みも同じ席に置く** — 席を「決着したバイト列」だけにすると、同じ
   * part への並行 read が全員 readFile へ入り、その瞬間だけ part 長 × 本数が生きる。
   * NOTE: 席は 1 つなので、並行に**別の** part を読むと後の読みが先の読みを席から追い出し、
   *       追い出された側の全量読みも最後まで走る。同時に生きる part は「並行に読む別 part の
   *       本数」まで増え、hub はその本数を抑えない（追い出した part をまた読むと全量読みが重複
   *       する）。追い出された part の器は、そこから切った view を呼び手が握っている間は生き残る。
   *       重みの読み手は part 順に直列に読むのでこの形にならない。資産の行読み（models の PLE —
   *       別の part を並行に読む）はこの形になり、prefill では行読みの同時発行の上限（models の
   *       `ROW_READ_CONCURRENCY` = 16 本）まで part の全量読みが同時に生きる。区間は
   *       `readAssetRange` が写すので読み終えた器は握らない（part ちょうどの資産は写さずに握るが、
   *       握る量は写しと同じ）。
   */
  let held: HeldPart | undefined;
  const readWhole = (index: number, ref: FileRef): Promise<Uint8Array<ArrayBuffer>> => {
    const current = held;
    if (current !== undefined && current.part === index) return current.bytes;
    const pending = origin.readFile(ref, {
      // この面は取得ではないので進捗は出さない（温めは呼び手が prefetchAssets で済ませている）。
      onProgress: () => {},
      sizeViolation: context.sizeViolation(ref),
    }).then((bytes) => assertTightView(bytes, ref.path));
    const entry: HeldPart = {
      part: index,
      // MUST: 失敗した全量読みは席に残さない（残すと以後の読みが同じ reject を返し続ける）。
      bytes: pending.catch((error: unknown) => {
        if (held === entry) held = undefined;
        throw error;
      }),
    };
    held = entry;
    return entry.bytes;
  };

  return {
    partCount: parts.length,
    verified,
    partLength: (index) => partAt(index).size,
    read: async (part, offset, length) => {
      const ref = partAt(part);
      if (ref.size === 0) {
        throw new Error(
          `hub: part ${part}（${ref.path}）は長さ 0 の part なので読めない` +
            `（読み手は partLength が 0 を答えた時点で読みに行かない）`,
        );
      }
      if (
        !Number.isInteger(offset) || !Number.isInteger(length) || offset < 0 || length < 0 ||
        offset + length > ref.size
      ) {
        throw new Error(
          `hub: part ${part}（${ref.path}）の区間 [${offset}, ${offset + length}) が不正` +
            `（宣言 size ${ref.size} — 非負整数で size に収まる区間だけを読める）`,
        );
      }
      const plan = await planFor(part);
      if (plan.kind === "seek") {
        return assertTightView(await plan.reader.read(offset, length), ref.path);
      }
      // MUST: 切り出しは写さない（写すと part ごとに器と写しで part 長を 2 重に持つ）。返るのは
      // 保持枠の part の器の view で、次の 2 つが成り立つので消費側はそのまま読める:
      //  - 整列: 器は tight（上の `assertTightView`）なので byteOffset = block.offset で、block
      //    開始は 64 B 整列（descriptor の parse が強制）。scale の Float32Array view に要る 4 B
      //    整列はこれで満たされる — `assertTightView` を外すとこの整列が偶然任せになる。
      //  - 寿命: 重みの読み手は batch の view をフェンスまでに使い切り、次の part を読み始める
      //    （= 枠が器を手放す）のはその後なので、view は枠が器を握る期間の内側に収まる。
      //    資産の読み手は `readAssetRange`（models）が区間を写してから持つ（part ちょうどの資産は
      //    写さずに器を共有するが、握る量は写しと同じ）。
      return (await readWhole(part, ref)).subarray(offset, offset + length);
    },
  };
};
