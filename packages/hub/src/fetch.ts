/**
 * 取得の共通層。**取得元そのものは持たない** — 取得元の面は `source.ts` の契約で、実装は
 * `sources/`（HF とローカルディレクトリ）。ここに残るのは、取得元が何であっても同じでなければ
 * ならない作法だけ:
 *
 * - 世代は**セッションあたり 1 回だけ**解決し、manifest も全ファイルも同一世代に固定して取得する
 *   （可変 ref のまま複数回解決すると manifest と重みが別の世代から来る）。例外は manifest が
 *   明示した**越境参照**（`FileRef` の `repo` / `revision` — ADR 0038 §7）だけで、その 1 本は
 *   セッションの世代ではなく宣言された (repo, revision) から取る。
 * - 取得と進捗総量の一意化（{@link fileRefKey}）・進捗の集計（`progress.ts`）。
 * - 同時取得の律速は面ごとに違う（全量面はバイト予算・逐次面の相 1 は本数 4）。
 * - `AbortSignal` は全取得へ透過し、**取得元が network に出ない区間でも**面の境界で明示的に見る。
 * - 引き渡すバイト列が buffer 全体を占めること（tight view）の検査。
 * - 失敗の文脈付け（`context.ts` — 真の第一失敗の復元を含む）。
 */

import { purgeLegacyCaches } from "./cache.ts";
import { createByteAdmission } from "./concurrency.ts";
import {
  createFetchContext,
  type FetchContext,
  manifestFetchFailure,
  manifestOversize,
  revisionResolutionFailure,
} from "./context.ts";
import { HubError, ManifestFormatError, ManifestReferenceError } from "./errors.ts";
import {
  type FileRef,
  fileRefKey,
  type Manifest,
  MANIFEST_FILENAME,
  parseManifest,
} from "./manifest.ts";
import { createProgressEmitter, type ProgressEmitter } from "./progress.ts";
import type { ResolvedFiles } from "./resolve.ts";
import {
  type FetchAssetsOptions,
  type HubRepoRef,
  LoadedManifest,
  type LoadManifestOptions,
  pinnedSourceOf,
  type StreamAssetsOptions,
} from "./session.ts";
import { DistributionSource, driverOf, type PinnedSource, sourceForRef } from "./source.ts";
import { createHfSource } from "./sources/hf.ts";

/**
 * 逐次面 {@link streamAssets} 相 1 の同時取得数（数十コンポーネントの manifest で接続を
 * 破綻させない）。相 1 は body をそのままキャッシュへ流す streaming なので、受信バッファの
 * 前確保が無い＝本数だけで RAM が決まらない。
 */
const CONCURRENCY = 4;

/**
 * 全量面 {@link fetchAssets} の in-flight バイト予算（1.5GiB = 1,610,612,736 バイト）。
 *
 * 全量面は 1 ファイルにつき `ref.size` ぶんの受信バッファを**受信前に**確保するので、律速を
 * 本数にすると RAM ピークが「同時本数 × その時点で一番大きいファイル」で決まってしまう
 * （実測: anima turbo i4 の先頭 4 本で計 2.503GiB を同時前確保 — 8GB 機ターゲットでは
 * ブラウザごと落ちる）。予算は前確保の合計に課すもので、これに加えて完走済みファイルの保持と
 * 検証の一時コピー（Chrome の `crypto.subtle.digest` は入力を Blink 内部へ全量コピーする）が
 * 乗るため、単一 ArrayBuffer の上限（Chromium は 2,145,386,496 バイトで打ち切る）よりも
 * 明確に下へ置く。
 *
 * NOTE: その digest は取得元の中（network 取得の検証）にあり、共通層からは直列化できない —
 * 検証を委ねた以上、同時に走る本数を決めるのはこの予算だけになった。キャッシュヒットは記録
 * ハッシュの文字列比較で済むので digest ごと起きない（2 回目以降の起動でこのピークは立たない）。
 *
 * 公開ノブにはしない — 「安全側へ下げる」以外の使い道が無い値であり、上げれば上のピークが
 * そのまま戻る。合わない配布が出たら定数ごと裁定し直す。
 */
const BYTE_BUDGET = 1.5 * 1024 * 1024 * 1024;

/**
 * 中断（呼び出し側の `AbortSignal`）かどうか。中断は取得失敗ではないので `HubFetchError` に
 * 包まず素通しする — 包むと呼び出し側の「自分が止めたのか落ちたのか」の判別が壊れる。
 *
 * MUST: 既定の `abort()` が作る DOMException だけを見ない — `abort(reason)` の custom reason
 * では `fetch` が **reason 自体**で reject するため、任意の値が中断の正体になり得る。判定は
 * 「実際に取得へ渡した signal が aborted で、捕まえた値がその reason と同一」に依る（合成した
 * 場合は合成後の signal — 上流の reason はそのまま伝播する）。
 */
const isAborted = (error: unknown, signal?: AbortSignal): boolean =>
  (error instanceof DOMException && error.name === "AbortError") ||
  (signal?.aborted === true && error === signal.reason);

/**
 * `openModel` は全量 ArrayBuffer を要求するため、返す bytes は buffer 全体を占めていなければ
 * ならない（slice で辻褄を合わせると RAM ピークが倍増する）。SharedArrayBuffer 背面は
 * ここで弾く（述語が主張する `Uint8Array<ArrayBuffer>` を型の上でも嘘にしない）。
 */
const isTightView = (bytes: Uint8Array): bytes is Uint8Array<ArrayBuffer> =>
  bytes.buffer instanceof ArrayBuffer && bytes.byteOffset === 0 &&
  bytes.byteLength === bytes.buffer.byteLength;

const assertTightView = (bytes: Uint8Array, path: string): Uint8Array<ArrayBuffer> => {
  if (!isTightView(bytes)) {
    throw new Error(
      `hub: ${path} の bytes が buffer 全体を占めていない` +
        `（byteOffset ${bytes.byteOffset} / byteLength ${bytes.byteLength} /` +
        ` buffer ${bytes.buffer.byteLength}）`,
    );
  }
  return bytes;
};

/** バイト列 → `Manifest` の唯一の変換点（取得元の検証フックの内側で呼ばれる）。 */
const decodeManifest = (bytes: Uint8Array): Manifest => {
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (error) {
    throw new ManifestFormatError(`manifest: ${MANIFEST_FILENAME} が UTF-8 として読めない`, {
      cause: error,
    });
  }
  return parseManifest(text);
};

/**
 * `karume.json` を取得して parse する。世代の解決はここで 1 回だけ行い、**取得元ごと**返り値に
 * 載せる（{@link fetchAssets} 以降はその取得元・その世代で取得する）。
 *
 * 第 1 引数は HF のリポ参照（{@link HubRepoRef}）か、取得元そのもの
 * （`localDirectory(...)` 等が返す不透明ハンドル）。前者は HF 取得元の省略記法。
 *
 * セッションの入口でもあるので、ここで旧名前空間（`karume/1` 系）を 1 回だけ回収する
 * （取得元に関わらず — 旧版の hub が残した写しは、今どの取得元を使っていても不要）。
 *
 * NOTE: manifest は資産と違い**期待 sha256 を事前に持てない**（正本の根なので）。したがって
 * キーは SHA 固定 resolve URL のままで、`parse` = UTF-8 decode + parse がバイト列 →
 * `Manifest` の唯一の変換点として残る（資産側の検証だけが取得元へ移った）。
 */
export const loadManifest = async (
  ref: HubRepoRef | DistributionSource,
  options: LoadManifestOptions = {},
): Promise<LoadedManifest> => {
  await purgeLegacyCaches(options);
  // 取得元の判別は**同一性**で行う（`source.ts` — ブランド欄も構造判別も持たせない）。
  const driver = driverOf(ref instanceof DistributionSource ? ref : createHfSource(ref));
  let generation: string;
  try {
    generation = await driver.resolveGeneration(options);
  } catch (error) {
    if (isAborted(error, options.signal)) throw error;
    throw revisionResolutionFailure(driver.origin, error);
  }
  // MUST: SHA 固定 URL のキャッシュヒットは network に出ない＝取得元の signal 監視が効かないので、
  // 取得の前後で明示的に中断を見る（見ないと中断済みの signal で呼んでも manifest が返り、
  // 取り消したはずのロードがそのまま先へ進む）。
  options.signal?.throwIfAborted();
  const pinned = driver.pin(generation, options);
  let manifest: Manifest | undefined;
  try {
    await pinned.readManifest({
      parse: (bytes) => {
        manifest = decodeManifest(bytes);
      },
      ...(options.signal === undefined ? {} : { signal: options.signal }),
      sizeViolation: manifestOversize(pinned.origin),
    });
  } catch (error) {
    if (error instanceof HubError || isAborted(error, options.signal)) throw error;
    throw manifestFetchFailure(pinned.origin, error);
  }
  // 取得を抜けた直後にも見る（この後は同期の組み立てだけなので、これが返却前の最後の関門）。
  options.signal?.throwIfAborted();
  if (manifest === undefined) {
    throw new Error(
      `hub: ${MANIFEST_FILENAME} の検証フックが走っていない（取得層の不変条件破れ）`,
    );
  }
  // 取得元は**この値が運ぶ**（識別欄から組み立て直さない — 復元手段の無い取得元が入れられない）。
  return new LoadedManifest(manifest, { driver, generation }, pinned.origin);
};

/**
 * 解決済みファイル表を取得する。取得と進捗総量は **path で一意化**され、同じ path を指す
 * 複数のキーには同一のバイト列が入る。
 *
 * 検証（size / sha256）は取得元へ委ねる — manifest の `sha256` / `size` を期待値として渡すので、
 * network 取得は受信中に照合され、通ったエントリには記録ハッシュが焼かれる。以後のヒットは記録
 * との文字列比較だけで済み（全量ハッシュ 0 回）、記録が食い違う・記録が無いのに実ハッシュが
 * 合わないエントリは evict → 取り直し（self-heal）になる。
 */
export const fetchAssets = async (
  loaded: LoadedManifest,
  files: ResolvedFiles,
  options: FetchAssetsOptions = {},
): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const source = pinnedSourceOf(loaded, options);
  const context = createFetchContext(loaded, source.origin);

  const keys = Object.keys(files);
  // MUST: 一意化は path ではなく {@link fileRefKey} で行う — 越境参照が入った以上、別リポの
  // 同名 path は別のバイト列であり、path で畳むと片方の bytes がもう片方に配られる。
  const unique = new Map<string, FileRef>();
  for (const key of keys) {
    const ref = files[key];
    const refKey = fileRefKey(ref);
    if (!unique.has(refKey)) unique.set(refKey, ref);
  }
  const targets = [...unique.values()];
  const progress = createProgressEmitter(unique, options.onProgress);

  const failure = new AbortController();
  const signal = options.signal === undefined
    ? failure.signal
    : AbortSignal.any([failure.signal, options.signal]);
  const bytesByRef = new Map<string, Uint8Array<ArrayBuffer>>();

  const fetchOne = async (ref: FileRef): Promise<void> => {
    // MUST: キャッシュヒットは network に出ない＝取得元の signal 監視が効かない区間なので、
    // 取得の前後で明示的に中断を見る（数 GB の読出しを回している最中に取り消しが効かないのは
    // 中断の透過が壊れているのと同じ — 逐次面 streamAssets と同型の確認）。
    signal.throwIfAborted();
    let bytes: Uint8Array;
    try {
      bytes = await sourceForRef(source, ref).readFile(ref, {
        signal,
        onProgress: (received) => progress.downloading(ref, received),
        sizeViolation: context.sizeViolation(ref),
      });
    } catch (error) {
      if (error instanceof HubError || isAborted(error, signal)) throw error;
      throw context.fetchFailure(ref, "取得", error);
    }
    // 取得を抜けた直後にも見る — 前段の確認だけだと「キャッシュ読出しの最中に中断された」形が
    // 観測されず、取り消したはずのファイルが complete まで進む。
    signal.throwIfAborted();
    bytesByRef.set(fileRefKey(ref), assertTightView(bytes, ref.path));
    progress.complete(ref);
  };

  // 送出は「本数」ではなく「in-flight の `ref.size` 合計」で律速する（{@link BYTE_BUDGET}）。
  // 待つのはこのループ 1 本だけなので、`targets`（= resolveFiles の順）の head-of-line
  // blocking がそのまま送出順になる — 後続の小さいファイルに追い越させないので、同じ
  // manifest なら同じ順に出る。
  const admission = createByteAdmission(BYTE_BUDGET);
  const running: Promise<void>[] = [];
  const run = async (ref: FileRef): Promise<void> => {
    try {
      await fetchOne(ref);
    } catch (error) {
      // 1 本でも落ちたら残りを止める（fail loud — 全体が reject するのに DL を続ける意味はない）。
      // MUST: ここで reject を外へ漏らさない — 決着を待つのは全送出の後なので、漏らすと
      // 送出待ちの間に unhandled rejection になる。真の失敗理由は `failure` の reason が持つ。
      failure.abort(error);
    } finally {
      // 成否を問わず席を返す（返さないと後続が永久に待つ）。
      admission.release(ref.size);
    }
  };
  for (const ref of targets) {
    // 失敗・中断の後に新しい取得を起こさない（`fetchOne` 冒頭の確認より前に止める）。
    if (signal.aborted) break;
    await admission.admit(ref.size);
    running.push(run(ref));
  }
  // MUST: 失敗しても送出済み全本の決着を待ってから抜ける（早期 reject だと呼び出し側が
  // catch した後も取得が背後で走り続け、abort の意味が無くなる）。`run` は reject しないので
  // ここは常に成功で返る。
  await Promise.all(running);
  // MUST: 巻き添えの失敗ではなく**真の第一失敗**を上げる。`failure.abort()` は最初の 1 回だけが
  // reason を決めるので、`failure.signal.reason` は必ず最初に落ちた 1 本の理由になる。ワーカーの
  // reject をそのまま拾うと、巻き添え側が先に決着した場合にそちらが表面化する — しかも巻き添えは
  // 中断由来なので `isAborted` で `HubFetchError` に包まれず（生の AbortError）、Chrome では
  // 理由の文言まで固定文言（"BodyStreamBuffer was aborted"）へ差し替えられて真因が消える。
  // 呼び手渡しの外部 signal による中断も、その reason がそのまま `failure` に載るので素通しの
  // ままになる（中断が取得失敗に化けない）。
  if (failure.signal.aborted) throw failure.signal.reason;
  // MUST: 全ファイルがキャッシュ済みの起動は 1 度も network に出ないため、決着後にも中断を見る
  // （この後は同期の組み立てだけなので、これが返却前の最後の関門になる）。
  signal.throwIfAborted();

  let assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  for (const key of keys) {
    const bytes = bytesByRef.get(fileRefKey(files[key]));
    if (bytes === undefined) {
      throw new Error(`hub: ${files[key].path} の bytes が揃っていない（取得層の不変条件破れ）`);
    }
    assets = { ...assets, [key]: bytes };
  }
  return assets;
};

/**
 * {@link streamAssets} が 1 本ずつ引き渡す shard。runtime の `ModelShard` と**構造互換**で、
 * `streamAssets(...)` をそのまま `createSessionFromShards` へ渡せる。
 */
export type StreamedAsset = {
  /**
   * id = manifest の path（`refs` に渡した順で届く）。runtime 側はこれを失敗とフェンスの
   * 帰属先に使う — 到着順の連番では配布形のどのファイルかが決まらない。
   */
  readonly id: string;
  /** 検証（size / sha256）を通ったバイト列。buffer 全体を占める。 */
  readonly bytes: Uint8Array<ArrayBuffer>;
};

/**
 * 相 1 / 逐次面が共通で行う入力検査と進捗の組み立て。**取得元に触れる前**に済ませる
 * （相 1 は全 shard を落としてしまうので、network に出た後で呼び出し側の誤りに気づいても
 * 帯域が戻らない）。
 *
 * `where` は入力検査の文言に載る面の名前（呼び出し側の誤りをどの面で弾いたかが分かる）。
 */
const preparePhase = (
  where: string,
  context: FetchContext,
  refs: readonly FileRef[],
  options: FetchAssetsOptions,
): ProgressEmitter => {
  const available = context.available;
  if (refs.length === 0) {
    throw new ManifestReferenceError(`${where}: 取得対象が 1 つも無い（${context.session}）`, {
      available,
    });
  }
  // 重複検査の表はそのまま進捗の per-file 引き当て（`fileTotal`）にも使う。同一性は
  // {@link fileRefKey}（越境参照は別リポの同名 path を別の 1 本として数える）。
  const declared = new Map<string, FileRef>();
  for (const ref of refs) {
    const refKey = fileRefKey(ref);
    if (declared.has(refKey)) {
      throw new ManifestReferenceError(
        `${where}: 参照 '${refKey}' が重複している（この面は渡された列をそのまま扱う —` +
          ` 同じ shard を 2 回渡すのは呼び出し側の誤り。全量面 fetchAssets は一意化する）`,
        { available },
      );
    }
    declared.set(refKey, ref);
  }
  return createProgressEmitter(declared, options.onProgress);
};

/**
 * 全 ref を「RAM に載せずに、後の全量読みが安く済む状態」にする**相 1**。{@link streamAssets} の
 * 相 1 と {@link prefetchAssets} の本体はこの 1 本（同じ機構を 2 つ書くと、同時取得数・真因の
 * 復元・進捗の綴りが片方だけ直る）。
 *
 * `emitComplete` は「この面が `complete` の発行者か」— 相 2 を持つ逐次面では引き渡しの直前が
 * 終端なので `false`（両方が出すと `downloading`* → `complete` を 1 ファイル 1 回とする
 * `AssetPhase` の契約が破れる）、相 2 を持たない {@link prefetchAssets} では終端がここしか
 * ないので `true`。
 */
const runPrefetchPhase = async (
  source: PinnedSource,
  context: FetchContext,
  refs: readonly FileRef[],
  options: FetchAssetsOptions,
  progress: ProgressEmitter,
  { emitComplete }: { readonly emitComplete: boolean },
): Promise<void> => {
  // 相 1 は取得元の **optional 能力**（`source.ts` ④）。持たない取得元は相 2 の逐次読みだけで
  // 同じ RAM 目標を満たすので、ここは何もせずに抜ける。
  if (source.prefetchFile === undefined) return;

  const failure = new AbortController();
  const prefetchSignal = options.signal === undefined
    ? failure.signal
    : AbortSignal.any([failure.signal, options.signal]);

  const prefetchOne = async (ref: FileRef): Promise<void> => {
    // MUST: 記録ハッシュが一致するエントリは network に出ない＝取得元の signal 監視が効かない
    // 区間なので、ファイルごとに明示的に中断を見る（見ないと、取り消しも第一失敗も温まっている
    // ファイルに対してだけ効かず、残り全 ref を舐め切ってから決着する）。全量面 fetchOne・
    // 相 2 と同じ綴り。
    prefetchSignal.throwIfAborted();
    const origin = sourceForRef(source, ref);
    const prefetchFile = origin.prefetchFile;
    if (prefetchFile === undefined) {
      // 越境先だけが相 1 を持たない形は取得元契約の破れ（`originFor` は同じ取得元の別座標を
      // 返すものであって、能力を落とす口ではない）。
      throw new Error(
        `hub: 越境先の取得元が相 1 を持たない（${fileRefKey(ref)} — 取得元契約の不変条件破れ）`,
      );
    }
    try {
      await prefetchFile(ref, {
        signal: prefetchSignal,
        onProgress: (received) => progress.downloading(ref, received),
        sizeViolation: context.sizeViolation(ref),
      });
    } catch (error) {
      if (error instanceof HubError || isAborted(error, prefetchSignal)) throw error;
      // MUST: 捕まえた値の同一性だけで中断を判定しない — 相 1 はバイト列を手元に持たない
      // 面なので、転送中断も put の reject として現れ、`cause` に沈めて包まれる。signal が
      // 落ちていればその reason（巻き添えなら最初の失敗そのもの）を素通しする。
      // NOTE: 上の `isAborted` を先に通る形（生の AbortError）ではここに来ないので、これだけでは
      //       真因の復元にならない。最終的な決着は相 1 の allSettled の後で reason から取る。
      if (prefetchSignal.aborted) throw prefetchSignal.reason;
      throw context.fetchFailure(ref, "事前取得", error);
    }
    if (emitComplete) progress.complete(ref);
  };

  let next = 0;
  const worker = async (): Promise<void> => {
    try {
      while (next < refs.length) await prefetchOne(refs[next++]);
    } catch (error) {
      // 1 本でも落ちたら残りを止める（fail loud — 全体が reject するのに DL を続ける意味はない）。
      failure.abort(error);
      throw error;
    }
  };
  // MUST: 失敗しても全ワーカーの決着を待ってから抜ける（全量面と同じ理由 — 早期 reject だと
  // 呼び出し側が catch した後も取得が背後で走り続ける）。
  await Promise.allSettled(
    Array.from({ length: Math.min(CONCURRENCY, refs.length) }, () => worker()),
  );
  // MUST: 巻き添えではなく**真の第一失敗**を上げる（全量面と同じ理由）。ワーカーの reject を
  // 配列順に拾うと、真犯人が worker[0] 以外だったときに巻き添え側が表面化する。
  if (failure.signal.aborted) throw failure.signal.reason;
};

/**
 * 資産を**先に温めるだけ**の面（逐次面 {@link streamAssets} の相 1 単体）。重み shard を先に
 * 落としておく面 — 後続の {@link streamAssets} 相 2 は network に出ない。
 *
 * 使いどころは「Session を遅延構築するパイプラインのロード時」— 構築が初回の実行まで遅れると
 * 重み shard の DL もそこまで遅れ、ロード進捗にも現れない。ここで先に落としておけば、進捗は
 * ロード中に出揃い、構築はキャッシュ読出しだけで済む。
 *
 * 機構は逐次面の相 1 と同一（同時 {@link CONCURRENCY} 本・`sha256` は通過中に照合・失敗は
 * 真因を復元して `HubFetchError`）。バイト列は返さない（RAM に載せない面なので、欲しい
 * ときは {@link streamAssets} / {@link fetchAssets} で読み直す — キャッシュヒットになる）。
 *
 * 進捗は `downloading`* に続けて**ファイルごとに `complete` を 1 回**出す（相 2 を伴わない
 * この面が終端 — `AssetPhase` の契約。キャッシュ済みのファイルは `complete` 1 点だけ）。
 *
 * **相 1 を持たない取得元（ローカルディレクトリ）では、入力検査だけを行って何もしない** —
 * 進捗も 1 つも出ない。fail loudly にはしない: この面の約束は「後続の読みが安く済む状態にする」
 * ことで、直接読める取得元では**最初から満たされている**（温めるべきキャッシュが無いのは失敗
 * ではない）。落とすと、取得元を差し替えられるはずのアプリが取得元ごとに分岐する羽目になる。
 *
 * NOTE: HF 取得元では `caches` が無い環境・キャッシュ書込み失敗（quota 超過等）は **fail loud**
 * （バイト列を手元に持たない面なので素 fetch へ縮退する余地が無い）。
 */
export const prefetchAssets = async (
  loaded: LoadedManifest,
  refs: readonly FileRef[],
  options: FetchAssetsOptions = {},
): Promise<void> => {
  const source = pinnedSourceOf(loaded, options);
  const context = createFetchContext(loaded, source.origin);
  const progress = preparePhase("prefetchAssets", context, refs, options);
  await runPrefetchPhase(source, context, refs, options, progress, {
    emitComplete: true,
  });
  // MUST: 全ファイルがキャッシュ済みの呼び出しは 1 度も network に出ない＝取得元の signal 監視が
  // 効かないので、決着後にも中断を見る（これが返却前の最後の関門）。
  options.signal?.throwIfAborted();
};

/**
 * shard を **2 相**で読み、1 本ずつ引き渡す逐次面（ADR 0070 決定 2）。
 *
 * - **相 1（prefetch）**: 最初の yield の前に、全 shard を「後の全量読みが安く済む状態」にする
 *   （HF では streaming で永続キャッシュへ落とす — RAM に全量を載せない・同時 4 本）。`sha256` は
 *   通過中に照合され、不一致はエントリ不成立で fail loud（帯域を捨てた後に全量を握って落ちない）。
 *   通ったエントリには記録ハッシュが焼かれ、既に記録が一致するエントリは network に出ずそのまま
 *   温存される。**相 1 を持たない取得元では丸ごと省かれる**（相 2 だけで同じ RAM 目標を満たす）。
 * - **相 2（逐次引き渡し）**: `refs` の順に 1 本ずつ「取得元から読む → 呼び手へ渡す →
 *   参照を手放す」。相 1 が焼いた記録と期待 sha256 の突合は取得元が行い（全量ハッシュ 0 回）、
 *   記録が食い違う・バイト数が合わないエントリは self-heal で 1 往復だけ取り直す。
 *
 * 全量面 {@link fetchAssets} との違いは**渡したバイト列への参照を残さない**ことで、RAM ピークが
 * O(最大 shard) に収まる（全量ホスト保持が成立しない検収モデル級のための面）。全量面は温存して
 * あるので、小モデルは従来どおりそちらを使う。
 *
 * MUST: 相 1 は**最初の `next()` まで開始されない**（async generator の遅延）。呼んだだけでは
 * 何も起きず、`for await` に入るか `next()` を呼んだ時点で DL が始まる。空の `refs`・重複 path も
 * その時点（取得元に触れる前）に {@link ManifestReferenceError} で弾く。
 *
 * NOTE: HF 取得元の相 1 は `caches` が無い環境・キャッシュ書込み失敗（quota 超過等）で
 * **fail loud** になる（バイト列を手元に持たない面なので素 fetch へ縮退する余地が無い。黙って
 * 縮退させると RAM ピークの目標が壊れる）。`onCacheError` の診断が届くのは相 2 だけで、相 1 の
 * cache I/O 失敗は `HubFetchError` として上がる。
 */
export const streamAssets = async function* (
  loaded: LoadedManifest,
  refs: readonly FileRef[],
  options: StreamAssetsOptions = {},
): AsyncGenerator<StreamedAsset, void, unknown> {
  const source = pinnedSourceOf(loaded, options);
  const context = createFetchContext(loaded, source.origin);

  // ---- 相 1: ここを抜けるまで 1 本も yield しない。入力検査もこの中（取得元に触れる前）で済む。
  // `complete` は相 2 が出すので発行しない。
  const progress = preparePhase("streamAssets", context, refs, options);
  await runPrefetchPhase(source, context, refs, options, progress, { emitComplete: false });

  // ---- 相 2: 1 本ずつ引き渡す。
  for (const ref of refs) {
    // 相 2 は大半がキャッシュ読出しで network に出ない＝取得元の signal 監視が効かない区間なので、
    // shard の切れ目で明示的に中断を見る（数 GB の読出しを何本も回している最中に取り消しが
    // 効かないのは中断の透過が壊れているのと同じ）。
    options.signal?.throwIfAborted();

    let bytes: Uint8Array;
    try {
      bytes = await sourceForRef(source, ref).readFile(ref, {
        ...(options.signal === undefined ? {} : { signal: options.signal }),
        // 相 1 が温めた分はキャッシュヒットなので、ここが発火するのは self-heal の取り直しだけ。
        // NOTE: self-heal は evict してから取り直すため、その 1 巡だけ `loaded` はそのファイル
        //       ぶん巻き戻る（phase 契約が認めている「最初からやり直し」と同じ 1 巡）。
        //       `fileLoaded` も同じ 1 巡だけそのファイルの先頭から数え直しになる。
        onProgress: (received) => progress.downloading(ref, received),
        sizeViolation: context.sizeViolation(ref),
      });
    } catch (error) {
      if (error instanceof HubError || isAborted(error, options.signal)) throw error;
      throw context.fetchFailure(ref, "取得", error);
    }
    // MUST: yield の直前にも中断を見る — 冒頭の確認だけだと「最終 shard のキャッシュ読出しの
    // 最中に中断された」形が観測されず、取り消したはずのロードが正常完了して下流の Session
    // 構築まで走る（検証済みバイトを配ってから止まるのでは中断の意味が無い）。
    options.signal?.throwIfAborted();
    const asset = assertTightView(bytes, ref.path);
    progress.complete(ref);
    // MUST: ここで手放す — 引き渡したバイト列を generator 側の表に溜めない（溜めた瞬間に
    // 全量面と同じ RAM 特性に戻り、この面の存在理由が消える）。次の反復に入れば `bytes` の
    // 束縛ごと到達不能になるので、常駐するのは「今の 1 本」だけ。
    yield { id: ref.path, bytes: asset };
  }
};
