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
 * - 同時取得の律速は面ごとに違う（全量面はバイト予算・温め面は本数 4）。
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
import {
  type FetchAssetsOptions,
  type HubRepoRef,
  LoadedManifest,
  type LoadManifestOptions,
  pinnedSourceOf,
} from "./session.ts";
import {
  type AssetRangeReader,
  type DistributionSource,
  driverOf,
  isDistributionSource,
  type PinnedSource,
  sourceForRef,
} from "./source.ts";
import { createHfSource } from "./sources/hf.ts";

/**
 * 温め面 {@link prefetchAssets}（相 1）の同時取得数（数十コンポーネントの manifest で接続を
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
 * 取得層が返す bytes は buffer 全体を占めていなければならない。理由は面ごとに 3 つ:
 *
 * - 全量面（`fetchAssets`）: 呼び手は `bytes.buffer` を全量 `ArrayBuffer` としてパーサへ渡す
 *   （slice で辻褄を合わせると RAM ピークが倍増する）。
 * - 区間読み口（`openAsset` と、`openContainerSource` の seek 経路）: `AssetRangeReader.read` の
 *   MUST（`source.ts`）どおり区間ちょうどの器を返す — 呼び手は `bytes.buffer` を写さずに区間
 *   そのものとして使え、byteOffset 0 なので整列要件のある view もそのまま作れる。
 * - 容器面（`openContainerSource` の scan 経路）: block はこの器の view として切り出すので、器が
 *   tight であることが「byteOffset = block.offset（64 の倍数）」— scale の Float32Array view に
 *   要る 4 B 整列 — の成立条件になる。
 *
 * SharedArrayBuffer 背面はここで弾く（述語が主張する `Uint8Array<ArrayBuffer>` を型の上でも嘘に
 * しない）。
 */
const isTightView = (bytes: Uint8Array): bytes is Uint8Array<ArrayBuffer> =>
  bytes.buffer instanceof ArrayBuffer && bytes.byteOffset === 0 &&
  bytes.byteLength === bytes.buffer.byteLength;

export const assertTightView = (bytes: Uint8Array, path: string): Uint8Array<ArrayBuffer> => {
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
  const driver = driverOf(isDistributionSource(ref) ? ref : createHfSource(ref));
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
 * 「キー → ファイル参照」の表を取得する。取得と進捗総量は **{@link fileRefKey} で一意化**され、
 * 同じ実体を指す複数のキーには同一のバイト列が入る。キーの綴りは**呼び手が決める** — hub は
 * 表を受け取って同じキーで返すだけで、意味を解釈しない。
 *
 * 検証（size / sha256）は取得元へ委ねる — manifest の `sha256` / `size` を期待値として渡すので、
 * network 取得は受信中に照合され、通ったエントリには記録ハッシュが焼かれる。以後のヒットは記録
 * との文字列比較だけで済み（全量ハッシュ 0 回）、記録が食い違う・記録が無いのに実ハッシュが
 * 合わないエントリは evict → 取り直し（self-heal）になる。
 */
export const fetchAssets = async (
  loaded: LoadedManifest,
  files: Readonly<Record<string, FileRef>>,
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
    // 中断の透過が壊れているのと同じ — 温め面 prefetchAssets と同型の確認）。
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
  // 待つのはこのループ 1 本だけなので、`targets`（= 渡された表の順）の head-of-line
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

  // 表は一度だけ組む。fromEntries は __proto__ も通常の own property にする。
  return Object.fromEntries(keys.map((key): readonly [string, Uint8Array<ArrayBuffer>] => {
    const bytes = bytesByRef.get(fileRefKey(files[key]));
    if (bytes === undefined) {
      throw new Error(`hub: ${files[key].path} の bytes が揃っていない（取得層の不変条件破れ）`);
    }
    return [key, bytes];
  }));
};

/**
 * 相 1 の入力検査と進捗の組み立て。**取得元に触れる前**に済ませる（相 1 は渡された全 ref を
 * 落としてしまうので、network に出た後で呼び出し側の誤りに気づいても帯域が戻らない）。
 */
const preparePhase = (
  context: FetchContext,
  refs: readonly FileRef[],
  options: FetchAssetsOptions,
): ProgressEmitter => {
  const available = context.available;
  if (refs.length === 0) {
    throw new ManifestReferenceError(
      `prefetchAssets: 取得対象が 1 つも無い（${context.session}）`,
      { available },
    );
  }
  // 重複検査の表はそのまま進捗の per-file 引き当て（`fileTotal`）にも使う。同一性は
  // {@link fileRefKey}（越境参照は別リポの同名 path を別の 1 本として数える）。
  const declared = new Map<string, FileRef>();
  for (const ref of refs) {
    const refKey = fileRefKey(ref);
    if (declared.has(refKey)) {
      throw new ManifestReferenceError(
        `prefetchAssets: 参照 '${refKey}' が重複している（この面は渡された列をそのまま扱う —` +
          ` 同じ参照を 2 回渡すのは呼び出し側の誤り。全量面 fetchAssets は一意化する）`,
        { available },
      );
    }
    declared.set(refKey, ref);
  }
  return createProgressEmitter(declared, options.onProgress);
};

/**
 * 全 ref を「RAM に載せずに、後の全量読みが安く済む状態」にする**相 1**（{@link prefetchAssets}
 * の本体）。
 *
 * MUST: 相 1 の能力は**ref ごとに**見る（`source.ts` ④は取得元ごとの optional 能力で、越境参照は
 * セッションと違う取得元から来る）。セッションの取得元だけで決めると、ローカルセッション +
 * 越境先が HF という正当な構成で越境ぶんの温めが丸ごと飛び、進捗も `signal` も届かないまま
 * 実際の取得が後の読みの中（`openFile` の温め直し）で無音のまま走る。
 */
const runPrefetchPhase = async (
  source: PinnedSource,
  context: FetchContext,
  refs: readonly FileRef[],
  options: FetchAssetsOptions,
  progress: ProgressEmitter,
): Promise<void> => {
  const failure = new AbortController();
  const prefetchSignal = options.signal === undefined
    ? failure.signal
    : AbortSignal.any([failure.signal, options.signal]);

  const prefetchOne = async (ref: FileRef): Promise<void> => {
    // MUST: 記録ハッシュが一致するエントリは network に出ない＝取得元の signal 監視が効かない
    // 区間なので、ファイルごとに明示的に中断を見る（見ないと、取り消しも第一失敗も温まっている
    // ファイルに対してだけ効かず、残り全 ref を舐め切ってから決着する）。全量面 fetchOne と
    // 同じ綴り。
    prefetchSignal.throwIfAborted();
    let origin: PinnedSource;
    try {
      origin = sourceForRef(source, ref);
    } catch (error) {
      // 未 mapping の越境（`sources/local.ts` の素の Error）は呼び手の設定不足なので、全量面と
      // 同じ文脈付きの失敗にする（面ごとに見え方を変えない）。
      if (error instanceof HubError) throw error;
      throw context.fetchFailure(ref, "事前取得", error);
    }
    const prefetchFile = origin.prefetchFile;
    if (prefetchFile === undefined) {
      // セッションが相 1 を持つのに越境先だけが持たない形は取得元契約の破れ（`originFor` は
      // 同じ取得元の別座標を返すものであって、能力を落とす口ではない）。
      if (source.prefetchFile !== undefined) {
        throw new Error(
          `hub: 越境先の取得元が相 1 を持たない（${fileRefKey(ref)} — 取得元契約の不変条件破れ）`,
        );
      }
      // 相 1 を持たない取得元（ローカルディレクトリ）の ref は温めずに飛ばす — 進捗も出さない
      // （`prefetchAssets` の「何もしない」はこの 1 本ぶんの no-op が並んだ形）。
      return;
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
    // この面が `complete` の発行者（終端はここしかない — `AssetPhase` の契約）。
    progress.complete(ref);
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
 * 資産を**先に温めるだけ**の面（相 1）。重みを先に落としておく面 — 後続の
 * {@link fetchAssets} や容器面（`openContainerSource`）は network に出ない。
 *
 * 使いどころは「Session を遅延構築するパイプラインのロード時」— 構築が初回の実行まで遅れると
 * 重みの DL もそこまで遅れ、ロード進捗にも現れない。ここで先に落としておけば、進捗は
 * ロード中に出揃い、構築はキャッシュ読出しだけで済む。
 *
 * 同時 {@link CONCURRENCY} 本・`sha256` は通過中に照合・失敗は真因を復元して `HubFetchError`。
 * バイト列は返さない（RAM に載せない面なので、欲しいときは {@link fetchAssets} や容器の区間読み
 * （{@link openAsset}）で読み直す — キャッシュヒットになる）。
 *
 * 進捗は `downloading`* に続けて**ファイルごとに `complete` を 1 回**出す（この面が終端 —
 * `AssetPhase` の契約。キャッシュ済みのファイルは `complete` 1 点だけ）。
 *
 * **相 1 を持たない取得元（ローカルディレクトリ）の ref は、入力検査だけを行って何もしない** —
 * その ref の進捗は 1 つも出ない（全 ref がそうなら面ごと no-op）。判定は**ref ごと**なので、
 * ローカルセッション + 越境先が HF という構成では越境ぶんだけが温まる。
 * fail loudly にはしない: この面の約束は「後続の読みが安く済む状態にする」
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
  const progress = preparePhase(context, refs, options);
  await runPrefetchPhase(source, context, refs, options, progress);
  // MUST: 全ファイルがキャッシュ済みの呼び出しは 1 度も network に出ない＝取得元の signal 監視が
  // 効かないので、決着後にも中断を見る（これが返却前の最後の関門）。
  options.signal?.throwIfAborted();
};

/**
 * 資産 1 本の**区間読み口**を開く（`source.ts` ⑧）。全量面 {@link fetchAssets} が「宣言 size を
 * 丸ごと 1 本」を単位にするのに対し、この面は同じ 1 本から
 * `[offset, offset + length)` だけを引く — 数百 MiB の表から数 KB の行だけが要る消費側のための面。
 *
 * **取得元がその能力を持たなければ `undefined`**（位置読みを持たないディレクトリアダプターを
 * 差したローカル取得元がその形）。fail loudly に
 * しないのは相 1（{@link prefetchAssets}）と同じ理由で、持たないことは失敗ではなく能力の差
 * だから — 呼び手は `undefined` を見て全量読みへ倒す（分岐は 1 箇所で済む）。
 *
 * 読み口は**費用の型**を名乗る（{@link AssetRangeReader.cost}）。呼び手はそれで「行読みに
 * 切り替えてよい行数の上限」を変える — `"scan"` の取得元では offset に比例した読み飛ばしが
 * 1 行ごとに乗るので、行数が増えると全量 1 回の方が安くなる。
 *
 * ref の取得元は他の面と同じ解決（越境参照は宣言された (repo, revision) の取得元）で決まる。
 * **開く動作が何に触るかは取得元による**（`source.ts` ⑧）— ローカル取得元は読み口を作るだけで
 * network にもキャッシュにも書かない。HF 取得元は在庫が無い参照に限り、相 1 と同じ温め
 * （全量 DL → 永続キャッシュ）を 1 度だけ挟んでから開く（この呼び出しの `signal` はその温めの
 * 中断に効く）。温め以外のバイト取得は起きず、区間の実体に触れるのは
 * {@link AssetRangeReader.read} を呼んだときだけ。取得の文脈付け（{@link HubError} 系への
 * 包み直し）は通らない: この面は取得ではないので、失敗は取得元の素の `Error` として上がる。
 *
 * この面が見るのは**宣言 `size` の境界だけ**で、全量面の size 門（`sizeViolation` →
 * `IntegrityError`）も sha256 の照合も掛からない（区間だけを読む以上、宣言と照合できる
 * 全量が手元に無い）。実体の破損は読んだ行の値として現れる。
 *
 * 面ごとの作法（`fetch` / `caches` / `headers` / `onCacheError` / `onRetry`）は他の面と同じく
 * **この呼び出しに渡した分だけ**が効く（ADR 0086 決定 1 — 取得元は生成時ではなく `pin` ごとに
 * 作法を受け取る）。ローカル取得元では 1 つも効かない（HTTP 取得元専用の語彙）。
 */
export const openAsset = async (
  loaded: LoadedManifest,
  ref: FileRef,
  options: LoadManifestOptions = {},
): Promise<AssetRangeReader | undefined> => {
  // 中断は口の有無より先に見る（取得元の能力差で中断の見え方が変わらない — 共通層の作法）。
  options.signal?.throwIfAborted();
  const source = sourceForRef(pinnedSourceOf(loaded, options), ref);
  const openFile = source.openFile;
  if (openFile === undefined) return undefined;
  const reader = await openFile(ref, {
    ...(options.signal === undefined ? {} : { signal: options.signal }),
  });
  // tight view の検査も共通層の作法（`source.ts` の MUST）— 取得元ごとに置くと、取得元が
  // 増えるたびに同じ不変条件を書き直すことになる。
  return {
    cost: reader.cost,
    read: async (offset, length, readOptions) =>
      assertTightView(await reader.read(offset, length, readOptions), ref.path),
  };
};
