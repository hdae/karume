/**
 * **ディレクトリ 1 つに固定した `CacheStorage`**（RAM ピーク harness 専用）。
 *
 * ## なぜ要るのか
 *
 * cold / warm を測り分けるには「空のキャッシュから始める」「直前の cold が温めたキャッシュを
 * そのまま使う」を**プロセスを跨いで**選べなければならない（1 構成 = 1 プロセス）。Deno の
 * 組み込み `caches` は `$DENO_DIR/location_data/web_cache` に固定で、置き場を選ぶ口は起動時の
 * `DENO_DIR` しかない — 計測プロセスの中から「この構成ぶんのキャッシュ」を指せない。
 * hub / 取得層（`@hdae/fetch-cache`）はどの面にも `caches` の差し替え口を持つ（`LoadManifestOptions`
 * の `caches`）ので、そこへ**このディレクトリ実装**を挿す。
 *
 * ## 実装が満たすべき 3 点
 *
 * 1. **`match` は本文をヒープに載せない**（`Deno.FsFile.readable` をそのまま `Response` の body に
 *    する）。全量を `Uint8Array` で返す実装にすると、取得層の区間読み（"stream" 戦略 —
 *    `fetch-cache` は Deno では既定でこちら）が part 長ぶんのヒープを毎回踏み、**測ろうとしている
 *    ピークそのものを harness が作る**。
 * 2. **ヘッダごと保持する**。取得層は記録ハッシュ（`x-fetch-cache-sha256`）をレスポンスヘッダへ
 *    焼き、warm のヒット判定を文字列比較だけで済ませる。ヘッダを落とすと warm が「記録の無い
 *    エントリ」になり、実環境では起きない全量再ハッシュ経路を測ることになる（検収③が壊れる）。
 * 3. **キー → ファイル名の写像に `crypto.subtle.digest` を使わない**。harness は digest の回数を
 *    数える（検収③）ので、キャッシュ実装が digest を呼ぶと計数が自分の道具で汚れる。ここは
 *    非暗号ハッシュ（FNV-1a 64bit）+ 可読な尾部で綴る。
 *
 * MUST: 使い捨ての計測用であって、`@karume/hub` の永続キャッシュの代わりではない（LRU も容量
 * 管理も無い）。置き場は `--cache-dir` が指す消して安全なディレクトリだけに向ける。
 */

/** キャッシュ I/O の実績（cold / warm の判別に使う — 検収③の傍証）。 */
export type DirectoryCacheStats = {
  /** `CacheStorage.open` の回数。 */
  readonly opens: number;
  /** `Cache.match` の回数。 */
  readonly matches: number;
  /** そのうち在庫が在った回数。 */
  readonly hits: number;
  /** そのうち在庫が無かった回数。 */
  readonly misses: number;
  /** `Cache.put` の回数（= 真実源から落としてきた本数）。 */
  readonly puts: number;
  /** `Cache.put` で書いた総バイト数。 */
  readonly putBytes: number;
  /** `Cache.delete` の回数。 */
  readonly deletes: number;
};

/** 実績を数えられるディレクトリ固定の `CacheStorage`。 */
export type DirectoryCacheStorage = CacheStorage & {
  /** 現時点の実績（呼んだ瞬間の写し）。 */
  readonly stats: () => DirectoryCacheStats;
};

/** 可変の集計器（{@link DirectoryCacheStats} と同じ欄を可変で持つ）。 */
type Tally = {
  opens: number;
  matches: number;
  hits: number;
  misses: number;
  puts: number;
  putBytes: number;
  deletes: number;
};

/** エントリの添え状（本文と対で置く — これが在る = `put` が決着した印）。 */
type EntryMeta = {
  /** 元のキー（`Cache.keys` が返す `Request` の URL）。 */
  readonly url: string;
  /** `put` されたレスポンスヘッダ（記録ハッシュを含む）。 */
  readonly headers: readonly (readonly [string, string])[];
};

const encoder = new TextEncoder();

/**
 * FNV-1a 64bit（非暗号・決定的）。プロセスを跨いで同じキーが同じファイル名になることだけが
 * 要件で、衝突耐性は要らない（衝突しても「別の URL が同じ在庫を指す」ではなく、添え状の
 * `url` 照合で miss に落ちる）。
 */
const fnv1a64 = (text: string): string => {
  let hash = 0xcbf29ce484222325n;
  for (const byte of encoder.encode(text)) {
    hash = BigInt.asUintN(64, (hash ^ BigInt(byte)) * 0x100000001b3n);
  }
  return hash.toString(16).padStart(16, "0");
};

/** ファイル名に使える尾部（人が `ls` で読めるようにするだけ — 一意性は担っていない）。 */
const readableTail = (url: string): string => {
  const tail = url.slice(url.lastIndexOf("/") + 1).replace(/[^A-Za-z0-9._-]/g, "_");
  return tail.length === 0 ? "entry" : tail.slice(0, 48);
};

const urlOf = (request: RequestInfo | URL): string =>
  typeof request === "string" ? request : request instanceof URL ? request.href : request.url;

/**
 * 名前空間 1 つ（= ディレクトリ 1 つ）。`Cache` のうち hub / 取得層が使う 4 本
 * （`match` / `put` / `delete` / `keys`）だけを実装し、残りは呼ばれたら落とす（黙って
 * 何もしない実装にすると、使われ始めたときに「効かないキャッシュ」として静かに劣化する）。
 */
class DirectoryCache implements Cache {
  readonly #dir: string;
  readonly #tally: Tally;

  constructor(dir: string, tally: Tally) {
    this.#dir = dir;
    this.#tally = tally;
    Deno.mkdirSync(dir, { recursive: true });
  }

  #base(request: RequestInfo | URL): string {
    const url = urlOf(request);
    return `${this.#dir}/${fnv1a64(url)}-${readableTail(url)}`;
  }

  async #meta(base: string): Promise<EntryMeta | undefined> {
    try {
      // 添え状は小さい（URL + ヘッダ）ので全量読みでよい。
      return JSON.parse(await Deno.readTextFile(`${base}.json`)) as EntryMeta;
    } catch (error) {
      if (error instanceof Deno.errors.NotFound) return undefined;
      throw error;
    }
  }

  async match(request: RequestInfo | URL): Promise<Response | undefined> {
    this.#tally.matches += 1;
    const base = this.#base(request);
    const meta = await this.#meta(base);
    if (meta === undefined || meta.url !== urlOf(request)) {
      this.#tally.misses += 1;
      return undefined;
    }
    let file: Deno.FsFile;
    try {
      file = await Deno.open(`${base}.bin`);
    } catch (error) {
      if (!(error instanceof Deno.errors.NotFound)) throw error;
      // 添え状だけが残った形（本文の据え替えに失敗した残骸）は在庫なしと同じに畳む。
      this.#tally.misses += 1;
      return undefined;
    }
    this.#tally.hits += 1;
    // 同名ヘッダの重複も保つ（`append` で積む — 実 Cache API はヘッダごと格納する）。
    const headers = new Headers();
    for (const [name, value] of meta.headers) headers.append(name, value);
    // MUST: body はファイルのストリームのまま返す（全量を読み出すとこの harness が測ろうと
    // しているホスト RAM のピークを harness 自身が作る — モジュール doc の①）。
    return new Response(file.readable, { headers });
  }

  async put(request: RequestInfo | URL, response: Response): Promise<void> {
    const url = urlOf(request);
    const base = this.#base(url);
    const partial = `${base}.partial`;
    // MUST: 上書きの put では古い添え状を先に消す。残すと本文の据え替えから新しい添え状までの間、
    // 古いヘッダ（古い記録ハッシュ）と新しい本文の組を在庫として返す。
    await Deno.remove(`${base}.json`).catch((error: unknown) => {
      if (!(error instanceof Deno.errors.NotFound)) throw error;
    });
    let written = 0;
    const counted = new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, controller) {
        written += chunk.byteLength;
        controller.enqueue(chunk);
      },
    });
    // body を持たない応答（0 バイト）も同じ経路へ寄せる（分岐を 2 本持たない）。
    const body = response.body ?? new Blob([await response.arrayBuffer()]).stream();
    const file = await Deno.open(partial, { create: true, write: true, truncate: true });
    try {
      // `pipeTo` は成功でも中断でも writable を閉じる（= fd を閉じる）ので、ここで close しない。
      await body.pipeThrough(counted).pipeTo(file.writable);
    } catch (error) {
      await Deno.remove(partial).catch(() => {});
      throw error;
    }
    // 本文 → 添え状の順で据える。添え状が在ることが「決着した」印なので、途中で落ちた put は
    // 本文の残骸を残すだけで、在庫としては見えない（`match` は添え状から入る）。
    await Deno.rename(partial, `${base}.bin`);
    const meta: EntryMeta = { url, headers: [...response.headers] };
    // 添え状も一時名に書いてから据える（書きかけの JSON を `match` に読ませない）。
    await Deno.writeTextFile(`${base}.json.partial`, JSON.stringify(meta));
    await Deno.rename(`${base}.json.partial`, `${base}.json`);
    this.#tally.puts += 1;
    this.#tally.putBytes += written;
  }

  async delete(request: RequestInfo | URL): Promise<boolean> {
    this.#tally.deletes += 1;
    const base = this.#base(request);
    let existed = false;
    for (const suffix of [".json", ".bin"]) {
      try {
        await Deno.remove(`${base}${suffix}`);
        existed = true;
      } catch (error) {
        if (!(error instanceof Deno.errors.NotFound)) throw error;
      }
    }
    return existed;
  }

  async keys(request?: RequestInfo | URL): Promise<readonly Request[]> {
    if (request !== undefined) {
      const meta = await this.#meta(this.#base(request));
      return meta === undefined ? [] : [new Request(meta.url)];
    }
    const found: Request[] = [];
    for await (const entry of Deno.readDir(this.#dir)) {
      if (!entry.isFile || !entry.name.endsWith(".json")) continue;
      const meta = JSON.parse(await Deno.readTextFile(`${this.#dir}/${entry.name}`)) as EntryMeta;
      found.push(new Request(meta.url));
    }
    return found;
  }

  add(): Promise<void> {
    throw new Error("ram-peak/dir-cache: Cache.add は使わない");
  }

  addAll(): Promise<void> {
    throw new Error("ram-peak/dir-cache: Cache.addAll は使わない");
  }

  matchAll(): Promise<readonly Response[]> {
    throw new Error("ram-peak/dir-cache: Cache.matchAll は使わない");
  }
}

/**
 * `root` の下だけを使う `CacheStorage` を作る（名前空間 = 直下のディレクトリ 1 つ）。
 *
 * 実ディレクトリは `open` の時に作る。空にしたいときは呼び手が `root` ごと消す
 * （{@link resetCacheDirectory}）。
 */
export const directoryCaches = (root: string): DirectoryCacheStorage => {
  const tally: Tally = {
    opens: 0,
    matches: 0,
    hits: 0,
    misses: 0,
    puts: 0,
    putBytes: 0,
    deletes: 0,
  };
  const opened = new Map<string, DirectoryCache>();
  const dirOf = (cacheName: string): string =>
    `${root}/${cacheName.replace(/[^A-Za-z0-9._-]/g, "_")}`;
  return {
    open: (cacheName: string): Promise<Cache> => {
      tally.opens += 1;
      const existing = opened.get(cacheName);
      if (existing !== undefined) return Promise.resolve(existing);
      const created = new DirectoryCache(dirOf(cacheName), tally);
      opened.set(cacheName, created);
      return Promise.resolve(created);
    },
    has: async (cacheName: string): Promise<boolean> => {
      try {
        return (await Deno.stat(dirOf(cacheName))).isDirectory;
      } catch (error) {
        if (error instanceof Deno.errors.NotFound) return false;
        throw error;
      }
    },
    delete: async (cacheName: string): Promise<boolean> => {
      opened.delete(cacheName);
      try {
        await Deno.remove(dirOf(cacheName), { recursive: true });
        return true;
      } catch (error) {
        if (error instanceof Deno.errors.NotFound) return false;
        throw error;
      }
    },
    keys: async (): Promise<string[]> => {
      const names: string[] = [];
      try {
        for await (const entry of Deno.readDir(root)) {
          if (entry.isDirectory) names.push(entry.name);
        }
      } catch (error) {
        if (!(error instanceof Deno.errors.NotFound)) throw error;
      }
      return names;
    },
    match: (): Promise<Response | undefined> => {
      throw new Error("ram-peak/dir-cache: CacheStorage.match は使わない");
    },
    stats: (): DirectoryCacheStats => ({ ...tally }),
  };
};

/**
 * キャッシュ置き場を**空の状態にする**（cold の前処理）。消してから作り直すので、直前の run が
 * 残した在庫は 1 件も残らない。
 */
export const resetCacheDirectory = async (root: string): Promise<void> => {
  await Deno.remove(root, { recursive: true }).catch((error: unknown) => {
    if (!(error instanceof Deno.errors.NotFound)) throw error;
  });
  await Deno.mkdir(root, { recursive: true });
};
