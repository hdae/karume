/**
 * GPU lab（掃引 → 幾何プロファイルの生成 → 注入して Anima を実行、を 1 ページで回す PoC —
 * ADR 0115 追記決定 6）のローカルサーバ。
 *
 * 配信の形は `tools/llm-speed/browser/server.ts` と同じ（localhost 限定・COOP / COEP / CORP・
 * 起動時に `deno bundle --platform browser`・モデルは根の中に閉じて Range 対応で配る）。
 * 区間配信と根の閉じ込めはそちらの実装をそのまま使う（同じ規則を二重に持たない）。配るのはページ
 * （`browser/index.html`）・bundle（`/main.js`）・`/config.json`・Anima の配布形（`/models/anima/…`）・
 * Wan2.1 の配布形（`/models/wan/…` — ADR 0118 段 9）・Wan2.2 の配布形（`/models/wan22/…` — ADR 0121 段 8・Wan の
 * タブの世代の選択で使う）・umT5 の配布形（`/models/umt5/…` — 2 世代の Wan の manifest の `text_encoder` が越境参照する
 * 先。Wan のタブの GPU 経路が取得元の `crossRepo` で結ぶ — ADR 0119 追記「段 10d の設計」）だけ。
 *
 * 既定の置き場に配布形（`karume.json` を持つディレクトリ）が無くても起動する — 掃引とプロファイルの
 * タブはモデルを使わない。そのときは `/config.json` の `source`（Anima）/ `wanSource`（Wan2.1）/ `wan22Source`
 * （Wan2.2）が null、その配布形の経路は 404 で、そのタブ（Wan2.2 は Wan のタブの世代）は操作を無効にする。
 * umT5 の配布形が無いときは `/models/umt5/…` が
 * 404 で、Wan のタブは事前計算の経路だけで回る（GPU 経路の読み込みが越境先の欠落を名指しで落ちる）。
 * `--source` / `--wan-source` / `--wan22-source` / `--umt5-source` を明示したのに配布形が無いときは起動しない（指定の誤りを
 * 黙ってそのタブ・その経路無しの起動にしない）。
 */
import { containedPath, fileResponse } from "../llm-speed/browser/server.ts";
import { readCheckout } from "../shared/checkout.ts";

const DEFAULT_PORT = 8790;
const DEFAULT_SOURCE = "models/karume-anima";
const DEFAULT_WAN_SOURCE = "models/karume-wan2.1";
const DEFAULT_WAN22_SOURCE = "models/karume-wan2.2";
const DEFAULT_UMT5_SOURCE = "models/karume-umt5-xxl";

const headers = (): Headers =>
  new Headers({
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
    "X-Content-Type-Options": "nosniff",
  });

/** `/config.json` の中身（ページが書き出す JSON にそのまま載る）。 */
export type ServerConfig = {
  readonly revision: string;
  readonly dirty: boolean;
  readonly bundleSha256: string;
  /**
   * Anima の配布形ディレクトリの名前（パスは出さない — 手元の構成をページへ漏らさない）。配布形が
   * 無ければ null（Anima のタブはこれを見て操作を無効にする）。
   */
  readonly source: string | null;
  /** Wan2.1 の配布形ディレクトリの名前（{@link ServerConfig.source} と同じ規則 — Wan のタブが見る）。 */
  readonly wanSource: string | null;
  /** Wan2.2 の配布形ディレクトリの名前（同じ規則 — Wan のタブの世代の選択で Wan2.2 を選んだときに見る）。 */
  readonly wan22Source: string | null;
};

/** 配る配布形の実 path（無いものは undefined — その経路は全て 404）。 */
export type Distributions = {
  readonly anima?: string;
  readonly wan?: string;
  readonly wan22?: string;
  /**
   * umT5 の配布形（Wan の GPU 経路の越境先）。`/config.json` には載せない — ページは Wan の manifest の越境の
   * 宣言を見て `/models/umt5/` を引き、無ければ 404 を名指しで出す（設定の欄を 2 か所に持たない）。
   */
  readonly umt5?: string;
};

/** 配布形を配る経路の前置（`/models/<名前>/`）。 */
const MODEL_ROUTES = ["anima", "wan", "wan22", "umt5"] as const;

/** `distributions` は配布形の実 path（無い配布形の経路 `/models/<名前>/…` は全て 404）。 */
export const createHandler = (
  distributions: Distributions,
  bundle: Uint8Array<ArrayBuffer>,
  config: ServerConfig,
): (req: Request) => Promise<Response> => {
  const page = decodeURIComponent(new URL("browser/index.html", import.meta.url).pathname);
  return async (req) => {
    const url = new URL(req.url);
    if (!["127.0.0.1", "localhost", "[::1]"].includes(url.hostname)) {
      return new Response("Localhost only", { status: 403 });
    }
    if (req.method !== "GET" && req.method !== "HEAD") return new Response(null, { status: 405 });
    try {
      const h = headers();
      if (url.pathname === "/config.json") return Response.json(config, { headers: h });
      if (url.pathname === "/main.js") {
        h.set("Content-Type", "text/javascript");
        return new Response(bundle, { headers: h });
      }
      if (url.pathname === "/") return await fileResponse(req, page);
      for (const name of MODEL_ROUTES) {
        const prefix = `/models/${name}/`;
        const root = distributions[name];
        if (url.pathname.startsWith(prefix) && root !== undefined) {
          return await fileResponse(
            req,
            await containedPath(root, url.pathname.slice(prefix.length)),
          );
        }
      }
      return new Response(null, { status: 404, headers: h });
    } catch (error) {
      if (error instanceof Deno.errors.NotFound) {
        return new Response("Asset not found", { status: 404, headers: headers() });
      }
      console.error(error);
      return new Response("Asset request failed; see server log", {
        status: 400,
        headers: headers(),
      });
    }
  };
};

/** 配布形の実 path（`karume.json` を持つディレクトリでなければ undefined — 無いこと自体は正常）。 */
const findDistribution = async (sourceRoot: string): Promise<string | undefined> => {
  try {
    const realSource = await Deno.realPath(sourceRoot);
    return (await Deno.stat(`${realSource}/karume.json`)).isFile ? realSource : undefined;
  } catch (error) {
    if (error instanceof Deno.errors.NotFound) return undefined;
    throw error;
  }
};

/** `--source` / `--wan-source` などで明示した置き場に配布形が無い（起動を止める）。 */
export class MissingDistributionError extends Error {
  override name = "MissingDistributionError";
}

/**
 * 配布形の実 path を決める。`explicit`（`option` を渡した）なのに無ければ
 * {@link MissingDistributionError} を投げ、既定の置き場に無いだけなら警告して undefined（`tab` のタブ無しで
 * 起動）。
 */
export const resolveDistribution = async (
  source: {
    readonly path: string;
    readonly explicit: boolean;
    /** 置き場を指定するオプション（`--source` / `--wan-source` / `--wan22-source` / `--umt5-source`）。 */
    readonly option: string;
    /** その配布形を使うタブの名前（`Anima` / `Wan` / `Wan (Wan2.2)` / `Wan GPU text encoder`）。 */
    readonly tab: string;
  },
): Promise<string | undefined> => {
  const { path, option, tab } = source;
  const realSource = await findDistribution(path);
  if (realSource !== undefined) return realSource;
  if (source.explicit) {
    throw new MissingDistributionError(
      `${option} ${path} has no karume.json (pass the distribution directory, or omit ${option} to start without the ${tab} tab)`,
    );
  }
  console.warn(
    `${path} has no karume.json — serving without the ${tab} distribution (the ${tab} tab is disabled; pass ${option} to enable it)`,
  );
  return undefined;
};

/** 配布形の実 path → ページへ出す名前（パスは出さない — {@link ServerConfig.source}）。 */
const directoryName = (realPath: string | undefined): string | null =>
  realPath === undefined ? null : realPath.slice(realPath.lastIndexOf("/") + 1);

/** 配布形の置き場の指定（{@link resolveDistribution} の引数）。 */
export type DistributionLocation = Parameters<typeof resolveDistribution>[0];

/**
 * コマンドラインの引数（`--help` 以外）→ port と配布形ごとの置き場。受理するオプションと既定の置き場をテストで縛るため
 * main から切り出した純関数（ファイルシステムは見ない — 置き場の有無は {@link resolveDistribution} が決める）。
 */
export const parseServerArgs = (argv: readonly string[]): {
  readonly port: number;
  readonly locations: Readonly<Record<keyof Distributions, DistributionLocation>>;
} => {
  const args = new Map<string, string>();
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i], value = argv[i + 1];
    if (
      !["--port", "--source", "--wan-source", "--wan22-source", "--umt5-source"].includes(key) ||
      value === undefined ||
      value.startsWith("--") ||
      args.has(key)
    ) throw Error(`Invalid option ${key}`);
    args.set(key, value);
  }
  const port = Number(args.get("--port") ?? DEFAULT_PORT);
  if (!Number.isInteger(port) || port < 1024 || port > 65535) throw Error("Invalid port");
  return {
    port,
    locations: {
      anima: {
        path: args.get("--source") ?? DEFAULT_SOURCE,
        explicit: args.has("--source"),
        option: "--source",
        tab: "Anima",
      },
      wan: {
        path: args.get("--wan-source") ?? DEFAULT_WAN_SOURCE,
        explicit: args.has("--wan-source"),
        option: "--wan-source",
        tab: "Wan",
      },
      wan22: {
        path: args.get("--wan22-source") ?? DEFAULT_WAN22_SOURCE,
        explicit: args.has("--wan22-source"),
        option: "--wan22-source",
        // Wan2.2 は独立のタブではなく Wan のタブの世代の選択 — 文言でタブを取り違えさせない。
        tab: "Wan (Wan2.2)",
      },
      umt5: {
        path: args.get("--umt5-source") ?? DEFAULT_UMT5_SOURCE,
        explicit: args.has("--umt5-source"),
        option: "--umt5-source",
        tab: "Wan GPU text encoder",
      },
    },
  };
};

const main = async (): Promise<void> => {
  if (Deno.args.includes("--help")) {
    console.log(
      `deno task bench:gpu-lab [--port ${DEFAULT_PORT}] [--source ${DEFAULT_SOURCE}] [--wan-source ${DEFAULT_WAN_SOURCE}] [--wan22-source ${DEFAULT_WAN22_SOURCE}] [--umt5-source ${DEFAULT_UMT5_SOURCE}]`,
    );
    return;
  }
  const { port, locations } = parseServerArgs(Deno.args);
  let distributions: Distributions;
  try {
    distributions = {
      anima: await resolveDistribution(locations.anima),
      wan: await resolveDistribution(locations.wan),
      wan22: await resolveDistribution(locations.wan22),
      umt5: await resolveDistribution(locations.umt5),
    };
  } catch (error) {
    if (!(error instanceof MissingDistributionError)) throw error;
    console.error(error.message);
    Deno.exit(1);
  }
  const build = await Deno.makeTempDir({ prefix: "karume-gpu-lab-" });
  try {
    const output = `${build}/main.js`;
    const command = new Deno.Command(Deno.execPath(), {
      args: [
        "bundle",
        "--platform",
        "browser",
        "--output",
        output,
        decodeURIComponent(new URL("browser/main.ts", import.meta.url).pathname),
      ],
      stdout: "inherit",
      stderr: "inherit",
    });
    if (!(await command.output()).success) throw Error("Browser bundle failed: browser/main.ts");
    const bundle = await Deno.readFile(output);
    const config: ServerConfig = {
      ...await readCheckout(),
      bundleSha256: Array.from(
        new Uint8Array(await crypto.subtle.digest("SHA-256", bundle)),
        (v) => v.toString(16).padStart(2, "0"),
      ).join(""),
      source: directoryName(distributions.anima),
      wanSource: directoryName(distributions.wan),
      wan22Source: directoryName(distributions.wan22),
    };
    console.log(`Open http://localhost:${port} in Chrome. Ctrl+C stops the server.`);
    const abort = new AbortController();
    const stop = (): void => abort.abort();
    Deno.addSignalListener("SIGINT", stop);
    try {
      await Deno.serve(
        { hostname: "127.0.0.1", port, signal: abort.signal },
        createHandler(distributions, bundle, config),
      ).finished;
    } finally {
      Deno.removeSignalListener("SIGINT", stop);
    }
  } finally {
    await Deno.remove(build, { recursive: true });
  }
};
if (import.meta.main) await main();
