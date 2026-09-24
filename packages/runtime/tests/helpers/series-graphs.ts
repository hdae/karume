/**
 * 系列出力（`outputs/series/`）の容器が名乗るグラフ名の表 — **この 1 本が正本**。
 *
 * MUST: 系列を読むテスト・tools・examples はグラフ名を綴らず、ここから引く。門番
 * （`assets_gate_test.ts`）と各 e2e が別々に表を持つと、片方だけ書き換えても誰も落ちない
 * 状態になる（門番は「期待どおり」と言い、e2e は別のグラフを読む）。
 *
 * グラフ名は**配布 manifest の weights キー**であって、置き場のディレクトリ名ではない
 * （container-v1 §2.1）。両者は綴りが一致しない — `caption-proj` のグラフは `caption_proj`・
 * `deberta/full-24layer` のグラフは `text_encoder`・`dacvae` の `decoder` / `encoder` は
 * `codec_decoder` / `codec_encoder`。ディレクトリ名からグラフ名を導く規則を書くと、家族
 * ごとの例外が読み手の数だけ生えるので、表を引く形に固定する。
 */

/**
 * 系列ディレクトリ名 → 部品ディレクトリ名（重みが系列の根直下なら空文字）→ グラフ名。
 *
 * NOTE: 載るのは**実重み e2e が実際に開く系列**。`outputs/series/` にはこれ以外の系列
 * （退役した実験・研究記録の `-probe`）も置かれうるが、読み手がいないものは表に要らない。
 */
export const SERIES_GRAPHS: Readonly<Record<string, Readonly<Record<string, string>>>> = {
  "birefnet-hr-1024": { "": "matte" },
  "birefnet-hr-2048": { "": "matte" },
  "lucida-1024": { "": "matte" },
  "lucida-2048": { "": "matte" },
  "dacvae-32dim": { decoder: "codec_decoder", encoder: "codec_encoder" },
  "deberta": { "full-24layer": "text_encoder" },
  "deberta-i8": { "full-24layer": "text_encoder", "sbv2-22layer": "text_encoder" },
  "depth-anything-v2-small-hf": { "": "depth" },
  "embeddinggemma-300m": { "": "model" },
  "gemma4-e2b": { "": "model" },
  "gemma4-e2b-decode": { "": "model" },
  "gemma4-e2b-decode-token": { "": "model" },
  "gemma4-e2b-drafter": { "": "drafter" },
  "gemma4-e2b-product": { "": "model" },
  "gemma4-qat-e2b-product": { "": "model" },
  "irodori-v4-small": {
    backbone: "backbone",
    "caption-proj": "caption_proj",
    dit: "dit",
    duration: "duration",
    speaker: "speaker",
    "text-proj": "text_proj",
  },
  "minicpm5-1b": { "": "model" },
  "minicpm5-1b-decode": { "": "model" },
  "sbv2-F1": { dp: "dp", front: "front", flow: "flow", dec: "dec", voice: "voice" },
  "sbv2-F1-f16": { dp: "dp", front: "front", flow: "flow", dec: "dec", voice: "voice" },
  "sbv2-F1-i8": { dp: "dp", front: "front", flow: "flow", dec: "dec", voice: "voice" },
  "siglip2-base-patch16-224": { "": "vision" },
  "siglip2-so400m-patch14-384": { "": "vision" },
  "vowel-detector-crnn-epoch3": { "": "crnn" },
};

/**
 * その系列の「部品ディレクトリ → グラフ名」（表に無い系列は fail loudly）。
 *
 * ターゲットの列挙と表の突合（`Object.keys(...)` の等値検査）を持つ e2e のための口。
 */
export const seriesComponents = (series: string): Readonly<Record<string, string>> => {
  if (!Object.hasOwn(SERIES_GRAPHS, series)) {
    throw new Error(
      `系列 ${series} がグラフ名の表に無い（helpers/series-graphs.ts）。` +
        `在るのは ${Object.keys(SERIES_GRAPHS).join(" / ")}。`,
    );
  }
  return SERIES_GRAPHS[series];
};

/**
 * 系列 + 部品ディレクトリ（重みが根直下なら省略）の容器が名乗るグラフ名。
 *
 * MUST: 表に無い組み合わせは fail loudly。`undefined` を `prepareContainer` へ渡すと型検査で
 * 止まるが、綴り違いを**別のグラフ名として黙って**渡す形（`?? "model"` のような既定値）は
 * 数値が通ってしまうので作らない。
 */
export const seriesGraph = (series: string, component = ""): string => {
  const components = seriesComponents(series);
  if (!Object.hasOwn(components, component)) {
    const known = Object.keys(components).map((name) => name === "" ? "<根直下>" : name);
    throw new Error(
      `系列 ${series} の部品 ${component === "" ? "<根直下>" : component} がグラフ名の表に無い` +
        `（helpers/series-graphs.ts）。在るのは ${known.join(" / ")}。`,
    );
  }
  return components[component];
};
