/**
 * The table of graph names the containers of the series output (`outputs/series/`) carry — **this
 * one file is the source of truth**.
 *
 * MUST: tests, tools and examples that read series do not spell graph names; they look them up
 * here. If the assets gate (`e2e_assets_gate_test.ts`) and each e2e kept their own tables, changing
 * only one of them would fail nothing (the gate says "as expected" while the e2e reads another
 * graph).
 *
 * A graph name is the **weights key of the distribution manifest**, not the name of the directory
 * it lives in (container-v1 §2.1). The two are spelled differently — the graph of `caption-proj`
 * is `caption_proj`, the graph of `deberta/full-24layer` is `text_encoder`, and `decoder` /
 * `encoder` of `dacvae` are `codec_decoder` / `codec_encoder`. A rule that derives graph names from
 * directory names would grow per-family exceptions for every reader, so the shape is fixed to a
 * table lookup.
 */

/**
 * Series directory name → component directory name (empty string when the weights sit directly
 * in the series root) → graph name.
 *
 * NOTE: the table lists **the series the real-weight e2e actually open**. `outputs/series/` may
 * hold other series too (retired experiments, the `-probe` of research records), but those without
 * a reader do not need to be in the table.
 */
export const SERIES_GRAPHS: Readonly<Record<string, Readonly<Record<string, string>>>> = {
  "birefnet-hr-1024-f16": { "": "matte" },
  "birefnet-hr-2048-f16": { "": "matte" },
  "lucida-1024": { "": "matte" },
  "lucida-1024-f16": { "": "matte" },
  "lucida-2048": { "": "matte" },
  "lucida-2048-f16": { "": "matte" },
  "dacvae-32dim": { decoder: "codec_decoder", encoder: "codec_encoder" },
  deberta: { "full-24layer": "text_encoder" },
  "deberta-i8": { "full-24layer": "text_encoder", "sbv2-22layer": "text_encoder" },
  "depth-anything-v2-small-hf": { "": "depth" },
  "embeddinggemma-300m": { "": "model" },
  "gemma4-e2b": { "": "model" },
  "gemma4-e2b-decode": { "": "model" },
  "gemma4-e2b-drafter": { "": "drafter" },
  "gemma4-e2b-product": { "": "model" },
  "gemma4-qat-e2b-product": { "": "model" },
  "irodori-v4-small": {
    backbone: "backbone",
    "caption-proj": "caption_proj",
    dit: "dit",
    "dit-context": "dit_context",
    duration: "duration",
    speaker: "speaker",
    "text-proj": "text_proj",
  },
  "minicpm5-1b": { "": "model" },
  "minicpm5-1b-decode": { "": "model" },
  // MOSS-Image's DiT (ADR 0138 decisions 2–4 — `moss.export_dit write`; `dit_skip22` borrows
  // every weight of `dit`).
  "moss-image-sft-i8-dyn": { dit: "dit", dit_skip22: "dit_skip22", dit_text: "dit_text" },
  // MOSS-Image's VAE decoder (ADR 0138 decision 8 — `moss.export_vae write`; a VAE-only series).
  "moss-image-sft-f32-dyn": {
    vae_decoder_pre: "vae_decoder_pre",
    vae_decoder_attn: "vae_decoder_attn",
    vae_decoder_post: "vae_decoder_post",
  },
  "qwen3.5-9b-text-moss-image-sft-i8-dyn": { text_encoder: "text_encoder" },
  "sbv2-F1": { dp: "dp", front: "front", flow: "flow", dec: "dec", voice: "voice" },
  "sbv2-F1-f16": { dp: "dp", front: "front", flow: "flow", dec: "dec", voice: "voice" },
  "sbv2-F1-i8": { dp: "dp", front: "front", flow: "flow", dec: "dec", voice: "voice" },
  "siglip2-base-patch16-224": { "": "vision" },
  "siglip2-so400m-patch14-384": { "": "vision" },
  "umt5-xxl-i8-dyn": { text_encoder: "text_encoder" },
  "vowel-detector-crnn-epoch3": { "": "crnn" },
  "wan2.1-t2v-1.3b-f16-dyn": {
    transformer: "transformer",
    vae_decoder_first: "vae_decoder_first",
    vae_decoder_next: "vae_decoder_next",
  },
  "wan2.1-t2v-1.3b-f16-dyn-probe": { transformer: "transformer" },
  "wan2.1-t2v-1.3b-i8-dyn": { transformer: "transformer" },
  "wan2.2-ti2v-5b-f16-dyn": {
    vae_decoder_first: "vae_decoder_first",
    vae_decoder_next: "vae_decoder_next",
    // The 32-latent pair that borrows the pair above (ADR 0134 — `wan.export_vae --tile 32 --shared`).
    vae_decoder_first_32: "vae_decoder_first_32",
    vae_decoder_next_32: "vae_decoder_next_32",
    vae_encoder_pre: "vae_encoder_pre",
    vae_encoder_attn: "vae_encoder_attn",
    vae_encoder_post: "vae_encoder_post",
  },
  // The int8 series also holds the VAE decoder's two chunk graphs (ADR 0133 decision 7 — written
  // by `wan.export_vae --model ti2v-5b --dtype i8` into the same series root as the DiT).
  "wan2.2-ti2v-5b-i8-dyn": {
    transformer: "transformer",
    vae_decoder_first: "vae_decoder_first",
    vae_decoder_next: "vae_decoder_next",
    vae_decoder_first_32: "vae_decoder_first_32",
    vae_decoder_next_32: "vae_decoder_next_32",
  },
};

/**
 * The series' "component directory → graph name" (fails loudly for a series not in the table).
 *
 * The entry point for e2e that enumerate targets and check them against the table (an equality
 * check on `Object.keys(...)`).
 */
export const seriesComponents = (series: string): Readonly<Record<string, string>> => {
  if (!Object.hasOwn(SERIES_GRAPHS, series)) {
    throw new Error(
      `series ${series} is not in the graph-name table (helpers/series-graphs.ts). ` +
        `Known series: ${Object.keys(SERIES_GRAPHS).join(" / ")}.`,
    );
  }
  return SERIES_GRAPHS[series];
};

/**
 * The graph name the container of a series + component directory carries (the component is
 * omitted when the weights sit directly in the root).
 *
 * MUST: a combination not in the table fails loudly. Passing `undefined` to `prepareContainer` is
 * stopped by the type checker, but a form that **silently** passes a misspelling **as another
 * graph name** (a default such as `?? "model"`) lets the numbers pass, so it is not written.
 */
export const seriesGraph = (series: string, component = ""): string => {
  const components = seriesComponents(series);
  if (!Object.hasOwn(components, component)) {
    const known = Object.keys(components).map((name) => (name === "" ? "<根直下>" : name));
    throw new Error(
      `series ${series} has no component ${component === "" ? "<根直下>" : component} in the graph-name table` +
        ` (helpers/series-graphs.ts). Known components: ${known.join(" / ")}.`,
    );
  }
  return components[component];
};
