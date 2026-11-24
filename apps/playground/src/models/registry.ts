/**
 * The pages of the playground: the sidebar and the index list these, grouped by {@link PageGroup}.
 * A new page = one route file under `src/routes/` plus one entry here (`tests/registry_test.ts`
 * fails when a route has no entry or an entry has no route).
 */

import type { MountName } from "@karume/local-distributions/mounts";
import type { FileRoutesByPath } from "@tanstack/react-router";
import {
  AudioLinesIcon,
  BoxesIcon,
  ImageIcon,
  type LucideIcon,
  MessageSquareTextIcon,
  MicVocalIcon,
  MountainIcon,
  ScissorsIcon,
  SparklesIcon,
  TextSearchIcon,
  VideoIcon,
} from "lucide-react";

/** The sidebar groups, in display order. */
export const PAGE_GROUPS = ["Video", "Image", "Vision", "Audio", "Text"] as const;
export type PageGroup = (typeof PAGE_GROUPS)[number];

/** Every file route except the index (the route tree is generated from `src/routes/`). */
export type PagePath = Exclude<keyof FileRoutesByPath, "/">;

export type ModelPage = {
  readonly path: PagePath;
  readonly title: string;
  readonly summary: string;
  readonly group: PageGroup;
  readonly icon: LucideIcon;
  /** The `/models/<name>/` mounts the page reads (`@karume/local-distributions/mounts`); empty for a kernel page. */
  readonly mounts: readonly MountName[];
};

export const MODEL_PAGES: readonly ModelPage[] = [
  {
    path: "/wan22",
    title: "Wan2.2 TI2V 5B",
    summary:
      "Text → video at 1280×704 / 704×1280. Checks the presets and the 32-latent VAE tile on a real GPU.",
    group: "Video",
    icon: VideoIcon,
    mounts: ["wan22", "umt5"],
  },
  {
    path: "/anima",
    title: "Anima",
    summary:
      "Text → image with the official Anima models, the third-party fine-tunes and the layer-expanded Anima-2.9B (which share the official text encoder and VAE).",
    group: "Image",
    icon: SparklesIcon,
    mounts: ["anima", "anima-extra", "anima-2.9b"],
  },
  {
    path: "/moss-image",
    title: "MOSS-Image",
    summary:
      "Text → image with MOSS-Image SFT (Qwen3.5 text encoder, int8 DiT, FLUX.2 VAE). Minutes per 1024² image.",
    group: "Image",
    icon: ImageIcon,
    mounts: ["moss-image"],
  },
  {
    path: "/birefnet",
    title: "BiRefNet",
    summary:
      "Background removal: an alpha matte of the foreground, with BiRefNet HR (photos) or Lucida (illustrations).",
    group: "Vision",
    icon: ScissorsIcon,
    mounts: ["birefnet-hr", "lucida"],
  },
  {
    path: "/depth-anything",
    title: "Depth Anything V2",
    summary: "Monocular depth estimation from one image.",
    group: "Vision",
    icon: MountainIcon,
    mounts: ["depth-anything"],
  },
  {
    path: "/siglip2",
    title: "SigLIP 2",
    summary: "Image and text embeddings, compared by similarity (base 224 / so400m 384).",
    group: "Vision",
    icon: BoxesIcon,
    mounts: ["siglip2"],
  },
  {
    path: "/irodori",
    title: "Irodori-TTS",
    summary:
      "Japanese text → speech with the v4 and v4.1 small models, the v4.1 small MeanFlow model and v4-large.",
    group: "Audio",
    icon: AudioLinesIcon,
    mounts: ["irodori-v4", "irodori-v4.1", "irodori-v4.1-mf", "irodori-v4-large"],
  },
  {
    path: "/sbv2",
    title: "Style-Bert-VITS2",
    summary: "Japanese text → speech with the JVNV speakers and their styles.",
    group: "Audio",
    icon: MicVocalIcon,
    mounts: ["sbv2"],
  },
  {
    path: "/gemma4",
    title: "Gemma 4 chat",
    summary:
      "A chat with Gemma 4 (E2B / E4B), its q4_0 QAT line (E2B / E4B / 12B, MTP drafters) or the fixed mobile QAT (E2B / E4B), streaming tokens on the GPU.",
    group: "Text",
    icon: MessageSquareTextIcon,
    mounts: ["gemma4", "gemma4-qat", "gemma4-qat-mobile"],
  },
  {
    path: "/embeddinggemma",
    title: "EmbeddingGemma",
    summary:
      "Text → sentence embedding with EmbeddingGemma 2's text model: a prompt per row, a chosen dimension and the rows' cosine similarity matrix.",
    group: "Text",
    icon: TextSearchIcon,
    mounts: ["embeddinggemma-2"],
  },
];

/**
 * The page whose path is exactly `routeId`, if any. Callers pass the id of the matched route
 * (`useCurrentRouteId`), not the raw pathname, so a trailing slash resolves as the router resolves it.
 */
export const pageAt = (routeId: string): ModelPage | undefined =>
  MODEL_PAGES.find(({ path }) => path === routeId);

/** The pages of each group, in {@link PAGE_GROUPS} order; empty groups are left out. */
export const pagesByGroup = (): readonly {
  readonly group: PageGroup;
  readonly pages: readonly ModelPage[];
}[] =>
  PAGE_GROUPS.map((group) => ({
    group,
    pages: MODEL_PAGES.filter((page) => page.group === group),
  })).filter(({ pages }) => pages.length > 0);
