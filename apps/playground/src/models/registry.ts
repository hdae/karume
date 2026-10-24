/**
 * The pages of the playground. A new page = one route file under `src/routes/` plus one entry here
 * (the index lists these as cards).
 */

export type ModelPage = {
  /** The route path (must match the file route under `src/routes/`). */
  readonly path: "/wan22" | "/attention-bench";
  readonly title: string;
  readonly summary: string;
  /** The `/models/<name>/` mounts the page reads (see `vite.config.ts`); empty for a kernel page. */
  readonly distributions: readonly string[];
};

export const MODEL_PAGES: readonly ModelPage[] = [
  {
    path: "/wan22",
    title: "Wan2.2 TI2V 5B",
    summary:
      "Text → video at 1280×704 / 704×1280. Checks the presets and the 32-latent VAE tile on a real GPU.",
    distributions: ["wan22", "umt5"],
  },
  {
    path: "/attention-bench",
    title: "Attention route bench",
    summary:
      "Per-kernel GPU time of the two a8 attention routes (S in f16 over three dispatches, and the online kernel) at Wan2.2's shapes. Needs no distribution.",
    distributions: [],
  },
];
