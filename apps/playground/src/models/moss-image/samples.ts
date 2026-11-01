/**
 * Sample prompts for the MOSS-Image page. The model takes natural-language prompts (its text
 * encoder is a chat model), so a descriptive sentence works better than a tag list.
 */

export type SamplePrompt = {
  readonly id: string;
  readonly label: string;
  readonly prompt: string;
};

export const SAMPLE_PROMPTS: readonly SamplePrompt[] = [
  {
    id: "fox-forest",
    label: "Fox in a snowy forest",
    prompt:
      "A red fox sitting in a snowy pine forest at dawn, soft golden light through the trees, breath visible in the cold air, photorealistic, shallow depth of field.",
  },
  {
    id: "reading-nook",
    label: "Reading nook in the rain",
    prompt:
      "A cozy reading nook by a large window, rain streaking down the glass, a warm lamp, a stack of books and a cup of tea on a wooden side table, watercolor illustration.",
  },
  {
    id: "lighthouse",
    label: "Lighthouse at night",
    prompt:
      "A lighthouse on a rocky coast under a starry night sky, its beam sweeping across calm water, long-exposure photograph, deep blue tones.",
  },
  {
    id: "ramen",
    label: "Bowl of ramen",
    prompt:
      "A close-up of a bowl of ramen on a dark wooden table, steam rising, a soft-boiled egg, sliced green onions and nori, food photography, natural window light.",
  },
];
