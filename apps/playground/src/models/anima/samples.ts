/**
 * Sample prompts for the Anima page. The models were trained on Danbooru-style tag lists, so a
 * comma-separated list of tags with the quality tags at the end composes more reliably than a
 * sentence.
 */

export type SamplePrompt = {
  readonly id: string;
  readonly label: string;
  readonly prompt: string;
};

const QUALITY_TAGS = "masterpiece, best quality, very aesthetic, absurdres";

export const SAMPLE_PROMPTS: readonly SamplePrompt[] = [
  {
    id: "sunflower-field",
    label: "Sunflower field",
    prompt: `1girl, solo, long hair, white dress, straw hat, sunflower field, summer, blue sky, cloud, wind, looking at viewer, smile, outdoors, ${QUALITY_TAGS}`,
  },
  {
    id: "neon-rain",
    label: "Neon street in the rain",
    prompt: `1girl, solo, short hair, black hoodie, hood down, city street at night, neon lights, rain, wet pavement, reflection, from side, upper body, cinematic lighting, ${QUALITY_TAGS}`,
  },
  {
    id: "autumn-shrine",
    label: "Shrine in autumn",
    prompt: `1girl, solo, miko, red hakama, white kimono, very long black hair, shrine, torii, autumn leaves, falling leaves, closed eyes, serene, upper body, ${QUALITY_TAGS}`,
  },
  {
    id: "sleeping-cat",
    label: "Cat on a windowsill",
    prompt: `no humans, cat, sleeping, windowsill, curtain, sunlight, potted plant, indoors, warm lighting, still life, depth of field, ${QUALITY_TAGS}`,
  },
];
