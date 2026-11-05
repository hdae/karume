/**
 * Sample messages for the chat input. The Japanese one is test material (multi-byte text through
 * the streaming detokenizer), so it stays in Japanese; this is the page's only file with Japanese.
 */

export type SampleMessage = { readonly label: string; readonly text: string };

export const SAMPLE_MESSAGES: readonly SampleMessage[] = [
  { label: "Short answer", text: "Name three primary colors, one per line." },
  {
    label: "Longer answer",
    text: "Explain in about 200 words how a GPU runs many threads at once.",
  },
  { label: "Japanese", text: "日本の四季について、それぞれ一文で説明してください。" },
  // For a chat with the sample tools checked (see sample-tools.ts).
  { label: "Tools: weather", text: "What is the weather like in Kyoto and in Paris right now?" },
  { label: "Tools: math", text: "What is (12.5 + 7) * 3^2 - 4 / 8? Use the calculator." },
  { label: "Tools: time", text: "What time is it in Tokyo and in New York?" },
];
