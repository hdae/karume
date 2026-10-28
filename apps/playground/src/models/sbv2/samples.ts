/**
 * The Japanese sample texts of the Style-Bert-VITS2 page. Japanese is the material itself here (the
 * model reads Japanese only), so this is the one file of the page that holds Japanese text.
 */

export type SampleText = {
  readonly id: string;
  /** What the select shows (English). */
  readonly label: string;
  readonly text: string;
};

export const SAMPLE_TEXTS: readonly SampleText[] = [
  {
    // The sentence of the SBV2 WAV regression test (packages/e2e/tests/e2e_sbv2_wav_test.ts): with
    // seed 0 and the distribution's defaults, a run is the same request the test makes.
    id: "reference",
    label: "Regression-test sentence (short)",
    text: "こんにちは、これはテストです。",
  },
  {
    id: "expressive",
    label: "Exclamations and a question (styles show most here)",
    text: "えっ、本当に？ずっと探していた鍵が、こんなところにあったなんて！もう、信じられないよ。",
  },
  {
    id: "narration",
    label: "Calm narration (longer)",
    text: "今日は朝から雨が降っていました。駅までの道はいつもより静かで、傘に当たる雨の音だけが聞こえていました。",
  },
];

/** The text the page starts with. */
export const DEFAULT_SAMPLE_TEXT: SampleText = SAMPLE_TEXTS[0];
