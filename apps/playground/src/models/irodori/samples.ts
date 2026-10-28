/**
 * The Japanese material of the Irodori-TTS page: the text to speak and the voice-design captions.
 * Irodori-TTS is a Japanese model, so its demo inputs are Japanese (they are the material the model
 * reads, not prose of the page). Every Japanese string of the page lives in this one file.
 */

/** A voice-design example: an English label for the button, the caption and a text that suits it. */
export type CaptionExample = {
  readonly id: string;
  readonly label: string;
  readonly caption: string;
  readonly text: string;
};

export const CAPTION_EXAMPLES: readonly [CaptionExample, ...CaptionExample[]] = [
  {
    id: "calm-woman",
    label: "Calm woman",
    caption: "落ち着いた大人の女性の声で、ゆっくりと丁寧に話している。",
    text: "皆さま、本日はご来場いただきありがとうございます。開演まで、もうしばらくお待ちください。",
  },
  {
    id: "bright-woman",
    label: "Bright young woman",
    caption: "明るく元気な若い女性の声で、少し早口に話している。",
    text: "ねえ聞いて！ずっと行きたかったお店、やっと予約が取れたんだ。今度一緒に行こうよ！",
  },
  {
    id: "low-man",
    label: "Low, steady man",
    caption: "落ち着いた低めの男性の声で、はっきりと話している。",
    text: "本日の会議は以上です。資料は後ほど共有しますので、ご確認をお願いします。",
  },
  {
    id: "sleepy-child",
    label: "Sleepy child",
    caption: "寝起きでまだ眠そうな、幼い子どもの声。小さな声で、ゆっくりと話している。",
    text: "ふぁあ……おはよう……。まだ眠いよ。あと五分だけ、寝ててもいい？",
  },
];

/** The text the page starts with (the first example's, so text and caption match). */
export const DEFAULT_TEXT = CAPTION_EXAMPLES[0].text;
export const DEFAULT_CAPTION = CAPTION_EXAMPLES[0].caption;
