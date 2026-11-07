import type { ChatMessage, Rgb8Image } from "@karume/models/gemma";
import { describe, expect, it } from "vite-plus/test";

import { rgb8OfRgba, thumbnailOf } from "../src/models/gemma4/image-input";
import { hasVision } from "../src/models/gemma4/settings";
import { deriveTranscript, type SentTurn, type TranscriptEntry } from "../src/models/gemma4/turns";

const image = (width: number, height: number): Rgb8Image => ({
  data: Uint8Array.from({ length: width * height * 3 }, (_, index) => index % 251),
  width,
  height,
});

describe("rgb8OfRgba", () => {
  it("drops the alpha without compositing", () => {
    const rgba = Uint8ClampedArray.of(1, 2, 3, 0, 4, 5, 6, 255);
    expect(rgb8OfRgba(rgba, 2, 1)).toEqual({
      data: Uint8Array.of(1, 2, 3, 4, 5, 6),
      width: 2,
      height: 1,
    });
  });
});

describe("thumbnailOf", () => {
  it("keeps a small image's size, opaque, with its pixels", () => {
    const { data, width, height } = thumbnailOf(image(2, 2), 256);
    expect([width, height]).toEqual([2, 2]);
    expect([...data.slice(0, 8)]).toEqual([0, 1, 2, 255, 3, 4, 5, 255]);
  });

  it("scales the longest side down to the limit and keeps the aspect", () => {
    const { data, width, height } = thumbnailOf(image(1000, 500), 256);
    expect([width, height]).toEqual([256, 128]);
    expect(data.length).toBe(256 * 128 * 4);
  });

  it("never draws a side of zero", () => {
    const { width, height } = thumbnailOf(image(2000, 1), 256);
    expect([width, height]).toEqual([256, 1]);
  });
});

describe("hasVision", () => {
  const manifest = (weights: Record<string, unknown>) =>
    ({ models: { "12b": { weights } } }) as unknown as Parameters<typeof hasVision>[0];

  it("needs the role on the regular entry point", () => {
    expect(hasVision(manifest({ vision: {} }), { mount: "gemma4-qat", model: "12b" })).toBe(true);
    expect(hasVision(manifest({}), { mount: "gemma4-qat", model: "12b" })).toBe(false);
    expect(hasVision(manifest({ vision: {} }), { mount: "gemma4-qat-mobile", model: "12b" })).toBe(
      false,
    );
    expect(hasVision(manifest({ vision: {} }), { mount: "gemma4-qat", model: "e2b" })).toBe(false);
  });
});

describe("deriveTranscript with images", () => {
  const picture = image(2, 2);
  const sent: SentTurn = { turn: 0, text: "Hi", reply: "Yes.", userId: 1, assistantId: 2 };
  const newId = (): number => 99;
  const history: readonly ChatMessage[] = [
    {
      role: "user",
      content: [
        { type: "image", image: picture },
        { type: "text", text: "Hi" },
      ],
    },
    { role: "assistant", content: "Yes." },
  ];

  it("lines a message with parts up by its text, and carries its images", () => {
    const { entries, kept } = deriveTranscript({
      previous: [],
      history,
      sent,
      dropped: 0,
      newId,
    });
    expect(kept).toBe(true);
    const [user, assistant] = entries as [TranscriptEntry, TranscriptEntry];
    expect(user).toMatchObject({ id: 1, role: "user", text: "Hi", turn: 0 });
    expect(user.images).toEqual([picture]);
    expect(assistant).toMatchObject({ id: 2, role: "assistant", text: "Yes.", turn: 0 });
    expect(assistant.images).toBeUndefined();
  });

  it("a message of images only has empty text", () => {
    const { entries } = deriveTranscript({
      previous: [],
      history: [{ role: "user", content: [{ type: "image", image: picture }] }, history[1]],
      sent: { ...sent, text: "" },
      dropped: 0,
      newId,
    });
    expect(entries[0]).toMatchObject({ id: 1, text: "", images: [picture] });
  });

  it("a string message has no images field", () => {
    const { entries } = deriveTranscript({
      previous: [],
      history: [{ role: "user", content: "Hi" }, history[1]],
      sent,
      dropped: 0,
      newId,
    });
    expect(Object.hasOwn(entries[0], "images")).toBe(false);
  });
});
