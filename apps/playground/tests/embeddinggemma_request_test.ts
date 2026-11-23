import type { PipelineSource } from "@karume/local-distributions/browser";
import { ModelInputError } from "@karume/models/embeddinggemma";
import { describe, expect, it } from "vite-plus/test";

import type { LoadedImage } from "../src/lib/image.ts";
import type { LoadedAudio } from "../src/models/embeddinggemma/audio.ts";
import { tokenCountView } from "../src/models/embeddinggemma/format.ts";

import {
  addRow,
  canReuse,
  type PipelineKey,
  removeRow,
  rowInput,
  rowTokenCount,
  type RowTokenizer,
  rowTokens,
  runBlocker,
  SAMPLE_ROWS,
  type TextRow,
  updateRow,
} from "../src/models/embeddinggemma/request.ts";

describe("the rows", () => {
  it("opens with a query and two documents, each with its prompt", () => {
    expect(SAMPLE_ROWS.map(({ prompt }) => prompt)).toEqual(["query", "document", "document"]);
    expect(new Set(SAMPLE_ROWS.map(({ id }) => id)).size).toBe(SAMPLE_ROWS.length);
  });

  it("adds an empty row without a prompt, with an id no other row has", () => {
    const rows = addRow(SAMPLE_ROWS);
    expect(rows).toHaveLength(SAMPLE_ROWS.length + 1);
    expect(rows.at(-1)).toEqual({ id: 3, text: "" });
  });

  it("never gives a new row the id of a removed one still in use elsewhere", () => {
    // Removing row 1 leaves ids 0 and 2: the next row must not take 2.
    const rows = addRow(removeRow(SAMPLE_ROWS, 1));
    expect(rows.map(({ id }) => id)).toEqual([0, 2, 3]);
  });

  it("removes the row by id and keeps the last one", () => {
    expect(removeRow(SAMPLE_ROWS, 1).map(({ id }) => id)).toEqual([0, 2]);
    const single: readonly TextRow[] = [{ id: 7, text: "only" }];
    expect(removeRow(single, 7)).toBe(single);
  });

  it("changes the text or the prompt of one row only", () => {
    const edited = updateRow(SAMPLE_ROWS, 2, { text: "changed" });
    expect(edited[2]).toEqual({ id: 2, text: "changed", prompt: "document" });
    expect(edited[0]).toBe(SAMPLE_ROWS[0]);
    expect(updateRow(SAMPLE_ROWS, 0, { prompt: "STS" })[0]?.prompt).toBe("STS");
  });

  it("clears a prompt by leaving the key out (no prompt, no prefix)", () => {
    const cleared = updateRow(SAMPLE_ROWS, 0, { prompt: undefined })[0];
    expect(cleared).toEqual({ id: 0, text: "how do I reset my password" });
    expect(cleared !== undefined && Object.hasOwn(cleared, "prompt")).toBe(false);
  });
});

describe("runBlocker", () => {
  const ready = {
    gpuReady: true,
    manifestReady: true,
    configError: false,
    running: false,
    rows: SAMPLE_ROWS,
  };

  it("can run with the rows as they are, an empty text included (the pipeline accepts it)", () => {
    expect(runBlocker(ready)).toBeUndefined();
    expect(runBlocker({ ...ready, rows: [{ id: 0, text: "" }] })).toBeUndefined();
  });

  it("names a running run, the manifest, the config and the GPU first, in that order", () => {
    expect(runBlocker({ ...ready, running: true, gpuReady: false })).toBe("A run is in progress.");
    expect(runBlocker({ ...ready, manifestReady: false, gpuReady: false })).toBe(
      "The manifest is not loaded.",
    );
    expect(runBlocker({ ...ready, configError: true, gpuReady: false })).toBe(
      "The model's pipelineConfig cannot be read.",
    );
    expect(runBlocker({ ...ready, gpuReady: false })).toBe("The GPU is not ready.");
  });

  it("needs at least one row", () => {
    expect(runBlocker({ ...ready, rows: [] })).toBe("Add a row.");
  });
});

describe("canReuse", () => {
  const gpu = { name: "gpu" };
  const direct: PipelineSource = { kind: "direct" };
  const held: PipelineKey<typeof gpu, PipelineSource> = {
    gpu,
    source: direct,
    model: "text",
    quant: "bf16",
  };

  it("reuses the held pipeline when the GPU, source, model and preset all match", () => {
    expect(canReuse(held, { gpu, source: direct, model: "text", quant: "bf16" })).toBe(true);
  });

  it("builds a new one when any of the four differs, or when nothing is held", () => {
    expect(
      canReuse(held, { gpu: { name: "gpu" }, source: direct, model: "text", quant: "bf16" }),
    ).toBe(false);
    expect(canReuse(held, { gpu, source: { kind: "direct" }, model: "text", quant: "bf16" })).toBe(
      false,
    );
    expect(canReuse(held, { gpu, source: direct, model: "image", quant: "bf16" })).toBe(false);
    expect(canReuse(held, { gpu, source: direct, model: "text", quant: "f32" })).toBe(false);
    expect(canReuse(undefined, { gpu, source: direct, model: "text", quant: "bf16" })).toBe(false);
  });
});

describe("a row's token count", () => {
  /**
   * A stand-in for `tokenize`: one token per word of the prefix and the text, plus <bos> and <eos>
   * (a text row's input is its string; image rows are covered below).
   */
  const calls: { readonly text: string; readonly prompt: string | undefined }[] = [];
  const tokenize: RowTokenizer = (input, options) => {
    if (typeof input !== "string") throw new Error("a text row sends its text as a string");
    const text = input;
    calls.push({ text, prompt: options.prompt });
    if (options.prompt !== undefined && options.prompt !== "query") {
      throw new ModelInputError(`prompt "${options.prompt}" is not one of this model's prompts`);
    }
    const prefix = options.prompt === undefined ? "" : "task: search result | query: ";
    return new Int32Array((prefix + text).split(" ").filter(Boolean).length + 2);
  };

  it("counts what the pipeline would embed for the row's text and prompt", () => {
    calls.length = 0;
    expect(rowTokenCount(tokenize, { id: 0, text: "reset my password", prompt: "query" })).toBe(10);
    expect(rowTokenCount(tokenize, { id: 1, text: "reset my password" })).toBe(5);
    expect(calls).toEqual([
      { text: "reset my password", prompt: "query" },
      { text: "reset my password", prompt: undefined },
    ]);
  });

  it("has no count for a row the pipeline cannot tokenize (an undeclared prompt)", () => {
    expect(rowTokenCount(tokenize, { id: 0, text: "a", prompt: "Query" })).toBeUndefined();
  });

  it("lets any other failure through (a disposed pipeline is a bug, not a row's state)", () => {
    const disposed: RowTokenizer = () => {
      throw new Error("embeddinggemma.Pipeline: cannot tokenize after dispose");
    };
    expect(() => rowTokenCount(disposed, { id: 0, text: "a" })).toThrow("after dispose");
  });

  it("shows the count against the limit and marks only a count over it", () => {
    expect(tokenCountView(12, 2048)).toEqual({ text: "12 / 2048 tokens", overLimit: false });
    expect(tokenCountView(2048, 2048).overLimit).toBe(false);
    expect(tokenCountView(2049, 2048)).toEqual({ text: "2049 / 2048 tokens", overLimit: true });
  });
});

describe("a row with an image", () => {
  const image: LoadedImage = {
    name: "cat.png",
    rgba: { data: new Uint8ClampedArray(2 * 1 * 4), width: 2, height: 1 },
    rgb: { data: new Uint8Array(2 * 1 * 3), width: 2, height: 1 },
  };
  const row: TextRow = { id: 4, text: "a cat", prompt: "query", image };

  it("sends a text row as its string, an image row as its text then its RGB8 image", () => {
    expect(rowInput({ id: 0, text: "a cat" })).toBe("a cat");
    expect(rowInput(row)).toEqual([
      { type: "text", text: "a cat" },
      { type: "image", image: image.rgb },
    ]);
  });

  it("sets and clears a row's image, leaving the key out when cleared", () => {
    const [set] = updateRow([{ id: 4, text: "a cat", prompt: "query" }], 4, { image });
    expect(set).toEqual(row);
    const [cleared] = updateRow([row], 4, { image: undefined });
    expect(cleared).toEqual({ id: 4, text: "a cat", prompt: "query" });
    expect(cleared !== undefined && Object.hasOwn(cleared, "image")).toBe(false);
    // Changing the prompt keeps the image.
    expect(updateRow([row], 4, { prompt: undefined })[0]?.image).toBe(image);
  });

  it("counts the parts the pipeline gets: the text's tokens and the image's soft tokens and markers", () => {
    const seen: unknown[] = [];
    const tokenize: RowTokenizer = (input, options) => {
      seen.push([input, options]);
      if (typeof input === "string") return new Int32Array(2);
      const count = input.reduce(
        (sum, part) => sum + (part.type === "text" ? part.text.split(" ").length : 256 + 2),
        2,
      );
      return new Int32Array(count);
    };
    expect(rowTokenCount(tokenize, row)).toBe(2 + 2 + 258);
    expect(seen).toEqual([[rowInput(row), { prompt: "query" }]]);
  });

  it("has no count when the pipeline refuses the image (a model without the vision tower)", () => {
    const refusing: RowTokenizer = () => {
      throw new ModelInputError(
        "image parts need the vision tower — build the pipeline with { vision: true }",
      );
    };
    expect(rowTokenCount(refusing, row)).toBeUndefined();
  });
});

describe("a row with an audio", () => {
  const audio: LoadedAudio = {
    name: "click.wav",
    audio: { data: new Float32Array(16100), sampleRate: 16000 },
    channels: 2,
  };
  const image: LoadedImage = {
    name: "cat.png",
    rgba: { data: new Uint8ClampedArray(2 * 1 * 4), width: 2, height: 1 },
    rgb: { data: new Uint8Array(2 * 1 * 3), width: 2, height: 1 },
  };
  const row: TextRow = { id: 5, text: "a click", prompt: "query", audio };

  it("sends the text part then the audio part, and the image between them when the row has both", () => {
    expect(rowInput(row)).toEqual([
      { type: "text", text: "a click" },
      { type: "audio", audio: audio.audio },
    ]);
    expect(rowInput({ ...row, image })).toEqual([
      { type: "text", text: "a click" },
      { type: "image", image: image.rgb },
      { type: "audio", audio: audio.audio },
    ]);
  });

  it("sets and clears a row's audio, leaving the key out when cleared, and keeps it across other edits", () => {
    const [set] = updateRow([{ id: 5, text: "a click", prompt: "query" }], 5, { audio });
    expect(set).toEqual(row);
    const [cleared] = updateRow([row], 5, { audio: undefined });
    expect(cleared).toEqual({ id: 5, text: "a click", prompt: "query" });
    expect(cleared !== undefined && Object.hasOwn(cleared, "audio")).toBe(false);
    expect(updateRow([row], 5, { prompt: undefined })[0]?.audio).toBe(audio);
    expect(updateRow([row], 5, { image })[0]?.audio).toBe(audio);
    expect(updateRow([{ ...row, image }], 5, { audio: undefined })[0]?.image).toBe(image);
  });

  it("counts the audio's soft tokens and markers as tokenize gives them", () => {
    const tokenize: RowTokenizer = (input) => {
      if (typeof input === "string") return new Int32Array(2);
      const count = input.reduce(
        (sum, part) => sum + (part.type === "text" ? part.text.split(" ").length : 25 + 2),
        2,
      );
      return new Int32Array(count);
    };
    expect(rowTokens(tokenize, row)).toEqual({ count: 2 + 2 + 27 });
    expect(rowTokenCount(tokenize, row)).toBe(31);
  });

  it("gives the library's refusal of a row it cannot tokenize (an audio over 30 s) word for word", () => {
    const message =
      "embeddinggemma.Pipeline: content part 1 is 480001 samples (30.00 s), over the 480000 (30 s)" +
      " this model reads — nothing is truncated; shorten the audio or split it";
    const refusing: RowTokenizer = () => {
      throw new ModelInputError(message);
    };
    expect(rowTokens(refusing, row)).toEqual({ refusal: message });
    expect(rowTokenCount(refusing, row)).toBeUndefined();
  });
});
