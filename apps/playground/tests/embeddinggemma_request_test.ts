import type { PipelineSource } from "@karume/local-distributions/browser";
import { describe, expect, it } from "vite-plus/test";

import {
  addRow,
  canReuse,
  type PipelineKey,
  removeRow,
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
