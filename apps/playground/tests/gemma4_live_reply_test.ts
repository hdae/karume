import { describe, expect, it } from "vite-plus/test";

import { createLiveReplyStore, type FlushScheduler } from "../src/models/gemma4/live-reply.ts";

/** A scheduler the test flushes by hand (in place of an animation frame). */
const manualScheduler = () => {
  const queued: (() => void)[] = [];
  let cancelled = 0;
  const schedule: FlushScheduler = (flush) => {
    queued.push(flush);
    return () => {
      cancelled += 1;
      queued.splice(queued.indexOf(flush), 1);
    };
  };
  return {
    schedule,
    pending: () => queued.length,
    cancelled: () => cancelled,
    runFrame: () => {
      for (const flush of queued.splice(0)) flush();
    },
  };
};

describe("createLiveReplyStore", () => {
  it("coalesces updates into one notification per frame, publishing only the latest", () => {
    const frames = manualScheduler();
    const store = createLiveReplyStore(frames.schedule);
    let notified = 0;
    store.subscribe(() => (notified += 1));
    store.update({ text: "He" });
    store.update({ text: "Hell" });
    store.update({ text: "Hello", stage: "decode 3 / 8 tokens" });
    expect(frames.pending()).toBe(1);
    expect(store.snapshot()).toEqual({ text: "" });
    expect(notified).toBe(0);
    frames.runFrame();
    expect(notified).toBe(1);
    expect(store.snapshot()).toEqual({ text: "Hello", stage: "decode 3 / 8 tokens" });
  });

  it("keeps the same snapshot object between flushes", () => {
    const frames = manualScheduler();
    const store = createLiveReplyStore(frames.schedule);
    store.update({ text: "a" });
    frames.runFrame();
    const first = store.snapshot();
    store.update({ text: "ab" });
    expect(store.snapshot()).toBe(first);
  });

  it("resets at once and drops a pending update", () => {
    const frames = manualScheduler();
    const store = createLiveReplyStore(frames.schedule);
    let notified = 0;
    store.subscribe(() => (notified += 1));
    store.update({ text: "a" });
    frames.runFrame();
    store.update({ text: "ab" });
    store.reset();
    expect(store.snapshot()).toEqual({ text: "" });
    expect(notified).toBe(2);
    expect(frames.cancelled()).toBe(1);
    frames.runFrame();
    expect(store.snapshot()).toEqual({ text: "" });
  });

  it("stops notifying an unsubscribed listener and cancels a pending flush on dispose", () => {
    const frames = manualScheduler();
    const store = createLiveReplyStore(frames.schedule);
    let notified = 0;
    const unsubscribe = store.subscribe(() => (notified += 1));
    unsubscribe();
    store.update({ text: "a" });
    store.dispose();
    expect(frames.pending()).toBe(0);
    expect(notified).toBe(0);
  });
});
