import { describe, expect, it } from "vite-plus/test";

import { holdWakeLock, type WakeLockApi, type WakeLockLock } from "../src/wake-lock.ts";

/** A lock the fake browser grants: `browserRelease` is the browser releasing it (the page hidden). */
const fakeLock = () => {
  const listeners: (() => void)[] = [];
  let released = false;
  const fire = (): void => {
    if (released) return;
    released = true;
    for (const listener of listeners) listener();
  };
  const lock: WakeLockLock = {
    release: () => {
      fire();
      return Promise.resolve();
    },
    addEventListener: (_type, listener) => listeners.push(listener),
  };
  return { lock, browserRelease: fire, released: () => released };
};

/** A fake `navigator.wakeLock` whose requests are settled by hand, in order. */
const fakeApi = () => {
  const requests: {
    readonly grant: () => ReturnType<typeof fakeLock>;
    readonly refuse: () => void;
  }[] = [];
  const api: WakeLockApi = {
    request: () =>
      new Promise((resolve, reject) => {
        requests.push({
          grant: () => {
            const made = fakeLock();
            resolve(made.lock);
            return made;
          },
          refuse: () => reject(new Error("NotAllowedError")),
        });
      }),
  };
  return { api, requests };
};

const settle = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

describe("holdWakeLock", () => {
  it("is held once the browser grants the lock, and follows the browser releasing it", async () => {
    const { api, requests } = fakeApi();
    const hold = holdWakeLock(api);
    expect(hold.state()).toBe("not-held");
    const granted = requests[0]?.grant();
    await settle();
    expect(hold.state()).toBe("held");
    granted?.browserRelease();
    expect(hold.state()).toBe("not-held");
  });

  it("asks again when renewed after the browser released it, and not while it holds one or one is pending", async () => {
    const { api, requests } = fakeApi();
    const hold = holdWakeLock(api);
    hold.renew();
    expect(requests).toHaveLength(1);
    const first = requests[0]?.grant();
    await settle();
    hold.renew();
    expect(requests).toHaveLength(1);
    first?.browserRelease();
    hold.renew();
    expect(requests).toHaveLength(2);
    requests[1]?.grant();
    await settle();
    expect(hold.state()).toBe("held");
  });

  it("is failed when the browser refuses, and held again when a later request is granted", async () => {
    const { api, requests } = fakeApi();
    const hold = holdWakeLock(api);
    requests[0]?.refuse();
    await settle();
    expect(hold.state()).toBe("failed");
    hold.renew();
    requests[1]?.grant();
    await settle();
    expect(hold.state()).toBe("held");
  });

  it("releases its lock when the run ends, keeps the state it ended with, and releases a late grant at once", async () => {
    const { api, requests } = fakeApi();
    const hold = holdWakeLock(api);
    const granted = requests[0]?.grant();
    await settle();
    hold.release();
    expect(granted?.released()).toBe(true);
    expect(hold.state()).toBe("held");
    hold.renew();
    expect(requests).toHaveLength(1);

    const late = fakeApi();
    const ended = holdWakeLock(late.api);
    ended.release();
    const grantedLate = late.requests[0]?.grant();
    await settle();
    expect(grantedLate?.released()).toBe(true);
  });

  it("is unavailable without the API", () => {
    expect(holdWakeLock(undefined).state()).toBe("unavailable");
  });
});
