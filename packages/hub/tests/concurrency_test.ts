import { assert, assertEquals, assertRejects } from "@std/assert";
import { createByteAdmission, createSerializer } from "../src/concurrency.ts";

/**
 * バイト予算の観測台。`fetchAssets` の送出ループと同じ形（1 本のループが admit → 送出、
 * 各仕事が終わったら release）で回し、in-flight 合計のピークを記録する。
 */
const dispatch = async (
  maxBytes: number,
  sizes: readonly number[],
): Promise<{ peak: number; order: number[] }> => {
  const admission = createByteAdmission(maxBytes);
  const order: number[] = [];
  const running: Promise<void>[] = [];
  let inFlight = 0;
  let peak = 0;
  for (const [index, size] of sizes.entries()) {
    await admission.admit(size);
    order.push(index);
    inFlight += size;
    peak = Math.max(peak, inFlight);
    running.push((async () => {
      await new Promise((resolve) => setTimeout(resolve, 5));
      inFlight -= size;
      admission.release(size);
    })());
  }
  await Promise.all(running);
  return { peak, order };
};

Deno.test("createByteAdmission: in-flight の合計が予算を超えない", async () => {
  const { peak, order } = await dispatch(100, [40, 40, 40, 40, 40]);
  assert(peak <= 100, `in-flight が ${peak} バイトまで膨らんだ`);
  assert(peak > 40, "1 本ずつしか通していない（予算に収まる 2 本目を待たせている）");
  assertEquals(order.length, 5, "全部が送出されていない");
});

Deno.test("createByteAdmission: in-flight 0 なら予算超の単独要求も通す（詰まらせない）", async () => {
  // 席を返す相手が居ないのに待てば永久に詰まる。予算 100 に対する単独 250 が通ること。
  const { peak, order } = await dispatch(100, [250]);
  assertEquals(order, [0]);
  assertEquals(peak, 250);
});

Deno.test("createByteAdmission: 予算超の巨大要求は他の 1 本とも重ならない", async () => {
  // 「in-flight 0 なら通す」の例外が、周りを巻き込んで予算を二重に破らないこと。
  const { peak } = await dispatch(100, [60, 250, 60]);
  assertEquals(peak, 250, `巨大要求が他と重なって ${peak} バイトになった`);
});

Deno.test("createByteAdmission: release が 2 連続でも待機者を取りこぼさない", async () => {
  const admission = createByteAdmission(100);
  await admission.admit(50);
  await admission.admit(50);
  let admitted = false;
  const pending = admission.admit(50).then(() => {
    admitted = true;
  });
  await Promise.resolve();
  assertEquals(admitted, false, "予算が埋まっているのに通してしまった");
  // 起こしの席は 1 つしか無い。2 連続の release で取りこぼすと待機者は永久に起きない。
  admission.release(50);
  admission.release(50);
  await pending;
  assertEquals(admitted, true, "release を取りこぼして詰まった（lost wakeup）");
});

Deno.test("createSerializer: 同時に 1 本しか走らせない", async () => {
  const serialize = createSerializer();
  let running = 0;
  let peak = 0;
  const task = async (): Promise<void> => {
    running += 1;
    peak = Math.max(peak, running);
    await new Promise((resolve) => setTimeout(resolve, 1));
    running -= 1;
  };
  await Promise.all([serialize(task), serialize(task), serialize(task)]);
  assertEquals(peak, 1, `同時 ${peak} 本走った`);
});

Deno.test("createSerializer: 先着順に実行し、結果は呼び出し元へそのまま返る", async () => {
  const serialize = createSerializer();
  const done: string[] = [];
  const task = (name: string) => async (): Promise<string> => {
    await new Promise((resolve) => setTimeout(resolve, 1));
    done.push(name);
    return name;
  };
  const results = await Promise.all([
    serialize(task("a")),
    serialize(task("b")),
    serialize(task("c")),
  ]);
  assertEquals(done, ["a", "b", "c"]);
  assertEquals(results, ["a", "b", "c"]);
});

Deno.test("createSerializer: 1 本の失敗を後続へ伝染させない", async () => {
  const serialize = createSerializer();
  const failing = serialize(() => Promise.reject(new Error("boom")));
  const following = serialize(() => Promise.resolve("ok"));
  await assertRejects(() => failing, Error, "boom");
  assertEquals(await following, "ok", "先行の失敗が後続を巻き添えにした");
});
