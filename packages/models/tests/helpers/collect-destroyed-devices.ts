/**
 * 新しい device を取る前に、破棄済みの device を V8 に回収させる（テストの GPU の取得口が呼ぶ）。
 *
 * NOTE: これは**対症療法**である（利用者の裁定 2026-10-05 — 根本の対処は後回し）。Deno 2.9.6 の
 * `GPUDevice.destroy()` は device に無効の印を立てるだけで VRAM を返さず、device とその資源が返るのは
 * V8 の GC が device のラッパを回収したとき。テストは 1 本ごとに device を作って捨てるので、GC まで
 * 残った破棄済みの device が同じ process の VRAM を食い、後のテストの大きな確保を OOM にする
 * （docs/known-issues.md「Deno: `GPUDevice.destroy()` が VRAM を返さず…」）。
 *
 * NOTE: 保証しないこと:
 * - 回収の時点は V8 / cppgc の実装の挙動で、仕様の保証ではない。2 回の `gc()` で返ることは B570 の
 *   実測（下の {@link GC_SETTLE_MS}）で見ただけで、版が変われば回数も待ちも変わりうる。
 * - 回収されたかは検査しない（DRM クライアントの数を見る門は後回し）。返っていなくても黙って進む。
 * - `gc` は `--v8-flags=--expose-gc` を付けた走行にだけ生える。deno.json の `deno test` のタスクは
 *   全てこのフラグを渡す（`packages/runtime/tests/verify_lanes_test.ts` が門）が、フラグ無しで直に
 *   `deno test` を回した走行では何もしない（緩和が無いだけで、テストの意味は変わらない）。
 * - まだ到達できる GpuContext（変数・closure が握っているもの）は回収されない。
 */

/**
 * `gc()` の後の待ち（ms）。
 *
 * probe（`outputs/diag/dead-device-probe.ts` の段 d — `gc()` → 100 ms → 観測）の実測
 * （`outputs/bench/2026-10-05_dead-device/p1-settle-gc.log`）: 1 回目の後の 100 ms では何も返らず、
 * 2 回目の後の 100 ms で破棄済みの device の DRM クライアントと vram0 / gtt が全て返った（クライアントは
 * 物理デバイスの fd の 1 本だけに戻った）。
 * 待ちは cppgc の finalizer が macrotask を跨いで走る機会を与えるためのもので、100 ms より短い値で
 * 返るかは測っていないので、実測した値より短くしない。
 */
export const GC_SETTLE_MS = 100;

const wait = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * `host.gc` を 2 回呼び、それぞれの後に {@link GC_SETTLE_MS} 待つ。`gc` が無ければ何もしない。
 *
 * `host` は `gc` を持つ物（既定は `globalThis`）。関数そのものではなく持ち主を受けるのは、既定引数が
 * `undefined` を「既定を使う」と読むため — 「gc が無い」を注入で表せるように、無い形は `{}` で渡す。
 */
export const collectDestroyedDevices = async (host: object = globalThis): Promise<void> => {
  const gc = "gc" in host ? host.gc : undefined;
  if (typeof gc !== "function") return;
  // 2 回: probe の実測では 1 回目では返らない。1 回目で GPUDevice が finalize されて GPUQueue への
  // 強参照が外れ、fd を閉じる GPUQueue の回収は次の GC になる、という連鎖で説明がつく（上流のソース
  // からの推論 — 回数だけが実測）。
  gc();
  await wait(GC_SETTLE_MS);
  gc();
  await wait(GC_SETTLE_MS);
};
