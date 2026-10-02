/**
 * workgroup storage（WGSL の `var<workgroup>`）の量を、WebGPU が device の上限と比べるのと同じ数え方で
 * 数える葉モジュール。
 *
 * WebGPU の compute パイプライン検証は「entry point が静的に使う `var<workgroup>` の型 T ごとの
 * `roundUp(16, SizeOf(T))` の総和 ≤ `maxComputeWorkgroupStorageSize`」（WebGPU 仕様の
 * GPUProgrammableStage の検証）。変数ごとに 16 バイトへ切り上げるので、要素数の総和にバイト幅を掛けた
 * 値は上限の比較に使えない（小さく出る）。
 */

/** `var<workgroup>` 1 本が上限の比較で占めるバイト数（`roundUp(16, SizeOf(T))`）。 */
export const workgroupVariableBytes = (sizeOf: number): number => Math.ceil(sizeOf / 16) * 16;
