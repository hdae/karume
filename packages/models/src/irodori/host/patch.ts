/**
 * 参照 latent の patch（ADR 0047 決定 4 のホスト残置）。
 *
 * `speaker` グラフの入力は patch 済みの `[1, tokens, latentDim×patchSize]` で、patch 自体は
 * 純粋な reshape なので IR に載せる意味が無い。
 */

/** patch 後の参照 latent。 */
type PatchedReference = {
  readonly data: Float32Array<ArrayBuffer>;
  readonly tokens: number;
  /** `latentDim × patchSize`。 */
  readonly width: number;
};

/**
 * 参照 latent `[frames, latentDim]` を `[frames/patchSize, latentDim×patchSize]` へ畳む。
 *
 * MUST: 端数のフレームは**捨てる**（上流 `patch_sequence_with_mask` の `usable`）。切り上げて
 * 0 で埋めると、存在しない無音フレームが参照話者の特徴として encoder に入る。
 */
export const patchReferenceLatent = (
  latent: Float32Array<ArrayBuffer>,
  latentDim: number,
  patchSize: number,
): PatchedReference => {
  if (latent.length % latentDim !== 0) {
    throw new Error(`参照 latent の長さ ${latent.length} が latentDim ${latentDim} の倍数でない`);
  }
  const frames = latent.length / latentDim;
  const tokens = Math.floor(frames / patchSize);
  if (tokens < 1) {
    throw new Error(
      `参照 latent のフレーム数 ${frames} が patch 幅 ${patchSize} に満たない（1 トークンも作れない）`,
    );
  }
  const width = latentDim * patchSize;
  // 端数を落とした prefix はそのまま行優先で並ぶ（patch は reshape でしかない）。
  return { data: latent.slice(0, tokens * width), tokens, width };
};
