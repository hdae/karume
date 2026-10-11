/**
 * 参照 latent の patch（ADR 0047 決定 4 のホスト残置）。
 *
 * `speaker` グラフの入力は patch 済みの `[1, tokens, latentDim×patchSize]` で、patch 自体は
 * 純粋な reshape なので IR に載せる意味が無い。
 */

/**
 * `speaker` グラフの記号次元 S（patch 後のトークン数）の下限。export の `Dim(min=MIN_SYM_LENGTH)`
 * （`tools/export-recipes/irodori/export.py` — torch.export の 0/1 特殊化を避ける既定の 2）そのもので、
 * IR には下限が載らないのでホストが写しを持つ。1 トークンの参照は**グラフに食わせられない**。
 */
export const SPEAKER_MIN_TOKENS = 2;

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
 *
 * MUST: 作れたトークンが {@link SPEAKER_MIN_TOKENS} に満たない参照は落とす（patch 幅 × 2 フレーム
 * 未満）。この関数は codec encoder の出力（= モデル出力）も受けるので素の `Error` — 呼び手が渡した
 * latent の経路は `conditioning.ts` の `patchRequestedLatent` が送出型を入力起因へ上げる。
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
  if (tokens < SPEAKER_MIN_TOKENS) {
    throw new Error(
      `参照 latent のフレーム数 ${frames} が patch 幅 ${patchSize} × ${SPEAKER_MIN_TOKENS} に満たない` +
        `（${tokens} トークン — speaker の記号次元の下限は ${SPEAKER_MIN_TOKENS} トークン）`,
    );
  }
  const width = latentDim * patchSize;
  // 端数を落とした prefix はそのまま行優先で並ぶ（patch は reshape でしかない）。
  return { data: latent.slice(0, tokens * width), tokens, width };
};
