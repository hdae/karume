/**
 * codec decoder の**タイル分割**（ホストグルーのみ — ランタイムは 1 行も関わらない）。
 *
 * ## なぜタイルが要るのか
 *
 * decoder は latent `[1,S,32]` を `[1,1,1920S]` へ展開する。S = 750（30 秒）では中間テンソルが
 * 553MB に達し、**`maxStorageBufferBindingSize` の既定 128MiB の機では S ≥ 183 で確保に失敗
 * する**。タイルはこのポータビリティのためのもので、確保天井の高い機では単発 decode がそのまま
 * 通る（`tileFrames` は性能ノブであって、結果を変えるものではない）。
 *
 * ## なぜ halo を捨てればビット一致するのか
 *
 * decoder の主経路には因果層が 1 つも無く、全層が対称 pad か厳密に `L·stride` の
 * `conv_transpose1d` なので、**平行移動同変**である。片側の受容野は実測 13,793 サンプル
 * （= 7.19 latent フレーム）で、{@link IrodoriPipelineConfig.codecHaloFrames} の 8 フレーム
 * （15,360 サンプル）がこれを覆う。karume の `conv1d` / `conv_transpose1d` は出力要素ごとに
 * 固定順（`(ic,k)` 昇順）で縮約する gather 形なので、同じ入力窓なら**縮約順まで同じ**になり、
 * 採用区間は全長実行と 1 ビットも違わない。実 GPU の門
 * （`tests/e2e_irodori_codec_test.ts`）がこれを Uint32 で毎回実測する。
 *
 * 先頭タイルの左端と末尾タイルの右端は**真の境界**なので halo が要らない（グラフのゼロ
 * padding が全長実行の端と同じ役をする）。
 *
 * ## 配置は等間隔 + 末尾スナップ（`anima/tiling.ts` と同じ流儀）
 *
 * タイルの decode 長を**全て `tileFrames` に揃える**ため、末尾タイルの開始位置を
 * `frames − tileFrames` へスナップする。同形なら Session の `PreparedExecutionPlan`
 * （ADR 0042）がタイル間で効き続ける — 形が 1 枚だけ違うと、そのぶんだけ導出をやり直す。
 */

/**
 * `tileFrames` の既定（= 1 回の decode に流す latent フレーム数の上限）。
 *
 * `maxStorageBufferBindingSize` が既定の 128MiB しか無い機でも通る大きさ（S = 183 で確保に
 * 失敗する実測の 1 つ下）。確保天井の高い機では S ≤ 750 が単発で通るので、そこでは
 * {@link IrodoriGenerateRequest.codecTileFrames} に大きい値を渡せばタイルが 1 枚に縮退する。
 */
export const DEFAULT_CODEC_TILE_FRAMES = 182;

/** タイル 1 枚（開始位置と長さは latent フレーム単位 — 1920 の格子が全段で揃う）。 */
export type CodecTile = {
  /** decode する区間の開始フレーム。 */
  readonly start: number;
  /** decode する区間の長さ（フレーム）。 */
  readonly length: number;
  /** 採用区間の開始（タイル先頭からの相対フレーム — 左 halo の幅）。 */
  readonly offset: number;
  /** 採用区間の長さ（フレーム）。全タイルの合計が S になる。 */
  readonly take: number;
};

/** {@link planCodecTiles} の入力（どちらも latent フレーム単位）。 */
export type CodecTilePlanOptions = {
  /** 1 回の decode に流す最大フレーム数。 */
  readonly tileFrames: number;
  /** 採用区間の両側へ足す文脈（{@link IrodoriPipelineConfig.codecHaloFrames}）。 */
  readonly haloFrames: number;
};

/**
 * latent 長 S を、採用区間が隙間なく並ぶタイル列へ割る。
 *
 * `frames <= tileFrames` なら 1 枚（= 単発 decode と同じ実行）。それ以外は
 * `step = tileFrames − 2·haloFrames` フレームずつ採用し、その両側へ halo を足した窓を
 * decode する。窓が latent の外へはみ出す端では halo が縮むが、そこは真の境界なので
 * 文脈が足りないのではなく**要らない**。
 */
export const planCodecTiles = (
  frames: number,
  options: CodecTilePlanOptions,
): readonly CodecTile[] => {
  const { tileFrames, haloFrames } = options;
  if (!Number.isInteger(frames) || frames <= 0) {
    throw new Error(`planCodecTiles: frames ${frames} が正の整数でない`);
  }
  if (!Number.isInteger(haloFrames) || haloFrames < 0) {
    throw new Error(`planCodecTiles: haloFrames ${haloFrames} が非負整数でない`);
  }
  if (!Number.isInteger(tileFrames) || tileFrames <= 2 * haloFrames) {
    throw new Error(
      `planCodecTiles: tileFrames ${tileFrames} が halo 2 枚ぶん（${2 * haloFrames}）` +
        "より大きい整数でない（採用できるフレームが 1 枚も残らない）",
    );
  }
  if (frames <= tileFrames) return [{ start: 0, length: frames, offset: 0, take: frames }];

  const step = tileFrames - 2 * haloFrames;
  const tiles: CodecTile[] = [];
  for (let takeStart = 0; takeStart < frames; takeStart += step) {
    const takeEnd = Math.min(frames, takeStart + step);
    // 左へ halo ぶん下げ、latent の両端へはみ出す場合だけ内側へスナップする。
    const start = Math.min(Math.max(0, takeStart - haloFrames), frames - tileFrames);
    tiles.push({
      start,
      length: tileFrames,
      offset: takeStart - start,
      take: takeEnd - takeStart,
    });
  }
  assertTileContext(tiles, frames, haloFrames);
  return tiles;
};

/** タイル 1 枚を decode する（`[1,length,latentDim]` → `[1,1,hopLength·length]` の生波形）。 */
export type CodecTileRunner = (
  latent: Float32Array<ArrayBuffer>,
  frames: number,
) => Promise<Float32Array>;

/** {@link decodeTiles} が要る幾何（`pipelineConfig` の 2 欄 + 計画）。 */
export type CodecDecodeOptions = {
  readonly latentDim: number;
  readonly hopLength: number;
  readonly tiles: readonly CodecTile[];
};

/**
 * タイルを順に decode して**全長ぶん**の波形へ貼り合わせる。
 *
 * `run` を注入にしてあるのは、この貼り合わせが「単発 decode とビット一致する」ことを実 GPU の
 * 門が**この関数そのもの**に対して測れるようにするため（テストが同じループを写経すると、
 * 写しが正しいだけで本体が誤っていても緑になる）。
 */
export const decodeTiles = async (
  latent: Float32Array<ArrayBuffer>,
  options: CodecDecodeOptions,
  run: CodecTileRunner,
): Promise<Float32Array<ArrayBuffer>> => {
  const { latentDim, hopLength, tiles } = options;
  if (latent.length % latentDim !== 0) {
    throw new Error(
      `decodeTiles: latent の要素数 ${latent.length} が latentDim ${latentDim} で割れない`,
    );
  }
  const frames = latent.length / latentDim;
  const waveform = new Float32Array(frames * hopLength) as Float32Array<ArrayBuffer>;
  for (const tile of tiles) {
    // MUST: 写して渡す（view のまま渡すと、バッファ全体を占めることを前提にした受け渡しの
    // 契約から外れる）。
    const slice = latent.slice(
      tile.start * latentDim,
      (tile.start + tile.length) * latentDim,
    ) as Float32Array<ArrayBuffer>;
    const decoded = await run(slice, tile.length);
    if (decoded.length !== tile.length * hopLength) {
      throw new Error(
        `decodeTiles: decoder の出力が ${decoded.length} サンプル` +
          `（${tile.length}×${hopLength} = ${tile.length * hopLength} のはず）`,
      );
    }
    waveform.set(
      decoded.subarray(tile.offset * hopLength, (tile.offset + tile.take) * hopLength),
      (tile.start + tile.offset) * hopLength,
    );
  }
  return waveform;
};

/**
 * 「採用区間の両側に halo ぶんの文脈があるか、そこが真の境界か」を確かめる。
 *
 * MUST: 落とさない。この条件だけがビット一致の根拠で、破れても値は**それらしいまま**
 * （継ぎ目付近が数 LSB 違うだけ）出てくる。幾何を触ったときに気づける席がここしか無い。
 */
const assertTileContext = (
  tiles: readonly CodecTile[],
  frames: number,
  haloFrames: number,
): void => {
  let covered = 0;
  for (const tile of tiles) {
    const takeStart = tile.start + tile.offset;
    const takeEnd = takeStart + tile.take;
    if (takeStart !== covered) {
      throw new Error(`planCodecTiles: 採用区間が ${covered} で途切れた（次は ${takeStart}）`);
    }
    if (tile.start + tile.length > frames) {
      throw new Error(
        `planCodecTiles: タイル [${tile.start}, ${
          tile.start + tile.length
        }) が S ${frames} を超えた`,
      );
    }
    if (tile.start > 0 && tile.offset < haloFrames) {
      throw new Error(
        `planCodecTiles: 左の文脈が ${tile.offset} フレームしかない（halo ${haloFrames} が要る）`,
      );
    }
    const right = tile.start + tile.length - takeEnd;
    if (tile.start + tile.length < frames && right < haloFrames) {
      throw new Error(
        `planCodecTiles: 右の文脈が ${right} フレームしかない（halo ${haloFrames} が要る）`,
      );
    }
    covered = takeEnd;
  }
  if (covered !== frames) {
    throw new Error(`planCodecTiles: 採用区間の合計 ${covered} が S ${frames} と違う`);
  }
};
