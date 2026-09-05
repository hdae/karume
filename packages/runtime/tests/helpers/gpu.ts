/**
 * 実 GPU テストの実行可否判定。
 *
 * アダプタが無い環境では `Deno.test({ ignore })` で**明示 SKIP** する（テスト自体を消して
 * 無音で緑にしない）。判定はテスト登録時点で必要なため、モジュール評価時に 1 回だけ行う。
 * `requestAdapter()` が例外を投げる環境は「壊れた環境」なので握り潰さず伝播させる。
 */
const detectAdapter = async (): Promise<boolean> => {
  const gpu: GPU | undefined = navigator.gpu;
  if (gpu === undefined) {
    return false;
  }
  return (await gpu.requestAdapter()) !== null;
};

export const GPU_AVAILABLE: boolean = await detectAdapter();

/**
 * f16 **計算**変種（ADR 0028）が使えるアダプタか。
 *
 * MUST: 列挙は「使える候補」の判定でしかない — 実際に動くかどうかの門は `acquireGpu` の
 * 実走カナリアで、そちらが落ちれば列挙があってもテストは赤になる（denoland/deno#23125）。
 * 重み**格納** f16（ADR 0018）はこの feature を要らないので、格納側のテストは無条件に走る。
 */
const detectShaderF16 = async (): Promise<boolean> => {
  const gpu: GPU | undefined = navigator.gpu;
  if (gpu === undefined) return false;
  const adapter = await gpu.requestAdapter();
  return adapter !== null && adapter.features.has("shader-f16");
};

export const SHADER_F16_AVAILABLE: boolean = await detectShaderF16();

/**
 * GPU 時間診断（ADR 0021）が使えるアダプタか。
 *
 * MUST: 内訳（パイプラインキー別の GPU 実時間）を見るテストは `acquireGpu({gpuTiming: true})`
 * を明示的に渡す。既定は**要求しない**ので、渡し忘れると `lastRunTiming` が undefined に
 * なり、キー検査が「entries が空なら何も見ない」形で**黙って空振りする**。`true` は feature
 * 不在で fail loudly するため、列挙が無い環境ではこのフラグでケースごと SKIP する。
 */
const detectTimestampQuery = async (): Promise<boolean> => {
  const gpu: GPU | undefined = navigator.gpu;
  if (gpu === undefined) return false;
  const adapter = await gpu.requestAdapter();
  return adapter !== null && adapter.features.has("timestamp-query");
};

export const TIMESTAMP_QUERY_AVAILABLE: boolean = await detectTimestampQuery();

/**
 * 内訳（パイプラインキー別の GPU 実時間）を読むテスト用の `acquireGpu` 引数。
 *
 * MUST: キー検査を持つテストはこれを渡す。素の `acquireGpu()` は計測を**要求しない**ので
 * `lastRunTiming` が undefined になり、`entries.length > 0` で守られたキー検査が**黙って
 * 空振りする**（数値だけ見て緑のまま通る）。列挙が無いアダプタでは `false` に落として
 * 数値側の被覆を残す（`true` は feature 不在で fail loudly するため）。
 */
export const TIMING_ACQUIRE_OPTIONS: { readonly gpuTiming: boolean } = {
  gpuTiming: TIMESTAMP_QUERY_AVAILABLE,
};

/**
 * 「アダプタ無しでの全 SKIP」を明示的に許可する opt-out。
 *
 * MUST: 既定は fail loudly（ADR 0005「全ケース SKIP は明示 FAIL」）。GPU テストが全て消えても
 * 緑になる状態は、検証していないことを検証済みと誤読させる無音の見かけ成功そのもの。門番は
 * `tests/gpu_gate_test.ts` に置き、抜けるには意図表明としてこの環境変数を要求する。
 */
export const ALLOW_NO_GPU: boolean = Deno.env.get("KARUME_ALLOW_NO_GPU") === "1";

/**
 * 「`shader-f16` 不在での SKIP」を明示的に許可する opt-out（{@link ALLOW_NO_GPU} と同型）。
 *
 * MUST: 既定は fail loudly。この feature が無い機では f16 計算変種（ADR 0028）のケースが
 * 丸ごと消え、書き出しガードを 1 度も検証しないまま verify が緑になる。機を替えた瞬間に
 * 検証範囲が黙って縮むのを止めるのがこの門（門番は `tests/gpu_gate_test.ts`）。
 */
export const ALLOW_NO_SHADER_F16: boolean = Deno.env.get("KARUME_ALLOW_NO_SHADER_F16") === "1";

/**
 * 「`timestamp-query` 不在での SKIP」を明示的に許可する opt-out（同上）。
 *
 * この feature が無い機では GPU 時間診断（ADR 0021）のケースが消えるだけでなく、
 * `if (keys.length > 0)` で守られたキー検査が**黙って空振り**する（数値だけ見て緑になる）。
 */
export const ALLOW_NO_TIMESTAMP_QUERY: boolean =
  Deno.env.get("KARUME_ALLOW_NO_TIMESTAMP_QUERY") === "1";

if (!GPU_AVAILABLE) {
  console.warn(
    "[karume] GPUAdapter が無いため実 GPU テストを SKIP する（リリース判定は実 GPU 緑が必須）。" +
      "GPU 無し環境で意図的に通すには KARUME_ALLOW_NO_GPU=1 を設定する",
  );
} else {
  if (!SHADER_F16_AVAILABLE) {
    console.warn(
      "[karume] アダプタが 'shader-f16' を列挙しないため f16 計算変種（ADR 0028）の実 GPU " +
        "テストを SKIP する。既定の f32 経路の検証には影響しない。" +
        "この機で意図的に通すには KARUME_ALLOW_NO_SHADER_F16=1 を設定する",
    );
  }
  if (!TIMESTAMP_QUERY_AVAILABLE) {
    console.warn(
      "[karume] アダプタが 'timestamp-query' を列挙しないため GPU 時間診断（ADR 0021）の " +
        "実 GPU テストを SKIP する。数値の検証には影響しない（キー検査だけが落ちる）。" +
        "この機で意図的に通すには KARUME_ALLOW_NO_TIMESTAMP_QUERY=1 を設定する",
    );
  }
}
