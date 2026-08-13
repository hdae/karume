// WAV（RIFF）の読み書きの挙動テスト。GPU も実資産も要らない。
//
// `decodeWav` は参照音声を食う唯一の入口で、**規約の取り違えが例外にならない**（int16 の
// スケールが 1 ずれても「少し音量が違う波形」が出るだけ）。したがって値そのものを固定する。
//
// 資産があれば golden `outputs/series/dacvae-32dim/host/meta.json` の `wavScale`（上流の
// リーダが int16 の両端で実測した値）とも突き合わせる（無ければその 1 本だけ SKIP）。

import { assertEquals, assertThrows } from "@std/assert";
import { decodeWav, encodeWav } from "../src/audio/wav.ts";

/** golden `meta.json` の置き場（`dacvae_host.py` の既定の出力先）。 */
const HOST_GOLDEN = new URL(
  "../../../outputs/series/dacvae-32dim/host/meta.json",
  import.meta.url,
);
const HOST_COMMAND =
  "cd tools/exporter && uv run --with descript-audiotools --with einops --with 'transformers==5.14.1' python dacvae_host.py";

const goldenText = await Deno.readTextFile(HOST_GOLDEN).catch(() => undefined);
if (goldenText === undefined) {
  console.warn(
    `[karume] wav のスケール突合を SKIP する（${HOST_GOLDEN.pathname} が要る）。生成: ${HOST_COMMAND}`,
  );
}

const ascii = (bytes: Uint8Array, offset: number, text: string): void => {
  for (let i = 0; i < text.length; i += 1) bytes[offset + i] = text.charCodeAt(i);
};

/**
 * 任意の format / bit 深度 / チャネル数の WAV を組む（異常系を 1 点ずつ壊すための土台）。
 * `payload` は data チャンクの中身そのもの。
 */
const buildWav = (options: {
  readonly format: number;
  readonly channels: number;
  readonly sampleRate: number;
  readonly bits: number;
  readonly payload: Uint8Array;
  /** `fmt ` の前に挟む未知チャンク（読み飛ばしの検査用）。 */
  readonly extraChunk?: { readonly id: string; readonly length: number };
}): Uint8Array<ArrayBuffer> => {
  const { format, channels, sampleRate, bits, payload, extraChunk } = options;
  const extraBytes = extraChunk === undefined ? 0 : 8 + extraChunk.length + (extraChunk.length % 2);
  const total = 12 + extraBytes + 8 + 16 + 8 + payload.length;
  const bytes = new Uint8Array(total);
  const view = new DataView(bytes.buffer);
  ascii(bytes, 0, "RIFF");
  view.setUint32(4, total - 8, true);
  ascii(bytes, 8, "WAVE");
  let cursor = 12;
  if (extraChunk !== undefined) {
    ascii(bytes, cursor, extraChunk.id);
    view.setUint32(cursor + 4, extraChunk.length, true);
    cursor += extraBytes;
  }
  ascii(bytes, cursor, "fmt ");
  view.setUint32(cursor + 4, 16, true);
  view.setUint16(cursor + 8, format, true);
  view.setUint16(cursor + 10, channels, true);
  view.setUint32(cursor + 12, sampleRate, true);
  view.setUint32(cursor + 16, (sampleRate * channels * bits) / 8, true);
  view.setUint16(cursor + 20, (channels * bits) / 8, true);
  view.setUint16(cursor + 22, bits, true);
  cursor += 24;
  ascii(bytes, cursor, "data");
  view.setUint32(cursor + 4, payload.length, true);
  bytes.set(payload, cursor + 8);
  return bytes;
};

const int16Payload = (values: readonly number[]): Uint8Array => {
  const bytes = new Uint8Array(values.length * 2);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setInt16(index * 2, value, true));
  return bytes;
};

const float32Payload = (values: readonly number[]): Uint8Array => {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setFloat32(index * 4, value, true));
  return bytes;
};

// ---- int16 のスケール（規約そのもの）------------------------------------

Deno.test("decodeWav: int16 は 32768 で割る（下端 −32768 が厳密に −1.0）", () => {
  const probe = [-32768, -32767, -1, 0, 1, 16384, 32767];
  const decoded = decodeWav(
    buildWav({ format: 1, channels: 1, sampleRate: 48000, bits: 16, payload: int16Payload(probe) }),
  );
  assertEquals(decoded.sampleRate, 48000);
  assertEquals(
    Array.from(decoded.data),
    [-1, -0.999969482421875, -3.0517578125e-5, 0, 3.0517578125e-5, 0.5, 0.999969482421875],
    "int16 → f32 のスケールが /32768 でない（/32767 との差は実音声では 3e-5 しか出ず、" +
      "golden との突合では tolerance に埋もれて見えない）",
  );
});

Deno.test({
  name: "decodeWav: int16 のスケールが golden の上流リーダ実測（wavScale）と一致する",
  ignore: goldenText === undefined,
  fn: () => {
    const meta = JSON.parse(goldenText as string) as {
      readonly wavScale: {
        readonly probeInt16: readonly number[];
        readonly probeFloat: readonly number[];
      };
    };
    const decoded = decodeWav(
      buildWav({
        format: 1,
        channels: 1,
        sampleRate: 48000,
        bits: 16,
        payload: int16Payload(meta.wavScale.probeInt16),
      }),
    );
    assertEquals(
      Array.from(decoded.data),
      [...meta.wavScale.probeFloat],
      "上流（soundfile）の int16 正規化と綴りが違う",
    );
  },
});

// ---- 往復（読みと書きのスケールが非対称であること自体を固定する）--------

Deno.test("encodeWav → decodeWav: 往復は ×32767 と /32768 の非対称で 1LSB 級ずれる", () => {
  const original = Float32Array.of(-1, -0.5, 0, 0.5, 1);
  const decoded = decodeWav(encodeWav(original, 48000));
  assertEquals(decoded.sampleRate, 48000);
  assertEquals(decoded.data.length, original.length);
  // フルスケールは 32767 で書かれ 32768 で読まれるので、**厳密には戻らない**。ここが揃って
  // いないと「聴き比べ」の相手（torch 台本）か参照音声の読み手のどちらかと綴りが割れる。
  assertEquals(decoded.data[0], -0.999969482421875, "−1.0 の往復");
  assertEquals(decoded.data[4], 0.999969482421875, "+1.0 の往復");
  // 0 だけは厳密に往復する（スケールが非対称でも 0 は 0）。
  assertEquals(decoded.data[2], 0);
  // 誤差は全域で 16bit の 1LSB（1/32768）以内に収まる。
  for (let i = 0; i < original.length; i += 1) {
    const error = Math.abs(decoded.data[i] - original[i]);
    assertEquals(error <= 1 / 32768, true, `要素 ${i} の往復誤差 ${error} が 1LSB を超えた`);
  }
});

// ---- f32 WAV / ステレオ --------------------------------------------------

Deno.test("decodeWav: IEEE float 32bit は恒等（丸めも割り算も挟まない）", () => {
  // 値は全て f32 で厳密に表せるものを選ぶ（丸めが挟まらないことを見たいので）。
  const values = [-1, -0.3333333432674408, 0, 3.0517578125e-5, 0.75, 2.5];
  const decoded = decodeWav(
    buildWav({
      format: 3,
      channels: 1,
      sampleRate: 44100,
      bits: 32,
      payload: float32Payload(values),
    }),
  );
  assertEquals(decoded.sampleRate, 44100);
  // 値域外（2.5）も切り詰めない — 正規化は呼び出し側の段。
  assertEquals(Array.from(decoded.data), values);
});

Deno.test("decodeWav: 複数チャネルはチャネル平均で mono 化する", () => {
  // 2ch のインターリーブ（L, R, L, R, …）。
  const decoded = decodeWav(
    buildWav({
      format: 3,
      channels: 2,
      sampleRate: 48000,
      bits: 32,
      payload: float32Payload([1, 0, 0.5, -0.5, -1, -1]),
    }),
  );
  assertEquals(Array.from(decoded.data), [0.5, 0, -1]);
});

Deno.test("decodeWav: 未知チャンク（LIST 等）は読み飛ばす", () => {
  const decoded = decodeWav(
    buildWav({
      format: 1,
      channels: 1,
      sampleRate: 48000,
      bits: 16,
      payload: int16Payload([16384]),
      // 奇数長 — RIFF の 1 バイト詰めを跨げないと次のチャンク境界を見失う。
      extraChunk: { id: "LIST", length: 7 },
    }),
  );
  assertEquals(Array.from(decoded.data), [0.5]);
});

// ---- fail loudly ---------------------------------------------------------

Deno.test("decodeWav: RIFF/WAVE でないバイト列は落とす", () => {
  const bytes = new Uint8Array(64);
  ascii(bytes, 0, "RIFX");
  ascii(bytes, 8, "WAVE");
  assertThrows(() => decodeWav(bytes), Error, "RIFF/WAVE ヘッダでない");
  assertThrows(() => decodeWav(new Uint8Array(8)), Error, "バイトしかない");
});

Deno.test("decodeWav: 未対応の format / bit 深度は落とす（黙って近似しない）", () => {
  // 24bit PCM。
  assertThrows(
    () =>
      decodeWav(
        buildWav({
          format: 1,
          channels: 1,
          sampleRate: 48000,
          bits: 24,
          payload: new Uint8Array(6),
        }),
      ),
    Error,
    "format 1 / 24bit に未対応",
  );
  // A-law。
  assertThrows(
    () =>
      decodeWav(
        buildWav({ format: 6, channels: 1, sampleRate: 8000, bits: 8, payload: new Uint8Array(4) }),
      ),
    Error,
    "format 6 / 8bit に未対応",
  );
  // WAVE_FORMAT_EXTENSIBLE は SubFormat を読まないと種別が決まらないので受理しない。
  assertThrows(
    () =>
      decodeWav(
        buildWav({
          format: 0xfffe,
          channels: 1,
          sampleRate: 48000,
          bits: 16,
          payload: new Uint8Array(4),
        }),
      ),
    Error,
    "に未対応",
  );
});

Deno.test("decodeWav: 'data' が無い / 長さが宣言と食い違うファイルは落とす", () => {
  const complete = buildWav({
    format: 1,
    channels: 1,
    sampleRate: 48000,
    bits: 16,
    payload: int16Payload([1, 2, 3, 4]),
  });
  // data チャンクの宣言長だけを実体より大きくする（末尾 8 バイト = 'data' + 長さ の直後）。
  const truncated = complete.slice(0, complete.length - 2) as Uint8Array<ArrayBuffer>;
  assertThrows(() => decodeWav(truncated), Error, "残りは");
  // fmt だけのファイル（data 無し）。
  const fmtOnly = complete.slice(0, 36) as Uint8Array<ArrayBuffer>;
  assertThrows(() => decodeWav(fmtOnly), Error, "'data' チャンクが無い");
});

Deno.test("decodeWav: フレーム境界で割り切れない data は落とす", () => {
  assertThrows(
    () =>
      decodeWav(
        buildWav({
          format: 1,
          channels: 2,
          sampleRate: 48000,
          bits: 16,
          // 2ch × 16bit = 4 バイト/フレームなのに 6 バイト。
          payload: new Uint8Array(6),
        }),
      ),
    Error,
    "割り切れない",
  );
});

Deno.test("encodeWav: u32 に収まらない sampleRate / byte rate は落とす", () => {
  const samples = Float32Array.of(0, 0.5);
  // 上限ちょうど（byte rate = 0xffff_fffe）は通り、ヘッダにその値がそのまま載る。
  const view = new DataView(encodeWav(samples, 0x7fff_ffff).buffer);
  assertEquals(view.getUint32(24, true), 0x7fff_ffff, "sample rate");
  assertEquals(view.getUint32(28, true), 0xffff_fffe, "byte rate");
  // 1 つ上は sampleRate 自身は u32 に収まるが byte rate（×2）が溢れる。検査が無いと
  // `setUint32` が mod 2^32 で巻き戻し、byte rate 0 を宣言した **valid な WAV** が出る。
  assertThrows(() => encodeWav(samples, 0x8000_0000), RangeError, "byte rate 4294967296");
  // sampleRate 自身が u32 を超える場合も同じ門で落ちる。
  assertThrows(() => encodeWav(samples, 0x1_0000_0000), RangeError, "u32 に収まらない");
});

Deno.test("encodeWav: RIFF チャンク長が u32 を超えるサンプル数は落とす", () => {
  // 上限は (0xffff_ffff − 36) / 2 = 2147483629 サンプル。実物は 8GB を超えて確保できないので、
  // 長さだけを名乗る器で境界計算を叩く（`encodeWav` は出力を確保する**前**に長さを見る）。
  // 上限ちょうど側は 4GB の確保が要るので置かない（通る側は既存の往復テストが押さえている）。
  const huge = { length: 2_147_483_630 } as unknown as Float32Array;
  assertThrows(
    () => encodeWav(huge, 48000),
    RangeError,
    "RIFF チャンク長 4294967296 が u32 に収まらない",
  );
});
