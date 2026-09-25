// WAV（RIFF）の読み書きの挙動テスト。GPU も実資産も要らない。
//
// `decodeWav` は参照音声を食う唯一の入口で、**規約の取り違えが例外にならない**（int16 の
// スケールが 1 ずれても「少し音量が違う波形」が出るだけ）。したがって値そのものを固定する。
//
// 資産があれば golden `outputs/series/dacvae-32dim/host/meta.json` の `wavScale`（上流の
// リーダが int16 の両端で実測した値）とも突き合わせる（無ければその 1 本だけ SKIP）。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { decodeWav, encodeWav } from "../src/audio/wav.ts";
import { ModelInputError } from "../src/errors.ts";
import { readTextIfPresent } from "./helpers/read-if-present.ts";

/** golden `meta.json` の置き場（`tools/export-recipes/irodori/dacvae/host.py` の既定の出力先）。 */
const HOST_GOLDEN = new URL(
  "../../../outputs/series/dacvae-32dim/host/meta.json",
  import.meta.url,
);
const HOST_COMMAND = "cd tools/export-recipes && uv run --with descript-audiotools --with einops " +
  "--with 'transformers==5.14.1' python -m irodori.dacvae.host";

const goldenText = await readTextIfPresent(HOST_GOLDEN);
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

Deno.test("decodeWav: RIFF が宣言した境界の外に余った物理バイトは走査しない", () => {
  const complete = buildWav({
    format: 1,
    channels: 1,
    sampleRate: 48000,
    bits: 16,
    payload: int16Payload([16384]),
  });
  // 論理終端（RIFF 宣言 = complete.length）の**外**に、長さが嘘のチャンクを足す。物理長で
  // 走査を止める実装はここまで読んでしまい「残りが足りない」で落ちる。
  const padded = new Uint8Array(complete.length + 8);
  padded.set(complete);
  ascii(padded, complete.length, "junk");
  new DataView(padded.buffer).setUint32(complete.length + 4, 0xffff_ffff, true);
  const decoded = decodeWav(padded);
  assertEquals(Array.from(decoded.data), [0.5]);
});

// ---- fail loudly ---------------------------------------------------------

Deno.test("decodeWav: RIFF/WAVE でないバイト列は落とす", () => {
  const bytes = new Uint8Array(64);
  ascii(bytes, 0, "RIFX");
  ascii(bytes, 8, "WAVE");
  assertThrows(() => decodeWav(bytes), ModelInputError, "RIFF/WAVE ヘッダでない");
  assertThrows(() => decodeWav(new Uint8Array(8)), ModelInputError, "バイトしかない");
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
    ModelInputError,
    "format 1 / 24bit に未対応",
  );
  // A-law。
  assertThrows(
    () =>
      decodeWav(
        buildWav({ format: 6, channels: 1, sampleRate: 8000, bits: 8, payload: new Uint8Array(4) }),
      ),
    ModelInputError,
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
    ModelInputError,
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
  assertThrows(() => decodeWav(truncated), ModelInputError, "残りは");
  // fmt だけのファイル（data 無し）。RIFF の宣言サイズも 36 バイトの器に揃える — 揃えないと
  // 「器が切り詰められている」ほうの門で先に落ち、data 欠落の門を踏まない。
  const fmtOnly = complete.slice(0, 36) as Uint8Array<ArrayBuffer>;
  new DataView(fmtOnly.buffer).setUint32(4, 36 - 8, true);
  assertThrows(() => decodeWav(fmtOnly), ModelInputError, "'data' チャンクが無い");
});

/** 器の末尾にチャンクを 1 本足し、RIFF の宣言サイズを揃える（奇数長は 1 バイト詰め）。 */
const appendChunk = (
  wav: Uint8Array<ArrayBuffer>,
  id: string,
  body: Uint8Array,
): Uint8Array<ArrayBuffer> => {
  const bytes = new Uint8Array(wav.length + 8 + body.length + (body.length % 2));
  bytes.set(wav);
  ascii(bytes, wav.length, id);
  const view = new DataView(bytes.buffer);
  view.setUint32(wav.length + 4, body.length, true);
  bytes.set(body, wav.length + 8);
  view.setUint32(4, bytes.length - 8, true);
  return bytes;
};

describe("decodeWav が同じ id のチャンクを 2 本持つ器を受けたとき", () => {
  const single = (): Uint8Array<ArrayBuffer> =>
    buildWav({
      format: 1,
      channels: 1,
      sampleRate: 48000,
      bits: 16,
      payload: int16Payload([16384]),
    });

  it("'data' が 2 本なら後勝ちで読まずに落とす", () => {
    const twoData = appendChunk(single(), "data", int16Payload([-16384]));
    assertThrows(() => decodeWav(twoData), ModelInputError, "チャンク 'data' が 2 本ある");
  });

  it("'fmt ' が 2 本なら後勝ちで読まずに落とす", () => {
    const wav = single();
    // 1 本目の `fmt ` の中身（offset 20 から 16 バイト）をそのまま写す — 中身が同じでも重複は重複。
    const twoFmt = appendChunk(wav, "fmt ", wav.slice(20, 36));
    assertThrows(() => decodeWav(twoFmt), ModelInputError, "チャンク 'fmt ' が 2 本ある");
  });

  it("読み飛ばす未知チャンク（LIST）は 2 本あっても通る", () => {
    const twoList = appendChunk(
      buildWav({
        format: 1,
        channels: 1,
        sampleRate: 48000,
        bits: 16,
        payload: int16Payload([16384]),
        extraChunk: { id: "LIST", length: 7 },
      }),
      "LIST",
      new Uint8Array(3),
    );
    assertEquals(Array.from(decodeWav(twoList).data), [0.5]);
  });
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
    ModelInputError,
    "割り切れない",
  );
});

Deno.test("decodeWav: RIFF の宣言サイズが物理長を超える器は落とす（切り詰め）", () => {
  const complete = buildWav({
    format: 1,
    channels: 1,
    sampleRate: 48000,
    bits: 16,
    payload: int16Payload([1, 2]),
  });
  // 物理長より 16 バイト多く名乗る。チャンク側は全て整合しているので、宣言サイズを読まない
  // 実装は最後まで問題なく読み切ってしまう。
  new DataView(complete.buffer).setUint32(4, complete.length - 8 + 16, true);
  assertThrows(
    () => decodeWav(complete),
    ModelInputError,
    `RIFF が ${complete.length + 8} バイトを宣言`,
  );
});

Deno.test("decodeWav: data の宣言が RIFF の論理終端をはみ出すファイルは落とす", () => {
  const complete = buildWav({
    format: 1,
    channels: 1,
    sampleRate: 48000,
    bits: 16,
    payload: int16Payload([1, 2, 3, 4]),
  });
  // 物理バイトは 8 だけ余らせ、data の宣言長をその分伸ばす。物理長で境界を見る実装は通し、
  // 容器の外にある 8 バイトを 4 サンプルとして読んでしまう。
  const padded = new Uint8Array(complete.length + 8);
  padded.set(complete);
  const view = new DataView(padded.buffer);
  view.setUint32(40, view.getUint32(40, true) + 8, true); // data チャンクの長さ欄（44 バイト定型）
  assertThrows(() => decodeWav(padded), ModelInputError, "チャンク 'data' が 16 バイトを宣言");
});

Deno.test("decodeWav: fmt の block align / byte rate が導出値と矛盾するファイルは落とす", () => {
  const build = () =>
    buildWav({
      format: 1,
      channels: 1,
      sampleRate: 48000,
      bits: 16,
      payload: int16Payload([1, 2]),
    });
  // block align だけを壊す（1ch × 16bit なら 2 バイト）。fmt 定型では offset 32。
  const badAlign = build();
  new DataView(badAlign.buffer).setUint16(32, 4, true);
  assertThrows(
    () => decodeWav(badAlign),
    ModelInputError,
    "block align 宣言 4 が、1ch × 16bit から出る 2 と食い違う",
  );
  // byte rate だけを壊す（48000 × 2 = 96000）。fmt 定型では offset 28。
  const badRate = build();
  new DataView(badRate.buffer).setUint32(28, 192000, true);
  assertThrows(
    () => decodeWav(badRate),
    ModelInputError,
    "byte rate 宣言 192000 が、48000Hz × block align 2 = 96000 と食い違う",
  );
});

Deno.test("encodeWav: 非有限サンプルは位置と値付きで落とす（無音・クリップに化けさせない）", () => {
  // クリップの `Math.max` / `Math.min` は NaN を素通しし、`Math.round(NaN)` → `setInt16` が
  // 0 を書く。±Infinity はフルスケールへ張り付く。どちらも例外にならない。
  for (
    const [value, text] of [[NaN, "NaN"], [Infinity, "Infinity"], [-Infinity, "-Infinity"]] as const
  ) {
    const samples = Float32Array.of(0.1, 0.2, value, 0.3);
    assertThrows(
      () => encodeWav(samples, 48000),
      ModelInputError,
      `2 番目のサンプル ${text} が非有限`,
    );
  }
  // 値域外の有限値は今までどおりクリップして通る（検査が全部を落としていない）。
  const clipped = new DataView(encodeWav(Float32Array.of(2, -2), 48000).buffer);
  assertEquals(clipped.getInt16(44, true), 32767);
  assertEquals(clipped.getInt16(46, true), -32767);
});

Deno.test("encodeWav: u32 に収まらない sampleRate / byte rate は落とす", () => {
  const samples = Float32Array.of(0, 0.5);
  // 上限ちょうど（byte rate = 0xffff_fffe）は通り、ヘッダにその値がそのまま載る。
  const view = new DataView(encodeWav(samples, 0x7fff_ffff).buffer);
  assertEquals(view.getUint32(24, true), 0x7fff_ffff, "sample rate");
  assertEquals(view.getUint32(28, true), 0xffff_fffe, "byte rate");
  // 1 つ上は sampleRate 自身は u32 に収まるが byte rate（×2）が溢れる。検査が無いと
  // `setUint32` が mod 2^32 で巻き戻し、byte rate 0 を宣言した **valid な WAV** が出る。
  assertThrows(() => encodeWav(samples, 0x8000_0000), ModelInputError, "byte rate 4294967296");
  // sampleRate 自身が u32 を超える場合も同じ門で落ちる。
  assertThrows(() => encodeWav(samples, 0x1_0000_0000), ModelInputError, "u32 に収まらない");
});

Deno.test("encodeWav: RIFF チャンク長が u32 を超えるサンプル数は落とす", () => {
  // 上限は (0xffff_ffff − 36) / 2 = 2147483629 サンプル。実物は 8GB を超えて確保できないので、
  // 長さだけを名乗る器で境界計算を叩く（`encodeWav` は出力を確保する**前**に長さを見る）。
  // 上限ちょうど側は 4GB の確保が要るので置かない（通る側は既存の往復テストが押さえている）。
  const huge = { length: 2_147_483_630 } as unknown as Float32Array;
  assertThrows(
    () => encodeWav(huge, 48000),
    ModelInputError,
    "RIFF チャンク長 4294967296 が u32 に収まらない",
  );
});

// ---- 入力起因かどうかの分類（ADR 0107）------------------------------------

describe("共通 audio 層の失敗をホストが 400 / 500 に振り分けるとき", () => {
  it("ホストが渡したバイト列の解析失敗は ModelInputError で捕まる", () => {
    const notRiff = new Uint8Array(64);
    ascii(notRiff, 0, "RIFX");
    ascii(notRiff, 8, "WAVE");
    assert(assertThrows(() => decodeWav(notRiff)) instanceof ModelInputError);
    // 形式の検査（未対応の bit 深度）も同じ枝に落ちる。
    const unsupported = buildWav({
      format: 1,
      channels: 1,
      sampleRate: 48000,
      bits: 24,
      payload: new Uint8Array(6),
    });
    assert(assertThrows(() => decodeWav(unsupported)) instanceof ModelInputError);
  });

  it("呼び手が渡した波形 / 周波数の違反も ModelInputError で捕まる", () => {
    assert(assertThrows(() => encodeWav(new Float32Array(1), 0)) instanceof ModelInputError);
    assert(
      assertThrows(() => encodeWav(Float32Array.of(0, NaN), 48000)) instanceof ModelInputError,
    );
  });

  // NOTE: 内部不変条件の破れ（ModelInputError **でない**素の Error）は、この層には到達経路が
  // 無い。`wav.ts` の throw は 17 本とも呼び手のバイト列 / 波形 / 周波数だけを見ており、資産にも
  // 配線にも由来しない。「ModelInputError でない」側は共通 image 層が持つ
  // （`image_preprocess_test.ts` の `resizePlaneF32` — 公開面に出ていない内部ヘルパ）。
});
