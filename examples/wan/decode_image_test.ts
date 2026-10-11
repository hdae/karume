// examples/wan の条件画像のデコーダ（`decode-image.ts`）の振る舞い。期待値は Pillow の `Image.open(...).convert("RGB")`
// の RGB8（上流 Wan2.2 の generate.py が条件画像を読む形）で、fixture と一緒に `emit-image-fixtures.py` が書き出す
// （Pillow 12.3.0・`fixtures/expected.json`）。PNG は画素単位で一致、JPEG はデコーダの実装差（IDCT・4:2:0 の色差の
// 補間）があるので最大絶対差の上限で縛る。GPU も実資産も要らない。

import { assert, assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { decodeImage } from "./decode-image.ts";

const FIXTURES = new URL("./fixtures/", import.meta.url);

type Case =
  | { readonly kind: "reject" }
  | {
    readonly kind: "exact" | "jpeg";
    readonly width: number;
    readonly height: number;
    /** RGB8 の 16 進（width × height × 3 バイト）。 */
    readonly rgb: string;
  };

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** expected.json の 1 件の形（`unknown` の境界 — fixture を作り直して欄が変わったらここで落とす）。 */
const isCase = (value: unknown): value is Case => {
  if (!isRecord(value)) return false;
  if (value.kind === "reject") return true;
  if (value.kind !== "exact" && value.kind !== "jpeg") return false;
  const { width, height, rgb } = value;
  return Number.isInteger(width) && Number.isInteger(height) && typeof rgb === "string" &&
    /^(?:[0-9a-f]{2})*$/.test(rgb) && rgb.length === Number(width) * Number(height) * 6;
};

const parseCases = (text: string): Readonly<Record<string, Case>> => {
  const parsed: unknown = JSON.parse(text);
  if (!isRecord(parsed) || !isRecord(parsed.cases)) {
    throw new Error("expected.json に cases が無い");
  }
  const cases: Record<string, Case> = {};
  for (const [name, value] of Object.entries(parsed.cases)) {
    if (!isCase(value)) {
      throw new Error(`expected.json の ${name} の形が違う: ${JSON.stringify(value)}`);
    }
    cases[name] = value;
  }
  return cases;
};
const cases = parseCases(Deno.readTextFileSync(new URL("expected.json", FIXTURES)));

const read = (name: string): Uint8Array => Deno.readFileSync(new URL(name, FIXTURES));

const hexBytes = (hex: string): Uint8Array =>
  Uint8Array.from(
    { length: hex.length / 2 },
    (_, at) => parseInt(hex.slice(at * 2, at * 2 + 2), 16),
  );

/** fixture の期待値（Pillow の RGB8）。 */
const expectedOf = (name: string) => {
  const entry = Object.hasOwn(cases, name) ? cases[name] : undefined;
  if (entry === undefined || entry.kind === "reject") {
    throw new Error(`expected.json に ${name} の期待値が無い`);
  }
  return { data: hexBytes(entry.rgb), width: entry.width, height: entry.height };
};

const maxAbsDiff = (a: Uint8Array, b: Uint8Array): number => {
  assertEquals(a.length, b.length);
  let max = 0;
  for (let at = 0; at < a.length; at += 1) max = Math.max(max, Math.abs(a[at] - b[at]));
  return max;
};

/** R と B を入れ替えた RGB8（チャネルの取り違えの模擬）。 */
const swapRedBlue = (rgb: Uint8Array): Uint8Array => {
  const swapped = rgb.slice();
  for (let at = 0; at < rgb.length; at += 3) {
    swapped[at] = rgb[at + 2];
    swapped[at + 2] = rgb[at];
  }
  return swapped;
};

const SUPPORTED = "対応形式は PNG";

/**
 * 8×8・全成分が平坦（DC も AC も 0）の baseline JPEG を `components` 成分で組む（各成分 1×1 の標本化・量子化表は
 * 全て 1・Huffman 表は DC / AC とも長さ 1 の符号 1 つ）。Pillow は 2 成分の JPEG を書けないので、成分数だけが違う
 * 正しい符号列をここで組む（JPEG には CRC が無い）。
 */
const flatJpeg = (components: number): Uint8Array => {
  const segment = (marker: number, body: readonly number[]): number[] => [
    0xff,
    marker,
    (body.length + 2) >> 8,
    (body.length + 2) & 0xff,
    ...body,
  ];
  const ids = Array.from({ length: components }, (_, index) => index + 1);
  // 符号長 1 の符号 1 つ（"0"）だけを持つ表 — DC は差分の大きさ 0・AC は EOB。
  const table = (tableClass: number): number[] => [tableClass, 1, ...new Array(15).fill(0), 0x00];
  // 1 ブロック = DC の "0" + EOB の "0"。MCU は成分ごとに 1 ブロックで、末尾を 1 で埋める。
  const bits = "00".repeat(components).padEnd(Math.ceil((components * 2) / 8) * 8, "1");
  const scan = Array.from(
    { length: bits.length / 8 },
    (_, at) => parseInt(bits.slice(at * 8, at * 8 + 8), 2),
  );
  return Uint8Array.from([
    0xff,
    0xd8,
    ...segment(0xdb, [0, ...new Array(64).fill(1)]),
    ...segment(0xc0, [8, 0, 8, 0, 8, components, ...ids.flatMap((id) => [id, 0x11, 0])]),
    ...segment(0xc4, [...table(0x00), ...table(0x10)]),
    ...segment(0xda, [components, ...ids.flatMap((id) => [id, 0x00]), 0, 63, 0]),
    ...scan,
    0xff,
    0xd9,
  ]);
};

/**
 * 画素単位で Pillow と一致する PNG（8 bit の 4 種・パレット 8 bit〈tRNS 付き〉/ 4 / 2 / 1 bit・Adam7 の 8 bit の
 * RGB / RGBA / Gray+Alpha / パレット）。1 / 2 bit は 1 バイトから 8 / 4 個の索引を取り、奇数幅 7 で行末がバイトの
 * 途中で切れる。Adam7 は fast-png が 1 画素のバイト数をチャネル数から数える経路を色の種類ごとに通す。
 */
const EXACT_PNG = [
  "rgb.png",
  "rgba.png",
  "l.png",
  "la.png",
  "p8.png",
  "p4.png",
  "p2.png",
  "p1.png",
  "adam7-rgb.png",
  "adam7-rgba.png",
  "adam7-la.png",
  "adam7-p8.png",
];

/**
 * JPEG の最大絶対差の上限 = この機（2026-10-10・jpeg-js 0.4.4 対 Pillow 12.3.0〈libjpeg-turbo〉）で測った最悪値 × 2。
 * 測定値: baseline 4:2:0 = 9・progressive = 9（同じ係数なので baseline と同じ値）・グレースケール = 1・
 * CMYK（Adobe）= 2・EXIF の向き付き（baseline と同じ画像）= 9。カラーの差の大半は 4:2:0 の色差の補間の流儀
 * （jpeg-js は最近傍・libjpeg-turbo は補間 — 4:4:4 なら最悪 3）。
 * RGB 符号（3 成分・色の変換なし）: Adobe の transform = 0 = 1・成分 ID 'R' 'G' 'B' だけ = 1・JFIF + Adobe 0
 * （JFIF が優先して YCbCr とみなす形）= 2。前の 2 つは jpeg-js の既定（3 成分を常に YCbCr とみなす）だと 255。
 */
const JPEG_BOUNDS: Readonly<Record<string, number>> = {
  "baseline-420.jpg": 18,
  "progressive.jpg": 18,
  "gray.jpg": 2,
  "cmyk.jpg": 4,
  "exif-orientation-6.jpg": 18,
  "rgb-adobe.jpg": 2,
  "rgb-ids.jpg": 2,
  "jfif-adobe0.jpg": 4,
};

describe("decodeImage — PNG", () => {
  for (const name of EXACT_PNG) {
    it(`${name} を Pillow の convert("RGB") と画素単位で同じ RGB8 にする`, () => {
      const decoded = decodeImage(read(name));
      const expected = expectedOf(name);
      assertEquals([decoded.width, decoded.height], [expected.width, expected.height]);
      assertEquals(decoded.data, expected.data);
    });
  }

  it("知らない色の種類（IHDR の色の種類 5）を対応形式を名指しして拒む", () => {
    const bytes = read("rgb.png").slice();
    // IHDR の本体の 10 バイト目（署名 8 + 長さ 4 + 型 4 + 幅 4 + 高さ 4 + 深度 1 の後）が色の種類。
    assertEquals(bytes[25], 2);
    bytes[25] = 5;
    const error = assertThrows(() => decodeImage(bytes), Error, "PNG の色の種類 5 を知らない");
    assertStringIncludes(error.message, SUPPORTED);
  });

  it("パレットの展開が 1 画素 3 / 4 バイトにならない PNG（tRNS が 2 本）を壊れたデータとして拒む", () => {
    assertThrows(
      () => decodeImage(read("p8-double-trns.png")),
      Error,
      "パレットの展開の長さ 175 が 35 画素の 3 / 4 倍でない",
    );
  });

  it("16 bit を対応形式を名指しして拒む（8 bit へ黙って丸めない）", () => {
    assertThrows(() => decodeImage(read("l16.png")), Error, "16 bit は受けない");
  });

  it("パレット以外の 8 bit 未満（1 bit の Gray）を拒む", () => {
    assertThrows(() => decodeImage(read("l1.png")), Error, "1 bit は受けない");
  });

  it("インターレース（Adam7）の 4 bit のパレットを拒む（fast-png の Adam7 の経路が 8 bit 未満を読み違える）", () => {
    assertThrows(() => decodeImage(read("adam7-p4.png")), Error, "インターレース");
  });

  for (const name of ["p8-bad-index.png", "p8-no-plte.png"]) {
    it(`壊れたパレット（${name} — CRC は正しい）を対応形式を名指しして拒む`, () => {
      assertThrows(() => decodeImage(read(name)), Error, SUPPORTED);
    });
  }

  it("途中で切れた PNG を拒む（どこで切れても）", () => {
    const bytes = read("rgb.png");
    for (const cut of [8, 20, 40, bytes.length - 20, bytes.length - 1]) {
      assertThrows(() => decodeImage(bytes.subarray(0, cut)), Error, SUPPORTED);
    }
  });

  it("CRC の合わないチャンク（壊れたデータ）を拒む", () => {
    const bytes = read("rgb.png").slice();
    // IDAT（署名 8 + IHDR 25 バイトの直後の chunk）の格納 CRC の 1 バイトを反転する — 中身は無傷なので、
    // CRC を検査しないと素通りする壊れ方。
    const idat = 33;
    assertEquals(String.fromCharCode(...bytes.subarray(idat + 4, idat + 8)), "IDAT");
    const length = new DataView(bytes.buffer).getUint32(idat);
    bytes[idat + 8 + length] ^= 0xff;
    assertThrows(() => decodeImage(bytes), Error, "CRC");
  });
});

describe("decodeImage — JPEG", () => {
  for (const [name, bound] of Object.entries(JPEG_BOUNDS)) {
    it(`${name} を Pillow の復号から最大 ${bound} 以内の RGB8 にする`, () => {
      const decoded = decodeImage(read(name));
      const expected = expectedOf(name);
      assertEquals([decoded.width, decoded.height], [expected.width, expected.height]);
      const diff = maxAbsDiff(decoded.data, expected.data);
      assert(diff <= bound, `${name} の最大絶対差 ${diff} が上限 ${bound} を超えた`);
    });
  }

  it("EXIF の向き（orientation = 6）を適用しない（縦横が入れ替わらない）", () => {
    const decoded = decodeImage(read("exif-orientation-6.jpg"));
    assertEquals([decoded.width, decoded.height], [37, 21]);
  });

  it("標本の精度が 8 bit でない JPEG（SOF の精度 12）を対応形式を名指しして拒む", () => {
    const bytes = read("baseline-420.jpg").slice();
    // SOF0（FF C0）の本体の先頭（長さ 2 バイトの後）が精度。
    const sof = bytes.findIndex((byte, at) => byte === 0xff && bytes[at + 1] === 0xc0);
    assertEquals(bytes[sof + 4], 8);
    bytes[sof + 4] = 12;
    assertThrows(() => decodeImage(bytes), Error, "精度 12 bit は受けない");
  });

  it("2 成分の JPEG を対応形式を名指しして拒む（jpeg-js の色の変換に 2 成分の枝が無い）", () => {
    // 対照: 同じ組み方の 1 成分は復号できる（拒否が組み方の誤りではなく成分数から来ること）。
    const gray = decodeImage(flatJpeg(1));
    assertEquals([gray.width, gray.height], [8, 8]);
    const error = assertThrows(() => decodeImage(flatJpeg(2)), Error, SUPPORTED);
    // 拒否の経路の確かめ: 復号は通り、RGB への変換（jpeg-js の `copyToImageData`）で落ちている。
    assertStringIncludes(String(error.cause), "Unsupported color mode");
  });

  it("Adobe の APP14 が無い 4 成分の JPEG を対応形式を名指しして拒む（CMYK か YCCK か決められない）", () => {
    const bytes = read("cmyk.jpg");
    const app14 = bytes.findIndex((byte, at) => byte === 0xff && bytes[at + 1] === 0xee);
    assert(app14 > 0, "前提: cmyk.jpg は Adobe の APP14 を持つ");
    const length = (bytes[app14 + 2] << 8) | bytes[app14 + 3];
    const stripped = new Uint8Array([
      ...bytes.subarray(0, app14),
      ...bytes.subarray(app14 + 2 + length),
    ]);
    const error = assertThrows(() => decodeImage(stripped), Error, SUPPORTED);
    assertStringIncludes(String(error.cause), "4 components");
  });

  for (const [label, marker] of [["SOF3（可逆）", 0xc3], ["SOF9（算術符号）", 0xc9]] as const) {
    it(`jpeg-js が知らない ${label} の JPEG を対応形式を名指しして拒む`, () => {
      const bytes = read("baseline-420.jpg").slice();
      const sof = bytes.findIndex((byte, at) => byte === 0xff && bytes[at + 1] === 0xc0);
      bytes[sof + 1] = marker;
      assertThrows(() => decodeImage(bytes), Error, SUPPORTED);
    });
  }

  it("途中で切れた JPEG を部分的な画像にせず拒む", () => {
    for (const name of ["baseline-420.jpg", "progressive.jpg"]) {
      const bytes = read(name);
      for (const cut of [4, Math.floor(bytes.length / 2), bytes.length - 2]) {
        assertThrows(() => decodeImage(bytes.subarray(0, cut)), Error, SUPPORTED);
      }
    }
  });
});

describe("decodeImage — 対応外の形式", () => {
  for (const name of ["gif.gif", "webp.webp"]) {
    it(`${name} をマジックバイトで見分けて対応形式を名指しして拒む`, () => {
      assertThrows(() => decodeImage(read(name)), Error, SUPPORTED);
    });
  }

  it("空のバイト列を拒む", () => {
    assertThrows(() => decodeImage(new Uint8Array(0)), Error, SUPPORTED);
  });
});

describe("fixture の検出力（上のテストが落ちうること）", () => {
  it("色の PNG の期待値は R と B を入れ替えると変わる（チャネルの取り違えを見逃さない）", () => {
    for (
      const name of [
        "rgb.png",
        "rgba.png",
        "p8.png",
        "p4.png",
        "p2.png",
        "p1.png",
        "adam7-rgb.png",
        "adam7-rgba.png",
        "adam7-p8.png",
      ]
    ) {
      const { data } = expectedOf(name);
      assert(maxAbsDiff(swapRedBlue(data), data) > 0, `${name} の R と B が同じ値`);
    }
  });

  it("カラーの JPEG の期待値は R と B を入れ替えると上限を大きく超える", () => {
    for (const name of ["baseline-420.jpg", "progressive.jpg", "cmyk.jpg"]) {
      const { data } = expectedOf(name);
      assert(maxAbsDiff(swapRedBlue(data), data) > JPEG_BOUNDS[name] * 2, name);
    }
  });
});
