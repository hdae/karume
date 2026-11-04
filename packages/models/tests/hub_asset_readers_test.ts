// The reader of fetched asset bytes (`src/hub/asset-readers.ts`). 7 families go through this one
// path, so widening its accepted set here loosens the asset checks of every family at once.
//
// What is checked, 5 points (the last one is for assets inside the container — `readWholeAsset`):
//
// ① It **does not copy**. What comes back is the view's `buffer` itself (the condition for GB-scale
//    weights not to double the RAM peak), which can be pinned by identity.
// ② A view that does not span its whole buffer is rejected. Letting it through would make
//    `openContainer` read the neighbouring bytes as a header.
// ③ The message labels (family name and manifest table) come out verbatim as the caller passed
//    them — each family's tests assert the message verbatim, so dropping them here would turn 5
//    files red at once.
// ④ The decode stage (invalid UTF-8) and the parse stage (JSON syntax error) fail with different
//    messages.

import { test } from "vite-plus/test";
import { assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import { openContainer } from "@karume/runtime";
import { readAssetBuffer, readAssetJson, readWholeAsset } from "../src/hub/asset-readers.ts";
import { tensorlessContainer } from "./helpers/container-fixture.ts";

/** A view spanning its whole buffer (the shape the hub's contract returns). */
const wholeView = (bytes: readonly number[]): Uint8Array<ArrayBuffer> => Uint8Array.from(bytes);

test("readAssetBuffer: returns the view's buffer itself (not a single byte is copied)", () => {
  const bytes = wholeView([1, 2, 3, 4]);
  assertStrictEquals(
    readAssetBuffer("sbv2", "weights / assets", { model: bytes }, "model"),
    bytes.buffer,
  );
});

test("readAssetBuffer: fails when the asset is missing, with the family, the table and the keys present in the message", () => {
  assertThrows(
    () => readAssetBuffer("sbv2", "weights / assets", { front: wholeView([0]) }, "voice"),
    Error,
    "sbv2: asset 'voice' is missing (the manifest's weights / assets needs voice)" +
      " (keys present: front)",
  );
});

test("readAssetBuffer: the manifest table is the caller's label as is (it differs per family)", () => {
  assertThrows(
    () => readAssetBuffer("birefnet", "weights", {}, "matte"),
    Error,
    "birefnet: asset 'matte' is missing (the manifest's weights needs matte) (keys present: )",
  );
});

test("readAssetBuffer: asset presence is checked with Object.hasOwn (prototype values are not picked up)", () => {
  // `assets["toString"]` returns a function from Object.prototype. Checking with `in` would try to
  // read `byteOffset` from it and fail somewhere other than "the asset is missing".
  assertThrows(
    () => readAssetBuffer("anima", "weights / assets", {}, "toString"),
    Error,
    "anima: asset 'toString' is missing",
  );
});

test("readAssetBuffer: a view not spanning its whole buffer fails as a broken fetch-layer invariant", () => {
  // A view that skips the first byte. Returning it would make `openContainer` read the neighbouring
  // bytes as a header.
  const backing = wholeView([1, 2, 3, 4]);
  const shifted = new Uint8Array(backing.buffer, 1, 2);
  assertThrows(
    () => readAssetBuffer("irodori", "weights / assets", { dit: shifted }, "dit"),
    Error,
    "irodori: the bytes of asset 'dit' do not cover the whole buffer (byteOffset 1 / byteLength 2 /" +
      " buffer 4)",
  );
});

/**
 * An asset that is closed as JSON but has one invalid UTF-8 byte (0xff) mixed into a string value.
 */
const brokenUtf8Asset = (): Uint8Array<ArrayBuffer> =>
  Uint8Array.from([
    ...new TextEncoder().encode('{"unkId":'),
    0x22,
    0xff,
    0x22,
    ...new TextEncoder().encode("}"),
  ]);

test("readAssetJson: an invalid UTF-8 asset fails with the decode-stage message (it does not reach the JSON stage)", () => {
  const broken = brokenUtf8Asset();
  // Pinning the premise: the default TextDecoder replaces 0xff with U+FFFD, so reading with
  // replacement silently lets **valid JSON with different contents** through. That path is closed.
  assertEquals(
    JSON.parse(new TextDecoder().decode(broken)),
    { unkId: "�" },
    "the premise that the replacing decode does not give valid JSON broke (this test's subject is gone)",
  );
  assertThrows(
    () => readAssetJson("sbv2", "weights / assets", { tokenizer: broken }, "tokenizer"),
    Error,
    "sbv2: asset 'tokenizer' cannot be read as UTF-8",
  );
});

test("readAssetJson: a JSON syntax error fails with a message different from decode (the valid domain is unchanged)", () => {
  const truncated = new TextEncoder().encode('{"unkId":');
  assertThrows(
    () => readAssetJson("irodori", "weights / assets", { tokenizer: truncated }, "tokenizer"),
    Error,
    "irodori: asset 'tokenizer' cannot be read as JSON",
  );
  // Correct UTF-8 passes as is, even with multi-byte characters.
  const valid = new TextEncoder().encode('{"unkId":0,"space":"あ"}');
  assertEquals(readAssetJson("irodori", "weights / assets", { tokenizer: valid }, "tokenizer"), {
    unkId: 0,
    space: "あ",
  });
});

test("readAssetJson: when the asset itself is missing, it fails with the buffer-stage message", () => {
  assertThrows(
    () => readAssetJson("sbv2", "weights / assets", {}, "symbols"),
    Error,
    "sbv2: asset 'symbols' is missing (the manifest's weights / assets needs symbols)",
  );
});

// ---- Container assets (`readWholeAsset`) ------------------------------------------
//
// Assets inside the container (`rope_base` / the PLE index) are not in the manifest's table, so the
// reader is a single `AssetReader`. Two points are checked: it returns **exactly the logical
// length** (no block padding is handed out), and the returned ArrayBuffer spans only the payload (a
// form `parseSafetensors` can read as is).

test("readWholeAsset: reads exactly the logical length and returns a buffer without padding", async () => {
  // An asset of length 6 (the block length rounds up to a multiple of 4, so 8 bytes = 2 bytes of
  // padding).
  const payload = Uint8Array.from([1, 2, 3, 4, 5, 6]) as Uint8Array<ArrayBuffer>;
  const written = await tensorlessContainer(
    "unit",
    { inputs: [{ name: "x", shape: [1, 2] }], output: { name: "y", shape: [1, 2] } },
    [{ name: "probe", role: "test-probe", bytes: payload }],
  );
  const opened = await openContainer({ kind: "bytes", bytes: written.single });
  const reader = opened.asset("probe");
  assertEquals(reader.role, "test-probe");
  assertEquals(reader.length, payload.byteLength);

  const buffer = await readWholeAsset(reader);
  assertEquals(buffer.byteLength, payload.byteLength, "padding was returned too");
  assertEquals([...new Uint8Array(buffer)], [...payload]);
});
