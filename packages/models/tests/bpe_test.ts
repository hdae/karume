// BPE 共通層（src/text/bpe.ts）と逐次復号器（src/text/detokenizer.ts）の振る舞い検証。
//
// 実資産には依存しない — 合成した小語彙で「rank と位置の優先順」「古くなった対の無効化」
// 「byte_fallback の展開」「byte run の畳み方」を直接見る。実資産との突合（id 列が上流の
// `tokenizers` と一致するか）はファミリ側（gemma_tokenizer_test.ts）の担当で、ここは
// **規則そのもの**と**計算量**を落とせる形に置く。

import { assertEquals, assertThrows } from "@std/assert";
import { bpeEncode, type BpeModel, createBpeModel } from "../src/text/bpe.ts";
import { detokenize, StreamingDetokenizer } from "../src/text/detokenizer.ts";

/** byte_fallback 語彙の綴り（資産側と同じ規約）。 */
const byteToken = (byte: number): string =>
  `<0x${byte.toString(16).toUpperCase().padStart(2, "0")}>`;

/**
 * 綴りの一覧と `[左, 右]` の併合規則から模型を組む。
 *
 * byte 語彙 256 本は素の語彙の**後ろ**へ足す（`createBpeModel` が要求する 256 本を、
 * テストごとに書かずに済ませるため）。
 */
const buildModel = (
  pieces: readonly string[],
  merges: readonly (readonly [string, string])[] = [],
): BpeModel => {
  const tokens = [...pieces, ...Array.from({ length: 256 }, (_, byte) => byteToken(byte))];
  const idOf = new Map(tokens.map((token, id) => [token, id] as const));
  return createBpeModel({
    vocab: tokens.map((token, id) => [id, token] as const),
    merges: merges.map(([left, right], rank) =>
      [idOf.get(left) as number, idOf.get(right) as number, rank] as const
    ),
    byteIds: Array.from({ length: 256 }, (_, byte) => idOf.get(byteToken(byte)) as number),
  });
};

/** id 列を綴り列へ戻す（意図が読める形で突き合わせるため）。 */
const spell = (model: BpeModel, ids: readonly number[]): string[] =>
  ids.map((id) => model.tokenOf.get(id) as string);

Deno.test("資産の門: 通すと沈黙誤値になる形を落とす", async (t) => {
  await t.step("byteIds が 256 本でないと落ちる", () => {
    assertThrows(
      () => createBpeModel({ vocab: [[0, "a"]], merges: [], byteIds: [0] }),
      Error,
      "256 本でない",
    );
  });

  await t.step("byteIds が指す綴りが `<0xHH>` でないと落ちる", () => {
    // `base + byte` の連番は実資産の実測事実であって schema の保証ではない。ずれた表を
    // 通すと byte 展開が合法な別の行を指す（例外にならない配布破損）。
    const tokens = ["x", ...Array.from({ length: 256 }, (_, byte) => byteToken(byte))];
    assertThrows(
      () =>
        createBpeModel({
          vocab: tokens.map((token, id) => [id, token] as const),
          merges: [],
          // 1 つ手前へずらす = 全部が 1 バイト分ずれた表。
          byteIds: Array.from({ length: 256 }, (_, byte) => byte),
        }),
      Error,
      "綴りが",
    );
  });

  await t.step("同じ対の併合規則が 2 本あると落ちる（上流は後勝ちで沈黙する）", () => {
    assertThrows(
      () => buildModel(["a", "aa"], [["a", "a"], ["a", "a"]]),
      Error,
      "重複",
    );
  });

  await t.step("併合先の綴りが語彙に無いと落ちる", () => {
    assertThrows(() => buildModel(["a", "b"], [["a", "b"]]), Error, "語彙に無い");
  });

  await t.step("語彙の綴りが重複すると落ちる（行番号 = id が引けなくなる）", () => {
    assertThrows(
      () => createBpeModel({ vocab: [[0, "a"], [1, "a"]], merges: [], byteIds: [] }),
      Error,
      "重複",
    );
  });
});

Deno.test("併合の順序: rank が最小のものから、同点は左から", async (t) => {
  await t.step("左から貪欲に取るのではなく rank の小さい対が先に潰れる", () => {
    // ("b","c") が rank 0・("a","b") が rank 1。左から取ると [ab, c] になるが、正本は
    // rank 順なので [a, bc] が正しい。
    const model = buildModel(["a", "b", "c", "ab", "bc"], [["b", "c"], ["a", "b"]]);
    assertEquals(spell(model, bpeEncode(model, "abc")), ["a", "bc"]);
  });

  await t.step("同じ rank が複数の位置に立つときは左が勝つ", () => {
    // "aaa" は位置 0 と 1 の両方で ("a","a") rank 0。左が勝てば [aa, a] → ("aa","a") で
    // [aaa] まで畳まれる。右を先に取ると [a, aa] で止まる（("a","aa") は規則に無い）。
    const model = buildModel(["a", "aa", "aaa"], [["a", "a"], ["aa", "a"]]);
    assertEquals(spell(model, bpeEncode(model, "aaa")), ["aaa"]);
  });

  await t.step("併合で入れ替わった位置の古い項目は捨てられる", () => {
    // ("y","z") rank 0 が先に潰れると、位置 0 に積んであった ("x","y") rank 1 の相手は
    // もう `y` ではない。同一性で照合しないと `x` と `yz` を `xy` として畳んでしまう。
    const model = buildModel(["x", "y", "z", "yz", "xy"], [["y", "z"], ["x", "y"]]);
    assertEquals(spell(model, bpeEncode(model, "xyz")), ["x", "yz"]);
  });

  await t.step("消えた位置の項目は読み飛ばされる", () => {
    // ("a","b") → ab で位置 1 が消えるが、キューには ("b","c") が残っている。
    const model = buildModel(
      ["a", "b", "c", "d", "ab", "cd", "bc", "abcd"],
      [["a", "b"], ["c", "d"], ["b", "c"], ["ab", "cd"]],
    );
    assertEquals(spell(model, bpeEncode(model, "abcd")), ["abcd"]);
  });
});

Deno.test("byte_fallback: 語彙に無い文字は UTF-8 バイトへ割れる", async (t) => {
  const model = buildModel(["a"]);

  await t.step("多バイト文字は 1 バイト 1 記号になる", () => {
    assertEquals(spell(model, bpeEncode(model, "あ")), ["<0xE3>", "<0x81>", "<0x82>"]);
  });

  await t.step("語彙にある文字は割れない", () => {
    assertEquals(spell(model, bpeEncode(model, "aあ")).slice(0, 1), ["a"]);
  });

  await t.step("空の断片は空の id 列", () => {
    assertEquals(bpeEncode(model, ""), []);
  });

  await t.step("対にならないサロゲートは落とす（正本の str に載らない値）", () => {
    assertThrows(() => bpeEncode(model, "\ud800"), Error, "サロゲート");
  });
});

Deno.test("併合キュー: 長文で計算量が線形に近い（O(n²) の全走査は落ちる）", () => {
  // 2 のべき乗で倍化する語彙を合成し、日本語 1 文字 65,536 個を 1 断片として食わせる。
  // 併合回数は 65,535 回で、素朴な「全隣接ペアを走査 → 最小 rank を splice」は
  // 併合ごとに記号列を舐めるので 10^9 回規模の探索になる（分単位）。merge queue なら
  // 押し込みと取り出しが記号数の定数倍で済む。
  //
  // MUST: この門は恒真でない — 実装を全走査へ書き戻すと確実に落ちる（時間予算は実測の
  // 100 倍以上の余裕を採ってあるので、機械の速度差では落ちない）。
  const LEVELS = 16;
  const pieces = Array.from({ length: LEVELS + 1 }, (_, level) => "あ".repeat(2 ** level));
  const merges = Array.from(
    { length: LEVELS },
    (_, level) => [pieces[level], pieces[level]] as const,
  );
  const model = buildModel(pieces, merges);

  const started = performance.now();
  const ids = bpeEncode(model, "あ".repeat(2 ** LEVELS));
  const elapsed = performance.now() - started;

  assertEquals(spell(model, ids), [pieces[LEVELS]]);
  const BUDGET_MS = 5_000;
  if (elapsed > BUDGET_MS) {
    throw new Error(
      `記号 ${2 ** LEVELS} 個の併合に ${elapsed.toFixed(0)}ms（予算 ${BUDGET_MS}ms）— ` +
        "全走査へ退行していないか確かめる",
    );
  }
});

/** 復号の口を合成する（byte 語彙 = `<0xHH>`・`skip` の id は綴りを返さない）。 */
const buildSource = (
  tokens: ReadonlyMap<number, string>,
  skip: ReadonlySet<number> = new Set(),
) => ({
  byteOf: (id: number): number | undefined => {
    const token = tokens.get(id);
    if (token === undefined || !/^<0x[0-9A-F]{2}>$/.test(token)) return undefined;
    return Number.parseInt(token.slice(3, 5), 16);
  },
  textOf: (id: number): string | undefined => {
    if (skip.has(id)) return undefined;
    const token = tokens.get(id);
    if (token === undefined) throw new Error(`復号: id ${id} が語彙に無い`);
    return token;
  },
});

/** `<0xHH>` を id 200+byte・普通の綴りを 0..9 に割り当てた復号用の表。 */
const DECODE_TOKENS = new Map<number, string>([
  [0, "hello"],
  [1, "<eos>"],
  [2, "world"],
  ...Array.from({ length: 256 }, (_, byte) => [200 + byte, byteToken(byte)] as const),
]);

Deno.test("逐次復号: byte run 以外は 1 トークンも溜めない", async (t) => {
  await t.step("byte でないトークンは push のたびに確定して出る", () => {
    // `finish()` まで溜めてから返す「偽 streaming」はここで落ちる（ADR 0084 決定 4）。
    const detokenizer = new StreamingDetokenizer(buildSource(DECODE_TOKENS));
    assertEquals(detokenizer.push(0), "hello");
    assertEquals(detokenizer.push(2), "world");
    assertEquals(detokenizer.finish(), "");
  });

  await t.step("byte run は閉じるまで出ない（閉じる = 非 byte トークンか finish）", () => {
    // 「3 バイト揃った時点で出す」にはできない。正本は run 全体を 1 単位で畳むので、後続の
    // 1 バイトで**先頭の妥当な部分まで**置換文字に変わりうる（下の run の畳み方の節）。
    const detokenizer = new StreamingDetokenizer(buildSource(DECODE_TOKENS));
    assertEquals(detokenizer.push(200 + 0xe3), "");
    assertEquals(detokenizer.push(200 + 0x81), "");
    assertEquals(detokenizer.push(200 + 0x82), "");
    assertEquals(detokenizer.finish(), "あ");
  });

  await t.step("run の直後の非 byte トークンと同時に確定する", () => {
    const detokenizer = new StreamingDetokenizer(buildSource(DECODE_TOKENS));
    detokenizer.push(200 + 0xe3);
    detokenizer.push(200 + 0x81);
    detokenizer.push(200 + 0x82);
    assertEquals(detokenizer.push(0), "あhello");
  });
});

Deno.test("逐次復号: byte run の畳み方は正本の String::from_utf8 に合わせる", async (t) => {
  const source = buildSource(DECODE_TOKENS);

  await t.step("run 全体が不正なら 1 バイトにつき 1 個の置換文字", () => {
    // 「先頭の妥当な部分だけを取り出す」形にはならない。
    assertEquals(detokenize(source, [200 + 0xe3, 200 + 0x81]), "��");
  });

  await t.step("run が非 byte トークンで割れると別々に畳まれる", () => {
    assertEquals(
      detokenize(source, [200 + 0xe3, 200 + 0x81, 0, 200 + 0x82]),
      "��hello�",
    );
  });

  await t.step("BOM は剥がされない（TextDecoder の既定は捨てる）", () => {
    // `ignoreBOM: true` を外すと、`EF BB BF` の run だけが静かに消える。
    assertEquals(detokenize(source, [200 + 0xef, 200 + 0xbb, 200 + 0xbf]), "﻿");
  });

  await t.step("終端の run は finish で確定する", () => {
    const detokenizer = new StreamingDetokenizer(source);
    detokenizer.push(200 + 0x41);
    assertEquals(detokenizer.finish(), "A");
  });
});

Deno.test("逐次復号: skip した特殊トークンは byte run を切らない", () => {
  // 正本は「id 列 → 綴り列」を作る段で skip し、デコーダ鎖はその後で走る。切ってしまうと
  // 特殊トークンを挟んだだけで文字が置換文字 2 個に化ける。
  const source = buildSource(DECODE_TOKENS, new Set([1]));
  assertEquals(detokenize(source, [200 + 0xe3, 1, 200 + 0x81, 200 + 0x82]), "あ");
});

Deno.test("逐次復号: 語彙に無い id は落とす（正本は黙って読み飛ばす）", () => {
  assertThrows(
    () => detokenize(buildSource(DECODE_TOKENS), [999]),
    Error,
    "語彙に無い",
  );
});
