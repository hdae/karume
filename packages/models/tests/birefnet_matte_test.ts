// BiRefNet の α マットの後処理（`src/birefnet/matte.ts`）の 2 つの手順を、系列ごとの上流の関数で
// 作った golden と突き合わせる（ADR 0125 — GPU 不要・資産不要）。
//
// golden は git 追跡の `fixtures/birefnet-matte/matte.{json,safetensors}`（書き手は recipe の
// `tools/export-recipes/birefnet/matte_fixture.py` — 上流の関数をそのまま呼び、量子化だけを四捨五入へ
// 差し替えたもの）。無ければ SKIP ではなく赤（上流との一致を誰も見ていない状態を緑にしない）。
//
// - `u8-pil-bicubic`（HR の上流の意図）: golden の `<case>.hr`（Pillow の 8bit の BICUBIC の最終 α）と
//   **0 不一致**。
// - `float-bilinear`（Lucida の上流の意図 = 今の経路）: golden の `<case>.lucida`（torch の丸める前の
//   `255α`）を四捨五入した最終 α と一致するか、golden の `255α` が `k + 0.5` から ε（fixture の
//   `lucida.epsilon`）以内のときだけ 1 段ずれてよい（torch の f32 の座標と重みの丸め — ADR 0125）。
//
// 加えて fixture の作り方に依らない仕様（量子化は四捨五入・既定は今の経路・未知の手順は落とす・
// 非有限の logit は画素座標つきで落ちる）を直接縛る。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { parseSafetensors, type SafetensorsFile } from "@karume/runtime";
import { MATTE_PROCEDURES, matteFromLogits, type MatteProcedure } from "../src/birefnet/matte.ts";
import { floatsOf, readBuffer, viewOf } from "./helpers/wan-ti2v-dit.ts";

const FIXTURE = new URL("fixtures/birefnet-matte/matte.json", import.meta.url);

/** fixture の入力表の 1 行（logit は safetensors の `<name>.logits`）。 */
type MatteInputRow = {
  readonly name: string;
  readonly width: number;
  readonly height: number;
};

/** fixture のケース表の 1 行（期待値は safetensors の `<name>.hr` / `<name>.lucida`）。 */
type MatteCaseRow = {
  readonly name: string;
  readonly input: string;
  readonly outWidth: number;
  readonly outHeight: number;
};

type MatteFixture = {
  readonly versions: Readonly<Record<string, string>>;
  readonly epsilon: number;
  readonly inputs: ReadonlyMap<string, MatteInputRow>;
  readonly cases: readonly MatteCaseRow[];
  readonly tensors: SafetensorsFile;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

const isPositiveInteger = (value: unknown): value is number =>
  typeof value === "number" && Number.isInteger(value) && value > 0;

const isInputRow = (value: unknown): value is MatteInputRow =>
  isRecord(value) && typeof value.name === "string" && isPositiveInteger(value.width) &&
  isPositiveInteger(value.height);

const isCaseRow = (value: unknown): value is MatteCaseRow =>
  isRecord(value) && typeof value.name === "string" && typeof value.input === "string" &&
  isPositiveInteger(value.outWidth) && isPositiveInteger(value.outHeight);

/**
 * fixture（入力表・ケース表の JSON + テンソルの safetensors）を読む。形が書き手の doc から外れたら
 * fail loudly。MUST: 無ければ投げる（git 追跡の fixture なので、無いのは取り違えか書き損じ）。
 */
const readFixture = async (): Promise<MatteFixture> => {
  const raw: unknown = JSON.parse(await Deno.readTextFile(FIXTURE));
  if (
    !isRecord(raw) || !isRecord(raw.source) || !isRecord(raw.source.versions) ||
    !isRecord(raw.lucida) || typeof raw.lucida.epsilon !== "number" ||
    typeof raw.tensors !== "string" || !Array.isArray(raw.inputs) ||
    !raw.inputs.every(isInputRow) || !Array.isArray(raw.cases) || !raw.cases.every(isCaseRow)
  ) {
    throw new Error(
      `${FIXTURE.pathname}: 形が {source.versions, lucida.epsilon, tensors, inputs, cases} でない`,
    );
  }
  const versions: Record<string, string> = {};
  for (const [name, version] of Object.entries(raw.source.versions)) {
    if (typeof version !== "string") {
      throw new Error(`${FIXTURE.pathname}: 版 ${name} が文字列でない`);
    }
    versions[name] = version;
  }
  return {
    versions,
    epsilon: raw.lucida.epsilon,
    inputs: new Map(raw.inputs.map((row) => [row.name, row])),
    cases: raw.cases,
    tensors: parseSafetensors(await readBuffer(new URL(raw.tensors, FIXTURE))),
  };
};

const fixture = await readFixture();

/** 入力の logit（F32 `[height, width]` — 形の食い違いは fail loudly）。 */
const logitsOf = (row: MatteInputRow): Float32Array<ArrayBuffer> => {
  const key = `${row.name}.logits`;
  assertEquals([...viewOf(fixture.tensors, key, "matte の fixture").shape], [
    row.height,
    row.width,
  ]);
  return floatsOf(fixture.tensors, key, "matte の fixture");
};

/** ケースの HR の最終 α（U8 `[outHeight, outWidth]`）。 */
const hrOf = (row: MatteCaseRow): Uint8Array<ArrayBuffer> => {
  const key = `${row.name}.hr`;
  const view = viewOf(fixture.tensors, key, "matte の fixture");
  assertEquals(view.dtype, "U8", `${key} の dtype`);
  assertEquals([...view.shape], [row.outHeight, row.outWidth], `${key} の形`);
  return new Uint8Array(fixture.tensors.buffer, view.byteOffset, view.byteLength);
};

/** ケースの Lucida の丸める前の `255α`（F32 `[outHeight, outWidth]`）。 */
const lucidaOf = (row: MatteCaseRow): Float32Array<ArrayBuffer> => {
  const key = `${row.name}.lucida`;
  assertEquals(
    [...viewOf(fixture.tensors, key, "matte の fixture").shape],
    [row.outHeight, row.outWidth],
    `${key} の形`,
  );
  return floatsOf(fixture.tensors, key, "matte の fixture");
};

const inputOf = (row: MatteCaseRow): MatteInputRow => {
  const input = fixture.inputs.get(row.input);
  if (input === undefined) throw new Error(`ケース ${row.name} の入力 ${row.input} が入力表に無い`);
  return input;
};

/** ケースを手順に通す。 */
const run = (row: MatteCaseRow, procedure: MatteProcedure): Uint8Array<ArrayBuffer> => {
  const input = inputOf(row);
  const matte = matteFromLogits(
    logitsOf(input),
    input.width,
    input.height,
    row.outWidth,
    row.outHeight,
    procedure,
  );
  assertEquals([matte.width, matte.height], [row.outWidth, row.outHeight], `${row.name} の寸法`);
  return matte.data;
};

describe("u8-pil-bicubic（HR の上流の意図 — Pillow の 8bit の BICUBIC）", () => {
  it("fixture の全ケースが上流の関数の最終 α と 0 不一致", () => {
    const failures: string[] = [];
    for (const row of fixture.cases) {
      const expected = hrOf(row);
      const actual = run(row, "u8-pil-bicubic");
      let mismatches = 0;
      let first = -1;
      for (let index = 0; index < expected.length; index += 1) {
        if (actual[index] === expected[index]) continue;
        mismatches += 1;
        if (first < 0) first = index;
      }
      if (mismatches > 0) {
        failures.push(
          `${row.name}: ${mismatches} / ${expected.length} 標本が違う` +
            `（最初は ${first}: ${actual[first]} 対 ${expected[first]}）`,
        );
      }
    }
    assertEquals(
      failures,
      [],
      `Pillow ${fixture.versions.pillow} の golden と ${fixture.cases.length} ケース中 ` +
        `${failures.length} ケースが違う`,
    );
  });

  it("量子化は 0.5 を切り上げる（上流の切り捨てではない）", () => {
    // 恒等寸法では resize が写しになり、量子化だけが残る。logit 0 → 255σ = 127.5 → 128（切り捨てなら
    // 127）、logit 7 → 254.77 → 255（切り捨てなら 254）。
    const matte = matteFromLogits(Float32Array.of(0, 7, -40, 40), 2, 2, 2, 2, "u8-pil-bicubic");
    assertEquals([...matte.data], [128, 255, 0, 255]);
  });
});

describe("float-bilinear（Lucida の上流の意図 = 今の経路）", () => {
  it("fixture の全ケースが上流の関数の最終 α と一致し、ずれは ε の内の 1 段だけ", () => {
    const { epsilon } = fixture;
    const failures: string[] = [];
    for (const row of fixture.cases) {
      const scaled = lucidaOf(row);
      const actual = run(row, "float-bilinear");
      for (let index = 0; index < scaled.length; index += 1) {
        const expected = Math.floor(scaled[index] + 0.5);
        if (actual[index] === expected) continue;
        const boundary = Math.min(actual[index], expected) + 0.5;
        if (
          Math.abs(actual[index] - expected) === 1 && Math.abs(scaled[index] - boundary) <= epsilon
        ) {
          continue;
        }
        failures.push(
          `${row.name}[${index}]: ${actual[index]} 対 ${expected}（golden の 255α = ${
            scaled[index]
          }）`,
        );
      }
    }
    assertEquals(failures.slice(0, 8), [], `ε ${epsilon} の外の不一致が ${failures.length} 標本`);
  });

  it("手順を省くと今の経路（パイプラインの既定は float-bilinear のまま）", () => {
    const row = fixture.cases.find((entry) => entry.outWidth > 1 && entry.outHeight > 1);
    assert(row !== undefined, "2 次元のケースが無い");
    const input = inputOf(row);
    const logits = logitsOf(input);
    assertEquals(
      [...matteFromLogits(logits, input.width, input.height, row.outWidth, row.outHeight).data],
      [...run(row, "float-bilinear")],
    );
  });
});

describe("手順の共通の約束", () => {
  it("fixture は拡大・縮小・横と縦の逆向き・片軸だけ・恒等を含み、グラフの 1024² を狙わない", () => {
    const shapes = fixture.cases.map((row) => ({ ...row, input: inputOf(row) }));
    const has = (predicate: (shape: (typeof shapes)[number]) => boolean, what: string): void =>
      assert(shapes.some(predicate), `${what}のケースが無い`);
    has(
      ({ input, outWidth, outHeight }) => outWidth > input.width && outHeight > input.height,
      "拡大",
    );
    has(
      ({ input, outWidth, outHeight }) => outWidth < input.width && outHeight < input.height,
      "縮小",
    );
    has(
      ({ input, outWidth, outHeight }) => outWidth < input.width && outHeight > input.height,
      "横は縮小・縦は拡大",
    );
    has(
      ({ input, outWidth, outHeight }) => outWidth !== input.width && outHeight === input.height,
      "横だけ",
    );
    has(
      ({ input, outWidth, outHeight }) => outWidth === input.width && outHeight !== input.height,
      "縦だけ",
    );
    has(
      ({ input, outWidth, outHeight }) =>
        outWidth === input.width && outHeight === input.height && input.width > 1,
      "恒等",
    );
    assert(
      shapes.every(({ outWidth, outHeight }) => outWidth !== 1024 || outHeight !== 1024),
      "グラフの寸法 1024² のケースがある（resize が恒等になり補間を見ない）",
    );
  });

  it("2 つの手順は段差の拡大で値が割れる（手順の取り違えを掴む）", () => {
    // 段差の拡大は bicubic の負のローブで bilinear と値が割れる（R6B-001 の 4 → 8 の実測の 1 例）。
    const step = Float32Array.of(-12, -12, 12, 12);
    const bilinear = matteFromLogits(step, 4, 1, 8, 1, "float-bilinear").data;
    const bicubic = matteFromLogits(step, 4, 1, 8, 1, "u8-pil-bicubic").data;
    assert([...bilinear].some((value, index) => value !== bicubic[index]), "2 つの手順が同じ値");
  });

  it("未知の手順の名前は落とす（Object.prototype のメンバも手順として受けない）", () => {
    for (const name of ["bilinear", "toString", "__proto__"]) {
      assertThrows(
        // 型の外の綴り（JS の呼び手）を模すための変換。
        () => matteFromLogits(new Float32Array(4), 2, 2, 2, 2, name as MatteProcedure),
        Error,
        "後処理の手順",
      );
    }
  });

  it("非有限の logit はどの手順でも画素座標つきで落ちる（NaN が透明画素に化けない）", () => {
    for (const procedure of MATTE_PROCEDURES) {
      assertThrows(
        () => matteFromLogits(Float32Array.of(0, 0, NaN, 0), 2, 2, 3, 3, procedure),
        Error,
        "マットの logit (x=0, y=1) が非有限値",
      );
    }
  });
});
