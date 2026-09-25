// 検証結果の書き出し（`outputs/verify/<環境キー>/<日付>_<系列>/`）。
//
// ここで見るのは 3 つ: 席が環境キーで分かれること、`record` のたびに `results.json` が
// **その時点までの全件**で書き直されること（途中で落ちても直前までが残る）、実物の置き場が
// 同じディレクトリ配下に閉じること。実 GPU も `outputs/` も要らない — 根と環境は注入する。

import {
  assert,
  assertEquals,
  assertRejects,
  assertStrictEquals,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import { parseResultsDocument } from "../../../tools/verify-diff/diff.ts";
import type { Environment } from "./helpers/environment.ts";
import {
  type Measurement,
  openResults,
  recordFailure,
  type ResultEntry,
  type Results,
  runRecordedCase,
} from "./helpers/results.ts";

const KEY = "deno-intel-graphics-bmg-g21";

const ENVIRONMENT: Environment = {
  key: KEY,
  runtime: { name: "deno", version: "2.9.6", v8: "15.0.245.2", typescript: "6.0.3" },
  adapter: { vendor: "32902", architecture: "", device: "57868", description: "test" },
  os: { platform: "linux", arch: "x86_64" },
};

type ResultsDocument = {
  readonly schema: number;
  readonly family: string;
  readonly environment: Environment;
  readonly checkout: { readonly sha: string; readonly dirty: boolean } | null;
  readonly startedAt: string;
  readonly cases: readonly ResultEntry[];
};

Deno.test("結果の書き出し: record のたびに全件で書き直され、実物は同じ席に置かれる", async () => {
  const temporary = Deno.makeTempDirSync({ prefix: "karume-verify-" });
  try {
    const root = new URL(`file://${temporary}/`);
    const results = openResults("irodori", { root, environment: ENVIRONMENT });
    const today = new Date().toISOString().slice(0, 10);
    assertEquals(results.dir.href, new URL(`${KEY}/${today}_irodori/`, root).href);

    const artifact = results.artifact("voice-clone.wav");
    assert(artifact.href.startsWith(results.dir.href), "実物が結果の席の外を指している");
    await Deno.writeFile(artifact, new Uint8Array([1, 2, 3]));

    await results.record({
      id: "voice-clone",
      status: "pass",
      expected: "a".repeat(64),
      actual: "a".repeat(64),
      artifact: "voice-clone.wav",
      elapsedMs: 12,
    });
    const afterFirst = JSON.parse(
      await Deno.readTextFile(new URL("results.json", results.dir)),
    ) as ResultsDocument;
    assertEquals(afterFirst.cases.length, 1, "1 件目が残っていない");

    await results.record({ id: "no-ref", status: "written", actual: "b".repeat(64), elapsedMs: 8 });
    const document = JSON.parse(
      await Deno.readTextFile(new URL("results.json", results.dir)),
    ) as ResultsDocument;
    assertEquals(document.schema, 1);
    assertEquals(document.family, "irodori");
    assertEquals(document.environment.key, KEY);
    assertEquals(document.cases.map((entry) => entry.id), ["voice-clone", "no-ref"]);
    assertEquals(document.cases[1].status, "written");
    // 参照値を作った回は expected を持たない（JSON.stringify が undefined の欄を落とす）。
    assertEquals(Object.hasOwn(document.cases[1], "expected"), false);
    assert(document.startedAt.endsWith("Z"), "startedAt が ISO 文字列でない");
    // このリポは git 管理下なのでチェックアウトは採れる（git が無い機でだけ null）。
    assert(
      document.checkout === null || /^[0-9a-f]{40}$/.test(document.checkout.sha),
      `checkout の sha が commit SHA でない: ${JSON.stringify(document.checkout)}`,
    );
  } finally {
    Deno.removeSync(temporary, { recursive: true });
  }
});

Deno.test("結果の書き出し: 席を作った時点で走行中（cases 空）の results.json が置かれる", async () => {
  const temporary = Deno.makeTempDirSync({ prefix: "karume-verify-" });
  try {
    const root = new URL(`file://${temporary}/`);
    const results = openResults("golden", { root, environment: ENVIRONMENT });
    // 前回の走行が同じ席（同じ日・同じ系列）に残した決着。
    Deno.mkdirSync(results.dir, { recursive: true });
    Deno.writeTextFileSync(
      new URL("results.json", results.dir),
      `${JSON.stringify({ schema: 1, cases: [{ id: "activations", status: "pass" }] })}\n`,
    );
    // 実物だけ置いて record には到達しない = 記録の前に例外で抜けた回。
    await Deno.writeFile(results.artifact("activations.png"), new Uint8Array([1, 2, 3]));
    const document = JSON.parse(
      await Deno.readTextFile(new URL("results.json", results.dir)),
    ) as ResultsDocument;
    assertEquals(document.cases, [], "前回の走行の決着が今回の実物と同じ席に残っている");
    assertEquals(document.family, "golden");
    assert(document.startedAt.endsWith("Z"), "走行中マーカーが今回の startedAt を名乗っていない");
  } finally {
    Deno.removeSync(temporary, { recursive: true });
  }
});

Deno.test("結果の書き出し: 許容差の実測が record に載せたとおりの形で残る", async () => {
  const temporary = Deno.makeTempDirSync({ prefix: "karume-verify-" });
  try {
    const root = new URL(`file://${temporary}/`);
    const results = openResults("golden", { root, environment: ENVIRONMENT });
    // 合格した回の実測（これが残らないと「どれだけ差が出たか」は赤くなるまで分からない）。
    const measurements: readonly Measurement[] = [
      {
        output: "sin",
        maxAbs: 2.68e-5,
        maxRel: 1.72e-5,
        tolerance: { atol: 2 ** -11, rtol: 0 },
        stage: "spec",
      },
      {
        output: "gelu",
        maxAbs: 1.19e-7,
        maxRel: 1.83e-6,
        tolerance: { atol: 1e-6, rtol: 1e-5 },
        stage: "karume",
      },
    ];
    await results.record({ id: "activations", status: "pass", elapsedMs: 31, measurements });
    const document = JSON.parse(
      await Deno.readTextFile(new URL("results.json", results.dir)),
    ) as ResultsDocument;
    assertEquals(document.cases[0].measurements, measurements);
  } finally {
    Deno.removeSync(temporary, { recursive: true });
  }
});

Deno.test("結果の書き出し: 非有限の実測は null として書かれる", async () => {
  const temporary = Deno.makeTempDirSync({ prefix: "karume-verify-" });
  try {
    const root = new URL(`file://${temporary}/`);
    const results = openResults("golden", { root, environment: ENVIRONMENT });
    // NaN / ±Inf（出力に非有限が混ざった回の実測）。JSON に綴りが無いので null になる —
    // 読む側（tools/verify-diff）はこの null を「測れなかった」として受ける。
    await results.record({
      id: "activations",
      status: "fail",
      elapsedMs: 9,
      measurements: [{
        output: "sin",
        maxAbs: Number.POSITIVE_INFINITY,
        maxRel: Number.NaN,
        tolerance: { atol: 1e-6, rtol: 1e-5 },
        stage: "karume",
      }],
    });
    const text = await Deno.readTextFile(new URL("results.json", results.dir));
    assertStringIncludes(text, `"maxAbs": null`);
    assertStringIncludes(text, `"maxRel": null`);
  } finally {
    Deno.removeSync(temporary, { recursive: true });
  }
});

// 失敗した回の決着は「残すのが務め」だが、その書き込みが落ちたときに throw すると、呼び手の
// catch が抱えている元の検証例外（何が壊れたのかを言う唯一の診断）が I/O 例外に置き換わる。
Deno.test("結果の書き出し: 記録できなくても recordFailure は throw せず、黙りもしない", async () => {
  const temporary = Deno.makeTempDirSync({ prefix: "karume-verify-" });
  try {
    const root = new URL(`file://${temporary}/`);
    // 故障注入: 席の親（環境キーのディレクトリ）を通常ファイルにする = 席を作れない置き場。
    Deno.writeTextFileSync(new URL(KEY, root), "");
    const results = openResults("golden", { root, environment: ENVIRONMENT });
    const entry: ResultEntry = { id: "activations", status: "fail", elapsedMs: 3 };
    // 故障注入が効いていることを先に固定する（素の record が通るなら下の検査は何も言わない）。
    await assertRejects(() => results.record(entry));

    const said: string[] = [];
    const original = console.error;
    console.error = (...args: unknown[]): void => {
      said.push(args.map(String).join(" "));
    };
    try {
      // 元の検証例外を置き換えないので、ここは投げずに戻る。
      await recordFailure(results, entry);
    } finally {
      console.error = original;
    }
    assertEquals(said.length, 1, "記録できなかったことを黙って飲み込んだ");
    assertStringIncludes(said[0], "activations");
  } finally {
    Deno.removeSync(temporary, { recursive: true });
  }
});

Deno.test("結果の書き出し: ディレクトリ区切りを含む実物の名前は throw する", () => {
  const temporary = Deno.makeTempDirSync({ prefix: "karume-verify-" });
  try {
    const results = openResults("anima", {
      root: new URL(`file://${temporary}/`),
      environment: ENVIRONMENT,
    });
    assertThrows(() => results.artifact("F1/i8.wav"), Error, "ディレクトリ区切り");
  } finally {
    Deno.removeSync(temporary, { recursive: true });
  }
});

// runRecordedCase（実重み golden の各ケースを包む枠）。書き出しの実体は上で見ているので、ここは
// 記録の口を fake にして「どの経路でも決着が同じ形で 1 件積まれる」ことだけを見る。

/** 積まれた決着を手元に残す fake（`record` を落とす故障注入つき）。 */
const fakeResults = (failRecord = false): { results: Results; entries: ResultEntry[] } => {
  const entries: ResultEntry[] = [];
  const results: Results = {
    dir: new URL("file:///fake/"),
    artifact: (name: string): URL => new URL(name, "file:///fake/"),
    record: (entry: ResultEntry): Promise<void> => {
      if (failRecord) return Promise.reject(new Error("results.json を書けない"));
      entries.push(entry);
      return Promise.resolve();
    },
  };
  return { results, entries };
};

const MEASUREMENT: Measurement = {
  output: "logits",
  maxAbs: 1e-4,
  maxRel: 2e-4,
  tolerance: { atol: 1e-3, rtol: 1e-3 },
  stage: "karume",
};

/** console.error を拾いながら `action` を待つ。 */
const captureConsoleError = async (action: () => Promise<void>): Promise<string[]> => {
  const said: string[] = [];
  const original = console.error;
  console.error = (...args: unknown[]): void => {
    said.push(args.map(String).join(" "));
  };
  try {
    await action();
  } finally {
    console.error = original;
  }
  return said;
};

Deno.test("runRecordedCase: 本体が戻った回は pass と積んだ実測を 1 件残す", async () => {
  const { results, entries } = fakeResults();
  await runRecordedCase(results, { id: "case0" }, ({ measurements }) => {
    measurements.push(MEASUREMENT);
    return Promise.resolve(undefined);
  });
  assertEquals(entries.length, 1);
  const [entry] = entries;
  assertEquals(Object.keys(entry), ["id", "status", "elapsedMs", "measurements"]);
  assertEquals(entry.id, "case0");
  assertEquals(entry.status, "pass");
  assertEquals(entry.measurements, [MEASUREMENT]);
  assert(Number.isInteger(entry.elapsedMs) && entry.elapsedMs >= 0, "elapsedMs が整数 ms でない");
});

Deno.test("runRecordedCase: 本体が投げた回は fail を残してから同じ例外を投げ直す", async () => {
  const { results, entries } = fakeResults();
  const cause = new Error("output.0: allclose 不一致");
  const order: string[] = [];
  const thrown = await assertRejects(() =>
    runRecordedCase(results, {
      id: "case1",
      onFailure: () => {
        order.push(`onFailure（積んだ件数 ${entries.length}）`);
      },
    }, ({ measurements }) => {
      // 落ちる前に測れた分も決着に載る（どこまで測れたかが失敗の診断になる）。
      measurements.push(MEASUREMENT);
      return Promise.reject(cause);
    })
  );
  assertStrictEquals(thrown, cause, "元の検証例外が別の例外に置き換わった");
  assertEquals(entries.length, 1);
  assertEquals(Object.keys(entries[0]), ["id", "status", "elapsedMs", "measurements"]);
  assertEquals(entries[0].status, "fail");
  assertEquals(entries[0].measurements, [MEASUREMENT]);
  // 系列側の「ケースの席に残った」印は記録より先に立つ。
  assertEquals(order, ["onFailure（積んだ件数 0）"]);
});

Deno.test("runRecordedCase: 本体が fail を返した回は投げずに note つきで残す", async () => {
  const { results, entries } = fakeResults();
  await runRecordedCase(
    results,
    { id: "gelu" },
    () => Promise.resolve({ status: "fail", note: "仕様帯で受理: sin" }),
  );
  assertEquals(entries.length, 1);
  assertEquals(Object.keys(entries[0]), ["id", "status", "elapsedMs", "note", "measurements"]);
  assertEquals(entries[0].status, "fail");
  assertEquals(entries[0].note, "仕様帯で受理: sin");
  assertEquals(entries[0].measurements, []);
});

Deno.test("runRecordedCase: 投げた回の note は failureNote が例外から作る", async () => {
  const { results, entries } = fakeResults();
  await assertRejects(
    () =>
      runRecordedCase(results, {
        id: "gelu",
        failureNote: (cause) => `例外: ${cause instanceof Error ? cause.message : String(cause)}`,
      }, () => Promise.reject(new Error("createSession 失敗"))),
    Error,
    "createSession 失敗",
  );
  assertEquals(Object.keys(entries[0]), ["id", "status", "elapsedMs", "note", "measurements"]);
  assertEquals(entries[0].note, "例外: createSession 失敗");
});

Deno.test("runRecordedCase: 投げた回に記録が落ちても元の例外を投げ、記録できなかったと言う", async () => {
  const { results } = fakeResults(true);
  const cause = new Error("shape 不一致");
  let thrown: unknown;
  const said = await captureConsoleError(async () => {
    thrown = await assertRejects(() =>
      runRecordedCase(results, { id: "case2" }, () => Promise.reject(cause))
    );
  });
  assertStrictEquals(thrown, cause, "元の検証例外が I/O 例外に置き換わった");
  assertEquals(said.length, 1, "記録できなかったことを黙って飲み込んだ");
  assertStringIncludes(said[0], "case2");
});

Deno.test("runRecordedCase: 戻った回に記録が落ちたら投げ、onFailure は呼ばない", async () => {
  const { results } = fakeResults(true);
  let failed = false;
  await assertRejects(
    () =>
      runRecordedCase(results, {
        id: "case3",
        onFailure: () => {
          failed = true;
        },
      }, () => Promise.resolve(undefined)),
    Error,
    "results.json を書けない",
  );
  // ケースの決着は残っていない — 系列側が「ケースの席に残った」と誤認すると、どこにも残らない。
  assertEquals(failed, false);
});

// 書き手（helpers/results.ts）と読み手（tools/verify-diff）は型を別々に持つ。ADR 0106 追記の
// 「schema は 1 のまま」は両者の一致に依存するので、書き手が実際に書いた文書を読み手へ通す。
Deno.test("往復: runRecordedCase が書いた results.json を verify-diff の読み手がそのまま受ける", async () => {
  const temporary = Deno.makeTempDirSync({ prefix: "karume-verify-" });
  try {
    const root = new URL(`file://${temporary}/`);
    const results = openResults("golden", { root, environment: ENVIRONMENT });
    await runRecordedCase(results, { id: "measured" }, ({ measurements }) => {
      measurements.push({ ...MEASUREMENT, maxAbs: Number.NaN });
      return Promise.resolve(undefined);
    });
    const cause = new Error("測る前に落ちた");
    await assertRejects(
      () =>
        runRecordedCase(
          results,
          { id: "thrown", failureNote: String },
          () => Promise.reject(cause),
        ),
    );

    const text = await Deno.readTextFile(new URL("results.json", results.dir));
    const document = parseResultsDocument(JSON.parse(text));
    assertEquals(document.family, "golden");
    assertEquals(document.environment.key, KEY);
    assertEquals(document.cases.map((entry) => [entry.id, entry.status]), [
      ["measured", "pass"],
      ["thrown", "fail"],
    ]);
    // 非有限は書き手の JSON.stringify で null になり、読み手は「測れなかった」として受ける。
    assertEquals(document.cases[0].measurements, [{ ...MEASUREMENT, maxAbs: null }]);
    // runRecordedCase 経由は測る前に投げた回も欄を持つ（空配列）。
    assertEquals(document.cases[1].measurements, []);
    assertEquals(document.cases[1].note, "Error: 測る前に落ちた");
  } finally {
    Deno.removeSync(temporary, { recursive: true });
  }
});
