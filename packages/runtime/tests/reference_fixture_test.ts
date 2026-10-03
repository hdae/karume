// 環境別の参照値 fixture（models 側 e2e は `packages/models/tests/fixtures/references/*.json`、
// runtime 側 e2e は `packages/runtime/tests/fixtures/references/*.json`）の読み書き。
//
// ここで縛るのは**モードの意味論**そのもの: 未設定は比較だけ、`write` は無い行を足すだけ、
// `rewrite` は現環境の行だけを上書きする。この 3 つのどれかが緩むと、門が「他の機の参照値と
// 突き合わせて緑」「割れた値を黙って焼き直して緑」のどちらかに化ける。
//
// 実 GPU も環境変数も要らない — 環境とモードは引数で注入する。

import { assert, assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import type { Environment } from "./helpers/environment.ts";
import { parseSafetensors } from "../mod.ts";
import {
  f32ArtifactBytes,
  openReferences,
  parseReferenceMode,
  referenceEntryFields,
  referenceGateFindings,
  referenceGatePasses,
  type ReferenceHolds,
  referenceMismatchMessage,
  type ReferenceSettlement,
  settleOrObserve,
  settleReference,
  sha256Hex,
} from "./helpers/reference.ts";
import { openResults, type Results } from "./helpers/results.ts";

const CURRENT = "deno-intel-graphics-bmg-g21";
const OTHER = "deno-nvidia-geforce-rtx-3080-ti";

const environment = (key: string): Environment => ({
  key,
  runtime: { name: "deno", version: "2.9.6", v8: "15.0.245.2", typescript: "6.0.3" },
  adapter: { vendor: "32902", architecture: "", device: "57868", description: "test" },
  os: { platform: "linux", arch: "x86_64" },
});

/** 1 件だけ他環境の行を持つ fixture を一時ディレクトリに作る。 */
const withFixture = (
  cases: Record<string, Record<string, string>>,
  body: (fixtureUrl: URL) => void,
): void => {
  const dir = Deno.makeTempDirSync({ prefix: "karume-reference-" });
  try {
    const fixtureUrl = new URL("references.json", `file://${dir}/`);
    Deno.writeTextFileSync(
      fixtureUrl,
      `${JSON.stringify({ schema: 1, kind: "sha256", cases }, undefined, 2)}\n`,
    );
    body(fixtureUrl);
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
};

const SHA_A = "a".repeat(64);
const SHA_B = "b".repeat(64);

Deno.test("参照 fixture: lookup は現環境の行だけを返す", () => {
  withFixture({ "case-1": { [OTHER]: SHA_A } }, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: undefined,
    });
    assertEquals(references.lookup("case-1"), undefined, "他環境の行が現環境として見えている");
    assertEquals(referenceGatePasses(references, ["case-1"]), false);
    assertEquals(references.lacksReference("case-1"), true);
    const other = openReferences(fixtureUrl, { environment: environment(OTHER), mode: undefined });
    assertEquals(other.lookup("case-1"), SHA_A);
    assertEquals(referenceGatePasses(other, ["case-1"]), true);
    assertEquals(other.lacksReference("case-1"), false);
  });
});

Deno.test("参照 fixture: write は無い行を足し、既存の行は上書きしない", () => {
  withFixture({ "case-1": { [OTHER]: SHA_A } }, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: "write",
    });
    // 行が無い ⇒ 実測を足して緑。
    assertEquals(references.check("case-1", SHA_B), { status: "written" });
    assertEquals(references.lookup("case-1"), SHA_B);
    // 行がある ⇒ 通常どおり比較（write でも上書きしない）。
    assertEquals(references.check("case-1", SHA_B), { status: "pass", expected: SHA_B });
    assertEquals(references.check("case-1", SHA_A), { status: "fail", expected: SHA_B });
    assertEquals(references.lookup("case-1"), SHA_B, "write モードが既存の行を書き換えた");
    // 書き戻した内容は次の読みでも同じ（他環境の行は生きたまま）。
    const reopened = openReferences(fixtureUrl, {
      environment: environment(OTHER),
      mode: undefined,
    });
    assertEquals(reopened.lookup("case-1"), SHA_A, "他環境の行が write で動いた");
  });
});

Deno.test("参照 fixture: rewrite は現環境の行だけを上書きする", () => {
  withFixture({ "case-1": { [CURRENT]: SHA_A, [OTHER]: SHA_A } }, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: "rewrite",
    });
    assertEquals(references.check("case-1", SHA_B), { status: "rewritten", previous: SHA_A });
    assertEquals(references.lookup("case-1"), SHA_B);
    const other = openReferences(fixtureUrl, { environment: environment(OTHER), mode: undefined });
    assertEquals(other.lookup("case-1"), SHA_A, "rewrite が他環境の行を巻き込んだ");
  });
});

Deno.test("参照 fixture: 先に 2 ハンドルを開いて順に書いても双方の環境の行が残る", () => {
  withFixture({}, (fixtureUrl) => {
    // 両方を**書く前に**開く（開いた時点の写しをそのまま書き戻すと、後から書いた側が先の行を消す）。
    const current = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: "write",
    });
    const other = openReferences(fixtureUrl, { environment: environment(OTHER), mode: "write" });
    current.check("case-1", SHA_A);
    other.check("case-1", SHA_B);
    const document = JSON.parse(Deno.readTextFileSync(fixtureUrl)) as {
      cases: Record<string, Record<string, string>>;
    };
    assertEquals(
      document.cases["case-1"],
      { [CURRENT]: SHA_A, [OTHER]: SHA_B },
      "後から書いた環境が、先に書かれた他環境の行を巻き込んだ",
    );
  });
});

Deno.test("参照 fixture: 同じ環境キーの 2 ハンドルでも、後発の write は先発の行を上書きしない", () => {
  withFixture({}, (fixtureUrl) => {
    // 両方を**書く前に**開く。開いた時点の写しで判定すると、後発は「行が無い」と見なして
    // 先発の行を上書きする（`write` の意味論が rewrite に化ける）。
    const first = openReferences(fixtureUrl, { environment: environment(CURRENT), mode: "write" });
    const second = openReferences(fixtureUrl, { environment: environment(CURRENT), mode: "write" });
    assertEquals(first.check("case-1", SHA_A), { status: "written" });
    // 後発は最新のディスク行と比較する側へ落ちる（違う実測なら赤）。
    assertEquals(second.check("case-1", SHA_B), { status: "fail", expected: SHA_A });
    const document = JSON.parse(Deno.readTextFileSync(fixtureUrl)) as {
      cases: Record<string, Record<string, string>>;
    };
    assertEquals(
      document.cases["case-1"],
      { [CURRENT]: SHA_A },
      "後発の write が先発の行を上書きした",
    );
    // 逆側: 同じ実測なら緑（「常に赤」で通しているのではないことを固定する）。
    assertEquals(second.check("case-1", SHA_A), { status: "pass", expected: SHA_A });
  });
});

// --- 参照門の緑条件（登録したケース集合で数える）---------------------------------
//
// 門が守るのは「この環境の参照値が無いケースの sha 門が SKIP された」を無音の緑にしないこと。
// fixture 全体を横断して数えると、改名・削除で残った孤児行がその状態を隠す。

Deno.test("参照門: 孤児行だけが現環境の行を持つ状態は緑にしない", () => {
  // `retired` は登録されていないケース（改名・削除の跡）。現役 2 件は行を持たない。
  withFixture({ retired: { [CURRENT]: SHA_A } }, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: undefined,
    });
    assertEquals(
      referenceGatePasses(references, ["case-1", "case-2"]),
      false,
      "孤児行が現役ケースの全 SKIP を緑で隠した",
    );
  });
});

Deno.test("参照門: 現役ケースの行が 1 件も無ければ緑にしない（作るモードは別）", () => {
  withFixture({ "case-1": { [OTHER]: SHA_A } }, (fixtureUrl) => {
    const cases = ["case-1", "case-2"];
    const references = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: undefined,
    });
    assertEquals(referenceGatePasses(references, cases), false);
    // 参照値を作る走行は行が無くて当然なので緑（ADR 0106）。
    const writing = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: "write",
    });
    assertEquals(referenceGatePasses(writing, cases), true);
  });
});

// 「どれか 1 本に行があれば緑」だと、新しく足したケースの行の書き忘れが警告付き SKIP のまま
// 緑で通る。門は登録ケースの全部に行があるか、無いケースが明示の held 行であることを求める。

Deno.test("参照門: 登録ケースの 1 本でも現環境の行が無ければ赤（無いケースを名指しする）", () => {
  withFixture({ "case-1": { [CURRENT]: SHA_A }, "case-2": { [OTHER]: SHA_B } }, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: undefined,
    });
    const cases = ["case-1", "case-2"];
    assertEquals(referenceGatePasses(references, cases), false, "書き忘れの行が緑で通った");
    assertEquals(referenceGateFindings(references, cases), { missing: ["case-2"], staleHolds: [] });
  });
});

Deno.test("参照門: 登録ケースの全部に現環境の行があれば緑", () => {
  withFixture({ "case-1": { [CURRENT]: SHA_A }, "case-2": { [CURRENT]: SHA_B } }, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: undefined,
    });
    assertEquals(referenceGatePasses(references, ["case-1", "case-2"]), true);
  });
});

Deno.test("参照門: 行が無いケースが現環境の held 行なら緑・他環境の held 行は数えない", () => {
  withFixture({ "case-1": { [CURRENT]: SHA_A } }, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: undefined,
    });
    const cases = ["case-1", "case-2"];
    const heldHere: ReferenceHolds = { "case-2": { [CURRENT]: "この機では device lost" } };
    assertEquals(referenceGatePasses(references, cases, heldHere), true);
    // 別の機の held 行はこの機の欠けを免除しない（全機共通で止めると走れる機の検証まで消える）。
    const heldElsewhere: ReferenceHolds = { "case-2": { [OTHER]: "あの機では device lost" } };
    assertEquals(referenceGatePasses(references, cases, heldElsewhere), false);
    assertEquals(referenceGateFindings(references, cases, heldElsewhere).missing, ["case-2"]);
  });
});

Deno.test("参照門: held 行と現環境の行が両方あるケースは赤（古い held 行を名指しする）", () => {
  withFixture({ "case-1": { [CURRENT]: SHA_A } }, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: undefined,
    });
    const holds: ReferenceHolds = { "case-1": { [CURRENT]: "解消済みのはず" } };
    assertEquals(referenceGatePasses(references, ["case-1"], holds), false);
    assertEquals(referenceGateFindings(references, ["case-1"], holds), {
      missing: [],
      staleHolds: ["case-1"],
    });
  });
});

Deno.test("参照門: opt-in のケースは登録から外した走行では数えない", () => {
  // `opt-in` は環境変数で有効になるケース（行は別の機にしか無い）。無効の走行では呼び手が登録に
  // 含めない（e2e_wan_pipeline_test.ts の CASE_IDS の条件つき登録）— 門は fixture 全体ではなく
  // 登録した集合だけを数える。
  withFixture(
    { "case-1": { [CURRENT]: SHA_A }, "opt-in": { [OTHER]: SHA_B } },
    (fixtureUrl) => {
      const references = openReferences(fixtureUrl, {
        environment: environment(CURRENT),
        mode: undefined,
      });
      assertEquals(referenceGatePasses(references, ["case-1"]), true, "無効の opt-in を数えた");
      assertEquals(referenceGatePasses(references, ["case-1", "opt-in"]), false);
    },
  );
});

Deno.test("参照門: 走れる門に登録ケースが 0 件なら赤", () => {
  withFixture({ "case-1": { [CURRENT]: SHA_A } }, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: undefined,
    });
    assertEquals(referenceGatePasses(references, []), false);
  });
});

Deno.test("参照 fixture: 未設定モードで行が無いケースを突き合わせると throw する", () => {
  withFixture({}, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: undefined,
    });
    assertThrows(
      () => references.check("case-1", SHA_A),
      Error,
      "KARUME_REFERENCE=write",
    );
  });
});

Deno.test("参照 fixture: 書き出しは辞書順で安定する（2 度書いてバイト同一）", () => {
  withFixture({ "case-2": { [OTHER]: SHA_A }, "case-1": { [OTHER]: SHA_A } }, (fixtureUrl) => {
    openReferences(fixtureUrl, { environment: environment(CURRENT), mode: "write" }).check(
      "case-2",
      SHA_B,
    );
    const first = Deno.readTextFileSync(fixtureUrl);
    openReferences(fixtureUrl, { environment: environment(CURRENT), mode: "rewrite" }).check(
      "case-2",
      SHA_B,
    );
    const second = Deno.readTextFileSync(fixtureUrl);
    assertEquals(second, first, "同じ内容を書き直したのにバイトが動いた");
    // JSON の鍵は挿入順で読み戻るので、並びの検査は parse した鍵の列で足りる。
    const document = JSON.parse(first) as { cases: Record<string, Record<string, string>> };
    assertEquals(Object.keys(document.cases), ["case-1", "case-2"], "ケース名が辞書順でない");
    assertEquals(
      Object.keys(document.cases["case-2"]),
      [CURRENT, OTHER].sort(),
      "環境キーが辞書順でない",
    );
    assert(first.endsWith("}\n"), "末尾改行が無い");
  });
});

Deno.test("参照 fixture: 読めない fixture は空として扱わず throw する", () => {
  withFixture({}, (fixtureUrl) => {
    assertThrows(
      () =>
        openReferences(new URL("missing.json", fixtureUrl), {
          environment: environment(CURRENT),
          mode: undefined,
        }),
      Error,
      "fixture が読めない",
    );
    Deno.writeTextFileSync(fixtureUrl, "{ broken");
    assertThrows(
      () => openReferences(fixtureUrl, { environment: environment(CURRENT), mode: undefined }),
      Error,
      "JSON として壊れている",
    );
    Deno.writeTextFileSync(fixtureUrl, JSON.stringify({ schema: 2, kind: "sha256", cases: {} }));
    assertThrows(
      () => openReferences(fixtureUrl, { environment: environment(CURRENT), mode: undefined }),
      Error,
      "schema が 2",
    );
  });
});

Deno.test("参照 fixture: KARUME_REFERENCE の未知の綴りは throw する", () => {
  assertEquals(parseReferenceMode(undefined), undefined);
  assertEquals(parseReferenceMode("write"), "write");
  assertEquals(parseReferenceMode("rewrite"), "rewrite");
  assertThrows(() => parseReferenceMode("1"), Error, "KARUME_REFERENCE");
  assertThrows(() => parseReferenceMode(""), Error, "KARUME_REFERENCE");
});

// --- 参照値の決着（settleReference と、その結果・診断への写し）-----------------------
//
// ここで縛るのは「実物は突合の成否に依らず結果の席に残る」「sha は実物のバイト列から採る」
// 「決着の写しが expected の有無を取り違えない」の 3 つ。どれかが崩れると、割れた回の A/B の
// 材料が消えるか、結果 JSON が参照値を作った回を「何かと突き合わせた」と誤読させる。

/** fixture と結果の席を同じ一時ディレクトリに置く（実 GPU も outputs/ も要らない）。 */
const withSeats = async (
  cases: Record<string, Record<string, string>>,
  body: (fixtureUrl: URL, results: Results) => Promise<void>,
): Promise<void> => {
  const dir = Deno.makeTempDirSync({ prefix: "karume-reference-settle-" });
  try {
    const base = new URL(`file://${dir}/`);
    const fixtureUrl = new URL("references.json", base);
    Deno.writeTextFileSync(
      fixtureUrl,
      `${JSON.stringify({ schema: 1, kind: "sha256", cases }, undefined, 2)}\n`,
    );
    const results = openResults("settle", {
      root: new URL("verify/", base),
      environment: environment(CURRENT),
    });
    await body(fixtureUrl, results);
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
};

const BYTES = new TextEncoder().encode("karume-artifact");

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

/** fixture 文書の `cases`（ケース → 環境キー → sha）の形か。 */
const isCasesDocument = (
  value: unknown,
): value is { readonly cases: Record<string, Record<string, string>> } =>
  isRecord(value) && isRecord(value.cases) &&
  Object.values(value.cases).every((rows) =>
    isRecord(rows) && Object.values(rows).every((sha) => typeof sha === "string")
  );

/** ディスク上の fixture の `cases` を読む（型ガードで絞る — 形が崩れていれば落ちる）。 */
const readCases = (fixtureUrl: URL): Record<string, Record<string, string>> => {
  const document: unknown = JSON.parse(Deno.readTextFileSync(fixtureUrl));
  assert(isCasesDocument(document), `${fixtureUrl.pathname} が { cases } の形でない`);
  return document.cases;
};

describe("sha256Hex", () => {
  it("FIPS 180-2 の既知ベクトル（'abc'）と一致する小文字 16 進 64 桁を返す", async () => {
    assertEquals(
      await sha256Hex(new TextEncoder().encode("abc")),
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
  });
});

describe("settleReference", () => {
  it("write モードで行が無ければ実測を行として書き、実物を結果の席に残す", async () => {
    await withSeats({}, async (fixtureUrl, results) => {
      const references = openReferences(fixtureUrl, {
        environment: environment(CURRENT),
        mode: "write",
      });
      const settlement = await settleReference(references, results, {
        id: "case-1",
        artifact: "case-1.png",
        bytes: BYTES,
      });
      const expectedSha = await sha256Hex(BYTES);
      assertEquals(settlement.check, { status: "written" });
      assertEquals(settlement.sha256, expectedSha);
      assertEquals(settlement.artifactUrl.href, new URL("case-1.png", results.dir).href);
      // sha は実物ファイルのバイト列と一致する（結果の actual と実物が食い違わない）。
      const onDisk = await Deno.readFile(settlement.artifactUrl);
      assertEquals(onDisk, BYTES);
      assertEquals(await sha256Hex(onDisk), settlement.sha256);
      assertEquals(readCases(fixtureUrl)["case-1"], { [CURRENT]: expectedSha });
    });
  });

  it("比較モードで行と食い違えば fail を返し、実物は書かれ、行は動かない", async () => {
    await withSeats({ "case-1": { [CURRENT]: SHA_A } }, async (fixtureUrl, results) => {
      const before = Deno.readTextFileSync(fixtureUrl);
      const references = openReferences(fixtureUrl, {
        environment: environment(CURRENT),
        mode: undefined,
      });
      const settlement = await settleReference(references, results, {
        id: "case-1",
        artifact: "case-1.safetensors",
        bytes: BYTES,
      });
      assertEquals(settlement.check, { status: "fail", expected: SHA_A });
      assertEquals(
        await Deno.readFile(settlement.artifactUrl),
        BYTES,
        "不一致の回に実物が残っていない",
      );
      assertEquals(Deno.readTextFileSync(fixtureUrl), before, "比較モードが参照値を書き換えた");
    });
  });

  it("比較モードで一致すれば pass を返す", async () => {
    const sha = await sha256Hex(BYTES);
    await withSeats({ "case-1": { [CURRENT]: sha } }, async (fixtureUrl, results) => {
      const references = openReferences(fixtureUrl, {
        environment: environment(CURRENT),
        mode: undefined,
      });
      const settlement = await settleReference(references, results, {
        id: "case-1",
        artifact: "case-1.png",
        bytes: BYTES,
      });
      assertEquals(settlement.check, { status: "pass", expected: sha });
    });
  });
});

// 追加検査としての sha の末端。行が無いケースの決着の形を 1 つに決めるのがこの関数の役目で、
// 崩れると「実測 sha を積む系列と何も積まない系列」が再び混ざる（環境横断の actual 比較が
// 一部の系列でしか効かなくなる）。
describe("settleOrObserve", () => {
  it("比較モードで行が無ければ突合を飛ばし、実物を書いて pass + actual + artifact（expected 無し）を返す", async () => {
    await withSeats({ "case-1": { [OTHER]: SHA_A } }, async (fixtureUrl, results) => {
      const before = Deno.readTextFileSync(fixtureUrl);
      const references = openReferences(fixtureUrl, {
        environment: environment(CURRENT),
        mode: undefined,
      });
      const outcome = await settleOrObserve(references, results, {
        id: "case-1",
        artifact: "case-1.safetensors",
        bytes: BYTES,
      });
      assertEquals(outcome.settlement, undefined);
      assertEquals(outcome.fields, {
        status: "pass",
        actual: await sha256Hex(BYTES),
        artifact: "case-1.safetensors",
      });
      assertEquals(Object.keys(outcome.fields), ["status", "actual", "artifact"]);
      assertEquals(
        await Deno.readFile(new URL("case-1.safetensors", results.dir)),
        BYTES,
        "突合を飛ばした回に実物が残っていない",
      );
      // 他環境の行（OTHER）とは突き合わせず、この環境の行も作らない。
      assertEquals(Deno.readTextFileSync(fixtureUrl), before, "比較モードが参照値を書き換えた");
    });
  });

  it("比較モードで行があれば突き合わせ、決着と referenceEntryFields の欄を返す", async () => {
    await withSeats({ "case-1": { [CURRENT]: SHA_A } }, async (fixtureUrl, results) => {
      const references = openReferences(fixtureUrl, {
        environment: environment(CURRENT),
        mode: undefined,
      });
      const outcome = await settleOrObserve(references, results, {
        id: "case-1",
        artifact: "case-1.safetensors",
        bytes: BYTES,
      });
      assertEquals(outcome.settlement?.check, { status: "fail", expected: SHA_A });
      assertEquals(outcome.fields, {
        status: "fail",
        expected: SHA_A,
        actual: await sha256Hex(BYTES),
        artifact: "case-1.safetensors",
      });
    });
  });

  it("write モードで行が無ければ行を作り、written の欄を返す", async () => {
    await withSeats({}, async (fixtureUrl, results) => {
      const references = openReferences(fixtureUrl, {
        environment: environment(CURRENT),
        mode: "write",
      });
      const outcome = await settleOrObserve(references, results, {
        id: "case-1",
        artifact: "case-1.safetensors",
        bytes: BYTES,
      });
      const sha = await sha256Hex(BYTES);
      assertEquals(outcome.settlement?.check, { status: "written" });
      assertEquals(outcome.fields, {
        status: "written",
        actual: sha,
        artifact: "case-1.safetensors",
      });
      assertEquals(readCases(fixtureUrl)["case-1"], { [CURRENT]: sha });
    });
  });
});

describe("f32ArtifactBytes", () => {
  const data = Float32Array.from([0.5, -1, 2.25, Number.MIN_VALUE, -0, 3]);

  it("テンソル名・F32・shape・値のビット列をそのまま持ち、metadata を持たない", () => {
    const bytes = f32ArtifactBytes("logits", [1, 2, 3], data);
    const parsed = parseSafetensors(bytes.buffer);
    assertEquals([...parsed.tensors.keys()], ["logits"]);
    assertEquals(parsed.metadata.size, 0);
    const view = parsed.tensors.get("logits");
    assert(view !== undefined);
    assertEquals(view.dtype, "F32");
    assertEquals(view.shape, [1, 2, 3]);
    // ビット列で比べる（-0 と 0 を同一視しない）。
    assertEquals(
      new Uint8Array(parsed.buffer, view.byteOffset, view.byteLength),
      new Uint8Array(data.buffer),
    );
  });

  it("同じ入力から同じバイト列を作る（sha が走行ごとに動かない）", () => {
    assertEquals(
      f32ArtifactBytes("logits", [1, 2, 3], data),
      f32ArtifactBytes("logits", [1, 2, 3], data),
    );
  });

  it("shape の要素数とデータ長が食い違えば throw する", () => {
    assertThrows(() => f32ArtifactBytes("logits", [1, 2, 2], data), Error, "食い違う");
  });
});

/** 写しの検査用の決着（実物の置き場は写しに関与しない）。 */
const settlementOf = (check: ReferenceSettlement["check"]): ReferenceSettlement => ({
  sha256: SHA_B,
  check,
  artifactUrl: new URL("file:///verify/case-1.png"),
});

describe("referenceEntryFields", () => {
  it("written は expected を欄ごと持たない", () => {
    const fields = referenceEntryFields(settlementOf({ status: "written" }), "case-1.png");
    assertEquals(fields, { status: "written", actual: SHA_B, artifact: "case-1.png" });
    assertEquals(Object.keys(fields), ["status", "actual", "artifact"]);
  });

  it("rewritten は焼き直す前の値を expected に持つ", () => {
    const fields = referenceEntryFields(
      settlementOf({ status: "rewritten", previous: SHA_A }),
      "case-1.png",
    );
    assertEquals(Object.keys(fields), ["status", "expected", "actual", "artifact"]);
    assertEquals(fields, {
      status: "rewritten",
      expected: SHA_A,
      actual: SHA_B,
      artifact: "case-1.png",
    });
  });

  it("pass / fail は突き合わせた行を expected に持つ", () => {
    for (const status of ["pass", "fail"] as const) {
      assertEquals(
        referenceEntryFields(settlementOf({ status, expected: SHA_A }), "case-1.png"),
        { status, expected: SHA_A, actual: SHA_B, artifact: "case-1.png" },
      );
    }
  });
});

describe("referenceMismatchMessage", () => {
  const fixtureUrl = new URL("file:///repo/fixtures/references/family.json");

  it("期待 / 実測の sha・実物の置き場・fixture の置き場を含む", () => {
    const message = referenceMismatchMessage(
      "case-1",
      settlementOf({ status: "fail", expected: SHA_A }),
      { fixtureUrl },
    );
    assertStringIncludes(message, "case-1");
    assertStringIncludes(message, `期待 ${SHA_A}`);
    assertStringIncludes(message, `実際 ${SHA_B}`);
    assertStringIncludes(message, "/verify/case-1.png");
    assertStringIncludes(message, "/repo/fixtures/references/family.json");
    assertStringIncludes(message, "tolerance に逃げない");
  });

  it("不一致でない決着を渡されたら throw する（呼び手の誤りを黙って文にしない）", () => {
    assertThrows(
      () => referenceMismatchMessage("case-1", settlementOf({ status: "written" }), { fixtureUrl }),
      Error,
      "'written'",
    );
  });
});
