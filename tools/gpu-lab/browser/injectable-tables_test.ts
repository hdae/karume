import { assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { DEFAULT_GEOMETRY_PROFILE } from "../../../packages/runtime/src/kernels/geometry-profile.ts";
import { geometryProfileJson, sweepCaseSetId } from "../../../packages/runtime/tune.ts";
import { geometryProfileKernelsId } from "../../../packages/runtime/src/tune/fingerprint.ts";
import { infinityJson } from "../../../packages/runtime/src/tune/derive.ts";
import {
  type InjectableTables,
  LAST_GENERATED_KEY,
  type ProfileStorage,
  readLastGenerated,
  resolveSavedProfile,
  type SavedProfile,
  savedProfileId,
  tableForValue,
  tableOptions,
  valueForTable,
  writeLastGenerated,
} from "./injectable-tables.ts";

type GeneratedProfile = SavedProfile["profile"];

/**
 * 同じ id でも別の表（作り直した表を模す — 中身の同一性は参照で区別される）。生成した表と同じく
 * `provenance` を持ち、adapter の description を `description` で変えられる。
 */
const table = (id: string, description = "Apple M2"): GeneratedProfile => ({
  ...DEFAULT_GEOMETRY_PROFILE,
  id,
  provenance: {
    sweep: "sweep.json",
    sha256: "sha",
    date: "2026-10-01T00:00:00.000Z",
    candidateSet: "quick+",
    adapter: { vendor: "apple", architecture: "metal-3", device: "", description },
    kernels: "0123456789abcdef",
    caseSet: "fedcba9876543210",
  },
});

const SAVED_AT = "2026-10-01T03:04:05.000Z";

const saved = (profile: GeneratedProfile): SavedProfile => ({
  savedAt: SAVED_AT,
  checkout: { revision: "0123456789abcdef", dirty: false },
  profile,
});

/** localStorage に置く形（`profileJson` は表の `geometryProfileJson` — `overrides` で欄を差し替える）。 */
const stored = (
  value: SavedProfile,
  overrides: Readonly<Record<string, unknown>> = {},
): string =>
  JSON.stringify({
    savedAt: value.savedAt,
    checkout: value.checkout,
    profileJson: geometryProfileJson(value.profile),
    ...overrides,
  });

/** Map を背にした Storage（`failing` に挙げた操作は投げる — 無効な localStorage を模す）。 */
class FakeStorage implements ProfileStorage {
  readonly items = new Map<string, string>();
  constructor(private readonly failing: ReadonlySet<keyof ProfileStorage> = new Set()) {}
  private guard(operation: keyof ProfileStorage): void {
    if (this.failing.has(operation)) throw new DOMException(`${operation} denied`, "SecurityError");
  }
  getItem(key: string): string | null {
    this.guard("getItem");
    return this.items.get(key) ?? null;
  }
  setItem(key: string, value: string): void {
    this.guard("setItem");
    this.items.set(key, value);
  }
  removeItem(key: string): void {
    this.guard("removeItem");
    this.items.delete(key);
  }
}

describe("gpu lab injectable tables", () => {
  describe("options of the generated tables", () => {
    it("keeps every generated table as its own option, in serial order", () => {
      const first = table("gpu-a"), second = table("gpu-a"), third = table("gpu-b");
      const tables: InjectableTables = {
        generated: [
          { serial: 2, profile: second },
          { serial: 1, profile: first },
          { serial: 3, profile: third },
        ],
      };
      assertEquals(tableOptions(tables, undefined), [
        { value: "generated:1", text: "生成した表 #1: gpu-a（注入）" },
        { value: "generated:2", text: "生成した表 #2: gpu-a（注入）" },
        { value: "generated:3", text: "生成した表 #3: gpu-b（注入）" },
      ]);
    });

    it("marks only the applied table, told apart from a rebuilt table with the same id", () => {
      const first = table("gpu-a"), second = table("gpu-a");
      const tables: InjectableTables = {
        generated: [{ serial: 1, profile: first }, { serial: 2, profile: second }],
      };
      assertEquals(tableOptions(tables, first).map(({ text }) => text), [
        "生成した表 #1: gpu-a（注入）（適用中）",
        "生成した表 #2: gpu-a（注入）",
      ]);
    });

    it("lists the saved table first with its save time and the adapter description of its provenance", () => {
      const restored = table("gpu-saved");
      const tables: InjectableTables = {
        saved: saved(restored),
        generated: [{ serial: 1, profile: table("gpu-a") }],
      };
      assertEquals(tableOptions(tables, restored)[0], {
        value: "saved",
        text: `保存した表: gpu-saved（${
          new Date(SAVED_AT).toLocaleString()
        }・Apple M2）（注入）（適用中）`,
      });
      assertEquals(tableOptions(tables, restored)[1].value, "generated:1");
    });

    it("names the adapter by vendor/architecture when its description is empty", () => {
      const options = tableOptions(
        { saved: saved(table("gpu-saved", "")), generated: [] },
        undefined,
      );
      assertEquals(options[0].text.includes("・apple/metal-3）"), true);
    });

    it("maps option values to tables and back", () => {
      const restored = table("gpu-saved"), first = table("gpu-a"), second = table("gpu-a");
      const tables: InjectableTables = {
        saved: saved(restored),
        generated: [{ serial: 1, profile: first }, { serial: 2, profile: second }],
      };
      assertEquals(tableForValue(tables, "saved"), restored);
      assertEquals(tableForValue(tables, "generated:2") === second, true);
      assertEquals(valueForTable(tables, second), "generated:2");
      assertEquals(valueForTable(tables, restored), "saved");
      assertEquals(tableForValue(tables, "generated:9"), undefined);
      assertEquals(tableForValue(tables, "builtin:default"), undefined);
      assertEquals(tableForValue({ generated: [] }, "saved"), undefined);
      assertEquals(valueForTable(tables, table("gpu-a")), undefined);
    });
  });

  describe("the last generated table in storage", () => {
    it("restores what was written, including the Infinity of the last rule", () => {
      const storage = new FakeStorage();
      const written = saved(table("gpu-a"));
      writeLastGenerated(storage, written);
      const raw = storage.items.get(LAST_GENERATED_KEY) ?? "";
      assertEquals(raw.includes("1e999"), true);
      // adapter は表の provenance にだけある（外側に重ねて持たない）
      assertEquals(Object.keys(JSON.parse(raw)), ["savedAt", "checkout", "profileJson"]);
      const restored = readLastGenerated(storage);
      assertEquals(restored, written);
      assertEquals(restored?.profile.gemmRows.at(-1)?.maxRows, Infinity);
    });

    it("overwrites the previous table (only the last one is kept)", () => {
      const storage = new FakeStorage();
      writeLastGenerated(storage, saved(table("gpu-a")));
      writeLastGenerated(storage, saved(table("gpu-b")));
      assertEquals(storage.items.size, 1);
      assertEquals(readLastGenerated(storage)?.profile.id, "gpu-b");
    });

    it("reads nothing when nothing was saved", () => {
      assertEquals(readLastGenerated(new FakeStorage()), undefined);
    });

    for (
      const [context, value, reason] of [
        ["text that is not JSON", "{not json", "JSON"],
        [
          "a table without its provenance",
          stored(saved(table("gpu-a")), {
            profileJson: geometryProfileJson({ ...DEFAULT_GEOMETRY_PROFILE, id: "gpu-a" }),
          }),
          "provenance が無い",
        ],
        [
          "a savedAt that is not a date",
          stored(saved(table("gpu-a")), { savedAt: "soon" }),
          "savedAt",
        ],
        // 素の JSON.stringify は Infinity を null に落とす — parseGeometryProfileJson が名指して拒む
        [
          "a table whose last maxRows lost its Infinity",
          stored(saved(table("gpu-a")), { profileJson: JSON.stringify(table("gpu-a")) }),
          "Infinity の欠落",
        ],
        [
          "a table that the runtime gate rejects",
          stored(saved(table("gpu-a")), {
            profileJson: geometryProfileJson({
              ...table("gpu-a"),
              attention: {
                qk: { regM: 3, regN: 4, wgX: 16, wgY: 16 },
                pv: table("x").attention.pv,
              },
            }),
          }),
          "attention.qk",
        ],
        [
          "a table without its fields",
          stored(saved(table("gpu-a")), { profileJson: JSON.stringify({ id: "x" }) }),
          "に欄 gemmRows, attention, conv2d, i8a8 が無い",
        ],
        // /1 の形（表をオブジェクトのまま持つ）は読まない
        [
          "a value in the earlier shape (the table as an object)",
          JSON.stringify({ ...saved(table("gpu-a")), profileJson: undefined }),
          "profileJson が文字列でない",
        ],
      ] as const
    ) {
      it(`removes the key and throws with the reason for ${context}`, () => {
        const storage = new FakeStorage();
        storage.items.set(LAST_GENERATED_KEY, value);
        const error = assertThrows(() => readLastGenerated(storage), Error, LAST_GENERATED_KEY);
        assertStringIncludes(error.message, reason);
        assertEquals(storage.items.has(LAST_GENERATED_KEY), false);
      });
    }

    it("does not read a table saved under the earlier /1 key", () => {
      const storage = new FakeStorage();
      storage.items.set("karume-gpu-lab/last-generated-profile/1", stored(saved(table("gpu-a"))));
      assertEquals(readLastGenerated(storage), undefined);
    });

    it("refuses to save a table without its provenance, writing nothing", () => {
      const storage = new FakeStorage();
      assertThrows(
        () =>
          writeLastGenerated(storage, {
            ...saved(table("gpu-a")),
            profile: { ...DEFAULT_GEOMETRY_PROFILE, id: "gpu-a" },
          }),
        Error,
        "provenance が無い",
      );
      assertEquals(storage.items.size, 0);
    });

    it("lets a storage that refuses access throw, for the caller to report", () => {
      assertThrows(
        () => readLastGenerated(new FakeStorage(new Set(["getItem"]))),
        DOMException,
        "getItem denied",
      );
      assertThrows(
        () => writeLastGenerated(new FakeStorage(new Set(["setItem"])), saved(table("gpu-a"))),
        DOMException,
        "setItem denied",
      );
    });
  });

  describe("stored table, matched and injected (resolveSavedProfile — ADR 0117 stage 7)", () => {
    const ADAPTER = { vendor: "apple", architecture: "metal-3", device: "", description: "" };

    /** 今の runtime で照合が通る表（指紋とケース集合の版を今の値で焼く — 生成器が書く表と同じ材料）。 */
    const matching = (id: string): GeneratedProfile => ({
      ...DEFAULT_GEOMETRY_PROFILE,
      id,
      provenance: {
        sweep: "sweep.json",
        sha256: "sha",
        date: "2026-10-02T00:00:00.000Z",
        candidateSet: "quick+",
        userAgent: ["Mozilla/5.0 (Macintosh) Chrome/154.0.0.0"],
        adapter: ADAPTER,
        kernels: geometryProfileKernelsId(DEFAULT_GEOMETRY_PROFILE),
        caseSet: sweepCaseSetId(),
      },
    });

    it("injects the saved table when it matches the adapter", () => {
      const profile = matching("gpu-saved");
      const resolution = resolveSavedProfile(stored(saved(profile)), ADAPTER);
      if (resolution.kind !== "matched") throw Error(`not matched: ${JSON.stringify(resolution)}`);
      assertEquals(resolution.id, "gpu-saved");
      assertEquals(resolution.profile, profile);
    });

    for (const field of ["vendor", "architecture", "device", "description"] as const) {
      it(`does not inject when the adapter's ${field} differs, naming the field`, () => {
        const resolution = resolveSavedProfile(
          stored(saved(matching("gpu-saved"))),
          { ...ADAPTER, [field]: "other" },
        );
        assertEquals(resolution.kind, "mismatched");
        if (resolution.kind !== "mismatched") return;
        assertEquals(resolution.id, "gpu-saved");
        assertStringIncludes(resolution.reason, `adapter の ${field} が違う`);
      });
    }

    it("does not inject when the kernel fingerprint differs from the current runtime's", () => {
      const profile = matching("gpu-saved");
      const stale = {
        ...profile,
        provenance: { ...profile.provenance, kernels: "0123456789abcdef" },
      };
      const resolution = resolveSavedProfile(stored(saved(stale)), ADAPTER);
      assertEquals(resolution.kind, "mismatched");
      if (resolution.kind !== "mismatched") return;
      assertStringIncludes(resolution.reason, "カーネルの指紋が違う");
    });

    it("does not inject when nothing is saved", () => {
      const resolution = resolveSavedProfile(undefined, ADAPTER);
      assertEquals(resolution.kind, "missing");
      if (resolution.kind !== "missing") return;
      assertStringIncludes(resolution.reason, LAST_GENERATED_KEY);
    });

    for (
      const [name, text, reason] of [
        ["not JSON", "{", "読めない"],
        [
          "a table the profile parser rejects",
          stored(saved(matching("gpu-saved")), { profileJson: '{"id":"x"}' }),
          "幾何プロファイルの JSON",
        ],
        [
          "a userAgent that is not an array of strings",
          stored(saved(matching("gpu-saved")), {
            // 連結した文字列（配列化の前の形）— 表の JSON と同じ書き手で欄だけ差し替える
            profileJson: infinityJson({
              ...matching("gpu-saved"),
              provenance: {
                ...matching("gpu-saved").provenance,
                userAgent: "Mozilla/5.0 (Macintosh) Chrome/154.0.0.0",
              },
            }),
          }),
          "provenance.userAgent が配列でない",
        ],
      ] as const
    ) {
      it(`marks a broken saved value (${name}) to be discarded and does not inject, with the reason`, () => {
        const resolution = resolveSavedProfile(text, ADAPTER);
        assertEquals(resolution.kind, "broken");
        if (resolution.kind !== "broken") return;
        assertStringIncludes(resolution.reason, LAST_GENERATED_KEY);
        assertStringIncludes(resolution.reason, reason);
      });
    }

    it("reads the saved table's id for the record (undefined when the value is broken)", () => {
      assertEquals(savedProfileId(stored(saved(matching("gpu-saved")))), "gpu-saved");
      assertEquals(savedProfileId("{"), undefined);
    });
  });
});
