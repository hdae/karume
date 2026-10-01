import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import type { GeometryProfile } from "../../../packages/runtime/mod.ts";
import { DEFAULT_GEOMETRY_PROFILE } from "../../../packages/runtime/src/kernels/geometry-profile.ts";
import {
  type InjectableTables,
  LAST_GENERATED_KEY,
  type ProfileStorage,
  readLastGenerated,
  type SavedProfile,
  tableForValue,
  tableOptions,
  valueForTable,
  writeLastGenerated,
} from "./injectable-tables.ts";

/** 同じ id でも別の表（作り直した表を模す — 中身の同一性は参照で区別される）。 */
const table = (id: string): GeometryProfile => ({ ...DEFAULT_GEOMETRY_PROFILE, id });

const SAVED_AT = "2026-10-01T03:04:05.000Z";

const saved = (profile: GeometryProfile, description = "Apple M2"): SavedProfile => ({
  savedAt: SAVED_AT,
  adapter: { vendor: "apple", architecture: "metal-3", description },
  checkout: { revision: "0123456789abcdef", dirty: false },
  profile,
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

    it("lists the saved table first with its save time and adapter description", () => {
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
        { saved: saved(table("gpu-saved"), ""), generated: [] },
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
      const [context, value] of [
        ["text that is not JSON", "{not json"],
        [
          "a value without the adapter",
          JSON.stringify({ ...saved(table("gpu-a")), adapter: null }),
        ],
        [
          "a savedAt that is not a date",
          JSON.stringify({ ...saved(table("gpu-a")), savedAt: "soon" }),
        ],
        // 素の JSON.stringify は Infinity を null に落とす — runtime の門が拒む
        ["a table that the runtime gate rejects", JSON.stringify(saved(table("gpu-a")))],
        [
          "a table without its fields",
          JSON.stringify({ ...saved(table("gpu-a")), profile: { id: "x" } }),
        ],
      ] as const
    ) {
      it(`removes the key and throws with the reason for ${context}`, () => {
        const storage = new FakeStorage();
        storage.items.set(LAST_GENERATED_KEY, value);
        assertThrows(() => readLastGenerated(storage), Error, LAST_GENERATED_KEY);
        assertEquals(storage.items.has(LAST_GENERATED_KEY), false);
      });
    }

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
});
