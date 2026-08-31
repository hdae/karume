/**
 * 取得元契約の分岐点（`sourceForRef`）の単体テスト。
 *
 * 越境の振り分けは面ごとに書くと片方だけ素通しする形の取り違えを生むため 1 箇所へ集約してある。
 * 公開面からは「越境参照が混ざったときのバイト列」でしか観測できないので、分岐そのものを
 * ここで固定する（将来ここへ「明示 mapping → 明示 fallback → fail loudly」の解決順が入る）。
 */

import { assertEquals, assertStrictEquals } from "@std/assert";
import type { FileRef } from "../src/manifest.ts";
import { type PinnedSource, sourceForRef } from "../src/source.ts";

const SHA = "b".repeat(40);

/** `originFor` の呼び出しだけを記録する取得元（読みの面は呼ばれない）。 */
const recordingSource = (): { source: PinnedSource; calls: [string, string][] } => {
  const calls: [string, string][] = [];
  const source: PinnedSource = {
    readManifest: () => Promise.reject(new Error("呼ばれてはいけない")),
    readFile: () => Promise.reject(new Error("呼ばれてはいけない")),
    prefetchFile: () => Promise.reject(new Error("呼ばれてはいけない")),
    originFor: (repo, revision) => {
      calls.push([repo, revision]);
      // 越境先は別の取得元（同一性が見えるように別オブジェクトを返す）。
      return { ...source, originFor: source.originFor };
    },
  };
  return { source, calls };
};

const ref = (extra: Partial<FileRef> = {}): FileRef => ({
  path: "transformer/model.safetensors",
  size: 4,
  sha256: "a".repeat(64),
  ...extra,
});

Deno.test("sourceForRef: 自リポ参照はセッションの取得元をそのまま使う", () => {
  const { source, calls } = recordingSource();
  assertStrictEquals(sourceForRef(source, ref()), source);
  assertEquals(calls, [], "越境でないのに originFor を引いている");
});

Deno.test("sourceForRef: repo と revision が揃った参照だけが宣言された取得元へ回る", () => {
  const { source, calls } = recordingSource();
  const crossed = sourceForRef(source, ref({ repo: "someone/other", revision: SHA }));
  assertEquals(calls, [["someone/other", SHA]]);
  assertStrictEquals(crossed === source, false, "越境先がセッションの取得元と同一になっている");
});

Deno.test("sourceForRef: 片方だけの宣言では越境しない（対でのみ意味を持つ）", async (t) => {
  // parse が対を強制するので manifest 経由では現れないが、分岐が片方だけを見ていると
  // 「越境先の revision を自リポへ効かせる」等の取り違えが黙って成立する。
  await t.step("repo だけ", () => {
    const { source, calls } = recordingSource();
    assertStrictEquals(sourceForRef(source, ref({ repo: "someone/other" })), source);
    assertEquals(calls, []);
  });
  await t.step("revision だけ", () => {
    const { source, calls } = recordingSource();
    assertStrictEquals(sourceForRef(source, ref({ revision: SHA })), source);
    assertEquals(calls, []);
  });
});
