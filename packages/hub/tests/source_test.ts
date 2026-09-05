/**
 * 取得元契約の分岐点（`sourceForRef`）の単体テスト。
 *
 * 越境の振り分けは面ごとに書くと片方だけ素通しする形の取り違えを生むため 1 箇所へ集約してある。
 * 公開面からは「越境参照が混ざったときのバイト列」でしか観測できないので、分岐そのものを
 * ここで固定する（将来ここへ「明示 mapping → 明示 fallback → fail loudly」の解決順が入る）。
 */

import { assertEquals, assertStrictEquals } from "@std/assert";
import { type FileRef, fileRefKey } from "../src/manifest.ts";
import { type PinnedSource, sourceForRef } from "../src/source.ts";

const SHA = "b".repeat(40);

/** `originFor` の呼び出しだけを記録する取得元（読みの面は呼ばれない）。 */
const recordingSource = (): { source: PinnedSource; calls: [string, string][] } => {
  const calls: [string, string][] = [];
  const source: PinnedSource = {
    origin: { label: "repo someone/anima @ " + SHA, integrity: "network" },
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

// 一意化キーは取得元の分岐と**同じ述語**（`crossRefOf`）で越境を見なければならない。片方だけの
// 宣言をキー側だけが越境として数えると、同じ 1 本のファイルが 2 本の別エントリになり（全量面は
// 同じ URL を 2 回取得し、進捗の total も二重に積む）、しかもバイト列は正しいので黙って通る。

Deno.test("fileRefKey: 対が揃った越境参照だけが宣言座標つきのキーになる", () => {
  assertEquals(
    fileRefKey(ref({ repo: "someone/other", revision: SHA })),
    `someone/other@${SHA}/transformer/model.safetensors`,
  );
});

Deno.test("fileRefKey: 片方だけの宣言は自リポの path キーへ畳まれる（sourceForRef と同じ判定）", async (t) => {
  await t.step("repo だけ", () => {
    assertEquals(fileRefKey(ref({ repo: "someone/other" })), ref().path);
  });
  await t.step("revision だけ", () => {
    assertEquals(fileRefKey(ref({ revision: SHA })), ref().path);
  });
});
