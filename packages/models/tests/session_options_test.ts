// manifest の `session`（3 キー固定の manifest 所有語彙）→ runtime `SessionOptions` の写像。
// ADR 0038 §3 の綴りの契約そのもので、抜けは**沈黙劣化**（未知キーは runtime が黙って無視する）
// になる。7 家族が同じ 1 本を使うので、門もここ 1 本に集約する（元は anima / sbv2 の
// pipeline テストへ 2 本に割れていて、残り 5 家族は写像を直接叩く門を持っていなかった）。

import { assertEquals } from "@std/assert";
import type { SessionSpec } from "@karume/hub";
import { toSessionOptions } from "../src/session/options.ts";

Deno.test("toSessionOptions: 3 キーを 1 つずつ写す（未指定は欄ごと作らない）", () => {
  assertEquals(toSessionOptions({}), {});
  assertEquals(toSessionOptions({ linearCompute: "a8" }), { linearCompute: "a8" });
  assertEquals(toSessionOptions({ attentionCompute: "f16" }), { attentionCompute: "f16" });
  assertEquals(toSessionOptions({ attentionScoreStorage: "f16" }), {
    attentionScoreStorage: "f16",
  });
  // 配布物の既定 quant（f16+dit8-a8-attn8-s16）の 3 キーが全て通ること。1 キーでも落とすと
  // 「名前だけ s16」の沈黙劣化になる。
  assertEquals(
    toSessionOptions({
      linearCompute: "a8",
      attentionCompute: "a8",
      attentionScoreStorage: "f16",
    }),
    {
      linearCompute: "a8",
      attentionCompute: "a8",
      attentionScoreStorage: "f16",
    },
  );
});

Deno.test("toSessionOptions: manifest 側に無いノブ（submitPolicy）は写さない", () => {
  // `SessionOptions.submitPolicy` は TDR 予算 = **ホスト政策**なので配布者に書かせない
  // （ADR 0038 §3 の理由 ③）。スプレッド素通しに書き換えるとここが素通りしうる。
  const mapped = toSessionOptions({ linearCompute: "a8" }) as Record<string, unknown>;
  assertEquals(Object.hasOwn(mapped, "submitPolicy"), false);
  assertEquals(Object.keys(mapped), ["linearCompute"]);
});

Deno.test("toSessionOptions: SessionSpec の全キーを写す（キー追加の取り残しを検出）", () => {
  // 写像の網羅は `src/session/options.ts` の `WRITERS`（`Required<SessionSpec>` の網羅表）が
  // 型で固定しており、キーが増えれば**コンパイルエラー**になる。ここはその型門が生きている
  // ことを実行時からも見る対（型を緩めた改変は型検査を通ってしまうため）—
  // 全キーを埋めた spec を渡し、出てくるキー集合が入力と一致することを確かめる。
  const full: Required<SessionSpec> = {
    linearCompute: "f16",
    attentionCompute: "f16",
    attentionScoreStorage: "f16",
  };
  const mapped = toSessionOptions(full) as Record<string, unknown>;
  assertEquals(Object.keys(mapped).sort(), Object.keys(full).sort());
});
