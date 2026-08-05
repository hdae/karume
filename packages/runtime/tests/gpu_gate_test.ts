// ADR 0005「全ケース SKIP は明示 FAIL」の門番。
//
// 実 GPU テストは `ignore: !GPU_AVAILABLE` で個別に SKIP されるため、アダプタ無しの環境では
// 段 3（実 GPU 数値検証）が丸ごと消えたまま verify が緑になる。それは「検証していない」を
// 「検証済み」と誤読させる無音の見かけ成功なので、ここで 1 本だけ落とす。
//
// この門番自身は GPU がある環境では**通る（緑の 1 件として見える）**。ignore にすると
// 「門番が効いているのか、門番ごと消えているのか」が区別できなくなるため。

import { assert } from "@std/assert";
import { ALLOW_NO_GPU, GPU_AVAILABLE } from "./helpers/gpu.ts";

Deno.test({
  name: "GPU 門番: アダプタ無しの全 SKIP は明示 FAIL（ADR 0005）",
  // opt-out は「GPU 無しを承知で通す」意図表明のときだけ。既定では ignore しない。
  ignore: ALLOW_NO_GPU,
  fn: () => {
    assert(
      GPU_AVAILABLE,
      "GPUAdapter が取得できず、実 GPU テストが全て SKIP された。ADR 0005 によりこれは FAIL " +
        "として扱う（リリース判定は実 GPU 緑が必須）。GPU の無い環境で意図的に通すには " +
        "KARUME_ALLOW_NO_GPU=1 を設定すること。",
    );
  },
});
