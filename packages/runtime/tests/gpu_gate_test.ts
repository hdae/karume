// ADR 0005「全ケース SKIP は明示 FAIL」の門番。
//
// 実 GPU テストは `ignore: !GPU_AVAILABLE` で個別に SKIP されるため、アダプタ無しの環境では
// 段 3（実 GPU 数値検証）が丸ごと消えたまま verify が緑になる。それは「検証していない」を
// 「検証済み」と誤読させる無音の見かけ成功なので、ここで 1 本だけ落とす。
//
// この門番自身は GPU がある環境では**通る（緑の 1 件として見える）**。ignore にすると
// 「門番が効いているのか、門番ごと消えているのか」が区別できなくなるため。

import { assert } from "@std/assert";
import {
  ALLOW_NO_GPU,
  ALLOW_NO_SHADER_F16,
  ALLOW_NO_TIMESTAMP_QUERY,
  GPU_AVAILABLE,
  SHADER_F16_AVAILABLE,
  TIMESTAMP_QUERY_AVAILABLE,
} from "./helpers/gpu.ts";

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

// アダプタはあるが feature が無い機でも、担当範囲だけで完全 SKIP が複数本発生し、痕跡は
// `helpers/gpu.ts` の console.warn 1 行だけになる（verify は緑）。アダプタ不在と同じ性格の
// 「無音の見かけ成功」なので、同じ形の opt-out つき門番を feature ごとに置く。
// アダプタ自体が無い機では上の門番が既に落ちているので、ここは二重に鳴らさない。
Deno.test({
  name: "GPU 門番: shader-f16 不在の SKIP は明示 FAIL（ADR 0005）",
  ignore: ALLOW_NO_SHADER_F16 || !GPU_AVAILABLE,
  fn: () => {
    assert(
      SHADER_F16_AVAILABLE,
      "アダプタが 'shader-f16' を列挙せず、f16 計算変種（ADR 0028）の実 GPU テストが全て " +
        "SKIP された。ADR 0005 によりこれは FAIL として扱う（f16 計算変種の書き出しガードを " +
        "1 度も検証していない状態になる）。この機で意図的に通すには " +
        "KARUME_ALLOW_NO_SHADER_F16=1 を設定すること。",
    );
  },
});

Deno.test({
  name: "GPU 門番: timestamp-query 不在の SKIP は明示 FAIL（ADR 0005）",
  ignore: ALLOW_NO_TIMESTAMP_QUERY || !GPU_AVAILABLE,
  fn: () => {
    assert(
      TIMESTAMP_QUERY_AVAILABLE,
      "アダプタが 'timestamp-query' を列挙せず、GPU 時間診断（ADR 0021）の実 GPU テストが " +
        "全て SKIP された。ADR 0005 によりこれは FAIL として扱う（内訳が undefined になり、" +
        "キー検査を持つケースも黙って空振りする）。この機で意図的に通すには " +
        "KARUME_ALLOW_NO_TIMESTAMP_QUERY=1 を設定すること。",
    );
  },
});
