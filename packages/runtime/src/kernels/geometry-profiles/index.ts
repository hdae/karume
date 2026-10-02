/**
 * 埋め込みの幾何プロファイルの一覧（DECIDED: ADR 0115 — 選択は src/kernels/geometry-profile.ts の
 * `selectGeometryProfile`）。
 *
 * MUST: 既定プロファイル（`DEFAULT_GEOMETRY_PROFILE`）は**含めない** — 既定はどれにも当たらない
 * adapter の受け皿で、一覧に入れると `match` 空の表が 2 本になる（選択が落とす）。
 * MUST: 並べるのは掃引から生成した `./<id>.ts` の定数だけ（副作用なし・配列リテラル 1 本）。
 * 並び順は選択に効かない（同じ順位に 2 本当たる一覧は選択が落とす）。
 *
 * 一覧は公開面（mod.ts）から出るので、配列も各表も深く凍結した**複製**を置く — アプリが一覧の表を
 * 書き換えると、同じ realm の全 Session の自動選択が黙って別の幾何になる。凍結は読み込み時の副作用では
 * ない: 各 `./<id>.ts` の定数には触れず、新しいオブジェクトを組んで凍結するだけの純粋な構築。
 */

import type { GemmGeometry } from "../gemm-geometry.ts";
import type { GeometryProfile } from "../geometry-profile.ts";
import type { I8a8Geometry } from "../i8a8-geometry.ts";
import { APPLE_METAL_3 } from "./apple-metal-3.ts";
import { NVIDIA_BLACKWELL } from "./nvidia-blackwell.ts";

/**
 * 表 1 本の深く凍結した複製（引数は書き換えない純関数）。
 *
 * NOTE: geometry-profile.ts に置かないのは、あちらがこの一覧を値として import する（選択の既定引数）
 * 循環のため — この module が先に評価される import 順では、あちらの `const` がまだ初期化前で落ちる。
 * 幾何はどれも数だけの平たいオブジェクトなので、1 段の複製で深い。`provenance` は `adapter` の 1 段と
 * `userAgent`（文字列の配列）を入れ子に持つので、そこも複製して凍結する。
 */
const freezeGeometryProfile = (profile: GeometryProfile): GeometryProfile => {
  const geometry = <T extends GemmGeometry | I8a8Geometry>(value: T): T =>
    Object.freeze({ ...value });
  return Object.freeze({
    id: profile.id,
    ...(profile.match === undefined ? {} : { match: Object.freeze({ ...profile.match }) }),
    gemmRows: Object.freeze(
      profile.gemmRows.map((rule) =>
        Object.freeze({ maxRows: rule.maxRows, geometry: geometry(rule.geometry) })
      ),
    ),
    attention: Object.freeze({
      qk: geometry(profile.attention.qk),
      pv: geometry(profile.attention.pv),
    }),
    conv2d: Object.freeze({
      rows64: geometry(profile.conv2d.rows64),
      rows32: geometry(profile.conv2d.rows32),
    }),
    i8a8: Object.freeze({
      linear: geometry(profile.i8a8.linear),
      attentionQk: geometry(profile.i8a8.attentionQk),
      attentionPv: geometry(profile.i8a8.attentionPv),
    }),
    ...(profile.provenance === undefined ? {} : {
      provenance: Object.freeze({
        ...profile.provenance,
        ...(profile.provenance.userAgent === undefined
          ? {}
          : { userAgent: Object.freeze([...profile.provenance.userAgent]) }),
        adapter: Object.freeze({ ...profile.provenance.adapter }),
      }),
    }),
  });
};

export const BUILTIN_GEOMETRY_PROFILES: readonly GeometryProfile[] = Object.freeze(
  [
    APPLE_METAL_3,
    NVIDIA_BLACKWELL,
  ].map(freezeGeometryProfile),
);
