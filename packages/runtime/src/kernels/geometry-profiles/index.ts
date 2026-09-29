/**
 * 埋め込みの幾何プロファイルの一覧（DECIDED: ADR 0115 — 選択は src/kernels/geometry-profile.ts の
 * `selectGeometryProfile`）。
 *
 * MUST: 既定プロファイル（`DEFAULT_GEOMETRY_PROFILE`）は**含めない** — 既定はどれにも当たらない
 * adapter の受け皿で、一覧に入れると `match` 空の表が 2 本になる（選択が落とす）。
 * MUST: 並べるのは掃引から生成した `./<id>.ts` の定数だけ（副作用なし・配列リテラル 1 本）。
 * 並び順は選択に効かない（同じ順位に 2 本当たる一覧は選択が落とす）。
 */

import type { GeometryProfile } from "../geometry-profile.ts";
import { APPLE_METAL_3 } from "./apple-metal-3.ts";
import { NVIDIA_BLACKWELL } from "./nvidia-blackwell.ts";

export const BUILTIN_GEOMETRY_PROFILES: readonly GeometryProfile[] = [
  APPLE_METAL_3,
  NVIDIA_BLACKWELL,
];
