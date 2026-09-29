/**
 * GEMM 幾何の**プロファイル**（shape × adapter の静的な表 — DECIDED: ADR 0115）。
 *
 * 幾何（どのスレッドがどの出力を担当するか）の最良点は GPU によって逆転する — Apple M2 で
 * 1.4〜1.7 倍速い幾何が Arc B570 では 0.72〜0.98 倍になる（docs/research/2026-09-27-k70-metal-per-op.md
 * §9）。共通の既定 1 本ではどちらかが負けるので、選択を「shape の純関数」から「shape ×
 * プロファイルの純関数」へ広げ、プロファイルは adapter の (vendor, architecture) から決定的な順で
 * 1 本選ぶ。
 *
 * MUST: プロファイルは**純データ**（副作用なし・実行時に外から読まない）。明示のチューニング
 * （tools/geometry-sweep の掃引）で測った結果を生成物としてソースに埋め込む
 * （`./geometry-profiles/`）。実行時オートチューン（実行中に測って選び直す）は禁止のまま
 * （ADR 0022 追記の MUST）— 同じ機・同じ shape なら毎回同じ幾何・同じキー・同じ WGSL になる。
 * MUST: 幾何が変えてよいのは担当割りだけで、数値契約（K 縮約順・積和の字面）は動かない
 * （src/kernels/gemm-geometry.ts / src/kernels/i8a8-geometry.ts）。幾何はパイプラインキーに載るので、
 * プロファイルが違えばキーも違い、device 寿命のパイプラインキャッシュで取り違えは起きない。
 *
 * 適用範囲（プロファイルを引く経路）は matmul / bmm / linear の行数バケット・融合 attention f32 の
 * ①QK / ③PV・conv2d の implicit GEMM・i8a8 の linear / 融合 attention ①QK / ③PV だけ。行数バケットの
 * 表は matmul / bmm / linear の 3 経路で共有する（同じ GEMM 骨格 — 幾何の意味・K 縮約順が同じで、
 * B 側の充填だけが違う）。掃引は 3 経路のケースを持ち、生成器はその全部で速い幾何だけを採る
 * （tools/geometry-sweep の profile）。states 形 attention（①ₜ / ③ₜ）・分解 attention の行ブロック・
 * conv1d の implicit GEMM は既定の選択のまま（掃引の対象外 — 測っていない経路を表で動かさない）。
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";
import {
  assertGemmGeometry,
  defaultGemmGeometry,
  GEMM_ROWS_BUCKETS,
  GEMM_TILE,
  type GemmGeometry,
} from "./gemm-geometry.ts";
import { GEMM_MTILE_SMALL, gemmMTileGeometry } from "./gemm.ts";
import { assertI8a8Geometry, defaultI8a8Geometry, type I8a8Geometry } from "./i8a8-geometry.ts";
import { BUILTIN_GEOMETRY_PROFILES } from "./geometry-profiles/index.ts";

/**
 * 行数バケットの規則 1 本。`rows <= maxRows` で最初に当たる規則を使う（{@link gemmRowsGeometry}）。
 * 規則列は `maxRows` の狭義昇順で、最後は `Number.POSITIVE_INFINITY`（どの行数も必ず当たる）。
 */
export type GemmRowsRule = { readonly maxRows: number; readonly geometry: GemmGeometry };

export type GeometryProfile = {
  /** プロファイルの名前（`"default"` / `"apple-metal-3"` など — 生成物のファイル名と一致）。 */
  readonly id: string;
  /**
   * どの adapter に当てるか（`GPUAdapterInfo` の `vendor` / `architecture` と文字列の完全一致）。
   * 両方 undefined は既定プロファイル（{@link DEFAULT_GEOMETRY_PROFILE}）だけの形。
   */
  readonly match: { readonly vendor?: string; readonly architecture?: string };
  /** linear / matmul / bmm（行数バケット）。 */
  readonly gemmRows: readonly GemmRowsRule[];
  /** 融合 attention f32 の ①QK / ③PV。S は両段の間で実体化されるので、段ごとに別の幾何でよい。 */
  readonly attention: { readonly qk: GemmGeometry; readonly pv: GemmGeometry };
  /**
   * conv2d の implicit GEMM。m タイルの**クラス**（`conv2dIgemmMTile` が返す 64 行 / 32 行）ごとの
   * 幾何で、クラスを決めるのは今までどおりその述語。実タイル辺の正本は幾何なので、32 行クラスに
   * tileM = 64 の幾何を割り当ててもよい（dispatch は幾何から導く）。
   */
  readonly conv2d: { readonly rows64: GemmGeometry; readonly rows32: GemmGeometry };
  readonly i8a8: {
    readonly linear: I8a8Geometry;
    readonly attentionQk: I8a8Geometry;
    readonly attentionPv: I8a8Geometry;
  };
  /** 掃引から生成したプロファイルだけが持つ出どころ（掃引 JSON の path と sha256・日付・adapter）。 */
  readonly provenance?: {
    readonly sweep: string;
    readonly sha256: string;
    readonly date: string;
    readonly adapter: string;
  };
};

/**
 * 既定プロファイル（どの生成物にも当たらない adapter が使う表）。
 *
 * MUST: 値は既存の選択関数を**参照して**組む（書き写さない — 写すと既定が 2 か所になり、片方だけ
 * 直したときに Session の経路と shape 純関数の経路が別の幾何を選ぶ）。この表が選ばれた機では、
 * キー・WGSL・dispatch がプロファイル導入前と 1 バイトも変わらない（codegen スナップショットと
 * tests/geometry_profile_test.ts が検出器）。
 */
export const DEFAULT_GEOMETRY_PROFILE: GeometryProfile = {
  id: "default",
  match: {},
  gemmRows: GEMM_ROWS_BUCKETS,
  attention: { qk: defaultGemmGeometry(), pv: defaultGemmGeometry() },
  conv2d: {
    rows64: gemmMTileGeometry(GEMM_TILE),
    rows32: gemmMTileGeometry(GEMM_MTILE_SMALL),
  },
  i8a8: {
    linear: defaultI8a8Geometry("linear"),
    attentionQk: defaultI8a8Geometry("attention_qk"),
    attentionPv: defaultI8a8Geometry("attention_pv"),
  },
};

/**
 * 行数バケットの規則列の門。**昇順でない列は後ろの規則が死に、最後が Infinity でない列は大きな
 * M で当たる規則が無くなる** — どちらも選択が黙って別物になる形なので fail loudly。
 */
const assertGemmRowsRules = (rules: readonly GemmRowsRule[], where: string): void => {
  if (rules.length === 0) {
    throw new CodegenError(`${where}: gemmRows が空（どの行数にも幾何が当たらない）`);
  }
  rules.forEach((rule, index) => {
    const last = index === rules.length - 1;
    if (last) {
      if (rule.maxRows !== Number.POSITIVE_INFINITY) {
        throw new CodegenError(
          `${where}: gemmRows の最後の maxRows は Infinity（${rule.maxRows} — それより大きい M に規則が無い）`,
        );
      }
    } else {
      assertU32Params(`${where}: gemmRows[${index}]`, { maxRows: rule.maxRows });
      const next = rules[index + 1].maxRows;
      if (!(rule.maxRows < next)) {
        throw new CodegenError(
          `${where}: gemmRows の maxRows が狭義昇順でない（[${index}] ${rule.maxRows} → [${
            index + 1
          }] ${next} — 後ろの規則に当たる行数が無くなる）`,
        );
      }
    }
    assertGemmGeometry(rule.geometry, `${where}: gemmRows[${index}]`);
  });
};

/**
 * プロファイル 1 本の門（match の形・規則列・全欄の幾何の整除条件）。
 *
 * 幾何は生成時（`gemmWgsl` / i8a8 の生成器）にも同じ門を通るが、ここで先に落とすのは、壊れた
 * プロファイルを**その op に当たるまで**気づけない形にしないため（Session 構築で 1 度だけ見る）。
 * NOTE: w4a8 の `groupSize % tileK` は資産の group 長に依るので、ここでは見られない（生成時の
 * `assertI8a8GroupGeometry` の担当）。
 */
export const assertGeometryProfile = (profile: GeometryProfile): void => {
  const where = `幾何プロファイル '${profile.id}'`;
  if (profile.id.length === 0) {
    throw new CodegenError("幾何プロファイル: id が空（診断で選ばれた表を名指せない）");
  }
  const { vendor, architecture } = profile.match;
  if (vendor === "" || architecture === "") {
    throw new CodegenError(
      `${where}: match の vendor / architecture は空文字にしない（未指定は省く）`,
    );
  }
  if (architecture !== undefined && vendor === undefined) {
    throw new CodegenError(`${where}: match の architecture は vendor と対で指定する`);
  }
  assertGemmRowsRules(profile.gemmRows, where);
  for (
    const [name, geometry] of [
      ["attention.qk", profile.attention.qk],
      ["attention.pv", profile.attention.pv],
      ["conv2d.rows64", profile.conv2d.rows64],
      ["conv2d.rows32", profile.conv2d.rows32],
    ] as const
  ) {
    assertGemmGeometry(geometry, `${where}: ${name}`);
  }
  for (
    const [name, geometry] of [
      ["i8a8.linear", profile.i8a8.linear],
      ["i8a8.attentionQk", profile.i8a8.attentionQk],
      ["i8a8.attentionPv", profile.i8a8.attentionPv],
    ] as const
  ) {
    assertI8a8Geometry(geometry, `${where}: ${name}`);
  }
};

/** 選択の照合に使う adapter の 2 欄（`GpuContext.adapterInfo` はそのまま渡せる）。 */
type AdapterIdentity = { readonly vendor: string; readonly architecture: string };

/**
 * adapter に当てるプロファイルを 1 本選ぶ（Session 構築で 1 度だけ呼ぶ）。
 *
 * 選択順（決定的）: ① `match` の vendor と architecture が両方一致 → ② architecture 未指定で
 * vendor だけ一致 → ③ {@link DEFAULT_GEOMETRY_PROFILE}。照合は文字列の完全一致（正規化も前方一致も
 * しない）。
 *
 * MUST: 同じ順位に 2 本当たりうる一覧は fail loudly — 一覧の並び順で黙って 1 本を選ぶと、一覧の
 * 編集だけで選択が変わる。**当たった adapter に限らず一覧そのもので落とす**（同じ `match` を持つ
 * 2 本は、その `match` の機でだけ衝突する — 他の機の CI では見えないまま残る）。
 * MUST: 類似度（GPU 名の近さ等）で最近傍を選ばない。環境キーごとの参照 sha の行（ADR 0106）と
 * 「その機がどの幾何で走ったか」の対応が崩れる。
 */
export const selectGeometryProfile = (
  adapter: AdapterIdentity,
  profiles: readonly GeometryProfile[] = BUILTIN_GEOMETRY_PROFILES,
): GeometryProfile => {
  const ids = new Set([DEFAULT_GEOMETRY_PROFILE.id]);
  const matches = new Set<string>();
  for (const profile of profiles) {
    assertGeometryProfile(profile);
    const { vendor, architecture } = profile.match;
    if (vendor === undefined) {
      throw new CodegenError(
        `幾何プロファイル '${profile.id}': match が空（全 adapter に当たる表は既定プロファイルだけ）`,
      );
    }
    if (ids.has(profile.id)) {
      throw new CodegenError(`幾何プロファイル '${profile.id}': id が重複（診断で表を名指せない）`);
    }
    ids.add(profile.id);
    // JSON の配列形は区切り文字の衝突が起きない（`a/b` + `c` と `a` + `b/c` を取り違えない）。
    const match = JSON.stringify([vendor, architecture ?? null]);
    if (matches.has(match)) {
      throw new CodegenError(
        `幾何プロファイル '${profile.id}': match（vendor=${vendor} / architecture=${
          architecture ?? "（未指定）"
        }）が他のプロファイルと同じ（同じ順位に 2 本当たる）`,
      );
    }
    matches.add(match);
  }
  const exact = profiles.find((profile) =>
    profile.match.vendor === adapter.vendor && profile.match.architecture !== undefined &&
    profile.match.architecture === adapter.architecture
  );
  if (exact !== undefined) return exact;
  const vendorOnly = profiles.find((profile) =>
    profile.match.vendor === adapter.vendor && profile.match.architecture === undefined
  );
  return vendorOnly ?? DEFAULT_GEOMETRY_PROFILE;
};

/**
 * 行数 M から幾何を引く（matmul / bmm / linear の 3 経路）。
 *
 * MUST: 導出相はキー・WGSL・dispatch の 3 つへ**この 1 回の返り値**を通す（片方だけ別の幾何に
 * なるとキャッシュに載った WGSL と dispatch 数が噛み合わず、出力タイルが例外なしに欠ける）。
 */
export const gemmRowsGeometry = (profile: GeometryProfile, rows: number): GemmGeometry => {
  assertU32Params("幾何の選択", { "行数 M": rows });
  assertGemmRowsRules(profile.gemmRows, `幾何プロファイル '${profile.id}'`);
  const rules = profile.gemmRows;
  // 規則列の門を通った列は最後が Infinity なので、u32 の行数は必ずどれかに当たる。
  return (rules.find((rule) => rows <= rule.maxRows) ?? rules[rules.length - 1]).geometry;
};

/**
 * conv2d の m タイルのクラス（`conv2dIgemmMTile` の返り値 64 / 32）からプロファイルの幾何を引く。
 *
 * MUST: クラスは 2 値だけ。それ以外の m タイルは述語と表の食い違い（片方だけ値を足した）なので
 * fail loudly — 近い方へ丸めると、測っていないクラスに別クラスの幾何が黙って当たる。
 */
export const conv2dProfileGeometry = (profile: GeometryProfile, mTile: number): GemmGeometry => {
  if (mTile === GEMM_TILE) return profile.conv2d.rows64;
  if (mTile === GEMM_MTILE_SMALL) return profile.conv2d.rows32;
  throw new CodegenError(
    `幾何プロファイル '${profile.id}': conv2d の m タイル ${mTile} に当たるクラスが無い（${GEMM_TILE} / ${GEMM_MTILE_SMALL} だけ）`,
  );
};
