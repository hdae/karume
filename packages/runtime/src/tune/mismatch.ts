/**
 * 保存した表を当ててよいかの照合（DECIDED: ADR 0117 決定 5）。
 *
 * runtime は注入時に照合しない（ADR 0115 追記決定 6 — 別の機の表を当てて A/B する用途を残す）。照合は
 * 純関数で公開し、呼ぶかどうかと不一致のときの打つ手（既定で走る・再掃引する・利用者に知らせる）はアプリが
 * 決める（`sessionOptionsViolation` と同じ流儀）。
 */
import type { GeometryProfile } from "../kernels/geometry-profile.ts";
import { geometryProfileKernelsId, KernelsIdError, sweepCaseSetId } from "./fingerprint.ts";

const ADAPTER_FIELDS = ["vendor", "architecture", "device", "description"] as const;

/**
 * 表 `profile` をこの adapter と今の runtime に当ててよいか。最初の不一致の文言を返し、全て一致なら
 * `undefined`。投げない。
 *
 * 照合する物（全て文字列の完全一致・この順）:
 * 1. `provenance` があること（無い表 = 手書きの表 → 照合の材料が無い）。
 * 2. adapter の 4 欄（`vendor` / `architecture` / `device` / `description` — 空文字も値として比べる）。
 * 3. カーネルの指紋（今の runtime で表から導き直した値と `provenance.kernels`）。
 * 4. ケース集合の版（今の runtime の値と `provenance.caseSet`）。
 *
 * 照合しない物: `candidateSet`（quick+ の表で足りるかはアプリの判断）・`match`（注入は `match` を見ない）・
 * ブラウザやドライバの版（取れる情報が無い — 更新で黙って遅くなりうるので、再掃引は利用者の明示操作で）。
 *
 * 限界: フラグ無しの Chrome は `device` と `description` を空で返し、Apple M2 と M5 は同じ 4 欄になる。
 * その 2 機の間で表を持ち運んだことはこの関数では検出できない。
 *
 * 指紋の導出に GPU は要らないが、掃引の全ケースで codegen を回す（約 1 MB の WGSL）。表の幾何から掃引の
 * shape のカーネルを組めない（codegen の門に落ちる・dispatch 数が上限を超える）表は、その理由を返す。
 */
export const geometryProfileMismatch = (
  profile: GeometryProfile,
  // 照合する adapter の欄だけを要求する（`GPUAdapterInfo`・`GpuContext.adapterInfo` はそのまま渡せる）
  adapterInfo: Pick<GPUAdapterInfo, "vendor" | "architecture" | "device" | "description">,
): string | undefined => {
  const { provenance } = profile;
  if (provenance === undefined) {
    return `表 '${profile.id}' に provenance が無い（照合の材料が無い — 手書きの表か、掃引から作っていない表）`;
  }
  for (const field of ADAPTER_FIELDS) {
    if (provenance.adapter[field] !== adapterInfo[field]) {
      return `表 '${profile.id}' の adapter の ${field} が違う（表 ${
        JSON.stringify(provenance.adapter[field])
      } / この adapter ${JSON.stringify(adapterInfo[field])} — 別の機か別の adapter で作った表）`;
    }
  }
  let kernels: string;
  try {
    kernels = geometryProfileKernelsId(profile);
  } catch (cause) {
    // 表の幾何が今の codegen の門か dispatch 数の上限に落ちる — 当てると Session 構築で落ちる表なので、
    // 投げずに不一致として返す（他の例外は実装の誤りなのでそのまま伝える）
    if (cause instanceof KernelsIdError) {
      return `表 '${profile.id}' の幾何から掃引の shape のカーネルを組めない（${cause.message}）`;
    }
    throw cause;
  }
  if (kernels !== provenance.kernels) {
    return `表 '${profile.id}' のカーネルの指紋が違う（表 ${provenance.kernels} / 今の runtime ${kernels}` +
      " — runtime の更新で表か既定の幾何のカーネルが変わった）";
  }
  const caseSet = sweepCaseSetId();
  if (caseSet !== provenance.caseSet) {
    return `表 '${profile.id}' のケース集合の版が違う（表 ${provenance.caseSet} / 今の runtime ${caseSet}` +
      " — runtime の更新で掃引のケースか gemmRows の段の境界が変わった）";
  }
  return undefined;
};
