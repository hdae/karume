/**
 * シナリオ束縛: 記号次元（gemma4 の `M` / `C`、anima の `S` など）に**どの実行の値**を入れて
 * census を採るかの指定。
 *
 * 静的 census は IR を読むだけなので、記号は記号のまま残る。数値 shape が要るところ
 * （`linear` の M×N×K、strided 実体化の要素数）は束縛が無いと 1 つも埋まらないので、
 * ここで与える。MUST: 未束縛のまま黙って記号を出力しない（census 加重が「実行 1 回の
 * dispatch 数」を意味しなくなる）。
 *
 * キーは `SYM`（家族全体）か `<component>.SYM`（そのコンポーネントだけ）。同じ記号名が
 * コンポーネントごとに別の意味を持つ資産があるので後者が要る — irodori の `T` は backbone で
 * トークン数・codec_encoder でフレーム数、という具合に割れている。
 */

/** 1 シナリオ（= census.jsonl の `scenario` 欄 1 値ぶん）。 */
export type Scenario = {
  readonly name: string;
  /** キーは `SYM` または `<component>.SYM`。 */
  readonly bindings: Readonly<Record<string, number>>;
  /** `default`（下表）か `cli`（`--scenario`）。census 行に残す。 */
  readonly source: "default" | "cli";
  /** 既定値の出どころ（summary.json に 1 度だけ載る）。 */
  readonly provenance: string;
};

/**
 * 家族ごとの既定シナリオ。
 *
 * MUST: 値の出どころを `provenance` に書く。代表値は「その家族を実際に走らせたときの形」で
 * あって普遍の定数ではないので、出どころの無い数字は次の読み手が更新できない。
 */
const DEFAULT_SCENARIOS: Readonly<Record<string, readonly Scenario[]>> = {
  gemma4: [
    {
      name: "decode",
      bindings: { M: 1, C: 4096 },
      source: "default",
      provenance: "decode 1 トークン（M=1）。C は full attention スロットの容量記号で、" +
        "sliding 側は配布形が 512 で焼き込み済み（ADR 0066）",
    },
    {
      name: "prefill",
      bindings: { M: 768, C: 4096 },
      source: "default",
      provenance: "prefill の物理 chunk 行数 M=768（可変 capacity 波の掃引で使った刻み）",
    },
  ],
  anima: [
    {
      name: "1024px",
      bindings: { T: 64, Tsrc: 64, Ttgt: 512, S: 4096 },
      source: "default",
      provenance: "1024px 生成（latent 64×64 = パッチトークン S=4096）。T / Tsrc / Ttgt は " +
        "packages/runtime/tests/assets_fusion_counts_test.ts が実資産へ与えている代表値",
    },
  ],
  irodori: [
    {
      name: "representative",
      bindings: { T: 256, S: 750, "codec_encoder.T": 750 },
      source: "default",
      provenance:
        "テキスト 256 トークン / latent 750 フレーム（assets_fusion_counts_test.ts の代表値と、" +
        "dit の参照 IO `io.dit-cond-max` の x_t [1,750,32] に一致）。" +
        "codec_encoder の T だけはトークン数ではなくフレーム数なので個別に束縛する",
    },
  ],
  sbv2: [
    {
      name: "p203",
      bindings: { "text_encoder.T": 35, "front.P": 203, "voice.T": 203 },
      source: "default",
      provenance: "各コンポーネントの参照 IO コーパスの最長ケース（front / voice = " +
        "outputs/series/sbv2-*/{front,voice}/io.p203.safetensors、text_encoder = " +
        "outputs/series/deberta-*/sbv2-22layer/io.case2.safetensors の input_ids [1,35]）",
    },
  ],
  siglip2: [
    {
      name: "native",
      bindings: {},
      source: "default",
      provenance: "記号次元なし（入力 pixel_values は宣言が数値 shape）",
    },
  ],
  birefnet: [
    {
      name: "native",
      bindings: {},
      source: "default",
      provenance: "記号次元なし（入力 pixel_values は宣言が数値 shape）",
    },
  ],
  "depth-anything": [
    {
      name: "native",
      bindings: {},
      source: "default",
      provenance: "記号次元なし（入力 pixel_values は宣言が数値 shape）",
    },
  ],
  "vowel-detector": [
    {
      name: "voiced",
      bindings: { T: 100 },
      source: "default",
      provenance:
        "参照 IO outputs/series/vowel-detector-*/io.voiced.safetensors の features [1,200,83]" +
        "（宣言は [1,2T,83] なので T=100）",
    },
  ],
};

/** 家族の既定シナリオ（無ければ fail loudly — 黙って空束縛で走らせない）。 */
export const defaultScenarios = (family: string): readonly Scenario[] => {
  if (!Object.hasOwn(DEFAULT_SCENARIOS, family)) {
    throw new Error(
      `家族 '${family}' の既定シナリオが無い（既知: ${
        Object.keys(DEFAULT_SCENARIOS).join(" / ")
      }）— --scenario で明示する`,
    );
  }
  return DEFAULT_SCENARIOS[family];
};

/** `名前=記号:値,記号:値` を 1 シナリオへ。 */
export const parseScenario = (text: string): Scenario => {
  const split = text.indexOf("=");
  if (split <= 0) {
    throw new Error(`--scenario '${text}' は '<名前>=<記号>:<値>[,…]' の形で書く`);
  }
  const name = text.slice(0, split);
  const body = text.slice(split + 1);
  const bindings: Record<string, number> = {};
  for (const pair of body.split(",")) {
    if (pair === "") continue;
    const at = pair.indexOf(":");
    if (at <= 0) throw new Error(`--scenario '${text}': 束縛 '${pair}' が '<記号>:<値>' でない`);
    const key = pair.slice(0, at);
    const value = Number(pair.slice(at + 1));
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new Error(`--scenario '${text}': 束縛 '${pair}' の値が正の整数でない`);
    }
    bindings[key] = value;
  }
  return { name, bindings, source: "cli", provenance: `--scenario ${text}` };
};

/**
 * 1 コンポーネントへ効く束縛（`<component>.SYM` が同名の `SYM` を上書きする）。
 *
 * MUST: `<component>.` の付いたキーのうち、どのコンポーネントにも当たらない綴りは
 * fail loudly（打ち間違いが「その束縛は無かった」として静かに通ると、別の shape の census が
 * 出る）。判定は呼び手が全コンポーネント名を知っている位置で行う（{@link assertBindingKeys}）。
 */
export const bindingsFor = (
  scenario: Scenario,
  component: string,
): Readonly<Record<string, number>> => {
  const plain: Record<string, number> = {};
  const qualified: Record<string, number> = {};
  for (const [key, value] of Object.entries(scenario.bindings)) {
    const dot = key.indexOf(".");
    if (dot < 0) {
      plain[key] = value;
      continue;
    }
    if (key.slice(0, dot) === component) qualified[key.slice(dot + 1)] = value;
  }
  return { ...plain, ...qualified };
};

/** `<component>.SYM` の component が実在することを確かめる。 */
export const assertBindingKeys = (
  scenario: Scenario,
  components: readonly string[],
): void => {
  for (const key of Object.keys(scenario.bindings)) {
    const dot = key.indexOf(".");
    if (dot < 0) continue;
    const component = key.slice(0, dot);
    if (!components.includes(component)) {
      throw new Error(
        `シナリオ '${scenario.name}' の束縛 '${key}': component '${component}' がこの資産に無い` +
          `（既知: ${components.join(" / ")}）`,
      );
    }
  }
};
