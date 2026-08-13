/**
 * 評価用サンプル画像 4 枚を **再現可能な形で**焼き直す台本（SigLIP2 の実画像 e2e が読む）。
 *
 *     deno task demo:eval-images --source models/karume-anima-turbo
 *
 * 中身は `examples/anima/main.ts` を seed とプロンプトを変えて 4 回呼ぶだけ。**プロンプト /
 * seed / 解像度の正本はこのファイル**で、生成物は `outputs/demo/` 直下（`rm -rf` で安全に
 * 消せる席 — docs/assets-layout.md）。
 *
 * MUST: `--source` は必須にする（既定を置かない）。Anima の配布形は untracked のローカル資産で
 * 置き場が環境ごとに違い、`main.ts` は `karume.json` を持たないパスを **HF リポジトリ名**と
 * 読んで取得しに行くので、既定を書くと打ち間違いが「知らないリポジトリの取得」に化ける。
 *
 * MUST: quant / steps は渡さない。ファイル名がその 2 つを綴る（{@link outputName}）ので、
 * 渡した瞬間に別名で焼かれ、golden（`io.photo-*.safetensors`）とは無関係の画像が
 * `outputs/demo/` に増えるだけになる。
 *
 * 画像を焼き直したら **golden も採り直す**（`tools/exporter/export_siglip2.py`）。採り直しを
 * 忘れた場合は e2e が落ちる（`io.photo-*.safetensors` に焼いた元画像の sha256 が入っており、
 * 突合するのは実 GPU 実行の前）。
 */

/** 4 枚に共通の解像度（`--resolution` の綴りそのもの = ファイル名にも入る）。 */
const RESOLUTION = "1024x1024";

/** 全ケース共通の品質タグ（プロンプトの前置き / 後置き）。 */
const QUALITY_PREFIX = "score_9, score_8_up, score_7_up";
const QUALITY_SUFFIX = "masterpiece, best quality";

/**
 * 焼く 4 枚。**人物 2 枚（42 / 45）と人物なし 2 枚（43 / 44）** の 2 群になっているのが要で、
 * SigLIP2 e2e の判別検査（人物どうしの cosine > 人物と風景の cosine）はこの群分けを見る。
 * ケース名（`photo-*`）は golden の綴りで、正本は `tools/exporter/export_siglip2.py` の
 * `REAL_CASES`（あちらがこのファイル名で PNG を読む）。
 */
const CASES: readonly { seed: number; case: string; subject: string; why: string }[] = [
  {
    seed: 42,
    case: "photo-portrait",
    subject: "1girl, solo, school uniform, cherry blossoms, outdoors, smile, upper body",
    why: "人物アップ + 桜並木（前景と背景が明確）",
  },
  {
    seed: 43,
    case: "photo-landscape",
    subject: "scenery, mountain, lake, forest, no humans, blue sky, clouds",
    why: "風景（山・湖・森・人物なし）",
  },
  {
    seed: 44,
    case: "photo-corridor",
    subject: "school hallway, perspective, vanishing point, indoor, windows, sunlight, no humans",
    why: "校舎の廊下（強い遠近・消失点）",
  },
  {
    seed: 45,
    case: "photo-street",
    subject: "1girl, solo, full body, standing, city street, buildings, detailed background",
    why: "全身人物 + 街並み（背景が複雑）",
  },
];

const prompt = (subject: string): string => `${QUALITY_PREFIX}, ${subject}, ${QUALITY_SUFFIX}`;

/**
 * `main.ts` が書くファイル名を写したもの（quant / steps 未指定 = `default` / `defaultstep`）。
 * 綴りの正本はあちらなので、1 枚ごとに存在を検査して**名前がずれたら落とす**。
 */
const outputName = (seed: number): string =>
  `anima-default-${RESOLUTION}-defaultstep-seed${seed}.png`;

const USAGE = "--source <Anima 配布形のパス|HF repo>";

/** `--key value` の対だけを受ける（`main.ts` と同じ規律 — 未知キーは落とす）。 */
const args = new Map<string, string>();
for (let at = 0; at < Deno.args.length; at += 2) {
  const [key, value] = [Deno.args[at], Deno.args[at + 1]];
  if (!key.startsWith("--") || value === undefined || value.startsWith("--")) {
    throw new Error(`引数 ${key} が --key value の対になっていない（使い方: ${USAGE}）`);
  }
  if (key !== "--source") throw new Error(`未知のオプション ${key}（使い方: ${USAGE}）`);
  args.set(key.slice(2), value);
}

const source = args.get("source");
if (source === undefined) throw new Error(`--source が無い（使い方: ${USAGE}）`);

const main = new URL("./main.ts", import.meta.url);

for (const entry of CASES) {
  const name = outputName(entry.seed);
  console.log(`[eval-images] ${entry.case} — ${entry.why}`);
  const { code } = await new Deno.Command(Deno.execPath(), {
    args: [
      "run",
      "-A",
      main.href,
      "--source",
      source,
      "--prompt",
      prompt(entry.subject),
      "--resolution",
      RESOLUTION,
      "--seed",
      String(entry.seed),
    ],
    stdout: "inherit",
    stderr: "inherit",
  }).output();
  if (code !== 0) throw new Error(`${entry.case}（seed ${entry.seed}）の生成が終了コード ${code}`);
  // MUST: 名前まで検査する。`main.ts` の綴りが変わっても生成自体は成功するので、ここで
  // 落とさないと「焼けているのに golden 側が読めない」形で後から気づくことになる。
  // 置き場が cwd 相対なのも `main.ts` の作法（リポジトリ直下から回す）。
  const path = `outputs/demo/${name}`;
  if (!(await Deno.stat(path).then((stat) => stat.isFile, () => false))) {
    throw new Error(`${path} が生成されていない（examples/anima/main.ts の命名が変わった）`);
  }
  console.log(`[eval-images] ${path}`);
}

console.log(
  `[eval-images] ${CASES.length} 枚。golden を採り直す: ` +
    "cd tools/exporter && uv run --group siglip2 python export_siglip2.py" +
    " --model-dir ../../inputs/siglip2/<系列名>",
);
