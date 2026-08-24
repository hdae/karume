/**
 * 母音検出の後処理 — グラフが出したロジット `[T20, 8]` を `.lab`（開始秒 終了秒 ラベル）へ畳む。
 *
 * 段は 4 つで、順番も含めて仕様の正本は Python 側
 * `training/src/vowel_detector/postprocess.py`（上流 `@hdae/vowel-detector`・MIT）:
 *
 * 1. log_softmax（ロジット → log 事後確率）
 * 2. **遷移ペナルティ付き Viterbi** — クラス切替に固定コストを課して短い断片を潰す
 * 3. **短区間マージ** — `MIN_DURATION_FRAMES` 未満の区間を、そこでの事後確率が高いほうの隣接
 *    クラスへ塗り替える（`pau` は知覚上重要な無音なので短くても残す）
 * 4. **cons 吸収** — 日本語のモーラ構造 C+V に基づき、子音区間を**後続**の母音/N へ吸収する
 *    （後続が母音/N でなければ先行区間へ）
 *
 * ## `src/text/unigram.ts` の Viterbi とは別物（共通化しない）
 *
 * あちらは「可変長の断片で文字列を覆う格子」の最短経路（辺の集合が入力ごとに変わる）で、
 * こちらは「固定 8 クラス × T フレームの格子」に一律の遷移ペナルティを置いた平滑化。状態も
 * 遷移も評価関数も共有しないので、名前が同じだけの 2 つを 1 本にしても引数で分岐するだけの
 * 抽象が増える。
 *
 * ## 定数は Python の module 定数がそのまま正本
 *
 * 配布形の `pipelineConfig` には載せない（載せると「学習時と違う平滑化を宣言できる」席が
 * できるが、そこを動かして良い根拠が無い）。値の一致は
 * `tests/vowel_detector_host_test.ts` が上流 `feature_config.json` の写しと突き合わせる。
 */

/** リップシンクの 8 クラス。**並びが id**（グラフの出力列と 1:1）。 */
export const LIPSYNC_CLASSES = ["a", "i", "u", "e", "o", "N", "pau", "cons"] as const;

/** 1 フレームの長さ（秒）。グラフ出力は 20ms グリッド（入力 10ms × stride 2）。 */
export const FRAME_SEC = 0.02;
/** クラス切替のコスト（log 確率スケール — 大きいほど平滑）。 */
const SWITCH_PENALTY = 4.0;
/** これ未満の区間は隣接へ吸収する（2 = 40ms。3 にすると実在する短母音まで消える）。 */
const MIN_DURATION_FRAMES = 2;

const CLASS_COUNT = LIPSYNC_CLASSES.length;
const classId = (label: typeof LIPSYNC_CLASSES[number]): number => LIPSYNC_CLASSES.indexOf(label);
const CONS_ID = classId("cons");
const PAU_ID = classId("pau");
/** cons を**後続へ**吸収してよい相手（母音と撥音）。 */
const VOWEL_OR_N: readonly number[] = (["a", "i", "u", "e", "o", "N"] as const).map(classId);

/** クラス列の連続区間 `[classId, start, end]`（start / end の単位は呼び出し側の倍率）。 */
type ClassRun = readonly [classId: number, start: number, end: number];

/**
 * 格子 `[frames, 8]` に非有限値が 1 つでもあれば座標を添えて落とす。
 *
 * MUST: 通してはならない。非有限値は例外にならず**書式として正当な `.lab`** に化ける —
 * NaN / `+Infinity` は log_softmax でその行の 8 要素すべてを NaN にし、`viterbiSmooth` の
 * 比較（`score[cls] > best` / `stays`）が軒並み偽になって backpointer がゼロ初期化のまま残る。
 * 結果は「発話全体が `a`」の 1 区間で、GPU の数値異常も壊れた資産もどこにも表面化しない。
 * 横断の不変条件「未対応・想定外は fail loudly（黙って近似しない）」の側に倒す。
 *
 * @param where エラー文言の接頭辞（関数名 + 何の格子か）。
 */
const assertFiniteGrid = (
  values: Float32Array | Float64Array,
  frames: number,
  where: string,
): void => {
  for (let frame = 0; frame < frames; frame += 1) {
    for (let cls = 0; cls < CLASS_COUNT; cls += 1) {
      const value = values[frame * CLASS_COUNT + cls];
      if (!Number.isFinite(value)) {
        throw new Error(
          `${where} frame ${frame} / class ${cls}（${LIPSYNC_CLASSES[cls]}）が非有限（${value}）`,
        );
      }
    }
  }
};

/** `.lab` の 1 行（秒）。 */
export type LabSegment = {
  readonly start: number;
  readonly end: number;
  readonly label: string;
};

/**
 * 遷移ペナルティ付き Viterbi でクラス列を平滑化する。
 *
 * 遷移は「留まる（コスト 0）」か「**その時点の最良クラスから**移る（コスト
 * `SWITCH_PENALTY`）」の 2 択しかない — 全対全の遷移行列を持たないので O(T·C) で回る。
 *
 * 門から名指しできるように出してある（漸化式の綴りが 1 箇所ずれても `.lab` は
 * 「それらしい別の区間割り」になるだけで、末端の文字列比較まで異常が見えない）。
 */
export const viterbiSmooth = (logProbabilities: Float64Array, frames: number): Int32Array => {
  if (logProbabilities.length !== frames * CLASS_COUNT) {
    throw new Error(
      `viterbiSmooth: log 事後確率が ${logProbabilities.length} 要素` +
        `（${frames}×${CLASS_COUNT} が要る）`,
    );
  }
  if (frames < 1) throw new Error(`viterbiSmooth: フレーム数 ${frames} が 1 未満`);
  // 公開関数なので単体でも叩ける（`logitsToSegments` 経由なら入口で既に検査済み）。
  assertFiniteGrid(logProbabilities, frames, "viterbiSmooth: log 事後確率");

  const score = new Float64Array(CLASS_COUNT);
  for (let cls = 0; cls < CLASS_COUNT; cls += 1) score[cls] = logProbabilities[cls];
  const backpointer = new Int32Array(frames * CLASS_COUNT);
  for (let frame = 1; frame < frames; frame += 1) {
    let best = Number.NEGATIVE_INFINITY;
    let argbest = 0;
    for (let cls = 0; cls < CLASS_COUNT; cls += 1) {
      if (score[cls] > best) {
        best = score[cls];
        argbest = cls;
      }
    }
    const switched = best - SWITCH_PENALTY;
    for (let cls = 0; cls < CLASS_COUNT; cls += 1) {
      const stays = score[cls] >= switched;
      backpointer[frame * CLASS_COUNT + cls] = stays ? cls : argbest;
      score[cls] = (stays ? score[cls] : switched) + logProbabilities[frame * CLASS_COUNT + cls];
    }
  }

  const path = new Int32Array(frames);
  let best = Number.NEGATIVE_INFINITY;
  for (let cls = 0; cls < CLASS_COUNT; cls += 1) {
    if (score[cls] > best) {
      best = score[cls];
      path[frames - 1] = cls;
    }
  }
  for (let frame = frames - 1; frame > 0; frame -= 1) {
    path[frame - 1] = backpointer[frame * CLASS_COUNT + path[frame]];
  }
  return path;
};

/** クラス列 → 連続区間（`frameSec` を掛けた尺度で返す）。 */
const toRuns = (path: Int32Array, frameSec: number): ClassRun[] => {
  const runs: ClassRun[] = [];
  let start = 0;
  for (let frame = 1; frame <= path.length; frame += 1) {
    if (frame === path.length || path[frame] !== path[start]) {
      runs.push([path[start], start * frameSec, frame * frameSec]);
      start = frame;
    }
  }
  return runs;
};

/**
 * `MIN_DURATION_FRAMES` 未満の区間を、その区間での事後確率が高いほうの隣接クラスへ塗り替える。
 *
 * 1 箇所塗るたびに走査をやり直す（塗り替えで run の構造そのものが変わるため）。収束するまで
 * 繰り返す。`pau` は短くても残す。
 */
const mergeShortRuns = (path: Int32Array, logProbabilities: Float64Array): Int32Array => {
  const merged = path.slice();
  let changed = true;
  while (changed) {
    changed = false;
    const runs = toRuns(merged, 1); // 倍率 1 = 区間の単位はフレーム
    for (let index = 0; index < runs.length; index += 1) {
      const [cls, start, end] = runs[index];
      if (end - start >= MIN_DURATION_FRAMES || cls === PAU_ID) continue;
      const candidates: number[] = [];
      if (index > 0) candidates.push(runs[index - 1][0]);
      if (index + 1 < runs.length) candidates.push(runs[index + 1][0]);
      if (candidates.length === 0) continue;
      let best = candidates[0];
      let bestScore = Number.NEGATIVE_INFINITY;
      for (const candidate of candidates) {
        let score = 0;
        for (let frame = start; frame < end; frame += 1) {
          score += logProbabilities[frame * CLASS_COUNT + candidate];
        }
        if (score > bestScore) {
          bestScore = score;
          best = candidate;
        }
      }
      merged.fill(best, start, end);
      changed = true;
      break;
    }
  }
  return merged;
};

/**
 * cons 区間を隣接区間へ吸収して 7 ラベル（a/i/u/e/o/N/pau）の列にする。
 *
 * 基本は**後続**へ吸収（モーラ構造 C+V なので、子音は次の母音の口形の一部として見える）。
 * 後続が母音/N でなければ先行区間を伸ばし、先頭でどちらも無い縮退ケース（発話全体が cons）は
 * `pau` に落とす。
 */
const absorbConsonants = (runs: readonly ClassRun[]): LabSegment[] => {
  const result: LabSegment[] = [];
  let pendingStart: number | undefined;
  for (let index = 0; index < runs.length; index += 1) {
    const [cls, start, end] = runs[index];
    if (cls === CONS_ID) {
      const next = index + 1 < runs.length ? runs[index + 1][0] : undefined;
      if (next !== undefined && VOWEL_OR_N.includes(next)) {
        // 後続区間の開始を前倒しする（連続する cons では最初の開始だけ覚える）。
        pendingStart ??= start;
      } else if (result.length > 0) {
        const last = result[result.length - 1];
        result[result.length - 1] = { start: last.start, end, label: last.label };
      } else {
        result.push({ start, end, label: "pau" });
      }
      continue;
    }
    const segmentStart = pendingStart ?? start;
    pendingStart = undefined;
    const label = LIPSYNC_CLASSES[cls];
    const last = result.at(-1);
    if (last !== undefined && last.label === label) {
      result[result.length - 1] = { start: last.start, end, label };
    } else {
      result.push({ start: segmentStart, end, label });
    }
  }
  return result;
};

/**
 * ロジット `[frames, 8]`（行優先）→ `.lab` セグメント列。
 *
 * log_softmax は最大値を引いてから通す（Python 側は素の `logsumexp` だが、`exp` の桁溢れを
 * 避けるこの形と log 事後確率としては同値 — 差は f64 の丸めぶんだけで、段 2 以降は
 * すべて**差**しか見ない）。
 */
export const logitsToSegments = (logits: Float32Array, frames: number): LabSegment[] => {
  if (logits.length !== frames * CLASS_COUNT) {
    throw new Error(
      `logitsToSegments: ロジットが ${logits.length} 要素（${frames}×${CLASS_COUNT} が要る）`,
    );
  }
  if (frames < 1) throw new Error(`logitsToSegments: フレーム数 ${frames} が 1 未満`);
  assertFiniteGrid(logits, frames, "logitsToSegments: ロジット");

  const logProbabilities = new Float64Array(frames * CLASS_COUNT);
  for (let frame = 0; frame < frames; frame += 1) {
    const row = frame * CLASS_COUNT;
    let peak = Number.NEGATIVE_INFINITY;
    for (let cls = 0; cls < CLASS_COUNT; cls += 1) peak = Math.max(peak, logits[row + cls]);
    let sum = 0;
    for (let cls = 0; cls < CLASS_COUNT; cls += 1) sum += Math.exp(logits[row + cls] - peak);
    const logSum = Math.log(sum) + peak;
    for (let cls = 0; cls < CLASS_COUNT; cls += 1) {
      logProbabilities[row + cls] = logits[row + cls] - logSum;
    }
  }
  const path = mergeShortRuns(viterbiSmooth(logProbabilities, frames), logProbabilities);
  return absorbConsonants(toRuns(path, FRAME_SEC));
};

/**
 * `.lab` 本文（`開始 終了 ラベル` の行を秒 7 桁で並べ、各行に改行を付ける）。
 *
 * MUST: 7 桁固定（上流 `lab.py` の `f"{s.start:.7f}"`）。桁を変えると既存の `.lab` 消費側
 * （リップシンクのタイムライン）との突合が文字列比較で外れる。
 */
export const toLab = (segments: readonly LabSegment[]): string =>
  segments.map((segment) =>
    `${segment.start.toFixed(7)} ${segment.end.toFixed(7)} ${segment.label}\n`
  ).join("");
