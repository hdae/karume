/**
 * denoise ループ（段 7）の 2 経路 — forward ごとにホストへ降りる {@link runDitLoopOnHost} と、
 * ループ全体を 1 batch に束ねる GPU 常駐経路 {@link runDitLoopResident}。ループ 1 本ぶんの
 * 材料（{@link DitLoop}）を組むのは呼び手で、ここは受け取った材料を回すだけ。
 *
 * 10 段の説明（何をどの順で回すか）は {@link "./pipeline.ts"} 冒頭の doc が正本。**どちらの
 * 経路を回すかを決めるのもここではない** — 選択は `./pipeline.ts` の `generateLatent` にあり、
 * 計測が有効な device（ADR 0021 — batch を開けない）と `onEvent` の購読がホスト経路を選ぶ。
 *
 * MUST: 2 経路を汎用の step executor へ畳まない。出力のビット同一は「積む演算が 1 演算ずつ
 * 同型」でだけ成立しており（{@link runDitLoopResident} の MUST）、共通化して片方の並びが
 * 動くと WAV の sha256 門が割れる。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 */

import { disposeSteps } from "../session/dispose-steps.ts";
import {
  createSession,
  openModel,
  type ResidentTensor,
  type Session,
  type SessionOptions,
  type Tensor,
} from "@karume/runtime";

import type { IrodoriSegment } from "./host/mask.ts";
import type { SequencePlan } from "./host/round.ts";
import { type CfgVariant, combineCfg, eulerStep } from "./host/sampler.ts";
import {
  COMBINE_INPUTS,
  COMBINE_OUTPUT,
  combineGraph,
  EULER_INPUTS,
  EULER_OUTPUT,
  eulerGraph,
} from "./host/sampler-graph.ts";
import { timestepEmbedding } from "./host/t-embed.ts";
import type { GeneratedLatent, IrodoriLatentSnapshot, IrodoriState } from "./pipeline.ts";
import {
  asF32,
  type EmitEvent,
  f32,
  observer,
  outputAt,
  outputNameAt,
  withStageSession,
} from "./stage.ts";
import type { ModelComponent } from "../hub/components.ts";

/**
 * 途中潜在を返す口を作る（**lazy copy** — 呼ばれたときだけ写す）。
 *
 * 進捗だけを購読する消費側にコピー費用が一切かからず、内部の配列を渡さないので「次 step の
 * 入力を購読側に握られる」事故も構造的に起きない。
 *
 * MUST: 呼ばれた時点ではなく**作った時点**の配列を写す（引数で束縛する）。DiT ループの `x` は
 * step ごとに**新しい配列へ差し替わる**ので、この束縛がそのまま「その step の潜在」になる。
 * ループ変数を閉じ込めると、後から呼んだ購読側に別 step の潜在が返る。
 *
 * MUST: `data` だけでなく `shape` も写す。参照のまま返すと同じ step で 2 回呼んだ写しが同一の
 * 配列を共有し、購読側が 1 回目の `shape` を書き換えると 2 回目の写しが黙って別の形を名乗る。
 * 公開イベント面の契約（`IrodoriGenerateEvent.copyLatents` / `AnimaGenerateEvent.copyLatents`）
 * を家族間で 1 本にするための揃え（anima 側 `src/anima/pipeline.ts` の同名関数と同じ形）。
 *
 * NOTE: `export` は GPU 無しで独立性を縛るテストのため（`mod.ts` / サブパス面には出さない —
 * ADR 0008）。
 */
export const latentSnapshot = (
  latent: Float32Array<ArrayBuffer>,
  shape: readonly number[],
): () => IrodoriLatentSnapshot =>
(): IrodoriLatentSnapshot => ({ data: new Float32Array(latent), shape: [...shape] });

/**
 * latent 段の結果。`plan.targetSamples` は波形の切り出しに要るので**決めた場所から持ち回る**
 * （`GeneratedLatent` は公開の面なので混ぜない）。
 */
export type LatentStage = {
  readonly latent: GeneratedLatent;
  readonly plan: SequencePlan;
};

/** CFG の 1 変種（落とす区間・強さ・その区間だけ False にしたマスク）。 */
export type UncondVariant = {
  readonly segment: IrodoriSegment;
  readonly scale: number;
  readonly mask: Tensor;
};

/**
 * DiT ループ 1 本ぶんの材料（2 つの経路が**同じもの**を読む — ホストの計算はどちらでも同一）。
 *
 * 条件 3 本を値と Tensor の両方で持つのは、常駐経路が {@link ResidentTensor.write} に生の
 * 配列を要り、ホスト経路が `run` に Tensor を要るため。**同じ配列の別の見方**であって、
 * 独立に更新される複製ではない。
 */
export type DitLoop = {
  readonly frames: number;
  /** 初期ノイズ `[frames × latentDim]`。 */
  readonly initial: Float32Array<ArrayBuffer>;
  readonly schedule: Float32Array<ArrayBuffer>;
  readonly frequencies: Float32Array<ArrayBuffer>;
  /** 右 pad 済みの条件 3 本（グラフ入力名 → 値）。 */
  readonly conditionValues: Readonly<Record<string, Float32Array<ArrayBuffer>>>;
  readonly conditions: Readonly<Record<string, Tensor>>;
  readonly condMask: Tensor;
  readonly uncondVariants: readonly UncondVariant[];
};

/** ループの結果（最終潜在と `dit` を回した回数）。 */
type DitLoopResult = {
  readonly x: Float32Array<ArrayBuffer>;
  readonly forwards: number;
};

/**
 * forward ごとにホストへ降りるループ（`run` → readback → `combineCfg` + `eulerStep` →
 * 再アップロード）。
 *
 * **数値の正本**であり、計測が有効な device（`gpuTiming` — 常駐経路が使う batch を開けない）
 * と生成イベントの購読（`onEvent` — 1 batch の途中は観測できない）での唯一の経路でもある。
 */
export const runDitLoopOnHost = async (
  state: IrodoriState,
  emit: EmitEvent,
  loop: DitLoop,
): Promise<DitLoopResult> => {
  const { config } = state;
  let x = loop.initial;
  let forwards = 0;
  await withStageSession(
    state,
    emit,
    "dit",
    state.dit,
    state.ditSessionOptions,
    async (run) => {
      for (let step = 0; step < config.steps; step += 1) {
        const t = loop.schedule[step];
        const tNext = loop.schedule[step + 1];
        const tEmbed = f32(timestepEmbedding(t, loop.frequencies), [1, config.timestepEmbedDim]);
        const xTensor = f32(x, [1, loop.frames, config.latentDim]);
        const cond = asF32(
          outputAt(
            state.dit,
            await run({ x_t: xTensor, t_embed: tEmbed, mask: loop.condMask, ...loop.conditions }),
            0,
          ),
          "dit の速度場",
        );
        forwards += 1;
        const variants: CfgVariant[] = [];
        if (t >= config.cfgMinT && t <= config.cfgMaxT) {
          // MUST: 合成順は SEGMENT_ORDER（text → speaker → caption）— `combineCfg` の doc。
          for (const variant of loop.uncondVariants) {
            const outputs = await run({
              x_t: xTensor,
              t_embed: tEmbed,
              mask: variant.mask,
              ...loop.conditions,
            });
            forwards += 1;
            variants.push({
              scale: variant.scale,
              velocity: asF32(
                outputAt(state.dit, outputs, 0),
                `dit の速度場（uncond ${variant.segment}）`,
              ),
            });
          }
        }
        x = eulerStep(x, combineCfg(cond, variants), Math.fround(tNext - t));
        await emit({
          kind: "denoise-step",
          step: step + 1,
          steps: config.steps,
          t,
          copyLatents: latentSnapshot(x, [loop.frames, config.latentDim]),
        });
      }
    },
  );
  return { x, forwards };
};

/**
 * ループ全体を **1 batch** に束ねる GPU 常駐経路（H-5）。
 *
 * 潜在・速度場・CFG の途中結果・条件 3 本を全て {@link ResidentTensor} に置き、`dit` と
 * ホストで組んだ小グラフ 2 本（{@link combineGraph} / {@link eulerGraph}）を `enqueue` で
 * 積むだけにする。ホストへ降りるのは最後の 1 回（`x_t.read()`）だけで、フェンスは
 * `batch.finish()` の 1 本に集約される。
 *
 * MUST: 区間の中で `Session.run` を待たない（自己デッドロック — `beginBatch` の doc）。
 * MUST: 演算の積み方は {@link runDitLoopOnHost} と 1 演算ずつ同型（変種順・差の基準・
 * 引数順）。ずれると最終桁が動き、WAV sha256 門が割れる。強さ `scale` はこちらが GPU へ
 * 渡す前に f32 へ丸めるのに対しホスト経路は JS の f64 で乗算するが、`parseCfgScales` が
 * f32 厳密な値しか受理しないので、2 経路の出力一致は**配布形に依らず無条件で**成立する。
 * MUST: 常駐テンソルを返すのは Session を全て畳んだ**後**（焼き込み参照が残っていると
 * `dispose` が fail loudly になる）。
 */
export const runDitLoopResident = async (
  state: IrodoriState,
  loop: DitLoop,
): Promise<DitLoopResult> => {
  const { config, gpu } = state;
  const { frames } = loop;
  const observe = observer(state, "dit");
  const latentBytes = frames * config.latentDim * 4;
  // 記号次元は常駐入力から束縛できない（常駐テンソルは shape を持たない）ので毎 enqueue 明示する。
  // 名前は admission が確定させたもの（「記号は 1 本」の検査もそこにある — `admitIrodori`）。
  const bindings = { [state.ditSymbol]: frames };
  const velocity = outputNameAt(state.dit, 0);
  const residents: ResidentTensor[] = [];
  const sessions: Session[] = [];
  let failure: { readonly error: unknown } | undefined;
  try {
    const createResident = async (bytes: number, label: string): Promise<ResidentTensor> => {
      const tensor = await gpu.createResident(bytes, label);
      residents.push(tensor);
      return tensor;
    };
    const xT = await createResident(latentBytes, "irodori.x_t");
    const vCond = await createResident(latentBytes, "irodori.v_cond");
    const vVariant = await createResident(latentBytes, "irodori.v_variant");
    const accumulator = await createResident(latentBytes, "irodori.cfg_acc");
    xT.write(loop.initial);
    const conditions: Record<string, ResidentTensor> = {};
    for (const [name, values] of Object.entries(loop.conditionValues)) {
      const tensor = await createResident(values.byteLength, `irodori.${name}`);
      // ループの前に 1 度だけ投入する（毎 forward の 3.8MB writeBuffer が丸ごと消える）。
      tensor.write(values);
      conditions[name] = tensor;
    }
    const open = async (model: ModelComponent, options: SessionOptions): Promise<Session> => {
      const session = await model.createSession(gpu, options);
      sessions.push(session);
      return session;
    };
    // ホストが組んだ小グラフは配布形を通らない（バイト列がその場にある）ので、コンテナでは
    // なく旧 IR 面（`openModel`）から直に Session にする — 容器に包む理由が無い。
    const openHostGraph = async (bytes: ArrayBuffer): Promise<Session> => {
      const session = await createSession(gpu, openModel(bytes), {});
      sessions.push(session);
      return session;
    };
    const dit = await open(state.dit, state.ditSessionOptions);
    const combine = await openHostGraph(combineGraph(frames, config.latentDim));
    const euler = await openHostGraph(eulerGraph(frames, config.latentDim));
    // 強さは step に依らないので 1 度だけ作る。
    const scales = loop.uncondVariants.map((variant) => f32(Float32Array.of(variant.scale), [1]));

    let forwards = 0;
    const batch = await gpu.beginBatch();
    let batchFailure: { readonly error: unknown } | undefined;
    try {
      for (let step = 0; step < config.steps; step += 1) {
        const t = loop.schedule[step];
        const tEmbed = f32(timestepEmbedding(t, loop.frequencies), [1, config.timestepEmbedDim]);
        const forward = async (mask: Tensor, target: ResidentTensor): Promise<void> => {
          await dit.enqueue(
            { x_t: xT, t_embed: tEmbed, mask, ...conditions },
            { batch, bindings, copyOutputs: { [velocity]: target } },
          );
          forwards += 1;
          if (observe !== undefined) observe(dit.diagnostics());
        };
        await forward(loop.condMask, vCond);
        // 区間の最初の 1 forward だけフェンスを張って推定を裏付ける（P-2）。これが無いと区間の
        // 間ずっと実測 0 = チャンクは初期値 16 のままで、約 96,000 dispatch が 6,000 回の submit に
        // 割れる。代償はフェンス 1 本（≈11 ms）。
        if (step === 0) await batch.settle();
        const guided = t >= config.cfgMinT && t <= config.cfgMaxT &&
          loop.uncondVariants.length > 0;
        if (guided) {
          // MUST: 合成順は SEGMENT_ORDER（text → speaker → caption）— `combineCfg` の doc。
          for (let index = 0; index < loop.uncondVariants.length; index += 1) {
            await forward(loop.uncondVariants[index].mask, vVariant);
            // k = 0 の被加数は cond そのもの（正本 `combineCfg` の `let value = base`）。同じ
            // バッファを acc_in と cond の 2 口で読むだけなので WebGPU 上も合法。
            await combine.enqueue({
              [COMBINE_INPUTS.accumulator]: index === 0 ? vCond : accumulator,
              [COMBINE_INPUTS.cond]: vCond,
              [COMBINE_INPUTS.variant]: vVariant,
              [COMBINE_INPUTS.scale]: scales[index],
            }, { batch, copyOutputs: { [COMBINE_OUTPUT]: accumulator } });
          }
        }
        await euler.enqueue({
          [EULER_INPUTS.x]: xT,
          [EULER_INPUTS.velocity]: guided ? accumulator : vCond,
          [EULER_INPUTS.deltaT]: f32(
            Float32Array.of(Math.fround(loop.schedule[step + 1] - t)),
            [1],
          ),
        }, { batch, copyOutputs: { [EULER_OUTPUT]: xT } });
      }
    } catch (error) {
      batchFailure = { error };
      throw error;
    } finally {
      // MUST: 区間は必ず閉じる（開いたままだと device 単位のロックが返らず、以後の run が
      // 永久に待つ）。
      await disposeSteps([
        () => {
          if (batchFailure !== undefined) throw batchFailure.error;
        },
        () => batch.finish(),
      ]);
    }
    return { x: new Float32Array(await xT.read()), forwards };
  } catch (error) {
    failure = { error };
    throw error;
  } finally {
    // Session の焼き込み参照を先に外す。失敗しても全資源の解放を試み、元の故障も残す。
    await disposeSteps([
      () => {
        if (failure !== undefined) throw failure.error;
      },
      ...sessions.map((session) => () => session.dispose()),
      ...residents.map((tensor) => () => tensor.dispose()),
    ]);
  }
};
