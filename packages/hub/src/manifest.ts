/**
 * `karume.json`（配布 manifest v4 = `karume/4`）の parse と全構造検査 — ADR 0041 の正本実装。
 *
 * MUST: **旧版は読まない**。`format` が `karume/4` 以外なら unsupported format で落とす
 * （2 形パースを持たない — ADR 0041 §1）。v4 は quant エントリへ表示欄（`label` /
 * `description` — ADR 0075）と `requiredLimits`、ファイル参照へ越境参照（`repo` /
 * `revision` — ADR 0038 §7）の席を足した版で、`session` の計算ノブの値は `i8a8` → `a8`
 * へ改名した（ADR 0074 決定 3）。どれも optional だが、**未知キーを fail loudly で拒否する
 * パーサの性質上、席を足した manifest は旧クライアントから読めない**ので major を繰り上げる
 * （ADR 0075 決定 4）。
 * MUST: 手書き parse・Web 標準 API のみ・未対応と想定外は fail loudly（黙って正規化しない）。
 * MUST: manifest 由来のマップは `Object.hasOwn` 経由でのみ引き、合成はスプレッドのみ
 * （`Object.assign` 禁止 — CLAUDE.md 横断不変条件 / ADR 0038 §1）。
 *
 * v1 からの語彙の対応（ADR 0041 §3）: `presets` → `quants` / `defaultPreset` → `defaultQuant` /
 * variant → `dtype` / `components` → `weights`（dtype キー必須のテンソル容器）+ `assets`
 * （quant 選択に依存しない無条件ファイル）。v3 で dtype エントリが `{file}` → `{shards}` に
 * なった（ADR 0071）。
 */

import {
  type AvailableLabels,
  ManifestFormatError,
  ManifestPathError,
  ManifestReferenceError,
  NO_LABELS,
} from "./errors.ts";

/** manifest のファイル名（リポジトリ直下の固定名 — ADR 0041 §1）。 */
export const MANIFEST_FILENAME = "karume.json";

/** manifest 本体の上限（DoS 防波堤 — ADR 0041 §7。取得中にも同じ値で abort する）。 */
export const MAX_MANIFEST_BYTES = 1024 * 1024;

/** 規模上限（ADR 0041 §7）。 */
const MAX_MODELS = 32;
const MAX_WEIGHTS = 32;
const MAX_ASSETS = 32;
const MAX_QUANTS = 32;
const MAX_PIPELINE_CONFIG_BYTES = 256 * 1024;
/**
 * manifest 全域走査の深さ上限。実在の manifest は envelope → models → weights → dtype →
 * shards → 要素の 6 段前後（`pipelineConfig` の入れ子を足しても十数段）で足りる。一方
 * 深さ検査が無いと、1MiB に収まる深いネストが `assertNoForbiddenKeys` の再帰でスタックを
 * 食い潰し、素の `RangeError` として `HubError` 契約の外へ抜ける（Deno 実測: 素の walk は
 * 配列 2,410 段で `RangeError`）。実用の要求より十分上・実測の破綻点よりはるか下に置き、
 * 深すぎる manifest は `ManifestFormatError` として fail loudly させる。
 */
const MAX_MANIFEST_DEPTH = 64;
/** 1 ファイルの上限バイト数（ADR 0038 §2 から据え置き）。 */
const MAX_FILE_BYTES = 16 * 2 ** 30;
/**
 * 1 dtype エントリの shard 数の上限。実在の配布は数十本で足りる（16GiB / shard の上限と
 * 併せれば 1024 本は現実の配布規模のはるか上）一方、上限が無いと 1MiB の manifest に
 * 数千の shard 参照を詰めた入力がそのまま取得計画になる。
 */
const MAX_SHARDS = 1024;
/**
 * shard 1 本の上限バイト数（1GiB — ADR 0081 の読み手契約 2。席による例外は無い）。
 *
 * MUST: 読み手側にも張る。上限の門は書き手（exporter の `pack_shards`）と読み返し
 * （`karume verify`）にしか無く、規則を守っていない shard 列（手で組んだ / 別実装が書いた /
 * 外部ツールの出力）は焼く側が全て緑で通す。読み手は **RAM ピーク O(最大 shard)**
 * （ADR 0070 決定 2）を前提に組まれているので、超過 shard を黙って受けるとその前提が崩れ、
 * Chromium の単一 `ArrayBuffer` 上限や取得層のバイト予算に**ブラウザで初めて**ぶつかる。
 * parse 時が正位置 — 読み手契約はフォーマット契約であり、「DL 開始後に初めて分かる」を
 * 許さないのが manifest 検査の目的そのもの（{@link parseManifest}）。
 *
 * MUST: 掛けるのは `shards` **だけ**。上限は shard 分割の契約であって、`assets` / `extras`
 * （単一ファイルで配る付帯資産）はこの規則の外にある — 混同すると 1GiB 超の実在資産
 * （例: PLE sidecar）が読めなくなる。全 FileRef 共通の天井は {@link MAX_FILE_BYTES}。
 *
 * MUST: 綴りは Python 正本 `tools/exporter/src/karume/shards.py` の `SHARD_BYTE_LIMIT` と
 * 同値に保つ（hub は exporter に依存しないので写しになる）。判定も向こうと同じ**閉区間**
 * （ちょうど上限は合法・超過だけを落とす）。
 *
 * NOTE: 数えるバイトは両側で厳密には同じでない。Python が数えるのは safetensors の**データ節**
 * だけ（ヘッダ長を決めるには所属が要り、所属を決めるにはヘッダ長が要るという循環を避けるため）
 * で、manifest の `size` は**ヘッダ込みのファイル全体**。よってこの門は書き手の契約より
 * ヘッダぶんだけ厳しい。それでよい — 読み手が RAM に載せるのはファイル全体なので、上限の根拠
 * （ArrayBuffer 天井・バイト予算）に対して正しいのはこちらの数え方で、差は実測で weight shard
 * 約 27KB（テンソル 1 本あたり 100 バイト前後）・グラフ shard で数 MB 級と、1GiB に対する
 * 余裕のうちに収まる（実配布の最大 shard は 993,725,828 バイト = 上限の 92.5%）。
 */
const MAX_SHARD_BYTES = 2 ** 30;
/** hub が理解する `format` の major。未知 major は fail loudly（ADR 0041 §1）。 */
const FORMAT_MAJOR = 4;
/** 表示欄の長さ上限（ADR 0075 決定 1）。 */
const MAX_LABEL_CHARS = 64;
const MAX_DESCRIPTION_CHARS = 200;

const SHA256_RE = /^[0-9a-f]{64}$/;
const SEGMENT_RE = /^[A-Za-z0-9._-]+$/;
const FORMAT_RE = /^karume\/([1-9][0-9]*)$/;
const PIPELINE_RE = /^([A-Za-z0-9_-]+)\/([1-9][0-9]*)$/;
/**
 * 越境参照の `revision`（40 桁小文字 hex の commit SHA）。
 *
 * MUST: ブランチ・タグ・短縮 SHA は拒否する — 越境参照は「別リポの**不変**コンポーネントを
 * 指す」席（ADR 0038 §7）であって、可変 ref を許すと自リポの revision を SHA へ固定した意味
 * （manifest と重みが同一コミットから来ること）が参照先で崩れる。短縮形も曖昧なので不可。
 */
const COMMIT_SHA_RE = /^[0-9a-f]{40}$/;
/**
 * 越境参照の `repo` の 1 セグメント（`owner` / `name`）。path と同じ許可リスト文字に加えて
 * **先頭ドットを禁止**する（`.` / `..` そのものもここで落ちる）。
 *
 * MUST: 禁止列挙ではなく許可リストにする — repo の `/` は取得層で構造要素として扱われ
 * percent-encode されない（`hfResolveUrl`）ので、列挙の抜けがそのまま SHA ピン外への
 * traversal になる（ADR 0038 §2 の path 検査と同じ理由）。
 */
const REPO_SEGMENT_RE = /^[A-Za-z0-9_-][A-Za-z0-9._-]*$/;

/**
 * プロトタイプ汚染に使われるキー。JSON.parse は `__proto__` を**自前プロパティ**として
 * 置くため汚染そのものは起きないが、下流が素朴な代入で読み直す事故を構造的に断つため
 * manifest 全域で拒否する（ADR 0038 §1）。
 */
const FORBIDDEN_KEYS: readonly string[] = ["__proto__", "constructor", "prototype"];

const ENVELOPE_KEYS: readonly string[] = ["format", "generator", "defaultModel", "models"];
const MODEL_KEYS: readonly string[] = [
  "pipeline",
  "weights",
  "assets",
  "quants",
  "defaultQuant",
  "pipelineConfig",
];
const WEIGHT_DTYPE_KEYS: readonly string[] = ["shards", "extras"];
const FILE_REF_KEYS: readonly string[] = ["path", "size", "sha256", "repo", "revision"];
const QUANT_KEYS: readonly string[] = [
  "weights",
  "session",
  "gpuFeatures",
  "label",
  "description",
  "requiredLimits",
];
const GPU_FEATURE_KEYS: readonly string[] = ["shaderF16"];

/**
 * `session` の allowlist（キーも値も — ADR 0038 §3 から据え置き）。runtime 型の素通しではない。
 *
 * 値の `i8a8` は 0.5.0 で `a8` へ改名した（ADR 0074 決定 3）。このノブが決めているのは
 * **活性の扱いだけ**で、重みの格納形は資産ヘッダが決める — 格納形を値に綴ると manifest と
 * ヘッダの二重持ちになり、混成資産（i4 + i8）でどちらか一方しか加速できない嘘になる。
 */
const LINEAR_COMPUTE: readonly LinearCompute[] = ["f32", "a8", "f16"];
const ATTENTION_COMPUTE: readonly AttentionCompute[] = ["f32", "f16", "a8"];
const SCORE_STORAGE: readonly ScoreStorage[] = ["f32", "f16"];
const SESSION_KEYS: readonly string[] = [
  "linearCompute",
  "attentionCompute",
  "attentionScoreStorage",
];

export type LinearCompute = "f32" | "a8" | "f16";
export type AttentionCompute = "f32" | "f16" | "a8";
export type ScoreStorage = "f32" | "f16";

/**
 * quant が要求する device limit の名前（ADR 0038 §7 が据え置き席として列挙していた
 * `requiredLimits`）。
 *
 * MUST: 語彙は **manifest 所有**で runtime 型の素通しではない（`session` と同じ規律 —
 * ADR 0038 §3）。綴りは `@karume/runtime` の `REQUIRED_LIMIT_KEYS`（`acquireGpu` が
 * `requestDevice` へ渡す requiredLimits のキー）と 1 対 1 に保つ。hub は runtime に依存しない
 * ので写しになるが、**未知名は fail loudly で拒否する**ので綴りのずれが黙って無視されること
 * はない（配布済み manifest を runtime 内部の綴りへ釘付けしないための境界でもある）。
 */
const REQUIRED_LIMIT_NAMES = [
  "maxBufferSize",
  "maxStorageBufferBindingSize",
  "maxUniformBufferBindingSize",
  "maxStorageBuffersPerShaderStage",
  "maxUniformBuffersPerShaderStage",
  "maxComputeWorkgroupStorageSize",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupSizeX",
  "maxComputeWorkgroupSizeY",
  "maxComputeWorkgroupSizeZ",
  "maxComputeWorkgroupsPerDimension",
] as const;

export type RequiredLimitName = (typeof REQUIRED_LIMIT_NAMES)[number];

/**
 * quant が要求する limit の**最小値**の表（`limit 名 → 値`）。WebGPU の
 * `GPUDeviceDescriptor.requiredLimits` と同じ「名前 → 満たすべき最小値」の形なので、
 * 消費側の判定は `adapter.limits[name] >= value` の素の比較になる。
 *
 * 書かれた名前だけが要求で、書かれていない limit は「要求しない」（部分写像）。
 */
export type RequiredLimitsSpec = Readonly<Partial<Record<RequiredLimitName, number>>>;

/** ファイル参照の 3 点セット（ADR 0038 §2）。3 点全ての存在と形式が parse 時の必須検査。 */
export type FileRef = {
  readonly path: string;
  /** Hub 上の保存形 raw のバイト数。 */
  readonly size: number;
  /** 小文字 hex 64 桁。 */
  readonly sha256: string;
  /**
   * 越境参照の取得元リポ（`owner/name` — ADR 0038 §7）。省略時は manifest 自身のリポ。
   *
   * MUST: {@link revision} と**両方同時**にのみ現れる（片方だけは fail loudly）。
   */
  readonly repo?: string;
  /** 越境参照の commit SHA（40 桁小文字 hex 固定）。{@link repo} と対。 */
  readonly revision?: string;
};

/**
 * ファイル参照の同一性キー。越境参照（別リポの `repo` / `revision`）が入った以上、
 * **`path` だけでは 1 本のファイルを指さない** — 別リポの同名 path は別のバイト列であり、
 * path で畳むと 3 点セット一致検査が正しい manifest を誤って拒否し、取得層では別のファイルの
 * バイト列を返してしまう。自リポ参照のキーは `path` そのままなので、v3 までの挙動は変わらない。
 */
export const fileRefKey = (ref: FileRef): string =>
  ref.repo === undefined ? ref.path : `${ref.repo}@${ref.revision}/${ref.path}`;

/**
 * weights の 1 dtype ぶんのファイル群（`{shards, extras?}`）。
 *
 * MUST: `shards` は**順序付き**（1 要素以上・{@link MAX_SHARDS} 以下で、1 本ずつが
 * {@link MAX_SHARD_BYTES} 以下）。宣言順は保存され、
 * 先頭 = グラフ shard（`karume_ir` を持つ）・後続 = 重み shard という**意味**を持つ
 * （ADR 0070 決定 3）。hub はその意味を検査しない — safetensors を開かないのが hub の境界で、
 * 順序の意味は shard を消費する runtime 側の契約。hub が保証するのは「宣言順のまま渡す」ことだけ。
 *
 * shard の識別子は**配列位置 = id・`size` = その shard のバイト数**として導出する。manifest に
 * id 欄は設けない（位置から導ける値を独立に更新される欄へ写すと正本が 2 つになる）。
 *
 * NOTE: `extras`（rope_base 等）は v1 と同じ席 = dtype エントリの内側に置く（ADR 0041 §3 は
 * components を weights / assets へ割るだけで、extras の位置は動かしていない）。dtype ごとに
 * 別の付帯資産を持てることに意味がある（同一実体なら同じ path を書けば取得は 1 回に畳まれる）。
 */
export type WeightFiles = {
  readonly shards: readonly FileRef[];
  readonly extras: Readonly<Record<string, FileRef>>;
};

/**
 * weights の 1 エントリ = **dtype ラベル → ファイル群**。v1 の `{file}` / `{variants}` の
 * 2 形は消え、i8 単体のコンポーネントも `{ "i8": … }` と書く（ADR 0041 §3）。
 */
export type WeightEntry = Readonly<Record<string, WeightFiles>>;

/** manifest 所有の実行ノブ語彙（3 キー固定 — ADR 0038 §3）。 */
export type SessionSpec = {
  readonly linearCompute?: LinearCompute;
  readonly attentionCompute?: AttentionCompute;
  readonly attentionScoreStorage?: ScoreStorage;
};

/** device 生成前に要る GPU feature（`shaderF16` のみ — ADR 0038 §3）。 */
export type GpuFeaturesSpec = {
  readonly shaderF16?: boolean;
};

/** 名前付きの量子化・精度モード（v1 の Preset — ADR 0041 §3 で改名）。 */
export type Quant = {
  /** weights 名 → その dtype ラベルへの**完全写像**。 */
  readonly weights: Readonly<Record<string, string>>;
  readonly session: SessionSpec;
  readonly gpuFeatures?: GpuFeaturesSpec;
  /**
   * 選択 UI に出す短い表示名（英語・{@link MAX_LABEL_CHARS} 文字以内 — ADR 0075 決定 1）。
   * 席 id（`i8-a8-attn8-s16`）が機械の都合なのに対し、こちらが人の読む側。未設定なら
   * 呼び手が id をそのまま出す。
   */
  readonly label?: string;
  /** 1 行の説明（英語・{@link MAX_DESCRIPTION_CHARS} 文字以内 — ADR 0075 決定 1）。 */
  readonly description?: string;
  /**
   * この席が要求する device limit の最小値（ADR 0038 §7）。
   *
   * NOTE: hub は受理・検査・型面への露出までを持ち、読み手は `@karume/models` の家族
   * admission（重み shard を 1 バイトも取る前に GPU 側の limits と突き合わせる — ADR 0089
   * 決定 5）。突き合わせ相手は、共有 GPU を渡された経路なら `GpuContext.limits`、自前で
   * device を取る経路なら `readAdapterLimits()` のアダプタ実測値。
   */
  readonly requiredLimits?: RequiredLimitsSpec;
};

/** パイプライン実装の契約名 + major（`anima/1`）。未知 major の判定は models 側の責務。 */
export type PipelineId = {
  readonly name: string;
  readonly major: number;
};

/**
 * 1 モデルぶんの宣言。`pipeline` が**モデル単位**なので、ファミリーリポに別アーキが混ざっても
 * 壊れない（ADR 0041 §2）。
 */
export type ModelEntry = {
  readonly pipeline: PipelineId;
  /** dtype 別のテンソル容器。 */
  readonly weights: Readonly<Record<string, WeightEntry>>;
  /** quant 選択に依存しない無条件ファイル（tokenizer / symbols / style_vectors 等）。 */
  readonly assets: Readonly<Record<string, FileRef>>;
  readonly quants: Readonly<Record<string, Quant>>;
  readonly defaultQuant: string;
  /** パイプライン所有 — hub は素通し（スキーマ検証は `@karume/models` の各実装）。 */
  readonly pipelineConfig: Readonly<Record<string, unknown>>;
  /** このモデルで選べるもの（エラー提示にも使う）。 */
  readonly available: AvailableLabels;
};

export type Manifest = {
  /** `karume/<major>`。major は hub が検査済み。 */
  readonly format: string;
  /** 焼いたツールの版（障害報告の照合用・実行意味論なし）。 */
  readonly generator: string;
  /** モデル未指定時に使うモデル名（必須 — ADR 0041 §2）。 */
  readonly defaultModel: string;
  readonly models: Readonly<Record<string, ModelEntry>>;
  /** リポ全体で選べるもの（`quants` / `dtypes` はモデル文脈が無いので空）。 */
  readonly available: AvailableLabels;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** 検査失敗の生成器。`available` を全エラーへ確実に載せるため、parse 中は必ずこれを通す。 */
type Fail = {
  format: (message: string, cause?: unknown) => ManifestFormatError;
  reference: (message: string) => ManifestReferenceError;
  path: (message: string, path: string) => ManifestPathError;
};

const createFail = (available: AvailableLabels): Fail => ({
  format: (message, cause) => new ManifestFormatError(message, { available, cause }),
  reference: (message) => new ManifestReferenceError(message, { available }),
  path: (message, path) => new ManifestPathError(message, { available, path }),
});

/**
 * manifest 全域から禁止キーを一掃する（`pipelineConfig` の内側も含む）。素通し先の実装が
 * 素朴な代入で読み直しても事故らないことを、hub の入口で一度だけ保証する。
 *
 * MUST: 深さを自前で数えて上限で落とす（{@link MAX_MANIFEST_DEPTH}）。スタックが尽きるまで
 * 潜ると `RangeError` という `HubError` 契約外の型で抜ける — 想定外は型付きで fail loudly。
 * これが parse の最初の全域走査なので、この門が下流の走査（`JSON.stringify` 等）も守る。
 */
const assertNoForbiddenKeys = (value: unknown, where: string, depth: number): void => {
  if (depth > MAX_MANIFEST_DEPTH) {
    throw new ManifestFormatError(`${where}: 入れ子が上限 ${MAX_MANIFEST_DEPTH} 段を超えた`);
  }
  if (Array.isArray(value)) {
    value.forEach((entry, index) => assertNoForbiddenKeys(entry, `${where}[${index}]`, depth + 1));
    return;
  }
  if (!isRecord(value)) return;
  for (const key of Object.keys(value)) {
    if (FORBIDDEN_KEYS.includes(key)) {
      throw new ManifestFormatError(`${where}: 禁止キー '${key}' が現れた`);
    }
    assertNoForbiddenKeys(value[key], `${where}.${key}`, depth + 1);
  }
};

/**
 * 検査を通す前に「利用可能なもの」を最善努力で拾う。壊れた manifest でも
 * 「では何なら動くのか」をエラーに載せられるようにするためだけの走査で、絶対に throw しない。
 *
 * `model` を渡すとそのモデルの quant / dtype まで拾う（モデルを跨いだ混合はしない — 別モデルの
 * quant 名を勧めるのは誤誘導）。
 */
const surveyLabels = (root: Record<string, unknown>, model?: string): AvailableLabels => {
  const modelsRaw = root["models"];
  if (!isRecord(modelsRaw)) return NO_LABELS;
  const models = Object.keys(modelsRaw);
  if (model === undefined) return { models, quants: [], dtypes: {} };
  const entry = modelsRaw[model];
  if (!isRecord(entry)) return { models, quants: [], dtypes: {} };
  const quantsRaw = entry["quants"];
  const quants = isRecord(quantsRaw) ? Object.keys(quantsRaw) : [];
  const weightsRaw = entry["weights"];
  let dtypes: Readonly<Record<string, readonly string[]>> = {};
  if (isRecord(weightsRaw)) {
    for (const name of Object.keys(weightsRaw)) {
      const labels = weightsRaw[name];
      if (!isRecord(labels)) continue;
      dtypes = { ...dtypes, [name]: Object.keys(labels) };
    }
  }
  return { models, quants, dtypes };
};

const assertAllowedKeys = (
  fail: Fail,
  value: Record<string, unknown>,
  allowed: readonly string[],
  where: string,
): void => {
  for (const key of Object.keys(value)) {
    if (!allowed.includes(key)) {
      throw fail.reference(`${where}: 未知キー '${key}'（許可: ${allowed.join(" / ")}）`);
    }
  }
};

/**
 * path の**許可リスト**検査（ADR 0038 §2・v2 も据え置き = ADR 0041 §6）。取得層はセグメントを
 * percent-encode してもドットを透過するため、禁止列挙では抜けがそのまま SHA ピン外への
 * traversal になる。
 */
const assertPath = (fail: Fail, path: string, where: string): void => {
  for (const segment of path.split("/")) {
    if (segment === "") {
      throw fail.path(
        `${where}: path '${path}' に空セグメントがある（先頭 / 末尾 / 連続スラッシュ）`,
        path,
      );
    }
    if (segment.startsWith(".")) {
      throw fail.path(
        `${where}: path '${path}' に先頭ドットのセグメント '${segment}' がある`,
        path,
      );
    }
    if (!SEGMENT_RE.test(segment)) {
      throw fail.path(
        `${where}: path '${path}' のセグメント '${segment}' が許可文字 [A-Za-z0-9._-] に一致しない`,
        path,
      );
    }
  }
};

/**
 * 越境参照の `repo`（`owner/name`）を検査する。取得層が URL へ綴り込む生文字列なので、
 * {@link REPO_SEGMENT_RE} の許可リストを 2 セグメントちょうどに掛ける。
 */
const assertRepo = (fail: Fail, repo: string, where: string): void => {
  const segments = repo.split("/");
  if (segments.length !== 2 || !segments.every((segment) => REPO_SEGMENT_RE.test(segment))) {
    throw fail.format(
      `${where}.repo: 'owner/name' の 2 セグメントでない / 許可文字 [A-Za-z0-9._-]（先頭ドット禁止）` +
        `に一致しない: '${repo}'`,
    );
  }
};

/**
 * 越境参照（`repo` + `revision`）を読む。**両方同時にのみ合法**で、片方だけの宣言は
 * fail loudly — `repo` だけなら「どのコミットか」が可変 ref に落ちて不変性が崩れ、
 * `revision` だけなら自リポの解決済み SHA を黙って上書きする経路になる（どちらも
 * 「宣言の意図が読み取れないまま何かを取りに行く」形なので、推測で補わない）。
 */
const parseCrossRepo = (
  fail: Fail,
  raw: Record<string, unknown>,
  where: string,
): { readonly repo: string; readonly revision: string } | undefined => {
  const hasRepo = Object.hasOwn(raw, "repo");
  const hasRevision = Object.hasOwn(raw, "revision");
  if (!hasRepo && !hasRevision) return undefined;
  if (hasRepo !== hasRevision) {
    throw fail.format(
      `${where}: 越境参照は 'repo' と 'revision' を両方同時に書く` +
        `（期待 両方 / 実際 ${hasRepo ? "'repo' だけ" : "'revision' だけ"}）`,
    );
  }
  const repo = raw["repo"];
  const revision = raw["revision"];
  if (typeof repo !== "string") throw fail.format(`${where}.repo: 文字列でない`);
  if (typeof revision !== "string") throw fail.format(`${where}.revision: 文字列でない`);
  assertRepo(fail, repo, where);
  if (!COMMIT_SHA_RE.test(revision)) {
    throw fail.format(
      `${where}.revision: 40 桁小文字 hex の commit SHA でなければならない` +
        `（ブランチ・タグ・短縮形は不可）: '${revision}'`,
    );
  }
  return { repo, revision };
};

/**
 * 3 点セットを検査して読む。同一ファイルの重複参照は合法（モデル間の共有はこれだけで成立する —
 * ADR 0041 §5）だが `{size, sha256}` の完全一致を要求する（矛盾 manifest は self-heal を
 * 振動させ、正しいキャッシュを evict し続ける）。同一性の判定は {@link fileRefKey}
 * （越境参照は repo と revision まで含めて 1 本）。
 */
const parseFileRef = (
  fail: Fail,
  raw: unknown,
  where: string,
  seen: Map<string, FileRef>,
): FileRef => {
  if (!isRecord(raw)) throw fail.format(`${where}: ファイル参照がオブジェクトでない`);
  assertAllowedKeys(fail, raw, FILE_REF_KEYS, where);
  const path = raw["path"];
  const size = raw["size"];
  const sha256 = raw["sha256"];
  if (typeof path !== "string") throw fail.format(`${where}: 'path' が無い / 文字列でない`);
  if (typeof size !== "number") throw fail.format(`${where}: 'size' が無い / 数値でない`);
  if (typeof sha256 !== "string") throw fail.format(`${where}: 'sha256' が無い / 文字列でない`);
  if (!SHA256_RE.test(sha256)) {
    throw fail.format(`${where}: 'sha256' は小文字 hex 64 桁でなければならない: '${sha256}'`);
  }
  if (!Number.isSafeInteger(size) || size <= 0 || size > MAX_FILE_BYTES) {
    throw fail.format(
      `${where}: 'size' は 0 < size <= ${MAX_FILE_BYTES} の安全整数でなければならない: ${size}`,
    );
  }
  assertPath(fail, path, where);
  const crossRepo = parseCrossRepo(fail, raw, where);
  const ref: FileRef = { path, size, sha256, ...(crossRepo ?? {}) };
  const key = fileRefKey(ref);
  const previous = seen.get(key);
  if (previous !== undefined) {
    if (previous.size !== size || previous.sha256 !== sha256) {
      throw fail.reference(
        `${where}: 重複参照 '${key}' の {size, sha256} が食い違う ` +
          `({${previous.size}, ${previous.sha256}} と {${size}, ${sha256}})`,
      );
    }
    return previous;
  }
  seen.set(key, ref);
  return ref;
};

/**
 * shard 列を宣言順のまま読む。並べ替えも重複畳み込みもしない — 位置が shard の id なので、
 * 列を触った瞬間に識別子が壊れる（同一 path の 3 点セット一致だけは {@link parseFileRef} の
 * 表が全域で見る）。
 *
 * バイト上限（{@link MAX_SHARD_BYTES}）を掛けるのはここ — 1 本ずつ独立に見る検査で、席
 * （先頭 / 末尾）による例外は無い。{@link parseFileRef} 側へ下ろさないのは、あの関数を
 * `assets` / `extras` と共有しているため（上限は shard 分割の契約に限る）。
 */
const parseShards = (
  fail: Fail,
  raw: unknown,
  where: string,
  seen: Map<string, FileRef>,
): readonly FileRef[] => {
  if (!Array.isArray(raw)) throw fail.format(`${where}: 無い / 配列でない`);
  if (raw.length === 0) throw fail.format(`${where}: 空（shard が 1 つ以上要る）`);
  if (raw.length > MAX_SHARDS) {
    throw fail.format(`${where}: ${raw.length} 件が上限 ${MAX_SHARDS} を超えた`);
  }
  return raw.map((entry, index) => {
    const ref = parseFileRef(fail, entry, `${where}[${index}]`, seen);
    if (ref.size > MAX_SHARD_BYTES) {
      throw fail.format(
        `${where}[${index}]: shard '${ref.path}' が ${ref.size} バイトで` +
          `上限 ${MAX_SHARD_BYTES} を超えた（分割規則 — ADR 0081）`,
      );
    }
    return ref;
  });
};

const parseWeightFiles = (
  fail: Fail,
  raw: unknown,
  where: string,
  seen: Map<string, FileRef>,
): WeightFiles => {
  if (!isRecord(raw)) throw fail.format(`${where}: dtype エントリがオブジェクトでない`);
  assertAllowedKeys(fail, raw, WEIGHT_DTYPE_KEYS, where);
  const shards = parseShards(fail, raw["shards"], `${where}.shards`, seen);
  const extrasRaw = raw["extras"];
  if (extrasRaw === undefined) return { shards, extras: {} };
  if (!isRecord(extrasRaw)) throw fail.format(`${where}.extras: オブジェクトでない`);
  let extras: Readonly<Record<string, FileRef>> = {};
  for (const name of Object.keys(extrasRaw)) {
    const ref = parseFileRef(fail, extrasRaw[name], `${where}.extras.${name}`, seen);
    extras = { ...extras, [name]: ref };
  }
  return { shards, extras };
};

/** weights の 1 エントリ（dtype ラベル → ファイル群）。**dtype キーは 1 つ以上必須**。 */
const parseWeightEntry = (
  fail: Fail,
  raw: unknown,
  where: string,
  seen: Map<string, FileRef>,
): WeightEntry => {
  if (!isRecord(raw)) throw fail.format(`${where}: weights エントリがオブジェクトでない`);
  const labels = Object.keys(raw);
  if (labels.length === 0) {
    throw fail.format(`${where}: 空（dtype ラベルが 1 つ以上要る — dtype キーは必須）`);
  }
  let entry: WeightEntry = {};
  for (const label of labels) {
    entry = { ...entry, [label]: parseWeightFiles(fail, raw[label], `${where}.${label}`, seen) };
  }
  return entry;
};

const readEnum = <T extends string>(
  fail: Fail,
  raw: Record<string, unknown>,
  key: string,
  values: readonly T[],
  where: string,
): T | undefined => {
  if (!Object.hasOwn(raw, key)) return undefined;
  const value = raw[key];
  if (typeof value !== "string") throw fail.format(`${where}.${key}: 文字列でない`);
  const found = values.find((candidate) => candidate === value);
  if (found === undefined) {
    throw fail.reference(`${where}.${key}: 未知の値 '${value}'（許可: ${values.join(" / ")}）`);
  }
  return found;
};

const parseSession = (fail: Fail, raw: unknown, where: string): SessionSpec => {
  if (raw === undefined) return {};
  if (!isRecord(raw)) throw fail.format(`${where}.session: オブジェクトでない`);
  assertAllowedKeys(fail, raw, SESSION_KEYS, `${where}.session`);
  const at = `${where}.session`;
  const linearCompute = readEnum(fail, raw, "linearCompute", LINEAR_COMPUTE, at);
  const attentionCompute = readEnum(fail, raw, "attentionCompute", ATTENTION_COMPUTE, at);
  const attentionScoreStorage = readEnum(fail, raw, "attentionScoreStorage", SCORE_STORAGE, at);
  return {
    ...(linearCompute === undefined ? {} : { linearCompute }),
    ...(attentionCompute === undefined ? {} : { attentionCompute }),
    ...(attentionScoreStorage === undefined ? {} : { attentionScoreStorage }),
  };
};

const parseGpuFeatures = (
  fail: Fail,
  raw: unknown,
  where: string,
): GpuFeaturesSpec | undefined => {
  if (raw === undefined) return undefined;
  if (!isRecord(raw)) throw fail.format(`${where}.gpuFeatures: オブジェクトでない`);
  assertAllowedKeys(fail, raw, GPU_FEATURE_KEYS, `${where}.gpuFeatures`);
  if (!Object.hasOwn(raw, "shaderF16")) return {};
  const shaderF16 = raw["shaderF16"];
  if (typeof shaderF16 !== "boolean") {
    throw fail.format(`${where}.gpuFeatures.shaderF16: 真偽値でない`);
  }
  return { shaderF16 };
};

/**
 * quant の weights 写像を読む。**weights に実在するキー → そのエントリに実在する dtype ラベル**
 * の完全写像であることを実行時にも検査する（型だけでは配布 JSON を縛れない — ADR 0041 §3）。
 */
const parseQuantWeights = (
  fail: Fail,
  raw: unknown,
  where: string,
  weights: Readonly<Record<string, WeightEntry>>,
): Readonly<Record<string, string>> => {
  if (!isRecord(raw)) throw fail.format(`${where}.weights: 無い / オブジェクトでない`);
  let mapping: Readonly<Record<string, string>> = {};
  for (const name of Object.keys(raw)) {
    if (!Object.hasOwn(weights, name)) {
      throw fail.reference(
        `${where}.weights: 未知の weights '${name}'` +
          `（利用可能: ${Object.keys(weights).join(" / ")}）`,
      );
    }
    const label = raw[name];
    if (typeof label !== "string") {
      throw fail.format(`${where}.weights.${name}: dtype ラベルが文字列でない`);
    }
    const entry = weights[name];
    if (!Object.hasOwn(entry, label)) {
      throw fail.reference(
        `${where}.weights: '${name}' に dtype '${label}' が無い` +
          `（利用可能: ${Object.keys(entry).join(" / ")}）`,
      );
    }
    mapping = { ...mapping, [name]: label };
  }
  for (const name of Object.keys(weights)) {
    if (!Object.hasOwn(mapping, name)) {
      throw fail.reference(
        `${where}.weights: '${name}' の dtype 指定が無い（完全写像が必要）`,
      );
    }
  }
  return mapping;
};

/**
 * 表示欄（ADR 0075 決定 1）を読む。**長さ上限を検査するだけで意味は解釈しない**
 * （決定 2 — 説明が実態と合っているかは検査できないし、しない。表示は呼び手の責任）。
 * 上限は「manifest は外部入力」という前提から来る境界検査。
 *
 * NOTE: 長さは**コードポイント数**で数える（UTF-16 の符号単位ではない）— 同じ見た目の
 * 文字列が綴り方（サロゲートペア・結合文字）で通ったり落ちたりするのを避ける。
 */
const parseText = (
  fail: Fail,
  raw: unknown,
  key: string,
  max: number,
  where: string,
): string | undefined => {
  if (raw === undefined) return undefined;
  if (typeof raw !== "string") {
    throw fail.format(`${where}.${key}: 期待 文字列 / 実際 ${typeof raw}`);
  }
  const length = [...raw].length;
  if (length > max) {
    throw fail.format(`${where}.${key}: 期待 ${max} 文字以内 / 実際 ${length} 文字`);
  }
  return raw;
};

/**
 * `requiredLimits`（limit 名 → 満たすべき最小値）を読む。名前は allowlist
 * （{@link REQUIRED_LIMIT_NAMES}）、値は正の安全整数だけを受ける。
 *
 * MUST: 未知名を黙って無視しない — device の limit 名は綴り違いが**静かな頭打ち**として
 * 現れる軸（要求しなかった limit は仕様既定値に落ちる）なので、無視は「宣言したのに効いて
 * いない」を沈黙で常態化させる。
 */
const parseRequiredLimits = (
  fail: Fail,
  raw: unknown,
  where: string,
): RequiredLimitsSpec | undefined => {
  if (raw === undefined) return undefined;
  const at = `${where}.requiredLimits`;
  if (!isRecord(raw)) throw fail.format(`${at}: オブジェクトでない`);
  assertAllowedKeys(fail, raw, REQUIRED_LIMIT_NAMES, at);
  let limits: RequiredLimitsSpec = {};
  for (const name of Object.keys(raw)) {
    const value = raw[name];
    if (typeof value !== "number" || !Number.isSafeInteger(value) || value <= 0) {
      throw fail.format(
        `${at}.${name}: 期待 正の安全整数 / 実際 ${JSON.stringify(value) ?? typeof value}`,
      );
    }
    limits = { ...limits, [name]: value };
  }
  return limits;
};

const parseQuant = (
  fail: Fail,
  raw: unknown,
  where: string,
  weights: Readonly<Record<string, WeightEntry>>,
): Quant => {
  if (!isRecord(raw)) throw fail.format(`${where}: quant がオブジェクトでない`);
  assertAllowedKeys(fail, raw, QUANT_KEYS, where);
  const mapping = parseQuantWeights(fail, raw["weights"], where, weights);
  const session = parseSession(fail, raw["session"], where);
  const gpuFeatures = parseGpuFeatures(fail, raw["gpuFeatures"], where);
  const label = parseText(fail, raw["label"], "label", MAX_LABEL_CHARS, where);
  const description = parseText(
    fail,
    raw["description"],
    "description",
    MAX_DESCRIPTION_CHARS,
    where,
  );
  const requiredLimits = parseRequiredLimits(fail, raw["requiredLimits"], where);
  return {
    weights: mapping,
    session,
    ...(gpuFeatures === undefined ? {} : { gpuFeatures }),
    ...(label === undefined ? {} : { label }),
    ...(description === undefined ? {} : { description }),
    ...(requiredLimits === undefined ? {} : { requiredLimits }),
  };
};

const parsePipeline = (fail: Fail, raw: unknown, where: string): PipelineId => {
  if (typeof raw !== "string") throw fail.format(`${where}.pipeline: 無い / 文字列でない`);
  const matched = PIPELINE_RE.exec(raw);
  if (matched === null) {
    throw fail.format(`${where}.pipeline: '<name>/<major>' の形でない: '${raw}'`);
  }
  return { name: matched[1], major: Number(matched[2]) };
};

const parseFormat = (fail: Fail, raw: unknown): string => {
  if (typeof raw !== "string") throw fail.format("format: 無い / 文字列でない");
  const matched = FORMAT_RE.exec(raw);
  if (matched === null) {
    throw fail.format(`format: 'karume/<major>' でない（取り違え）: '${raw}'`);
  }
  const major = Number(matched[1]);
  if (major !== FORMAT_MAJOR) {
    throw fail.format(
      `format: 未対応の major '${raw}' — この版は karume/${FORMAT_MAJOR} のみ読む` +
        `（旧版のパーサは持たない。配布形を karume/${FORMAT_MAJOR} で上げ直すか、` +
        `その manifest を読める版の @karume/hub を使う）`,
    );
  }
  return raw;
};

const parsePipelineConfig = (
  fail: Fail,
  raw: unknown,
  where: string,
): Readonly<Record<string, unknown>> => {
  if (!isRecord(raw)) {
    throw fail.format(`${where}.pipelineConfig: 無い / オブジェクトでない（空でも {} を明示）`);
  }
  const bytes = new TextEncoder().encode(JSON.stringify(raw)).length;
  if (bytes > MAX_PIPELINE_CONFIG_BYTES) {
    throw fail.format(
      `${where}.pipelineConfig: ${bytes} バイトが上限 ${MAX_PIPELINE_CONFIG_BYTES} を超えた`,
    );
  }
  return raw;
};

/**
 * 名前付きマップを上限つきで読む。`allowEmpty` は「宣言は必須だが中身が無くてもよい」席
 * （`assets` — 重みだけのモデルに空でない assets を捏造させない）にだけ使う。
 */
const parseMap = <T>(
  fail: Fail,
  raw: unknown,
  where: string,
  max: number,
  parseEntry: (name: string, value: unknown) => T,
  allowEmpty = false,
): Readonly<Record<string, T>> => {
  if (!isRecord(raw)) throw fail.format(`${where}: 無い / オブジェクトでない`);
  const names = Object.keys(raw);
  if (names.length === 0 && !allowEmpty) throw fail.format(`${where}: 空（1 つ以上が要る）`);
  if (names.length > max) {
    throw fail.format(`${where}: ${names.length} 件が上限 ${max} を超えた`);
  }
  let parsed: Readonly<Record<string, T>> = {};
  for (const name of names) {
    parsed = { ...parsed, [name]: parseEntry(name, raw[name]) };
  }
  return parsed;
};

const parseModel = (
  root: Record<string, unknown>,
  name: string,
  raw: unknown,
  seen: Map<string, FileRef>,
): ModelEntry => {
  const available = surveyLabels(root, name);
  const fail = createFail(available);
  const where = `models.${name}`;
  if (!isRecord(raw)) throw fail.format(`${where}: モデルエントリがオブジェクトでない`);
  assertAllowedKeys(fail, raw, MODEL_KEYS, where);

  const pipeline = parsePipeline(fail, raw["pipeline"], where);
  const weights = parseMap(
    fail,
    raw["weights"],
    `${where}.weights`,
    MAX_WEIGHTS,
    (weightName, value) => parseWeightEntry(fail, value, `${where}.weights.${weightName}`, seen),
  );
  const assets = parseMap(
    fail,
    raw["assets"],
    `${where}.assets`,
    MAX_ASSETS,
    (assetName, value) => parseFileRef(fail, value, `${where}.assets.${assetName}`, seen),
    true,
  );
  const quants = parseMap(
    fail,
    raw["quants"],
    `${where}.quants`,
    MAX_QUANTS,
    (quantName, value) => parseQuant(fail, value, `${where}.quants.${quantName}`, weights),
  );

  const defaultQuant = raw["defaultQuant"];
  if (typeof defaultQuant !== "string") {
    throw fail.format(`${where}.defaultQuant: 無い / 文字列でない`);
  }
  if (!Object.hasOwn(quants, defaultQuant)) {
    throw fail.reference(
      `${where}.defaultQuant: '${defaultQuant}' は quants に無い` +
        `（利用可能: ${Object.keys(quants).join(" / ")}）`,
    );
  }
  return {
    pipeline,
    weights,
    assets,
    quants,
    defaultQuant,
    pipelineConfig: parsePipelineConfig(fail, raw["pipelineConfig"], where),
    available,
  };
};

/**
 * `karume.json` のテキストを検査して読む。ADR 0041 の検査を**全て** parse 時に走らせ、1 つでも
 * 破れたら fetch を開始せずに throw する（「DL 開始後に初めて欠けが分かる」を許さないのが
 * manifest 導入の目的そのもの）。
 */
export const parseManifest = (text: string): Manifest => {
  const bytes = new TextEncoder().encode(text).length;
  if (bytes > MAX_MANIFEST_BYTES) {
    throw new ManifestFormatError(
      `manifest: ${bytes} バイトが上限 ${MAX_MANIFEST_BYTES} を超えた`,
      { available: NO_LABELS },
    );
  }
  let root: unknown;
  try {
    root = JSON.parse(text);
  } catch (error) {
    throw new ManifestFormatError(`manifest: JSON として読めない`, { cause: error });
  }
  if (!isRecord(root)) throw new ManifestFormatError("manifest: 最上位がオブジェクトでない");
  assertNoForbiddenKeys(root, "manifest", 0);

  const available = surveyLabels(root);
  const fail = createFail(available);
  // MUST: `format` を未知キー検査より**先**に見る。v1 manifest はトップレベルの綴りが丸ごと
  // 違うので、順が逆だと「未知キー 'components'」という的外れな診断が出て、本当の理由
  // （この版は karume/4 のみ読む）が隠れる（ADR 0041 §1）。
  const format = parseFormat(fail, root["format"]);
  assertAllowedKeys(fail, root, ENVELOPE_KEYS, "manifest");
  const generator = root["generator"];
  if (typeof generator !== "string" || generator === "") {
    throw fail.format("generator: 無い / 非空文字列でない");
  }

  // MUST: path の重複検査は manifest 全域で 1 つの表を共有する — モデル間の共有は「同じ path を
  // 書く」だけで成立する形（ADR 0041 §5）なので、モデルごとに表を分けると矛盾を見逃す。
  const seenPaths = new Map<string, FileRef>();
  const models = parseMap(
    fail,
    root["models"],
    "models",
    MAX_MODELS,
    (name, value) => parseModel(root, name, value, seenPaths),
  );

  const defaultModel = root["defaultModel"];
  if (typeof defaultModel !== "string") throw fail.format("defaultModel: 無い / 文字列でない");
  if (!Object.hasOwn(models, defaultModel)) {
    throw fail.reference(
      `defaultModel: '${defaultModel}' は models に無い` +
        `（利用可能: ${Object.keys(models).join(" / ")}）`,
    );
  }
  return { format, generator, defaultModel, models, available };
};
