/**
 * hub のエラー型（ADR 0038 §5「エラーの形」・語彙は ADR 0041 の v2 へ更新）。
 *
 * 分類の軸は**利用者の分岐先**:
 * - {@link ManifestFormatError} — JSON / 構造 / 規模（manifest が形として壊れている）
 * - {@link ManifestReferenceError} — 参照と語彙（defaultModel / defaultQuant・weights 写像・未知キー）
 * - {@link ManifestPathError} — path 許可リスト違反（SHA ピン外への traversal 防波堤）
 * - {@link IntegrityError} — 取得物が 3 点セットと食い違う（size / sha256 / content-length）
 * - {@link HubFetchError} — 取得層由来（404・認証・revision 解決失敗）の文脈付き透過
 *
 * MUST: 全エラーに**利用可能な model / quant / dtype ラベル一覧**（{@link AvailableLabels}）を
 * 載せる — GGUF 利用者が README の quant 表で得ている情報の代替であり、失敗時に「では何なら
 * 動くのか」を一次情報として返すのが manifest 導入の目的の一部だから（ADR 0038 §5 / 0041 §8）。
 * 空になるのは列挙する材料が無い 2 つの場合だけ — ①manifest の構造が壊れていて列挙できない
 * ②manifest をまだ取得できていない（revision 解決失敗・`karume.json` の 404 等）。
 */

/**
 * 失敗時に提示する「今このリポで選べるもの」の一覧。
 *
 * NOTE: `quants` / `dtypes` は**文脈のモデル**のもの（v2 は 1 manifest = 複数モデルなので、
 * モデルを跨いで混ぜると「別モデルの quant 名」を勧める誤誘導になる）。モデル文脈が定まらない
 * 位置（トップレベルの検査・取得層）では空になり、`models` だけが埋まる。
 */
export type AvailableLabels = {
  /** manifest に実在するモデル名（v2 で初めて機械可読になった列挙 — ADR 0041 §8）。 */
  readonly models: readonly string[];
  /** 文脈のモデルに実在する quant 名。 */
  readonly quants: readonly string[];
  /** 文脈のモデルの weights 名 → その dtype ラベル一覧。 */
  readonly dtypes: Readonly<Record<string, readonly string[]>>;
};

/** manifest の構造が壊れていてラベルを列挙できないときの値。 */
export const NO_LABELS: AvailableLabels = { models: [], quants: [], dtypes: {} };

type HubErrorOptions = {
  readonly available?: AvailableLabels;
  readonly cause?: unknown;
};

/**
 * hub が投げる全エラーの基底。直接 throw はしない（`instanceof HubError` で一括して
 * 捌けるようにするためだけに公開する）。
 */
export class HubError extends Error {
  /** 失敗時点で判明している利用可能 model / quant / dtype ラベル。 */
  readonly available: AvailableLabels;

  constructor(message: string, options: HubErrorOptions = {}) {
    super(message, { cause: options.cause });
    this.name = "HubError";
    this.available = options.available ?? NO_LABELS;
  }
}

/** JSON 不正・構造違反・規模上限超過（ADR 0041 §2「トップレベル形」・§7「規模上限」）。 */
export class ManifestFormatError extends HubError {
  constructor(message: string, options: HubErrorOptions = {}) {
    super(message, options);
    this.name = "ManifestFormatError";
  }
}

/**
 * 参照と語彙の違反（ADR 0041 §2/§3）。宙吊りの `defaultModel` / `defaultQuant`・`weights`
 * 写像の過不足・未知キー・allowlist に無い `session` 値・重複 path の 3 点セット不一致。
 */
export class ManifestReferenceError extends HubError {
  constructor(message: string, options: HubErrorOptions = {}) {
    super(message, options);
    this.name = "ManifestReferenceError";
  }
}

type ManifestPathErrorOptions = HubErrorOptions & { readonly path: string };

/** path 許可リスト違反（ADR 0038 §2）。 */
export class ManifestPathError extends HubError {
  /** 違反した生の path 文字列（URL 組み立て前）。 */
  readonly path: string;

  constructor(message: string, options: ManifestPathErrorOptions) {
    super(message, options);
    this.name = "ManifestPathError";
    this.path = options.path;
  }
}

/** 完全性検証の失敗元（キャッシュ読出し / network 取得）。 */
export type IntegritySource = "cache" | "network";

type IntegrityErrorOptions = HubErrorOptions & {
  readonly repo: string;
  readonly revisionSha: string;
  readonly path: string;
  readonly expected: string;
  readonly actual: string;
  readonly source: IntegritySource;
};

/**
 * 取得物が manifest の 3 点セットと食い違った（ADR 0038 §5）。
 *
 * NOTE: キャッシュ読出し側の不一致は取得層が self-heal（evict → 取り直し・1 往復まで）に
 * 使うため通常は外へ出ない。外へ出るのは network 取得物の不一致（`source: "network"`）が主。
 */
export class IntegrityError extends HubError {
  readonly repo: string;
  /** 取得を固定していた commit SHA（40 桁）。 */
  readonly revisionSha: string;
  readonly path: string;
  /** manifest が宣言していた値（sha256 / バイト数）。 */
  readonly expected: string;
  /** 実際に得られた値。 */
  readonly actual: string;
  readonly source: IntegritySource;

  constructor(message: string, options: IntegrityErrorOptions) {
    super(message, options);
    this.name = "IntegrityError";
    this.repo = options.repo;
    this.revisionSha = options.revisionSha;
    this.path = options.path;
    this.expected = options.expected;
    this.actual = options.actual;
    this.source = options.source;
  }
}

type HubFetchErrorOptions = HubErrorOptions & {
  readonly repo: string;
  readonly revisionSha?: string;
  readonly path?: string;
};

/**
 * 取得層由来の失敗（404・認証・revision 解決失敗）を**文脈付きで透過**する容れ物
 * （ADR 0038 §5）。原因は `cause` にそのまま残す。
 */
export class HubFetchError extends HubError {
  readonly repo: string;
  /** 解決済み commit SHA。revision 解決自体に失敗した場合は undefined。 */
  readonly revisionSha?: string;
  /** 対象ファイルの path。manifest 取得前 / revision 解決失敗では undefined。 */
  readonly path?: string;

  constructor(message: string, options: HubFetchErrorOptions) {
    super(message, options);
    this.name = "HubFetchError";
    this.repo = options.repo;
    if (options.revisionSha !== undefined) this.revisionSha = options.revisionSha;
    if (options.path !== undefined) this.path = options.path;
  }
}
