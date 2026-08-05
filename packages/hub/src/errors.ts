/**
 * hub のエラー型（ADR 0038 §5「エラーの形」）。
 *
 * 分類の軸は**利用者の分岐先**:
 * - {@link ManifestFormatError} — JSON / 構造 / 規模（manifest が形として壊れている）
 * - {@link ManifestReferenceError} — 参照と語彙（defaultPreset・weights 写像・未知キー / 未知値）
 * - {@link ManifestPathError} — path 許可リスト違反（SHA ピン外への traversal 防波堤）
 * - {@link IntegrityError} — 取得物が 3 点セットと食い違う（size / sha256 / content-length）
 * - {@link HubFetchError} — 取得層由来（404・認証・revision 解決失敗）の文脈付き透過
 *
 * MUST: 全エラーに**利用可能な preset / variant ラベル一覧**（{@link AvailableLabels}）を載せる
 * — GGUF 利用者が README の quant 表で得ている情報の代替であり、失敗時に「では何なら動くのか」を
 * 一次情報として返すのが manifest 導入の目的の一部だから（ADR 0038 §5）。manifest の構造が
 * 壊れていて列挙できない場合だけ空になる。
 */

/** 失敗時に提示する「今このリポで選べるもの」の一覧。 */
export type AvailableLabels = {
  /** manifest に実在する preset 名。 */
  readonly presets: readonly string[];
  /** variants を持つコンポーネント名 → そのラベル一覧。 */
  readonly variants: Readonly<Record<string, readonly string[]>>;
};

/** manifest の構造が壊れていてラベルを列挙できないときの値。 */
export const NO_LABELS: AvailableLabels = { presets: [], variants: {} };

export type HubErrorOptions = {
  readonly available?: AvailableLabels;
  readonly cause?: unknown;
};

/**
 * hub が投げる全エラーの基底。直接 throw はしない（`instanceof HubError` で一括して
 * 捌けるようにするためだけに公開する）。
 */
export class HubError extends Error {
  /** 失敗時点で判明している利用可能 preset / variant ラベル。 */
  readonly available: AvailableLabels;

  constructor(message: string, options: HubErrorOptions = {}) {
    super(message, { cause: options.cause });
    this.name = "HubError";
    this.available = options.available ?? NO_LABELS;
  }
}

/** JSON 不正・構造違反・規模上限超過（ADR 0038 §1「parse 時の構造検査」「規模上限」）。 */
export class ManifestFormatError extends HubError {
  constructor(message: string, options: HubErrorOptions = {}) {
    super(message, options);
    this.name = "ManifestFormatError";
  }
}

/**
 * 参照と語彙の違反（ADR 0038 §1/§2/§3）。宙吊りの `defaultPreset` / preset 名・`weights` の
 * 過不足・未知キー・allowlist に無い `session` 値・重複 path の 3 点セット不一致。
 */
export class ManifestReferenceError extends HubError {
  constructor(message: string, options: HubErrorOptions = {}) {
    super(message, options);
    this.name = "ManifestReferenceError";
  }
}

export type ManifestPathErrorOptions = HubErrorOptions & { readonly path: string };

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

export type IntegrityErrorOptions = HubErrorOptions & {
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

export type HubFetchErrorOptions = HubErrorOptions & {
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
