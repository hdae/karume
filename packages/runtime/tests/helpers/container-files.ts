/**
 * 系列出力 / 配布形ミラーの**容器ファイル**（`krm` の part 列）を、runtime の容器面
 * （`openContainer`）へ繋ぐテスト用の面。テストが持つのは「ファイルの見つけ方」と
 * 「区間の読み方」だけである。
 *
 * Python 側 `karume.container.resolve_sequence` / `container_parts`
 * （tools/exporter/src/karume/container.py）の**鏡像**。MUST: 連番の綴りを 2 箇所で別々に
 * 育てない — 焼く側と読み返す側で規則が割れると、前回の書き出しの残骸を今回の期待値で読む形が
 * 黙って通る。
 *
 * ## part の番号づけ（file 番号と `BlockSource` の添字は 1 ずれる）
 *
 * ファイル名の連番は **1 始まり**で、`model-00001-of-000NN.krm` が `BlockSource` の **part 0**
 * （`[ヘッダ][グラフ記述][モデル記述]`）である。以降 `-00002-` が part 1（const 領域）、
 * `-00003-` 以降が part 2..（重み / 資産）。{@link fileBlockSource} は `parts[i]` をそのまま
 * part `i` に対応づけるので、{@link resolveParts} が返す**添字順を崩してはならない** MUST。
 *
 * ## 読む量
 *
 * `read` は毎回 `Deno.open` + `seek` で区間だけを取る（ファイルを丸ごと RAM に載せない）。
 * `verified: false` MUST — ローカルのファイル列は誰も検証していないので、`openContainer` 側で
 * block ごとの sha256 を掛けてもらう（true を名乗ると改ざんが素通りする — container-v1 §7）。
 */

import {
  type BlockSource,
  type DescriptorExpectation,
  openContainer,
  type OpenedContainer,
} from "../../mod.ts";

/** 連番の桁数（`-NNNNN-of-NNNNN`）— Python 側 `SEQUENCE_DIGITS` と同値。 */
const INDEX_DIGITS = 5;

const escapeRegExp = (text: string): string => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * この代表 path と同じ容器の part ファイル名に一致する正規表現。
 *
 * MUST: stem / suffix は escape する — 実 path にはドットもハイフンも入る（`model.i8.krm`）ので、
 * 素で埋めると無関係なファイルを拾う（glob を使わないのも同じ理由）。
 */
const sequencePattern = (name: string): RegExp => {
  const dot = name.lastIndexOf(".");
  const stem = dot <= 0 ? name : name.slice(0, dot);
  const suffix = dot <= 0 ? "" : name.slice(dot);
  return new RegExp(
    `^${escapeRegExp(stem)}-(\\d{${INDEX_DIGITS}})-of-(\\d{${INDEX_DIGITS}})${
      escapeRegExp(suffix)
    }$`,
  );
};

/** URL の最終要素（`%` エスケープを解いた実ファイル名）。 */
const baseName = (url: URL): string => decodeURIComponent(url.pathname).split("/").pop() ?? "";

/**
 * ファイルの有無。
 * MUST: NotFound 以外は伝播させる — 全 I/O エラーを「資産が無い」に丸めると、資産ルートの
 * マウント異常が SKIP に化けて、実行されていない検証が静かに緑になる。
 */
const isFile = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

/** ディレクトリの列挙（不在は空 — 中身の異常は伝播させる）。 */
const listDir = (url: URL): readonly Deno.DirEntry[] => {
  try {
    return [...Deno.readDirSync(url)];
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return [];
    throw cause;
  }
};

/**
 * 容器の**代表 path**（`…/model.krm` / `…/model.i8.krm`）→ 実在する part ファイル列（添字順 =
 * `BlockSource` の part 順）。単一形と存在しない容器はどちらも `[representative]` を返す
 * （不在の診断は呼び手の既存の門が持つ — ここで先回りすると綴りが 2 つに割れる）。
 *
 * MUST: 曖昧な現場は fail loudly。単一形と連番の同居・`of` の食い違い・番号の欠けやはみ出しは、
 * どれも「どのバイト列を読むか」が一意に決まらない。黙って一方を選ぶと、前回の書き出しを
 * 今回の期待値で突き合わせる形になる。
 */
export const resolveParts = (representative: URL): readonly URL[] => {
  const parent = new URL("./", representative);
  const pattern = sequencePattern(baseName(representative));
  const found = new Map<number, URL>();
  const totals = new Set<number>();
  for (const entry of listDir(parent)) {
    if (!entry.isFile) continue;
    const match = pattern.exec(entry.name);
    if (match === null) continue;
    found.set(Number(match[1]), new URL(encodeURIComponent(entry.name), parent));
    totals.add(Number(match[2]));
  }
  if (found.size === 0) return [representative];
  const where = decodeURIComponent(representative.pathname);
  if (isFile(representative)) {
    throw new Error(
      `${where}: 単一形と part 連番（${found.size} 本）が同居している` +
        " — 前回の書き出しの残骸を消してから読む",
    );
  }
  if (totals.size !== 1) {
    throw new Error(`${where}: part 連番の総数が ${[...totals].sort()} と食い違っている`);
  }
  const [total] = totals;
  const files: URL[] = [];
  for (let index = 1; index <= total; index += 1) {
    const file = found.get(index);
    if (file === undefined) {
      throw new Error(`${where}: part 連番 1..${total} のうち ${index} 本目が無い`);
    }
    files.push(file);
  }
  if (found.size !== total) {
    // MUST: 下にも上にもはみ出しを数える。上側だけを見ると `model-00000-of-00003.krm` のような
    // 0 以下の番号が混ざった現場で「はみ出した番号 []」と空リストで落ち、診断が読めなくなる
    // （落ちること自体は本数の不一致が保証する）。
    const surplus = [...found.keys()]
      .filter((index) => index < 1 || index > total)
      .sort((a, b) => a - b);
    throw new Error(`${where}: part 連番 1..${total} からはみ出した番号 ${surplus} がある`);
  }
  return files;
};

/**
 * 容器が実在するか（単一形 / 連番のどちらでも）。資産の有無で SKIP を決める門が
 * 「`model.krm` があるか」を直に見ていた席の置き換え。
 *
 * MUST: 同期であること — 呼び手はモジュール先頭の `const ANY_PRESENT = …` で引く。
 */
export const modelPresent = (representative: URL): boolean =>
  isFile(resolveParts(representative)[0]);

/**
 * part ファイル列 → `BlockSource`（区間読み）。`parts[i]` が part `i`。
 *
 * part 長はここで一度だけ `stat` して持つ（`BlockSource.partLength` は同期の面で、
 * `openContainer` が宣言長との突合に使う）。
 */
export const fileBlockSource = (parts: readonly URL[]): BlockSource => {
  const lengths = parts.map((part) => Deno.statSync(part).size);
  const at = (index: number): { readonly url: URL; readonly length: number } => {
    const url = parts[index];
    const length = lengths[index];
    if (url === undefined || length === undefined) {
      throw new Error(`test: part ${index} は無い（part 列は ${parts.length} 本）`);
    }
    return { url, length };
  };
  return {
    partCount: parts.length,
    // MUST: ローカルのファイル列は誰も検証していない（container-v1 §7）。
    verified: false,
    partLength: (index) => at(index).length,
    read: async (part, offset, length) => {
      const { url, length: partBytes } = at(part);
      if (offset < 0 || length < 0 || offset + length > partBytes) {
        throw new Error(
          `test: ${url.pathname}: 区間 [${offset}, ${
            offset + length
          }) が part 長 ${partBytes} の外`,
        );
      }
      const handle = await Deno.open(url, { read: true });
      try {
        await handle.seek(offset, Deno.SeekMode.Start);
        const into = new Uint8Array(new ArrayBuffer(length));
        for (let filled = 0; filled < length;) {
          const read = await handle.read(into.subarray(filled));
          if (read === null) {
            throw new Error(
              `test: ${url.pathname}: offset ${offset} からの ${length} バイトを読み切れない`,
            );
          }
          filled += read;
        }
        return into;
      } finally {
        handle.close();
      }
    },
  };
};

/**
 * 系列出力 / ミラーの容器を代表 path から開く（part 列の解決 → 区間読みの `BlockSource` →
 * `openContainer`）。
 *
 * `expect` は外側が持つ 2 文書の期待値（manifest の `container.descriptor`）。系列出力には
 * manifest が無いので省略する — その場合 2 文書の完全性は誰も保証しないが、block ごとの
 * sha256 は `verified: false` の経路で掛かる。
 */
export const openSeriesContainer = async (
  representative: URL,
  expect?: DescriptorExpectation,
): Promise<OpenedContainer> =>
  // MUST: `async` にする — part 列の解決と `stat` は同期に落ちるので、素の式にすると
  // 欠番やマウント異常だけが**同期の throw**で返る。呼び手が `await` と `.catch` のどちらで
  // 受けても同じ形で届くことを、面の側で揃える。
  await openContainer(
    { kind: "source", source: fileBlockSource(resolveParts(representative)) },
    expect,
  );
