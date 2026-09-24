/**
 * メモリ内容器 — 手元のバイト列（合成した重み・別の器から読み出した実体）を**容器を書かずに**
 * Session 構築へ渡す供給面。
 *
 * 面は `krm` を開いた {@link "./open.ts"} の `OpenedContainer` と同じ 2 つ（`graphs` と `readBlock`
 * = `BoundContainer`）で、束縛規則の所有者は合流層（{@link bindDeclarations}）**1 つのまま**
 * である。ここがするのは「宣言 → 供給」の対応を確かめて、合流層が要る block 目次を**合成する**
 * ことで、規則①〜④・codec × 意味論 dtype・payload 長・piece の被覆はすべて合流層が見る
 * （二重実装しない — 片方だけ直されると受理集合が割れる）。
 *
 * ただし**合成そのものに閉じた門はこの層が持つ**（容器を開く経路には対応物が無く、合流層からは
 * 見えない）— 宣言との対応（shared でない initializer の不足 / 余剰・未知のグラフ名）、`pieces`
 * が 2 本以上、合成 id の衝突（規則①の等価物）、companion scale のバイト位置の整列、そして
 * `readBlock` の取得長と合成した宣言長の突合の 5 つ（{@link openMemoryContainer}）。
 *
 * **sha256 は掛けない**: 呼び手が渡したバイト列はネットワークもディスクも通っていないので、
 * 検証すべき「宣言と現物の食い違い」が存在しない（`krm` の block ごと digest は未検証の取得元を
 * 前提にした門 — container-v1 §7）。
 */

import {
  bindDeclarations,
  type BoundContainer,
  type Locator,
  pieceRowBytes,
  type SupplyRecord,
} from "./bind.ts";
import { codecEntry, type CodecName } from "./codecs.ts";
import type { Encoding, WeightSupply } from "./descriptor.ts";
import { ContainerFormatError } from "./header.ts";
import { DEFAULT_PART_BYTES } from "./limits.ts";
import type { IrDeclaration } from "../ir.ts";

/**
 * 格納の宣言（`Encoding` のうち呼び手が決める欄 — `packing` は codec 台帳から写す）。
 *
 * NOTE: `zeroPoint` の欄は**無い**（台帳の 4 codec は全て `zeroPoint: "forbidden"` なので
 * 現時点では表現できない形が存在しない）。非対称量子化の codec を台帳に足すときは、ここも
 * 同時に見る — `krm` では表現できてメモリ内容器では表現できない形が静かに生まれる。
 */
export type MemoryEncoding = {
  readonly codec: CodecName;
  /** 量子化 codec のみ・省略時 0。 */
  readonly rowAxis?: 0 | 1;
  /** 量子化 codec のみ・per-channel なら行長。 */
  readonly groupSize?: number;
  /** 量子化 codec は必須（f32・rank 2 `[rows, groups]` の並び）。 */
  readonly scale?: Uint8Array<ArrayBuffer>;
};

/** 分割供給の 1 本（先頭次元の半開区間と、その区間ぶんの payload の読み口）。 */
export type MemoryPiece = {
  /** 先頭次元の半開区間。 */
  readonly rows: readonly [number, number];
  /** その piece の payload（詰め物なし）。 */
  read(): Promise<Uint8Array<ArrayBuffer>>;
};

/**
 * initializer 1 本の供給。丸ごと 1 本（`bytes`）か行分割（`pieces` — 2 本以上）の排他。
 *
 * 分割形の値は「読み口」であって実体ではない: {@link openMemoryContainer} はバイト列を 1 つも
 * 抱えず、`readBlock` のたびに `read()` を呼ぶ。器 1 本ぶんの RAM しか使わない供給（ファイルを
 * 部分読みする呼び手など）はこの形で渡す。
 */
export type MemoryTensor =
  | {
    readonly encoding: MemoryEncoding;
    readonly bytes: Uint8Array<ArrayBuffer>;
    readonly pieces?: undefined;
  }
  | {
    readonly encoding: MemoryEncoding;
    readonly bytes?: undefined;
    readonly pieces: readonly MemoryPiece[];
  };

export type MemoryContainerInput = {
  /**
   * 載せたグラフは**全部**供給が要る（shared でない initializer が 1 本でも欠けたら fail
   * loudly）。複数グラフのモデルのうち 1 本だけ Session にしたい呼び手は、`graphs` 自体を
   * その 1 本に絞って渡す — 宣言だけを持つグラフ（`krg` に相当）はメモリ内容器では作れない。
   */
  readonly graphs: Readonly<Record<string, IrDeclaration>>;
  /** グラフ名 → initializer 名 → 供給。shared でない initializer は全部要る。 */
  readonly tensors: Readonly<Record<string, Readonly<Record<string, MemoryTensor>>>>;
};

/**
 * 丸ごと 1 本（`bytes`）で供給される実体を積み始める part。
 *
 * part 割りが意味を持つのは、Session 構築（`containerBatches` → `buildSessionState`）が
 * **part ごとに submit 1 回を出して完了まで待つ**ためである。`queue.writeBuffer` は staging を
 * 確保して溜め込み、submit の完了までそれを解放しない（`session-build.ts` の batch ループの
 * MUST — 数 GiB を一度に上げると VRAM が二重計上のまま最初の run に入り、初回ピークが重み
 * 1 本ぶん押し上がる）。したがって part 割りは**構築時の staging VRAM の上限**そのもので、
 * ホスト側でバイト列を既に握っているかどうかとは別の軸になる。
 *
 * - 丸ごと供給は宣言順に積み、累積が {@link DEFAULT_PART_BYTES} を超えるところで次の part へ
 *   移る（容器の書き手が part 長で切るのと同じ刻み）。companion scale は実体と同じ part。
 * - `pieces` は **piece 1 本 = 1 part = フェンス 1 回**。staging を piece 1 本ぶんに抑えるための
 *   割り方で、ホスト RAM は part の割り方に依らない（Session 構築は item を 1 本ずつ上げて手放す —
 *   container-v1 §11）。規則③どおり scale は piece 1 と同じ part に置く。
 *
 * 丸ごと供給の累積はグラフ横断で共有する（`containerBatches` は 1 グラフぶんしか消費しないので、
 * 多グラフ容器ではフェンスが余分に増えるだけ — staging が閾値を超える過小分割は起きない）。
 */
const WHOLE_PART = 2;

/** companion scale の dtype（初版は f32 の 1 通り — container-v1 §6.1）。 */
const SCALE_DTYPE = "f32" as const;

/** scale のバイト位置が満たす整列（f32 の view を張る側の要求 — `session-build.ts` の `scaleTensor`）。 */
const SCALE_ALIGN = 4;

/**
 * 呼び手の宣言 → `Encoding`（`packing` は codec 台帳の写し）。規則は容器の書き手と同じ:
 * 非量子化 codec は scale / groupSize / rowAxis を持てず、量子化 codec は scale と groupSize が要る。
 */
const resolveEncoding = (
  input: MemoryEncoding,
  scaleBlock: string | undefined,
  where: string,
): Encoding => {
  const entry = codecEntry(input.codec);
  if (entry.scale === "forbidden") {
    if (input.scale !== undefined || input.groupSize !== undefined || input.rowAxis !== undefined) {
      throw new ContainerFormatError(
        `${where}: codec '${input.codec}' は scale / groupSize / rowAxis を持てない`,
      );
    }
    return { codec: input.codec, packing: entry.packing };
  }
  if (input.scale === undefined || input.groupSize === undefined || scaleBlock === undefined) {
    throw new ContainerFormatError(`${where}: codec '${input.codec}' は scale と groupSize が要る`);
  }
  return {
    codec: input.codec,
    packing: entry.packing,
    rowAxis: input.rowAxis ?? 0,
    groupSize: input.groupSize,
    scale: { block: scaleBlock, dtype: SCALE_DTYPE },
  };
};

/**
 * 手元のバイト列から供給面を組む（同期・読みは lazy）。
 *
 * block 目次は合成する: id は `<graph>/<initializer>`、piece は `<graph>/<initializer>#piece<k>`
 * （k は 1 始まり）、scale は `<graph>/<initializer>#scale`。`length` は payload（詰め物なし）で
 * `offset` は 0 — メモリ内には「区間を切り出す元の連続領域」が無いため。
 *
 * ここで見るのは**この層が自分で合成したもの**だけ — 宣言との対応（shared でない initializer の
 * 不足 / 余剰・未知のグラフ名）、`pieces` が 2 本以上、合成 id の衝突、scale のバイト位置の整列、
 * そして `readBlock` の取得長と目次の宣言長の突合。束縛規則そのものは合流層が見る（モジュール doc）。
 */
export const openMemoryContainer = (input: MemoryContainerInput): BoundContainer => {
  const located = new Map<string, {
    readonly part: number;
    readonly record: SupplyRecord;
    readonly read: () => Promise<Uint8Array<ArrayBuffer>>;
  }>();
  const register = (
    id: string,
    part: number,
    length: number,
    read: () => Promise<Uint8Array<ArrayBuffer>>,
  ): void => {
    // 規則①の等価物: 合成した id が衝突したら fail loudly（後勝ちで上書きすると、別
    // initializer の実体が他人の scale として dequant される沈黙誤値になる）。衝突源は id の
    // 組み立てに使う区切りを名前が含む形（initializer 名 / グラフ名の `#` や `/`）で、名前の
    // 規則そのものは変えない。
    if (located.has(id)) {
      throw new ContainerFormatError(
        `合成した block '${id}' が二重に束縛されている（1 block ≤ 1 binding — initializer 名 / グラフ名が id の区切り '#' / '/' を含んでいる）`,
      );
    }
    located.set(id, { part, record: { id, offset: 0, length }, read });
  };

  for (const graphName of Object.keys(input.tensors)) {
    if (!Object.hasOwn(input.graphs, graphName)) {
      throw new ContainerFormatError(`供給に未宣言のグラフ '${graphName}' がある`);
    }
  }

  const binding: Record<string, Record<string, WeightSupply>> = {};
  // part の払い出し（{@link WHOLE_PART} の doc）。丸ごと供給は `wholePart` に積み上げ、
  // `pieces` は 1 本 1 part を `nextPart` から取る。
  let wholePart = WHOLE_PART;
  let wholeBytes = 0;
  let nextPart = WHOLE_PART + 1;
  /** 丸ごと供給 1 本（実体 + companion scale）を置く part。超えたぶんは次の part へ。 */
  const takeWholePart = (bytes: number): number => {
    if (wholeBytes > 0 && wholeBytes + bytes > DEFAULT_PART_BYTES) {
      wholePart = nextPart;
      nextPart += 1;
      wholeBytes = 0;
    }
    wholeBytes += bytes;
    return wholePart;
  };
  for (const [graphName, declaration] of Object.entries(input.graphs)) {
    const supplied = input.tensors[graphName] ?? {};
    // 突合集合は「shared でない initializer」— shared はバイトを持たない宣言（貸し手の重みを
    // 借りる）なので、供給されたら余剰として落ちる。
    const expected = Object.entries(declaration.initializers)
      .filter(([, init]) => !init.shared)
      .map(([name]) => name);
    // 不足も余剰も**全件列挙**する（1 件ずつ落とすと、供給を組む側が何本足りないのか分からない）。
    const missing = expected.filter((name) => !Object.hasOwn(supplied, name));
    const surplus = Object.keys(supplied).filter((name) => !expected.includes(name));
    if (missing.length > 0 || surplus.length > 0) {
      throw new ContainerFormatError(
        `グラフ '${graphName}' の供給が initializer 宣言と一致しない: 不足 [${
          missing.join(", ")
        }] / 余剰 [${surplus.join(", ")}]（shared 宣言は供給しない）`,
      );
    }
    const supplies: Record<string, WeightSupply> = {};
    for (const name of expected) {
      const tensor = supplied[name];
      const where = `graph '${graphName}' initializer '${name}'`;
      const id = `${graphName}/${name}`;
      const scale = tensor.encoding.scale;
      const encoding = resolveEncoding(
        tensor.encoding,
        scale === undefined ? undefined : `${id}#scale`,
        where,
      );
      if (scale !== undefined && scale.byteOffset % SCALE_ALIGN !== 0) {
        // scale は GPU 構築で `Float32Array` の view をコピーせずに張る（`session-build.ts` の
        // `scaleTensor` の MUST「バイト位置の 4 バイト整列は供給元が保証する」）。整列していない
        // view を通すと、そこで素の RangeError が出て転送層の文言に化ける。
        throw new ContainerFormatError(
          `${where}: scale の byteOffset ${scale.byteOffset} が ${SCALE_ALIGN} バイト整列でない（${SCALE_ALIGN} バイト整列した view を渡す）`,
        );
      }
      let scalePart: number;
      if (tensor.pieces === undefined) {
        // 実体と companion scale は同じ part（丸ごと供給の 1 本ぶんの staging をまとめて数える）。
        scalePart = takeWholePart(tensor.bytes.byteLength + (scale?.byteLength ?? 0));
        register(id, scalePart, tensor.bytes.byteLength, () => Promise.resolve(tensor.bytes));
        supplies[name] = { block: id, encoding };
      } else {
        if (tensor.pieces.length < 2) {
          throw new ContainerFormatError(
            `${where}: pieces は 2 本以上（1 本なら bytes で渡す）`,
          );
        }
        const shape = declaration.values[name].shape.map(Number);
        const rowBytes = pieceRowBytes(encoding.codec, shape, where);
        // 規則③: companion scale は piece 1 と同じ part。
        const firstPart = nextPart;
        scalePart = firstPart;
        const pieces = tensor.pieces.map((piece, index) => {
          const block = `${id}#piece${index + 1}`;
          register(
            block,
            firstPart + index,
            (piece.rows[1] - piece.rows[0]) * rowBytes,
            () => piece.read(),
          );
          return { block, rows: piece.rows };
        });
        nextPart += tensor.pieces.length;
        supplies[name] = { pieces, encoding };
      }
      if (scale !== undefined) {
        register(`${id}#scale`, scalePart, scale.byteLength, () => Promise.resolve(scale));
      }
    }
    binding[graphName] = supplies;
  }

  const locate: Locator = (id, by) => {
    const found = located.get(id);
    if (found === undefined) throw new ContainerFormatError(`${by}: 未宣言の block '${id}'`);
    return found;
  };

  return {
    graphs: bindDeclarations({
      graphs: input.graphs,
      // const 領域はグラフ容器（`krg` / `krm` の part 1）だけが持つ概念で、メモリ内容器には無い
      // — 定数も重みと同じ `tensors` から供給する。
      constants: [],
      binding,
      locate,
    }),
    readBlock: async (id) => {
      const found = located.get(id);
      if (found === undefined) throw new ContainerFormatError(`未宣言の block '${id}'`);
      // 丸ごとの実体と scale は**渡された器をそのまま**返す（複製しない — `BoundContainer`
      // の MUST）。piece は呼ぶたびに `read()` を引き直す（保持しない = 器 1 本ぶんの RAM）。
      const bytes = await found.read();
      // MUST: 取得長 = 合成した目次の宣言長（`open.ts` の `readBlock` と同じ門）。piece の
      // 長さは宣言から導いた値なので、`read()` が長いバイト列を返しても合流層の突合は恒真に
      // なり、`containerBatches` の `subarray(0, payloadBytes)` が黙って切り詰める — 呼び手が
      // 行範囲と読み口を取り違えた形が沈黙誤値として GPU に載る。
      if (bytes.byteLength !== found.record.length) {
        throw new ContainerFormatError(
          `block '${id}': 取得長 ${bytes.byteLength} が宣言 ${found.record.length} と違う`,
        );
      }
      return bytes;
    },
  };
};
