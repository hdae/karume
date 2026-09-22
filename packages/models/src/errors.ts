/**
 * `@karume/models` の**家族横断**の入力起因エラー。
 *
 * 分類の軸は**呼び手の分岐先**（`@karume/hub` の `HubError` と同じ流儀）: これが飛ぶのは
 * 「渡した要求そのものが受理できない = 入力を直せば通る」ときだけで、HTTP サーバーなら 400 に
 * 当たる。内部不変条件の破れ・資産（manifest / tokenizer / shard）の齟齬・GPU 容量の不足は
 * 素の `Error` や既存の型のまま投げる — あれは 500 であって、呼び手が入力を直しても直らない。
 *
 * 型が家族ごとに割れていると、8 家族を同じホストに載せた側は**メッセージの文字列を読む**しか
 * 400 / 500 を分ける手が無くなる。1 本にするのはそのためで、家族の数だけ専用型を作らない。
 *
 * ## 適用範囲
 *
 * **生成要求の値域・型・組合せ**の検査だけ。次の 2 つは打つ手が違うので同じ型に混ぜない:
 *
 * - model / quant / sampler 名の綴り違い（打つ手は「受理集合を引き直す」）
 * - 呼び出し手順の違反（`dispose` 済みの再利用・二重生成 — 打つ手は「呼ぶ順を直す」）
 *
 * どちらも素の `Error` のままにする。範囲を後から**足す**のは呼び手にとって互換だが、
 * 狭めるのは既に分岐している側を壊すので破壊変更である。
 *
 * MUST: 型を分けるだけで**メッセージの質は落とさない**（期待と実際の両方を書く）。型は分岐の
 * ため、メッセージは人のためで、どちらも要る。
 *
 * NOTE: 派生は `Sbv2InputError`（`./sbv2/errors.ts`）と `GenerationCapacityError`
 * （`./generation/sequence.ts`）の 2 本だけ。前者は ADR 0072 決定 6 の線引きを仕様として持ち、
 * 後者は切り詰めの計算に要る実値を欄で運ぶ — どちらも `instanceof` の外に**追加の情報**が
 * あるから型が要る。情報が増えない分岐先を型で割らない。
 *
 * DECIDED: [ADR 0107](../../../docs/decisions/0107-model-input-error.md)。
 */
export class ModelInputError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "ModelInputError";
  }
}
