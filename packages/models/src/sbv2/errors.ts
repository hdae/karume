import { ModelInputError } from "../errors.ts";

/**
 * SBV2 の入力起因エラー。
 *
 * 分類の軸は**呼び手の分岐先**（`@karume/hub` の `HubError` と同じ流儀）:
 * これが飛ぶのは「渡した要求そのものが受理できない」ときだけで、HTTP サーバーなら 400 に当たる。
 * 内部不変条件の破れ（`sum(word2ph)` の不一致・tile 走査の破れ・資産の齟齬）は素の `Error` の
 * まま投げる — あれは 500 であって、呼び手が入力を直しても直らない。
 *
 * MUST: 型を分けるだけで**メッセージの質は落とさない**（期待と実際の両方を書く）。型は分岐の
 * ため、メッセージは人のためで、どちらも要る。
 *
 * NOTE: {@link ModelInputError} の派生であり、**これ以上の派生は作らない**。家族横断の
 * 「入力を直せ」は親が受け持つので、この型が残っているのは ADR 0072 決定 6 の線引き（SBV2 の
 * どの検査が入力起因で、どれが内部不変条件か）を名前で持つためである。hub が 5 種に割って
 * いるのは利用者の分岐先が実際に違う（形式 / 参照 / path / 完全性 / 取得）からで、こちらは
 * 「入力を直せ」の 1 つしかない。
 */
export class Sbv2InputError extends ModelInputError {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "Sbv2InputError";
  }
}
