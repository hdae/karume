# 0079: SBV2 の合成入力は 2 層（フレーズ層 → モーラ層）— yomi 依存の分離

- Status: accepted（2026-08-25 — ユーザー裁定。設計対話で段階的に確定し「OK, それで確定に
  しましょう」で最終形・命名も同日裁定。実装同日 `7cd82c8`）
- Date: 2026-08-25
- 関連: ADR [0072](0072-sbv2-text-injection.md)（注入席 — 本 ADR が supersede）/
  [0008](0008-public-api.md)（薄い公開面）/ backlog later（バレル・ファミリープレフィックス
  見直し）
- 実装: `packages/models/src/sbv2/`（text/utterance.ts / text/model-input.ts / pipeline.ts）

## Context

`@karume/models` は SBV2 のテキスト解析を `@hdae/yomi` に依存して内部で行っていた（text 席・
辞書 19MB の取得と Promise キャッシュ・overlay 解決・`analyzeProsody` — hub を経由しない唯一の
ネットワーク経路でもあった）。yomi を models の依存から外す方針は 0.5.0 検討時に確定していたが
（backlog）、「呼び手が yomi を直接叩くと手数が増える」ヘルパー設計が未決で延期されていた。

モデルの実効入力は 3 本の並行構造（音素列・トーン列・word2ph×BERT）で、成立条件は
①`sum(word2ph) === 音素数` ②音素列とトーン列が同長 ③全てが同一解析由来であること。
旧設計（ADR 0072）の prosody 下書き往復はこの同期を守るための装置（派生欄を持たせない・
核クランプ・「同じ解析と対で戻す」規則・音素列内容一致門）で、把握コストが高かった。

## Decision

**models は yomi 非依存の 2 つの表現（層）と、その間の変換 1 本だけを所有する。**

```
text ──(yomi: analyzeWithWords)──▶ フレーズ層 ──(toSbv2Utterance)──▶ モーラ層 ──(generate)──▶ 音声
                                    編集 = アクセント核（メイン UI）    編集 = tone の高低直指定
```

1. **フレーズ層 = `Sbv2Phrases`**: yomi `analyzeWithWords` の返り値（`{ result, words }`）が
   **構造的に満たす** karume 所有型。アクセント指定の編集面はここ（`accentPhrases[i].accentNucleus`
   を直す・句を割る）。編集後は再変換するだけで、核→トーン規則を呼び手が再実装することはない。
2. **変換 = `toSbv2Utterance(phrases)`**: 純関数 1 箇所。核→トーン展開（上流 g2p と同一規則:
   上端クランプ k=min(核, モーラ数)・平板/頭高/中高尾高・負核は fail loudly）・句 punctuations を
   末尾モーラへ付け替え・上流が促音を `cl` と綴る場合は音素 `q` へ畳む（`Sbv2Mora.vowel` は
   **音素そのもの**）。命名は `toSbv2Utterance` — `sbv2Utterance` だと族名の「2」が変換
   イディオム（x2y）に誤読される（バレル見直しでプレフィックスが外れれば `toUtterance` へ
   収斂できる — later 起票）。
3. **モーラ層 = `Sbv2Utterance`**: フラット（句の入れ子なし）。
   `{ leadingPunctuations, moras: [{ kana, consonant?, vowel, tone: 0|1, punctuations? }], words }`。
   **`tone` が唯一の実効編集点**（核で書けない任意の 0/1 パターンはここで直指定 — 旧 givenTone を
   構造ごと吸収）。words は読み取り専用（word2ph と BERT テキストの源 — BERT 入力は words の
   surface から導出されるため**元テキスト欄は持たない**）。
4. **`generate(utterance, options?)` の第一引数化** — 経路 1 本。必須はちょうど utterance 1 個で、
   ノブ（speaker/style/…/seed）は全 optional の options bag（`fromPretrained(ref, options)` と同型）。
5. **廃止**（0.6.0 breaking・シム無し）: text 席・`dictionary` / `overlay` 注入席・
   `analyzeProsody`・`Sbv2Prosody` / 下書き型・`givenTone` 席。プレビュー関数（sbv2PhoneTone 案）も
   作らない（tone と音素はオブジェクト内に見えている — 必要になれば追加のみで足せる）。
6. **門は全て models に残る**: moras 由来音素列 ↔ words の内容一致（位置ごと）・
   `sum(word2ph) === P`・tone の 0|1。読み（音素列）の変更は overlay で辞書を直して再解析する
   一本道（adjust_word2ph 不採用は ADR 0072 決定 8 のまま — 原則は解析器側の責務として存続）。

## 設計の経緯（棄却した中間案）

- **A 案（analysis + prosody/givenTone の 3 席）**: 分離前の「pipeline が text を再解析して
  突き合わせる」世界の名残で、経路が複数あり混乱する（ユーザー指摘）→ 1 本化。
- **完全フラット単層案**: 核クランプ・平板/尾高縮退・givenTone を一掃できるが、編集の主面で
  ある**フレーズ構造を合成入力から追放するのはやり過ぎ**（ユーザー裁定）→ フレーズ層を公式の
  編集面として復権させ 2 層に。
- **1 モーラ句トリック**（1 phrase 1 mora で任意トーン）: 可能（核 0→[0]・核 1→[1]）だが、
  フレーズ表示を壊すハックを正規手段にしない — 直指定はモーラ層で。

## Consequences

- breaking は models の SBV2 API のみ（0.6.0）。**配布形・manifest・pin は不変**（HF 再
  アップロード不要）。
- **吸収のビット同一性の証明 = e2e WAV sha256 門 3 本の不変**（i8 / i8+bert4 / i4 — 期待値に
  1 文字も触れず移行し、実 GPU で全一致を確認済み）。
- 「同じ解析と対で戻す」規則は消滅（1 オブジェクト = 1 解析の産物として構造で成立）。
- words の surface 書き換えは門で検出できない宣言された自由度（BERT 入力が変わる）—
  読み取り専用の規律は型 readonly + doc で受ける。
- 辞書の支度・overlay は呼び手側の責務。写経見本は `examples/sbv2/`（glue パッケージは実需が
  出たら追加で切り出せる — 2026-08-25 裁定）。
- `deno publish --dry-run` の出力に yomi の痕跡なし（src/ の import 0 件）。公開依存グラフから
  消えたことの最終確定は 0.6.0 publish 後の meta.json で行う。
