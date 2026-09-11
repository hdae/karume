# 0097 — Gemma 4 QAT mobile を別ファミリで統合する

- Status: accepted（2026-09-11 — 利用者が実測後の段階案を承認、family `gemma4-qat` と E2B / E4B を指定）
- 根拠: [INT2 / SRQ の試作と段階案](../research/2026-09-10-codex-mtp-optimization.md#qat-mobile-の-int2-と固定丸め2026-09-11)
- 関連: [0069](0069-packed-w4-storage.md)（packed 格納）、[0085](0085-ple-host-gather.md)（PLE）、
  [0092](0092-distribution-repos-and-sources.md)（配布と取得元）、[0096](0096-speculative-decoding.md)（MTP）

## 背景

公式 mobile QAT は通常 Gemma 4 と共通する骨格を持つが、固定の INT2 / INT4 / INT8 重みと
SRQ（Static Range Quantization、固定 scale による活性値の丸め）を使う。
単に既存モデルの量子化名を変えたり、重みを f32 に広げたりするだけでは同じ計算にならない。
単体実測では INT2 の容量と速度に利得があった。CPU と GPU の線形縮約差が丸め境界を越す例も残るため、
単体の一致だけで全モデルの品質を保証できない。

## 決定

1. 配布・利用者向けの family は **`gemma4-qat`**。その中の model を **`e2b` / `e4b`** とする。
   通常 `gemma4` のモデル名や既定量子化は変更しない。対象は公式 `*-qat-mobile-transformers` の text 生成。
   同じ QAT 名でも GGUF / unquantized / compressed-tensors の各形式を自動で同じものとは扱わない。
2. 将来の配布先は `karume-gemma4-qat`、manifest の pipeline 名は `gemma4-qat` とする。
   旧 Gemma reader がこの pipeline を拒否する性質を維持する。公開前の source pin は作らず、
   検収ではローカル配布形を使用する。実際の公開・push はこの統合作業に含めない。
3. 実行・tokenizer・会話・RoPE・取得処理の共通部分は既存 Gemma の実装を再利用する。
   family の追加を理由にパイプライン本体を複製しない。QAT の入口では対応形式と固定丸めの存在を明示的に検証する。
4. 固定整数・scale の保存値を再量子化せず保持する。IR の INT2、固定 SRQ、PLE の INT4 を実装する。
   保存形式の shape / byte 数 / packing / scale、丸めと非有限値の契約は実装単位で本 ADR へ追記し、
   リーダ・ライタ・CPU 参照・GPU を揃える。既存の資産と数値許容差を変更しない。
5. 通常生成の一致・性能・資源解放を先に検収する。head と token embedding の共有は全バイト一致を確認して扱う。
   MTP drafter の I8 借用を INT2 へ暗黙に置き換えない。MTP は通常生成の検収後に別設計とする。

## 実装単位と検収

- INT2 格納: IR / safetensors / loader / CPU 展開 / 見積り / exporter の読書きと packing を揃え、
  GEMV・prefill GEMM・embedding の packed GPU 経路を検収する。端と符号の全値、既存格納型との u32 一致を検査する。
- 固定 SRQ: 明示 op の CPU / GPU を実装する。除算の誤差を丸め境界から除く試作を基に、
  scale=0、境界両隣、正負ゼロ、飽和、非有限値と許可する scale 範囲を検査する。
- QAT recipe / PLE: 公式の固定重みを保持した E2B / E4B の変換と、PLE INT4 の全量・行読取を実装する。
  元の I8 読取を維持し、全量と区間読みで同じ PLE を得ることを検査する。
- family / 生成: ローカル資産から使える入口を用意し、公式 CPU 参照、Deno、Chrome で通常生成・複数ターン・
  中断・解放を検収する。RAM / VRAM は実測と見積りを区別する。M2 の GPU 数値検収は別実機の作業。

各単位は全体検証後に独立コミットする。exporter / recipe 変更時は両 Python パッケージの pytest も実行する。
実験出力は `outputs/bench/karume/2026-09-11_qat-integration/`、既存資産は上書きしない。

## 承認の扱い

この段階案は承認済みであり、含まれる実装に段階ごとの再承認は要らない。
仕様変更は実装前に説明・記録し、承認範囲を超える変更や前提を覆す想定外の問題が出た場合に再確認する。
