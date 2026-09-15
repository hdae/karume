# 0102: states attention の行統計と PV の任意融合

- Status: accepted（2026-09-15・利用者のattention最適化依頼の範囲）
- 関連: [0067](0067-autoregressive-attention-vocabulary.md)、
  [0058](0058-numerics-opt-in-contract.md)、[0096](0096-speculative-decoding.md)
- 根拠: [調査・測定](../research/2026-09-15-attention-fusion.md)

## 問題と選択

Gemma E2B の decode では各層が QK、行統計、PV の3 dispatchを発行する。
行統計とPVを1カーネルにまとめ、行統計の一時バッファと1 dispatchを省く。
QK・append・stateの所有権・commit順は変えない。

行統計の最大値と分母は256 lane、PVの部分和は16 laneで従来の順に計算する。
行統計のlane番号は `lid.y * 16 + lid.x`。最大値を全員が読み終えてから
分母で共有メモリを書き換え、分母を読み終えてからPVの部分和に使い回す。
空行の `(amax, inv) = (0, 0)`、pad行の厳密0、ringの法がcapacityであることを維持する。

## 指定と互換性

`SessionOptions.stateAttentionReduce` に `"parallel-fused"` を追加する。
`"sequential"` と `"parallel"` の意味・既定・既存WGSL snapshotは維持する。
Gemma通常版/QATのpipelineは既存の同名オプションで受ける。
quant/defaultQuantとモデルの既定はこの段階では変更しない。

融合は `M <= 8` かつ静的列上限 `colCap <= 1024` のstates形に限る。
列上限はfullでcapacity、slidingでwindow−1+M。短い列でも実行時の論理長から
別のpipelineを選ばない。対象外の形とreadonly attentionは従来のparallel経路を使う。
深さタイルごとに行統計を再計算するため、長い列へ無条件には広げない。

融合前のparallelと加算順を保ち、実GPUではu32比較で検収する。
sequentialとの同一性を新たに主張する変更ではない。
既存の許容差・goldenを更新せず、MTP verify行0とdecodeの一致も別に確認する。

## メモリと検収

`planStateAttention` が融合の適用と中間バイト数を決め、実行と見積りで共有する。
`EstimateOptions.stateAttentionReduce` を指定すると同じ経路を見積る。
省略時は従来の参照経路。Gemmaの見積りにはSession構築に使った不変の設定を渡す。
この追加は保存IR/manifestの変更を伴わない。

単体の非有限値分類、GQA、pad、行分割、ring周回、u32近傍の論理位置、
Session経由の適用/非適用キー、メモリ見積り、実モデルの生成とMTPを検証する。
ブラウザ比較の初期選択は両E2Bでparallel→parallel-fused→parallel-fused→parallel、
8ロード・80生成。M2での採用判断を経てからモデル既定への昇格を検討する。

## 見送った案

PVの担当幅を64/256へ広げる案は文脈が長いほど遅くなり、256幅は4096列の1条件で
既存の絶対許容差5e-6も超えた。検証条件は緩めず見送る。
正規化済みの重みを共有する案も検証したが、全体の追加利得が明確でないため、
共有配列を増やさない融合を先に採る。外部のソースコードは複製していない。
