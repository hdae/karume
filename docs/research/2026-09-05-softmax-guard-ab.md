# safe-softmax ガードの「-inf 源」判定変更（W-E1-3）の A/B — 公開 8 家族の再 export 実測

> 時点スナップショット（2026-09-05・エクスポータ = 網羅レビュー修正波 ffa1de9 以降・torch 2.13.0+cpu）。
> 生データと系列別の表は `outputs/bench/softmax-guard-ab/REPORT.md`（git 追跡外）が正本で、ここは結論と
> 読み方だけを残す。

## 何を測ったか

網羅レビューの修正波で、SDPA 分解の safe-softmax ガードを落とす判定が「-inf のリテラル定数だけを源と
見る」から「-inf を**生む** op（`log` / `log1p` / 除数が 0 と示せない `div`）まで遡る」へ変わった
（`tools/exporter/src/karume/normalize.py` の `_produces_neg_inf`）。この変更が公開済みの配布資産に
何をもたらすかが未実測だったので、公開 8 リポが指す **46 系列を全て新エクスポータで再 export** し、
再 export の前後（A = 旧エクスポータで焼いた現物 / B = 新エクスポータの出力）で次の 3 点を突き合わせた:
①グラフ shard の op 名ヒストグラム ②ガード関連 op（`softmax` / `safe_softmax` / `attention` /
`masked_fill` / `where`）の本数 ③`io.*.safetensors` の全 output の sha256。

## 結果

| 観点                                           | 結果                                                                                   |
| ---------------------------------------------- | -------------------------------------------------------------------------------------- |
| op ヒストグラム（46 系列・全コンポーネント）   | 差分 0 件（ノード数・initializer 数も全一致）                                          |
| golden 出力の sha256                           | 893 本すべて一致（`gemma4-e2b-product` は io を持たないので対象外）                    |
| 実配布グラフのガード発火サイト                 | 105 か所 — 分岐①（ガード除去）93 / 分岐②（`safe_softmax`）12 は**変更前と同じ**        |
| 陽性対照（旧判定 vs 現行を同一プロセスで対比） | `log` / テンソル除算を経由した合成 SDPA で `softmax 1 → safe_softmax 1` と観測量が動く |

**結論: 既存の公開資産と現行エクスポータの出力はビット一致で、再 export も上げ直しも不要。**
判定は 93 サイトで評価されたうえで動かなかった — attention のスコア側の `div` は `scores / sqrt(d)`
のように除数が非ゼロの数値リテラルで、`_produces_neg_inf` が「非ゼロリテラルは 0 除算の反証」として
源から外すため。レビューの反証注が警告した「`div` を op 名だけで源にすると全部が②へ倒れる」を
実装が回避できていることの実測でもある。irodori の dit 12 サイトは変更前から②（実行時マスクで
不活性を証明できない）で、源集合を広げても行き先は変わらない。sbv2 / deberta（自前の
`masked_fill` + `softmax` でガードの形を作らない）と anima transformer / gemma4（SDPA を `attention`
op として保存し分解経路に入らない）は発火 0。

## 注意

- A 側は `outputs/series/` の現物（旧エクスポータで焼いた系列）で、系列ディレクトリはコピーせず
  op ヒストグラム・sha256 だけを `before/` に記録した。再 export は同じ既定の出力先へ行ったので、
  `outputs/series/` の中身は結果として**ビット同一のまま**。
- 置き場は `docs/assets-layout.md` の規約（`outputs/bench/<model>/<日付>_<目的>/`）から外れて
  `outputs/bench/softmax-guard-ab/` にある（計測契約の指定）。
- `anima-f16/transformer`（静的 DiT・配布が指さない）は再 export していない。
