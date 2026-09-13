# 0100: RMS正規化の32レーンsubgroup縮約を明示指定で追加する

- Status: accepted（2026-09-13）。任意経路として提供。既定quantへの採用は未決。
- 関連: [0017](0017-rms-norm-conv2d-clamp-min.md)、[0058](0058-numerics-opt-in-contract.md)、[0099](0099-rms-norm-add-fusion.md)。

## 背景と代替案

RMSの元の256部分和と加算木を保持して128/64スレッドへ配る候補は遅くなった。
vec2/vec4化は幅に依存し、元の木をsubgroupShuffleXorで保つ案もモデル全体の改善は小さい。
subgroupAddで部分和を計算するスカラー案はQAT E2Bで全体の利得があり、小規模の品質評価に大幅な劣化を認めなかった。
加算順が変わるので元のキーの実装へ差し替えず、利用者が承認した任意指定の方針に従う。
数値の正本は[調査記録](../research/2026-09-13-rms-subgroup-reduction.md)。

## 決定

1. `AcquireGpuOptions.subgroups?: boolean`を追加する。省略・falseでは新機能を要求しない。
   trueはWebGPUの`subgroups`・`subgroup-size-control`とWGSLの`subgroup_id`を必須とし、不足を拒否する。
   さらに32レーン×8グループで既知解を全256レーンへ返すカーネルを実走する。
   必要な機能の列挙だけを動作証明にせず、生成・実行・読み戻しの失敗やdevice消失を例外にする。
   能力の取得自体は推論の加算方式を切り替えない。
2. `SessionOptions.rmsNormReduce?: "workgroup" | "subgroup32"`と型`RmsNormReduce`を追加する。
   既定はworkgroup。不正な値と必要なdevice機能の不足は重み転送前に拒否する。
   subgroup32は幅128を超えるf32 RMSと、`fuseRmsNormAdd: true`で既に適格なRMS→addに適用する。
   幅128以下は従来版。融合のshape・使用数・公開値・引数順の条件は変えない。
3. 新カーネルは256スレッド、`@subgroup_size(32)`。各スレッドの入力走査は従来と同じ。
   subgroupAddの8部分和を共有メモリへ書き、同期後、各グループが同じ8値を読んで合計する。
   subgroup IDとlocal IDの配置の対応は仮定しない。全レーンが縮約へ参加する。
   行のgrid-strideと、共有部分和の読み終わりを揃える同期を維持する。
   RMS→addの整数丸め障壁と引数順も維持する。元のRMSキー・WGSLは変更しない。
4. 加算順の変化による下位bit・生成列の変化を許す別経路である。
   参照golden・SHA・許容誤差は変更しない。同じ設定のM=1/4/8の先頭行は検査するが、
   それだけで実モデルの投機生成全体を検収済みとはしない。
5. Gemma通常/QATのpipelineへ`rmsNormReduce`を渡せるようにする。
   GPUを内部取得する場合はsubgroup32指定時だけ必要な能力を要求し、渡されたGPUはSessionの門で検査する。
   target/drafterへ同じ設定を渡す。quantの保存語彙・配布重み・既定・公開revisionは変更しない。
   Deno 2.9.6では必要な機能が未提供で、共通の既定quantに指定すると既存利用者が起動できなくなる。
   M2の速度・品質も未検収なので、今はChromeで明示指定する範囲に留める。
6. 比較画面はQATのみ、並列GEMV・細分化chunk64・RMS→add融合・投入上限768を共通条件とする。
   初期選択をworkgroup→subgroup32→subgroup32→workgroupの4設定・40生成にする。
   通常版・参照経路・前回の3設定往復も残す。選んだ縮約方式と有効なGPU機能を保存する。

## 仕様と検収

[subgroupAdd](https://www.w3.org/TR/WGSL/#subgroupadd-builtin)と
[WGSL仕様](https://www.w3.org/TR/WGSL/)のsubgroup_size属性・subgroup_idの条件に従う。
固定32レーンを要求する実測案を採用し、未計測の可変サイズ案を代わりに実行しない。
外部実装のソースコードは複製していない。

- 参照WGSLの既存snapshot、3種類の新WGSLの別snapshot、融合キーの分離。
- 必要な機能・不正値・沈黙した全0・一部レーンの誤値・device消失の検査。
- CPU参照との既存許容帯、端数幅、行数1/4/8/19、強制grid-stride、融合の両引数順と後続入力の寿命。
- Chromeで実モデルの生成速度・固定入力列の帰属実験・固定品質評価。
- Denoの全体verifyでは新機能の実走だけを明示SKIPする。Chromeで同じ数値検査を実走する。
- M2の追試後に対象モデルと提供範囲を判断する。未計測のE4B・長文・投機へ既定適用しない。
