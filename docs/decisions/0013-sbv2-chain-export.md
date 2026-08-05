# 0013 — SBV2 チェーンの export 戦略

- Status: accepted（2026-08-02）
- 根拠資料: [../research/2026-08-02-sbv2-chain-recon.md](../research/2026-08-02-sbv2-chain-recon.md)

## 決定

- **emit ターゲットは 5 本**: `dp`（単体 — 現行語彙で通る唯一のモジュール、早期 E2E の足場）/
  `front`（enc_p + dp + sdp(reverse) の融合 1 グラフ、動的 P、Pmax=512）/ `flow` / `dec` /
  `voice`（flow + dec の融合）。ゴールデンは各ターゲットで出す。融合の根拠は中間 hidden の
  readback 往復排除（プロトタイプ実測）。段階検証のため単体ターゲットも残す。
- **乱数・実行時ノブはグラフに焼かない**（P2 の規約を踏襲）: sdp reverse の乱数は外部入力
  `z_noise` へ昇格（noise_scale 乗算はホスト側・同 seed でゴールデン決定性を担保）。
  sdp_ratio 混合・durations 化もホスト側。
- **パッチ層**（エクスポータ内、モデルパッケージ本体は変更しない）: ① spline の分岐フリー・
  非破壊同値実装 ② 相対位置注意の gather 化（2P−1 等の二次 shape 式を除去 — export 成立の
  構造的前提） ③ FFN の明示 pad を conv padding へ畳む（奇数 kernel かつ非 causal のみ等価 —
  assert で固定）。パッチはクラス属性のプロセス全域差し替えのため**「パッチ前の参照」は
  1 プロセス 1 回だけ** — verify 系オプションの排他を CLI が機械的に拒否し、順序は
  「全ケースの参照値を確定 → パッチ適用 → 比較」を必須とする。
- **相対位置注意の表は二方式を使い分ける**:
  - front（P ≤ 512）: P2 方式のまま **Tmax 焼き込み + sym_prefix_slice**（約 2MB — 実害なし）。
  - flow（T ≤ 4096）: 焼き込みは O(Tmax²) ≈ 134MB になるため**グラフ入力へ昇格**
    （idx_k / valid）。実 Ty の表は**ホスト TS が生成**し、Python 側ゴールデン生成器と
    **バイト一致のパリティテスト**で式レベルの一致を固定する。表は clamp 済み・idx_v は
    pad 済み長に常に範囲内のため、gather の OOB 規約（NaN 汚染 — ADR 0009）には抵触しない。
  - window_size == 4 は load 時 assert（生成器の焼き込み幅と食い違うと shape エラーに
    ならないまま誤った埋め込みを読む — 沈黙誤値クラス）。
- **sym_max はターゲット別既定**: front=512（P=音素数）/ flow・dec・voice=4096（T=フレーム数）。
  機械的強制が無く誤値が沈黙する罠があるため、emit CLI がターゲットごとの既定値を持ち、
  逸脱は明示フラグでのみ許す。
- **weight_norm は dec のみ**（enc_p/dp/sdp/flow は 0 件 — 実測済み）。export 前に
  remove_weight_norm（冪等ラッパ）を通し、将来の f16/i8 丸めは **remove 後の実効重みに
  当てる**順序制約を今から明文化する。flow に導出パラメータが無いことは assert で固定。

## 検討した代替案

- モジュール別 export のみ: 中間 readback 往復と dispatch 分断でチェーン実行が細切れになる。
- 全チェーン 1 グラフ: front と voice の間に長さ計算（durations → y_mask）のホスト処理が
  挟まるため構造的に不可能。
- flow の表も Tmax 焼き込み: sym_max=4096 で定数 134MB — 成果物肥大と発話長の人工上限。却下。

## 帰結

- ホスト TS に表生成器という**新しい検証責務**が生まれる（Python 鏡像とのパリティテスト、
  window_size assert）。
- エクスポータ CLI は verify 系の排他検査を持つ（恒真化 = 偽 PASS の遮断）。
