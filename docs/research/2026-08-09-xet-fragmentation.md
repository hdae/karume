# HF Xet の再構成断片化が DL を律速する — 実測と機序

> NOTE: 時点スナップショット。数値は 2026-08-09 の実測（開発機 / 開発回線・`curl` と Deno の
> `fetch` 直叩き）に基づく。HF 側の CAS・CDN の挙動は変わりうるので、再現には §9 の手順で
> 測り直すこと。先行実験は [2026-08-08-xet-split-probe.md](2026-08-08-xet-split-probe.md)（E1）。

発端は公開直後の `hdae/karume-anima-turbo` で text_encoder の DL が体感 2 MB/s 程度しか出ない
という観測。E1 は「速度を決めているのはストリーム数だけ」と結論したが、**なぜ per-stream が
遅いのか**には踏み込んでいなかった。本記録はその機序を特定し、E1 §5 の推測 1 点を撤回する。

**結論を先に**:

1. 遅さの機序は **Xet の再構成断片化**。resolve URL の実体はサーバ側再組み立て（xet-bridge）で、
   text_encoder は 1.19 GB を **1545 回のレンジ取得**から組み立てている。per-stream は
   1.5〜2.4 MB/s に張り付く。
2. 律速は **HTTP/2 ストリーム単位**であって TCP 接続数ではない（1 接続に 8 ストリームを
   多重化して 21 MB/s を実測）。したがって **Range 並列はブラウザでも同じだけ効く**。
3. 断片化は**アップロード時に確定し、既定設定のまま再アップロードしても治らない**（dedup が
   断片化した祖先を引き当てるため — 実証済み）。**E1 §5 の「再アップロードすれば治る公算が
   大きい」は誤りだったので撤回する。**
4. ただし **dedup を止めて上げ直せば直る**。`HF_XET_DEDUPLICATION_*` の 4 つを設定して同じ
   バイト列を同じパスへ上げ直すと、再構成が 1 xorb = 1 term の健全形に戻り、**コミットは 1 つも
   増えない**。実施済み（§5）— anima の text_encoder は 3069 → 18 term・1.92 → 9.61 MB/s。
   恒久の公開手順は [assets-layout.md](../assets-layout.md) の「公開」節。

## 1. 速度と断片化の対応

`https://cas-server.xethub.hf.co/v1/reconstructions/<x-xet-hash>` が返す term
（= xorb 内のチャンク範囲）と、`fetch_info` の URL レンジ数を実測速度と並べる。
**以下は修復前の値**（修復後は §5 の「本番適用の結果」）。

| ファイル                             |   サイズ | terms | xorbs | fetch レンジ数 | MiB/レンジ | per-stream 実測 |
| ------------------------------------ | -------: | ----: | ----: | -------------: | ---------: | --------------: |
| `text_encoder/model.safetensors`     | 1139 MiB |  3069 |    22 |       **1545** |       0.74 |    **1.9 MB/s** |
| `text_conditioner/model.safetensors` |  257 MiB |   531 |     8 |        **326** |       0.79 |    **3.2 MB/s** |
| `transformer/model.f16.safetensors`  | 3732 MiB |   125 |    61 |             98 |       38.1 |        （未測） |
| `transformer/model.i8.safetensors`   | 1872 MiB |    30 |    30 |             30 |       62.4 |       15.3 MB/s |
| `vae_decoder/model.safetensors`      |   48 MiB |     1 |     1 |              1 |       48.4 |       11.4 MB/s |

MiB/レンジと速度が単調に対応する。Xet の仕様は defrag 制御で **term 平均 8 チャンク**を
目標に置く（`huggingface.co/docs/xet/en/file-reconstruction`）が、text_encoder の term は
**チャンク数中央値 2** で、この制御が効いていない。

## 2. 律速は接続数ではなくストリーム数

| 測定                                    | 接続数 | ストリーム数 |                          スループット |
| --------------------------------------- | -----: | -----------: | ------------------------------------: |
| 単一 GET（curl）                        |      1 |            1 |                         1.9〜2.1 MB/s |
| 単一 GET（Deno `fetch`）                |      1 |            1 |                         1.4〜2.4 MB/s |
| 5 ファイル同時（Deno・別ファイル）      |      2 |            5 | 合計 17.6（text_encoder は 1.6〜2.4） |
| **Range 8 並列（curl・HTTP/2 多重化）** |  **1** |        **8** |                         **21.0 MB/s** |

`curl --parallel`（`--parallel-immediate` なし）で 8 転送のうち新規接続は 1 本のみ
（`%{num_connects}` が `0 0 0 1 0 0 0 0`）、`/proc/net/tcp` でも CDN 宛は 1 本だった。
**1 本の接続に多重化しても 8 ストリームぶんスケールする**ので、ブラウザで同一ホストへの接続が
1 本に集約されても Range 並列の効果は失われない。

同一ファイルへの Range 並列（text_encoder・128〜256 MiB 区間）:

| 並列度 |   1 |   4 |    8 |       16 |   32 |
| ------ | --: | --: | ---: | -------: | ---: |
| MB/s   | 1.9 | 6.8 | 12.2 | **28.4** | 32.5 |

E1 §4-4 の「並列は 4 で飽和」は**健全なオブジェクト（帯域律速）限定**だった。断片化した
オブジェクトでは 16 付近まで伸び、単一ストリーム比 **15 倍**になる。

なお公式 `hf_xet` クライアントはレンジ並列取得（旧ドキュメントで既定 16、現行は adaptive で
上限 64）で落とすため、この問題を回避している。`@karume/hub` は Web 標準 API のみで素の
resolve URL を 1 本の GET で読む設計（ADR 0038）なので、回避策を丸ごと取り逃していた。

## 3. 既定設定の再アップロードでは治らない（E1 §5 の撤回）

`hdae/anima-turbo`（2026-08-05）と `hdae/karume-anima-turbo`（2026-08-09）の text_encoder は
**`x-xet-hash` が完全一致**（`fec27d57…0548`）で、リポジトリ内の全ファイルの oid も一致する。
Xet は content-addressed なので、**別リポへ再アップロードしても同じ CAS オブジェクトを指す
だけ**で、チャンク配置は作り直されない。

E1 で摂動版（`hdae/karume-probe` の `d/`）が速かったのは、salt でバイト列が変わって別オブジェクト
になったからであり、「新しいリポだから速い」ではなかった。**配布形を作り直しても、バイト列が
同じである限り速度は変わらない。**

## 4. 断片化の形 — dedup のヒット/ミスの交替

text_encoder が使う 22 個の xorb は、はっきり 2 群に分かれる。

| 群            | 個数 | 1 xorb あたり使用チャンク | xorb 内の被覆率 | ファイル内での出現                 | term 長                       |
| ------------- | ---: | ------------------------: | --------------: | ---------------------------------- | ----------------------------- |
| A（領域）     |   19 |           ~900（~55 MiB） |         78〜85% | 連続した狭い窓（例 term 653..816） | 長い run（最大 873 チャンク） |
| B（散らばり） |    3 |                  484〜986 |        90〜100% | ファイルの 1/3 ずつの全域          | **1 チャンクが 987 個**       |

- 連続する term の **3065/3068 が別 xorb へ飛ぶ**（同一 xorb 内の飛びは 3 回だけ）。
- 群 B が占めるのは全 18374 チャンクのうち **2303（12.5%）**。群 B の出現間隔（間に挟まる
  群 A のチャンク数）は中央値 7・25% 分位 3・75% 分位 14 と幾何分布的で、ラウンドロビンでは
  ない。
- ファイル内部の自己重複は無い（64 KiB 整列ブロックの重複率 0.04%・ゼロ埋め 19 ブロック）。

つまり**大半のチャンクは既存 xorb にヒットして長い run になり、ヒットしなかった 12.5% が
新規 xorb に入って run を刻んでいる**。1 チャンクの term が 987 個あることが、これが
「ヒット/ミスの交替」であることを示している。

### ヒット元は特定できていない

アカウント配下の全 8 リポジトリを走査したが、群 A の xorb を参照するファイルは
`hdae/anima-turbo` の同一ファイル（同じ CAS オブジェクト）以外に無かった。上流モデルとの
dedup でもない — 同モデルの diffusers 版 `hdae/diffusers-anima-preview/text_encoder`
（1,192,133,232 B・2026-02-23）とは **xorb 共有ゼロ**である。

その diffusers 版は **terms=21 / xorbs=18・1 xorb = 1 term・被覆率 100%** と完全に健全で、
同一アカウント内に「正常なアップロードの対照」が存在する形になっている。

現時点で残る候補は「**現在は見えない先行アップロード**（削除済みリポ・他アカウントを含む
Hub 全体の dedup 対象）にほぼ同一バイト列があり、そこへ部分ヒットした」。確認手段が無いため
**未確定**として残す（§10）。

ローカルの xet キャッシュ（`~/.cache/huggingface/xet/`）には、**2026-08-05 18:51 付の shard
`.mdb` が 4 本**ある。アップロードのコミットは 18:54:31 なので、これは**アップロード中の
global dedup クエリでサーバから取り寄せた shard**である。同セッションのログは
`query_dedup` 98 回 / `upload_xorb` 92 回。リポジトリ全体 7.39 GB は無 dedup なら約 110 xorb
を要するので、**約 1.15 GB（≒ text_encoder 1 本ぶん）が既存データへ dedup された**計算に
なる。2026-08-02 のセッションは download のみで、祖先はこの開発機から上げたものではない。

## 5. 機序 — dedup と断片化防止のヒステリシス（`xet-core` v1.5.2 のソース確認）

`xet-core` を v1.5.2（`cb96dfe`）で読んだ結果、観測は次の機構で説明できる。

- **1 ファイルにつき開いている新規 xorb は 1 本だけ**（`FileDeduper::new_data`）。切り出しは
  64 MiB / 8192 チャンクに達したときのみ（`xet_core_structures/src/xorb_object/constants.rs`）。
  したがって群 B（散らばり）は「dedup されなかった残りだけが詰まる新規 xorb」であり、
  ファイルを 1/3 ずつ進みながら埋まって切れる。群 A は CAS に既にあった連続レイアウトの
  コピーで、そこへ dedup ヒットしている。
- **term が切れるのは供給元が切り替わるたび**。継続条件は「直前 term と同一 xorb かつ
  インデックス連続」（`file_deduplication.rs` の `file_data_sequence_continues_current`）
  だけなので、［ヒット］［新規］の交互で毎回外れる。**隣接 term 連続率 0% はこれで説明が付く。**
- **断片化防止は「効かなかった」のではなく、効いた結果の平衡**。
  `defrag_prevention.rs` の `allow_dedup_on_next_range` は、直近 128 range の平均
  chunk/range が閾値を下回り**かつ**次のマッチがその平均より小さいときだけ dedup を蹴る。
  既定は `min_n_chunks_per_range = 8.0` / ヒステリシス係数 `0.5` なので許容帯は **4〜8**。
  実測の平均 18374/3069 = **5.99 チャンク/term** はその帯のちょうど中央にある。
  仕様文の「平均 8 チャンク」は上側の閾値であって保証値ではない。
- **これは片道ラチェット**。この機構は断片化を 4〜8 で下げ止めるだけで、**CAS 上の断片化した
  レイアウトを連続に戻す機能は持たない**。既定設定で再アップロードするたびに祖先の断片化を
  継承する。
- **クライアントは「その file hash の reconstruction が既にあるか」を照会しない**。毎回
  自分の dedup 判断だけから `MDBFileInfo` を組み立てて新しい shard として上げる
  （`finalize()` → `add_file_reconstruction_info` → `upload_and_register_session_shards`）。
  §8 のリポジトリ間差はこれで説明できる。

### レバーの実測（probe リポでの A/B・2026-08-09）

`config_group!` の env 上書きは release ビルドでも有効（`xet_runtime/src/config/macros.rs` の
`apply_env_overrides` に `cfg` ガードは無い）。変数名は `HF_XET_DEDUPLICATION_<定数名の大文字>`。

同一バイト列（text_conditioner・257 MiB・元 531 term）を private の `hdae/karume-probe` へ
段階的に上げ直した。設定が読まれたことは hf_xet のログの `Config: … (user set)` 行で確認済み。

| 設定                                                           | New Data Upload | terms | xorbs |
| -------------------------------------------------------------- | --------------: | ----: | ----: |
| 既定（対照）                                                   |      **0.00 B** |   531 |     8 |
| `MIN_N_CHUNKS_PER_RANGE=1000000` + `..._HYSTERESIS_FACTOR=1.0` |         77.2 MB |   261 |     9 |
| ＋ `NRANGES_IN_STREAMING_FRAGMENTATION_ESTIMATOR=1`            |          141 MB |   152 |    10 |
| ＋ `GLOBAL_DEDUP_QUERY_ENABLED=false` かつ shard-cache 退避    |          223 MB | **6** |     5 |

- **既定では New Data Upload が 0.00 B** — 別リポへ上げてもラチェットが断片化をそのまま継承する。
- 推定器の窓が既定 128 range なのが 2 行目が中途半端な理由。**このファイルは全 261 range しか
  無いので前半がまるごと無防備**だった（`defrag_prevention.rs` の `rolling_chunks_per_range` は
  窓が埋まるまで `None` を返して素通しする）。窓を 1 にすると即座に効く。
- **記録はリポジトリ単位**。probe 内では最後のアップロードが記録を置き換え、先に上げた別パスの
  再構成も 6 term に変わった。一方その時点で本番リポは 531 term のままだった。
- **同一パス・同一内容でも xet の preupload は走る**（`No files have been modified since last
  commit. Skipping to prevent empty commit.` が出てもコミットが省かれるだけ）。つまり
  **実パスへ上げ直せば直り、コミットは 1 つも増えない**。
- **修復後は既定設定で上げても壊れない** — dedup が連続レイアウトの長いマッチを引き当てるため。

### 本番適用の結果（2026-08-09）

上表 4 行目の設定で、実パスへ上げ直した（コミットは 1 つも増えていない）。

| リポ / ファイル           |         terms |    MiB/レンジ | per-stream（同一範囲・修復後は全域 cold） |
| ------------------------- | ------------: | ------------: | ----------------------------------------: |
| anima `text_encoder`      | 3069 → **18** |  0.74 → 63.27 |                      1.92 → **9.61** MB/s |
| anima `text_conditioner`  |   531 → **9** |  0.79 → 42.89 |                     2.43 → **15.89** MB/s |
| jvnv `F1/voice/model.f16` |    65 → **2** |  2.98 → 52.15 |                                  （未測） |
| jvnv `F2/voice/model.f16` |    17 → **3** | 10.43 → 34.77 |                                  （未測） |
| jvnv `M1/voice/model.f16` |    93 → **3** |  2.13 → 34.77 |                             **7.83** MB/s |
| jvnv `M2/voice/model.f16` |    77 → **3** |  2.61 → 34.77 |                                  （未測） |
| jvnv `M1/voice/model.i8`  |    87 → **3** |  1.18 → 26.47 |                                  （未測） |

アップロード所要は anima の 2 本で計 120 秒、jvnv の 5 本で計 142 秒。手順は
[assets-layout.md](../assets-layout.md) の「公開」節に恒久ルールとして書いた。

### 効かない手段

| 手段                                         | 理由                                                                     |
| -------------------------------------------- | ------------------------------------------------------------------------ |
| 1 ファイルずつ上げる / `upload_large_folder` | **無効**。断片化は 1 ファイル内で完結する現象                            |
| xorb / chunk サイズの env 調整               | **不可**。該当 env は `#[cfg(debug_assertions)]` でしか compile されない |

なお群 A の被覆欠損（232.3 MiB）と群 B の実体（152.5 MiB）は**同オーダーだが一致はしない**ので、
「蹴られた dedup 区間がそのまま群 B に落ちた」だけでは全量を説明できない。祖先 xorb に
このファイル以外の内容が含まれている可能性が残る。

## 6. 断片化はリポジトリ横断でランダムに起きる

`hdae/karume-sbv2-jvnv`（同日アップ）では、同形・同サイズの兄弟ファイル間で揃わない。

| ファイル                       |  サイズ | terms | fetch レンジ数 | MiB/レンジ |
| ------------------------------ | ------: | ----: | -------------: | ---------: |
| `shared/text_encoder/model.i8` | 319 MiB |     5 |              5 |       63.8 |
| `F1/voice/model.f16`           | 104 MiB |    65 |             35 |       2.98 |
| `F2/voice/model.f16`           | 104 MiB |    17 |             10 |      10.43 |
| `M1/voice/model.f16`           | 104 MiB |    93 |             49 |       2.13 |
| `M2/voice/model.f16`           | 104 MiB |    77 |             40 |       2.61 |

こちらは**兄弟ファイル間で xorb を共有している**（F1 は 6 xorb 中 2 個を F2 と、4 個を M1 と
共有）。話者違いの近縁モデルどうしが部分 dedup し、その分だけ term が刻まれた形で、§4 と
同じ機序が弱く出ている。

## 7. karume 固有ではない — Hub 全体で起きている

第三者の公開モデルでも同じ相関が成立する（同一回線・同一時間帯の実測）。

| リポジトリ                   |   サイズ | terms | fetch レンジ数 | MiB/レンジ | per-stream 実測 |
| ---------------------------- | -------: | ----: | -------------: | ---------: | --------------: |
| `Qwen/Qwen2.5-1.5B-Instruct` | 2944 MiB |  1396 |            609 |       4.83 |   **2.95** MB/s |
| `openai/whisper-large-v3`    | 2944 MiB |  1082 |            651 |       4.52 |        （未測） |
| `Qwen/Qwen3-0.6B`            | 1434 MiB |   183 |             98 |      14.63 |      22.91 MB/s |
| `stabilityai/sdxl-vae`       |  319 MiB |    25 |             25 |      12.77 |      21.30 MB/s |

`Qwen/Qwen2.5-1.5B-Instruct` は term のチャンク数中央値が **2** で、karume の text_encoder と
同型に断片化している。**断片化は Xet の一般的な性質であって、karume のアップロード手順の
瑕疵ではない**（ただし karume の text_encoder は 0.74 MiB/レンジと、Qwen2.5 の 4.83 よりさらに
6 倍以上ひどい端に位置する）。

健全なオブジェクトは per-stream で 21〜23 MB/s 出るので、**回線が 2 MB/s の原因ではない**ことも
同時に確認できる。

## 8. 同一ファイルハッシュでもリポジトリごとに reconstruction が違う

| リポジトリ                | fileHash       | terms | xorbs |
| ------------------------- | -------------- | ----: | ----: |
| `hdae/karume-anima-turbo` | `fec27d57b4bc` |  3069 |    22 |
| `hdae/anima-turbo`        | `fec27d57b4bc` |  3105 |    21 |

各リポ内では 3 回測って完全に安定（決定的）だが、リポ間では一致しない。CAS 側に同一内容の
xorb が複数あり、リポジトリごとに見える範囲が違うことを示唆する。**この観測があるため、
「初回アップロード時点の配置」を今から遡って測ることはできない。**

## 9. 再現手順

```sh
# 1. resolve URL のヘッダから x-xet-hash を得る
curl -sS -I "https://huggingface.co/<repo>/resolve/main/<path>" | grep -i x-xet-hash

# 2. reconstruction を取る（read token はリポごとに発行される）
curl -sS "https://huggingface.co/api/models/<repo>/xet-read-token/main"   # -> casUrl / accessToken
curl -sS -H "Authorization: Bearer <accessToken>" "<casUrl>/v1/reconstructions/<x-xet-hash>"
```

返る JSON の `terms[]`（`hash` = xorb・`range` = xorb 内チャンク範囲・`unpacked_length`）と
`fetch_info`（実際の HTTP レンジ）を数える。速度は resolve URL への `Range` 付き GET で測る
（TTFB は全構成で 0.65〜0.81 秒の固定費）。

## 10. 未解決（openQuestions）

- **群 A の xorb を作ったアップロードが特定できていない**（§4）。現在のアカウントには存在
  しない。削除済みリポの可能性が残るが、外から確認する手段が無い。
- **初回アップロードで断片化したのか、後続の再アップロードで断片化したのか**が切り分けられない
  （§8 の理由）。ただし E1 §5 の記録（2026-08-05 / 08-08 の実測で `hdae/anima-turbo` が
  3.1〜4.7 MB/s）から、**2026-08-09 の再アップロード以前に既に遅かった**ことは確か。
- **修復した記録がどれだけ保つかは未検証**。同一環境での再アップロードでは壊れないことを
  確認したが（§5）、CAS 側の GC / 再パックや、他者が同じバイト列を既定設定で上げた場合に
  記録が入れ替わるかは分からない。**公開後にときどき term 数を測り直すのが安全**。
- **`karume-sbv2-fn` は未適用**（HF 非公開のためローカル運用のみ）。公開する場合は
  [assets-layout.md](../assets-layout.md) の「公開」節に従うこと。
- Range 並列を `@karume/hub` の `fetch` 経路で実装したときに同じ数値が出るかは未検証
  （E1 §7 から引き継ぎ）。並列度は 16 を基準に再測が要る。

## 追記 2026-09-04 — hf_xet 1.6.0 での global dedup 停止と回復

0.9.0 の初回公開（`hdae/karume-siglip2` / `hdae/karume-depth-anything-v2`）で断片化が再発した。
使ったのは nix の `hf`（huggingface_hub 1.10.2 / hf_xet 1.4.3）+ 当時の手順どおりの env 3 本。

### 初回アップロードの断片化（env 3 本・hf_xet 1.4.3）

| リポ / 対象               | 対象 shard | MiB/term                        | global dedup 照会 |  ヒット |
| ------------------------- | ---------: | ------------------------------- | ----------------: | ------: |
| siglip2 `so400m`          |          7 | 5 本が **4.2〜8.9**（目安未達） |    19（リポ全体） | 8（同） |
| siglip2 `base`            |          — | 10.7 / 20.3                     |                〃 |      〃 |
| depth-anything-v2 `small` |          — | **47**（健全）                  |                 2 |       0 |

ヒットの内訳: reconstruction 全 1988 MiB のうち **1128 MiB（57%）が自分の上げていない 36 個の
xorb** を指していた。ヒット先のリポは特定できていない（上流 `google` の so400m f32 safetensors
とは xorb 共有ゼロ = 別物）。§4 の「ヒット元は特定できていない」と同型の観測である。

### 実害（single-stream 実測）

| 対象                        |    per-stream |
| --------------------------- | ------------: |
| siglip2 の断片化 shard      |  **4.0 MB/s** |
| 同リポの健全 shard          | 8.9〜9.7 MB/s |
| gemma4 の健全 shard（対照） |     14.1 MB/s |

### 停止ノブは 1.6.0 で復活している

`HF_XET_DEDUPLICATION_GLOBAL_DEDUP_QUERY_ENABLED=false` は **hf_xet 1.4.3 には無く、1.6.0
（`tools/.venv` の `hf` = huggingface_hub 1.27.0）にはある**。バイナリの env 名一覧で確認し、
実行時も hf_xet のログに `global_dedup_query_enabled = false (user set)` が出た。
1.4.3 で後継とされた `HF_XET_MIN_SPACING_BETWEEN_GLOBAL_DEDUP_QUERIES` のほうは、巨大値に
しても効かない（2026-08-29 実測）。

### 回復は可能（1.4.3 での「片道ラチェット」は 1.6.0 では覆る）

shard-cache（`~/.cache/huggingface/xet/*/shard-cache`）を退避し、env 4 本 + `tools/.venv` の
`hf` で**同一バイト**を上げ直すと、CAS 照会 0 回で xorb を新規に書く。

| 段階                                               | CAS 照会         |   MiB/term |
| -------------------------------------------------- | ---------------- | ---------: |
| 初回（1.4.3・env 3 本）                            | 19 回中 8 ヒット |  4.2〜20.3 |
| 同一バイトの private probe リポ（1.6.0・env 4 本） | 0 回             |   **46.4** |
| 本番（リポ削除 → 再作成 → 再アップロード）         | 0 回             | **46〜61** |

- 実施した回復手順は**リポ削除 → 再作成・再アップロード**（2026-09-04 ユーザー裁定）。
- **同一リポ内の 2 コミット法（delete → 再 up）は未検証**。probe と同じ機構なので効く見込みだが
  実測していない。
- したがって §3 / §5 の「片道ラチェット・回復手段なし」は **hf_xet 1.4.3 での結論**であり、
  1.6.0 + 停止ノブ + shard-cache 退避の組では成立しない。恒久手順は
  [release-runbook](../release-runbook.md) §2。
