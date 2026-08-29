# 0071: manifest `karume/3` — dtype エントリの shard 欄

- Status: accepted（2026-08-20 — ユーザー裁定「karume/3 へ format bump + shards 単一形」。
  release 波の R1 のうち **HF 公開前締切ぶんの先行実施**〈同日のユーザー裁定 — release 節
  「後回し（2026-08-19）」の部分変更〉）
- Date: 2026-08-20
- 関連: ADR [0041](0041-manifest-v2.md)（manifest v2 — 本 ADR が format を上書き）/
  [0070](0070-shard-loading-admission.md)（shard ローダ契約の正本 — 決定 1 が本欄を予約）/
  [0063](0063-safetensors-physical-layout.md)（shard 単体の物理配置は不変）/
  [0038](0038-manifest-v1.md)（FileRef 3 点セットと検証規則 — 据え置き）

## Context

ADR 0070 は shard ロードの 2 相契約（graph shard 先頭・各 shard 独立整合 safetensors・
co-shard MUST・**振り分け表は manifest が正本**）を runtime / hub 側で先に固定し、manifest の
欄だけを R1 に送った。hub は単一形パースしかしない（ADR 0041 決定 1）ため、この欄は
**HF 公開前が締切** — 公開後に形を変えると全リポの manifest 差し替えと公開済みクライアントの
断絶が同時に起きる。公開直前の今なら、既存資産は 1 要素として書くだけで席が空く。

## Decision

### 1. format は `karume/3`。単一形パースを維持する

`format: "karume/3"` のみを受理する（karume/2 は既存の unsupported format 系エラーで loud に
落ちる）。v2 のまま欄を差し替える案・`file` | `shards` の 2 形受理案は不採用 — format 文字列が
スキーマ形を識別しなくなる／「hub は 1 形だけを読む」原則（ADR 0041 決定 1）に反する。
公開済みの v2 資産は HF 上の 2 リポのみで、どちらも本波で上げ直すため移行コストは今がゼロ点。

### 2. dtype エントリ: `{file, extras?}` → `{shards, extras?}`

```json
"front": {
  "i8": {
    "shards": [{ "path": "FN4/front/model.i8.safetensors", "size": 0, "sha256": "…" }],
    "extras": { "…": "…" }
  }
}
```

- `shards` は**順序付き・1 要素以上・1024 要素以下**（空・超過は fail loudly — 上限は
  ADR 0041 決定 7 と同じ DoS 防波堤の性格）。各要素は従来の FileRef 3 点セット検査
  （ADR 0038 決定 2・同一 path の {size, sha256} 一致要求）をそのまま通す。
- **順序が意味を持つ**: 先頭 = graph shard（`karume_ir` 保持）。意味検査は runtime が持つ
  （ADR 0070 決定 3 — hub は素通し）。
- **shard identity は導出**: id = 配列位置・bytes = `size`。manifest に id 欄は設けない —
  ADR 0070 の hub↔runtime 境界で使う `{id, bytes}` が完全導出可能なため、将来の API 工事で
  format を再度動かさない。

### 3. 混成 dtype の正本化（スキーマ変更なし）

dtype ラベルは**開語彙の選択子**であり、格納 dtype を主張しない。1 ファイル内の実 dtype は
safetensors ヘッダが正（dist の門が検査 — 既に `i4` ラベルの系列は i4+i8 混成）。R1 が席と
していた「1 コンポーネント内混成 dtype」は、この規約の明文化で完結する — 新欄は作らない。

### 4. exporter は自動分割しない（席のみ）

dist は常に 1 要素の `shards` を書く。複数 shard への分割規則（co-shard 保証・分割閾値 —
ADR 0070 決定 1）は最初の実需（LLM 級の配布）まで実装しない。

> **撤回（2026-08-29 — R1 統合波）**: 実需が LLM 級より先に来た（Chromium の単一 ArrayBuffer
> 上限で anima Base f16 がロード不能 — limitations）。分割規則は `karume.shards`
> （データ節 1GiB 固定・書き出し順逐次詰め・weight/scale 原子対・連番名）として実装し、dist は
> 現物から解決した複数要素を書く。1GiB 以下の資産は従来どおり単一ファイルでバイト不変。
> 詳細は ADR 0070 追記 2026-08-29。

### 5. R1 同席の API 工事 4 件は切り離し

shard identity API・`prepareModel → estimate → createSession` の 2 段境界・estimator 報告
改名・`ResidentWeight` union 化（2026-08-19 レビューの同席裁定）は**本 ADR に含めない** —
backlog release 節に残置。凍結が要るのは資産側の形だけで、コード API は公開後も動かせる。

## 追記（2026-08-20 — 配布リポ直下の法的テキスト席）

リポ内レイアウト（ADR 0041 §9）の宣言外ファイル例外に、直下の **`LICENSE.md` / `NOTICE.md`**
を追加した（`karume.dist.LEGAL_PATHS` — `Pipeline.root_files` で渡す・既定は空・受理名は
この 2 つだけで他名は fail loudly）。上流の重みライセンスが再配布の条件としてライセンス文の
コピーや Attribution Notice の同梱を要求する場合の席で、manifest は宣言しない（モデルの
資産ではなく配布リポそのものに掛かる）。初出は anima（CircleStone Non-Commercial License
§3(a)/(b)/(d) — 逐語原文は recipe の `circlestone_license.txt`）。

## Consequences

- 配布 4 リポ（sbv2-fn / sbv2-jvnv / irodori-v4-small / anima-turbo）の dist 再生成が必要
  （manifest のみ変化・系列バイトは不変）。
- JSR 0.3.x の hub は karume/3 リポを「unsupported format」で拒否する（pre-1.0 の破壊的
  変更・公開実績ゼロ時点の切替）。models パッケージの pin（ADR 0073）により、公開後の
  パッケージ↔資産の版ずれは以後 pin が吸収する。
- LLM 級モデルの複数 shard 配布は「exporter の分割規則 + dist の複数要素書き」だけで
  可能になる（hub / runtime / manifest は本 ADR + ADR 0070 で受け口完備）。
