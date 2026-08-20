"""MiniCPM5-1B の**校正コーパス**（GPTQ / AWQ が見る活性を作る 48 文）。

`karume.quant_calib` の校正付き丸めは「その層に実際に流れる活性」から丸め先を選び直す方式
なので、コーパスの性格がそのまま丸めの偏りになる。ここは `sweep_w4.py` の校正付き構成
（{@link minicpm5.sweep_w4.CALIB_CONFIGS}）が読む唯一の入力で、選定方針は次の 3 点:

- **評価文と分離する** — {@link minicpm5.export.GOLDEN_CASES}（= 波 E の greedy 期待列の
  素材）と 1 文も重ねない。話題も重ねない（「フランス / 日本の首都」系は 1 文も入れない）。
  重ねると「校正で見た文をそのまま評価する」形になり、teacher / greedy の一致は**校正の
  質ではなく漏れ**を測る数になる。
- **素の言語モデルの分布へ寄せる** — 模型は base LM（chat template は載せない — `export.py`）
  なので、校正も chat 形式にせず素の散文で採る。英日混合の一般散文（叙述・問い・コード
  断片）で、話題は互いに離す（1 話題へ寄せると活性の偏りがその話題の形に張り付く）。
- **長さを散らす** — 短文（10 token 級）から段落（60 token 級）まで。GPTQ の `H = Σ XᵀX` は
  トークン数で重み付くので、長文だけだと長文の活性が、短文だけだと文頭の活性が支配する。

内訳は英語 26 / 日本語 16 / コード断片 6。固有の人名・実在組織は書かない（校正の入力が
特定の固有名の活性を引き当てるのを避ける — 一般語彙の分布だけを見たい）。
"""

from __future__ import annotations

#: 校正入力の 48 文（この順で先頭から使う — `--calib-limit N` は**先頭 N 文**を採る）。
#: 順序は英語 → 日本語 → コードではなく**混ぜて**並べる: 上限を掛けた縮小実行でも
#: 言語とスタイルの混合が保たれる（先頭 4 文だけで英語 / 日本語 / コードが揃う）。
CALIB_TEXTS: tuple[str, ...] = (
    "A compiler turns source text into machine instructions, and a good error message often "
    "matters more than the speed of the pass that produced it.",
    "分散システムでは、返事が来ないことと相手が死んでいることを区別できない。",
    "What happens to a queued job when the worker process is killed before it acknowledges "
    "the message?",
    "def merge(left, right):\n"
    "    out = []\n"
    "    while left and right:\n"
    "        out.append(left.pop(0) if left[0] <= right[0] else right.pop(0))\n"
    "    return out + left + right",
    "The library keeps a small cache of parsed configuration files, and invalidates an entry "
    "whenever the file's modification time changes.",
    "ログを読むときは、まず時刻の並びが本当に単調かどうかを疑ったほうがいい。",
    "Rain fell for three days, and by the fourth the river had risen high enough to cover the "
    "lower path along the bank.",
    "駅前の商店街は昼のうちは静かで、日が落ちてから人の流れが変わる。",
    "Why do floating point sums give different results when the terms are added in a "
    "different order?",
    "for (let i = 0; i < items.length; i += 1) {\n"
    "  if (!seen.has(items[i].id)) {\n"
    "    seen.add(items[i].id);\n"
    "    unique.push(items[i]);\n"
    "  }\n"
    "}",
    "She read the manual twice before touching the machine, then wrote down the steps she "
    "thought were missing from it.",
    "この関数はなぜ引数を二つ取るのですか。片方は呼び出し側で計算できるように見えます。",
    "A hash table degrades to a linear scan when every key collides, which is why the hash "
    "function matters as much as the table size.",
    "キャッシュを入れると速くなるが、古い値が返る条件を説明できないなら入れないほうがましだ。",
    "The train arrives at platform four, waits for two minutes, and leaves again without "
    "anyone stepping off.",
    "SELECT user_id, count(*) AS orders FROM purchases WHERE created_at >= '2024-01-01' "
    "GROUP BY user_id HAVING count(*) > 3 ORDER BY orders DESC;",
    "How much memory does a browser tab need before the operating system starts swapping "
    "pages to disk?",
    "山道は途中から舗装が切れ、そこから先は車を降りて歩くことになった。",
    "Most of the cost in this pipeline is not arithmetic but memory bandwidth, so the fastest "
    "kernel is the one that reads each byte once.",
    "型が通ることと仕様を満たすことは別で、前者は後者のごく一部しか保証しない。",
    "He kept a notebook of failed experiments, and the notebook turned out to be more useful "
    "than the record of successes.",
    "手紙は三週間かかって届き、書かれていた予定はすでに過ぎていた。",
    "The specification says the field is optional, but every implementation in the wild "
    "refuses the message when it is missing.",
    "type Result<T> = { ok: true; value: T } | { ok: false; error: string };\n"
    "\n"
    "const unwrap = <T,>(r: Result<T>): T => {\n"
    "  if (!r.ok) throw new Error(r.error);\n"
    "  return r.value;\n"
    "};",
    "Sorting a nearly sorted array with insertion sort is fast, while quicksort on the same "
    "input hits its worst case if the pivot is chosen badly.",
    "並行処理の不具合は再現しないことが多いので、観測を足すより不変条件を書き出すほうが早い。",
    "When the power came back the clocks were all wrong, and it took the rest of the evening "
    "to set them again.",
    "台所の窓から見えるのは隣家の壁と、その上に伸びた一本の柿の木だけだった。",
    "Can a cache be correct if two writers update the same key at the same moment without any "
    "coordination?",
    "impl Counter {\n"
    "    fn increment(&mut self, key: &str) {\n"
    "        *self.counts.entry(key.to_string()).or_insert(0) += 1;\n"
    "    }\n"
    "}",
    "The garden was small, but the wall behind it caught the afternoon sun and kept the herbs "
    "alive well into autumn.",
    "この設定を変えると何が壊れる可能性がありますか。影響範囲を先に知りたいのですが。",
    "A test that never fails is not evidence that the code works; it is evidence that the "
    "test does not look at anything.",
    "浮動小数点の比較を等号で書くと、計算の順序が変わっただけで結果が食い違う。",
    "Reading a file line by line uses less memory than loading it whole, and the difference "
    "stops mattering once the file fits in the page cache.",
    "祖母は同じ話を何度もしたが、細部は毎回少しずつ違っていた。",
    "The meeting ended without a decision, so the same three options were carried over to the "
    "following week.",
    "$ git log --oneline --since='2 weeks ago' -- src/ | head -20\n$ git show --stat HEAD~3",
    "Where does the extra latency come from when a request crosses a network boundary that "
    "used to be a function call?",
    "測定の前に何を測るかを決めておかないと、出た数字に合う説明を後から作ってしまう。",
    "Numerical code often looks simple and behaves badly, because the failure shows up as a "
    "slightly wrong number rather than a crash.",
    "雪は夜のうちに止み、朝には屋根の上だけが白く残っていた。",
    "The ship left the harbour at dawn with a load of timber and returned eleven days later "
    "carrying salt and dried fish.",
    "依存を一つ足すたびに、更新の判断を他人へ預ける範囲が少しずつ広がる。",
    "An index makes reads faster and writes slower, and a table with many indexes can spend "
    "most of its time maintaining them.",
    "If the process is restarted between the write and the rename, which of the two files "
    "does the reader see?",
    "The old bridge was replaced twice, and only the stone footings on the north side belong "
    "to the original structure.",
    "Documentation written after the fact tends to describe what the author remembers, not "
    "what the program actually does.",
)
