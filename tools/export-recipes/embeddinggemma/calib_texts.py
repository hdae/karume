"""EmbeddingGemma-300m の**校正コーパス**（GPTQ / AWQ が見る活性を作る 48 文）。

`karume.quant_calib` の校正付き丸めは「その層に実際に流れる活性」から丸め先を選び直す方式
なので、コーパスの性格がそのまま丸めの偏りになる。ここは `measure_quant.py` の校正付き構成
（{@link embeddinggemma.measure_quant.CALIB_CONFIGS}）が読む唯一の入力で、選定方針は次の
3 点:

- **評価文と分離する** — {@link embeddinggemma.export.GOLDEN_CASES}（= 全 cosine の分母を
  作る 5 ケース）と 1 文も重ねない。話題も重ねない（「フランスの首都」系と `bare` の
  pangram は 1 文も入れない）。重ねると校正で見た文をそのまま評価する形になり、cosine は
  **校正の質ではなく漏れ**を測る数になる。
- **検索の使われ方へ寄せる** — EG は query / document の非対称なプロンプト接頭辞を持つ
  （`config_sentence_transformers.json` の `prompts` が正本）。校正入力にも同じ接頭辞を
  付け、query 風の短い問い合わせ・document 風の説明文・接頭辞なしの素の文を混ぜる。
  接頭辞を落とすと、実運用で必ず先頭に来るトークン列の活性を 1 度も見ないまま丸めることに
  なる。
- **長さと言語を散らす** — 5〜10 語の問い合わせから 30〜40 語の段落まで。英語中心
  （検索の主用途）に日本語を 10 文だけ混ぜ、CJK のトークン列でも活性を見ておく。

内訳は query 20 / document 20 / 接頭辞なし 8、うち日本語 10。固有の人名・実在組織は
書かない（校正の入力が特定の固有名の活性を引き当てるのを避ける）。
"""

from __future__ import annotations

#: 校正入力の 48 文（`(プロンプト種別 | None, 本文)`）。種別は
#: {@link embeddinggemma.export.load_prompts} が返す辞書のキー（`query` / `document`）で、
#: `None` は接頭辞なし。この順で先頭から使う（`--calib-limit N` は**先頭 N 文**）—
#: 3 種を混ぜて並べるので、上限を掛けた縮小実行でも役割の混合が保たれる。
CALIB_TEXTS: tuple[tuple[str | None, str], ...] = (
    ("query", "how do i rotate log files without restarting the service"),
    (
        "document",
        "Rotating a log file safely means the writer must reopen the path after the rename; "
        "otherwise it keeps writing into the unlinked inode and the new file stays empty.",
    ),
    (None, "The index returns candidates by vector distance, and a second pass reranks them."),
    ("query", "difference between a mutex and a semaphore"),
    (
        "document",
        "A mutex allows exactly one holder at a time, while a counting semaphore admits a "
        "fixed number of holders and is often used to bound concurrency rather than to "
        "protect state.",
    ),
    ("query", "best way to store timestamps in a database"),
    (
        "document",
        "Storing timestamps as an integer count of seconds in UTC avoids ambiguity around "
        "daylight saving transitions, and the local zone is applied only when a value is "
        "displayed.",
    ),
    (
        None,
        "Chunk boundaries decide what can be retrieved at all; a fact split across two chunks "
        "is often found by neither.",
    ),
    ("query", "what causes a memory leak in a long running server process"),
    (
        "document",
        "Memory that is still reachable from a global cache is not garbage, so a process can "
        "grow without bound while the collector reports that nothing is leaking.",
    ),
    ("query", "how to resize an image in the browser before uploading"),
    (
        "document",
        "Downscaling an image before upload cuts transfer time and drops camera metadata, and "
        "drawing the source into a canvas of the target size is enough for most photographs.",
    ),
    ("query", "symptoms of a failing solid state drive"),
    (
        "document",
        "Drives usually fail gradually: read errors are retried, latency rises, and "
        "reallocated sector counts climb long before the controller reports the disk as "
        "unhealthy.",
    ),
    ("query", "recipe for bread with a long cold fermentation"),
    (
        "document",
        "A long cold fermentation develops flavour without extra yeast; the dough rests in "
        "the refrigerator overnight and is shaped straight from the cold.",
    ),
    ("query", "why is my wireless connection slower in the evening"),
    (
        "document",
        "Shared wireless spectrum gets busier in the evening because neighbouring networks "
        "are active at the same time, and the router falls back to lower rates as "
        "interference grows.",
    ),
    (
        None,
        "Two passages with almost the same wording can answer different questions once the "
        "surrounding context is restored.",
    ),
    ("query", "how to read a nutrition label"),
    (
        "document",
        "Nutrition labels list values per serving, and the serving size is chosen by the "
        "manufacturer, so two similar packages can look very different for the same amount "
        "of food.",
    ),
    ("query", "train timetable changes during public holidays"),
    (
        "document",
        "Holiday timetables run on a reduced schedule; the first departure is later, the "
        "evening service ends earlier, and some connections are not guaranteed.",
    ),
    ("query", "how do heat pumps work in cold climates"),
    (
        "document",
        "A heat pump moves heat rather than generating it, so its output falls as the outside "
        "temperature drops, and cold climate units use a larger compressor to compensate.",
    ),
    (
        None,
        "Embedding a query and embedding a passage are the same operation; the prompt prefix "
        "is what tells the model which role it is playing.",
    ),
    ("query", "cheapest way to ship a heavy parcel abroad"),
    (
        "document",
        "Shipping cost depends on volumetric weight as much as on actual weight, and a dense "
        "parcel is often cheaper to send than a large light one.",
    ),
    ("query", "how to prune an apple tree in winter"),
    (
        "document",
        "Winter pruning removes crossing branches and opens the centre of the tree, which "
        "improves air flow and lets the fruit ripen more evenly the following season.",
    ),
    ("query", "return policy for opened electronics"),
    (
        "document",
        "Opened electronics can normally be returned within thirty days if all accessories "
        "are present, though a restocking fee may apply once the seal is broken.",
    ),
    ("query", "how many hours of sleep do teenagers need"),
    (
        "document",
        "Adolescents need more sleep than adults, and a later school start time has been "
        "associated with longer total sleep rather than with later bedtimes.",
    ),
    (
        None,
        "A stopword list that is too aggressive removes the only distinguishing term in a "
        "short query.",
    ),
    ("query", "permit needed to convert a garage into a home office"),
    (
        "document",
        "Converting a garage usually requires approval when the change alters insulation, "
        "ventilation, or parking provision, even if the exterior walls are untouched.",
    ),
    (
        None,
        "Numbers, units, and product codes survive tokenisation poorly, which is why exact "
        "match still complements dense retrieval.",
    ),
    ("query", "ロードバイクのチェーンはどのくらいで交換するべきか"),
    (
        "document",
        "チェーンの伸びは専用のゲージで測るのが確実で、伸びを放置するとスプロケットまで"
        "摩耗が進む。",
    ),
    ("query", "確定申告で医療費控除を受ける条件"),
    (
        "document",
        "医療費控除は生計を一にする家族の分を合算でき、保険金などで補填された額は差し引いて"
        "計算する。",
    ),
    ("query", "赤ちゃんの離乳食はいつから始めるか"),
    (
        "document",
        "離乳食は首が据わり、支えれば座れるようになった頃が目安で、初めは滑らかにすりつぶした"
        "一品から始める。",
    ),
    ("query", "電気ケトルの水垢を落とす方法"),
    (
        "document",
        "電気ケトルの水垢はクエン酸を溶かした水を沸かして放置すると落ちやすく、その後は"
        "数回すすいで臭いを抜く。",
    ),
    (None, "短い問い合わせは語数が少ないぶん、表記の揺れが検索結果に効きやすい。"),
    (
        None,
        "検索の評価では、正解が上位に入るかどうかと、順位そのものの安定性を分けて見る必要がある。",
    ),
)
