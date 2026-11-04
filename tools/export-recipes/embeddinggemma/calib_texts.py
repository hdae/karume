"""The **calibration corpus** of EmbeddingGemma-300m (48 texts that produce the activations GPTQ /
AWQ see).

The calibrated rounding of `karume.quant_calib` re-chooses the rounding targets from "the
activations that actually flow into that layer", so the character of the corpus becomes the bias of
the rounding as is. This is the only input read by the calibrated configs of `measure_quant.py`
({@link embeddinggemma.measure_quant.CALIB_CONFIGS}), and it is chosen by three rules:

- **Keep it apart from the evaluation texts** — not a single text overlaps
  {@link embeddinggemma.export.GOLDEN_CASES} (= the 5 cases that form the denominator of every
  cosine). The topics do not overlap either (nothing on "the capital of France" and no pangram like
  `bare`). With an overlap, the evaluation runs on texts the calibration has already seen, and the
  cosine measures **leakage, not the quality of the calibration**.
- **Follow how retrieval uses the model** — EG has asymmetric query / document prompt prefixes
  (`prompts` in `config_sentence_transformers.json` is the source of truth). The calibration inputs
  carry the same prefixes, mixing short query-style questions, document-style explanations and
  plain texts without a prefix. Dropping the prefixes would round without ever seeing the
  activations of the token sequence that always comes first in real use.
- **Spread lengths and languages** — from 5–10-word queries to 30–40-word paragraphs. Mostly English
  (the main use of retrieval), with 10 Japanese texts mixed in so that activations of CJK token
  sequences are seen as well.

The split is query 20 / document 20 / no prefix 8, of which 10 are Japanese. No personal names or
real organisations are written (so that the calibration inputs do not pull in the activations of
specific proper nouns).
"""

from __future__ import annotations

#: The 48 calibration inputs (`(prompt kind | None, body)`). The kind is a key of the dict returned
#: by {@link embeddinggemma.export.load_prompts} (`query` / `document`), and `None` means no prefix.
#: They are used in this order from the start (`--calib-limit N` takes **the first N texts**) — the
#: three kinds are interleaved, so a reduced run with a limit still keeps the mix of roles.
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
