"""Calibration corpus of DeBERTa (the text front of SBV2) (48 sentences that produce the
activations GPTQ sees).

The calibrated rounding of `karume.quant_calib` re-picks rounding targets from "the activations
that actually flow through the layer", so the character of the corpus becomes the bias of the
rounding directly. There are two readers, and the key point is that **both see the same 48
sentences** (if the rounding that measured well and the rounding that ships were decided from
different corpora, the measurement would not predict the distribution):

- the i4 series of `deberta.export` (the **shipping path** of calibrated rounding —
  {@link deberta.calib})
- the calibrated configurations of `sbv2.measure_quant` (`CALIB_CONFIGS` — **measuring** quality)

The side that owns the model owns the corpus, so it lives here (`deberta`), and `sbv2` imports it
by name. The selection policy has three points:

- **Keep it disjoint from the evaluation sentences** — not one sentence overlaps the utterances of
  the dump (`meta["text"]` / `meta["bertText"]`) (down to partial matches,
  {@link sbv2.measure_quant.assert_calib_disjoint} fails loudly). With an overlap, SNR / LSD would
  measure "whether the sentences seen in calibration could be reproduced as is", i.e. a leak rather
  than the quality of the calibration. No copy of the evaluation sentences is kept here because
  re-recording the dump would leave only one side stale.
- **Lean towards the read-aloud distribution of real TTS use** — the model is a Japanese char-wwm
  DeBERTa, and what SBV2 feeds it is the Japanese text to be read aloud itself, so the calibration
  uses Japanese prose only. It mixes narration, conversation (questions and answers) and readings
  of numbers and symbols, and keeps the topics apart from one another (leaning towards one topic
  pins the bias of the activations to the shape of that topic).
- **Spread the lengths** — from short sentences (about 5 characters) to long single sentences
  (about 40 characters). GPTQ's `H = Σ XᵀX` is weighted by token count, so with only long
  sentences the activations of long sentences dominate, and with only short ones the activations
  at the start of a sentence dominate (with char-wwm the character count is about the token count).

No proper names of people or real organisations are written (so the calibration input does not
pull up the activations of specific proper nouns — only the distribution of general vocabulary is
wanted).
"""

from __future__ import annotations

#: The 48 calibration sentences (used from the start in this order — `--calib-limit N` takes **the
#: first N sentences**). The order is **mixed** rather than narration → conversation → numbers: a
#: reduced run with a limit keeps the mix of roles (the first 4 sentences alone cover narration /
#: question / number reading).
CALIB_TEXTS: tuple[str, ...] = (
    "春の日差しが川面に散らばり、対岸の桜並木は満開の枝を水の上へ低く伸ばしていた。",
    "ねえ、その話をもう一度だけ聞かせてくれる？",
    "会議は十四時三十分から、第二会議室で開きます。資料は三部ずつ用意してください。",
    "雨は明け方までに上がるでしょう。",
    "「そこは触らないで」と言われたので、あわてて手を引っ込めた。",
    "気温は摂氏二十六度、湿度は七十八パーセント、風速は毎秒三メートルです。",
    "港の倉庫はもう使われておらず、割れた窓から差し込む光だけが床の埃を照らしていた。",
    "えっ、それで本当に大丈夫なんですか？",
    "今朝はよく眠れた。",
    "電話番号は、〇三の一二三四の五六七八です。",
    "山の稜線がゆっくりと暮れていき、谷のほうから鐘の音が二度聞こえた。",
    "すみません、次の停留所で降りたいのですが、こちらで合っていますか？",
    "商品名は二重引用符で囲み、価格はカンマ区切り、末尾にセミコロンを付けてください。",
    "祖父の書斎には背表紙の色が褪せた本が並び、どれも同じ癖のある字で名前が書き込まれていた。",
    "それって、昨日の続きということでいいんですよね？",
    "一九九八年の夏、私は初めて一人で電車に乗り、隣の県まで出かけた。",
    "風が強くなってきた。",
    "もしもし、少し電波が悪いようですが、こちらの声は聞こえていますか？",
    "台所の湯気が窓を白く曇らせ、外の景色は輪郭だけになっていた。",
    "締め切りまで残り三日、原稿は四十枚のうち二十二枚まで進んでいます。",
    "いいえ、そちらではなく、右側の棚の一番上にある箱です。",
    "海沿いの道は夏のあいだだけ賑わい、九月に入ると店の半分は板を打ち付けて冬を待つ。",
    "お先に失礼します。",
    "記号は半角、括弧は全角で統一し、単位との間に空白を入れないでください。",
    "昨日の夜、外で変な音がしませんでしたか？",
    "蝉の声が途切れた午後、庭の木陰だけが妙に静かだった。",
    "一日に二回、朝の八時と夜の九時に薬を飲んでください。",
    "その説明では、どこから手を付ければいいのか分かりません。順番に整理してもらえますか？",
    "列車は長いトンネルを抜け、窓の外は一面の雪になった。",
    "それでお願いします。",
    "縦横比は十六対九、解像度は千九百二十かける千八十を基準にします。",
    "手紙の最後には、来年の春には必ず戻ると書かれていたが、その春が来ることはなかった。",
    "どうしてそんなに急いでいるの？",
    "街灯が一つずつ灯り、濡れた舗道に細長い光が伸びた。",
    "参加者は男性十七名、女性二十三名、合わせて四十名でした。",
    "ちょっと待ってください、今メモを取りますので。",
    "古い時計は毎正時に短く鳴り、その音を合図に家じゅうの誰かが必ず立ち上がった。",
    "空が高い。",
    "この道で合っているのか、少し不安になってきませんか？",
    "三分の二を水で薄め、残りは翌日まで冷蔵庫で寝かせます。",
    "砂浜には流木と貝殻が散らばり、波打ち際だけが絶えず形を変えていた。",
    "さっきの件ですが、担当の方に確認したところ、来週まで待ってほしいとのことでした。",
    "章の見出しは第一章、節はドット区切りで一・一のように書きます。",
    "郵便受けには折り曲げられた新聞が刺さったままだった。",
    "やっぱり、そう思っていたんですね？",
    "谷の奥にある集落へは日に三本のバスしか通わず、最終便を逃すと歩くほかなかった。",
    "湯が沸いた。",
    "それはいつ頃になりそうですか？",
)
