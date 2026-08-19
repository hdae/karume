"""成果物の transactional な公開 — staging へ作り、通ってから final へ据える（ADR 0052）。

書き手（配布形の組み立て・系列ディレクトリの書き出し・単体 export）が違っても規律は 1 つ:
**final は最後の rename まで 1 バイトも触らない**。書き込みの途中で落ちても、検証門が
拒否しても、Ctrl-C でも、正規 path には前回の last-known-good がそのまま残り、「書けたが
読めない」資産も「新旧が混ざった」資産も現れない。

面は 2 つだけ:

- {@link staged_publication} — 席の用意から据え替えまでの全体。`with` の本体が「書き込み +
  検証門」で、例外なく抜けたときにだけ据わる。呼び手は席の綴りも後片付けも持たない。
- {@link swap_into_place} — 据え替えだけ。既に埋まった席を持っている呼び手向け。

作業席（`<final の名前>.staging`）と退避席（同 `.old`）は**必ず final と同じ親**に作る —
rename が原子的なのは同一ファイルシステム内だけで、`/tmp` などへ逃がすと据え替えが跨デバイス
コピーへ落ちて原子性ごと消える。どちらも**固定名**で、中断が残した席は次の実行が黙って捨てる
（一意名で撒くと、拾い直せない残骸が実行のたびに溜まる）。
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

#: 組み立て中の成果物を置く席（final と同じ親・固定名）。
STAGING_SUFFIX = ".staging"

#: 据え替えの直前まで last-known-good を残す席（同上）。
SUPERSEDED_SUFFIX = ".old"


class ArtifactSwapError(OSError):
    """据え替えの rename が失敗した（原因の OSError は `__cause__` に連鎖する）。

    OSError を継承するのは、呼び手にとってこれが I/O 故障そのものだから — 素の `os.replace`
    を呼んでいた頃の `except OSError` が意味を変えずにそのまま効く。
    """


def _discard(path: Path) -> None:
    """`path` を（在れば）丸ごと消す — 作業席・退避席の後始末。

    中断が残した席を踏み直すと、書き手が計画していないファイルが新しい成果物へ黙って混ざる。
    """
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _reclaim_superseded(final: Path, superseded: Path) -> None:
    """退避席を空にする — ただし last-known-good なら捨てずに final へ戻す。

    退避席だけが在って final が無いのは、前回が rename 2 回の**間**で落ちた形。その退避席は
    last-known-good そのものなので、捨てると「手元の成果物が完全消失した」状態から次の実行が
    始まる（final が在るときの退避席は、据え替えを終えた実行が消し損ねたただの残骸）。
    """
    if superseded.exists() and not final.exists():
        os.replace(superseded, final)
    _discard(superseded)


def swap_into_place(staged: Path, final: Path) -> None:
    """埋まった作業席 `staged` を `final` へ**丸ごと**据える（元の中身は 1 つも引き継がない）。

    非空ディレクトリの上へは rename できないので、ディレクトリの据え替えは退避 → 昇格の 2 段に
    なる。2 段の間で落ちると final は**不在**になるが、不在は読み手が確実に検出できる — 作って
    はならないのは「静かに読めてしまう新旧混成」で、この手順はそれを構造的に作れない。昇格に
    失敗したら退避席から戻すので、呼び手の手元は last-known-good のまま残る。

    ファイル（と不在の final）は `os.replace` 1 回で据わる — 退避席を作らない分だけ強く、
    final が不在になる瞬間そのものが無い。

    作業席は消さない — 席の持ち主は呼び手（{@link staged_publication} 経由なら向こうが捨てる）。
    """
    superseded = final.with_name(final.name + SUPERSEDED_SUFFIX)
    _reclaim_superseded(final, superseded)
    if not final.is_dir():
        try:
            os.replace(staged, final)
        except OSError as error:
            raise ArtifactSwapError(f"{final} への据え替え（rename）に失敗した: {error}") from error
        return
    try:
        os.replace(final, superseded)
    except OSError as error:
        # os.replace は原子的 — 失敗しても final は無傷のまま残る。
        raise ArtifactSwapError(f"{final} の退避（rename）に失敗した: {error}") from error
    try:
        os.replace(staged, final)
    except OSError as error:
        # 唯一の正常な成果物が退避席にしか無い状態で止めない。
        os.replace(superseded, final)
        raise ArtifactSwapError(f"{final} への据え替え（rename）に失敗した: {error}") from error
    _discard(superseded)


@contextmanager
def staged_publication(final: Path) -> Iterator[Path]:
    """作業席を渡し、`with` を**例外なく**抜けたときにだけ `final` へ据える。

    本体が「書き込み + 検証門」の全部で、どこで落ちても（Ctrl-C でも）作業席だけが消えて
    final は 1 バイトも変わらない。門を本体の中に置ける形なので、「書く → 検証する → その
    検証を通った値から更に書き足す」順序も同じ席の中で組める。

    MUST: 席の外に掛かる検査（呼び手が用意する全体の前提）は `with` へ入る**前**に済ませる —
    中で落とすと、捨てるだけとはいえ数 GB を並べ切ってからの破棄になる。

    作業席そのものは**作らない** — ディレクトリが要る書き手は自分で mkdir し、1 ファイルを
    書く書き手はその path へ直接書く（どちらが要るかは書き手しか知らない）。
    """
    staging = final.with_name(final.name + STAGING_SUFFIX)
    superseded = final.with_name(final.name + SUPERSEDED_SUFFIX)
    _discard(staging)
    _reclaim_superseded(final, superseded)
    try:
        yield staging
        swap_into_place(staging, final)
    except BaseException:
        _discard(staging)
        raise
