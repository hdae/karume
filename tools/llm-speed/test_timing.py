"""壁時計の単位と、最初のtokenを除く速度の勘定を確認する。"""

import pytest
from torch_bench import timing


@pytest.mark.parametrize(
    ("first", "end", "count", "ttft", "rate"),
    [
        (None, 1, 0, None, None),
        (0.25, 0.5, 1, 250, None),
        (0.25, 0.75, 3, 250, 4),
        (0.25, 0.25, 3, 250, None),
    ],
)
def test_delivery_timeline_separates_first_token_and_total_time(first, end, count, ttft, rate):
    result = timing(0, first, end, count)
    assert result == {
        "elapsedMs": end * 1000,
        "generatedTokens": count,
        "ttftMs": ttft,
        "decodeTokensPerSecond": rate,
    }
