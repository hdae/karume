"""`anima.resolution` — the spelling of resolutions (#23).

Small functions used only at the entry of the reference scripts, but of the kind where **a
breakage raises nothing at run time and only the numbers silently drift**: reading `1344x768` as
`768x1344` keeps the element count and the types, and only the reference fixture becomes a
transposed picture. The round-trip identity and the axis order are pinned here.
"""

from __future__ import annotations

import pytest

from anima.resolution import (
    GRANULARITY,
    format_resolution,
    parse_resolution,
    resolution_meta,
)


class TestParseResolution:
    def test_wxh_keeps_the_axis_order(self):
        """MUST: width first, height second (the spelling's order). Swapped, the element count
        still matches."""
        assert parse_resolution("1344x768") == (1344, 768)
        assert parse_resolution("768x1344") == (768, 1344)

    def test_a_bare_number_is_the_square_shorthand(self):
        assert parse_resolution("512") == (512, 512)

    def test_uppercase_x_is_accepted(self):
        assert parse_resolution("1024X1024") == (1024, 1024)

    @pytest.mark.parametrize("text", ["", "1344*768", "1344x", "x768", "1344x768x1", "-512", "1e3"])
    def test_a_malformed_spelling_is_refused(self, text):
        """Input that is not a valid spelling does not fall back to the default (a typo would
        silently run another shape)."""
        with pytest.raises(ValueError, match="WxH"):
            parse_resolution(text)

    @pytest.mark.parametrize("text", ["1352x768", "768x1350", "0"])
    def test_a_side_off_the_grid_is_refused(self, text):
        """patchify cannot be built for a shape whose latent sides are not divisible by the patch
        size 2."""
        with pytest.raises(ValueError, match=str(GRANULARITY)):
            parse_resolution(text)


class TestFormatResolution:
    def test_a_square_uses_the_shorthand(self):
        """MUST: a square uses the shorthand (the directory names of existing fixtures do not
        change by a single character)."""
        assert format_resolution(1024, 1024) == "1024"

    def test_a_non_square_spells_both_sides(self):
        assert format_resolution(1344, 768) == "1344x768"

    @pytest.mark.parametrize("text", ["512", "1024", "1344x768", "896x1152"])
    def test_the_round_trip_is_the_identity(self, text):
        assert format_resolution(*parse_resolution(text)) == text


class TestResolutionMeta:
    def test_a_square_keeps_the_int_field(self):
        """MUST: a square's `resolution` stays an int (readers take this field of `tiling.json`
        as a number)."""
        assert resolution_meta(1024, 1024) == {"resolution": 1024, "width": 1024, "height": 1024}

    def test_a_non_square_spells_the_field(self):
        """There is no single side, so the spelling goes in. The source of truth for the size is
        width / height."""
        assert resolution_meta(1344, 768) == {
            "resolution": "1344x768",
            "width": 1344,
            "height": 768,
        }
