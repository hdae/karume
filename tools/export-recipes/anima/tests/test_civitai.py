"""The civitai intake (`anima.civitai`) — parsing the target, naming, the sha comparison and the
shape of the record.

Nothing goes to the network (`urllib.request.urlopen` is replaced). The responses are cut down to
the essentials from the real ones (`GET /api/v1/model-versions/2983680` and
`GET /api/v1/models/2544636`, measured 2026-09-01), and **a fake of a few dozen bytes is enough for
the 4GB body** — what this observes is not the contents of the bytes but that nothing is put under
the finished name unless the sha matches.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from pathlib import Path

import pytest

from anima import civitai

#: The fake bytes of the body, and the spelling the API returns (upper case — checks that the
#: comparison casefolds).
FILE_BODY = b"weights-of-the-checkpoint"
FILE_SHA256 = hashlib.sha256(FILE_BODY).hexdigest()

DOWNLOAD_URL = "https://civitai.com/api/download/models/2983680?fileId=2863158"

VERSION_RESPONSE = {
    "id": 2983680,
    "modelId": 2544636,
    "name": "v1.0(base 1.0)",
    "baseModel": "Anima",
    "description": None,
    "usageControl": "Download",
    "air": "urn:air:anima:checkpoint:civitai:2544636@2983680",
    "files": [
        {
            "name": "waiANIMA_v10Base10_txt.safetensors",
            "sizeKB": 1164194.4296875,
            "type": "Text Encoder",
            "hashes": {
                "SHA256": "CD2A512003E2F9F3CD3C32A9C3573F820BB28C940F73C57B1DDAA983D9223EBA"
            },
            "primary": False,
            "downloadUrl": "https://civitai.com/api/download/models/2983680?fileId=2863150",
        },
        {
            "name": "waiANIMA_v10Base10.safetensors",
            "sizeKB": 4084212.8671875,
            "type": "Model",
            "hashes": {"SHA256": FILE_SHA256.upper()},
            "primary": True,
            "downloadUrl": DOWNLOAD_URL,
        },
        {
            "name": "qwen_image_vae.safetensors",
            "sizeKB": 247857.662109375,
            "type": "VAE",
            "hashes": {
                "SHA256": "A70580F0213E67967EE9C95F05BB400E8FB08307E017A924BF3441223E023D1F"
            },
            "primary": False,
            "downloadUrl": "https://civitai.com/api/download/models/2983680?type=VAE",
        },
    ],
}

MODEL_RESPONSE = {
    "id": 2544636,
    "name": "WAI-ANIMA",
    "type": "Checkpoint",
    "description": "<p>WAI-ANIMA</p>",
    "allowNoCredit": True,
    "allowCommercialUse": ["Image", "RentCivit"],
    "allowDerivatives": True,
    "allowDifferentLicense": True,
    "modelVersions": [
        {
            "id": 2983680,
            "name": "v1.0(base 1.0)",
            "files": [{"name": "waiANIMA_v10Base10.safetensors", "primary": True, "type": "Model"}],
        },
        {
            "id": 2859702,
            "name": "PW3",
            "files": [{"name": "waiANIMA_pw3.safetensors", "primary": True, "type": "Model"}],
        },
    ],
}


class _FakeResponse:
    """The minimal form of what `urlopen` returns (usable with `with`, readable in pieces with
    `read(size)`)."""

    def __init__(self, payload: bytes) -> None:
        self._buffer = io.BytesIO(payload)

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


class _FakeNetwork:
    """URL prefix → body. Records the URLs called (we also want to observe that **no call**
    happens)."""

    def __init__(self, bodies: dict[str, bytes]) -> None:
        self.bodies = bodies
        self.calls: list[str] = []

    def __call__(self, request: object) -> _FakeResponse:
        url = str(getattr(request, "full_url", request))
        self.calls.append(url)
        for prefix, body in self.bodies.items():
            if url.startswith(prefix):
                return _FakeResponse(body)
        raise AssertionError(f"unexpected GET: {url}")


@pytest.fixture
def network(monkeypatch: pytest.MonkeyPatch) -> _FakeNetwork:
    fake = _FakeNetwork(
        {
            f"{civitai.API_ROOT}/model-versions/2983680": json.dumps(VERSION_RESPONSE).encode(),
            f"{civitai.API_ROOT}/models/2544636": json.dumps(MODEL_RESPONSE).encode(),
            DOWNLOAD_URL: FILE_BODY,
        }
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


@pytest.fixture
def token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(civitai.TOKEN_ENV, "secret-token")


class TestParseAir:
    """An AIR is one word that includes "which version" — nothing ambiguous is accepted."""

    def test_it_reads_ecosystem_type_and_ids(self) -> None:
        air = civitai.parse_air("urn:air:anima:checkpoint:civitai:2544636@2983680")

        assert (air.ecosystem, air.type) == ("anima", "checkpoint")
        assert (air.model_id, air.version_id) == (2544636, 2983680)

    def test_it_rebuilds_the_urn_for_comparison(self) -> None:
        """It is compared with the value the API issues, so what was read can be turned back into
        the original spelling."""
        text = "urn:air:anima:checkpoint:civitai:2544636@2983680"

        assert civitai.parse_air(text).urn == text

    def test_it_refuses_an_air_without_a_version(self) -> None:
        """Without a version, "the latest" would be picked arbitrarily (upstream ships unordered
        lines side by side)."""
        with pytest.raises(SystemExit, match="Specify the version too"):
            civitai.parse_air("urn:air:anima:checkpoint:civitai:2544636")

    def test_it_refuses_a_source_other_than_civitai(self) -> None:
        with pytest.raises(SystemExit, match="A source other than civitai"):
            civitai.parse_air("urn:air:anima:checkpoint:huggingface:2544636@2983680")

    @pytest.mark.parametrize(
        "text",
        [
            "urn:air:anima:checkpoint:civitai:wai@2983680",
            "urn:air:civitai:2544636@2983680",
            "https://civitai.com/models/2544636",
            "",
        ],
    )
    def test_it_refuses_a_malformed_air(self, text: str) -> None:
        with pytest.raises(SystemExit, match="Cannot be read as an AIR"):
            civitai.parse_air(text)


class TestParseUrl:
    """A model page URL does not always carry a version (the form without one goes to guidance
    mode)."""

    def test_it_reads_the_version_from_the_query(self) -> None:
        target = civitai.parse_url(
            "https://civitai.com/models/2544636/wai-anima?modelVersionId=2983680"
        )

        assert (target.model_id, target.version_id) == (2544636, 2983680)

    def test_it_leaves_the_version_unset_when_the_url_has_none(self) -> None:
        target = civitai.parse_url("https://civitai.com/models/2544636")

        assert (target.model_id, target.version_id) == (2544636, None)

    @pytest.mark.parametrize(
        "text",
        [
            "https://civitai.com/models/2544636",
            "https://civitai.com/models/2544636/",
            "https://civitai.com/models/2544636/wai-anima",
            "https://www.civitai.com/models/2544636/wai-anima/",
        ],
    )
    def test_it_accepts_the_slug_being_present_or_absent(self, text: str) -> None:
        assert civitai.parse_url(text).model_id == 2544636

    def test_it_refuses_another_host(self) -> None:
        with pytest.raises(SystemExit, match=r"Not a civitai\.com URL"):
            civitai.parse_url("https://example.com/models/2544636")

    def test_it_refuses_a_page_that_is_not_a_model(self) -> None:
        with pytest.raises(SystemExit, match="Cannot be read as a model page URL"):
            civitai.parse_url("https://civitai.com/user/WAI0731")

    def test_it_refuses_a_non_numeric_version(self) -> None:
        with pytest.raises(SystemExit, match="modelVersionId is not a number"):
            civitai.parse_url("https://civitai.com/models/2544636?modelVersionId=latest")


class TestDeriveModelName:
    """The distribution name is derived mechanically from upstream's own naming (ADR 0077 — the
    version notation itself is not normalized)."""

    @pytest.mark.parametrize(
        ("model_name", "version_name", "expected"),
        [
            ("WAI-ANIMA", "v1.0(base 1.0)", "anima-wai-v1.0"),
            ("copycat-anima", "20260610", "anima-copycat-20260610"),
            ("WAI-ANIMA", "PW3", "anima-wai-pw3"),
        ],
    )
    def test_it_reproduces_the_names_already_distributed(
        self, model_name: str, version_name: str, expected: str
    ) -> None:
        assert civitai.derive_model_name(model_name, version_name) == expected

    def test_it_drops_a_parenthesised_note_from_the_version(self) -> None:
        """The parenthesized part is a note on the upstream page and is not needed to identify the
        version."""
        assert civitai.derive_model_name("Foo", "v2 (fp16 pruned)") == "anima-foo-v2"

    def test_it_squeezes_separators_instead_of_stacking_them(self) -> None:
        assert civitai.derive_model_name("Foo // Bar", "v1 @ final") == "anima-foo-bar-v1-final"

    @pytest.mark.parametrize(
        ("model_name", "version_name"),
        [("Anima", "v1.0"), ("WAI-ANIMA", "(base 1.0)"), ("", "v1.0")],
    )
    def test_it_refuses_when_a_half_comes_out_empty(
        self, model_name: str, version_name: str
    ) -> None:
        """A model name that is only the family name, or a version name that is only parentheses,
        gives nothing to derive a name from."""
        with pytest.raises(SystemExit, match="Cannot derive a name"):
            civitai.derive_model_name(model_name, version_name)

    def test_it_refuses_a_name_outside_the_accepted_set(self) -> None:
        with pytest.raises(SystemExit, match="has characters that are not allowed"):
            civitai._assert_name("anima-wai v1.0")

    @pytest.mark.parametrize("name", [".hidden", "..", ".x"])
    def test_it_refuses_a_leading_dot(self, name: str) -> None:
        """The accepted set of `--name` is the same as `karume.dist`'s — a leading dot is a spelling
        the build step rejects.

        If this were wider, a wrong `--name` would make it into `civitai.json` and the "command to
        run next" guidance, and fail only in `karume dist` tens of minutes later.
        """
        with pytest.raises(SystemExit, match="has characters that are not allowed"):
            civitai._assert_name(name)

    def test_it_refuses_a_trailing_newline(self) -> None:
        """Python's `$` also matches right before a trailing newline — rejected, like the source of
        truth `MODEL_NAME_RE`."""
        with pytest.raises(SystemExit, match="has characters that are not allowed"):
            civitai._assert_name("anima-wai\n")

    def test_it_refuses_the_shared_seat(self) -> None:
        """`shared` is the location of the distribution repository's shared files (`SHARED_DIRNAME`
        of `karume.dist`)."""
        with pytest.raises(SystemExit, match="collides with the"):
            civitai._assert_name("shared")

    def test_it_gates_the_derived_name_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Even if the normalization misses some upstream spelling, the accepted-set check applies
        last."""
        monkeypatch.setattr(civitai, "_slug", lambda text: "wai v1.0")

        with pytest.raises(SystemExit, match="has characters that are not allowed"):
            civitai.derive_model_name("WAI-ANIMA", "v1.0")


class TestAssertBasename:
    """A file name returned by the API is confined to one level below the intake directory (the
    path-join rules must not take it outside)."""

    @pytest.mark.parametrize(
        "name",
        [
            "../x.safetensors",
            "/abs/x.safetensors",
            "a/b.safetensors",
            "a\\b.safetensors",
            "",
            ".",
            "..",
        ],
    )
    def test_it_refuses_a_name_that_is_not_a_basename(self, name: str) -> None:
        with pytest.raises(SystemExit, match="is not a basename"):
            civitai._assert_basename(name)

    @pytest.mark.parametrize("name", ["D:x.safetensors", "C:weights.safetensors"])
    def test_it_refuses_a_windows_drive_relative_name(self, name: str) -> None:
        """`D:x` contains no separator, yet on Windows it is a drive-relative name.

        `PureWindowsPath("C:/intake/civitai-123") / "D:x"` becomes `D:x`, and the leading part of
        the join is discarded entirely (4GB lands directly in the working directory of another
        drive).
        """
        with pytest.raises(SystemExit, match="is not a basename"):
            civitai._assert_basename(name)

    def test_it_passes_an_ordinary_upstream_name_through(self) -> None:
        assert civitai._assert_basename("waiANIMA_v10Base10.safetensors") == (
            "waiANIMA_v10Base10.safetensors"
        )


class TestSelectFile:
    """The bundled Text Encoder / VAE are not taken (base's are shared)."""

    def test_it_picks_the_primary_file(self) -> None:
        chosen = civitai.select_file(VERSION_RESPONSE["files"])

        assert chosen["name"] == "waiANIMA_v10Base10.safetensors"

    def test_it_falls_back_to_a_single_model_file(self) -> None:
        files = [{"name": "a.safetensors", "type": "Model"}, {"name": "b.png", "type": "Archive"}]

        assert civitai.select_file(files)["name"] == "a.safetensors"

    def test_it_ignores_a_primary_that_is_not_a_model_file(self) -> None:
        """Upstream does not guarantee that primary = the Model type (ADR 0088 addendum)."""
        files = [
            {"name": "vae.safetensors", "type": "VAE", "primary": True},
            {"name": "body.safetensors", "type": "Model", "primary": False},
        ]

        assert civitai.select_file(files)["name"] == "body.safetensors"

    def test_it_stops_when_no_model_file_is_present(self) -> None:
        files = [{"name": "vae.safetensors", "type": "VAE", "primary": True}]

        with pytest.raises(SystemExit, match="Cannot identify the main file"):
            civitai.select_file(files)

    def test_it_stops_when_the_body_cannot_be_told_apart(self) -> None:
        files = [
            {"name": "a.safetensors", "type": "Model"},
            {"name": "b.safetensors", "type": "Model"},
        ]

        with pytest.raises(SystemExit, match="Cannot identify the main file"):
            civitai.select_file(files)


class TestDownload:
    """The file goes under the finished name only once the sha matches (a truncated download is not
    read as fetched)."""

    def test_it_writes_the_file_when_the_hash_matches(
        self, tmp_path: Path, network: _FakeNetwork, token: None
    ) -> None:
        dest = tmp_path / "waiANIMA_v10Base10.safetensors"

        civitai.download(DOWNLOAD_URL, dest, FILE_SHA256.upper())

        assert dest.read_bytes() == FILE_BODY
        assert not dest.with_name(f"{dest.name}.part").exists()

    def test_it_keeps_the_part_file_when_the_hash_differs(
        self, tmp_path: Path, network: _FakeNetwork, token: None
    ) -> None:
        """The downloaded bytes are kept — a person decides whether upstream replaced the file or it
        broke."""
        dest = tmp_path / "waiANIMA_v10Base10.safetensors"

        with pytest.raises(SystemExit, match="sha256 differs from the API's value"):
            civitai.download(DOWNLOAD_URL, dest, "0" * 64)

        assert not dest.exists()
        assert dest.with_name(f"{dest.name}.part").read_bytes() == FILE_BODY

    def test_it_masks_the_token_in_what_it_prints(
        self, tmp_path: Path, network: _FakeNetwork, token: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        civitai.download(DOWNLOAD_URL, tmp_path / "weights.safetensors", FILE_SHA256)

        printed = capsys.readouterr().out
        assert "secret-token" not in printed
        assert "token=***" in printed
        assert "token=secret-token" in network.calls[0]

    def test_it_skips_a_file_that_is_already_there(
        self, tmp_path: Path, network: _FakeNetwork, token: None
    ) -> None:
        """A rerun does not fetch the 4GB again (a matching sha is the definition of "fetched")."""
        dest = tmp_path / "weights.safetensors"
        dest.write_bytes(FILE_BODY)

        civitai.download(DOWNLOAD_URL, dest, FILE_SHA256)

        assert network.calls == []

    def test_it_stops_when_a_different_file_sits_on_the_name(
        self, tmp_path: Path, network: _FakeNetwork, token: None
    ) -> None:
        dest = tmp_path / "weights.safetensors"
        dest.write_bytes(b"something else")

        with pytest.raises(SystemExit, match="A different file is in place under the same name"):
            civitai.download(DOWNLOAD_URL, dest, FILE_SHA256)

        assert network.calls == []

    def test_it_asks_for_the_token_before_going_to_the_network(
        self, tmp_path: Path, network: _FakeNetwork, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(civitai.TOKEN_ENV, raising=False)

        with pytest.raises(SystemExit, match=civitai.TOKEN_ENV):
            civitai.download(DOWNLOAD_URL, tmp_path / "weights.safetensors", FILE_SHA256)

        assert network.calls == []


class TestResolveAir:
    """The value the API issues is the source of truth for the AIR — if it disagrees with the one
    we built, say so, then take the API's."""

    def test_it_reports_a_disagreement_and_takes_the_served_value(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        requested = civitai.parse_air("urn:air:sd1:model:civitai:2544636@2983680")

        resolved = civitai._resolve_air(VERSION_RESPONSE, requested)

        assert resolved == VERSION_RESPONSE["air"]
        assert "warning: the AIR differs from the requested one" in capsys.readouterr().out

    def test_it_falls_back_to_the_requested_urn_when_the_api_has_none(self) -> None:
        requested = civitai.parse_air("urn:air:anima:checkpoint:civitai:2544636@2983680")

        assert civitai._resolve_air({}, requested) == requested.urn


class TestFetchCheckpoint:
    """One intake — places only the one body file and `civitai.json`."""

    @pytest.fixture
    def taken(self, tmp_path: Path, network: _FakeNetwork, token: None) -> Path:
        return civitai.fetch_checkpoint(2544636, 2983680, out=tmp_path)

    def test_it_places_the_upstream_file_under_the_version_directory(
        self, tmp_path: Path, taken: Path
    ) -> None:
        assert taken == tmp_path / "civitai-2983680" / "waiANIMA_v10Base10.safetensors"
        assert taken.read_bytes() == FILE_BODY

    def test_it_leaves_only_the_body_and_the_record(self, tmp_path: Path, taken: Path) -> None:
        """The bundled Text Encoder / VAE are not taken, since base's are shared."""
        assert sorted(path.name for path in taken.parent.iterdir()) == [
            "civitai.json",
            "waiANIMA_v10Base10.safetensors",
        ]

    def test_it_records_the_provenance_the_machine_owns(self, taken: Path) -> None:
        record = json.loads((taken.parent / civitai.PROVENANCE_FILE).read_text(encoding="utf-8"))

        assert sorted(record) == [
            "air",
            "base_model",
            "derived_name",
            "descriptions",
            "fetched_at",
            "file",
            "model_id",
            "model_name",
            "permissions",
            "version_id",
            "version_name",
        ]
        assert record["derived_name"] == "anima-wai-v1.0"
        assert record["air"] == "urn:air:anima:checkpoint:civitai:2544636@2983680"
        assert record["permissions"] == {
            "allowNoCredit": True,
            "allowCommercialUse": ["Image", "RentCivit"],
            "allowDerivatives": True,
            "allowDifferentLicense": True,
            "usageControl": "Download",
        }
        assert record["descriptions"] == {"model": "<p>WAI-ANIMA</p>", "version": None}

    def test_it_records_the_hash_in_lower_case(self, taken: Path) -> None:
        """Both the comparison and the distribution card use lower case (the API returns upper
        case)."""
        record = json.loads((taken.parent / civitai.PROVENANCE_FILE).read_text(encoding="utf-8"))

        assert record["file"]["sha256"] == FILE_SHA256

    def test_it_keeps_the_token_out_of_the_record(self, taken: Path) -> None:
        record = json.loads((taken.parent / civitai.PROVENANCE_FILE).read_text(encoding="utf-8"))

        assert record["file"]["download_url"] == DOWNLOAD_URL
        assert "token" not in record["file"]["download_url"]

    def test_it_hands_over_to_the_next_step(
        self,
        tmp_path: Path,
        network: _FakeNetwork,
        token: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        civitai.fetch_checkpoint(2544636, 2983680, out=tmp_path)

        printed = capsys.readouterr().out
        assert "python -m anima.single_file" in printed
        assert "anima-diffusers/anima-wai-v1.0" in printed

    def test_it_refuses_a_file_name_that_escapes_the_version_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, token: None
    ) -> None:
        """An escaping name fails before the join — the weights alone must not end up outside the
        intake directory."""
        escaping = json.loads(json.dumps(VERSION_RESPONSE))
        for entry in escaping["files"]:
            if entry["primary"]:
                entry["name"] = "../escaped.safetensors"
        fake = _FakeNetwork(
            {
                f"{civitai.API_ROOT}/model-versions/2983680": json.dumps(escaping).encode(),
                f"{civitai.API_ROOT}/models/2544636": json.dumps(MODEL_RESPONSE).encode(),
            }
        )
        monkeypatch.setattr(urllib.request, "urlopen", fake)

        with pytest.raises(SystemExit, match="is not a basename"):
            civitai.fetch_checkpoint(2544636, 2983680, out=tmp_path)

        assert sorted(path.name for path in tmp_path.rglob("*")) == ["civitai-2983680"]

    def test_it_refuses_a_version_id_that_is_not_an_integer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, token: None
    ) -> None:
        """The first level of the intake directory is an upstream value too — an escaping id creates
        nothing outside `out`.

        `mkdir(parents=True)` creates the parents including any `..` on the way, so letting it
        through would put both the weights and `civitai.json` outside `out` (here, directly under
        `tmp_path`).
        """
        escaping = json.loads(json.dumps(VERSION_RESPONSE))
        escaping["id"] = "/../../escape"
        fake = _FakeNetwork(
            {
                f"{civitai.API_ROOT}/model-versions/2983680": json.dumps(escaping).encode(),
                f"{civitai.API_ROOT}/models/2544636": json.dumps(MODEL_RESPONSE).encode(),
            }
        )
        monkeypatch.setattr(urllib.request, "urlopen", fake)
        out = tmp_path / "intake"

        with pytest.raises(ValueError, match="invalid literal for int"):
            civitai.fetch_checkpoint(2544636, 2983680, out=out)

        assert list(tmp_path.iterdir()) == []

    def test_it_overrides_the_derived_name_when_asked(
        self, tmp_path: Path, network: _FakeNetwork, token: None
    ) -> None:
        taken = civitai.fetch_checkpoint(2544636, 2983680, out=tmp_path, name="anima-wai-custom")
        record = json.loads((taken.parent / civitai.PROVENANCE_FILE).read_text(encoding="utf-8"))

        assert record["derived_name"] == "anima-wai-custom"


class TestGuidanceMode:
    """When no version was chosen, it prints only what is needed to choose again and downloads
    nothing."""

    def test_it_lists_the_versions_and_downloads_nothing(
        self,
        network: _FakeNetwork,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            "sys.argv", ["civitai", "--url", "https://civitai.com/models/2544636/wai-anima"]
        )

        civitai.main()

        printed = capsys.readouterr().out
        assert "2983680  v1.0(base 1.0)  waiANIMA_v10Base10.safetensors" in printed
        assert "2859702  PW3  waiANIMA_pw3.safetensors" in printed
        assert network.calls == [f"{civitai.API_ROOT}/models/2544636"]
