"""civitai 取り込み（`anima.civitai`）— 指定のパース・命名・sha 突合・記録の形。

ネットワークへは出ない（`urllib.request.urlopen` を差し替える）。応答は実物
（`GET /api/v1/model-versions/2983680` と `GET /api/v1/models/2544636` の 2026-09-01 実測）から
要点だけ残して縮めたもので、**本体 4GB は数十バイトの偽物で足りる** — ここで観測したいのは
バイト列の中身ではなく、sha が合わない限り完成品の名前へ据えないことだから。
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from pathlib import Path

import pytest

from anima import civitai

#: 本体の偽バイト列と、API が返す綴り（大文字 — 突合が casefold しているかを見る）。
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
    """`urlopen` が返すものの最小形（`with` で使えて `read(size)` で刻める）。"""

    def __init__(self, payload: bytes) -> None:
        self._buffer = io.BytesIO(payload)

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


class _FakeNetwork:
    """URL の前置 → 本文。呼ばれた URL を覚える（**呼ばれない**ことも観測したい）。"""

    def __init__(self, bodies: dict[str, bytes]) -> None:
        self.bodies = bodies
        self.calls: list[str] = []

    def __call__(self, request: object) -> _FakeResponse:
        url = str(getattr(request, "full_url", request))
        self.calls.append(url)
        for prefix, body in self.bodies.items():
            if url.startswith(prefix):
                return _FakeResponse(body)
        raise AssertionError(f"想定外の GET: {url}")


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
    """AIR は「どの版か」まで含んだ 1 語 — 曖昧なものは受けない。"""

    def test_it_reads_ecosystem_type_and_ids(self) -> None:
        air = civitai.parse_air("urn:air:anima:checkpoint:civitai:2544636@2983680")

        assert (air.ecosystem, air.type) == ("anima", "checkpoint")
        assert (air.model_id, air.version_id) == (2544636, 2983680)

    def test_it_rebuilds_the_urn_for_comparison(self) -> None:
        """API 発行値との突合に使うので、読んだものから元の綴りへ戻せる。"""
        text = "urn:air:anima:checkpoint:civitai:2544636@2983680"

        assert civitai.parse_air(text).urn == text

    def test_it_refuses_an_air_without_a_version(self) -> None:
        """版が無いと「最新」を勝手に選ぶことになる（上流は順序の付かない系統を同時に配る）。"""
        with pytest.raises(SystemExit, match="版まで指定する"):
            civitai.parse_air("urn:air:anima:checkpoint:civitai:2544636")

    def test_it_refuses_a_source_other_than_civitai(self) -> None:
        with pytest.raises(SystemExit, match="civitai 以外の source"):
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
        with pytest.raises(SystemExit, match="AIR として読めない"):
            civitai.parse_air(text)


class TestParseUrl:
    """モデルページの URL は版を持つとは限らない（持たない形は案内モードへ回る）。"""

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
        with pytest.raises(SystemExit, match=r"civitai\.com の URL ではない"):
            civitai.parse_url("https://example.com/models/2544636")

    def test_it_refuses_a_page_that_is_not_a_model(self) -> None:
        with pytest.raises(SystemExit, match="モデルページの URL として読めない"):
            civitai.parse_url("https://civitai.com/user/WAI0731")

    def test_it_refuses_a_non_numeric_version(self) -> None:
        with pytest.raises(SystemExit, match="modelVersionId が数値でない"):
            civitai.parse_url("https://civitai.com/models/2544636?modelVersionId=latest")


class TestDeriveModelName:
    """配布名は上流の名乗りから機械で導く（ADR 0077 — 版表記そのものは正規化しない）。"""

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
        """丸括弧の中は上流ページ上の但し書きで、版の識別には要らない。"""
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
        """ファミリ名だけのモデル名や括弧だけの版名は、名前を導く材料になっていない。"""
        with pytest.raises(SystemExit, match="名前を導けない"):
            civitai.derive_model_name(model_name, version_name)

    def test_it_refuses_a_name_outside_the_accepted_set(self) -> None:
        with pytest.raises(SystemExit, match="配布名に使えない文字がある"):
            civitai._assert_name("anima-wai v1.0")

    @pytest.mark.parametrize("name", [".hidden", "..", ".x"])
    def test_it_refuses_a_leading_dot(self, name: str) -> None:
        """`--name` の受理集合は `karume.dist` と同じ — 先頭のドットは組み立て段で落ちる綴り。

        ここが広いと、誤った `--name` は `civitai.json` と「次に叩くコマンド」の案内まで
        載ってから、数十分後の `karume dist` で初めて落ちる。
        """
        with pytest.raises(SystemExit, match="配布名に使えない文字がある"):
            civitai._assert_name(name)

    def test_it_refuses_the_shared_seat(self) -> None:
        """`shared` は配布リポの共有ファイルの席（`karume.dist` の `SHARED_DIRNAME`）。"""
        with pytest.raises(SystemExit, match="共有ファイルの席と衝突する"):
            civitai._assert_name("shared")

    def test_it_gates_the_derived_name_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """正規化が上流の綴りを取りこぼしても、受理集合の門は最後に効く。"""
        monkeypatch.setattr(civitai, "_slug", lambda text: "wai v1.0")

        with pytest.raises(SystemExit, match="配布名に使えない文字がある"):
            civitai.derive_model_name("WAI-ANIMA", "v1.0")


class TestSelectFile:
    """同梱の Text Encoder / VAE は取らない（base 側を共有する）。"""

    def test_it_picks_the_primary_file(self) -> None:
        chosen = civitai.select_file(VERSION_RESPONSE["files"])

        assert chosen["name"] == "waiANIMA_v10Base10.safetensors"

    def test_it_falls_back_to_a_single_model_file(self) -> None:
        files = [{"name": "a.safetensors", "type": "Model"}, {"name": "b.png", "type": "Archive"}]

        assert civitai.select_file(files)["name"] == "a.safetensors"

    def test_it_stops_when_the_body_cannot_be_told_apart(self) -> None:
        files = [
            {"name": "a.safetensors", "type": "Model"},
            {"name": "b.safetensors", "type": "Model"},
        ]

        with pytest.raises(SystemExit, match="本体のファイルを特定できない"):
            civitai.select_file(files)


class TestDownload:
    """完成品の名前へ据えるのは sha が合ってから（途中で切れた物を取得済みと読まない）。"""

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
        """落ちたバイト列は残す — 上流が差し替えたのか壊れたのかは人が見て決める。"""
        dest = tmp_path / "waiANIMA_v10Base10.safetensors"

        with pytest.raises(SystemExit, match="sha256 が API の値と違う"):
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
        """再実行で 4GB を取り直さない（sha 一致が「取得済み」の定義）。"""
        dest = tmp_path / "weights.safetensors"
        dest.write_bytes(FILE_BODY)

        civitai.download(DOWNLOAD_URL, dest, FILE_SHA256)

        assert network.calls == []

    def test_it_stops_when_a_different_file_sits_on_the_name(
        self, tmp_path: Path, network: _FakeNetwork, token: None
    ) -> None:
        dest = tmp_path / "weights.safetensors"
        dest.write_bytes(b"something else")

        with pytest.raises(SystemExit, match="同名で別物が置かれている"):
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
    """AIR は API 発行値が正 — 自前組み立てと食い違ったら、その旨を出してから API 側を採る。"""

    def test_it_reports_a_disagreement_and_takes_the_served_value(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        requested = civitai.parse_air("urn:air:sd1:model:civitai:2544636@2983680")

        resolved = civitai._resolve_air(VERSION_RESPONSE, requested)

        assert resolved == VERSION_RESPONSE["air"]
        assert "警告: AIR が指定と違う" in capsys.readouterr().out

    def test_it_falls_back_to_the_requested_urn_when_the_api_has_none(self) -> None:
        requested = civitai.parse_air("urn:air:anima:checkpoint:civitai:2544636@2983680")

        assert civitai._resolve_air({}, requested) == requested.urn


class TestFetchCheckpoint:
    """取り込み 1 回分 — 本体 1 本と `civitai.json` だけを置く。"""

    @pytest.fixture
    def taken(self, tmp_path: Path, network: _FakeNetwork, token: None) -> Path:
        return civitai.fetch_checkpoint(2544636, 2983680, out=tmp_path)

    def test_it_places_the_upstream_file_under_the_version_directory(
        self, tmp_path: Path, taken: Path
    ) -> None:
        assert taken == tmp_path / "civitai-2983680" / "waiANIMA_v10Base10.safetensors"
        assert taken.read_bytes() == FILE_BODY

    def test_it_leaves_only_the_body_and_the_record(self, tmp_path: Path, taken: Path) -> None:
        """同梱の Text Encoder / VAE は base 側を共有するので取らない。"""
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
        """突合も配布カードも小文字で書く（API は大文字で返す）。"""
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

    def test_it_overrides_the_derived_name_when_asked(
        self, tmp_path: Path, network: _FakeNetwork, token: None
    ) -> None:
        taken = civitai.fetch_checkpoint(2544636, 2983680, out=tmp_path, name="anima-wai-custom")
        record = json.loads((taken.parent / civitai.PROVENANCE_FILE).read_text(encoding="utf-8"))

        assert record["derived_name"] == "anima-wai-custom"


class TestGuidanceMode:
    """版を選ばずに来たときは、選び直す材料だけ出して何も落とさない。"""

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
