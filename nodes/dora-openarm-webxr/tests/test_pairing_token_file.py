# Addition to the vendored dora-openarm-webxr tests (Apache-2.0, like the
# rest of this directory).
#
# PATCHES.md 19: the pairing token from a file.

import pytest

from dora_openarm_webxr.main import pairing_token_from_environment


def test_unset_is_none():
    assert pairing_token_from_environment({}) is None


def test_token_from_file_drops_the_newline(tmp_path):
    path = tmp_path / "pairing_token"
    path.write_text("abcDEF_123-xyz-abcdef\n")
    assert pairing_token_from_environment({"PAIRING_TOKEN_FILE": str(path)}) == (
        "abcDEF_123-xyz-abcdef"
    )


def test_environment_value_wins_over_the_file(tmp_path):
    path = tmp_path / "pairing_token"
    path.write_text("from-file")
    environ = {"PAIRING_TOKEN": "from-env", "PAIRING_TOKEN_FILE": str(path)}
    assert pairing_token_from_environment(environ) == "from-env"


def test_missing_file_stops_instead_of_accepting_everyone(tmp_path):
    with pytest.raises(SystemExit, match="webxr-setup"):
        pairing_token_from_environment(
            {"PAIRING_TOKEN_FILE": str(tmp_path / "missing")}
        )


def test_empty_file_stops(tmp_path):
    path = tmp_path / "pairing_token"
    path.write_text("\n")
    with pytest.raises(SystemExit, match="empty"):
        pairing_token_from_environment({"PAIRING_TOKEN_FILE": str(path)})
