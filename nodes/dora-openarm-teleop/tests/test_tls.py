"""`scripts/webxr-setup`: SANs, chain, idempotence, both backends."""

from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from dora_openarm_teleop import tls

IPS = ["10.0.0.42", "10.0.0.7"]


def _backends() -> list[str]:
    found = []
    if tls._have_cryptography():
        found.append("cryptography")
    if shutil.which("openssl"):
        found.append("openssl")
    return found


@pytest.mark.parametrize("backend", _backends())
def test_generate_sans_and_chain(tmp_path: Path, backend: str) -> None:
    created, used = tls.generate(tmp_path, hostname="host.local", ips=IPS, backend=backend)
    assert created and used == backend
    paths = tls.TlsPaths(tmp_path)
    assert paths.complete()

    info = tls.certificate_info(paths.server_crt)
    assert "host.local" in info.dns_names
    assert "host" in info.dns_names
    for ip in [*IPS, "127.0.0.1"]:
        assert ip in info.ip_addresses
    assert info.not_after is not None
    assert len(info.sha256.split(":")) == 32

    assert tls.verify_chain(paths.root_crt, paths.server_crt)
    # A certificate is not its own issuer: the check must actually look at the signature.
    assert not tls.verify_chain(paths.server_crt, paths.server_crt)

    for key in (paths.root_key, paths.server_key):
        assert stat.S_IMODE(key.stat().st_mode) == 0o600


@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl CLI not installed")
def test_openssl_cli_verifies_chain(tmp_path: Path) -> None:
    tls.generate(tmp_path, hostname="host.local", ips=IPS)
    paths = tls.TlsPaths(tmp_path)
    result = subprocess.run(
        ["openssl", "verify", "-CAfile", str(paths.root_crt), str(paths.server_crt)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    key_modulus = subprocess.run(
        ["openssl", "rsa", "-in", str(paths.server_key), "-noout", "-modulus"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    cert_modulus = subprocess.run(
        ["openssl", "x509", "-in", str(paths.server_crt), "-noout", "-modulus"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert key_modulus == cert_modulus

    # The openssl-based reader (used without `cryptography`) must agree.
    by_cli = tls._certificate_info_openssl(paths.server_crt)
    by_default = tls.certificate_info(paths.server_crt)
    assert sorted(by_cli.dns_names) == sorted(by_default.dns_names)
    assert sorted(by_cli.ip_addresses) == sorted(by_default.ip_addresses)
    assert by_cli.sha256 == by_default.sha256
    assert by_cli.not_after is not None
    assert abs((by_cli.not_after - by_default.not_after).total_seconds()) < 1


def test_idempotent_unless_forced(tmp_path: Path) -> None:
    assert tls.generate(tmp_path, hostname="a.local", ips=IPS)[0]
    before = tls.certificate_info(tmp_path / "server.crt").sha256
    created, _ = tls.generate(tmp_path, hostname="b.local", ips=["10.1.1.1"])
    assert not created
    assert tls.certificate_info(tmp_path / "server.crt").sha256 == before
    created, _ = tls.generate(tmp_path, hostname="b.local", ips=["10.1.1.1"], force=True)
    assert created
    info = tls.certificate_info(tmp_path / "server.crt")
    assert info.sha256 != before
    assert "b.local" in info.dns_names and "10.1.1.1" in info.ip_addresses


def test_rejects_bad_ip(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        tls.generate(tmp_path, hostname="a.local", ips=["not-an-ip"])
    assert not tls.TlsPaths(tmp_path).complete()


def test_virtual_interfaces_filtered() -> None:
    assert tls.is_virtual_interface("docker0", "172.17.0.1")
    assert tls.is_virtual_interface("tailscale0", "100.70.45.86")
    assert tls.is_virtual_interface("eth0", "100.100.1.1")  # CGNAT range
    assert tls.is_virtual_interface("lo", "127.0.0.1")
    assert not tls.is_virtual_interface("wlP9s9", "10.0.0.42")
    assert not tls.is_virtual_interface("enP7s7", "10.0.0.5")


def test_lan_ips_exclude_virtual(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = [
        tls.Interface("lo", "127.0.0.1"),
        tls.Interface("wlP9s9", "10.0.0.42"),
        tls.Interface("tailscale0", "100.70.45.86"),
        tls.Interface("docker0", "172.17.0.1"),
    ]
    monkeypatch.setattr(tls, "interfaces", lambda: fake)
    assert tls.lan_ipv4s() == ["10.0.0.42"]
    assert tls.lan_ipv4s(include_virtual=True) == ["10.0.0.42", "100.70.45.86", "172.17.0.1"]
    monkeypatch.setattr(tls, "mdns_hostname", lambda: "host.local")
    assert tls.quest_urls(8443) == ["https://10.0.0.42:8443/", "https://host.local:8443/"]


def test_pairing_token_is_persistent_and_private(tmp_path: Path) -> None:
    directory = tmp_path / "tls"
    assert tls.read_pairing_token(directory) is None
    token, created = tls.ensure_pairing_token(directory)
    assert created
    assert tls.PAIRING_TOKEN_PATTERN.match(token) and len(token) >= 32
    path = tls.TlsPaths(directory).pairing_token
    assert path == directory / "pairing_token"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    # One token per machine: a second run (or the dataflow) sees the same one.
    assert tls.ensure_pairing_token(directory) == (token, False)
    assert tls.read_pairing_token(directory) == token
    rotated, created = tls.ensure_pairing_token(directory, rotate=True)
    assert created and rotated != token
    assert tls.read_pairing_token(directory) == rotated
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [p.name for p in directory.iterdir()] == ["pairing_token"]  # no scratch left
    # A damaged file is an error, never silently "no token" (which would open the node).
    path.write_text("\n")
    with pytest.raises(ValueError, match="rotate-token"):
        tls.read_pairing_token(directory)
    with pytest.raises(ValueError):
        tls.ensure_pairing_token(directory)


def test_quest_urls_carry_the_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tls, "interfaces", lambda: [tls.Interface("wlP9s9", "10.0.0.42")])
    monkeypatch.setattr(tls, "mdns_hostname", lambda: "host.local")
    assert tls.quest_urls(8443, token="abc_DEF-123") == [
        "https://10.0.0.42:8443/#k=abc_DEF-123",
        "https://host.local:8443/#k=abc_DEF-123",
    ]


def test_webxr_setup_creates_and_rotates_the_token(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(argv: list[str]) -> int:
        with pytest.raises(SystemExit) as stopped:
            tls.main(argv)
        return stopped.value.code

    monkeypatch.setattr(tls, "interfaces", lambda: [tls.Interface("eth0", "10.0.0.7")])
    base = ["--dir", str(tmp_path), "--hostname", "a.local", "--ip", "10.0.0.7"]
    assert run([*base, "--port", "9443"]) == 0
    token = tls.read_pairing_token(tmp_path)
    assert token is not None
    out = capsys.readouterr().out
    assert f"https://10.0.0.7:9443/#k={token}" in out
    assert run(base) == 0
    assert tls.read_pairing_token(tmp_path) == token  # kept
    assert run([*base, "--rotate-token"]) == 0
    rotated = tls.read_pairing_token(tmp_path)
    assert rotated not in (None, token)
    out = capsys.readouterr().out
    assert f"#k={rotated}" in out and "作り直しました" in out


def test_tls_paths_takes_a_string_directory(tmp_path: Path) -> None:
    """Regression: TlsPaths(str) raised TypeError out of whichever property was touched.

    Several callers hold the TLS directory as a string, because it arrives from a
    config file or an argparse default, and the failure was
    `unsupported operand type(s) for /: 'str' and 'str'` a long way from the
    constructor that accepted it.
    """
    from_str = tls.TlsPaths(str(tmp_path))
    from_path = tls.TlsPaths(tmp_path)
    assert isinstance(from_str.directory, Path)
    assert from_str == from_path  # frozen dataclass: normalised before it is compared
    assert from_str.pairing_token == tmp_path / tls.PAIRING_TOKEN_FILE
    assert from_str.all() == from_path.all()
    assert from_str.complete() is False
    tls.ensure_pairing_token(str(tmp_path))
    assert tls.read_pairing_token(str(tmp_path)) is not None
