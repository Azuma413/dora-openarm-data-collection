"""Self-signed TLS material for the WebXR page (docs/teleop.md, Headset setup).

WebXR only runs on an HTTPS page, and the Quest Browser reaches this machine
by LAN IP or by `<hostname>.local`, so the server certificate carries both as
subjectAltNames. The upstream `prepare_tls.sh` only puts a DNS name in it,
which makes browsing by IP a name mismatch on top of the untrusted CA.

The certificates are made with `cryptography` when it is importable (it comes
with aiortc) and with the openssl CLI otherwise.

The same directory holds the headset pairing token. TLS only proves the page
came from this machine; it does not stop another device on the LAN from
POSTing an offer and taking over the arms. The webxr node therefore requires
the token (env PAIRING_TOKEN) with every offer, and the page gets it from the
URL fragment `#k=<token>`, which the browser never sends over the wire. One
token per machine, kept until `scripts/webxr-setup --rotate-token`, so a bookmarked URL on
the headset keeps working across sessions.
"""

from __future__ import annotations

import argparse
import datetime
import ipaddress
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Interfaces that never carry the headset's traffic. Their addresses in the
# certificate would only be noise, and a URL printed for them would send the
# operator to an address the Quest cannot reach.
VIRTUAL_INTERFACE_PREFIXES: tuple[str, ...] = (
    "lo",
    "docker",
    "br-",
    "veth",
    "virbr",
    "tailscale",
    "zt",
    "wg",
    "tun",
    "tap",
    "cni",
    "flannel",
    "vxlan",
    "lxc",
    "lxd",
    "podman",
    "vmnet",
    "vboxnet",
    "kube",
)

DEFAULT_DAYS = 365
FILE_NAMES = ("root.crt", "root.key", "server.crt", "server.key")
PAIRING_TOKEN_FILE = "pairing_token"
# 24 random bytes -> 32 URL-safe characters: short enough to type once into a
# headset bookmark, far too long to guess over HTTPS.
PAIRING_TOKEN_BYTES = 24
# token_urlsafe's alphabet, so the token needs no escaping in a URL fragment.
PAIRING_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,256}$")


@dataclass(frozen=True)
class TlsPaths:
    """The four files `scripts/webxr-setup` writes into one directory.

    ``directory`` accepts a ``str`` as well as a ``Path`` and is normalised to a
    ``Path`` here. It used to be used as a ``Path`` straight away, so a caller
    holding the directory as a string -- several do, because it arrives from a
    config file or an argparse default -- got ``TypeError: unsupported operand
    type(s) for /: 'str' and 'str'`` out of whichever property it touched first,
    a long way from the constructor that accepted the string.
    """

    directory: Path

    def __post_init__(self) -> None:
        if not isinstance(self.directory, Path):
            # frozen, so plain assignment raises: that is the point of frozen.
            object.__setattr__(self, "directory", Path(self.directory))

    @property
    def root_crt(self) -> Path:
        return self.directory / "root.crt"

    @property
    def root_key(self) -> Path:
        return self.directory / "root.key"

    @property
    def server_crt(self) -> Path:
        return self.directory / "server.crt"

    @property
    def server_key(self) -> Path:
        return self.directory / "server.key"

    @property
    def pairing_token(self) -> Path:
        return self.directory / PAIRING_TOKEN_FILE

    def all(self) -> list[Path]:
        return [self.directory / name for name in FILE_NAMES]

    def complete(self) -> bool:
        return all(path.is_file() for path in self.all())


@dataclass
class CertificateInfo:
    """What the checks need to know about one certificate."""

    subject: str
    dns_names: list[str] = field(default_factory=list)
    ip_addresses: list[str] = field(default_factory=list)
    not_after: datetime.datetime | None = None
    sha256: str = ""


@dataclass
class Interface:
    """One IPv4 address on one network interface."""

    name: str
    address: str

    @property
    def virtual(self) -> bool:
        return is_virtual_interface(self.name, self.address)


# ---------------------------------------------------------------------------
# Network facts


def is_virtual_interface(name: str, address: str) -> bool:
    """Whether an interface/address pair cannot be the headset's LAN."""
    if name.startswith(VIRTUAL_INTERFACE_PREFIXES):
        return True
    ip = ipaddress.ip_address(address)
    # Tailscale hands out CGNAT addresses; link-local means no DHCP lease.
    return ip.is_loopback or ip.is_link_local or ip in ipaddress.ip_network("100.64.0.0/10")


def interfaces() -> list[Interface]:
    """Every IPv4 address of this host, from `ip -j`, or the default route's."""
    ip_cli = shutil.which("ip")
    if ip_cli:
        try:
            out = subprocess.run(
                [ip_cli, "-j", "-4", "addr", "show"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
            found = []
            for link in json.loads(out):
                for info in link.get("addr_info", []):
                    if info.get("family", "inet") == "inet" and "local" in info:
                        found.append(Interface(link["ifname"], info["local"]))
            return found
        except (OSError, subprocess.SubprocessError, ValueError, KeyError):
            pass
    # Fallback: the source address the kernel would use to leave the host.
    # A UDP connect sends no packet.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))
            return [Interface("default", probe.getsockname()[0])]
    except OSError:
        return []


def lan_ipv4s(include_virtual: bool = False) -> list[str]:
    """LAN IPv4 addresses the headset could use, in interface order."""
    seen: list[str] = []
    for iface in interfaces():
        wanted = not iface.virtual or (
            include_virtual and not ipaddress.ip_address(iface.address).is_loopback
        )
        if wanted and iface.address not in seen:
            seen.append(iface.address)
    return seen


def mdns_hostname() -> str:
    """`<hostname>.local`, the name Avahi announces for this machine."""
    return f"{socket.gethostname().split('.')[0]}.local"


def quest_urls(port: int, include_virtual: bool = False, token: str | None = None) -> list[str]:
    """URLs to open in the Quest Browser, IP first (mDNS is not guaranteed).

    With a pairing token the URLs end in `#k=<token>`: the page reads it from
    the fragment, which stays on the headset (no server log, no Referer).
    """
    suffix = f"#k={token}" if token else ""
    urls = [f"https://{ip}:{port}/{suffix}" for ip in lan_ipv4s(include_virtual)]
    urls.append(f"https://{mdns_hostname()}:{port}/{suffix}")
    return urls


# ---------------------------------------------------------------------------
# Pairing token


def read_pairing_token(directory: Path) -> str | None:
    """The machine's pairing token, or None when the file does not exist.

    Raises ValueError for a file that exists but does not hold a token: an
    empty or truncated token would silently pair nothing (or everything, if a
    caller treated it as "unset").
    """
    path = TlsPaths(Path(directory)).pairing_token
    try:
        token = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    if not PAIRING_TOKEN_PATTERN.match(token):
        raise ValueError(
            f"{path} does not hold a valid pairing token; run `scripts/webxr-setup --rotate-token`"
        )
    return token


def ensure_pairing_token(directory: Path, rotate: bool = False) -> tuple[str, bool]:
    """Return (token, created); create the token file (0600) when missing or rotating."""
    directory = Path(directory)
    if not rotate:
        existing = read_pairing_token(directory)
        if existing is not None:
            return existing, False
    token = secrets.token_urlsafe(PAIRING_TOKEN_BYTES)
    directory.mkdir(parents=True, exist_ok=True)
    # Written next to the target and renamed, so a crash never leaves a half
    # token that would lock the headset out, and the file is 0600 from the
    # first byte (the machine is shared).
    fd, scratch = tempfile.mkstemp(dir=directory, prefix=".pairing_token-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            out.write(token + "\n")
        os.chmod(scratch, 0o600)
        os.replace(scratch, TlsPaths(directory).pairing_token)
    except BaseException:
        Path(scratch).unlink(missing_ok=True)
        raise
    return token, True


# ---------------------------------------------------------------------------
# Certificate generation


def _have_cryptography() -> bool:
    try:
        import cryptography  # noqa: F401
    except ImportError:
        return False
    return True


def _san_entries(hostname: str, ips: list[str]) -> tuple[list[str], list[str]]:
    dns = [hostname]
    short = hostname.removesuffix(".local")
    if short != hostname:
        dns.append(short)
    dns.append("localhost")
    all_ips = list(dict.fromkeys([*ips, "127.0.0.1"]))
    for ip in all_ips:
        ipaddress.ip_address(ip)  # reject garbage before it lands in a certificate
    return list(dict.fromkeys(dns)), all_ips


def _write_private(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(data)


def _generate_cryptography(out: Path, hostname: str, ips: list[str], days: int) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    dns_names, ip_list = _san_entries(hostname, ips)
    now = datetime.datetime.now(datetime.UTC)
    # A day of slack: a headset whose clock runs slightly behind would
    # otherwise see a certificate from the future.
    not_before = now - datetime.timedelta(days=1)
    not_after = now + datetime.timedelta(days=days)

    # RSA 2048 like the upstream script, which is known to work in Quest Browser.
    root_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    root_name = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "dora-openarm-teleop"),
            x509.NameAttribute(NameOID.COMMON_NAME, f"dora-openarm-teleop local CA ({hostname})"),
        ]
    )
    root_ski = x509.SubjectKeyIdentifier.from_public_key(root_key.public_key())
    root_crt = (
        x509.CertificateBuilder()
        .subject_name(root_name)
        .issuer_name(root_name)
        .public_key(root_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(root_ski, critical=False)
        .sign(root_key, hashes.SHA256())
    )

    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    sans: list[x509.GeneralName] = [x509.DNSName(name) for name in dns_names]
    sans += [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ip_list]
    server_crt = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)]))
        .issuer_name(root_name)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.SubjectAlternativeName(sans), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(root_ski),
            critical=False,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(server_key.public_key()), critical=False
        )
        .sign(root_key, hashes.SHA256())
    )

    pem = serialization.Encoding.PEM
    no_password = serialization.NoEncryption()
    key_format = serialization.PrivateFormat.TraditionalOpenSSL
    _write_private(out / "root.key", root_key.private_bytes(pem, key_format, no_password))
    _write_private(out / "server.key", server_key.private_bytes(pem, key_format, no_password))
    (out / "root.crt").write_bytes(root_crt.public_bytes(pem))
    (out / "server.crt").write_bytes(server_crt.public_bytes(pem))


def _openssl(*args: str, cwd: Path) -> None:
    subprocess.run(["openssl", *args], check=True, cwd=cwd, capture_output=True, timeout=60)


def _generate_openssl(out: Path, hostname: str, ips: list[str], days: int) -> None:
    if shutil.which("openssl") is None:
        raise RuntimeError("neither the cryptography package nor the openssl CLI is available")
    dns_names, ip_list = _san_entries(hostname, ips)
    san = ",".join([f"DNS:{name}" for name in dns_names] + [f"IP:{ip}" for ip in ip_list])
    ca_subject = f"/O=dora-openarm-teleop/CN=dora-openarm-teleop local CA ({hostname})"
    _openssl(
        "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256",
        "-keyout", "root.key", "-out", "root.crt", "-days", str(days),
        "-subj", ca_subject,
        "-addext", "basicConstraints=critical,CA:TRUE,pathlen:0",
        "-addext", "keyUsage=critical,keyCertSign,cRLSign",
        cwd=out,
    )  # fmt: skip
    _openssl(
        "req", "-newkey", "rsa:2048", "-nodes", "-sha256",
        "-keyout", "server.key", "-out", "server.csr", "-subj", f"/CN={hostname}",
        cwd=out,
    )  # fmt: skip
    # Extensions go through an extfile rather than -copy_extensions so this
    # also works with OpenSSL 1.1.
    (out / "server.ext").write_text(
        "basicConstraints=critical,CA:FALSE\n"
        "keyUsage=critical,digitalSignature,keyEncipherment\n"
        "extendedKeyUsage=serverAuth\n"
        f"subjectAltName={san}\n"
        "authorityKeyIdentifier=keyid\n"
        "subjectKeyIdentifier=hash\n"
    )
    serial = "0x" + os.urandom(16).hex()
    _openssl(
        "x509", "-req", "-sha256", "-in", "server.csr",
        "-CA", "root.crt", "-CAkey", "root.key", "-set_serial", serial,
        "-days", str(days), "-extfile", "server.ext", "-out", "server.crt",
        cwd=out,
    )  # fmt: skip
    for scratch in ("server.csr", "server.ext"):
        (out / scratch).unlink(missing_ok=True)
    for key in ("root.key", "server.key"):
        (out / key).chmod(0o600)


def generate(
    directory: Path,
    hostname: str | None = None,
    ips: list[str] | None = None,
    days: int = DEFAULT_DAYS,
    force: bool = False,
    backend: str = "auto",
) -> tuple[bool, str]:
    """Write the CA and server certificate into `directory`.

    Returns (created, backend). Existing complete material is kept unless
    `force`: regenerating changes the fingerprint, and the operator then has
    to click through the browser warning on the headset again.
    """
    paths = TlsPaths(Path(directory))
    if paths.complete() and not force:
        return False, "existing"
    hostname = hostname or mdns_hostname()
    ips = lan_ipv4s() if ips is None else ips
    if backend == "auto":
        backend = "cryptography" if _have_cryptography() else "openssl"
    paths.directory.mkdir(parents=True, exist_ok=True)
    # Build in a scratch directory next to the target, then move, so a
    # failure never leaves a key that does not match its certificate.
    with tempfile.TemporaryDirectory(dir=paths.directory, prefix=".tls-") as scratch:
        scratch_path = Path(scratch)
        if backend == "cryptography":
            _generate_cryptography(scratch_path, hostname, ips, days)
        elif backend == "openssl":
            _generate_openssl(scratch_path, hostname, ips, days)
        else:
            raise ValueError(f"unknown backend {backend!r}")
        for name in FILE_NAMES:
            os.replace(scratch_path / name, paths.directory / name)
    return True, backend


# ---------------------------------------------------------------------------
# Inspection


def certificate_info(path: Path) -> CertificateInfo:
    """Subject, SANs, expiry and SHA-256 fingerprint of a PEM certificate."""
    data = Path(path).read_bytes()
    if _have_cryptography():
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes

        cert = x509.load_pem_x509_certificate(data)
        info = CertificateInfo(
            subject=cert.subject.rfc4514_string(),
            not_after=cert.not_valid_after_utc,
            sha256=cert.fingerprint(hashes.SHA256()).hex(":").upper(),
        )
        try:
            san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        except x509.ExtensionNotFound:
            return info
        info.dns_names = san.get_values_for_type(x509.DNSName)
        info.ip_addresses = [str(ip) for ip in san.get_values_for_type(x509.IPAddress)]
        return info
    return _certificate_info_openssl(Path(path))


def _certificate_info_openssl(path: Path) -> CertificateInfo:
    out = subprocess.run(
        [
            "openssl", "x509", "-in", str(path), "-noout", "-subject", "-enddate",
            "-fingerprint", "-sha256", "-ext", "subjectAltName",
        ],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout  # fmt: skip
    info = CertificateInfo(subject="")
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("subject="):
            info.subject = line.removeprefix("subject=").strip()
        elif line.startswith("notAfter="):
            info.not_after = datetime.datetime.strptime(
                line.removeprefix("notAfter="), "%b %d %H:%M:%S %Y %Z"
            ).replace(tzinfo=datetime.UTC)
        elif "Fingerprint=" in line:
            info.sha256 = line.split("=", 1)[1]
        elif line.startswith(("DNS:", "IP")):
            for entry in line.split(","):
                kind, _, value = entry.strip().partition(":")
                if kind == "DNS":
                    info.dns_names.append(value)
                elif kind.startswith("IP"):
                    info.ip_addresses.append(value)
    return info


def verify_chain(root_crt: Path, server_crt: Path) -> bool:
    """Whether `server_crt` is signed by `root_crt` and is currently valid."""
    if _have_cryptography():
        from cryptography import x509
        from cryptography.exceptions import InvalidSignature

        root = x509.load_pem_x509_certificate(Path(root_crt).read_bytes())
        server = x509.load_pem_x509_certificate(Path(server_crt).read_bytes())
        try:
            server.verify_directly_issued_by(root)
        except (InvalidSignature, ValueError, TypeError):
            return False
        now = datetime.datetime.now(datetime.UTC)
        return server.not_valid_before_utc <= now <= server.not_valid_after_utc
    result = subprocess.run(
        ["openssl", "verify", "-CAfile", str(root_crt), str(server_crt)],
        capture_output=True,
        timeout=10,
        check=False,
    )
    return result.returncode == 0


def missing_ips(server_crt: Path, include_virtual: bool = False) -> list[str]:
    """Current LAN IPs the certificate does not name (the reason to regenerate)."""
    named = set(certificate_info(server_crt).ip_addresses)
    return [ip for ip in lan_ipv4s(include_virtual) if ip not in named]


# ---------------------------------------------------------------------------
# CLI


def add_arguments(parser: argparse.ArgumentParser, default_dir: Path) -> None:
    """Options of `scripts/webxr-setup`."""
    parser.add_argument("--dir", type=Path, default=default_dir, help="output directory")
    parser.add_argument("--force", action="store_true", help="regenerate existing files")
    parser.add_argument("--hostname", help="DNS name (default: <hostname>.local)")
    parser.add_argument(
        "--ip",
        action="append",
        default=None,
        help="IPv4 address to include (repeatable; default: every LAN address)",
    )
    parser.add_argument(
        "--include-virtual",
        action="store_true",
        help="also include docker/tailscale/VPN addresses",
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--backend", choices=("auto", "cryptography", "openssl"), default="auto")
    parser.add_argument(
        "--rotate-token",
        action="store_true",
        help="replace the headset pairing token (every saved URL stops working)",
    )
    parser.add_argument("--port", type=int, default=8443, help="port shown in the printed URLs")


def run(args: argparse.Namespace) -> int:
    """`scripts/webxr-setup`: create (or report on) the certificates and the pairing token."""
    ips = args.ip if args.ip is not None else lan_ipv4s(args.include_virtual)
    created, backend = generate(
        args.dir,
        hostname=args.hostname,
        ips=ips,
        days=args.days,
        force=args.force,
        backend=args.backend,
    )
    paths = TlsPaths(Path(args.dir))
    if created:
        print(f"証明書を作成しました（{backend}）: {paths.directory}")
    else:
        print(f"既存の証明書を使います（作り直すには --force）: {paths.directory}")
    root = certificate_info(paths.root_crt)
    server = certificate_info(paths.server_crt)
    print(f"  CA     SHA-256: {root.sha256}")
    print(f"  server SHA-256: {server.sha256}")
    print(f"  SAN DNS: {', '.join(server.dns_names)}")
    print(f"  SAN IP : {', '.join(server.ip_addresses)}")
    if server.not_after is not None:
        print(f"  有効期限: {server.not_after:%Y-%m-%d %H:%M} UTC")
    if not verify_chain(paths.root_crt, paths.server_crt):
        print("NG: server.crt が root.crt で検証できません。--force で作り直してください。")
        return 1
    absent = [ip for ip in ips if ip not in server.ip_addresses]
    if absent:
        print(
            f"WARN: 現在の IP {', '.join(absent)} が証明書に入っていません。"
            "`scripts/webxr-setup --force` で作り直してください（Quest で警告の通過がやり直しになります）。"
        )
    if created:
        print(
            "Quest Browser では初回と証明書を作り直した後に「詳細設定」→「アクセスする」が必要です。"
        )
    try:
        token, token_created = ensure_pairing_token(paths.directory, rotate=args.rotate_token)
    except (OSError, ValueError) as error:
        print(f"NG: ペアリングトークンを用意できません: {error}")
        return 1
    if args.rotate_token:
        print(
            f"ペアリングトークンを作り直しました: {paths.pairing_token}\n"
            "  以前の URL（#k=...）は使えなくなります。ヘッドセットで下の URL を開き直してください。"
        )
    elif token_created:
        print(f"ペアリングトークンを作成しました: {paths.pairing_token}")
    else:
        print(f"既存のペアリングトークンを使います: {paths.pairing_token}")
    print("Quest Browser で開く URL（#k= 以降がトークン。他人に見せないでください）:")
    for url in quest_urls(args.port, args.include_virtual, token):
        print(f"  {url}")
    return 0


def main(argv: list[str] | None = None) -> None:
    """`scripts/webxr-setup`: TLS certificate and pairing token for the WebXR page."""
    parser = argparse.ArgumentParser(prog="webxr-setup", description=main.__doc__)
    add_arguments(parser, Path.cwd() / "tls")
    args = parser.parse_args(argv)
    code = run(args)
    if code == 0:
        token = read_pairing_token(Path(args.dir))
        print(
            "USB 接続（scripts/webxr-usb/start.sh）のときは:\n"
            f"  https://localhost:{args.port}/#k={token}"
        )
    raise SystemExit(code)
