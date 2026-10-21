"""Bounded, provenance-preserving PDF, web, and public GitHub loaders."""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
import socket
import tempfile
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, cast
from urllib.parse import urljoin, urlsplit, urlunsplit

from rag_app.contracts import SourceType
from rag_app.errors import AppError, ErrorCode
from rag_app.ingestion.models import DocumentSegment, NormalizedDocument

_PDF_MAX_BYTES = 25 * 1024 * 1024
_PDF_MAX_PAGES = 200
_WEB_MAX_BYTES = 5 * 1024 * 1024
_WEB_TIMEOUT_SECONDS = 30
_GITHUB_MAX_FILES = 1000
_GITHUB_MAX_BYTES = 50 * 1024 * 1024
_GITHUB_MAX_FILE_BYTES = 1024 * 1024


def _invalid(reason: str) -> AppError:
    return AppError(ErrorCode.INVALID_SOURCE, reason, {"reason": reason})


def _default_resolver(host: str) -> list[str]:
    try:
        return sorted({cast(str, result[4][0]) for result in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)})
    except socket.gaierror:
        return []


def is_public_http_url(url: str, resolver: Callable[[str], list[str]] = _default_resolver) -> bool:
    """Reject credentials, non-HTTP URLs, and every non-public DNS answer."""
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            return False
        literal_address: str | None
        try:
            literal_address = str(ipaddress.ip_address(parsed.hostname))
        except ValueError:
            literal_address = None
        addresses = [literal_address] if literal_address else resolver(parsed.hostname)
    except (TypeError, ValueError):
        return False
    if not addresses:
        return False
    try:
        return all(_is_public_address(address) for address in addresses)
    except ValueError:
        return False


def _is_public_address(address: str) -> bool:
    value = ipaddress.ip_address(address)
    return value.is_global


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


@dataclass
class FetchedResponse:
    body: bytes
    url: str
    redirects: list[str] = field(default_factory=list)

    async def iter_bytes(self) -> Any:
        yield self.body


@dataclass(frozen=True)
class ParsedPdf:
    page_count: int
    chunks: Iterable[Any]


def resolve_public_url(url: str, resolver: Callable[[str], list[str]] = _default_resolver) -> str:
    """Return the exact public address approved for the next connection."""
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise _invalid("URL must be a public HTTP(S) address")
    try:
        address = str(ipaddress.ip_address(parsed.hostname))
    except ValueError as error:
        answers = resolver(parsed.hostname)
        if not answers or not all(_is_public_address(answer) for answer in answers):
            raise _invalid("URL must be a public HTTP(S) address") from error
        return answers[0]
    if not _is_public_address(address):
        raise _invalid("URL must be a public HTTP(S) address")
    return address


class WebLoader:
    def __init__(
        self,
        fetcher: Callable[..., Awaitable[Any] | Any] | None = None,
        extractor: Callable[[bytes, str], tuple[str, str]] | None = None,
        resolver: Callable[[str], list[str]] = _default_resolver,
        max_bytes: int = _WEB_MAX_BYTES,
        timeout_seconds: int = _WEB_TIMEOUT_SECONDS,
    ) -> None:
        self._resolver = resolver
        self._fetcher = fetcher or self._httpx_fetch
        self._extractor = extractor or self._trafilatura_extract
        self._max_bytes = max_bytes
        self._timeout_seconds = timeout_seconds

    async def load(self, url: str) -> NormalizedDocument:
        pinned_address = resolve_public_url(url, self._resolver)
        response = await self._fetch(url, pinned_address)
        final_url = str(response.url)
        redirect_hops = list(getattr(response, "redirects", [])) + list(getattr(response, "history", [])) + [final_url]
        for hop in redirect_hops:
            resolve_public_url(str(hop), self._resolver)
        body = bytearray()
        async for part in response.iter_bytes():
            body.extend(part)
            if len(body) > self._max_bytes:
                raise _invalid("Web response exceeds the byte limit")
        title, content = self._extractor(bytes(body), final_url)
        if not content.strip():
            raise _invalid("Web page did not contain extractable text")
        return NormalizedDocument(
            source_type=SourceType.WEB,
            canonical_uri=_canonical_http_url(final_url),
            title=title.strip() or _canonical_http_url(final_url),
            content=content,
            provenance={"final_url": _canonical_http_url(final_url), "title": title.strip()},
        )

    async def _fetch(self, url: str, pinned_address: str) -> Any:
        parameters = inspect.signature(self._fetcher).parameters
        if len(parameters) >= 3:
            return await _maybe_await(cast(Any, self._fetcher)(url, self._timeout_seconds, pinned_address))
        return await _maybe_await(cast(Any, self._fetcher)(url, self._timeout_seconds))

    async def _httpx_fetch(self, url: str, timeout_seconds: int, pinned_address: str) -> FetchedResponse:
        import httpx

        redirects: list[str] = []
        current = url
        async with httpx.AsyncClient(follow_redirects=False, timeout=timeout_seconds) as client:
            for _ in range(10):
                pinned_address = resolve_public_url(current, self._resolver)
                parsed = urlsplit(current)
                headers = {"Host": parsed.netloc}
                extensions = {"sni_hostname": parsed.hostname}
                pinned_url = urlunsplit(
                    (parsed.scheme, _netloc_with_ip(parsed, pinned_address), parsed.path, parsed.query, "")
                )
                request = client.build_request("GET", pinned_url, headers=headers, extensions=extensions)
                async with client.stream(
                    request.method, request.url, headers=request.headers, extensions=request.extensions
                ) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise _invalid("Redirect did not provide a location")
                        redirects.append(current)
                        current = urljoin(current, location)
                        continue
                    response.raise_for_status()
                    body = bytearray()
                    async for part in response.aiter_bytes():
                        body.extend(part)
                        if len(body) > self._max_bytes:
                            raise _invalid("Web response exceeds the byte limit")
                    # Never expose the IP-pinned connection URL as provenance;
                    # canonical identity stays on the validated logical hostname.
                    return FetchedResponse(bytes(body), current, redirects)
        raise _invalid("Too many redirects")

    @staticmethod
    def _trafilatura_extract(body: bytes, final_url: str) -> tuple[str, str]:
        try:
            import trafilatura
        except ImportError as error:
            raise _invalid("Web extraction support is unavailable") from error
        text = trafilatura.extract(body, url=final_url) or ""
        metadata = trafilatura.extract_metadata(body)
        title = metadata.title if metadata and metadata.title else ""
        return title, text


class PdfLoader:
    def __init__(
        self,
        docling_loader: Callable[[bytes], ParsedPdf] | None = None,
        max_bytes: int = _PDF_MAX_BYTES,
        max_pages: int = _PDF_MAX_PAGES,
    ) -> None:
        self._docling_loader = docling_loader
        self._max_bytes = max_bytes
        self._max_pages = max_pages

    async def load(self, filename: str, data: bytes, page_count: int | None = None) -> NormalizedDocument:
        if not filename.lower().endswith(".pdf") or not data.startswith(b"%PDF-"):
            raise _invalid("Only PDF files are supported")
        if len(data) > self._max_bytes:
            raise _invalid("PDF exceeds the byte limit")
        if page_count is not None and page_count > self._max_pages:
            raise _invalid("PDF exceeds the page limit")
        parsed = await asyncio.to_thread(self._load_docling, data)
        if parsed.page_count > self._max_pages:
            raise _invalid("PDF exceeds the page limit")
        chunks = list(parsed.chunks)
        segments = [_docling_segment(chunk) for chunk in chunks]
        if not segments:
            raise _invalid("PDF did not contain extractable text")
        return NormalizedDocument(
            source_type=SourceType.PDF,
            canonical_uri=filename,
            title=PurePosixPath(filename.replace("\\", "/")).name,
            content="\n\n".join(segment.content for segment in segments),
            segments=segments,
            provenance={"filename": filename},
        )

    def _load_docling(self, data: bytes) -> ParsedPdf:
        if self._docling_loader is not None:
            return self._docling_loader(data)
        try:
            from docling.chunking import HybridChunker  # type: ignore[attr-defined]
            from docling.document_converter import DocumentConverter
        except ImportError as error:
            raise _invalid("PDF ingestion support is unavailable") from error
        # Docling's converter accepts a source path. The temporary file is
        # removed before control returns, so rejected/processed uploads are not retained.
        with tempfile.NamedTemporaryFile(suffix=".pdf") as upload:
            upload.write(data)
            upload.flush()
            converted = DocumentConverter().convert(upload.name)
            pages = getattr(converted.document, "pages", {})
            return ParsedPdf(page_count=len(pages), chunks=list(HybridChunker().chunk(converted.document)))


def _docling_segment(chunk: Any) -> DocumentSegment:
    if isinstance(chunk, str):
        return DocumentSegment(content=chunk)
    if isinstance(chunk, dict):
        return DocumentSegment(content=str(chunk.get("text", "")), metadata=dict(chunk.get("metadata", {})))
    text = str(getattr(chunk, "text", chunk))
    chunk_meta = getattr(chunk, "meta", None)
    doc_items = getattr(chunk_meta, "doc_items", []) if chunk_meta else []
    pages: list[int] = []
    for item in doc_items or []:
        provenance = getattr(item, "prov", []) or []
        if not isinstance(provenance, (list, tuple)):
            provenance = [provenance]
        for entry in provenance:
            page = getattr(entry, "page_no", None)
            if isinstance(page, int) and page not in pages:
                pages.append(page)
    headings = getattr(chunk_meta, "headings", getattr(chunk, "heading", None))
    if isinstance(headings, (list, tuple)):
        headings = " > ".join(str(item).strip() for item in headings if str(item).strip()) or None
    metadata = {
        key: value
        for key, value in {
            "page": pages[0] if pages else getattr(chunk, "page", None),
            "pages": pages or None,
            "heading": headings,
        }.items()
        if value is not None
    }
    return DocumentSegment(content=text, metadata=metadata)


class GitHubLoader:
    def __init__(
        self,
        client: Any | None = None,
        max_files: int = _GITHUB_MAX_FILES,
        max_bytes: int = _GITHUB_MAX_BYTES,
    ) -> None:
        self._client = client
        self._max_files = max_files
        self._max_bytes = max_bytes

    async def load(self, repository_url: str) -> NormalizedDocument:
        owner_repo = _github_owner_repo(repository_url)
        client = self._client or self._create_client()
        repo = await asyncio.to_thread(client.get_repo, owner_repo)
        if repo.private:
            raise _invalid("Private GitHub repositories are not supported")
        branch = repo.default_branch
        commit = repo.get_branch(branch).commit.sha
        tree = await asyncio.to_thread(repo.get_git_tree, branch, recursive=True)
        selected: list[tuple[str, bytes]] = []
        total = 0
        for item in tree.tree:
            if not self.is_eligible_tree_entry(
                item.path, item.type, getattr(item, "mode", ""), getattr(item, "size", None)
            ):
                continue
            contents = await asyncio.to_thread(repo.get_contents, item.path, ref=branch)
            data = contents.decoded_content
            if not self.is_eligible(item.path, data):
                continue
            selected.append((item.path, data))
            total += len(data)
            if len(selected) > self._max_files or total > self._max_bytes:
                raise _invalid("GitHub repository exceeds ingestion limits")
        if not selected:
            raise _invalid("GitHub repository contains no eligible text files")
        segments = [
            DocumentSegment(
                content=data.decode("utf-8"), metadata={"path": path, "commit": commit, "language": _language(path)}
            )
            for path, data in selected
        ]
        canonical = f"https://github.com/{owner_repo}"
        return NormalizedDocument(
            source_type=SourceType.GITHUB,
            canonical_uri=canonical,
            title=owner_repo,
            content="\n\n".join(segment.content for segment in segments),
            segments=segments,
            provenance={"repository": canonical, "branch": branch, "commit": commit},
        )

    @staticmethod
    def is_eligible(path: str, data: bytes) -> bool:
        return (
            GitHubLoader._eligible_path(path)
            and len(data) <= _GITHUB_MAX_FILE_BYTES
            and b"\x00" not in data
            and _is_utf8(data)
        )

    @staticmethod
    def is_eligible_tree_entry(path: str, item_type: str, mode: str, size: int | None) -> bool:
        return (
            item_type == "blob"
            and mode != "120000"
            and mode != "160000"
            and size is not None
            and size <= _GITHUB_MAX_FILE_BYTES
            and GitHubLoader._eligible_path(path)
        )

    @staticmethod
    def _eligible_path(path: str) -> bool:
        lowered = path.lower()
        parts = lowered.split("/")
        disallowed_directories = {".git", "vendor", "node_modules", "build", "dist", "coverage", "__pycache__"}
        if any(part in disallowed_directories for part in parts[:-1]):
            return False
        name = parts[-1]
        if name.startswith(".env") or name in {
            ".npmrc",
            ".pypirc",
            ".netrc",
            ".awsconfig",
            "credentials",
            "credential",
            "config.json",
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            "poetry.lock",
            "uv.lock",
        }:
            return False
        return not any(marker in name for marker in ("secret", "credential", ".pem", ".key"))

    @staticmethod
    def _create_client() -> Any:
        try:
            from github import Github
        except ImportError as error:
            raise _invalid("GitHub ingestion support is unavailable") from error
        return Github()


def _github_owner_repo(url: str) -> str:
    parsed = urlsplit(url)
    parts = [part for part in parsed.path.split("/") if part]
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "github.com"
        or len(parts) != 2
        or parsed.username
        or parsed.password
    ):
        raise _invalid("A public HTTPS GitHub repository URL is required")
    return f"{parts[0]}/{parts[1].removesuffix('.git')}"


def _canonical_http_url(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", parsed.query, ""))


def _netloc_with_ip(parsed: Any, address: str) -> str:
    host = f"[{address}]" if ":" in address else address
    return f"{host}:{parsed.port}" if parsed.port else host


def _is_utf8(data: bytes) -> bool:
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _language(path: str) -> str | None:
    return {".py": "python", ".ts": "typescript", ".js": "javascript", ".md": "markdown", ".java": "java"}.get(
        PurePosixPath(path).suffix.lower()
    )
