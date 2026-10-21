from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from rag_app.errors import AppError, ErrorCode
from rag_app.ingestion.loaders import GitHubLoader, ParsedPdf, PdfLoader, WebLoader, is_public_http_url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://[::1]/admin",
        "http://169.254.169.254/latest/meta-data",
        "http://10.0.0.1/private",
        "https://user:pass@example.com/secret",
        "file:///etc/passwd",
    ],
)
def test_public_url_validator_rejects_ssrf_targets(url: str) -> None:
    assert is_public_http_url(url, resolver=lambda _host: ["93.184.216.34"]) is False


def test_public_url_validator_rejects_dns_rebinding_to_private_address() -> None:
    assert is_public_http_url("https://example.com/guide", resolver=lambda _host: ["10.1.2.3"]) is False


def test_public_url_validator_accepts_a_public_http_url() -> None:
    assert is_public_http_url("https://example.com/guide", resolver=lambda _host: ["93.184.216.34"]) is True


class FakeResponse:
    def __init__(self, body: bytes, url: str = "https://example.com/final", title: str = "Guide") -> None:
        self.body = body
        self.url = url
        self.title = title
        self.redirects: list[str] = []

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        yield self.body


@pytest.mark.asyncio
async def test_web_loader_pins_the_validated_address_and_revalidates_each_redirect() -> None:
    calls: list[tuple[str, str]] = []
    answers = iter([["93.184.216.34"], ["93.184.216.34"], ["93.184.216.35"]])

    def resolver(_host: str) -> list[str]:
        return next(answers)

    def fetcher(url: str, _timeout: int, pinned_address: str) -> FakeResponse:
        calls.append((url, pinned_address))
        response = FakeResponse(b"evidence", url="https://example.com/final")
        response.redirects = ["https://example.com/original"]
        return response

    loader = WebLoader(fetcher=fetcher, extractor=lambda _body, _url: ("Guide", "evidence"), resolver=resolver)

    await loader.load("https://example.com/original")

    assert calls == [("https://example.com/original", "93.184.216.34")]


@pytest.mark.asyncio
async def test_web_loader_rejects_an_unsafe_redirect_after_initial_public_resolution() -> None:
    answers = iter([["93.184.216.34"], ["127.0.0.1"]])
    def unsafe_response(_url: str, _timeout: int, _ip: str) -> FakeResponse:
        response = FakeResponse(b"evidence")
        response.redirects = ["http://127.0.0.1/admin"]
        return response

    loader = WebLoader(
        fetcher=unsafe_response,
        extractor=lambda _body, _url: ("Guide", "evidence"),
        resolver=lambda _host: next(answers),
    )

    with pytest.raises(AppError) as caught:
        await loader.load("https://example.com/original")

    assert caught.value.code is ErrorCode.INVALID_SOURCE


@pytest.mark.asyncio
async def test_web_loader_stops_streaming_at_byte_limit_without_persisting() -> None:
    loader = WebLoader(
        fetcher=lambda _url, _timeout: FakeResponse(b"x" * 11),
        extractor=lambda _body, _url: ("title", "text"),
        resolver=lambda _host: ["93.184.216.34"],
        max_bytes=10,
    )

    with pytest.raises(AppError) as caught:
        await loader.load("https://example.com/guide")

    assert caught.value.code is ErrorCode.INVALID_SOURCE


@pytest.mark.asyncio
async def test_web_loader_retains_final_url_and_title_after_safe_redirect() -> None:
    loader = WebLoader(
        fetcher=lambda _url, _timeout: FakeResponse(b"<article>evidence</article>"),
        extractor=lambda _body, _url: ("Final title", "evidence"),
        resolver=lambda _host: ["93.184.216.34"],
    )

    document = await loader.load("https://example.com/original")

    assert document.canonical_uri == "https://example.com/final"
    assert document.title == "Final title"
    assert document.provenance["final_url"] == "https://example.com/final"


@pytest.mark.asyncio
async def test_pdf_loader_rejects_non_pdf_and_page_limit_before_docling_import() -> None:
    loader = PdfLoader(docling_loader=lambda _data: ParsedPdf(page_count=201, chunks=[]))

    with pytest.raises(AppError, match="invalid"):
        await loader.load("notes.txt", b"plain text")
    with pytest.raises(AppError, match="invalid"):
        await loader.load("report.pdf", b"%PDF-1.7")


@pytest.mark.asyncio
async def test_pdf_page_limit_uses_docling_parsed_count_not_caller_claim() -> None:
    loader = PdfLoader(docling_loader=lambda _data: ParsedPdf(page_count=201, chunks=["native chunk"]))

    with pytest.raises(AppError) as caught:
        await loader.load("report.pdf", b"%PDF-1.7", page_count=1)

    assert caught.value.code is ErrorCode.INVALID_SOURCE


@pytest.mark.asyncio
async def test_docling_real_shape_flattens_nested_provenance_and_normalizes_headings() -> None:
    chunk = SimpleNamespace(
        text="Native Docling chunk",
        meta=SimpleNamespace(
            headings=["Chapter 1", "Evidence"],
            doc_items=[
                SimpleNamespace(prov=[]),
                SimpleNamespace(prov=[SimpleNamespace(page_no=7), SimpleNamespace(page_no=8)]),
            ],
        ),
    )
    loader = PdfLoader(docling_loader=lambda _data: ParsedPdf(page_count=8, chunks=[chunk]))

    document = await loader.load("report.pdf", b"%PDF-1.7")

    assert document.segments[0].metadata == {
        "page": 7,
        "pages": [7, 8],
        "heading": "Chapter 1 > Evidence",
    }


def test_github_filters_secrets_generated_vendor_and_large_or_binary_files() -> None:
    loader = GitHubLoader(client=object())

    assert loader.is_eligible("src/main.py", b"print('ok')") is True
    assert loader.is_eligible(".env", b"TOKEN=x") is False
    assert loader.is_eligible(".npmrc", b"//registry/:_authToken=x") is False
    assert loader.is_eligible("config.json", b'{"api_key":"x"}') is False
    assert loader.is_eligible("vendor/lib.js", b"x") is False
    assert loader.is_eligible("package-lock.json", b"{}") is False
    assert loader.is_eligible("image.png", b"\x89PNG\x00") is False
    assert loader.is_eligible("big.txt", b"x" * (1024 * 1024 + 1)) is False


def test_github_tree_entries_reject_symlink_submodule_and_oversized_blobs_before_download() -> None:
    loader = GitHubLoader(client=object())

    assert loader.is_eligible_tree_entry("src/main.py", "blob", "100644", 12) is True
    assert loader.is_eligible_tree_entry("src/link.py", "blob", "120000", 12) is False
    assert loader.is_eligible_tree_entry("deps/module", "commit", "160000", 12) is False
    assert loader.is_eligible_tree_entry("src/large.py", "blob", "100644", 1024 * 1024 + 1) is False
