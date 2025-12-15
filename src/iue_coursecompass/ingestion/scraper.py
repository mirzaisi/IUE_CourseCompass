"""
Scraper Module - Web scraping with caching, rate limiting, and retries.
=======================================================================

Implements the "scrape once, use saved data" principle:
- File-based caching of raw HTML
- Rate limiting to be respectful to servers
- Automatic retries with exponential backoff
- Idempotent operations (safe to re-run)
"""

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from iue_coursecompass.shared.config import get_settings
from iue_coursecompass.shared.logging import get_logger
from iue_coursecompass.shared.utils import ensure_directory

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ScrapedPage:
    """Represents a scraped web page."""

    url: str
    html: str
    status_code: int
    scraped_at: datetime
    from_cache: bool = False
    cache_path: Optional[Path] = None

    @property
    def is_success(self) -> bool:
        """Check if the page was successfully scraped."""
        return 200 <= self.status_code < 300

    @property
    def content_length(self) -> int:
        """Get the content length in bytes."""
        return len(self.html.encode("utf-8"))


@dataclass
class ScraperStats:
    """Statistics for scraping session."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    successful: int = 0
    failed: int = 0
    total_bytes: int = 0
    start_time: datetime = field(default_factory=datetime.utcnow)

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    @property
    def success_rate(self) -> float:
        """Calculate success rate (excluding cache hits)."""
        actual_requests = self.cache_misses
        if actual_requests == 0:
            return 1.0
        return self.successful / actual_requests


# ─────────────────────────────────────────────────────────────────────────────
# Scraper Class
# ─────────────────────────────────────────────────────────────────────────────


class Scraper:
    """
    Web scraper with caching, rate limiting, and retries.

    Features:
    - File-based HTML caching in data/raw/
    - Configurable rate limiting
    - Automatic retries with exponential backoff
    - Respectful user agent
    - Statistics tracking

    Example:
        >>> scraper = Scraper()
        >>> page = scraper.scrape("https://example.com/course")
        >>> if page.is_success:
        ...     print(page.html[:100])
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        rate_limit: Optional[float] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        user_agent: Optional[str] = None,
        cache_enabled: Optional[bool] = None,
        cache_expiry_days: Optional[int] = None,
    ):
        """
        Initialize the scraper.

        Args:
            cache_dir: Directory for cached HTML files
            rate_limit: Minimum seconds between requests
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            user_agent: User agent string
            cache_enabled: Whether to use caching
            cache_expiry_days: Days until cache expires (0 = never)
        """
        settings = get_settings()
        scraping_config = settings.scraping

        # Configuration
        self.cache_dir = cache_dir or settings.resolved_paths.raw_dir
        self.rate_limit = rate_limit if rate_limit is not None else scraping_config.rate_limit
        self.timeout = timeout if timeout is not None else scraping_config.timeout
        self.max_retries = max_retries if max_retries is not None else scraping_config.max_retries
        self.user_agent = user_agent or scraping_config.user_agent
        self.cache_enabled = (
            cache_enabled if cache_enabled is not None else scraping_config.cache_enabled
        )
        self.cache_expiry_days = (
            cache_expiry_days if cache_expiry_days is not None else scraping_config.cache_expiry_days
        )

        # Retry settings
        self.retry_min_wait = scraping_config.retry_min_wait
        self.retry_max_wait = scraping_config.retry_max_wait

        # State
        self._last_request_time: Optional[float] = None
        self._session: Optional[requests.Session] = None
        self.stats = ScraperStats()

        # Ensure cache directory exists
        if self.cache_enabled:
            ensure_directory(self.cache_dir)

        logger.debug(
            f"Scraper initialized: cache={self.cache_enabled}, "
            f"rate_limit={self.rate_limit}s, timeout={self.timeout}s"
        )

    @property
    def session(self) -> requests.Session:
        """Get or create the requests session."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(
                {
                    "User-Agent": self.user_agent,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "keep-alive",
                }
            )
        return self._session

    def _get_cache_path(self, url: str) -> Path:
        """Generate cache file path for a URL."""
        # Create a deterministic filename from URL hash
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        
        # Extract domain for subdirectory organization
        parsed = urlparse(url)
        domain = parsed.netloc.replace(".", "_")
        
        # Create path: cache_dir/domain/hash.html
        cache_subdir = self.cache_dir / domain
        ensure_directory(cache_subdir)
        
        return cache_subdir / f"{url_hash}.html"

    def _get_metadata_path(self, cache_path: Path) -> Path:
        """Get metadata file path for a cache file."""
        return cache_path.with_suffix(".meta")

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if a cached file is still valid."""
        if not cache_path.exists():
            return False

        # Check expiry if configured
        if self.cache_expiry_days > 0:
            file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
            expiry_time = file_time + timedelta(days=self.cache_expiry_days)
            if datetime.now() > expiry_time:
                logger.debug(f"Cache expired: {cache_path}")
                return False

        return True

    def _load_from_cache(self, url: str) -> Optional[ScrapedPage]:
        """Load a page from cache if available and valid."""
        if not self.cache_enabled:
            return None

        cache_path = self._get_cache_path(url)

        if not self._is_cache_valid(cache_path):
            return None

        try:
            html = cache_path.read_text(encoding="utf-8")
            
            # Try to load metadata
            meta_path = self._get_metadata_path(cache_path)
            scraped_at = datetime.fromtimestamp(cache_path.stat().st_mtime)
            status_code = 200
            
            if meta_path.exists():
                import json
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                scraped_at = datetime.fromisoformat(meta.get("scraped_at", scraped_at.isoformat()))
                status_code = meta.get("status_code", 200)

            logger.debug(f"Cache hit: {url}")
            self.stats.cache_hits += 1

            return ScrapedPage(
                url=url,
                html=html,
                status_code=status_code,
                scraped_at=scraped_at,
                from_cache=True,
                cache_path=cache_path,
            )

        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return None

    def _save_to_cache(self, page: ScrapedPage) -> None:
        """Save a scraped page to cache."""
        if not self.cache_enabled:
            return

        cache_path = self._get_cache_path(page.url)

        try:
            # Save HTML
            cache_path.write_text(page.html, encoding="utf-8")
            
            # Save metadata
            import json
            meta_path = self._get_metadata_path(cache_path)
            meta = {
                "url": page.url,
                "status_code": page.status_code,
                "scraped_at": page.scraped_at.isoformat(),
                "content_length": page.content_length,
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            
            page.cache_path = cache_path
            logger.debug(f"Cached: {page.url} -> {cache_path}")

        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limiting."""
        if self._last_request_time is not None and self.rate_limit > 0:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.rate_limit:
                sleep_time = self.rate_limit - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)

    def _make_request(self, url: str) -> ScrapedPage:
        """Make an HTTP request with retries."""
        
        @retry(
            retry=retry_if_exception_type((requests.RequestException, requests.Timeout)),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(min=self.retry_min_wait, max=self.retry_max_wait),
            before_sleep=lambda retry_state: logger.warning(
                f"Retry {retry_state.attempt_number}/{self.max_retries} for {url}"
            ),
        )
        def _request_with_retry() -> requests.Response:
            self._wait_for_rate_limit()
            self._last_request_time = time.time()
            return self.session.get(url, timeout=self.timeout)

        try:
            response = _request_with_retry()
            
            page = ScrapedPage(
                url=url,
                html=response.text,
                status_code=response.status_code,
                scraped_at=datetime.utcnow(),
                from_cache=False,
            )

            self.stats.successful += 1
            self.stats.total_bytes += page.content_length

            return page

        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            self.stats.failed += 1
            
            # Return a failed page object
            return ScrapedPage(
                url=url,
                html="",
                status_code=0,
                scraped_at=datetime.utcnow(),
                from_cache=False,
            )

    def scrape(self, url: str, force_refresh: bool = False) -> ScrapedPage:
        """
        Scrape a URL with caching support.

        Args:
            url: URL to scrape
            force_refresh: If True, ignore cache and re-scrape

        Returns:
            ScrapedPage object with HTML content and metadata
        """
        self.stats.total_requests += 1

        # Try cache first (unless force refresh)
        if not force_refresh:
            cached = self._load_from_cache(url)
            if cached is not None:
                return cached

        self.stats.cache_misses += 1
        logger.info(f"Scraping: {url}")

        # Make actual request
        page = self._make_request(url)

        # Cache successful responses
        if page.is_success:
            self._save_to_cache(page)

        return page

    def scrape_many(
        self,
        urls: list[str],
        force_refresh: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> list[ScrapedPage]:
        """
        Scrape multiple URLs.

        Args:
            urls: List of URLs to scrape
            force_refresh: If True, ignore cache and re-scrape all
            progress_callback: Optional callback(current, total, url) for progress

        Returns:
            List of ScrapedPage objects
        """
        pages = []
        total = len(urls)

        for i, url in enumerate(urls, 1):
            if progress_callback:
                progress_callback(i, total, url)

            page = self.scrape(url, force_refresh=force_refresh)
            pages.append(page)

        return pages

    def clear_cache(self, url: Optional[str] = None) -> int:
        """
        Clear cached files.

        Args:
            url: If provided, clear only this URL's cache. Otherwise clear all.

        Returns:
            Number of files deleted
        """
        if not self.cache_enabled:
            return 0

        deleted = 0

        if url:
            # Clear specific URL
            cache_path = self._get_cache_path(url)
            meta_path = self._get_metadata_path(cache_path)
            
            for path in [cache_path, meta_path]:
                if path.exists():
                    path.unlink()
                    deleted += 1
        else:
            # Clear all cache
            for html_file in self.cache_dir.rglob("*.html"):
                html_file.unlink()
                deleted += 1
                
                meta_file = self._get_metadata_path(html_file)
                if meta_file.exists():
                    meta_file.unlink()

        logger.info(f"Cleared {deleted} cached files")
        return deleted

    def get_stats(self) -> ScraperStats:
        """Get scraping statistics."""
        return self.stats

    def close(self) -> None:
        """Close the scraper session."""
        if self._session:
            self._session.close()
            self._session = None

    def __enter__(self) -> "Scraper":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        """Context manager exit."""
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────


def scrape_url(url: str, force_refresh: bool = False) -> ScrapedPage:
    """
    Convenience function to scrape a single URL.

    Args:
        url: URL to scrape
        force_refresh: If True, ignore cache

    Returns:
        ScrapedPage object
    """
    with Scraper() as scraper:
        return scraper.scrape(url, force_refresh=force_refresh)


def scrape_urls(urls: list[str], force_refresh: bool = False) -> list[ScrapedPage]:
    """
    Convenience function to scrape multiple URLs.

    Args:
        urls: List of URLs
        force_refresh: If True, ignore cache

    Returns:
        List of ScrapedPage objects
    """
    with Scraper() as scraper:
        return scraper.scrape_many(urls, force_refresh=force_refresh)
