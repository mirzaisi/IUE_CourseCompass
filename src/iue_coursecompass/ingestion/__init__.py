"""
Ingestion Module - Scrape, parse, clean, and chunk course data.
===============================================================

This module handles the entire data ingestion pipeline:

- scraper: Web scraping with caching, rate limiting, and retries
- parser: HTML parsing to extract course information
- cleaner: Text cleaning and normalization
- chunker: Semantic chunking for vector storage

Pipeline flow:
    URLs → Scraper → Raw HTML → Parser → CourseRecords → Cleaner → Chunker → ChunkRecords
"""

from iue_coursecompass.ingestion.scraper import Scraper, ScrapedPage
from iue_coursecompass.ingestion.parser import (
    CourseParser,
    IUECourseParser,
    ECTSCourseParser,
    EBSCourseParser,  # Alias for backward compatibility
    parse_course_page,
    parse_curriculum_page,
    get_parser,
)
from iue_coursecompass.ingestion.cleaner import TextCleaner, clean_course_record
from iue_coursecompass.ingestion.chunker import Chunker, chunk_course

__all__ = [
    # Scraper
    "Scraper",
    "ScrapedPage",
    # Parser
    "CourseParser",
    "IUECourseParser",
    "ECTSCourseParser",
    "EBSCourseParser",  # Alias for backward compatibility
    "parse_course_page",
    "parse_curriculum_page",
    "get_parser",
    # Cleaner
    "TextCleaner",
    "clean_course_record",
    # Chunker
    "Chunker",
    "chunk_course",
]
