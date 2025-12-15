"""
CLI Module - Command-line interface for IUE CourseCompass.
==========================================================

Provides CLI commands for:
- Scraping course data from IUE website
- Building vector index
- Querying the RAG system
- Running evaluation harness
- Managing data and indices

Usage:
    coursecompass --help
    coursecompass scrape --department se
    coursecompass index --provider sbert
    coursecompass query "What are the prerequisites for SE 301?"
    coursecompass eval --questions data/questions.json

Components:
- main: Typer CLI application
"""

from iue_coursecompass.cli.main import app, cli

__all__ = ["app", "cli"]
