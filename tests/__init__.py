"""
Tests Package - Unit and integration tests for IUE CourseCompass.
=================================================================

Test modules:
- test_ingestion: Scraper, parser, cleaner, chunker tests
- test_indexing: Embedding and vector store tests
- test_rag: Retriever, generator, grounding tests
- test_evaluation: Metrics and runner tests

Run tests with:
    pytest tests/
    pytest tests/ -v --cov=src/iue_coursecompass
    make test
"""
