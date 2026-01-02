# 🧭 IUE CourseCompass

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A Retrieval-Augmented Generation (RAG) system for querying course information at Izmir University of Economics' Faculty of Engineering.**

Ever had trouble navigating course catalogs, figuring out prerequisites, or comparing programs across departments? CourseCompass helps you ask natural language questions about courses and get accurate, citation-backed answers.

---

## 📋 Table of Contents

- [Features](#-features)
- [How It Works](#-how-it-works)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Evaluation](#-evaluation)
- [Development](#-development)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🌐 **Web Scraping** | Automated scraping of IUE ECTS course catalogs with rate limiting and caching |
| 📝 **Smart Chunking** | Semantic text chunking that respects document structure and maintains context |
| 🔍 **Vector Search** | ChromaDB-powered semantic search with SBERT (free, local) or Gemini embeddings |
| 🤖 **Grounded Generation** | Gemini-powered answers that cite their sources - no hallucinations |
| 🛡️ **Hallucination Prevention** | Built-in grounding verification and "trap question" detection |
| 📊 **Evaluation Suite** | Comprehensive metrics including MRR, Recall@K, and grounding rate |
| 🖥️ **Streamlit GUI** | Interactive web interface for queries, comparisons, and data management |
| ⌨️ **CLI** | Full command-line interface for automation and scripting |

---

## 🔬 How It Works

CourseCompass uses a RAG (Retrieval-Augmented Generation) pipeline to answer questions about IUE engineering courses:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           User Interfaces                                   │
│                    ┌─────────────┐      ┌─────────────┐                     │
│                    │  Streamlit  │      │     CLI     │                     │
│                    │     GUI     │      │   (Typer)   │                     │
│                    └──────┬──────┘      └──────┬──────┘                     │
└───────────────────────────┼──────────────────────┼──────────────────────────┘
                            │                      │
┌───────────────────────────▼──────────────────────▼──────────────────────────┐
│                          RAG Pipeline                                       │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐             │
│   │ Retriever│ → │ Prompts  │ → │Generator │ → │  Grounding  │             │
│   │          │   │ Builder  │   │ (Gemini) │   │   Checker   │             │
│   └────┬─────┘   └──────────┘   └──────────┘   └─────────────┘             │
└────────┼────────────────────────────────────────────────────────────────────┘
         │
┌────────▼────────────────────────────────────────────────────────────────────┐
│                        Indexing Layer                                       │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐               │
│   │ Vector Store │   │  Embeddings  │   │     Manifest     │               │
│   │  (ChromaDB)  │   │ SBERT/Gemini │   │     Manager      │               │
│   └──────────────┘   └──────────────┘   └──────────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
         ▲
┌────────┴────────────────────────────────────────────────────────────────────┐
│                       Ingestion Pipeline                                    │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐                │
│   │ Scraper  │ → │  Parser  │ → │ Cleaner  │ → │ Chunker  │                │
│   │ (ECTS)   │   │  (HTML)  │   │  (Text)  │   │(Semantic)│                │
│   └──────────┘   └──────────┘   └──────────┘   └──────────┘                │
└─────────────────────────────────────────────────────────────────────────────┘
```

**The flow:**
1. **Scrape** → Pull course data from IUE's ECTS portal
2. **Parse & Clean** → Extract structured information from HTML
3. **Chunk** → Split content into semantic chunks for better retrieval
4. **Embed** → Convert chunks to vectors using SBERT or Gemini
5. **Retrieve** → Find relevant chunks using semantic similarity
6. **Generate** → Create grounded answers with citations

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Gemini API Key** - Get one free at [Google AI Studio](https://makersuite.google.com/app/apikey) (required for answer generation)

### Installation

```bash
# Clone the repository
git clone https://github.com/mirzaisi/IUE_CourseCompass.git
cd IUE_CourseCompass

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e ".[dev]"

# Set up your API key
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### First Run

```bash
# 1. Scrape course data (takes ~5 minutes)
coursecompass scrape

# 2. Build the search index
coursecompass index --provider sbert  # Free, runs locally

# 3. Start asking questions!
coursecompass query "What are the prerequisites for SE 301?"

# Or launch the web interface
coursecompass gui
```

---

## 📖 Usage

### Command Line Interface

```bash
# Get system info and status
coursecompass info

# Scrape all departments
coursecompass scrape

# Scrape specific department
coursecompass scrape --department se

# Build index with different embedding providers
coursecompass index --provider sbert   # Free, local (default)
coursecompass index --provider gemini  # Requires API key, better quality

# Ask questions
coursecompass query "What courses cover machine learning?"
coursecompass query "Compare SE and CE database courses" --department se --department ce

# Run the evaluation suite
coursecompass eval --questions data/evaluation_questions.json
```

### Web Interface

Launch the Streamlit GUI for an interactive experience:

```bash
coursecompass gui
# Or: make app
```

The GUI provides:
- 💬 Natural language Q&A with source citations
- 🔄 Cross-department course comparisons
- 📊 Retrieval statistics and confidence scores
- ⚙️ Data management (scrape, index, configure)

### Python API

```python
from iue_coursecompass.rag import Retriever, Generator
from iue_coursecompass.rag.grounding import check_grounding

# Initialize components
retriever = Retriever()
generator = Generator()

# Retrieve relevant chunks
query = "What is SE 301 about?"
hits = retriever.retrieve(query, top_k=5)

# Generate a grounded answer
response = generator.generate(query, hits)
print(response.answer)

# Verify the answer is grounded in sources
grounding = check_grounding(response.answer, hits)
print(f"Grounded: {grounding.is_grounded} (score: {grounding.grounding_score:.2f})")
```

---

## ⚙️ Configuration

Configuration is split between `config/settings.yaml` and environment variables:

### Environment Variables (`.env`)

```bash
# Required for answer generation
GEMINI_API_KEY=your-api-key-here

# Optional: Override defaults
EMBEDDING_PROVIDER=sbert  # or "gemini"
GEMINI_MODEL=gemini-2.0-flash-exp
RETRIEVAL_TOP_K=5
```

### Settings File (`config/settings.yaml`)

```yaml
# Departments to scrape
departments:
  - id: "se"
    name: "Software Engineering"
  - id: "ce" 
    name: "Computer Engineering"
  - id: "eee"
    name: "Electrical & Electronics Engineering"
  - id: "ie"
    name: "Industrial Engineering"

# Retrieval settings
retrieval:
  top_k: 5
  similarity_threshold: 0.3

# Generation settings
generation:
  model_name: "gemini-2.0-flash-exp"
  temperature: 0.3
```

---

## 📁 Project Structure

```
IUE_CourseCompass/
├── config/
│   └── settings.yaml          # Main configuration
├── src/iue_coursecompass/
│   ├── ingestion/             # Data pipeline
│   │   ├── scraper.py         # Web scraper with caching
│   │   ├── parser.py          # HTML → structured data
│   │   ├── cleaner.py         # Text normalization
│   │   └── chunker.py         # Semantic chunking
│   ├── indexing/              # Vector storage
│   │   ├── embeddings_sbert.py   # Local SBERT embeddings
│   │   ├── embeddings_gemini.py  # Gemini API embeddings
│   │   ├── vector_store.py       # ChromaDB wrapper
│   │   └── manifest.py           # Index versioning
│   ├── rag/                   # RAG pipeline
│   │   ├── retriever.py       # Semantic search
│   │   ├── prompts.py         # Prompt engineering
│   │   ├── generator.py       # LLM generation
│   │   ├── grounding.py       # Citation verification
│   │   └── quantitative.py    # Counting queries
│   ├── evaluation/            # Testing & metrics
│   │   ├── questions.py       # Question bank
│   │   ├── metrics.py         # MRR, Recall@K, etc.
│   │   └── runner.py          # Evaluation harness
│   ├── app/                   # Streamlit GUI
│   └── cli/                   # Command-line interface
├── tests/                     # Unit & integration tests
├── data/                      # Data directory (gitignored)
├── pyproject.toml             # Package config & dependencies
├── Makefile                   # Common commands
└── README.md
```

---

## 📊 Evaluation

The evaluation suite measures both retrieval quality and generation accuracy:

| Metric | What it Measures |
|--------|------------------|
| **MRR** | Mean Reciprocal Rank - how high relevant results appear |
| **Recall@K** | Fraction of relevant docs found in top-K results |
| **Precision@K** | Fraction of top-K results that are relevant |
| **Hit Rate** | Queries with at least one relevant result |
| **Grounding Rate** | Answers properly cited from sources |
| **Trap Accuracy** | Correctly refusing to answer unanswerable questions |

Run the evaluation:

```bash
# Run full evaluation
coursecompass eval --questions data/evaluation_questions.json

# Output results to file
coursecompass eval --output results.json
```

---

## 🛠️ Development

### Setup

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks (optional)
pre-commit install
```

### Code Quality

```bash
# Lint code
make lint

# Format code
make format

# Type checking
make typecheck

# Run all checks
make check
```

### Testing

```bash
# Run tests
make test

# Run with coverage
make test-cov
pytest tests/ -v --cov=src/iue_coursecompass --cov-report=html
```

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes
4. **Test** your changes (`make lint && make test`)
5. **Commit** (`git commit -m 'Add amazing feature'`)
6. **Push** (`git push origin feature/amazing-feature`)
7. **Open** a Pull Request

Please make sure your code passes linting and tests before submitting.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Izmir University of Economics](https://www.iue.edu.tr/) - for the course data
- [Sentence Transformers](https://www.sbert.net/) - for local embeddings
- [ChromaDB](https://www.trychroma.com/) - for vector storage
- [Google Gemini](https://deepmind.google/technologies/gemini/) - for LLM generation
- [Streamlit](https://streamlit.io/) - for the web interface
- [Typer](https://typer.tiangolo.com/) - for the CLI framework

---

<p align="center">
  Made with ☕ for IUE Engineering students
</p>
