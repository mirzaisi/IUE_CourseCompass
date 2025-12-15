# 🧭 IUE CourseCompass

A production-quality **Retrieval-Augmented Generation (RAG)** system for querying Izmir University of Economics (IUE) Faculty of Engineering course information.

## ✨ Features

- 🌐 **Web Scraping**: Automated scraping of IUE course catalogs and curricula
- 📝 **Smart Chunking**: Semantic text chunking with overlap for better retrieval
- 🔍 **Vector Search**: ChromaDB-powered semantic search with SBERT or Gemini embeddings
- 🤖 **Grounded Generation**: Gemini-powered answers with citation requirements
- 🛡️ **Hallucination Prevention**: Grounding verification and trap question detection
- 📊 **Evaluation Harness**: Comprehensive metrics (MRR, Recall@K, grounding rate)
- 🖥️ **Streamlit GUI**: Interactive web interface for queries and comparisons
- ⌨️ **CLI**: Full command-line interface for all operations

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interfaces                          │
│                  ┌─────────────┐  ┌─────────────┐               │
│                  │ Streamlit   │  │    CLI      │               │
│                  │    GUI      │  │  (Typer)    │               │
│                  └──────┬──────┘  └──────┬──────┘               │
└─────────────────────────┼────────────────┼──────────────────────┘
                          │                │
┌─────────────────────────▼────────────────▼──────────────────────┐
│                         RAG Pipeline                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │ Retriever│→ │ Prompts  │→ │Generator │→ │  Grounding   │    │
│  └────┬─────┘  └──────────┘  │ (Gemini) │  │  Checker     │    │
│       │                      └──────────┘  └──────────────┘    │
└───────┼─────────────────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────────────────┐
│                      Indexing Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐      │
│  │ VectorStore  │  │  Embeddings  │  │     Manifest     │      │
│  │  (ChromaDB)  │  │ SBERT/Gemini │  │     Manager      │      │
│  └──────────────┘  └──────────────┘  └──────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
        ▲
┌───────┴─────────────────────────────────────────────────────────┐
│                     Ingestion Pipeline                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Scraper  │→ │  Parser  │→ │ Cleaner  │→ │ Chunker  │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Gemini API Key](https://makersuite.google.com/app/apikey) (for generation)

### Installation

```bash
# Clone the repository
git clone https://github.com/mirzaisi/IUE_CourseCompass.git
cd IUE_CourseCompass

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package with dependencies
pip install -e ".[dev]"

# Set up environment variables
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Basic Usage

#### 1. Scrape Course Data

```bash
# Scrape all departments
coursecompass scrape

# Scrape specific department
coursecompass scrape --department se
```

#### 2. Build Vector Index

```bash
# Build index with SBERT embeddings (free, local)
coursecompass index --provider sbert

# Or with Gemini embeddings (requires API key)
coursecompass index --provider gemini
```

#### 3. Ask Questions

```bash
# CLI query
coursecompass query "What are the prerequisites for SE 301?"

# With department filter
coursecompass query "How many ECTS credits in year 3?" --department se
```

#### 4. Launch GUI

```bash
# Start Streamlit interface
coursecompass gui
# Or: make gui
```

## 📖 Usage Examples

### CLI Commands

```bash
# Show system information
coursecompass info

# Run evaluation harness
coursecompass eval --questions data/questions.json

# Query with specific top-k
coursecompass query "Compare SE and CE programming courses" --top-k 10
```

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

# Generate answer
response = generator.generate(query, hits)
print(response.answer)

# Verify grounding
grounding = check_grounding(response.answer, hits)
print(f"Grounded: {grounding.is_grounded} (score: {grounding.grounding_score:.2f})")
```

### Evaluation

```python
from iue_coursecompass.evaluation import (
    QuestionBank,
    EvaluationRunner,
    run_evaluation,
)

# Load questions
questions = QuestionBank.from_file("data/questions.json")

# Run evaluation
result = run_evaluation(questions, output_path="results.json")
print(result.summary())
```

## 📁 Project Structure

```
IUE_CourseCompass/
├── config/
│   └── settings.yaml        # Configuration file
├── src/iue_coursecompass/
│   ├── ingestion/           # Scraping, parsing, chunking
│   │   ├── scraper.py       # Web scraper with caching
│   │   ├── parser.py        # HTML parser
│   │   ├── cleaner.py       # Text normalization
│   │   └── chunker.py       # Semantic chunking
│   ├── indexing/            # Embeddings and vector storage
│   │   ├── embeddings_*.py  # SBERT/Gemini providers
│   │   ├── vector_store.py  # ChromaDB wrapper
│   │   └── manifest.py      # Index versioning
│   ├── rag/                 # RAG pipeline
│   │   ├── retriever.py     # Chunk retrieval
│   │   ├── prompts.py       # Prompt templates
│   │   ├── generator.py     # LLM generation
│   │   ├── grounding.py     # Citation verification
│   │   └── quantitative.py  # Counting queries
│   ├── evaluation/          # Evaluation harness
│   │   ├── questions.py     # Question bank
│   │   ├── metrics.py       # MRR, Recall@K, etc.
│   │   └── runner.py        # Evaluation execution
│   ├── app/                 # Streamlit GUI
│   │   └── streamlit_app.py
│   ├── cli/                 # Command-line interface
│   │   └── main.py
│   └── shared/              # Shared utilities
│       ├── config.py        # Configuration loader
│       ├── schemas.py       # Pydantic models
│       └── utils.py         # Helper functions
├── tests/                   # Unit and integration tests
├── data/                    # Data directory (gitignored)
├── pyproject.toml           # Package configuration
├── Makefile                 # Common commands
└── README.md
```

## ⚙️ Configuration

Configuration is managed via `config/settings.yaml`:

```yaml
departments:
  se:
    name: "Software Engineering"
    curriculum_url: "https://..."
  ce:
    name: "Computer Engineering"
    curriculum_url: "https://..."

embeddings:
  provider: "sbert"  # or "gemini"
  model: "all-MiniLM-L6-v2"

retrieval:
  top_k: 5
  score_threshold: 0.3

generation:
  model: "gemini-1.5-flash"
  temperature: 0.3
```

Environment variables (`.env`):

```bash
GEMINI_API_KEY=your_api_key_here
COURSECOMPASS_ENV=development
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
pytest tests/ --cov=src/iue_coursecompass --cov-report=html

# Run specific test file
pytest tests/test_rag.py -v

# Skip integration tests
pytest tests/ -m "not integration"
```

## 📊 Evaluation Metrics

The evaluation harness measures:

| Metric | Description |
|--------|-------------|
| **MRR** | Mean Reciprocal Rank of first relevant result |
| **Recall@K** | Fraction of relevant docs in top-K |
| **Precision@K** | Fraction of top-K that are relevant |
| **Hit Rate** | Queries with at least one relevant result |
| **Grounding Rate** | Answers properly grounded in sources |
| **Trap Accuracy** | Correctly rejecting non-existent topics |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`make lint && make test`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linters
make lint

# Format code
make format

# Type check
make typecheck
```


## 🙏 Acknowledgments

- [Izmir University of Economics](https://www.iue.edu.tr/) for course data
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Google Gemini](https://deepmind.google/technologies/gemini/) for LLM generation
- [Streamlit](https://streamlit.io/) for the GUI framework

---
