"""
CLI Main - Typer command-line interface.
========================================

Commands:
- scrape: Scrape course data from IUE website
- index: Build vector index from scraped data
- query: Ask questions about courses
- eval: Run evaluation harness
- info: Show system information
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from iue_coursecompass.shared.logging import get_logger

logger = get_logger(__name__)

app = typer.Typer(
    name="coursecompass",
    help="""🧭 IUE CourseCompass - RAG System for IUE Engineering Courses

A Retrieval-Augmented Generation (RAG) system for querying course information
from Izmir University of Economics Engineering departments.

Supported Departments:
  • SE  - Software Engineering
  • CE  - Computer Engineering  
  • EEE - Electrical & Electronics Engineering
  • IE  - Industrial Engineering

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMMANDS OVERVIEW:

  scrape   Scrape course data from IUE ECTS Portal
           -d, --department   Filter by department (se/ce/eee/ie)
           --no-syllabi       Skip fetching detailed course content (faster)
           
  index    Build vector embeddings index from scraped data  
           -p, --provider     Embedding provider: sbert (local) or gemini (API)
           -r, --rebuild      Delete and rebuild index from scratch
           
  query    Ask questions about courses using RAG
           -d, --department   Filter results to specific department
           -k, --top-k        Number of chunks to retrieve (default: 5)
           --no-sources       Hide source citations
           
  eval     Run evaluation benchmark on question bank
           -q, --questions    Custom questions file (YAML/JSON)
           -k, --top-k        Chunks per query (default: 10, use 100 for full eval)
           -o, --output       Save results to JSON file
           -r, --retrieval-only  Skip LLM generation (faster)
           
  info     Show system configuration and index status
  
  gui      Launch interactive Streamlit web interface

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUICK START:

  coursecompass scrape                    # Step 1: Scrape all departments
  coursecompass index                     # Step 2: Build search index
  coursecompass query "What is SE 301?"   # Step 3: Ask questions
  
EVALUATION:

  coursecompass eval -k 100 -o results.json   # Full 60-question benchmark

Use 'coursecompass <command> --help' for detailed command options.
""",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Scrape Command
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def scrape(
    department: Optional[str] = typer.Option(
        None,
        "--department", "-d",
        help="Department to scrape (se, ce, eee, ie). Omit for all departments.",
    ),
    output_dir: Path = typer.Option(
        Path("data/processed"),
        "--output", "-o",
        help="Output directory for scraped/processed course data.",
    ),
    cache: bool = typer.Option(
        True,
        "--cache/--no-cache",
        help="Use cached pages if available.",
    ),
    fetch_syllabi: bool = typer.Option(
        True,
        "--syllabi/--no-syllabi",
        help="Fetch full syllabus content for each course.",
    ),
    curriculum: str = typer.Option(
        "2020-2024",
        "--curriculum", "-y",
        help="Curriculum year range (2020-2024 for ECTS portal).",
    ),
):
    """
    🌐 Scrape course data from IUE ECTS Portal (2020-2024 curriculum).

    Scrapes curriculum pages and optionally full course details for each course.
    By default scrapes ALL departments. Use -d to scrape a specific one.
    
    The ECTS portal (ects.ieu.edu.tr) contains the 2020-2024 curriculum data.
    
    Examples:
        coursecompass scrape              # Scrape all departments
        coursecompass scrape -d se        # Scrape only Software Engineering
        coursecompass scrape --no-syllabi # Only curriculum tables (faster)
    """
    import time
    from iue_coursecompass.shared.config import get_settings
    from iue_coursecompass.ingestion.scraper import Scraper
    from iue_coursecompass.ingestion.parser import ECTSCourseParser
    from iue_coursecompass.ingestion.cleaner import TextCleaner
    from iue_coursecompass.shared.utils import save_jsonl
    from iue_coursecompass.shared.schemas import CourseRecord

    settings = get_settings()

    # Determine departments to scrape
    if department:
        departments = [department.lower()]
    else:
        departments = settings.get_department_ids()

    console.print(Panel(
        f"[bold]Scraping Configuration (ECTS Portal)[/bold]\n"
        f"Source: ects.ieu.edu.tr (2020-2024 Curriculum)\n"
        f"Departments: {', '.join(d.upper() for d in departments)}\n"
        f"Fetch Course Details: {'Yes' if fetch_syllabi else 'No (curriculum only)'}\n"
        f"Curriculum: {curriculum}\n"
        f"Output: {output_dir}\n"
        f"Cache: {'enabled' if cache else 'disabled'}",
        title="🌐 Scrape ECTS Portal Courses",
    ))

    scraper = Scraper(cache_enabled=cache)
    ects_parser = ECTSCourseParser()
    cleaner = TextCleaner()

    output_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[CourseRecord] = []
    
    for dept in departments:
        dept_config = settings.get_department(dept)
        if not dept_config:
            console.print(f"[yellow]Department '{dept}' not found in config[/yellow]")
            continue
            
        curriculum_url = dept_config.curriculum_url
        if not curriculum_url:
            console.print(f"[yellow]No curriculum URL for {dept.upper()}[/yellow]")
            continue

        console.print(f"\n[bold blue]═══ {dept_config.full_name} ═══[/bold blue]")
        
        # Step 1: Fetch curriculum page from ECTS portal
        console.print(f"  Fetching curriculum from ECTS portal...")
        console.print(f"  URL: {curriculum_url}")
        page = scraper.scrape(curriculum_url)
        if not page.is_success:
            console.print(f"  [red]Failed to fetch curriculum page (status: {page.status_code})[/red]")
            continue
        
        # Step 2: Parse curriculum table using ECTS parser
        console.print(f"  Parsing curriculum table (ECTS format)...")
        courses = ects_parser.parse_curriculum_page(page.html, curriculum_url, dept, curriculum)
        console.print(f"  [green]Found {len(courses)} courses in curriculum[/green]")
        
        # Step 3: Optionally fetch full course details for each course
        if fetch_syllabi and dept_config.course_url_template:
            console.print(f"  Fetching course details from ECTS portal...")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
            ) as progress:
                task = progress.add_task(f"  {dept.upper()} course details", total=len(courses))
                
                for course in courses:
                    course_url = dept_config.get_course_url(course.course_code)
                    if course_url:
                        try:
                            course_page = scraper.scrape(course_url)
                            if course_page.is_success:
                                # Parse course details and merge into course record
                                ects_parser.parse_course_page(
                                    course_page.html, course_url, dept, curriculum, course
                                )
                            time.sleep(0.3)  # Rate limiting
                        except Exception as e:
                            logger.warning(f"Failed to scrape course {course.course_code}: {e}")
                    
                    progress.update(task, advance=1)
        
        # Step 4: Clean text content (only string fields, not lists)
        for course in courses:
            if course.description:
                course.description = cleaner.clean(course.description)
            if course.objectives:
                course.objectives = cleaner.clean(course.objectives)
            # learning_outcomes and weekly_topics are lists, clean each item
            if course.learning_outcomes:
                course.learning_outcomes = [cleaner.clean(item) for item in course.learning_outcomes]
            if course.weekly_topics:
                course.weekly_topics = [cleaner.clean(item) for item in course.weekly_topics]
        
        all_records.extend(courses)
        console.print(f"  [green]✓ Completed {dept.upper()}: {len(courses)} courses[/green]")

    # Save results
    if all_records:
        output_file = output_dir / "courses.jsonl"
        records_dict = [r.model_dump() for r in all_records]
        save_jsonl(output_file, records_dict)
        
        console.print(f"\n[bold green]{'═' * 50}[/bold green]")
        console.print(f"[bold green]✓ Saved {len(all_records)} total courses to {output_file}[/bold green]")
        
        # Summary by department
        dept_counts = {}
        for r in all_records:
            dept_counts[r.department] = dept_counts.get(r.department, 0) + 1
        for d, c in dept_counts.items():
            console.print(f"  • {d.upper()}: {c} courses")
    else:
        console.print("[yellow]No courses scraped.[/yellow]")


# ─────────────────────────────────────────────────────────────────────────────
# Index Command
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def index(
    input_file: Path = typer.Option(
        Path("data/processed/courses.jsonl"),
        "--input", "-i",
        help="Path to JSONL file containing scraped course data.",
    ),
    provider: str = typer.Option(
        None,
        "--provider", "-p",
        help="Embedding provider: 'sbert' (local, free) or 'gemini' (API, requires GOOGLE_API_KEY). Default: from config/settings.yaml.",
    ),
    collection: str = typer.Option(
        None,
        "--collection", "-c",
        help="ChromaDB collection name. Default: 'courses_<provider>'.",
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild", "-r",
        help="Delete existing index and rebuild from scratch. Use after re-scraping.",
    ),
):
    """
    📊 Build vector index from course data.

    Chunks course text into smaller pieces and creates vector embeddings
    for semantic search using ChromaDB.

    Embedding Providers:
      • sbert  - Local SBERT model (free, no API key needed, 384 dimensions)
      • gemini - Google Gemini API (requires GOOGLE_API_KEY, 768 dimensions)

    Examples:
        coursecompass index                    # Index with default settings
        coursecompass index -p gemini          # Use Gemini embeddings
        coursecompass index -r                 # Rebuild index from scratch
        coursecompass index -i custom.jsonl    # Index custom data file
    """
    from iue_coursecompass.shared.utils import load_jsonl
    from iue_coursecompass.shared.schemas import CourseRecord
    from iue_coursecompass.shared.config import get_settings
    from iue_coursecompass.ingestion.chunker import Chunker
    from iue_coursecompass.indexing.vector_store import VectorStore

    # Get provider from config if not specified
    settings = get_settings()
    if provider is None:
        provider = settings.embeddings.provider
    
    # Generate collection name if not specified
    if collection is None:
        collection = f"courses_{provider}"
    
    if not input_file.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        console.print("Run 'coursecompass scrape' first.")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]Indexing Configuration[/bold]\n"
        f"Input: {input_file}\n"
        f"Provider: {provider}\n"
        f"Collection: {collection}\n"
        f"Rebuild: {rebuild}",
        title="📊 Index",
    ))

    # Load courses
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading courses...", total=None)
        data = load_jsonl(input_file)
        courses = [CourseRecord(**item) for item in data]
        progress.remove_task(task)
        console.print(f"[green]✓ Loaded {len(courses)} courses[/green]")

        # Chunk
        task = progress.add_task("Chunking courses...", total=None)
        chunker = Chunker()
        chunks = []
        for course in courses:
            course_chunks = chunker.chunk_course(course)
            chunks.extend(course_chunks)
        progress.remove_task(task)
        console.print(f"[green]✓ Created {len(chunks)} chunks[/green]")

        # Index
        task = progress.add_task(f"Building {provider} embeddings...", total=None)
        store = VectorStore(
            collection_name=collection,
            embedding_provider=provider,
        )

        if rebuild:
            store.clear()

        store.add_chunks(chunks)
        progress.remove_task(task)

    console.print(f"\n[bold green]✓ Indexed {len(chunks)} chunks to '{collection}'[/bold green]")


# ─────────────────────────────────────────────────────────────────────────────
# Query Command
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def query(
    question: str = typer.Argument(
        ...,
        help="Natural language question about IUE courses (wrap in quotes).",
    ),
    department: Optional[str] = typer.Option(
        None,
        "--department", "-d",
        help="Filter to specific department: se, ce, eee, or ie.",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k", "-k",
        help="Number of source chunks to retrieve (1-100). Higher = more context but slower.",
    ),
    show_sources: bool = typer.Option(
        True,
        "--sources/--no-sources",
        help="Display the source chunks used to generate the answer.",
    ),
):
    """
    💬 Ask a question about IUE courses.

    Uses RAG (Retrieval-Augmented Generation) to:
      1. Embed your question as a vector
      2. Retrieve the most similar course chunks from the index
      3. Generate an answer using an LLM with retrieved context

    The system auto-detects semester references (e.g., "3rd semester").

    Examples:
        coursecompass query "What are the prerequisites for SE 301?"
        coursecompass query "List all 6 ECTS courses" -k 20
        coursecompass query "Machine learning courses" -d ce
        coursecompass query "Mandatory courses in semester 3 for SE" -k 10
    """
    from iue_coursecompass.rag.retriever import Retriever, extract_semester_from_query
    from iue_coursecompass.rag.generator import Generator
    from iue_coursecompass.rag.grounding import check_grounding

    console.print(f"\n[bold]Question:[/bold] {question}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Retrieve
        task = progress.add_task("Searching courses...", total=None)
        retriever = Retriever()
        departments = [department] if department else None
        
        # Auto-detect semester from question
        semester = extract_semester_from_query(question)
        if semester:
            logger.info(f"Auto-detected semester filter: {semester}")
        
        hits = retriever.retrieve(
            query=question, 
            top_k=top_k, 
            departments=departments,
            semester=semester
        )
        progress.remove_task(task)

        if not hits:
            console.print("[yellow]No relevant sources found.[/yellow]")
            raise typer.Exit(0)

        # Generate
        task = progress.add_task("Generating answer...", total=None)
        generator = Generator()
        response = generator.generate(query=question, hits=hits)
        progress.remove_task(task)

    # Display answer
    console.print(Panel(response.answer, title="💬 Answer", border_style="green"))

    # Grounding info
    grounding = check_grounding(response.answer, hits)
    status = "✓ Grounded" if grounding.is_grounded else "⚠ Low Confidence"
    console.print(f"\n[dim]{status} (score: {grounding.grounding_score:.2f})[/dim]")

    # Show sources
    if show_sources:
        console.print("\n[bold]📚 Sources:[/bold]")
        table = Table(show_header=True)
        table.add_column("Course", style="cyan")
        table.add_column("Department")
        table.add_column("Score", justify="right")

        for hit in hits[:5]:
            table.add_row(
                f"{hit.course_code} - {hit.course_title[:30]}",
                hit.department.upper(),
                f"{hit.score:.3f}",
            )

        console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Eval Command
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def eval(
    questions_file: Optional[Path] = typer.Option(
        None,
        "--questions", "-q",
        help="Path to YAML/JSON file with evaluation questions. Default: built-in question bank.",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save detailed results to JSON file for analysis.",
    ),
    top_k: int = typer.Option(
        10,
        "--top-k", "-k",
        help="Number of chunks to retrieve per question (10-100 typical). Higher = better recall but slower.",
    ),
    skip_generation: bool = typer.Option(
        False,
        "--retrieval-only", "-r",
        help="Skip answer generation, only evaluate retrieval metrics (much faster).",
    ),
    sample_size: Optional[int] = typer.Option(
        None,
        "--sample", "-s",
        help="Run on random sample of N questions instead of full set.",
    ),
):
    """
    📊 Run evaluation harness on question bank.

    Evaluates RAG system quality using predefined questions with known answers.

    Metrics Computed:
      • Retrieval: MRR, Recall@K, Hit Rate, Precision@1
      • Answer Quality: Grounding Rate, Citation Accuracy, Trap Accuracy
      • Completeness: Answer Completeness, Hallucination Rate

    Question Categories:
      • A: Mandatory courses by semester (curriculum lookup)
      • B: Simple factual queries
      • C: Comparison/analysis queries
      • D: Counting queries
      • E: Trap questions (non-existent courses)

    Examples:
        coursecompass eval                                    # Run with defaults
        coursecompass eval -k 100 -o results.json             # Full eval, save results
        coursecompass eval -q custom_questions.yaml           # Use custom questions
        coursecompass eval -r                                 # Retrieval metrics only (fast)
        coursecompass eval -s 10                              # Quick test with 10 questions
    """
    from iue_coursecompass.evaluation import (
        QuestionBank,
        EvaluationRunner,
    )
    from iue_coursecompass.evaluation.questions import create_sample_questions

    # Load questions
    if questions_file and questions_file.exists():
        questions = QuestionBank.from_file(questions_file)
        console.print(f"[green]✓ Loaded {len(questions)} questions from {questions_file}[/green]")
    else:
        console.print("[yellow]Using sample questions...[/yellow]")
        questions = create_sample_questions()

    # Sample if requested
    if sample_size and sample_size < len(questions):
        question_list = questions.sample(sample_size)
        console.print(f"[dim]Sampled {sample_size} questions[/dim]")
    else:
        question_list = questions.questions

    console.print(Panel(
        f"[bold]Evaluation Configuration[/bold]\n"
        f"Questions: {len(question_list)}\n"
        f"Top-K: {top_k}\n"
        f"Mode: {'Retrieval Only' if skip_generation else 'Full RAG'}",
        title="📊 Evaluate",
    ))

    # Run evaluation
    runner = EvaluationRunner(
        top_k=top_k,
        skip_generation=skip_generation,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(question_list))

        def callback(current, total, qid):
            progress.update(task, completed=current, description=f"Evaluating {qid}...")

        result = runner.evaluate(question_list, progress_callback=callback)

    # Display results
    console.print("\n" + result.summary())

    # Save if requested
    if output_file:
        result.save(output_file)
        console.print(f"\n[green]✓ Results saved to {output_file}[/green]")


# ─────────────────────────────────────────────────────────────────────────────
# Info Command
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def info():
    """
    ℹ️ Show system information and configuration.

    Displays:
      • Version information
      • Configured departments
      • Data paths and their existence status
      • Current embedding provider settings

    Useful for debugging and verifying setup.
    """
    from iue_coursecompass.shared.config import get_settings
    from iue_coursecompass import __version__

    settings = get_settings()

    console.print(Panel(
        f"[bold]IUE CourseCompass[/bold]\n"
        f"Version: {__version__}\n"
        f"Config: config/settings.yaml",
        title="ℹ️ Info",
    ))

    # Departments
    console.print("\n[bold]Configured Departments:[/bold]")
    table = Table()
    table.add_column("Code")
    table.add_column("Name")

    for dept in settings.departments:
        table.add_row(dept.id.upper(), dept.name)

    console.print(table)

    # Paths
    console.print("\n[bold]Data Paths:[/bold]")
    resolved_paths = settings.resolved_paths
    path_dict = {
        "data_dir": resolved_paths.data_dir,
        "raw_dir": resolved_paths.raw_dir,
        "processed_dir": resolved_paths.processed_dir,
        "courses_file": resolved_paths.courses_file,
        "chunks_file": resolved_paths.chunks_file,
        "index_dir": resolved_paths.index_dir,
    }
    for name, path in path_dict.items():
        exists = "✓" if path.exists() else "✗"
        console.print(f"  {name}: {path} [{exists}]")


# ─────────────────────────────────────────────────────────────────────────────
# GUI Command
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def gui():
    """
    🖥️ Launch Streamlit web interface.

    Opens an interactive GUI in your browser with:
      • Chat interface for asking questions
      • Pipeline management (scrape, index)
      • Evaluation dashboard
      • Source visualization

    The GUI runs at http://localhost:8501 by default.
    Press Ctrl+C to stop the server.
    """
    import subprocess
    import sys

    app_path = Path(__file__).parent.parent / "app" / "streamlit_app.py"

    console.print("[bold]🚀 Launching CourseCompass GUI...[/bold]")
    console.print(f"[dim]Running: streamlit run {app_path}[/dim]\n")

    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────


def cli():
    """CLI entry point."""
    app()


if __name__ == "__main__":
    cli()
