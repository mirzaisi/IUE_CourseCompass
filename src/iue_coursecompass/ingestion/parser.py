"""
Parser Module - Extract course information from HTML pages.
==========================================================

Parses scraped HTML to extract structured course data.
Uses BeautifulSoup with configurable selectors.

NOTE: Selectors are marked with TODO comments and should be
updated based on the actual IUE website structure.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from bs4 import BeautifulSoup, Tag

from iue_coursecompass.shared.logging import get_logger
from iue_coursecompass.shared.schemas import CourseRecord, CourseType

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Selector Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ParserSelectors:
    """
    CSS selectors for extracting course information.

    TODO: Update these selectors based on actual IUE website structure.
    These are placeholder patterns that should be customized.
    """

    # Course listing page selectors
    course_list_container: str = ".course-list, .curriculum-table, #courses"
    course_list_item: str = ".course-item, tr.course-row, .course-entry"
    course_link: str = "a[href*='course'], a[href*='ders']"

    # Course detail page selectors
    course_code: str = ".course-code, .ders-kodu, h1.title span.code, #courseCode"
    course_title: str = ".course-title, .ders-adi, h1.title, #courseTitle"
    
    # Content sections
    objectives: str = "#objectives, .course-objectives, .amaclar, [data-section='objectives']"
    description: str = "#description, .course-description, .aciklama, [data-section='description']"
    prerequisites: str = "#prerequisites, .prerequisites, .onkosul, [data-section='prerequisites']"
    weekly_topics: str = "#weekly-topics, .weekly-schedule, .haftalik-konular, table.topics"
    learning_outcomes: str = "#outcomes, .learning-outcomes, .kazanimlar, [data-section='outcomes']"
    assessment: str = "#assessment, .assessment-methods, .degerlendirme, [data-section='assessment']"

    # Credit information
    ects_credits: str = ".ects, .ects-credit, [data-field='ects']"
    local_credits: str = ".local-credit, .kredi, [data-field='credit']"
    
    # Classification
    semester_info: str = ".semester, .donem, [data-field='semester']"
    course_type_info: str = ".course-type, .ders-turu, [data-field='type']"

    # Table selectors (for curriculum tables)
    curriculum_table: str = "table.curriculum, table.ders-plani, .curriculum-table"
    table_row: str = "tr"
    table_cell: str = "td, th"


@dataclass
class ParsedCourseData:
    """Intermediate parsed data before creating CourseRecord."""

    course_code: Optional[str] = None
    course_title: Optional[str] = None
    objectives: Optional[str] = None
    description: Optional[str] = None
    prerequisites: Optional[str] = None
    weekly_topics: Optional[list[str]] = None
    learning_outcomes: Optional[list[str]] = None
    assessment_methods: Optional[str] = None
    ects: Optional[float] = None
    local_credits: Optional[float] = None
    semester: Optional[int] = None
    course_type: Optional[str] = None
    raw_html: str = ""
    parse_errors: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Parser Class
# ─────────────────────────────────────────────────────────────────────────────


class CourseParser:
    """
    Parser for extracting course information from HTML.

    Supports:
    - Single course detail pages
    - Curriculum listing pages
    - Flexible selector configuration

    Example:
        >>> parser = CourseParser()
        >>> course = parser.parse_course_page(html, url, "se", "2023-2024")
        >>> print(course.course_code)
    """

    def __init__(self, selectors: Optional[ParserSelectors] = None):
        """
        Initialize the parser.

        Args:
            selectors: Custom selector configuration (uses defaults if None)
        """
        self.selectors = selectors or ParserSelectors()

    def _create_soup(self, html: str) -> BeautifulSoup:
        """Create BeautifulSoup object from HTML."""
        return BeautifulSoup(html, "lxml")

    def _extract_text(
        self,
        soup: BeautifulSoup,
        selector: str,
        default: Optional[str] = None,
    ) -> Optional[str]:
        """
        Extract text content using a CSS selector.

        Args:
            soup: BeautifulSoup object
            selector: CSS selector (can be comma-separated for multiple options)
            default: Default value if not found

        Returns:
            Extracted text or default
        """
        # Try each selector in the comma-separated list
        for sel in selector.split(","):
            sel = sel.strip()
            element = soup.select_one(sel)
            if element:
                text = element.get_text(strip=True)
                if text:
                    return text

        return default

    def _extract_html(
        self,
        soup: BeautifulSoup,
        selector: str,
    ) -> Optional[str]:
        """Extract inner HTML content using a CSS selector."""
        for sel in selector.split(","):
            sel = sel.strip()
            element = soup.select_one(sel)
            if element:
                return str(element)
        return None

    def _extract_list(
        self,
        soup: BeautifulSoup,
        selector: str,
    ) -> Optional[list[str]]:
        """
        Extract a list of text items (e.g., weekly topics, outcomes).

        Handles:
        - <ul>/<ol> lists
        - Tables with rows
        - Numbered paragraphs
        """
        for sel in selector.split(","):
            sel = sel.strip()
            container = soup.select_one(sel)
            if not container:
                continue

            items = []

            # Try list items first
            list_items = container.select("li")
            if list_items:
                items = [li.get_text(strip=True) for li in list_items if li.get_text(strip=True)]

            # Try table rows
            if not items:
                rows = container.select("tr")
                for row in rows:
                    cells = row.select("td")
                    if cells:
                        # Combine cell text
                        row_text = " - ".join(
                            cell.get_text(strip=True) for cell in cells if cell.get_text(strip=True)
                        )
                        if row_text:
                            items.append(row_text)

            # Try paragraphs or divs
            if not items:
                paragraphs = container.select("p, div.item, .topic-item")
                items = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]

            if items:
                return items

        return None

    def _extract_number(
        self,
        soup: BeautifulSoup,
        selector: str,
        default: Optional[float] = None,
    ) -> Optional[float]:
        """Extract a numeric value from text."""
        text = self._extract_text(soup, selector)
        if not text:
            return default

        # Try to extract number from text
        # Handles formats like "6 ECTS", "ECTS: 6", "6.0", etc.
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

        return default

    def _extract_semester(self, soup: BeautifulSoup) -> Optional[int]:
        """Extract semester number."""
        text = self._extract_text(soup, self.selectors.semester_info)
        if not text:
            return None

        # Try to extract semester number
        # Handles: "Semester 3", "3rd Semester", "Dönem 3", etc.
        match = re.search(r"(\d+)", text)
        if match:
            semester = int(match.group(1))
            if 1 <= semester <= 8:
                return semester

        return None

    def _extract_course_type(self, soup: BeautifulSoup) -> CourseType:
        """Determine course type from page content."""
        text = self._extract_text(soup, self.selectors.course_type_info, "")
        if text:
            text_lower = text.lower()

            if "mandatory" in text_lower or "zorunlu" in text_lower:
                return CourseType.MANDATORY
            elif "technical elective" in text_lower or "teknik seçmeli" in text_lower:
                return CourseType.TECHNICAL_ELECTIVE
            elif "non-technical" in text_lower or "teknik dışı" in text_lower:
                return CourseType.NON_TECHNICAL_ELECTIVE
            elif "elective" in text_lower or "seçmeli" in text_lower:
                return CourseType.ELECTIVE

        return CourseType.UNKNOWN

    def _parse_course_code_title(
        self,
        soup: BeautifulSoup,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Extract course code and title.

        Handles various formats:
        - Separate elements for code and title
        - Combined "SE 301 - Software Engineering" format
        - Table cells
        """
        # Try separate selectors
        code = self._extract_text(soup, self.selectors.course_code)
        title = self._extract_text(soup, self.selectors.course_title)

        # If we have both, return them
        if code and title:
            # Clean code from title if duplicated
            if title.startswith(code):
                title = title[len(code):].strip(" -:")
            return code, title

        # Try to parse combined format from title
        if title and not code:
            # Pattern: "SE 301 - Software Engineering" or "SE301: Title"
            match = re.match(
                r"^([A-Z]{2,4}\s*\d{3}[A-Z]?)\s*[-:]\s*(.+)$",
                title,
                re.IGNORECASE,
            )
            if match:
                return match.group(1).strip(), match.group(2).strip()

        # Try h1 or main title
        h1 = soup.select_one("h1")
        if h1:
            h1_text = h1.get_text(strip=True)
            match = re.match(
                r"^([A-Z]{2,4}\s*\d{3}[A-Z]?)\s*[-:]\s*(.+)$",
                h1_text,
                re.IGNORECASE,
            )
            if match:
                return match.group(1).strip(), match.group(2).strip()

        return code, title

    def parse_course_page(
        self,
        html: str,
        source_url: str,
        department: str,
        year_range: str,
    ) -> Optional[CourseRecord]:
        """
        Parse a single course detail page.

        Args:
            html: Raw HTML content
            source_url: URL of the page
            department: Department ID
            year_range: Academic year range

        Returns:
            CourseRecord or None if parsing fails
        """
        if not html or not html.strip():
            logger.warning(f"Empty HTML for {source_url}")
            return None

        soup = self._create_soup(html)
        parsed = ParsedCourseData(raw_html=html)

        # Extract course code and title
        code, title = self._parse_course_code_title(soup)
        parsed.course_code = code
        parsed.course_title = title

        if not parsed.course_code:
            logger.warning(f"Could not extract course code from {source_url}")
            parsed.parse_errors.append("Missing course code")

        if not parsed.course_title:
            logger.warning(f"Could not extract course title from {source_url}")
            parsed.parse_errors.append("Missing course title")

        # Extract content sections
        parsed.objectives = self._extract_text(soup, self.selectors.objectives)
        parsed.description = self._extract_text(soup, self.selectors.description)
        parsed.prerequisites = self._extract_text(soup, self.selectors.prerequisites)
        parsed.weekly_topics = self._extract_list(soup, self.selectors.weekly_topics)
        parsed.learning_outcomes = self._extract_list(soup, self.selectors.learning_outcomes)
        parsed.assessment_methods = self._extract_text(soup, self.selectors.assessment)

        # Extract credits
        parsed.ects = self._extract_number(soup, self.selectors.ects_credits)
        parsed.local_credits = self._extract_number(soup, self.selectors.local_credits)

        # Extract classification
        parsed.semester = self._extract_semester(soup)
        course_type = self._extract_course_type(soup)

        # Create CourseRecord if we have minimum required data
        if not parsed.course_code or not parsed.course_title:
            logger.error(f"Failed to parse course from {source_url}: missing required fields")
            return None

        try:
            course = CourseRecord(
                course_code=parsed.course_code,
                course_title=parsed.course_title,
                department=department,
                year_range=year_range,
                course_type=course_type,
                semester=parsed.semester,
                ects=parsed.ects,
                local_credits=parsed.local_credits,
                objectives=parsed.objectives,
                description=parsed.description,
                prerequisites=parsed.prerequisites,
                weekly_topics=parsed.weekly_topics,
                learning_outcomes=parsed.learning_outcomes,
                assessment_methods=parsed.assessment_methods,
                source_url=source_url,
                scraped_at=datetime.utcnow(),
            )

            logger.debug(f"Parsed course: {course.course_code} - {course.course_title}")
            return course

        except Exception as e:
            logger.error(f"Failed to create CourseRecord from {source_url}: {e}")
            return None

    def parse_curriculum_page(
        self,
        html: str,
        source_url: str,
        department: str,
        year_range: str,
    ) -> list[CourseRecord]:
        """
        Parse a curriculum listing page with multiple courses.

        Args:
            html: Raw HTML content
            source_url: URL of the page
            department: Department ID
            year_range: Academic year range

        Returns:
            List of CourseRecords (may be partial records)
        """
        if not html or not html.strip():
            logger.warning(f"Empty HTML for curriculum page {source_url}")
            return []

        soup = self._create_soup(html)
        courses: list[CourseRecord] = []

        # Try to find curriculum table
        table = soup.select_one(self.selectors.curriculum_table)
        
        if table:
            courses.extend(self._parse_curriculum_table(table, source_url, department, year_range))
        else:
            # Try to find course list container
            container = soup.select_one(self.selectors.course_list_container)
            if container:
                courses.extend(
                    self._parse_course_list(container, source_url, department, year_range)
                )

        logger.info(f"Parsed {len(courses)} courses from curriculum page {source_url}")
        return courses

    def _parse_curriculum_table(
        self,
        table: Tag,
        source_url: str,
        department: str,
        year_range: str,
    ) -> list[CourseRecord]:
        """Parse courses from a curriculum table."""
        courses = []
        current_semester: Optional[int] = None

        rows = table.select(self.selectors.table_row)

        for row in rows:
            cells = row.select(self.selectors.table_cell)
            if not cells:
                continue

            # Check if this is a semester header row
            row_text = row.get_text(strip=True).lower()
            semester_match = re.search(r"semester\s*(\d+)|dönem\s*(\d+)|(\d+)\.\s*(?:semester|dönem)", row_text)
            if semester_match:
                current_semester = int(next(g for g in semester_match.groups() if g))
                continue

            # Try to extract course from row
            course = self._parse_table_row(cells, source_url, department, year_range, current_semester)
            if course:
                courses.append(course)

        return courses

    def _parse_table_row(
        self,
        cells: list[Tag],
        source_url: str,
        department: str,
        year_range: str,
        semester: Optional[int],
    ) -> Optional[CourseRecord]:
        """Parse a single table row into a CourseRecord."""
        if len(cells) < 2:
            return None

        # Common table formats:
        # [Code, Title, Credits, ECTS, ...]
        # [Code, Title, Type, Credits, ...]
        
        cell_texts = [cell.get_text(strip=True) for cell in cells]
        
        # Skip header rows
        if any(h in cell_texts[0].lower() for h in ["code", "kod", "course", "ders"]):
            return None

        # Try to identify code (usually first column)
        course_code = None
        course_title = None
        ects = None
        local_credits = None
        course_type = CourseType.UNKNOWN

        for i, text in enumerate(cell_texts):
            # Course code pattern
            if not course_code and re.match(r"^[A-Z]{2,4}\s*\d{3}[A-Z]?$", text, re.IGNORECASE):
                course_code = text
                # Title is usually next
                if i + 1 < len(cell_texts):
                    course_title = cell_texts[i + 1]
                continue

            # ECTS (look for small numbers 1-30)
            if not ects and re.match(r"^\d{1,2}$", text):
                num = int(text)
                if 1 <= num <= 30:
                    # Could be ECTS or local credits
                    if ects is None:
                        ects = float(num)
                    elif local_credits is None:
                        local_credits = float(num)

            # Course type
            text_lower = text.lower()
            if "mandatory" in text_lower or "zorunlu" in text_lower:
                course_type = CourseType.MANDATORY
            elif "elective" in text_lower or "seçmeli" in text_lower:
                course_type = CourseType.ELECTIVE

        if not course_code or not course_title:
            return None

        try:
            return CourseRecord(
                course_code=course_code,
                course_title=course_title,
                department=department,
                year_range=year_range,
                course_type=course_type,
                semester=semester,
                ects=ects,
                local_credits=local_credits,
                source_url=source_url,
                scraped_at=datetime.utcnow(),
            )
        except Exception:
            return None

    def _parse_course_list(
        self,
        container: Tag,
        source_url: str,
        department: str,
        year_range: str,
    ) -> list[CourseRecord]:
        """Parse courses from a list container."""
        courses = []

        items = container.select(self.selectors.course_list_item)
        
        for item in items:
            # Try to extract code and title from item
            code_elem = item.select_one(self.selectors.course_code)
            title_elem = item.select_one(self.selectors.course_title)

            code = code_elem.get_text(strip=True) if code_elem else None
            title = title_elem.get_text(strip=True) if title_elem else None

            # Fallback: try link text
            if not code or not title:
                link = item.select_one(self.selectors.course_link)
                if link:
                    link_text = link.get_text(strip=True)
                    match = re.match(
                        r"^([A-Z]{2,4}\s*\d{3}[A-Z]?)\s*[-:]\s*(.+)$",
                        link_text,
                        re.IGNORECASE,
                    )
                    if match:
                        code = code or match.group(1).strip()
                        title = title or match.group(2).strip()

            if code and title:
                try:
                    course = CourseRecord(
                        course_code=code,
                        course_title=title,
                        department=department,
                        year_range=year_range,
                        source_url=source_url,
                        scraped_at=datetime.utcnow(),
                    )
                    courses.append(course)
                except Exception:
                    continue

        return courses

    def extract_course_links(self, html: str, base_url: str = "") -> list[str]:
        """
        Extract links to individual course pages from a listing page.

        Args:
            html: HTML content
            base_url: Base URL for resolving relative links

        Returns:
            List of absolute URLs to course detail pages
        """
        from urllib.parse import urljoin

        soup = self._create_soup(html)
        links = []

        for a_tag in soup.select(self.selectors.course_link):
            href = a_tag.get("href")
            if href:
                # Resolve relative URLs
                absolute_url = urljoin(base_url, href)
                if absolute_url not in links:
                    links.append(absolute_url)

        return links


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────


def parse_course_page(
    html: str,
    source_url: str,
    department: str,
    year_range: str,
    selectors: Optional[ParserSelectors] = None,
) -> Optional[CourseRecord]:
    """
    Convenience function to parse a single course page.

    Args:
        html: Raw HTML content
        source_url: URL of the page
        department: Department ID
        year_range: Academic year range
        selectors: Optional custom selectors

    Returns:
        CourseRecord or None
    """
    parser = CourseParser(selectors=selectors)
    return parser.parse_course_page(html, source_url, department, year_range)


def parse_curriculum_page(
    html: str,
    source_url: str,
    department: str,
    year_range: str,
    selectors: Optional[ParserSelectors] = None,
) -> list[CourseRecord]:
    """
    Convenience function to parse a curriculum listing page.

    Args:
        html: Raw HTML content
        source_url: URL of the page
        department: Department ID
        year_range: Academic year range
        selectors: Optional custom selectors

    Returns:
        List of CourseRecords
    """
    parser = CourseParser(selectors=selectors)
    return parser.parse_curriculum_page(html, source_url, department, year_range)
