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
    CSS selectors for extracting course information from IUE website.
    
    Updated for actual IUE website structure (2025):
    - Curriculum pages: table.curr with semester headers
    - Syllabus URL pattern: /syllabus_v2/type/read/id/{COURSE_CODE}
    """

    # Curriculum table selector (main course listing table)
    curriculum_table: str = "table.curr, table.table-bordered.table-condensed"
    
    # Table row and cell selectors
    table_row: str = "tr"
    table_cell: str = "td, th"
    
    # Course listing page selectors (if not using table)
    course_list_container: str = ".card-body, .content"
    course_list_item: str = "tr"
    course_link: str = "a[href*='syllabus']"

    # Syllabus page selectors - Course header info tables
    course_code: str = "table td:contains('SE '), table td:contains('CE '), table td:contains('EEE '), table td:contains('IE '), table td:contains('MATH '), table td:contains('PHYS '), table td:contains('ENG '), table td:contains('FENG ')"
    course_title: str = "table tr:first-child td:last-child"
    
    # Content sections (syllabus page)
    objectives: str = "table tr:has(td:contains('Course Objectives')) td:last-child"
    description: str = "table tr:has(td:contains('Course Description')) td:last-child"
    prerequisites: str = "table tr:has(td:contains('Prerequisites')) td:last-child"
    weekly_topics: str = "h3:contains('WEEKLY SUBJECTS') + table, table:has(th:contains('Week'))"
    learning_outcomes: str = "table tr:has(td:contains('Learning Outcomes')) td:last-child"
    assessment: str = "h3:contains('EVALUATION SYSTEM') + table, table:has(th:contains('Semester Activities'))"

    # Credit information
    ects_credits: str = "table tr:has(td:contains('ECTS')) td:last-child, table th:contains('ECTS')"
    local_credits: str = "table tr:has(td:contains('Local Credits')) td:last-child"
    
    # Classification
    semester_info: str = "table tr:has(td:contains('Semester')) td:last-child"
    course_type_info: str = "table tr:has(td:contains('Course Type')) td:last-child"


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

        # Try to find ALL curriculum tables (one per semester)
        tables = soup.select(self.selectors.curriculum_table)
        
        if tables:
            for table in tables:
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
        """
        Parse courses from IUE curriculum table.
        
        IUE table structure:
        - Header rows: "1. Year Fall Semester", "Elective Courses", etc.
        - Column headers: Code | Pre. | Course Name | Theory | App/Lab | Local Credits | ECTS
        - Course rows with data in those columns
        """
        courses = []
        current_semester: Optional[int] = None
        current_section: str = "mandatory"  # mandatory, elective, tmd
        
        rows = table.select(self.selectors.table_row)
        
        for row in rows:
            cells = row.select(self.selectors.table_cell)
            if not cells:
                continue

            row_text = row.get_text(strip=True)
            row_text_lower = row_text.lower()
            
            # Check if this is a semester/section header row
            # IUE format: "1. Year Fall Semester", "2. Year Spring Semester", etc.
            year_semester_match = re.search(
                r"(\d+)\.\s*year\s*(fall|spring)\s*semester",
                row_text_lower
            )
            if year_semester_match:
                year = int(year_semester_match.group(1))
                season = year_semester_match.group(2)
                # Calculate semester: Year 1 Fall = 1, Year 1 Spring = 2, etc.
                current_semester = (year - 1) * 2 + (1 if season == "fall" else 2)
                current_section = "mandatory"
                continue
            
            # Check for elective courses section
            if "elective courses" in row_text_lower or "seçmeli" in row_text_lower:
                current_section = "elective"
                continue
            
            # Check for TMD/Complementary courses section  
            if "complementary" in row_text_lower or "tmd" in row_text_lower:
                current_section = "tmd"
                continue
            
            # Skip header rows (Code, Pre., Course Name, etc.)
            if len(cells) >= 3:
                first_cell = cells[0].get_text(strip=True).lower()
                if first_cell in ["code", "kod", ""]:
                    continue
                
                # Skip "Total" rows
                if "total" in first_cell:
                    continue
            
            # Try to extract course from row
            course = self._parse_iue_table_row(
                cells, source_url, department, year_range, 
                current_semester, current_section
            )
            if course:
                courses.append(course)

        return courses

    def _parse_iue_table_row(
        self,
        cells: list[Tag],
        source_url: str,
        department: str,
        year_range: str,
        semester: Optional[int],
        section: str,
    ) -> Optional[CourseRecord]:
        """
        Parse a single IUE curriculum table row.
        
        Expected columns: Code | Pre. | Course Name | Theory | App/Lab | Local Credits | ECTS
        """
        if len(cells) < 6:
            return None
        
        cell_texts = [cell.get_text(strip=True) for cell in cells]
        
        # Extract course code (first column)
        course_code = cell_texts[0] if cell_texts[0] else None
        
        # Validate course code format (e.g., SE 115, MATH 153, ENG 101)
        if not course_code or not re.match(r"^[A-Z]{2,4}\s*\d{3}[A-Z]?$", course_code, re.IGNORECASE):
            # Could be a placeholder like "ELEC 001" - still valid
            if not course_code or not re.match(r"^[A-Z]+\s*\d+$", course_code, re.IGNORECASE):
                return None
        
        # Prerequisites (second column) - "On kosul" means "prerequisite"
        prerequisites = cell_texts[1] if len(cells) > 1 and cell_texts[1] not in ["", " "] else None
        if prerequisites == "On kosul":
            prerequisites = "Has prerequisite (see syllabus)"
        
        # Course name (third column)
        course_title = cell_texts[2] if len(cells) > 2 else None
        if not course_title:
            return None
        
        # Theory hours (fourth column)
        theory_hours = None
        if len(cells) > 3:
            try:
                theory_hours = int(cell_texts[3]) if cell_texts[3].isdigit() else None
            except (ValueError, IndexError):
                pass
        
        # Lab hours (fifth column)
        lab_hours = None
        if len(cells) > 4:
            try:
                lab_hours = int(cell_texts[4]) if cell_texts[4].isdigit() else None
            except (ValueError, IndexError):
                pass
        
        # Local credits (sixth column)
        local_credits = None
        if len(cells) > 5:
            try:
                local_credits = float(cell_texts[5]) if cell_texts[5] else None
            except (ValueError, IndexError):
                pass
        
        # ECTS (seventh column)
        ects = None
        if len(cells) > 6:
            try:
                ects = float(cell_texts[6]) if cell_texts[6] else None
            except (ValueError, IndexError):
                pass
        
        # Determine course type based on section
        if section == "elective":
            course_type = CourseType.TECHNICAL_ELECTIVE
        elif section == "tmd":
            course_type = CourseType.NON_TECHNICAL_ELECTIVE
        else:
            course_type = CourseType.MANDATORY
        
        # Get syllabus URL from link if available
        syllabus_url = None
        link = cells[0].select_one("a")
        if link and link.get("href"):
            syllabus_url = link.get("href")
        
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
                prerequisites=prerequisites,
                source_url=syllabus_url or source_url,
                scraped_at=datetime.utcnow(),
            )
        except Exception as e:
            logger.debug(f"Failed to create CourseRecord: {e}")
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

    def parse_iue_syllabus_page(
        self,
        html: str,
        source_url: str,
        department: str,
        year_range: str,
        base_course: Optional[CourseRecord] = None,
    ) -> Optional[CourseRecord]:
        """
        Parse an IUE syllabus page to extract full course details.
        
        IUE syllabus pages contain tables with:
        - Course Name, Code, Semester, Theory, App/Lab, Local Credits, ECTS
        - Prerequisites, Course Language, Course Type
        - Course Objectives, Learning Outcomes, Course Description
        - Weekly Subjects table
        - Evaluation System table
        
        Args:
            html: Raw HTML content
            source_url: URL of the syllabus page
            department: Department ID
            year_range: Academic year range
            base_course: Optional base CourseRecord to update with details
            
        Returns:
            CourseRecord with full details
        """
        if not html or not html.strip():
            logger.warning(f"Empty HTML for syllabus {source_url}")
            return base_course

        soup = self._create_soup(html)
        
        # Extract course code and title from first table
        course_code = None
        course_title = None
        
        # Find all tables
        tables = soup.select("table")
        
        for table in tables:
            rows = table.select("tr")
            for row in rows:
                cells = row.select("td")
                if len(cells) >= 2:
                    header = cells[0].get_text(strip=True)
                    value = cells[1].get_text(strip=True)
                    
                    if "Course Name" in header:
                        course_title = value
                    elif header == "Code" or "Course Code" in header:
                        course_code = value
        
        # Also try to get code from URL
        if not course_code:
            url_match = re.search(r"/id/([A-Z]+\+?\d+)", source_url)
            if url_match:
                course_code = url_match.group(1).replace("+", " ")
        
        # Extract other fields using table row matching
        objectives = None
        description = None
        prerequisites = None
        learning_outcomes = []
        weekly_topics = []
        assessment_methods = None
        ects = None
        local_credits = None
        semester = None
        course_type = CourseType.UNKNOWN
        
        for table in tables:
            rows = table.select("tr")
            for row in rows:
                cells = row.select("td, th")
                if len(cells) >= 2:
                    header_text = cells[0].get_text(strip=True)
                    value_text = cells[-1].get_text(strip=True)
                    
                    header_lower = header_text.lower()
                    
                    if "course objectives" in header_lower or "objectives" == header_lower:
                        objectives = value_text
                    elif "course description" in header_lower or "description" == header_lower:
                        description = value_text
                    elif "prerequisites" in header_lower:
                        prerequisites = value_text if value_text.lower() != "none" else None
                    elif "learning outcomes" in header_lower:
                        # Learning outcomes might be in a list
                        outcomes_text = value_text
                        if outcomes_text:
                            learning_outcomes = [o.strip() for o in outcomes_text.split("\n") if o.strip()]
                    elif "ects" == header_lower.strip():
                        try:
                            ects = float(re.search(r"(\d+)", value_text).group(1))
                        except (AttributeError, ValueError):
                            pass
                    elif "local credits" in header_lower or "credits" == header_lower:
                        try:
                            local_credits = float(re.search(r"(\d+)", value_text).group(1))
                        except (AttributeError, ValueError):
                            pass
                    elif "semester" in header_lower:
                        if "fall" in value_text.lower():
                            semester = 1  # Will be adjusted based on year
                        elif "spring" in value_text.lower():
                            semester = 2
                    elif "course type" in header_lower:
                        value_lower = value_text.lower()
                        if "required" in value_lower or "core" in value_lower or "mandatory" in value_lower:
                            course_type = CourseType.MANDATORY
                        elif "elective" in value_lower:
                            course_type = CourseType.ELECTIVE
        
        # Parse weekly topics table
        weekly_table = None
        for table in tables:
            headers = table.select("th")
            header_text = " ".join(h.get_text(strip=True).lower() for h in headers)
            if "week" in header_text and "subjects" in header_text:
                weekly_table = table
                break
        
        if weekly_table:
            rows = weekly_table.select("tr")
            for row in rows[1:]:  # Skip header
                cells = row.select("td")
                if len(cells) >= 2:
                    week = cells[0].get_text(strip=True)
                    subject = cells[1].get_text(strip=True)
                    if week and subject and week.isdigit():
                        weekly_topics.append(f"Week {week}: {subject}")
        
        # Parse assessment/evaluation table
        eval_table = None
        for table in tables:
            headers = table.select("th")
            header_text = " ".join(h.get_text(strip=True).lower() for h in headers)
            if "semester activities" in header_text or "number" in header_text and "weighting" in header_text:
                eval_table = table
                break
        
        if eval_table:
            assessment_parts = []
            rows = eval_table.select("tr")
            for row in rows[1:]:
                cells = row.select("td")
                if len(cells) >= 3:
                    activity = cells[0].get_text(strip=True)
                    count = cells[1].get_text(strip=True)
                    weight = cells[2].get_text(strip=True)
                    if activity and count and weight and activity.lower() != "total":
                        assessment_parts.append(f"{activity}: {count}x, {weight}%")
            if assessment_parts:
                assessment_methods = "; ".join(assessment_parts)
        
        # Create or update CourseRecord
        if base_course:
            # Update existing course with syllabus details
            if objectives:
                base_course.objectives = objectives
            if description:
                base_course.description = description
            if prerequisites and not base_course.prerequisites:
                base_course.prerequisites = prerequisites
            if weekly_topics:
                base_course.weekly_topics = weekly_topics
            if learning_outcomes:
                base_course.learning_outcomes = learning_outcomes
            if assessment_methods:
                base_course.assessment_methods = assessment_methods
            if course_type != CourseType.UNKNOWN:
                base_course.course_type = course_type
            return base_course
        
        # Create new CourseRecord
        if not course_code or not course_title:
            logger.warning(f"Could not extract course code/title from {source_url}")
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
                objectives=objectives,
                description=description,
                prerequisites=prerequisites,
                weekly_topics=weekly_topics if weekly_topics else None,
                learning_outcomes=learning_outcomes if learning_outcomes else None,
                assessment_methods=assessment_methods,
                source_url=source_url,
                scraped_at=datetime.utcnow(),
            )
        except Exception as e:
            logger.error(f"Failed to create CourseRecord from syllabus: {e}")
            return None

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

    @staticmethod
    def build_syllabus_url(department: str, course_code: str) -> str:
        """
        Build IUE syllabus URL for a course.
        
        Args:
            department: Department ID (se, ce, eee, ie)
            course_code: Course code (e.g., "SE 115")
            
        Returns:
            Syllabus URL
        """
        # URL encode the course code (space becomes +)
        encoded_code = course_code.replace(" ", "+")
        return f"https://{department}.ieu.edu.tr/en/syllabus_v2/type/read/id/{encoded_code}"


# ─────────────────────────────────────────────────────────────────────────────
# IUE-Specific Parser
# ─────────────────────────────────────────────────────────────────────────────


class IUECourseParser(CourseParser):
    """
    Specialized parser for IUE website structure.
    
    Handles:
    - Curriculum pages at /{dept}.ieu.edu.tr/en/curr
    - Syllabus pages at /{dept}.ieu.edu.tr/en/syllabus_v2/type/read/id/{CODE}
    """
    
    def __init__(self):
        """Initialize with IUE-specific selectors."""
        super().__init__(selectors=ParserSelectors())
    
    def get_department_from_url(self, url: str) -> Optional[str]:
        """Extract department ID from URL."""
        match = re.search(r"https?://(\w+)\.ieu\.edu\.tr", url)
        if match:
            dept = match.group(1).lower()
            if dept in ["se", "ce", "eee", "ie"]:
                return dept
        return None
    
    def parse_syllabus_page(self, html: str, source_url: str) -> Optional[dict]:
        """
        Parse a syllabus page and return extracted data as dictionary.
        
        This is a simplified wrapper for CLI/GUI use.
        
        Args:
            html: Raw HTML content
            source_url: URL of the syllabus page
            
        Returns:
            Dictionary with syllabus fields or None
        """
        if not html or not html.strip():
            return None
            
        soup = self._create_soup(html)
        
        result = {
            "objectives": None,
            "description": None,
            "prerequisites": None,
            "weekly_topics": None,
            "learning_outcomes": None,
            "assessment": None,
            "ects": None,
            "local_credits": None,
        }
        
        # Find all tables and extract data
        tables = soup.select("table")
        
        for table in tables:
            rows = table.select("tr")
            for row in rows:
                cells = row.select("td, th")
                if len(cells) >= 2:
                    header_text = cells[0].get_text(strip=True).lower()
                    value_text = cells[-1].get_text(strip=True)
                    
                    if not value_text or value_text.lower() in ["none", "-", "n/a"]:
                        continue
                    
                    if "course objectives" in header_text or header_text == "objectives":
                        result["objectives"] = value_text
                    elif "course description" in header_text or header_text == "description":
                        result["description"] = value_text
                    elif "prerequisites" in header_text:
                        result["prerequisites"] = value_text
                    elif "learning outcomes" in header_text:
                        outcomes = [o.strip() for o in value_text.split("\n") if o.strip()]
                        if outcomes:
                            result["learning_outcomes"] = outcomes
                    elif header_text.strip() == "ects":
                        try:
                            result["ects"] = float(re.search(r"(\d+)", value_text).group(1))
                        except (AttributeError, ValueError):
                            pass
                    elif "local credits" in header_text:
                        try:
                            result["local_credits"] = float(re.search(r"(\d+)", value_text).group(1))
                        except (AttributeError, ValueError):
                            pass
        
        # Try to extract weekly topics from "WEEKLY SUBJECTS" section
        weekly_header = soup.find(string=re.compile(r"WEEKLY\s*SUBJECTS", re.IGNORECASE))
        if weekly_header:
            weekly_table = weekly_header.find_next("table")
            if weekly_table:
                topics = []
                for row in weekly_table.select("tr")[1:]:  # Skip header
                    cells = row.select("td")
                    if len(cells) >= 2:
                        topic = cells[1].get_text(strip=True)
                        if topic:
                            topics.append(topic)
                if topics:
                    result["weekly_topics"] = topics
        
        # Try to extract assessment from "EVALUATION SYSTEM" section
        eval_header = soup.find(string=re.compile(r"EVALUATION\s*SYSTEM", re.IGNORECASE))
        if eval_header:
            eval_table = eval_header.find_next("table")
            if eval_table:
                assessments = []
                for row in eval_table.select("tr")[1:]:  # Skip header
                    cells = row.select("td")
                    if len(cells) >= 2:
                        activity = cells[0].get_text(strip=True)
                        weight = cells[-1].get_text(strip=True)
                        if activity and weight:
                            assessments.append(f"{activity}: {weight}")
                if assessments:
                    result["assessment"] = "; ".join(assessments)
        
        return result
    
    def parse_curriculum_and_syllabi(
        self,
        curriculum_html: str,
        curriculum_url: str,
        department: str,
        year_range: str,
        syllabus_fetcher=None,
    ) -> list[CourseRecord]:
        """
        Parse curriculum page and optionally fetch syllabus details.
        
        Args:
            curriculum_html: HTML of curriculum page
            curriculum_url: URL of curriculum page
            department: Department ID
            year_range: Academic year range
            syllabus_fetcher: Optional async function to fetch syllabus HTML
            
        Returns:
            List of CourseRecords with full details
        """
        # First, parse the curriculum table
        courses = self.parse_curriculum_page(
            curriculum_html, curriculum_url, department, year_range
        )
        
        logger.info(f"Parsed {len(courses)} courses from curriculum table")
        
        # If we have a syllabus fetcher, get detailed info for each course
        if syllabus_fetcher:
            for course in courses:
                # Skip placeholder courses (ELEC, TMD, etc.)
                if re.match(r"^(ELEC|TMD|SEST)\s*\d+$", course.course_code):
                    continue
                    
                syllabus_url = self.build_syllabus_url(department, course.course_code)
                try:
                    syllabus_html = syllabus_fetcher(syllabus_url)
                    if syllabus_html:
                        self.parse_iue_syllabus_page(
                            syllabus_html, syllabus_url, department, year_range, course
                        )
                except Exception as e:
                    logger.warning(f"Failed to fetch syllabus for {course.course_code}: {e}")
        
        return courses


# ─────────────────────────────────────────────────────────────────────────────
# ECTS Portal Parser (2020-2024 Curriculum)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ECTSSelectors:
    """CSS selectors for ECTS portal (ects.ieu.edu.tr)."""
    
    # Curriculum table selector - uses table.curr class
    curriculum_table: str = "table.curr"
    
    # Table elements
    table_row: str = "tr"
    table_cell: str = "td"
    table_header: str = "th, tr.head"
    semester_title: str = "td.title"
    
    # Course detail page selectors
    course_info_table: str = "table.table-bordered"
    section_header: str = "h3, strong"


# Alias for backward compatibility
EBSSelectors = ECTSSelectors


class ECTSCourseParser:
    """
    Parser for IUE ECTS Portal (ects.ieu.edu.tr) - 2020-2024 curriculum.
    
    The ECTS portal structure:
    - Curriculum: akademik.php?section=XX&sid=curr_before_2025&lang=en
    - Course details: syllabus.php?section=XX&course_code=YY&currType=before_2025
    
    Curriculum table structure (class="table curr"):
    - Title row: "1. Year Fall Semester"
    - Header row: Code | Pre. | Course Name | Theory | Application | Local Credits | ECTS
    - Course rows with links to syllabus pages
    - Total row at the end
    """
    
    def __init__(self, selectors: Optional[ECTSSelectors] = None):
        self.selectors = selectors or ECTSSelectors()
    
    def _create_soup(self, html: str) -> BeautifulSoup:
        """Create BeautifulSoup object from HTML."""
        return BeautifulSoup(html, "html.parser")
    
    def parse_curriculum_page(
        self,
        html: str,
        source_url: str,
        department: str,
        year_range: str,
    ) -> list[CourseRecord]:
        """
        Parse ECTS curriculum page to extract all courses.
        
        Args:
            html: Raw HTML content
            source_url: URL of the curriculum page
            department: Department ID (se, ce, eee, ie)
            year_range: Academic year range (e.g., "2020-2024")
            
        Returns:
            List of CourseRecord objects
        """
        if not html or not html.strip():
            logger.warning(f"Empty HTML for ECTS curriculum page {source_url}")
            return []
        
        soup = self._create_soup(html)
        courses: list[CourseRecord] = []
        seen_codes: set[str] = set()  # Track unique courses
        
        # Find all curriculum tables (one per semester + electives)
        tables = soup.select(self.selectors.curriculum_table)
        
        if not tables:
            # Try alternative selectors
            tables = soup.select("table.table-bordered")
        
        logger.info(f"Found {len(tables)} curriculum tables in ECTS page")
        
        # Process all tables
        for table in tables:
            # Get semester info from title row
            title_row = table.select_one(self.selectors.semester_title)
            semester_info = self._parse_semester_title(title_row)
            
            # Check for elective section
            is_elective = 'elective' in table.get('class', [])
            table_classes = ' '.join(table.get('class', []))
            if 'elective' in table_classes.lower():
                is_elective = True
            
            # Parse rows in this table
            rows = table.select(self.selectors.table_row)
            
            for row in rows:
                # Skip title rows and header rows
                if row.select_one(self.selectors.semester_title):
                    continue
                if 'head' in row.get('class', []):
                    continue
                
                # Skip total rows
                row_text = row.get_text(strip=True).lower()
                if row_text.startswith('total'):
                    continue
                
                cells = row.select(self.selectors.table_cell)
                
                # Skip if not enough cells
                if len(cells) < 5:
                    continue
                
                course = self._parse_ects_table_row(
                    cells, source_url, department, year_range,
                    semester_info, is_elective
                )
                if course and course.course_code not in seen_codes:
                    courses.append(course)
                    seen_codes.add(course.course_code)
        
        logger.info(f"Parsed {len(courses)} unique courses from ECTS curriculum page {source_url}")
        return courses
    
    def _parse_semester_title(self, title_elem) -> dict:
        """Parse semester title to get year and semester info."""
        if not title_elem:
            return {'year': 1, 'semester': 1, 'is_elective': False}
        
        text = title_elem.get_text(strip=True).lower()
        
        year = 1
        semester_num = 1
        is_elective = 'elective' in text
        
        # Parse year: "1. Year", "2. Year", etc.
        year_match = re.search(r'(\d+)\.?\s*year', text)
        if year_match:
            year = int(year_match.group(1))
        
        # Parse semester: "Fall" or "Spring"
        if 'fall' in text:
            semester_num = (year - 1) * 2 + 1
        elif 'spring' in text:
            semester_num = (year - 1) * 2 + 2
        
        return {
            'year': year,
            'semester': semester_num,
            'is_elective': is_elective,
        }
    
    def _parse_ects_table_row(
        self,
        cells: list[Tag],
        source_url: str,
        department: str,
        year_range: str,
        semester_info: dict,
        is_elective: bool,
    ) -> Optional[CourseRecord]:
        """
        Parse a single row from ECTS curriculum table.
        
        ECTS table columns: Code | Pre. | Course Name | Theory | Application | Local Credits | ECTS
        """
        if len(cells) < 5:
            return None
        
        cell_texts = [cell.get_text(strip=True) for cell in cells]
        
        # Extract course code (first column, may have link)
        code_cell = cells[0]
        link = code_cell.find('a')
        course_code = link.get_text(strip=True) if link else cell_texts[0]
        
        if not course_code:
            return None
        
        # Skip header-like rows or total rows
        if course_code.lower() in ['code', 'kod', 'total', 'toplam', '']:
            return None
        
        # Skip POOL courses (general education course placeholders)
        if course_code.startswith('POOL'):
            return None
        
        # Determine column indices based on number of columns
        # Standard: Code | Pre. | Name | Theory | App | Local | ECTS (7 cols)
        # Some tables may have fewer columns
        
        if len(cells) >= 7:
            # Full table with prerequisite column
            # Col 0: Code, Col 1: Pre, Col 2: Name, Col 3: Theory, Col 4: App, Col 5: Local, Col 6: ECTS
            name_idx = 2
            theory_idx = 3
            app_idx = 4
            local_idx = 5
            ects_idx = 6
        elif len(cells) == 6:
            # No prerequisite column
            name_idx = 1
            theory_idx = 2
            app_idx = 3
            local_idx = 4
            ects_idx = 5
        else:
            # Minimal table
            name_idx = 1
            theory_idx = 2
            app_idx = 3
            local_idx = -2
            ects_idx = -1
        
        # Course name
        course_title = cell_texts[name_idx] if len(cells) > name_idx else None
        if not course_title or course_title.lower() in ['course name', 'ders adı']:
            return None
        
        # Local credits
        local_credits = None
        if len(cells) > local_idx:
            try:
                local_credits = float(cell_texts[local_idx]) if cell_texts[local_idx].replace('.', '').isdigit() else None
            except (ValueError, IndexError):
                pass
        
        # ECTS
        ects = None
        if len(cells) > ects_idx:
            try:
                ects_text = cell_texts[ects_idx]
                ects = float(ects_text) if ects_text.replace('.', '').isdigit() else None
            except (ValueError, IndexError):
                pass
        
        # Determine course type
        if is_elective or course_code.startswith('ELEC'):
            course_type = CourseType.TECHNICAL_ELECTIVE
        elif course_code.startswith(('TMD', 'NTE')):
            course_type = CourseType.NON_TECHNICAL_ELECTIVE
        elif course_code.startswith(('SEST', 'POOL')):
            # Summer training / internship and pool courses treated as mandatory
            course_type = CourseType.MANDATORY
        else:
            course_type = CourseType.MANDATORY
        
        # Get course link if available
        course_link = None
        if link and link.get('href'):
            href = link.get('href')
            if href.startswith('syllabus.php'):
                course_link = f"https://ects.ieu.edu.tr/new/{href}"
            else:
                course_link = href
        
        try:
            return CourseRecord(
                course_code=course_code.strip(),
                course_title=course_title.strip(),
                department=department.lower(),
                year_range=year_range,
                course_type=course_type,
                semester=semester_info.get('semester'),
                ects=ects,
                local_credits=local_credits,
                source_url=course_link or source_url,
            )
        except Exception as e:
            logger.warning(f"Failed to create CourseRecord for {course_code}: {e}")
            return None
    
    def parse_course_page(
        self,
        html: str,
        source_url: str,
        department: str,
        year_range: str,
        base_course: Optional[CourseRecord] = None,
    ) -> Optional[CourseRecord]:
        """
        Parse ECTS course detail page (syllabus.php).
        
        Extracts:
        - Course objectives
        - Course description/content
        - Learning outcomes
        - Weekly topics
        - Assessment methods
        - Prerequisites
        
        Args:
            html: Raw HTML content
            source_url: URL of the course page
            department: Department ID
            year_range: Academic year range
            base_course: Optional base course to update with details
            
        Returns:
            CourseRecord with full details
        """
        if not html or not html.strip():
            logger.warning(f"Empty HTML for ECTS course page {source_url}")
            return base_course
        
        soup = self._create_soup(html)
        
        # Start with base course or create new one
        course_data = {
            'department': department,
            'year_range': year_range,
            'source_url': source_url,
        }
        
        if base_course:
            course_data.update({
                'course_code': base_course.course_code,
                'course_title': base_course.course_title,
                'ects': base_course.ects,
                'local_credits': base_course.local_credits,
                'semester': base_course.semester,
                'course_type': base_course.course_type,
            })
        
        # Find course name from first table
        course_name_div = soup.select_one('#course_name')
        if course_name_div:
            course_data['course_title'] = course_name_div.get_text(strip=True)
        
        # Find all tables for course info
        tables = soup.select("table.table-bordered, table.table-condensed")
        
        for table in tables:
            rows = table.select("tr")
            for row in rows:
                cells = row.select("td")
                if len(cells) >= 2:
                    label = cells[0].get_text(strip=True).lower()
                    value = cells[1].get_text(strip=True)
                    
                    # Course code and name
                    if 'course name' in label:
                        if not course_data.get('course_title') or len(value) > len(course_data.get('course_title', '')):
                            course_data['course_title'] = value
                    elif label == 'code' or label.startswith('cod'):
                        if not course_data.get('course_code'):
                            course_data['course_code'] = value
                    
                    # Prerequisites
                    elif 'prerequisite' in label:
                        if value and value.lower() not in ['none', '-', 'yok', '']:
                            course_data['prerequisites'] = value
                    
                    # ECTS
                    elif 'ects' in label:
                        try:
                            course_data['ects'] = float(value)
                        except ValueError:
                            pass
                    
                    # Local credits
                    elif 'local credit' in label:
                        try:
                            course_data['local_credits'] = float(value)
                        except ValueError:
                            pass
                    
                    # Course objectives
                    elif 'objective' in label:
                        course_data['objectives'] = value
                    
                    # Course description
                    elif 'description' in label or 'content' in label:
                        course_data['description'] = value
        
        # Extract learning outcomes from list
        outcomes_list = soup.select('#outcome li')
        if outcomes_list:
            outcomes = [li.get_text(strip=True) for li in outcomes_list if li.get_text(strip=True)]
            if outcomes:
                course_data['learning_outcomes'] = outcomes
        
        # Extract weekly topics
        weekly = self._extract_weekly_topics(soup)
        if weekly:
            course_data['weekly_topics'] = weekly
        
        # Extract assessment methods
        assessment = self._extract_assessment(soup)
        if assessment:
            course_data['assessment_methods'] = assessment
        
        # Extract course notes/textbooks
        for row in soup.select('tr'):
            cells = row.select('td')
            if len(cells) >= 2:
                label = cells[0].get_text(strip=True).lower()
                if 'textbook' in label or 'course notes' in label:
                    course_data['textbook'] = cells[1].get_text(strip=True)
                elif 'suggested' in label or 'reading' in label:
                    course_data['suggested_readings'] = cells[1].get_text(strip=True)
        
        # Create or update CourseRecord
        try:
            if base_course:
                # Update base course with new data
                for key, value in course_data.items():
                    if value is not None and hasattr(base_course, key):
                        setattr(base_course, key, value)
                return base_course
            else:
                return CourseRecord(**course_data)
        except Exception as e:
            logger.warning(f"Failed to create CourseRecord from ECTS page: {e}")
            return base_course
    
    def _extract_weekly_topics(self, soup: BeautifulSoup) -> Optional[list[str]]:
        """Extract weekly topics table as a list."""
        # Find weekly topics table by id
        weeks_table = soup.select_one('#weeks')
        if weeks_table:
            topics = []
            rows = weeks_table.select('tr')[1:]  # Skip header row
            
            for row in rows:
                cells = row.select('td')
                if len(cells) >= 2:
                    week = cells[0].get_text(strip=True)
                    topic = cells[1].get_text(strip=True)
                    if week and topic and topic != '&nbsp;':
                        topics.append(f"Week {week}: {topic}")
            
            if topics:
                return topics
        
        # Fallback: find by header text
        for table in soup.select('table'):
            headers = table.select('th, td.text-center strong')
            header_text = ' '.join(h.get_text(strip=True).lower() for h in headers)
            
            if 'week' in header_text or 'subjects' in header_text:
                topics = []
                rows = table.select('tr')[1:]
                
                for row in rows:
                    cells = row.select('td')
                    if len(cells) >= 2:
                        week = cells[0].get_text(strip=True)
                        topic = cells[1].get_text(strip=True)
                        if week and topic:
                            topics.append(f"Week {week}: {topic}")
                
                if topics:
                    return topics
        
        return None
    
    def _extract_assessment(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract assessment methods."""
        # Find evaluation tables
        for table_id in ['evaluation_table1', 'evaluation_table2']:
            table = soup.select_one(f'#{table_id}')
            if table:
                methods = []
                rows = table.select('tr')[1:]  # Skip header row
                
                for row in rows:
                    cells = row.select('td')
                    if len(cells) >= 3:
                        method = cells[0].get_text(strip=True)
                        number = cells[1].get_text(strip=True)
                        weight = cells[2].get_text(strip=True)
                        
                        # Skip empty or placeholder values
                        if method and number and number != '-' and weight and weight != '-':
                            methods.append(f"{method}: {weight}%")
                
                if methods:
                    return '; '.join(methods)
        
        return None


# Alias for backward compatibility
EBSCourseParser = ECTSCourseParser


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────


def parse_course_page(
    html: str,
    source_url: str,
    department: str,
    year_range: str,
    selectors: Optional[ParserSelectors] = None,
    use_ects_parser: bool = True,
) -> Optional[CourseRecord]:
    """
    Convenience function to parse a single course page.

    Args:
        html: Raw HTML content
        source_url: URL of the page
        department: Department ID
        year_range: Academic year range
        selectors: Optional custom selectors
        use_ects_parser: If True, use ECTS portal parser (2020-2024 curriculum)

    Returns:
        CourseRecord or None
    """
    if use_ects_parser:
        parser = ECTSCourseParser()
        return parser.parse_course_page(html, source_url, department, year_range)
    
    parser = CourseParser(selectors=selectors)
    return parser.parse_course_page(html, source_url, department, year_range)


def parse_curriculum_page(
    html: str,
    source_url: str,
    department: str,
    year_range: str,
    selectors: Optional[ParserSelectors] = None,
    use_ects_parser: bool = True,
) -> list[CourseRecord]:
    """
    Convenience function to parse a curriculum listing page.

    Args:
        html: Raw HTML content
        source_url: URL of the page
        department: Department ID
        year_range: Academic year range
        selectors: Optional custom selectors
        use_ects_parser: If True, use ECTS portal parser (2020-2024 curriculum)

    Returns:
        List of CourseRecords
    """
    if use_ects_parser:
        parser = ECTSCourseParser()
        return parser.parse_curriculum_page(html, source_url, department, year_range)
    
    parser = CourseParser(selectors=selectors)
    return parser.parse_curriculum_page(html, source_url, department, year_range)


def get_parser(use_ects: bool = True) -> ECTSCourseParser | CourseParser:
    """
    Get the appropriate parser based on curriculum type.
    
    Args:
        use_ects: If True, return ECTS parser (2020-2024), else return standard parser
        
    Returns:
        ECTSCourseParser or CourseParser instance
    """
    if use_ects:
        return ECTSCourseParser()
    return CourseParser()
