"""Script to analyze course data for evaluation question answers."""
import json
from collections import Counter, defaultdict

# Load all courses
with open('data/raw/courses.jsonl', 'r', encoding='utf-8') as f:
    courses = [json.loads(line) for line in f]

print(f"Total courses: {len(courses)}")

# Count by department
dept_counts = Counter(c['department'] for c in courses)
print("\nCourses by department:")
for d, count in sorted(dept_counts.items()):
    print(f"  {d}: {count}")

# A01: SE Year 2 mandatory courses (semesters 3 and 4)
print("\n" + "="*60)
print("A01: SE Year 2 (Sem 3-4) Mandatory Courses")
print("="*60)
se_y2 = [c for c in courses if c['department'] == 'se' and c['course_type'] == 'mandatory' and c.get('semester') in [3, 4]]
for c in sorted(se_y2, key=lambda x: (x.get('semester',0), x['course_code'])):
    print(f"  {c['course_code']}: {c['course_title']} (Sem {c.get('semester')}, {c.get('ects')} ECTS)")

# A02: CE Semester 5 mandatory courses
print("\n" + "="*60)
print("A02: CE Semester 5 Mandatory Courses")
print("="*60)
ce_s5 = [c for c in courses if c['department'] == 'ce' and c['course_type'] == 'mandatory' and c.get('semester') == 5]
for c in sorted(ce_s5, key=lambda x: x['course_code']):
    print(f"  {c['course_code']}: {c['course_title']} ({c.get('ects')} ECTS)")

# A03: EEE Signal Processing courses
print("\n" + "="*60)
print("A03: EEE Signal Processing Courses")
print("="*60)
eee_signal = [c for c in courses if c['department'] == 'eee' and 
              ('signal' in (c.get('course_title','') + ' ' + c.get('description','')).lower() or
               'signals' in (c.get('course_title','') + ' ' + c.get('description','')).lower())]
for c in eee_signal:
    print(f"  {c['course_code']}: {c['course_title']}")

# A04: IE Optimization / OR courses
print("\n" + "="*60)
print("A04: IE Optimization/Operations Research Courses")
print("="*60)
ie_opt = [c for c in courses if c['department'] == 'ie' and 
          any(term in (c.get('course_title','') + ' ' + c.get('description','') + ' ' + c.get('objectives','')).lower() 
              for term in ['optimization', 'operations research', 'linear programming', 'integer programming'])]
for c in ie_opt:
    print(f"  {c['course_code']}: {c['course_title']}")

# A05: SE Database courses and prerequisites
print("\n" + "="*60)
print("A05: SE Database Courses + Prerequisites")
print("="*60)
se_db = [c for c in courses if c['department'] == 'se' and 
         'database' in (c.get('course_title','') + ' ' + c.get('description','')).lower()]
for c in se_db:
    print(f"  {c['course_code']}: {c['course_title']}")
    print(f"    Prerequisites: {c.get('prerequisites', 'None')}")

# A06: CE courses with machine learning / AI
print("\n" + "="*60)
print("A06: CE Machine Learning / AI Courses")
print("="*60)
ce_ml = [c for c in courses if c['department'] == 'ce' and 
         any(term in (c.get('course_title','') + ' ' + c.get('description','') + ' ' + c.get('objectives','')).lower() 
             for term in ['machine learning', 'artificial intelligence', 'neural network', 'deep learning'])]
for c in ce_ml:
    print(f"  {c['course_code']}: {c['course_title']}")

# A07: EEE courses for ABET outcomes
print("\n" + "="*60)
print("A07: EEE Control Systems Courses")
print("="*60)
eee_control = [c for c in courses if c['department'] == 'eee' and 
               'control' in (c.get('course_title','') + ' ' + c.get('description','')).lower()]
for c in eee_control:
    print(f"  {c['course_code']}: {c['course_title']}")

# A08: IE courses that require calculus
print("\n" + "="*60)
print("A08: IE Courses with Calculus Prerequisites")
print("="*60)
ie_calc = [c for c in courses if c['department'] == 'ie' and 
           c.get('prerequisites') and 'math' in c.get('prerequisites','').lower()]
for c in ie_calc:
    print(f"  {c['course_code']}: {c['course_title']}")
    print(f"    Prerequisites: {c.get('prerequisites')}")

# A09: SE courses covering testing
print("\n" + "="*60)
print("A09: SE Testing/QA Courses")
print("="*60)
se_test = [c for c in courses if c['department'] == 'se' and 
           any(term in (c.get('course_title','') + ' ' + c.get('description','') + ' ' + c.get('objectives','')).lower() 
               for term in ['testing', 'quality assurance', 'verification', 'validation'])]
for c in se_test:
    print(f"  {c['course_code']}: {c['course_title']}")

# A10: CE courses on networks
print("\n" + "="*60)
print("A10: CE Network/Communication Courses")
print("="*60)
ce_net = [c for c in courses if c['department'] == 'ce' and 
          any(term in (c.get('course_title','') + ' ' + c.get('description','')).lower() 
              for term in ['network', 'communication', 'protocol'])]
for c in ce_net:
    print(f"  {c['course_code']}: {c['course_title']}")

# Topic-based questions B01-B10
print("\n" + "="*60)
print("B01: ALL courses covering Python programming")
print("="*60)
python_courses = [c for c in courses if 'python' in (c.get('description','') + ' ' + str(c.get('weekly_topics',''))).lower()]
for c in python_courses:
    print(f"  {c['course_code']} ({c['department'].upper()}): {c['course_title']}")

print("\n" + "="*60)
print("B02: ALL courses covering cybersecurity/security")
print("="*60)
security = [c for c in courses if any(term in (c.get('course_title','') + ' ' + c.get('description','') + ' ' + c.get('objectives','')).lower() 
                                       for term in ['security', 'cybersecurity', 'cryptograph'])]
for c in security:
    print(f"  {c['course_code']} ({c['department'].upper()}): {c['course_title']}")

print("\n" + "="*60)
print("B03: ALL courses covering embedded systems")
print("="*60)
embedded = [c for c in courses if any(term in (c.get('course_title','') + ' ' + c.get('description','') + ' ' + c.get('objectives','')).lower() 
                                       for term in ['embedded', 'microcontroller', 'microprocessor'])]
for c in embedded:
    print(f"  {c['course_code']} ({c['department'].upper()}): {c['course_title']}")

print("\n" + "="*60)
print("B04: ALL project management courses")
print("="*60)
pm = [c for c in courses if any(term in (c.get('course_title','') + ' ' + c.get('description','') + ' ' + c.get('objectives','')).lower() 
                                 for term in ['project management', 'agile', 'scrum'])]
for c in pm:
    print(f"  {c['course_code']} ({c['department'].upper()}): {c['course_title']}")

print("\n" + "="*60)
print("B05: ALL courses covering probability/statistics")
print("="*60)
stats = [c for c in courses if any(term in (c.get('course_title','') + ' ' + c.get('description','')).lower() 
                                    for term in ['probability', 'statistic', 'random'])]
for c in stats:
    print(f"  {c['course_code']} ({c['department'].upper()}): {c['course_title']}")

# D questions - Quantitative
print("\n" + "="*60)
print("D01: Total ECTS per year for SE")
print("="*60)
se_ects_by_year = defaultdict(float)
for c in courses:
    if c['department'] == 'se' and c['course_type'] == 'mandatory':
        sem = c.get('semester', 0)
        year = (sem + 1) // 2 if sem else 0
        se_ects_by_year[year] += c.get('ects', 0) or 0
for year in sorted(se_ects_by_year.keys()):
    print(f"  Year {year}: {se_ects_by_year[year]} ECTS")

print("\n" + "="*60)
print("D02: Count mandatory courses in CE by year")
print("="*60)
ce_mandatory_by_year = defaultdict(int)
for c in courses:
    if c['department'] == 'ce' and c['course_type'] == 'mandatory':
        sem = c.get('semester', 0)
        year = (sem + 1) // 2 if sem else 0
        ce_mandatory_by_year[year] += 1
for year in sorted(ce_mandatory_by_year.keys()):
    print(f"  Year {year}: {ce_mandatory_by_year[year]} mandatory courses")

print("\n" + "="*60)
print("D03: Count 6+ ECTS courses across all departments")
print("="*60)
high_ects = [c for c in courses if (c.get('ects') or 0) >= 6]
print(f"  Total courses with 6+ ECTS: {len(high_ects)}")
for dept in ['se', 'ce', 'eee', 'ie']:
    count = len([c for c in high_ects if c['department'] == dept])
    print(f"    {dept.upper()}: {count}")

print("\n" + "="*60)
print("D04: Count elective courses per department")
print("="*60)
for dept in ['se', 'ce', 'eee', 'ie']:
    electives = [c for c in courses if c['department'] == dept and c['course_type'] != 'mandatory']
    print(f"  {dept.upper()}: {len(electives)} electives")

print("\n" + "="*60)
print("D05: Highest ECTS single course per department")
print("="*60)
for dept in ['se', 'ce', 'eee', 'ie']:
    dept_courses = [c for c in courses if c['department'] == dept and c.get('ects')]
    if dept_courses:
        max_course = max(dept_courses, key=lambda x: x.get('ects', 0))
        print(f"  {dept.upper()}: {max_course['course_code']} - {max_course['course_title']} ({max_course['ects']} ECTS)")

# More analysis for remaining questions
print("\n" + "="*60)
print("B06: ALL courses covering object-oriented programming")
print("="*60)
oop = [c for c in courses if any(term in (c.get('course_title','') + ' ' + c.get('description','') + ' ' + str(c.get('weekly_topics',''))).lower() 
                                  for term in ['object-oriented', 'object oriented', 'oop', 'inheritance', 'polymorphism'])]
for c in oop:
    print(f"  {c['course_code']} ({c['department'].upper()}): {c['course_title']}")

print("\n" + "="*60)
print("B07: ALL courses covering cloud computing")
print("="*60)
cloud = [c for c in courses if any(term in (c.get('course_title','') + ' ' + c.get('description','') + ' ' + c.get('objectives','')).lower() 
                                    for term in ['cloud', 'distributed', 'aws', 'azure'])]
for c in cloud:
    print(f"  {c['course_code']} ({c['department'].upper()}): {c['course_title']}")

print("\n" + "="*60)
print("B08: ALL courses covering data analysis/analytics")
print("="*60)
analytics = [c for c in courses if any(term in (c.get('course_title','') + ' ' + c.get('description','')).lower() 
                                        for term in ['data analy', 'analytics', 'data mining', 'data science'])]
for c in analytics:
    print(f"  {c['course_code']} ({c['department'].upper()}): {c['course_title']}")

print("\n" + "="*60)
print("B09: ALL courses covering simulation")
print("="*60)
sim = [c for c in courses if 'simulation' in (c.get('course_title','') + ' ' + c.get('description','')).lower()]
for c in sim:
    print(f"  {c['course_code']} ({c['department'].upper()}): {c['course_title']}")

print("\n" + "="*60)
print("B10: ALL courses covering software architecture/design patterns")
print("="*60)
arch = [c for c in courses if any(term in (c.get('course_title','') + ' ' + c.get('description','') + ' ' + c.get('objectives','')).lower() 
                                   for term in ['software architecture', 'design pattern', 'architectural'])]
for c in arch:
    print(f"  {c['course_code']} ({c['department'].upper()}): {c['course_title']}")

# C questions - Comparison
print("\n" + "="*60)
print("C01: SE vs CE - Software Development Methodologies")
print("="*60)
for dept in ['se', 'ce']:
    sdlc = [c for c in courses if c['department'] == dept and 
            any(term in (c.get('course_title','') + ' ' + c.get('description','') + ' ' + c.get('objectives','')).lower() 
                for term in ['software development', 'agile', 'scrum', 'software lifecycle', 'sdlc', 'software methodology'])]
    print(f"  {dept.upper()}: {[c['course_code'] for c in sdlc]}")

print("\n" + "="*60)
print("C02: EEE vs CE - Hardware-related courses")
print("="*60)
for dept in ['eee', 'ce']:
    hw = [c for c in courses if c['department'] == dept and 
          any(term in (c.get('course_title','') + ' ' + c.get('description','')).lower() 
              for term in ['hardware', 'circuit', 'digital design', 'microprocessor', 'microcontroller', 'vlsi', 'fpga'])]
    print(f"  {dept.upper()}: {[c['course_code'] for c in hw]}")

print("\n" + "="*60)
print("C03: IE vs SE - Project Management depth")
print("="*60)
for dept in ['ie', 'se']:
    pm = [c for c in courses if c['department'] == dept and 
          any(term in (c.get('course_title','') + ' ' + c.get('description','') + ' ' + c.get('objectives','')).lower() 
              for term in ['project management', 'agile', 'scrum', 'planning', 'scheduling'])]
    print(f"  {dept.upper()}: {[c['course_code'] + ' - ' + c['course_title'] for c in pm]}")

print("\n" + "="*60)
print("C04: All depts - Database courses comparison")
print("="*60)
for dept in ['se', 'ce', 'eee', 'ie']:
    db = [c for c in courses if c['department'] == dept and 
          'database' in (c.get('course_title','') + ' ' + c.get('description','')).lower()]
    print(f"  {dept.upper()}: {[c['course_code'] + ' - ' + c['course_title'] for c in db]}")

print("\n" + "="*60)
print("D06: Total mandatory courses per department")
print("="*60)
for dept in ['se', 'ce', 'eee', 'ie']:
    mandatory = [c for c in courses if c['department'] == dept and c['course_type'] == 'mandatory']
    print(f"  {dept.upper()}: {len(mandatory)} mandatory courses")

print("\n" + "="*60)
print("D07: Courses with 'Lab' or 'Laboratory' in title")
print("="*60)
lab = [c for c in courses if any(term in c.get('course_title','').lower() for term in ['lab', 'laboratory'])]
print(f"  Total: {len(lab)}")
for c in lab:
    print(f"    {c['course_code']} ({c['department'].upper()}): {c['course_title']}")

print("\n" + "="*60)
print("D08: Average ECTS per department")
print("="*60)
for dept in ['se', 'ce', 'eee', 'ie']:
    dept_courses = [c for c in courses if c['department'] == dept and c.get('ects')]
    avg = sum(c['ects'] for c in dept_courses) / len(dept_courses) if dept_courses else 0
    print(f"  {dept.upper()}: {avg:.2f} ECTS average")

print("\n" + "="*60)
print("D09: Semester with most courses (all departments)")
print("="*60)
sem_counts = Counter(c.get('semester') for c in courses if c.get('semester'))
for sem, count in sorted(sem_counts.items()):
    print(f"  Semester {sem}: {count} courses")

print("\n" + "="*60)
print("D10: 7+ ECTS mandatory courses")
print("="*60)
high_ects_mandatory = [c for c in courses if c['course_type'] == 'mandatory' and (c.get('ects') or 0) >= 7]
print(f"  Total: {len(high_ects_mandatory)}")
for c in high_ects_mandatory:
    print(f"    {c['course_code']} ({c['department'].upper()}): {c['course_title']} ({c['ects']} ECTS)")

# Check for trap question subjects - looking for things that DON'T exist
print("\n" + "="*60)
print("VERIFICATION: Does 'Quantum Computing' exist?")
print("="*60)
quantum = [c for c in courses if 'quantum' in (c.get('course_title','') + ' ' + c.get('description','')).lower()]
print(f"  Found: {len(quantum)} courses")

print("\n" + "="*60)
print("VERIFICATION: Does 'Blockchain' exist?")
print("="*60)
blockchain = [c for c in courses if 'blockchain' in (c.get('course_title','') + ' ' + c.get('description','')).lower()]
print(f"  Found: {len(blockchain)} courses")

print("\n" + "="*60)
print("VERIFICATION: Does 'Augmented Reality' exist?")
print("="*60)
ar = [c for c in courses if 'augmented reality' in (c.get('course_title','') + ' ' + c.get('description','')).lower()]
print(f"  Found: {len(ar)} courses")

print("\n" + "="*60)
print("VERIFICATION: Does 'Natural Language Processing' exist?")
print("="*60)
nlp = [c for c in courses if 'natural language' in (c.get('course_title','') + ' ' + c.get('description','')).lower()]
print(f"  Found: {len(nlp)} courses")
