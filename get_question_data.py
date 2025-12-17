"""Get specific data for evaluation questions."""
import json
from collections import Counter, defaultdict

with open('data/raw/courses.jsonl', 'r', encoding='utf-8') as f:
    courses = [json.loads(line) for line in f]

# A category data
print('=== A01: SE Semester 3 Mandatory ===')
for c in sorted([c for c in courses if c['department']=='se' and c['course_type']=='mandatory' and c.get('semester')==3], key=lambda x: x['course_code']):
    print(f"  {c['course_code']}: {c['course_title']}")

print('\n=== A02: SE Semester 4 Mandatory ===')
for c in sorted([c for c in courses if c['department']=='se' and c['course_type']=='mandatory' and c.get('semester')==4], key=lambda x: x['course_code']):
    print(f"  {c['course_code']}: {c['course_title']}")

print('\n=== A03: CE Semester 5 Mandatory ===')
for c in sorted([c for c in courses if c['department']=='ce' and c['course_type']=='mandatory' and c.get('semester')==5], key=lambda x: x['course_code']):
    print(f"  {c['course_code']}: {c['course_title']}")

print('\n=== A04: CE Semester 6 Mandatory ===')
for c in sorted([c for c in courses if c['department']=='ce' and c['course_type']=='mandatory' and c.get('semester')==6], key=lambda x: x['course_code']):
    print(f"  {c['course_code']}: {c['course_title']}")

print('\n=== A05: EEE Semester 3 Mandatory ===')
for c in sorted([c for c in courses if c['department']=='eee' and c['course_type']=='mandatory' and c.get('semester')==3], key=lambda x: x['course_code']):
    print(f"  {c['course_code']}: {c['course_title']}")

print('\n=== A06: EEE Semester 4 Mandatory ===')
for c in sorted([c for c in courses if c['department']=='eee' and c['course_type']=='mandatory' and c.get('semester')==4], key=lambda x: x['course_code']):
    print(f"  {c['course_code']}: {c['course_title']}")

print('\n=== A07: IE Semester 5 Mandatory ===')
for c in sorted([c for c in courses if c['department']=='ie' and c['course_type']=='mandatory' and c.get('semester')==5], key=lambda x: x['course_code']):
    print(f"  {c['course_code']}: {c['course_title']}")

print('\n=== A08: IE Semester 6 Mandatory ===')
for c in sorted([c for c in courses if c['department']=='ie' and c['course_type']=='mandatory' and c.get('semester')==6], key=lambda x: x['course_code']):
    print(f"  {c['course_code']}: {c['course_title']}")

print('\n=== A09: SE Semester 7 Mandatory ===')
for c in sorted([c for c in courses if c['department']=='se' and c['course_type']=='mandatory' and c.get('semester')==7], key=lambda x: x['course_code']):
    print(f"  {c['course_code']}: {c['course_title']}")

print('\n=== A10: CE Semester 7 Mandatory ===')
for c in sorted([c for c in courses if c['department']=='ce' and c['course_type']=='mandatory' and c.get('semester')==7], key=lambda x: x['course_code']):
    print(f"  {c['course_code']}: {c['course_title']}")

# B category - topic based (all departments)
print('\n=== B01: Courses with "Machine Learning" in title ===')
ml = [c for c in courses if 'machine learning' in c.get('course_title','').lower()]
for c in ml:
    print(f"  {c['course_code']} ({c['department'].upper()})")
print(f"  UNIQUE CODES: {sorted(set(c['course_code'] for c in ml))}")

print('\n=== B02: Courses with "Database" in title ===')
db = [c for c in courses if 'database' in c.get('course_title','').lower()]
for c in db:
    print(f"  {c['course_code']} ({c['department'].upper()})")
print(f"  UNIQUE CODES: {sorted(set(c['course_code'] for c in db))}")

print('\n=== B03: Courses with "Network" in title ===')
net = [c for c in courses if 'network' in c.get('course_title','').lower()]
for c in net:
    print(f"  {c['course_code']} ({c['department'].upper()}): {c['course_title']}")
print(f"  UNIQUE CODES: {sorted(set(c['course_code'] for c in net))}")

print('\n=== B04: Courses with "Optimization" in title ===')
opt = [c for c in courses if 'optimization' in c.get('course_title','').lower()]
for c in opt:
    print(f"  {c['course_code']} ({c['department'].upper()})")
print(f"  UNIQUE CODES: {sorted(set(c['course_code'] for c in opt))}")

print('\n=== B05: Courses with "Signal" in title ===')
sig = [c for c in courses if 'signal' in c.get('course_title','').lower()]
for c in sig:
    print(f"  {c['course_code']} ({c['department'].upper()}): {c['course_title']}")
print(f"  UNIQUE CODES: {sorted(set(c['course_code'] for c in sig))}")

print('\n=== B06: Courses with "Control" in title ===')
ctrl = [c for c in courses if 'control' in c.get('course_title','').lower()]
for c in ctrl:
    print(f"  {c['course_code']} ({c['department'].upper()}): {c['course_title']}")
print(f"  UNIQUE CODES: {sorted(set(c['course_code'] for c in ctrl))}")

print('\n=== B07: Courses with "Programming" in title ===')
prog = [c for c in courses if 'programming' in c.get('course_title','').lower()]
for c in prog:
    print(f"  {c['course_code']} ({c['department'].upper()}): {c['course_title']}")
print(f"  UNIQUE CODES: {sorted(set(c['course_code'] for c in prog))}")

print('\n=== B08: Courses with "Software" in title ===')
sw = [c for c in courses if 'software' in c.get('course_title','').lower()]
for c in sw:
    print(f"  {c['course_code']} ({c['department'].upper()}): {c['course_title']}")
print(f"  UNIQUE CODES: {sorted(set(c['course_code'] for c in sw))}")

print('\n=== B09: Courses with "Simulation" in title ===')
sim = [c for c in courses if 'simulation' in c.get('course_title','').lower()]
for c in sim:
    print(f"  {c['course_code']} ({c['department'].upper()}): {c['course_title']}")
print(f"  UNIQUE CODES: {sorted(set(c['course_code'] for c in sim))}")

print('\n=== B10: Courses with "Security" in title ===')
sec = [c for c in courses if 'security' in c.get('course_title','').lower()]
for c in sec:
    print(f"  {c['course_code']} ({c['department'].upper()}): {c['course_title']}")
print(f"  UNIQUE CODES: {sorted(set(c['course_code'] for c in sec))}")

# C category - comparison questions
print('\n=== C01-C10: Mandatory course counts by department ===')
for dept in ['se', 'ce', 'eee', 'ie']:
    mandatory = len([c for c in courses if c['department']==dept and c['course_type']=='mandatory'])
    print(f"  {dept.upper()}: {mandatory}")

print('\n=== C: Total ECTS for mandatory courses by department ===')
for dept in ['se', 'ce', 'eee', 'ie']:
    total_ects = sum(c.get('ects',0) or 0 for c in courses if c['department']==dept and c['course_type']=='mandatory')
    print(f"  {dept.upper()}: {total_ects}")

print('\n=== C: Elective counts by department ===')
for dept in ['se', 'ce', 'eee', 'ie']:
    electives = len([c for c in courses if c['department']==dept and c['course_type']!='mandatory'])
    print(f"  {dept.upper()}: {electives}")

print('\n=== C: 7+ ECTS mandatory courses by department ===')
for dept in ['se', 'ce', 'eee', 'ie']:
    high_ects = len([c for c in courses if c['department']==dept and c['course_type']=='mandatory' and (c.get('ects',0) or 0) >= 7])
    print(f"  {dept.upper()}: {high_ects}")

# D category - counts
print('\n=== D01: Total courses per department ===')
for dept in ['se', 'ce', 'eee', 'ie']:
    print(f"  {dept.upper()}: {len([c for c in courses if c['department']==dept])}")

print('\n=== D02: 6 ECTS courses count ===')
six_ects = [c for c in courses if c.get('ects') == 6]
print(f"  Total: {len(six_ects)}")

print('\n=== D03: 5 ECTS courses count ===')
five_ects = [c for c in courses if c.get('ects') == 5]
print(f"  Total: {len(five_ects)}")

print('\n=== D04: Courses starting with SE prefix ===')
se_prefix = [c for c in courses if c['course_code'].startswith('SE ')]
print(f"  Total unique: {len(set(c['course_code'] for c in se_prefix))}")

print('\n=== D05: Courses starting with CE prefix ===')
ce_prefix = [c for c in courses if c['course_code'].startswith('CE ')]
print(f"  Total unique: {len(set(c['course_code'] for c in ce_prefix))}")

print('\n=== D06: Courses starting with EEE prefix ===')
eee_prefix = [c for c in courses if c['course_code'].startswith('EEE ')]
print(f"  Total unique: {len(set(c['course_code'] for c in eee_prefix))}")

print('\n=== D07: Courses starting with IE prefix ===')
ie_prefix = [c for c in courses if c['course_code'].startswith('IE ')]
print(f"  Total unique: {len(set(c['course_code'] for c in ie_prefix))}")

print('\n=== D08: Courses starting with MATH prefix ===')
math_prefix = [c for c in courses if c['course_code'].startswith('MATH ')]
print(f"  Total unique: {len(set(c['course_code'] for c in math_prefix))}")
for code in sorted(set(c['course_code'] for c in math_prefix)):
    print(f"    {code}")

print('\n=== D09: Courses with 4 ECTS ===')
four_ects = [c for c in courses if c.get('ects') == 4]
print(f"  Total: {len(four_ects)}")

print('\n=== D10: 8 ECTS courses ===')
eight_ects = [c for c in courses if c.get('ects') == 8]
print(f"  Total: {len(eight_ects)}")
for c in eight_ects:
    print(f"    {c['course_code']} ({c['department'].upper()})")

# Verify trap topics don't exist
print('\n=== TRAP VERIFICATION ===')
topics_to_check = [
    'quantum', 'blockchain', 'cryptocurrency', 'virtual reality', 
    'augmented reality', 'natural language processing', 'nlp',
    'robotics', 'autonomous', 'drone', 'bioinformatics',
    'game theory', 'renewable energy', 'solar', 'wind power',
    'photovoltaic', '3d printing', 'additive manufacturing',
    'devops', 'kubernetes', 'docker', 'terraform'
]

for topic in topics_to_check:
    found = [c for c in courses if topic in (c.get('course_title','') + ' ' + c.get('description','') + ' ' + c.get('objectives','')).lower()]
    status = f"FOUND ({len(found)})" if found else "NOT FOUND - GOOD FOR TRAP"
    print(f"  '{topic}': {status}")
