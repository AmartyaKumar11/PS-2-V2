import re
import csv
import string

input_file_path = "C:/Users/AMARTYA KUMAR/Desktop/PS-2/PS-2 functions/skills_it.txt"
output_file_path = "C:/Users/AMARTYA KUMAR/Desktop/PS-2/PS-2 functions/parsed_resumes.csv"

# Helper to generate person names: A-Z, AA, AB, ...
def person_name_generator():
    alphabet = string.ascii_uppercase
    n = 1
    while True:
        for name in (''.join(chars) for chars in __import__('itertools').product(alphabet, repeat=n)):
            yield name
        n += 1

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

section_headers = [
    "Work Experience",
    "Skills",
    "Education",
    "Certifications/Licenses",
    "Links",
    "Additional Information"
]

# Build regex pattern to extract sections
header_pattern = r'(' + '|'.join([re.escape(h) for h in section_headers]) + r')[:\s]'

with open(input_file_path, 'r', encoding='utf-8') as f:
    content = f.read()

resume_blocks = content.strip().split('::::::')
parsed_resumes = []
person_names = person_name_generator()

for block in resume_blocks:
    block = block.strip()
    if not block:
        continue
    parts = block.split(':::', 2)
    if len(parts) < 3:
        continue
    person_name = next(person_names)
    raw_resume_text = parts[2].strip()
    current_resume_data = {"Person Name": person_name}
    # Extract each section using regex
    for i, header in enumerate(section_headers):
        # Build pattern for this header
        pattern = re.compile(r'{}:?'.format(re.escape(header)), re.IGNORECASE)
        match = pattern.search(raw_resume_text)
        if match:
            start = match.end()
            # Find the next header after this one
            next_start = len(raw_resume_text)
            for next_header in section_headers[i+1:]:
                next_pattern = re.compile(r'{}:?'.format(re.escape(next_header)), re.IGNORECASE)
                next_match = next_pattern.search(raw_resume_text, start)
                if next_match:
                    next_start = min(next_start, next_match.start())
            section_text = raw_resume_text[start:next_start]
            current_resume_data[header] = preprocess_text(section_text)
        else:
            current_resume_data[header] = ""
    parsed_resumes.append(current_resume_data)

csv_fieldnames = ["Person Name"] + section_headers
with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(parsed_resumes)

print(f"Successfully parsed resumes and saved to {output_file_path}")