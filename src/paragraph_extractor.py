import re

def extract_paragraph(text):
    match = re.search(r"(§\s*\d+[a-zA-Z]*)", text)
    if match:
        return match.group(1)
    return "Non trouvé"