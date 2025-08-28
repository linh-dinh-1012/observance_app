import re

def normalize_text(text: str) -> str:
    text = text.replace('\x0c', '')
    text = re.sub(r'(cid:\d+)', '', text)
    text = re.sub(r'[ﬁﬂ]', '', text)
    text = re.sub(r"[’`‘]", "'", text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()


def clean_text(text: str) -> str:
    text = re.sub(r'--- Page \d+ --- \(OCR\)', '', text)
    text = re.sub(r'^[=|_\-\[\]<>\\/#*]{3,}.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\bla \d+/\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*.*\(source\s*:\s*dossier\)\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'(?<=[a-z,;])\n(?=[a-zéèàù])', ' ', text)
    text = re.sub(r'\n?\d{1,2}/\d{1,2}\n?', '', text)
    text = re.sub(r'\.([^\s])', r'. \1', text)
    text = re.sub(r':([^\s])', r': \1', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()

