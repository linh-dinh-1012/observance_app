import re
import os
import pandas as pd
from pathlib import Path

# Normalization
def normalize_text(text):
    text = text.replace('\x0c', '')  
    text = re.sub(r'(cid:\d+)', '', text)
    text = re.sub(r'[ﬁﬂ]', '', text)
    text = re.sub(r"[’`‘]", "'", text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

# Cleaning
def clean_text(text):
    text = re.sub(r'--- Page \d+ --- \(OCR\)', '', text)
    text = re.sub(r'AVIS D[ÉE]LIB[ÉE]R[ÉE] N[°ºo]\s?\d{4}-\d+.*?région.*?\n?', '', text, flags=re.IGNORECASE)
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

def remove_first_and_last_sentence_synthese(text):
    if not isinstance(text, str):
        return ""   
    text = re.sub(r"^synth[èeéê]se de l[’'`]avis[\s\n]*", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"Les recommandations émises par l[’'`]autorité environnementale.*?ci[\s\-]*joint\.(.*)$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return text.strip()

# Extraction
def extract_sections(text):
    text = re.sub(r'\r\n?', '\n', text)

    synthese, avis_detaille, conclusion = None, None, None

    # Synthèse
    match_synth = re.search(r'synth[èeéê]se de l[’\'`]avis', text, re.IGNORECASE)
    if match_synth:
        start = match_synth.start()

        # Avis détaillé
        match_avis = re.search(r'^\s*Avis d[éeèéê]taill[éeéèê]', text, re.IGNORECASE | re.MULTILINE)
        if match_avis:
            synthese = text[start:match_avis.start()].strip()
            avis_detaille = text[match_avis.start():].strip()
        else:
            synthese = text[start:].strip()

    else:
        avis_detaille = text.strip()

    # Conclusion 
    match_conclusion = re.search(r'conclusion[\s:\-–]{0,2}\n?(.*)', text, re.IGNORECASE)
    if match_conclusion:
        conclusion = match_conclusion.group(1).strip()

    return synthese or None, avis_detaille or None, conclusion or None

# Pipeline
def process_txt_folder(input_dir, output_dir, output_csv):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = []

    for txt_file in input_dir.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        normalized = normalize_text(raw_text)
        cleaned = clean_text(normalized)

        # Save cleaned
        cleaned_path = output_dir / f"{txt_file.stem}_cleaned.txt"
        with open(cleaned_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        synthese, avis_detaille, conclusion = extract_sections(cleaned)
        synthese = remove_first_and_last_sentence_synthese(synthese)

        data.append({
            "file_name": txt_file.name,
            "synthese": synthese,
            "avis_detaille": avis_detaille,
            "conclusion": conclusion
        })

        print(f"Fichier traité : {txt_file.name}")

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n Résultat sauvegardé : {output_csv}")


if __name__ == "__main__":
    folders = [
        ("data/avis_txt", "data/avis_cleaned", "data/avis_sections.csv"),
        ("data/experimentation_txt/avis_txt", "data/experimentation_avis_cleaned", "data/experimentation_avis_sections.csv")
    ]

    for input_dir, output_dir, output_csv in folders:
        print(f"\n Traitement du dossier: {input_dir}")
        process_txt_folder(input_dir, output_dir, output_csv)
