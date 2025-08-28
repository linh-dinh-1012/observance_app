import os
from processing.extract_text import extract_text_with_hybrid_mode

def process_pdf_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

    print(f"Traitement du dossier : {input_dir} (Total : {len(files)} fichiers PDF)")

    for idx, file in enumerate(files, 1):
        pdf_path = os.path.join(input_dir, file)
        txt_name = os.path.splitext(file)[0]
        txt_path = os.path.join(output_dir, txt_name + ".txt")
        log_path = os.path.join(output_dir, txt_name + ".log")

        print(f"\n ({idx}/{len(files)}) Fichier: {file}")
        content, log_lines = extract_text_with_hybrid_mode(pdf_path)

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Text sauvegardé: {txt_path}")

        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(log_lines))
        print(f"Log sauvegardé: {log_path}")
    
    print("Terminé : Tous les fichiers ont été traités!")

if __name__ == "__main__":
    input_folders = [
        "data/avis_pdf",
        "data/reponse_pdf",
        "data/couples_pdf",
        "data/experimentation_pdf"
    ]
    
    for folder in input_folders:
        folder_name = os.path.basename(folder)
        output_folder = f"data/{folder_name.replace('_pdf', '_txt')}"
        process_pdf_folder(folder, output_folder)