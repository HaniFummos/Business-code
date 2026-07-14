import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import re
import time
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tkinter import filedialog, Tk
import pdfplumber
from dotenv import load_dotenv
from pathlib import Path
from Retcon import generate_summary
#    ^input^ one of these prompting strategy files: BasePrompt Persona StaticExemplars Retcon RetconPos

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
api_key = os.getenv("MISTRAL_API_KEY")

OUTPUT_FOLDER = "C:/Users/HaniA/OneDrive - Vrije Universiteit Amsterdam/Desktop/UNI/Business track/Generated summaries"
TEMPERATURES = [1.0]
MAX_WORKERS = 20

def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def call_summariser_with_retry(api_key, original_text, temperature, max_retries=5):
    for attempt in range(max_retries):
        try:
            return generate_summary(api_key, original_text, temperature), None
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = (2 ** attempt) + random.random()
                print(f"    Rate limited! Retry {attempt+1}/{max_retries}, waiting {wait:.1f}s...")
                time.sleep(wait)
            else:
                return None, str(e)
    return None, "Max retries exceeded"

def process(pdf_path, temperature, idx, total):
    print(f"[{idx}/{total}] Temp {temperature}: {os.path.basename(pdf_path)}")
    original = extract_text(pdf_path)
    
    summary, error = call_summariser_with_retry(api_key, original, temperature)
    
    if error:
        print(f"  -> [{idx}/{total}] Temp {temperature}: FAILED - {error}")
        os._exit(1)
    
    char_count = len(summary)
    
    out_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_temp{temperature}_summary.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"  -> [{idx}/{total}] Temp {temperature}: Summary saved | Chars={char_count}")
    return {'temperature': temperature, 'chars': char_count}

print("\n[Select PDF files]")
root = Tk()
root.withdraw()
files = filedialog.askopenfilenames(filetypes=[("PDF", "*.pdf")])
root.destroy()
if not files: exit()

print(f"\nProcessing {len(files)} files x {len(TEMPERATURES)} = {len(files)*len(TEMPERATURES)} tasks")
print(f"Max workers: {MAX_WORKERS}")
print("="*60)

results = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process, f, temp, i, len(files)) 
               for i, f in enumerate(files, 1) for temp in TEMPERATURES]
    for future in as_completed(futures):
        result = future.result()
        if result:
            results.append(result)

if results:
    print("="*60)
    print("AVERAGES BY TEMPERATURE")
    print("="*60)
    
    for temp in TEMPERATURES:
        r = [x for x in results if x['temperature'] == temp]
        if r:
            avg_chars = sum(x['chars'] for x in r)/len(r)
            print(f"Temp = {temp}: Avg Chars = {avg_chars:.0f}")
    print("="*60)
    print("GENERATION COMPLETE")
    print("="*60)