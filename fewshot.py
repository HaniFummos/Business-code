import os
from mistralai.client import Mistral

MODEL = "mistral-small-2506"

def load_exemplars(exemplar_folder="exemplars"):
    exemplars = []
    
    for filename in os.listdir(exemplar_folder):
        if filename.endswith("_input.txt"):
            input_path = os.path.join(exemplar_folder, filename)
            output_path = input_path.replace("_input.txt", "_output.txt")
            
            if os.path.exists(output_path):
                with open(input_path, 'r', encoding='utf-8') as f:
                    input_text = f.read()
                with open(output_path, 'r', encoding='utf-8') as f:
                    output_text = f.read()
                
                exemplars.append(f"Rapport:\n{input_text}\n\nSamenvatting:\n{output_text}")
    
    return "\n\n---\n\n".join(exemplars)

EXEMPLARS = ""
exemplar_folder = "exemplars"
if os.path.exists(exemplar_folder):
    EXEMPLARS = load_exemplars(exemplar_folder)
    print(f"Loaded {EXEMPLARS.count('Rapport:')} exemplars")
else:
    print("FAIL")

SYSTEM_PROMPT = (
    "Vat het onderstaande inspectierapport samen voor leerlingen van 10-12 jaar (groep 7/8).\n\n"
    
    "Regels:\n\n"

    "1. Opmaak\n"
    "- Lengte: 250 woorden\n"
    "- Eerste zin: '**Samenvatting voor leerlingen**', met daaronder een uitleggend zin met het eindoordeel.\n"

    "- Verdeel daarna de samenvatting in de kopjes:\n"
    "'Wat gaat er al goed?'\n"
    "'Wat kan beter?' (alleen toevoegen als er verbeterpunten zijn die niet verplicht zijn - gebruik hier altijd het woord 'kan')\n"
    "'Wat moet beter?' (gebruik hier altijd het woord 'moet')\n"
    "'Hoe gaat het nu verder?'\n"
    "- Na deze kopjes eindigt u het document\n"

    "- Gebruik onder elk kopje een reeks bullet points (geen lange alinea's)\n"
    "- Elk bullet point moet de vraag in het kopje beantwoorden door gebruik te maken van concrete feiten/voorbeelden uit het rapport\n"

    "2. Inhoud\n"
    "- Zorg dat uit de samenvatting duidelijk blijkt: of leerlingen genoeg leren, of de school hun ontwikkeling goed bijhoudt, "
    "of ze goed les krijgen, en of ze veilig zijn.\n"

    "3. Schrijfstijl:\n"
    "- Gebruik korte, eenvoudige zinnen\n"
    "- Gebruik altijd het woord 'leraren' in plaats van 'leerkrachten'\n"
    "- Gebruik GEEN emoji's\n"
    "- Spreek de leerlingen aan met 'jullie'\n"
)

if EXEMPLARS:
    SYSTEM_PROMPT += (
        "Hier zijn voorbeelden van goede samenvattingen:\n\n"
        f"{EXEMPLARS}\n\n"
        "---\n\n"
    )


USER_PROMPT = "Het inspectierapport:\n\n"

def generate_summary(api_key, original_text, temp):
    response = Mistral(api_key=api_key).chat.complete(
        model=MODEL,
        temperature=temp,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT + original_text}
        ],
        random_seed=21
    )
    return response.choices[0].message.content