import os
from mistralai.client import Mistral

MODEL = "mistral-small-2506"

SYSTEM_PROMPT = (
    "Vat het onderstaande inspectierapport samen voor leerlingen van 10-12 jaar (groep 7/8).\n\n"
    
    "Regels:\n\n"

    "1. Opmaak\n"
    "- Lengte: 250 woorden\n"
    "- Eerste zin: '**Samenvatting voor leerlingen**', met daaronder een uitleggende zin met het eindoordeel.\n"

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