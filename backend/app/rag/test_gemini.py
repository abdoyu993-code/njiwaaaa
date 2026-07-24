from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash",
)

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY est absente du fichier .env"
    )

print("Clé Gemini détectée.")
print(f"Modèle utilisé : {model_name}")
print("Envoi de la requête à Gemini...")

client = genai.Client(
    api_key=api_key
)

response = client.models.generate_content(
    model=model_name,
    contents=(
        "Réponds en français en une seule phrase : "
        "quel est le rôle principal de la NARSA au Maroc ?"
    ),
)

if not response.text:
    raise RuntimeError(
        "Gemini n'a retourné aucun texte."
    )

print("\nRéponse Gemini :")
print(response.text)