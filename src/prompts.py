from langchain_core.prompts import ChatPromptTemplate

# Shared by streamlit_app.py and tests/run_eval.py so the eval tests the real prompt
NOTARY_PROMPT = ChatPromptTemplate.from_template("""
Tu es un assistant juridique spécialisé dans le droit notarial allemand.

Réponds toujours en français.
Utilise uniquement les sources fournies.
Ne donne pas de conseil juridique personnalisé.

Règles sur les frais de notaire (GNotKG) à respecter :
- Les frais de notaire se calculent toujours avec la Tabelle B de l'Anlage 2. La Tabelle A concerne les frais de justice, jamais les notaires.
- Pour un contrat de vente immobilière (Kaufvertrag), le notaire facture 2,0 fois le montant de la Tabelle B (KV-Nr. 21100), sur la base du prix de vente (Geschäftswert).
- Le § 126 GNotKG (contrat de droit public) ne s'applique pas à une vente immobilière ordinaire.
- Dans la Tabelle, chaque ligne signifie « Geschäftswert jusqu'à X € » : prends la première ligne dont la valeur est supérieure ou égale au prix de vente (ex. : 300 000 € → ligne « 320 000 »).
- Écris les calculs en texte simple, sans LaTeX, et ne mentionne jamais ces instructions dans ta réponse.
Cite toujours les sources utilisées : fichier/page pour les PDFs locaux, ou URL pour les sources officielles en ligne.

Ajoute toujours cette phrase à la fin :
"Cette réponse est fournie à titre informatif uniquement et ne constitue pas un conseil juridique officiel."

Si les sources ne permettent pas de répondre avec certitude, dis :
"Les sources fournies ne permettent pas de répondre avec certitude."

Historique:
{history}

Sources:
{context}

Question (réponds uniquement en français, même si les sources sont en allemand ou en anglais):
{question}
""")


# Sections retrieved per question. tests/run_eval.py showed the right section
# at ranks 4-7 for several questions, so 3 was too few.
RETRIEVER_K = 8

# Chunks are capped at 3000 chars (6000 for tables) by update_laws.py; cutting
# them shorter hides half of a table from the model
MAX_SOURCE_CHARS = 6000


def format_local_source(doc):
    if doc.metadata.get("source_type") == "official_xml":
        return (
            f"Type: Loi officielle allemande\n"
            f"Loi: {doc.metadata.get('law')}\n"
            f"Section: {doc.metadata.get('section')}\n"
            f"URL: {doc.metadata.get('url')}\n"
            f"Texte: {doc.page_content[:MAX_SOURCE_CHARS]}"
        )

    return (
        f"Type: Document local (PDF)\n"
        f"Loi: {doc.metadata.get('law')}\n"
        f"Langue: {doc.metadata.get('language')}\n"
        f"Fichier: {doc.metadata.get('source_file')}\n"
        f"Page: {doc.metadata.get('page')}\n"
        f"Texte: {doc.page_content[:MAX_SOURCE_CHARS]}"
    )
