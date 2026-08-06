# Vue d'ensemble — NARSA AI Assistant

## Objet

Ce dépôt contient un assistant conversationnel d'information sur les services de la NARSA au Maroc. Il associe :

- une classification locale des intentions ;
- une recherche documentaire RAG sur des documents Markdown versionnés ;
- une réponse extractive lorsque le contexte est très fiable ;
- une génération contrôlée par Gemini dans les autres cas ;
- une API FastAPI et une interface Streamlit.

Les données et documents couvrent sept domaines : `accident`, `contact_reclamation`, `infractions`, `permis`, `prevention`, `vehicule` et `visite_technique`. Le classifieur livré traite 60 intentions.

## Architecture globale

```text
Navigateur
  │
  ▼
Streamlit (frontend/app.py)
  │ POST /ask  — JSON, timeout 120 s
  ▼
FastAPI (backend/app/main.py)
  │
  ├─ réponses locales : salutations, formulaires, suivi contextuel
  └─ NarsaGenerationService
       └─ NarsaRAGService
            ├─ IntentClassifierService
            │    └─ SentenceTransformer + LogisticRegression
            └─ NarsaRetriever
                 └─ ChromaDB persistante
                       ▲
                       └─ ingestion des Markdown de backend/data/documents
       └─ Gemini (si l'extraction déterministe ne suffit pas)
```

L'API fournit aussi des endpoints autonomes de classification, de décision et de téléchargement des PDF. Le dossier `database` existe mais ne contient pas d'implémentation ; SQLAlchemy est déclaré sans être utilisé par le code applicatif.

## Rôle des dossiers et fichiers principaux

| Emplacement | Rôle |
|---|---|
| `backend/app/main.py` | Point d'entrée FastAPI ; crée l'application et monte les routeurs. |
| `backend/app/routes/` | Contrôleurs HTTP : assistant, classification, décision et formulaires. |
| `backend/app/schemas/` | Contrats Pydantic des requêtes et réponses. |
| `backend/app/classifier/` | Entraînement, évaluation, vérification/correction de jeu de données et service d'inférence. |
| `backend/app/rag/` | Ingestion ChromaDB, retriever, orchestration RAG et génération Gemini. |
| `backend/app/decision/` | Routage conceptuel d'une intention vers `clarification`, `mcp`, `direct_response` ou `rag`. |
| `backend/app/core/`, `database/`, `mcp_tools/` | Paquets réservés, actuellement vides. Aucun outil MCP ni accès base SQL n'est branché. |
| `backend/data/documents/` | 32 sources Markdown, classées par domaine ; elles constituent le corpus indexé. Les PDF placés ici ne sont pas ingérés. |
| `backend/data/forms/` | Les cinq PDF effectivement servis par `/forms`. |
| `backend/data/chroma_db/` | Index vectoriel persistant ChromaDB, collection `narsa_documents`. Ignoré par Git. |
| `backend/data/*.csv` | Jeux d'entraînement/test (12 000 / 3 000 lignes) et copies `*_original.csv`. |
| `backend/models/` | Artefacts du classifieur et embeddings pré-calculés de train/test. |
| `frontend/app.py` | Application Streamlit, chat, affichage des formulaires, liens et détails techniques. |
| `frontend/rendering.py` | Fonctions pures de préparation de l'affichage et de contexte conversationnel. |
| `tests/` | Tests pytest fonctionnels et unitaires. |
| `reports/results/` | CSV/JSON issus de contrôle, entraînement et évaluation. |
| `reports/figures/` | Graphiques de distribution, confiance et matrices de confusion. |
| `notebooks/` | Dossier prévu pour notebooks ; il est vide dans l'état analysé. |
| `requirements.txt` | Dépendances Python déclarées. |
| `.env` | Configuration locale, notamment attendue pour `GEMINI_API_KEY`; il ne doit pas être versionné. |

## Flux d'une requête utilisateur

1. L'utilisateur saisit son message dans `frontend/app.py`. L'interface conserve les messages dans `st.session_state`, retrouve le dernier contexte métier valable (`question`, `domain`, `intent`) et envoie `POST http://127.0.0.1:8000/ask` avec `question`, `top_k` et éventuellement `context`.
2. `backend/app/routes/ask.py` traite d'abord localement les salutations/remerciements. Il intercepte aussi les demandes explicites de PDF et les suivis contextuels de formulaire : ces chemins n'appellent ni le RAG ni Gemini.
3. Pour une question dépendante courte (« et les documents ? », « donne-moi le formulaire »), la question précédente est concaténée à la requête effective. Une question indépendante ignore ce contexte.
4. `generate_answer()` appelle `NarsaGenerationService.answer()`, qui appelle `NarsaRAGService.search()`.
5. Le RAG classifie la question, refuse les prédictions sous le seuil 0,60, et demande une précision pour une demande générique de carte grise sans cas identifié.
6. Il enrichit ensuite la requête avec des mots-clés spécifiques à l'intention, recherche dans ChromaDB, reclassifie les candidats et écarte ceux sous le score 0,30.
7. Si aucun contexte n'est fiable, l'API retourne une réponse de prudence. Si un unique contexte est très solide, une réponse extractive est construite localement. Sinon, au plus deux contextes distincts sont injectés dans un prompt Gemini.
8. Gemini reçoit des contraintes explicites : répondre en français, s'appuyer exclusivement sur les extraits, ne rien inventer et ne pas exposer les détails internes. Après une erreur temporaire ou une sortie incomplète, un seul nouvel essai est effectué ; ensuite une réponse extractive de repli est retournée.
9. La route `/ask` ajoute les formulaires locaux correspondant à l'intention, les sources, le statut, et les métriques de temps. Streamlit affiche la réponse ; les sources et métriques ne sont visibles qu'en mode technique.

## RAG : indexation et recherche

### Ingestion

Le script `backend/app/rag/ingest_documents.py` charge uniquement les fichiers `*.md` de `backend/data/documents/`. Il extrait le titre, le domaine (premier sous-dossier) et les intentions déclarées dans la section Markdown « Intentions concernées ». Il découpe chaque document en chunks de 1 200 caractères avec un chevauchement de 200, en privilégiant une coupure sur titre, paragraphe ou phrase.

Chaque chunk est enregistré dans la collection Chroma `narsa_documents` avec un identifiant SHA-256 stable, son texte, et les métadonnées `domain`, `title`, `filename`, `relative_path`, `chunk_index`, `document_type` et `intents`. L'ingestion supprime et reconstruit l'index existant : c'est une opération destructive à exécuter délibérément.

### Embeddings

Le modèle d'embedding est `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, de dimension 384.

- Les embeddings du corpus RAG sont stockés dans ChromaDB, sous `backend/data/chroma_db/` (notamment `chroma.sqlite3` et les segments binaires). Ils ne sont pas conservés dans un fichier `.npy` dédié.
- Les embeddings du jeu de classification sont stockés séparément dans `backend/models/train_embeddings.npy` et `backend/models/test_embeddings.npy`.
- À l'exécution, le classifieur et le retriever rechargent chacun un `SentenceTransformer` du même modèle. Cela implique deux chargements mémoire indépendants.

### Retriever

Le retriever se trouve dans `backend/app/rag/retriever.py`, classe `NarsaRetriever`. Il encode la requête enrichie, filtre ChromaDB par domaine quand possible, récupère jusqu'à `max(top_k * 4, 12)` candidats, puis applique un score composite : similarité cosinus, bonus d'intention (+0,18), bonus de facettes de contexte et bonus de procédure. Des règles supplémentaires évitent le mélange entre les documents de visite technique et de mutation ; les requêtes procédurales peuvent reconstituer le contenu ordonné d'un document à partir de ses chunks.

`backend/app/rag/rag_service.py` est l'orchestrateur de recherche : il fait classifier, construit la requête avec `INTENT_QUERY_HINTS`, applique les seuils, choisit les sources et produit les statuts `clarification_required`, `no_context_found` ou `context_found`.

### Générateur

Le générateur est `backend/app/rag/generation_service.py`, classe `NarsaGenerationService`.

- Chemin rapide : extraction locale si une seule source et un seul chunk correspondent à l'intention, avec score >= 0,88, couverture suffisante et contenu court.
- Chemin génératif : `google-genai`, modèle `gemini-3.5-flash` par défaut, surchargeable par `GEMINI_MODEL`; température 0,3, maximum 600 tokens, niveau de réflexion faible.
- Résilience : une seule nouvelle tentative pour les erreurs 429/5xx, délais et réponses incomplètes ; repli documentaire sans génération en cas d'échec.

## Classification et décision

`backend/app/classifier/classifier_service.py` charge `intent_classifier.joblib` (une régression logistique), `label_encoder.joblib`, `intent_to_domain.json` et le nom du modèle d'embedding. Il normalise la question, l'encode, calcule `predict_proba`, renvoie l'intention/domaine et les alternatives. Le seuil par défaut est 0,60.

Les artefacts indiquent 60 intentions, 7 domaines, une accuracy de test de 0,9843 et un macro-F1 de 0,9842 sur 3 000 exemples. Ces chiffres sont des résultats historiques sur ce jeu de test, pas une garantie de production.

`backend/app/decision/decision_router.py` expose une politique distincte et limitée : les intentions de rendez-vous/suivi/paiement sont marquées `mcp`, `appel_urgence` est `direct_response`, sinon `rag`. Toutefois, la route principale `/ask` ne consomme pas ce routeur : elle passe directement par le pipeline génération/RAG. Les catégories MCP ne sont donc pas implémentées dans l'expérience de chat actuelle.

## Routes FastAPI

Les routes sont montées dans `backend/app/main.py`.

| Méthode | Route | Implémentation | Usage |
|---|---|---|---|
| GET | `/` | `main.py` | État/version de l'API. |
| GET | `/health` | `main.py` | Health check minimal. |
| POST | `/ask` | `routes/ask.py` | Endpoint principal de conversation. |
| POST | `/classify` | `routes/classification.py` | Classification seule. |
| POST | `/route` | `routes/decision.py` | Classification + décision théorique. |
| GET | `/forms` | `routes/forms.py` | Liste blanche des PDF locaux disponibles. |
| GET | `/forms/{filename}` | `routes/forms.py` | Téléchargement sécurisé d'un PDF autorisé. |

La communication Streamlit-backend est concentrée dans `frontend/app.py` : `API_BASE_URL = "http://127.0.0.1:8000"`, `ASK_API_URL = f"{API_BASE_URL}/ask"` et `call_api()` appelle `requests.post(..., timeout=120)`. Les boutons de téléchargement complètent les URL relatives `/forms/...` avec la même base API.

## Technologies utilisées

- Python, FastAPI, Uvicorn, Pydantic et `python-multipart` pour l'API.
- Streamlit et Requests pour l'interface web.
- Sentence Transformers, scikit-learn, NumPy, Pandas et Joblib pour les embeddings et la classification.
- ChromaDB pour la recherche vectorielle persistante.
- Google GenAI / Gemini pour la génération finale.
- Pytest et HTTPX pour les tests ; Matplotlib/Seaborn/Jupyter pour l'analyse.
- LangChain, `langchain-community`, `langchain-text-splitters`, PyPDF et SQLAlchemy figurent dans les dépendances mais ne sont pas importés par le chemin applicatif actuel.

## Tests

Les tests se trouvent dans `tests/` et s'exécutent avec `pytest`.

- `test_ask.py` : chemins de l'endpoint `/ask` (messages simples, défaut `top_k`, formulaires, clarification).
- `test_conversation.py` : continuité de contexte et clarification carte grise.
- `test_generation_service.py` : extraction, prompt/Gemini simulé, retry et replis.
- `test_visit_technique_retrieval.py` : règles de désambiguïsation visite technique/mutation.
- `test_forms.py` : inventaire et téléchargement réel des PDF.
- `test_frontend_rendering.py` : fonctions de rendu et visibilité conditionnelle des détails.
- `conftest.py` : ajoute la racine du projet au `sys.path`.

## Points sensibles et risques

1. **Secrets et dépendances externes.** `GEMINI_API_KEY` est indispensable dès que `NarsaGenerationService` est instancié. L'absence de clé, de package `google-genai` ou l'indisponibilité de Gemini fait échouer la génération ; le repli n'aide qu'après l'initialisation réussie du service. `google-genai` est importé mais n'apparaît pas dans `requirements.txt` : l'environnement doit l'avoir installé autrement.
2. **Cohérence du corpus.** Tout ajout ou changement des Markdown impose de relancer `ingest_documents.py`. Sinon ChromaDB reste périmée. L'ingestion efface l'index actuel avant reconstruction.
3. **Artefacts versionnés et compatibilité.** Le classifieur dépend de quatre fichiers cohérents dans `backend/models/`. Un nouvel entraînement doit publier ensemble modèle, encodeur, mapping, nom de modèle et embeddings d'évaluation.
4. **Règles heuristiques fragiles.** La détection de suivis, de formulaires, de procédure, de carte grise et de contexte repose sur des listes de marqueurs français. Les formulations, accents, darija ou messages plus longs peuvent contourner ces chemins.
5. **Écart entre routeur et chat.** `/route` annonce un branchement MCP/direct, mais `/ask` ne l'utilise pas et `mcp_tools/` est vide. Il faut éviter de présumer qu'une prise de rendez-vous ou un paiement est réellement exécuté.
6. **Performance/mémoire.** Le modèle d'embedding est chargé deux fois dans le même processus (classification et retriever), et Gemini peut ajouter un appel réseau puis un retry. Le frontend attend jusqu'à 120 secondes.
7. **Contrat frontend déployé localement.** L'URL FastAPI est codée en dur sur `127.0.0.1:8000`; sans configuration, une mise en conteneur ou un déploiement distant cassera la communication.
8. **Sources et sécurité des réponses.** Les données documentaires sont le référentiel réel. Les seuils 0,30/0,60/0,88 et les hints d'intention déterminent fortement ce qui est répondu. La génération est contrainte par prompt, mais une vérification métier/versionnage des documents reste nécessaire, en particulier pour des tarifs, délais et procédures réglementaires.
9. **Formulaires séparés du RAG.** Les PDF sont servis uniquement depuis `backend/data/forms` via une liste blanche. Les copies sous `data/documents/.../formulaires` ne sont pas indexées ni directement servies ; maintenir les deux emplacements peut créer des divergences.
10. **État de travail.** Lors de cette analyse, plusieurs fichiers applicatifs et tests étaient déjà modifiés/non suivis dans le répertoire de travail. Cette documentation n'en déduit pas l'auteur ni la validité ; leurs changements doivent être préservés et revus séparément.

## Commandes usuelles

```powershell
# API
uvicorn backend.app.main:app --reload

# Interface, dans un second terminal
streamlit run frontend/app.py

# Reconstruction volontaire de l'index RAG
python -m backend.app.rag.ingest_documents

# Tests
pytest
```

Avant de lancer l'API conversationnelle, installer les dépendances effectives et définir `GEMINI_API_KEY` dans l'environnement ou dans `.env`. Le fichier `.env.example` est actuellement vide, donc il ne documente pas cette configuration.
