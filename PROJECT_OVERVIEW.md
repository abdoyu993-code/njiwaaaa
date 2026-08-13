# Vue d'ensemble — NARSA AI Assistant

## Objet

Ce dépôt contient la plateforme numérique de la NARSA (Maroc). Elle associe :

- une classification locale des intentions ;
- une recherche documentaire RAG sur des documents Markdown versionnés ;
- une réponse extractive lorsque le contexte est très fiable ;
- une génération contrôlée par Gemini dans les autres cas ;
- un module de réclamations (dépôt, référence de suivi, traitement par un agent) ;
- un espace administration avec Dashboard IA ;
- une API FastAPI, une interface web servie par l'API (`frontend/web`) et une
  interface Streamlit historique.

Les données et documents couvrent sept domaines : `accident`, `contact_reclamation`, `infractions`, `permis`, `prevention`, `vehicule` et `visite_technique`. Le classifieur livré traite 60 intentions.

## Architecture globale

```text
Navigateur
  │  interface web (frontend/web) servie par FastAPI
  │  — ou Streamlit (frontend/app.py)
  ▼
FastAPI (backend/app/main.py)
  │
  ├─ /ask ─ réponses locales : salutations, formulaires, suivi contextuel
  │     └─ NarsaGenerationService
  │          └─ NarsaRAGService
  │               ├─ IntentClassifierService
  │               │    └─ SentenceTransformer + LogisticRegression
  │               └─ NarsaRetriever
  │                    └─ ChromaDB persistante
  │                          ▲
  │                          └─ ingestion des Markdown de backend/data/documents
  │          └─ Gemini (si l'extraction déterministe ne suffit pas
  │                     et si GEMINI_API_KEY est configurée)
  │
  ├─ /complaints ─ dépôt et suivi des réclamations
  └─ /admin ─ authentification agent, traitement des réclamations,
              Dashboard IA
                    │
                    ▼
              SQLite (SQLAlchemy) : réclamations, étapes de traitement,
              journal des conversations
```

L'API fournit aussi des endpoints autonomes de classification, de décision et de téléchargement des PDF.

## Rôle des dossiers et fichiers principaux

| Emplacement | Rôle |
|---|---|
| `backend/app/main.py` | Point d'entrée FastAPI ; crée l'application et monte les routeurs. |
| `backend/app/routes/` | Contrôleurs HTTP : assistant, classification, décision et formulaires. |
| `backend/app/schemas/` | Contrats Pydantic des requêtes et réponses. |
| `backend/app/classifier/` | Entraînement, évaluation, vérification/correction de jeu de données et service d'inférence. |
| `backend/app/rag/` | Ingestion ChromaDB, retriever, orchestration RAG et génération Gemini. |
| `backend/app/decision/` | Routage conceptuel d'une intention vers `clarification`, `mcp`, `direct_response` ou `rag`. |
| `backend/app/core/` | Configuration centrale (`config.py`, lecture du `.env`) et sécurité de l'espace agent (`security.py`, jetons signés HMAC). |
| `backend/app/database/` | Session SQLAlchemy et modèles `Complaint`, `ComplaintEvent`, `ConversationLog`. |
| `backend/app/complaints/` | Logique métier des réclamations : référence unique, création, recherche, changement de statut. |
| `backend/app/analytics/` | Agrégation des indicateurs du Dashboard IA, journalisation des échanges (`service.py`) et analyses qualité/angles morts (`insights.py`). |
| `backend/app/ai/` | Fonctions d'assistance transverses : détection de langue et normalisation vers le français (`language.py`), catalogue de questions liées (`suggestions.py`). |
| `backend/app/notifications/` | Envoi des emails de notification (création, réponse, changement de statut) avec mode console si aucun SMTP n'est configuré. |
| `backend/app/core/embeddings.py` | Instance unique du modèle d'embedding, partagée par le classifieur et le retriever, avec cache des encodages de requêtes. |
| `backend/app/rag/answer_cache.py` | Cache des réponses documentaires, invalidé dès que les Markdown sources changent. |
| `backend/app/journeys/` | Parcours administratifs guidés : catalogue d'étapes tirées des documents officiels (`catalogue.py`), détection de la démarche et suivi de progression (`service.py`). |
| `backend/app/mcp_tools/` | Paquet réservé, toujours vide : aucun outil MCP n'est branché. |
| `backend/data/documents/` | 32 sources Markdown, classées par domaine ; elles constituent le corpus indexé. Les PDF placés ici ne sont pas ingérés. |
| `backend/data/forms/` | Les cinq PDF effectivement servis par `/forms`. |
| `backend/data/chroma_db/` | Index vectoriel persistant ChromaDB, collection `narsa_documents`. Ignoré par Git. |
| `backend/data/*.csv` | Jeux d'entraînement/test (12 000 / 3 000 lignes) et copies `*_original.csv`. |
| `backend/models/` | Artefacts du classifieur et embeddings pré-calculés de train/test. |
| `frontend/web/` | Interface web de référence (HTML/CSS/JS sans dépendance externe) servie par FastAPI : accueil, assistant, réclamations, suivi, connexion agent et Dashboard IA. |
| `frontend/app.py` | Application Streamlit historique, chat, affichage des formulaires, liens et détails techniques. |
| `frontend/rendering.py` | Fonctions pures de préparation de l'affichage et de contexte conversationnel. |
| `tests/` | Tests pytest fonctionnels et unitaires. |
| `reports/results/` | CSV/JSON issus de contrôle, entraînement et évaluation. |
| `reports/figures/` | Graphiques de distribution, confiance et matrices de confusion. |
| `notebooks/` | Dossier prévu pour notebooks ; il est vide dans l'état analysé. |
| `requirements.txt` | Dépendances Python déclarées. |
| `.env` | Configuration locale, notamment attendue pour `GEMINI_API_KEY`; il ne doit pas être versionné. |

## Flux d'une requête utilisateur

1. L'utilisateur saisit son message dans l'interface web (`frontend/web/app.js`) ou dans Streamlit. L'interface conserve l'historique, retrouve le dernier contexte métier valable (`question`, `domain`, `intent`) et envoie `POST /ask` avec `question`, `top_k`, un `session_id` et éventuellement `context`.
2. `backend/app/routes/ask.py` traite d'abord localement les salutations/remerciements. Il intercepte aussi les demandes explicites de PDF et les suivis contextuels de formulaire : ces chemins n'appellent ni le RAG ni Gemini.
3. Pour une question dépendante courte (« et les documents ? », « donne-moi le formulaire »), la question précédente est concaténée à la requête effective. Une question indépendante ignore ce contexte.
4. `generate_answer()` appelle `NarsaGenerationService.answer()`, qui appelle `NarsaRAGService.search()`.
5. Le RAG classifie la question. Sous le seuil 0,60, il tente d'abord une recherche documentaire directe : si un extrait atteint un score composite d'au moins 0,55, la réponse est produite malgré l'incertitude (`low_confidence_rescue`, aucun formulaire n'est alors proposé) ; sinon une reformulation est demandée avec la liste des sujets couverts. Une demande générique de carte grise sans cas identifié déclenche toujours une clarification.
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
| GET | `/` | `StaticFiles` | Interface web (`frontend/web/index.html`). |
| GET | `/api` | `main.py` | État/version de l'API et endpoints exposés. |
| GET | `/health` | `main.py` | Health check ; indique si la génération est active. |
| POST | `/ask` | `routes/ask.py` | Endpoint principal de conversation. |
| POST | `/classify` | `routes/classification.py` | Classification seule. |
| POST | `/route` | `routes/decision.py` | Classification + décision théorique. |
| GET | `/forms` | `routes/forms.py` | Liste blanche des PDF locaux disponibles. |
| GET | `/forms/{filename}` | `routes/forms.py` | Téléchargement sécurisé d'un PDF autorisé. |
| GET | `/journeys/catalogue` | `routes/journeys.py` | Parcours administratifs disponibles. |
| POST | `/journeys/plan` | `routes/journeys.py` | Construit un parcours à partir d'un objectif en texte libre. |
| POST | `/journeys/start/{cle}` | `routes/journeys.py` | Démarre un parcours choisi dans le catalogue. |
| GET | `/journeys/{ref}` | `routes/journeys.py` | Reprend un parcours et sa progression. |
| POST | `/journeys/{ref}/steps/{i}` | `routes/journeys.py` | Coche ou décoche une étape. |
| GET | `/documents` | `routes/documents.py` | Inventaire des documents consultables. |
| GET | `/documents/content` | `routes/documents.py` | Texte lisible d'un document source (liste blanche). |
| GET | `/complaints/categories` | `routes/complaints.py` | Catégories du formulaire de dépôt. |
| POST | `/complaints/analyze` | `routes/complaints.py` | Pré-diagnostic IA : catégorie, priorité, réponse immédiate éventuelle. |
| POST | `/complaints` | `routes/complaints.py` | Dépôt d'une réclamation, retourne la référence. |
| POST | `/complaints/track` | `routes/complaints.py` | Suivi citoyen par référence + email. |
| POST | `/admin/login` | `routes/admin.py` | Authentification agent (jeton signé). |
| GET | `/admin/me` | `routes/admin.py` | Profil de l'agent connecté. |
| GET | `/admin/dashboard` | `routes/admin.py` | Indicateurs agrégés du Dashboard IA. |
| GET | `/admin/insights` | `routes/admin.py` | Alertes qualité et angles morts documentaires. |
| GET | `/admin/conversations` | `routes/admin.py` | Historique des échanges journalisés. |
| GET | `/admin/conversations/export` | `routes/admin.py` | Export CSV de l'historique. |
| GET | `/admin/complaints` | `routes/admin.py` | Liste filtrable (statut, priorité, recherche). |
| GET | `/admin/complaints/export` | `routes/admin.py` | Export CSV des réclamations filtrées. |
| GET | `/admin/complaints/{ref}` | `routes/admin.py` | Détail d'une réclamation. |
| PATCH | `/admin/complaints/{ref}` | `routes/admin.py` | Changement de statut et note au citoyen. |

Toutes les routes `/admin` sauf `/admin/login` exigent un en-tête
`Authorization: Bearer <jeton>`. Le jeton est signé en HMAC-SHA256 avec
`ADMIN_SECRET` et expire après `ADMIN_TOKEN_TTL_MINUTES`.

## Réclamations et espace administration

Le dépôt d'une réclamation crée une référence unique `NARSA-<année>-<6 caractères>`
et un premier événement de suivi. Le citoyen consulte l'avancement avec sa
référence **et** l'adresse email utilisée lors du dépôt. Un agent peut faire
évoluer le statut (`recue`, `en_cours`, `traitee`, `rejetee`) en ajoutant une
note qui devient visible dans le suivi citoyen ; chaque changement est
historisé dans `complaint_events`.

## Parcours administratifs guidés

`backend/app/journeys/catalogue.py` décrit cinq démarches (achat d'un véhicule
d'occasion, permis perdu ou volé, contestation d'une contravention,
immatriculation, visite technique). Chaque étape porte son intitulé, son détail,
sa durée indicative, les documents à préparer, les formulaires PDF associés, les
liens officiels et le **document source dont elle est tirée** — ce qui permet à
l'interface d'afficher le passage d'origine.

`/journeys/plan` reçoit un objectif en texte libre, le confronte aux marqueurs
d'expression de chaque parcours et à l'intention détectée par le classifieur
(prise en compte seulement au-dessus de 0,60), puis crée un parcours identifié
par une référence `PARC-<année>-<6 caractères>`. La progression (étapes cochées)
est stockée en base et se reprend avec la référence. Si aucune démarche n'est
reconnue, le catalogue complet est proposé plutôt qu'un parcours arbitraire.

Avant le dépôt, `/complaints/analyze` classe la demande, en estime la priorité
(`urgente` sur un signal de sécurité, `haute` sur un dossier relancé ou une
échéance réglementaire, `normale` sinon) et vérifie si une réponse documentaire
existe déjà : le citoyen peut alors renoncer au dépôt. La même analyse est
rejouée au moment de l'enregistrement pour stocker la priorité et l'intention
détectée, avec la justification écrite dans le premier événement de suivi. Ces
estimations sont une aide au tri : l'agent reste libre de les corriger.

Chaque appel à `/ask` est journalisé dans `conversation_logs` (question,
réponse, domaine, intention, confiance, statut, temps de traitement) lorsque
`LOG_CONVERSATIONS` est actif. Le Dashboard IA agrège ces lignes : conversations
et interactions, confiance moyenne, temps moyen, clarifications, faibles
confiances, évolution quotidienne, répartition par domaine, top intentions,
nature des réponses et historique récent. Chaque indicateur est comparé à la
période précédente de même durée.

`/admin/insights` complète ce tableau par deux analyses :

- les **alertes qualité** (taux de clarification, part des questions sans
  document, confiance moyenne, réclamations en souffrance ou urgentes) ;
- les **angles morts documentaires** : les questions restées sans réponse sont
  regroupées par proximité sémantique (modèle d'embedding déjà chargé, seuil
  cosinus 0,72) pour révéler les sujets à documenter en priorité.

Enfin, `backend/app/ai/suggestions.py` construit à partir de `train.csv` un
catalogue de questions lisibles par intention et par domaine ; `/ask` renvoie
trois questions proches dans `related_questions`.

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
- `test_complaints.py` : dépôt, suivi, contrôles de validation des réclamations.
- `test_admin.py` : authentification, protection des routes, jetons expirés ou falsifiés, dashboard, changement de statut.
- `test_conversation_quality.py` : salutations et darija translittérée, messages d'orientation quand aucun document ne convient.
- `test_ai_features.py` : priorités estimées, pré-diagnostic, questions liées, regroupement des angles morts.
- `test_journeys.py` : détection des parcours, progression, cohérence du catalogue avec les formulaires et documents réels, protection du lecteur de sources.
- `conftest.py` : ajoute la racine du projet au `sys.path` et isole la base de données de test.

## Points sensibles et risques

1. **Secrets et dépendances externes.** `GEMINI_API_KEY` active la génération. Depuis la version 2.0, son absence n'empêche plus le démarrage : le service bascule en mode documentaire et construit ses réponses uniquement à partir des extraits (`generation_error: generation_disabled`). `google-genai` figure désormais dans `requirements.txt`. En production, `ADMIN_SECRET` et `ADMIN_PASSWORD` doivent impérativement être changés.
2. **Cohérence du corpus.** Tout ajout ou changement des Markdown impose de relancer `ingest_documents.py`. Sinon ChromaDB reste périmée. L'ingestion efface l'index actuel avant reconstruction.
3. **Artefacts versionnés et compatibilité.** Le classifieur dépend de quatre fichiers cohérents dans `backend/models/`. Un nouvel entraînement doit publier ensemble modèle, encodeur, mapping, nom de modèle et embeddings d'évaluation.
4. **Règles heuristiques fragiles.** La détection de suivis, de formulaires, de procédure, de carte grise et de contexte repose sur des listes de marqueurs français. Les formulations, accents, darija ou messages plus longs peuvent contourner ces chemins.
5. **Écart entre routeur et chat.** `/route` annonce un branchement MCP/direct, mais `/ask` ne l'utilise pas et `mcp_tools/` est vide. Il faut éviter de présumer qu'une prise de rendez-vous ou un paiement est réellement exécuté.
6. **Performance/mémoire.** Le modèle d'embedding est chargé deux fois dans le même processus (classification et retriever), et Gemini peut ajouter un appel réseau puis un retry. Le frontend attend jusqu'à 120 secondes.
7. **Contrat frontend.** L'interface web utilise l'origine courante : servie par FastAPI, elle fonctionne sans configuration. L'interface Streamlit historique reste codée en dur sur `127.0.0.1:8000`.
8. **Sources et sécurité des réponses.** Les données documentaires sont le référentiel réel. Les seuils 0,30/0,60/0,88 et les hints d'intention déterminent fortement ce qui est répondu. La génération est contrainte par prompt, mais une vérification métier/versionnage des documents reste nécessaire, en particulier pour des tarifs, délais et procédures réglementaires.
9. **Formulaires séparés du RAG.** Les PDF sont servis uniquement depuis `backend/data/forms` via une liste blanche. Les copies sous `data/documents/.../formulaires` ne sont pas indexées ni directement servies ; maintenir les deux emplacements peut créer des divergences.
10. **État de travail.** Lors de cette analyse, plusieurs fichiers applicatifs et tests étaient déjà modifiés/non suivis dans le répertoire de travail. Cette documentation n'en déduit pas l'auteur ni la validité ; leurs changements doivent être préservés et revus séparément.

## Commandes usuelles

```bash
# API + interface web sur http://127.0.0.1:8000
uvicorn backend.app.main:app --reload

# Interface Streamlit historique, dans un second terminal
streamlit run frontend/app.py

# Reconstruction volontaire de l'index RAG (destructive)
python -m backend.app.rag.ingest_documents

# Vérification de la connexion Gemini
python -m backend.app.rag.check_gemini

# Tests
pytest
```

Avant le premier lancement, copier `.env.example` vers `.env`. La clé
`GEMINI_API_KEY` est facultative : sans elle, l'assistant répond en mode
documentaire. `.env.example` documente toutes les variables disponibles.


## Performance du pipeline

Les latences ont été mesurées sur 24 appels `/ask` (12 questions,
2 tours), avant et après optimisation :

| Étape | Avant | Après |
|---|---|---|
| classification | 13 ms | 8 ms |
| recherche documentaire | 19 ms | 10 ms |
| génération | 1 948 ms | 857 ms |
| **total (moyenne)** | **1 982 ms** | **877 ms** |
| total (médiane) | 1 229 ms | 406 ms |
| total (p95) | 6 797 ms | 3 578 ms |
| question déjà posée | 1 229 ms | **8 ms** |

Quatre leviers expliquent ce gain :

1. **Un seul modèle d'embedding.** Le classifieur et le retriever
   chargeaient chacun leur `SentenceTransformer` : l'instance est
   désormais partagée (mémoire et démarrage divisés par deux).
2. **Cache des encodages de requêtes.** Une question déjà posée
   n'est pas ré-encodée.
3. **Cache des réponses.** Une question déjà traitée est resservie
   telle quelle tant que les documents sources sont inchangés ;
   les questions contenant des données personnelles (numéros,
   emails) sont exclues du cache.
4. **Gestion des quotas Gemini.** Un dépassement de quota (429)
   n'est plus réessayé : le modèle est mis au repos et l'assistant
   bascule sur un modèle de repli. Quand tous sont épuisés, la
   réponse documentaire est produite immédiatement au lieu de
   perdre une seconde sur un appel voué à l'échec.
