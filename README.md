# Plateforme NARSA — Assistant IA et réclamations

Plateforme numérique de services de la NARSA (Maroc) :

- **Assistant intelligent** : classification d'intentions locale, recherche
  documentaire RAG sur les documents officiels, réponse extractive ou générée
  par Gemini ;
- **Réclamations** : dépôt en ligne avec référence de suivi, consultation de
  l'avancement par le citoyen ;
- **Espace administration** : authentification agent, traitement des
  réclamations et Dashboard IA (volumétrie, confiance, domaines, intentions,
  nature des réponses) ;
- **Formulaires officiels** : téléchargement des PDF autorisés.

### Fonctions d'intelligence artificielle

| Fonction | Où | Ce qu'elle fait |
|---|---|---|
| **Parcours administratif guidé** | Accueil → `/#/parcours` | « J'ai acheté une voiture d'occasion » → toutes les étapes dans l'ordre, documents, formulaires, délais réglementaires et progression sauvegardée (2/6 étapes). |
| **Lecteur de source** | Assistant et parcours | Chaque réponse et chaque étape renvoie au passage exact du document officiel utilisé. |
| **Confiance en clair** | Assistant | « Réponse fortement étayée » / « Sources partielles » / « Informations insuffisantes » au lieu d'un score brut. |
| Assistant RAG | `/#/assistant` | Répond à partir des documents officiels, sans invention. |
| **Multilingue** | Assistant | Comprend le français, l'arabe, la darija en alphabet arabe et l'arabizi ; répond dans la langue du citoyen en préservant URLs, noms de formulaires et organismes. |
| **Feedback 👍/👎** | Assistant | L'avis est rattaché à l'interaction journalisée et alimente le Dashboard IA. |
| **Cache documentaire** | API | Une question déjà traitée est resservie instantanément, invalidée dès qu'un document change. |
| **Repli de modèle** | Génération | En cas de quota Gemini épuisé, bascule automatique sur un modèle de repli, puis réponse documentaire immédiate. |
| Questions liées | Assistant | Propose après chaque réponse des questions proches réellement traitées, issues du corpus d'entraînement. |
| Rattrapage de faible confiance | Assistant | Tente une recherche documentaire directe quand le classifieur hésite, avec un seuil relevé. |
| Pré-diagnostic de réclamation | `/#/reclamation` | Détecte la catégorie, estime la priorité et affiche la réponse officielle si le sujet est déjà documenté (évite un dépôt inutile). |
| Tri automatique des réclamations | Espace agent | Priorité `urgente` / `haute` / `normale` calculée au dépôt, avec justification tracée dans le suivi. |
| Angles morts documentaires | Dashboard IA | Regroupe sémantiquement les questions restées sans réponse : la liste des documents à écrire en priorité. |
| Alertes qualité | Dashboard IA | Taux de clarification, questions sans document, confiance en baisse, réclamations en souffrance. |

L'interface web (`frontend/web`) est servie directement par l'API FastAPI.

## Démarrage rapide

```bash
# 1. Environnement
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# 2. Configuration
cp .env.example .env
# renseignez GEMINI_API_KEY si vous voulez la génération
# (sans clé, l'assistant répond en mode documentaire)

# 3. Index documentaire (uniquement au premier lancement
#    ou après modification des Markdown — opération destructive)
.venv/bin/python -m backend.app.rag.ingest_documents

# 4. Lancement
.venv/bin/python -m uvicorn backend.app.main:app --reload
```

Puis ouvrez **http://127.0.0.1:8000** :

| Page | Adresse |
|---|---|
| Accueil | `/#/accueil` |
| Mon parcours guidé | `/#/parcours` |
| Assistant IA | `/#/assistant` |
| Déposer une réclamation | `/#/reclamation` |
| Suivre une réclamation | `/#/suivi` |
| Espace administration | `/#/admin` |
| Documentation de l'API | `/docs` |

Identifiants d'administration par défaut : `agent` / `narsa2026`
(à changer via `ADMIN_USERNAME` et `ADMIN_PASSWORD` dans `.env`).

## Endpoints principaux

| Méthode | Route | Rôle |
|---|---|---|
| POST | `/ask` | Question à l'assistant (RAG + génération, multilingue). |
| POST | `/feedback` | Avis du citoyen sur une réponse (👍/👎). |
| POST | `/classify` | Classification d'intention seule. |
| POST | `/route` | Classification + décision théorique. |
| GET | `/forms`, `/forms/{fichier}` | Formulaires PDF officiels. |
| GET | `/journeys/catalogue` | Parcours administratifs disponibles. |
| POST | `/journeys/plan` | Construit un parcours depuis un objectif en texte libre. |
| GET | `/journeys/{ref}` | Reprend un parcours et sa progression. |
| POST | `/journeys/{ref}/steps/{i}` | Coche ou décoche une étape. |
| GET | `/documents/content?path=` | Texte lisible d'un document source (liste blanche). |
| GET | `/complaints/categories` | Catégories de réclamation. |
| POST | `/complaints/analyze` | Pré-diagnostic IA avant dépôt. |
| POST | `/complaints` | Dépôt d'une réclamation (retourne la référence). |
| POST | `/complaints/track` | Suivi par référence + email. |
| POST | `/admin/login` | Authentification agent (jeton signé). |
| GET | `/admin/dashboard` | Indicateurs du Dashboard IA. |
| GET | `/admin/insights` | Alertes qualité et angles morts documentaires. |
| GET | `/admin/complaints` | Liste filtrable (statut, priorité, recherche). |
| GET | `/admin/complaints/export` | Export CSV des réclamations. |
| PATCH | `/admin/complaints/{ref}` | Changement de statut + note. |
| POST | `/admin/complaints/{ref}/response` | Réponse officielle de l'agent (enregistrée + email). |
| GET | `/admin/email/status` | Configuration SMTP effective + dernières tentatives (sans secret). |
| POST | `/admin/email/test` | Envoi d'un email de contrôle pour valider la chaîne SMTP. |
| GET | `/admin/conversations` | Historique des échanges. |
| GET | `/admin/conversations/export` | Export CSV de l'historique. |
| GET | `/health`, `/api` | État de l'API. |

## Configuration (`.env`)

| Variable | Défaut | Rôle |
|---|---|---|
| `GEMINI_API_KEY` | *(vide)* | Active la génération. Sans clé, l'assistant reste utilisable en mode documentaire. |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Modèle de génération. |
| `DATABASE_URL` | `sqlite:///backend/data/narsa.db` | Base des réclamations et des statistiques. |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` | `agent` / `narsa2026` | Compte agent. |
| `ADMIN_SECRET` | `narsa-secret-local` | Clé de signature des jetons — **à changer**. |
| `ADMIN_TOKEN_TTL_MINUTES` | `480` | Durée de validité du jeton. |
| `CORS_ORIGINS` | `*` | Origines autorisées. |
| `SERVE_FRONTEND` | `true` | Sert `frontend/web` depuis l'API. |
| `LOG_CONVERSATIONS` | `true` | Journalise les échanges pour le dashboard. |
| `PRELOAD_MODELS` | `true` | Charge les modèles au démarrage (première question plus rapide). |
| `GEMINI_FALLBACK_MODELS` | `gemini-3.5-flash-lite,gemini-2.5-flash` | Modèles essayés quand le quota du principal est épuisé. |
| `GEMINI_QUOTA_COOLDOWN_SECONDS` | `900` | Durée de mise au repos d'un modèle après un 429. |
| `ANSWER_CACHE_ENABLED` | `true` | Cache des réponses documentaires. |
| `EMAIL_ENABLED` | `true` | Notifications email (console si `SMTP_HOST` est vide). |
| `SMTP_HOST` / `SMTP_PORT` | *(vide)* / `587` | Serveur d'envoi des notifications. |
| `SMTP_USERNAME` / `SMTP_PASSWORD` | *(vide)* | Compte SMTP. Le mot de passe ne vit que dans `.env`, jamais dans `.env.example` ni dans le code. |
| `SMTP_USE_TLS` / `SMTP_USE_SSL` | `true` / `false` | STARTTLS (port 587) ou SSL direct (port 465). |
| `SMTP_TIMEOUT` | `10` | Délai maximum d'une tentative SMTP, en secondes. |
| `EMAIL_SENDER` | `no-reply@narsa.gov.ma` | Expéditeur des emails. |
| `PORTAL_URL` | `http://127.0.0.1:8000` | URL du portail insérée dans les emails. |

### Notifications email

Le mail n'est qu'une **notification** : la réponse officielle complète reste
consultable sur le portail de suivi (`/#/suivi`), jamais dans l'email.

Une panne SMTP ne fait perdre aucune donnée : la réponse de l'agent est
enregistrée et committée **avant** toute tentative d'envoi. L'échec éventuel
est journalisé (`narsa.email`) avec un diagnostic, et remonté à l'agent dans
l'espace d'administration — le mot de passe n'apparaît jamais, ni dans les
journaux ni dans les réponses de l'API.

Exemple Gmail / Google Workspace dans `.env` :

```env
EMAIL_ENABLED=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=votre.compte@gmail.com
SMTP_PASSWORD=xxxxxxxxxxxxxxxx   # mot de passe d'application, 16 caractères
SMTP_USE_TLS=true
SMTP_USE_SSL=false
EMAIL_SENDER=votre.compte@gmail.com
PORTAL_URL=http://127.0.0.1:8000
```

Le mot de passe d'application se crée sur
<https://myaccount.google.com/apppasswords>, après activation de la validation
en 2 étapes. Le mot de passe principal du compte Google est systématiquement
refusé par `smtp.gmail.com`.

Pour vérifier la configuration sans toucher à un dossier réel :

```bash
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/admin/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"agent","password":"narsa2026"}' \
  | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

curl -s http://127.0.0.1:8000/admin/email/status -H "Authorization: Bearer $TOKEN"

curl -s -X POST http://127.0.0.1:8000/admin/email/test \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"to":"destinataire@exemple.ma"}'
```

Un envoi n'est réussi que si le statut retourné est `sent`. Les statuts
`console` (aucun `SMTP_HOST`), `disabled` (`EMAIL_ENABLED=false`) et `failed`
signifient qu'**aucun email n'est parti**.

## Tests

```bash
.venv/bin/python -m pytest
```

## Structure

```text
backend/app/
  analytics/    agrégation des indicateurs du Dashboard IA
  classifier/   entraînement et service d'inférence des intentions
  complaints/   logique métier des réclamations
  core/         configuration et sécurité (jetons agents)
  database/     modèles SQLAlchemy et session
  decision/     routage conceptuel d'une intention
  rag/          ingestion ChromaDB, retriever, RAG, génération Gemini
  routes/       endpoints FastAPI
  schemas/      contrats Pydantic
frontend/web/   interface web servie par l'API (HTML/CSS/JS)
frontend/app.py interface Streamlit historique (optionnelle)
backend/data/   documents Markdown, formulaires PDF, index Chroma, base SQLite
```

`PROJECT_OVERVIEW.md` détaille le fonctionnement interne du pipeline.

L'interface Streamlit reste disponible (`streamlit run frontend/app.py`) mais
l'interface de référence est désormais `frontend/web`.
