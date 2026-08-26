from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# Les tests ne doivent jamais écrire dans la base locale
# de développement : une base temporaire est utilisée.
_TEST_DB_DIR = tempfile.mkdtemp(prefix="narsa-tests-")

os.environ.setdefault(
    "DATABASE_URL",
    f"sqlite:///{Path(_TEST_DB_DIR) / 'narsa-test.db'}",
)
# Les tests ne doivent JAMAIS utiliser le vrai serveur SMTP du
# poste : sans cela, un `.env` correctement configuré ferait
# partir de vrais emails à chaque exécution de la suite.
#
# `load_dotenv()` n'écrase pas les variables déjà présentes dans
# l'environnement : les fixer ici neutralise le `.env` local.
# Les tests qui vérifient la mécanique SMTP la simulent
# explicitement (voir `tests/test_response_notification.py`).
os.environ["SMTP_HOST"] = ""
os.environ["SMTP_USERNAME"] = ""
os.environ["SMTP_PASSWORD"] = ""

os.environ.setdefault("ADMIN_USERNAME", "agent")
os.environ.setdefault("ADMIN_PASSWORD", "narsa2026")
os.environ.setdefault(
    "ADMIN_SECRET",
    "secret-de-test",
)
