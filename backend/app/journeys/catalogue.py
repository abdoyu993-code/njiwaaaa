from __future__ import annotations

from typing import Any


"""
Parcours administratifs guidés.

Chaque étape est construite à partir des documents officiels
indexés dans `backend/data/documents`. Le champ `source`
indique le document dont l'étape est tirée : il permet à
l'interface de renvoyer vers le passage d'origine.
"""


JOURNEYS: dict[str, dict[str, Any]] = {
    "achat_vehicule_occasion": {
        "title": "J'ai acheté un véhicule d'occasion",
        "summary": (
            "Mettre la carte grise à votre nom (mutation) "
            "dans le délai réglementaire."
        ),
        "domain": "vehicule",
        "intents": (
            "mutation_vehicule",
            "changement_proprietaire",
            "achat_vehicule",
            "vehicule_vendu",
            "carte_grise_demande",
        ),
        "markers": (
            "achete une voiture",
            "achete un vehicule",
            "acheté une voiture",
            "voiture d occasion",
            "vehicule d occasion",
            "voiture doccasion",
            "changement de proprietaire",
            "changer la carte grise",
            "mutation",
            "j ai achete",
        ),
        "deadline": (
            "30 jours après la légalisation des signatures "
            "de la déclaration de vente"
        ),
        "warning": (
            "Au-delà de 30 jours, une amende administrative "
            "de 500 dirhams peut être appliquée, majorée de "
            "10 % par mois de retard."
        ),
        "steps": [
            {
                "title": "Faire la déclaration de vente sur le portail",
                "detail": (
                    "Le vendeur crée ou complète la déclaration de "
                    "vente, puis l'acheteur complète les informations "
                    "relatives à l'achat sur le portail de mutation "
                    "des véhicules."
                ),
                "duration": "Le jour de la transaction",
                "documents": [],
                "links": [
                    {
                        "label": "Portail de mutation des véhicules",
                        "url": "https://khadamatnarsa.ma",
                    }
                ],
                "forms": [],
                "source": "vehicule/mutation_vehicule_narsa.md",
            },
            {
                "title": "Faire légaliser les signatures",
                "detail": (
                    "Les signatures de la déclaration de vente "
                    "doivent être légalisées. C'est cette date qui "
                    "fait courir le délai de 30 jours."
                ),
                "duration": "1 jour",
                "documents": ["Déclaration de vente signée"],
                "links": [],
                "forms": [],
                "source": "vehicule/mutation_vehicule_narsa.md",
            },
            {
                "title": "Passer le contrôle technique de mutation",
                "detail": (
                    "Un procès-verbal de contrôle technique "
                    "spécifique à la mutation, délivré par un centre "
                    "agréé et établi pour le véhicule concerné, est "
                    "demandé dans le dossier."
                ),
                "duration": "1 jour, sur rendez-vous",
                "documents": [
                    "Procès-verbal de contrôle technique pour la mutation",
                ],
                "links": [
                    {
                        "label": "Prendre rendez-vous pour le contrôle technique",
                        "url": "https://controletechnique.ma/Rendez-vous",
                    }
                ],
                "forms": [],
                "source": (
                    "visite_technique/controle_technique_mutation_narsa.md"
                ),
            },
            {
                "title": "Constituer le dossier de mutation",
                "detail": (
                    "Réunissez les pièces demandées. Si une "
                    "opposition ou un crédit est enregistré sur le "
                    "véhicule, une mainlevée est indispensable avant "
                    "le dépôt."
                ),
                "duration": "Quelques jours",
                "documents": [
                    "Justificatifs de résidence",
                    "Original du certificat d'immatriculation au nom du vendeur",
                    "Procès-verbal de contrôle technique pour la mutation",
                    "Contrat de financement en cas de crédit",
                    "Mainlevée en cas d'opposition sur le véhicule",
                    "Reçu de paiement des droits de timbre et de rémunération",
                ],
                "links": [],
                "forms": [
                    "formulaire_carte_grise_formule_bi.pdf",
                    "formulaire_carte_grise_formule_bii.pdf",
                ],
                "source": (
                    "visite_technique/controle_technique_mutation_narsa.md"
                ),
            },
            {
                "title": "Déposer le dossier en agence",
                "detail": (
                    "Le dossier se dépose auprès d'une agence "
                    "accréditée Al Barid Bank ou Barid Cash. Un "
                    "récépissé de dépôt valable 60 jours vous est "
                    "remis."
                ),
                "duration": "À faire avant la fin des 30 jours",
                "documents": ["Dossier de mutation complet"],
                "links": [],
                "forms": [],
                "source": "vehicule/mutation_vehicule_narsa.md",
            },
            {
                "title": "Recevoir la carte grise à votre nom",
                "detail": (
                    "La NARSA vérifie et valide la mutation, puis le "
                    "certificat d'immatriculation est établi au nom "
                    "du nouveau propriétaire."
                ),
                "duration": "Après validation par la NARSA",
                "documents": [],
                "links": [],
                "forms": [],
                "source": "vehicule/mutation_vehicule_narsa.md",
            },
        ],
    },

    "permis_perdu": {
        "title": "J'ai perdu mon permis de conduire",
        "summary": (
            "Obtenir un duplicata de votre permis après une "
            "perte ou un vol."
        ),
        "domain": "permis",
        "intents": (
            "permis_perte",
            "permis_vol",
            "permis_duplicata",
        ),
        "markers": (
            "perdu mon permis",
            "perdu le permis",
            "permis perdu",
            "vol de mon permis",
            "permis vole",
            "duplicata du permis",
            "duplicata permis",
            "refaire mon permis",
        ),
        "deadline": "Aucun délai réglementaire imposé",
        "warning": (
            "Conduire sans permis en cours de validité vous "
            "expose à une sanction : faites la demande sans "
            "attendre."
        ),
        "steps": [
            {
                "title": "Déclarer la perte ou le vol",
                "detail": (
                    "Faites établir une déclaration de perte ou de "
                    "vol auprès des services compétents. Elle fait "
                    "partie du dossier."
                ),
                "duration": "1 jour",
                "documents": ["Déclaration de perte ou de vol"],
                "links": [],
                "forms": [],
                "source": "permis/permis_perte_vol_narsa.md",
            },
            {
                "title": "Obtenir un certificat médical",
                "detail": (
                    "Un certificat médical de moins de trois mois, "
                    "délivré par un médecin agréé, atteste votre "
                    "aptitude physique et mentale."
                ),
                "duration": "1 à 3 jours",
                "documents": ["Certificat médical de moins de 3 mois"],
                "links": [],
                "forms": [],
                "source": "permis/permis_perte_vol_narsa.md",
            },
            {
                "title": "Payer les droits et rassembler les pièces",
                "detail": (
                    "Payez les droits de timbre et de rémunération "
                    "auprès d'une perception de la Trésorerie "
                    "Générale du Royaume, puis réunissez les pièces "
                    "du dossier."
                ),
                "duration": "1 jour",
                "documents": [
                    "Formulaire spécial rempli et signé",
                    "Copie valide de la CNIE ou du passeport marocain",
                    "Reçu de paiement des droits de timbre et de rémunération",
                    "Deux photos d'identité 35 × 45 mm, fond bleu",
                ],
                "links": [],
                "forms": [],
                "source": "permis/permis_perte_vol_narsa.md",
            },
            {
                "title": "Déposer la demande en agence",
                "detail": (
                    "La demande se dépose dans une agence Al Barid "
                    "Bank ou Barid Cash offrant les services NARSA. "
                    "L'agence saisit le dossier dans le système."
                ),
                "duration": "Le jour du dépôt",
                "documents": ["Dossier complet"],
                "links": [
                    {
                        "label": "Service officiel de duplicata du permis",
                        "url": (
                            "https://khadamatnarsa.ma/fr/services/"
                            "duplicata-de-mon-permis-de-conduire"
                        ),
                    }
                ],
                "forms": [],
                "source": "permis/permis_duplicata_narsa.md",
            },
            {
                "title": "Récupérer le duplicata",
                "detail": (
                    "Après validation par les services de la NARSA, "
                    "un SMS vous invite à retirer le permis dans "
                    "l'agence où le dossier a été déposé."
                ),
                "duration": "Suivi possible sur KHADAMAT NARSA",
                "documents": [],
                "links": [
                    {
                        "label": "Suivi des dossiers",
                        "url": "https://khadamatnarsa.ma/fr/suivi-des-dossiers",
                    }
                ],
                "forms": [],
                "source": "permis/permis_duplicata_narsa.md",
            },
        ],
    },

    "contester_contravention": {
        "title": "Je conteste une contravention",
        "summary": (
            "Déposer une réclamation sur un avis de "
            "contravention dans le délai de 30 jours."
        ),
        "domain": "infractions",
        "intents": (
            "contestation_amende",
            "reclamation_infraction",
        ),
        "markers": (
            "contester une amende",
            "contester une contravention",
            "contester l amende",
            "reclamation contravention",
            "amende injuste",
            "avis de contravention",
            "amende que je n ai pas commise",
        ),
        "deadline": (
            "30 jours à compter de la notification de l'avis "
            "de contravention"
        ),
        "warning": (
            "Passé ce délai, la réclamation n'est plus "
            "recevable."
        ),
        "steps": [
            {
                "title": "Vérifier votre motif de réclamation",
                "detail": (
                    "Les motifs admis sont notamment : véhicule "
                    "déclaré volé avant l'infraction, plaque "
                    "utilisée abusivement, véhicule vendu ou cédé "
                    "avant l'infraction, véhicule non enregistré à "
                    "votre nom, ou véhicule en mission d'urgence."
                ),
                "duration": "Immédiat",
                "documents": ["Avis de contravention reçu"],
                "links": [],
                "forms": [],
                "source": "infractions/reclamation_infraction_narsa.md",
            },
            {
                "title": "Remplir le formulaire de réclamation",
                "detail": (
                    "Le formulaire officiel doit être rempli et "
                    "signé. Il précise notamment les informations "
                    "du véhicule et le motif invoqué."
                ),
                "duration": "30 minutes",
                "documents": ["Formulaire de réclamation rempli et signé"],
                "links": [],
                "forms": ["formulaire_reclamation_contravention.pdf"],
                "source": "infractions/reclamation_infraction_narsa.md",
            },
            {
                "title": "Joindre les justificatifs du motif",
                "detail": (
                    "Véhicule volé : copie de la déclaration de vol. "
                    "Véhicule vendu : document prouvant la vente. "
                    "Mission d'urgence ou usage abusif de la plaque : "
                    "tout document le démontrant."
                ),
                "duration": "1 à 2 jours",
                "documents": [
                    "Copie de l'avis de contravention",
                    "Pièces justificatives du motif invoqué",
                ],
                "links": [],
                "forms": [],
                "source": "infractions/reclamation_infraction_narsa.md",
            },
            {
                "title": "Déposer la réclamation",
                "detail": (
                    "Le dépôt s'effectue auprès d'une agence Al "
                    "Barid Bank ou Barid Cash proposant les services "
                    "NARSA. Le formulaire officiel mentionne "
                    "également les services extérieurs de la NARSA : "
                    "vérifiez le lieu indiqué sur la version en "
                    "vigueur."
                ),
                "duration": "Avant la fin des 30 jours",
                "documents": ["Dossier de réclamation complet"],
                "links": [],
                "forms": [],
                "source": "infractions/reclamation_infraction_narsa.md",
            },
            {
                "title": "Suivre la réponse",
                "detail": (
                    "Conservez le récépissé de dépôt. Vous pouvez "
                    "aussi déposer une réclamation sur cette "
                    "plateforme pour garder une trace du suivi."
                ),
                "duration": "Variable",
                "documents": [],
                "links": [],
                "forms": [],
                "source": "contact_reclamation/suivi_dossiers_narsa.md",
            },
        ],
    },

    "immatriculer_vehicule": {
        "title": "J'immatricule un véhicule neuf ou importé",
        "summary": (
            "Obtenir la première carte grise d'un véhicule "
            "neuf ou importé."
        ),
        "domain": "vehicule",
        "intents": (
            "immatriculation_vehicule",
            "vehicule_neuf",
            "vehicule_importe",
        ),
        "markers": (
            "voiture neuve",
            "vehicule neuf",
            "vehicule importe",
            "voiture importee",
            "importer un vehicule",
            "importer une voiture",
            "importe un vehicule",
            "importation d un vehicule",
            "premiere immatriculation",
            "immatriculer",
            "immatriculation",
        ),
        "deadline": "Selon la situation du véhicule",
        "warning": (
            "Les formules I et II ne s'appliquent pas aux "
            "mêmes cas : vérifiez la formule applicable "
            "auprès du centre d'immatriculation."
        ),
        "steps": [
            {
                "title": "Réunir les documents d'origine du véhicule",
                "detail": (
                    "Selon qu'il s'agit d'un véhicule neuf ou "
                    "importé, les justificatifs d'acquisition et de "
                    "dédouanement diffèrent."
                ),
                "duration": "Quelques jours",
                "documents": [
                    "Justificatif d'acquisition du véhicule",
                    "Documents douaniers pour un véhicule importé",
                ],
                "links": [],
                "forms": [],
                "source": "vehicule/immatriculation_vehicules_narsa.md",
            },
            {
                "title": "Télécharger et remplir la bonne formule",
                "detail": (
                    "Deux formules existent : la Formule I et la "
                    "Formule II. Vérifiez celle qui correspond à "
                    "votre situation."
                ),
                "duration": "30 minutes",
                "documents": ["Formulaire d'immatriculation rempli"],
                "links": [],
                "forms": [
                    "formulaire_immatriculation_formule_i.pdf",
                    "formulaire_immatriculation_formule_ii.pdf",
                ],
                "source": "vehicule/immatriculation_vehicules_narsa.md",
            },
            {
                "title": "Payer les droits et déposer le dossier",
                "detail": (
                    "Le dossier se dépose auprès d'une agence "
                    "accréditée, accompagné du reçu de paiement des "
                    "droits de timbre et de rémunération."
                ),
                "duration": "Le jour du dépôt",
                "documents": [
                    "Reçu de paiement des droits",
                    "Dossier complet",
                ],
                "links": [],
                "forms": [],
                "source": "vehicule/immatriculation_vehicules_narsa.md",
            },
            {
                "title": "Recevoir le certificat d'immatriculation",
                "detail": (
                    "Après vérification, la NARSA établit le "
                    "certificat d'immatriculation du véhicule."
                ),
                "duration": "Après validation",
                "documents": [],
                "links": [],
                "forms": [],
                "source": "vehicule/immatriculation_vehicules_narsa.md",
            },
        ],
    },

    "visite_technique": {
        "title": "Je dois passer la visite technique",
        "summary": (
            "Prendre rendez-vous dans un centre de contrôle "
            "technique agréé."
        ),
        "domain": "visite_technique",
        "intents": (
            "visite_technique_rdv",
            "visite_technique_centre",
            "visite_technique_obligation",
        ),
        "markers": (
            "visite technique",
            "controle technique",
            "rendez vous visite",
            "rdv visite technique",
        ),
        "deadline": "Selon la périodicité applicable au véhicule",
        "warning": (
            "La plateforme de rendez-vous est une source "
            "complémentaire : les règles officielles "
            "(périodicité, tarifs, contre-visite) doivent "
            "être vérifiées auprès de la NARSA."
        ),
        "steps": [
            {
                "title": "Préparer les informations du véhicule",
                "detail": (
                    "Le type de véhicule et le type de visite "
                    "demandé sont nécessaires pour la prise de "
                    "rendez-vous."
                ),
                "duration": "Immédiat",
                "documents": ["Carte grise du véhicule"],
                "links": [],
                "forms": [],
                "source": "visite_technique/rendez_vous_controle_technique.md",
            },
            {
                "title": "Choisir un centre agréé",
                "detail": (
                    "Sélectionnez un centre disponible et vérifiez "
                    "qu'il est agréé pour le type de contrôle "
                    "demandé."
                ),
                "duration": "Immédiat",
                "documents": [],
                "links": [
                    {
                        "label": "Plateforme de rendez-vous",
                        "url": "https://controletechnique.ma/Rendez-vous",
                    }
                ],
                "forms": [],
                "source": "visite_technique/rendez_vous_controle_technique.md",
            },
            {
                "title": "Confirmer la date et l'heure",
                "detail": (
                    "Renseignez les informations du client, "
                    "sélectionnez une date et une heure disponibles, "
                    "puis confirmez le rendez-vous."
                ),
                "duration": "Immédiat",
                "documents": [],
                "links": [],
                "forms": [],
                "source": "visite_technique/rendez_vous_controle_technique.md",
            },
            {
                "title": "Se présenter au contrôle",
                "detail": (
                    "Présentez-vous au centre avec la carte grise du "
                    "véhicule à la date choisie."
                ),
                "duration": "Le jour du rendez-vous",
                "documents": ["Carte grise", "Confirmation de rendez-vous"],
                "links": [],
                "forms": [],
                "source": "visite_technique/rendez_vous_controle_technique.md",
            },
        ],
    },
}


def journey_keys() -> list[str]:
    return list(JOURNEYS)


def get_journey(key: str) -> dict[str, Any] | None:
    return JOURNEYS.get(key)
