# Essai de Pulsar.ml sur des données ouvertes du gouvernement du Canada

Date de l'essai : 4 novembre 2026. Branche `fonction/essai-donnees-gc` du dépôt `pulsar_demo`.

## 1. But et portée

Le projet vise le TRL 5 : des composants intégrés et testés ensemble dans un environnement pertinent. Jusqu'ici,
Pulsar.ml a été validé sur des données américaines (Folktables, Bike Sharing, taxis de New York). Cet essai le fait
tourner sur des données ouvertes fédérales canadiennes, avec des changements connus :

- partie 1 : un modèle tabulaire traverse le confinement de la COVID-19 au printemps 2020 ;
- partie 2 : un agent de recherche documentaire répond sur le Registre de l'IA du gouvernement du Canada, puis on
  provoque des changements.

C'est une démonstration de taille modeste, sur un seul poste, sans GPU. Les scripts, les sommes de contrôle et les
résultats sont versionnés dans `pulsar_demo/essai_donnees_gc/`.

## 2. Données et licences

| Jeu de données | Éditeur | Licence | Ce qui est utilisé |
|---|---|---|---|
| Programme national de surveillance de la pollution atmosphérique (SNPA / NAPS), NO2 horaire | ECCC | Licence du gouvernement ouvert – Canada ([fiche](https://open.canada.ca/data/dataset/1b36a356-defd-4813-acea-47bc3abd859b)) | 6 fichiers annuels 2016-2021 ; 8 stations de Montréal |
| Données climatiques historiques, horaires, station MONTREAL INTL A (ID climatologique 7025251) | ECCC | Licence d'utilisation finale des serveurs de données d'ECCC, version 2.1.1 (usage commercial permis, attribution) | 72 fichiers mensuels 2016-2021 |
| Registre de l'IA du gouvernement du Canada (produit minimal viable) | SCT | Licence du gouvernement ouvert – Canada ([fiche](https://open.canada.ca/data/dataset/fcbc0200-79ba-4fa4-94a6-00e32facea6b)) | 412 systèmes, version d'avril 2026 |

- `download.py` télécharge les 79 fichiers (58 Mo) et écrit `sources.json` : URL, date, taille et SHA-256 de chaque
  fichier. Les fichiers bruts ne sont pas dans git.
- Aucune donnée personnelle : des mesures agrégées et des descriptions de systèmes. Du registre, on écarte les
  colonnes fournisseur et fichiers de renseignements personnels. Une recherche de courriels et de numéros de
  téléphone dans toutes les colonnes n'a rien trouvé.
- Pourquoi le NO2 : une dimension temporelle longue (52 607 heures utilisables), une licence ouverte, et un
  changement réel bien daté. Le Québec a déclaré l'urgence sanitaire le 13 mars 2020, fermé les écoles à partir du
  16 mars et les commerces non essentiels à partir du 25 mars. La circulation, principale source de NO2 en ville, a
  chuté. Le modèle, lui, ne voit que la météo et le calendrier : il doit surestimer le NO2 après le 16 mars 2020.

## 3. Environnement exact

| Élément | Valeur |
|---|---|
| Processeur | AMD Ryzen 7 7730U, 8 cœurs, 16 fils, aucun GPU NVIDIA |
| Mémoire | 38,8 Go (hôte) ; machine virtuelle WSL2 de Docker : 16 processeurs, 19 Go |
| Système | Windows 11 Pro 10.0.26300 ; noyau WSL2 6.6.87.2 |
| Docker | Docker Desktop, moteur 28.0.4 |
| Base | InfluxDB 2.6.1 (conteneur `pulsar-validation-influxdb`), seau `essai_gc`, rétention 90 jours |
| Serveur de modèles | Ollama 0.6.8 (conteneur `pulsar-ollama`), CPU seulement, 2 requêtes simultanées par modèle, contexte 4096 par requête |
| Python | 3.12.10 |
| Pulsar | pulsar-metrics 0.12.0 (commit `cd18c94` de main), pulsar-data-collection 0.11.0 (commit `e67491a` de main) |
| Autres paquets | scikit-learn 1.9.1, pandas 2.3.3, numpy 2.5.4, scipy 1.18.1, influxdb-client 1.50.0, openai 3.42.0, opentelemetry-sdk 1.45.2, torch 2.15.0+cpu, transformers 4.57.6 |
| Modèle de l'agent | `llama3.2:3b` (Meta, Q4_K_M, 3,2 G de paramètres), condensé `a80c4f17acd5…` |
| Modèle de remplacement | `granite3.3:2b` (IBM, Q4_K_M, 2,5 G), condensé `07bd1f170855…` |
| Plongements des requêtes | `nomic-embed-text` (Nomic AI, F16, 137 M), condensé `0a109f422b47…` |
| Fidélité (NLI) | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (MIT), révision `b5113eb3…`, sur CPU |

Le poste était partagé avec d'autres travaux. Un calcul lourd d'un autre lot (16 processus) a tourné de 20 h 45 à
21 h 24 UTC : pendant ce temps, une seule complétion du modèle prenait plus de 5 minutes. La campagne de la partie 2
a été relancée après, sur une machine au repos ; les durées de la partie 1 ont été mesurées sur le poste partagé.

## 4. Partie 1 : modèle NO2 de Montréal

### Méthode

- Cible : moyenne horaire du NO2 de 8 stations de Montréal (au moins 5 stations présentes dans l'heure).
- Variables : heure, jour de la semaine, jour de l'année (sinus, cosinus), température, point de rosée, humidité,
  vent (vitesse, direction en sinus et cosinus), pression, à l'aéroport Montréal-Trudeau.
- Modèle : `HistGradientBoostingRegressor` de scikit-learn 1.9.1 (300 itérations, pas 0,05), entraîné sur 2016-2018.
- Référence de surveillance : prédictions hors échantillon de 2016-2018 (chaque année prédite par un modèle entraîné
  sur les deux autres) ; 26 303 heures ; MAE 2,65 ppb.
- Production simulée : les 157 semaines de 2019 à 2021 (26 304 heures) passent par `Pulse` dans InfluxDB, une
  capture par semaine, avec la vérité terrain. 26 304 lignes écrites sur 26 304, aucune perdue.
- Analyse, semaine par semaine contre la référence, trois groupes de signaux, chacun corrigé par Benjamini-Hochberg
  (FDR 5 %) et confirmé sur 2 semaines consécutives (`DriftPersistence`) :
  - erreur du modèle (observé moins prédit) : `ks_2samp` et `wasserstein`, seuils calibrés par tirages A/A
    (`AACalibrator`, `compare_to="reference"`, fenêtres d'une semaine) ;
  - performance : MAE de la semaine au-dessus du 95e centile des MAE hebdomadaires de la référence ;
  - entrées : `ks_2samp` sur les 7 variables météo et la prédiction. C'est un témoin négatif : un confinement ne
    change pas la météo.
- Deux références : `période` (mêmes semaines de 2016-2018 ± 4 semaines, `period_reference`, réglage recommandé pour
  des données saisonnières) et `annuelle` (toute la période 2016-2018).
- Période stable : semaines commençant avant le 9 mars 2020 (62 semaines). Semaine du 9 mars : transition, exclue.
  Période changée : à partir du 16 mars 2020 (94 semaines).

### Résultats

Erreur moyenne du modèle (observé moins prédit), par période :

| Période | Semaines | Erreur moyenne (ppb) | MAE moyenne (ppb) |
|---|---|---|---|
| Référence hors échantillon 2016-2018 | 156 | +0,02 | 2,65 |
| Stable, janv. 2019 à début mars 2020 | 62 | −0,31 | 2,66 |
| Printemps 2020 (16 mars au 30 juin) | 16 | −2,60 | 3,09 |
| Toute la période changée (16 mars 2020 à déc. 2021) | 94 | −2,10 | 2,88 |

L'écart-type des erreurs moyennes hebdomadaires de la référence est de 1,64 ppb : le changement fait environ
1,5 écart-type, pas plus.

Détections (intervalles de Clopper-Pearson à 95 %) :

| Référence | Signal | Fausses alarmes, semaines stables (brutes) | Fausses alarmes confirmées | 1re alerte après le 16 mars 2020 | 1re confirmation | Semaines changées confirmées |
|---|---|---|---|---|---|---|
| période | erreur | 0 / 62 [0 ; 0,058] | 0 / 62 [0 ; 0,058] | 30 mars (+2 sem.) | 6 avril (+3 sem.) | 34 / 94 [0,27 ; 0,47] |
| période | MAE | 1 / 62 [0 ; 0,087] | 0 / 62 | 4 janv. 2021 (+42 sem.) | jamais | 0 / 94 |
| période | entrées (témoin) | 2 / 62 [0,004 ; 0,11] | 0 / 62 | — | jamais | 0 / 94 [0 ; 0,038] |
| annuelle | erreur | 1 / 62 [0 ; 0,087] | 0 / 62 | 16 mars (+0 sem.) | 23 mars (+1 sem.) | 13 / 94 [0,08 ; 0,23] |
| annuelle | MAE | 2 / 62 [0,004 ; 0,11] | 0 / 62 | 30 mars (+2 sem.) | 11 janv. 2021 (+43 sem.) | 2 / 94 |
| annuelle | entrées (témoin) | 1 / 62 [0 ; 0,087] | 0 / 62 | — | 11 janv. 2021 (prédiction) | 2 / 94 |

Ce qu'on en retient :

- Le changement réel est vu par le signal d'erreur, dans les 1 à 3 semaines selon la référence, et confirmé sur tout
  le printemps 2020 avec la référence par période (13 semaines sur 13 au 2e trimestre 2020).
- Aucune fausse alarme confirmée en 62 semaines stables, pour aucun signal. La borne haute est de 5,8 % par semaine.
- La MAE ne voit presque rien : le modèle se trompe de 2,7 ppb en moyenne en temps normal, et un biais de −2,6 ppb ne
  la fait monter qu'à 3,1. Un test sur la distribution de l'erreur (biais compris) est plus sensible qu'un seuil de
  MAE.
- Après l'été 2020, le NO2 reste sous la prédiction (−1,6 à −2,5 ppb par trimestre), mais le signal ne confirme qu'une
  partie des semaines : avec des fenêtres d'une semaine, un écart de 1,5 écart-type est à la limite de la puissance.
- Le témoin négatif (météo) n'est jamais confirmé avec la référence par période. Avec la référence annuelle, la
  prédiction est confirmée 2 semaines en janvier 2021 : c'est une météo d'hiver hors de la plage de la référence
  annuelle, pas le confinement. Les mêmes semaines, la MAE est confirmée : janvier 2021 a l'erreur moyenne la plus
  forte de l'essai (−3,8 ppb sur le mois ; −6,9 ppb la semaine du 4 janvier), pendant la fermeture des fêtes puis le
  couvre-feu de janvier 2021 au Québec. On ne peut pas séparer ici l'effet de ces mesures de celui de la météo.
- La référence annuelle confirme plus vite (1 semaine contre 3) mais moins souvent (13 semaines contre 34). La
  référence par période ne compte que 27 fenêtres : ses seuils sont marqués `reliable=False` et sont plus larges.
- Coût : 71 minutes pour la référence par période (une calibration par semaine, poste partagé), 5 minutes pour la
  référence annuelle (une seule calibration).

## 5. Partie 2 : agent sur le Registre de l'IA

### Méthode

- Agent : celui de la démo `genai_agent/` (même boucle, mêmes spans OpenTelemetry GenAI, `PulsarSpanExporter` →
  `Pulse` → InfluxDB). On lui donne un seul outil, `search_register` : recherche BM25 sur le nom, l'organisation et la
  description des 412 systèmes, 3 fiches renvoyées. Au plus 4 étapes et 200 jetons par réponse, température 0.
- Questions à réponse connue, tirées des champs structurés du registre, sur 150 systèmes au nom unique : organisation
  responsable, statut (production, développement, hors service), renseignements personnels (oui ou non). Notation
  par mots-clés, stricte : une organisation donnée par son seul sigle compte comme fausse.
- Chaque exécution est notée dans le processus ; seuls les scores vont dans InfluxDB (champs `det_*` de `agent_run`) :
  meilleur score de recherche, recherche vide, fidélité de la réponse aux fiches (NLI), et exactitude par rapport au
  registre (évaluation seulement, jamais surveillée).
- Phases, 2 requêtes simultanées :

| Phase | Exécutions | Questions | Corpus | Modèle |
|---|---|---|---|---|
| référence | 80 | français | complet | llama3.2:3b |
| témoin (aucun changement) | 40 | français | complet | llama3.2:3b |
| perte de corpus | 40 | français | 75 des 150 systèmes des questions retirés | llama3.2:3b |
| passage à l'anglais | 40 | anglais | complet | llama3.2:3b |
| changement de modèle | 40 | français | complet | granite3.3:2b |

- Signaux de Pulsar, par fenêtres de 8 exécutions comparées à la référence, confirmés sur 2 fenêtres :
  - S2 de la démo (`analyze_drift.py`) sur les enregistrements `agent_run` relus dans InfluxDB : 11 variables
    (étapes, appels d'outils, jetons, latence, scores, premier outil, fin, succès), `chi2` et `ks_2samp` calibrés
    par tirages A/A, Benjamini-Hochberg ;
  - qualité de la recherche, comme l'analyse S3 de la démo : Mann-Whitney unilatéral sur la fidélité et sur le
    meilleur score de recherche ;
  - MMD sur les plongements des questions (`nomic-embed-text`) ;
  - égalité de modèle sur un ensemble canari (48 invites × 3 réponses par ensemble).
- Mêmes tests avec des fenêtres de 20 exécutions (sans la partie S2, qui demande 10 fenêtres de référence).

### Résultats

Exactitude par rapport au registre (intervalles de Clopper-Pearson à 95 %) :

| Phase | Exactitude | Fiche visée retrouvée | Secondes par exécution (moyenne) |
|---|---|---|---|
| référence | 0,90 [0,81 ; 0,96] | 0,88 | 45 |
| témoin | 0,88 [0,73 ; 0,96] | 0,88 | 41 |
| perte de corpus | 0,57 [0,41 ; 0,73] | 0,53 | 46 |
| passage à l'anglais | 0,90 [0,76 ; 0,97] | 0,95 | 37 |
| changement de modèle | 0,25 [0,13 ; 0,41] | 0,00 | 8 |

Par type de question, en référence : organisation 0,80 (30 questions), statut 0,94 (18), renseignements personnels
0,97 (32).

Détection, fenêtres de 8 exécutions (5 fenêtres par phase) : fenêtres en alerte / première alerte / première
confirmation.

| Phase | S2 (trajectoires et scores) | Fidélité, unilatéral | Score de recherche, unilatéral | MMD des questions |
|---|---|---|---|---|
| témoin | 2 / 5, fenêtre 1, jamais confirmé | 0 / 5 | 0 / 5 | 1 / 5, fenêtre 2, jamais confirmé |
| perte de corpus | 1 / 5, fenêtre 5, jamais | 2 / 5, fenêtre 1, jamais | 2 / 5, fenêtre 2, jamais | 0 / 5 |
| passage à l'anglais | 3 / 5, fenêtre 3, confirmé fenêtre 4 (`output_tokens`) | 0 / 5 | 0 / 5 | 5 / 5, fenêtre 1, confirmé fenêtre 2 |
| changement de modèle | 5 / 5, fenêtre 1, confirmé fenêtre 2 (9 variables) | sans objet (aucune fiche) | 5 / 5, fenêtre 1, confirmé fenêtre 2 | 0 / 5 |

Avec des fenêtres de 20 exécutions, la perte de corpus est confirmée par la fidélité à la 2e fenêtre (après 40
exécutions) ; le témoin reste sans alerte sur la fidélité, le score de recherche et le MMD.

Ensemble canari (48 invites, 3 réponses chacune, température 0,7), test d'égalité de modèle contre l'ensemble de
référence de llama3.2:3b :

| Ensemble | Modèle | p | Statistique |
|---|---|---|---|
| même modèle, second tirage | llama3.2:3b | 0,039 | 0,043 |
| changement | granite3.3:2b | 0,001 (plancher de 1 000 permutations) | 0,257 |

Le changement de modèle est net (statistique 6 fois plus grande). Mais le second tirage du même modèle passe aussi
sous 0,05 : c'est une fausse alarme du test canari, avec un seul test, donc sans possibilité de la confirmer sur 2
fenêtres. Avec 144 réponses par ensemble, le seuil de 0,05 ne suffit pas à séparer le bruit d'échantillonnage du
changement ; il faudrait plus de réponses ou répéter l'ensemble avant d'alerter.

Ce qu'on en retient :

- Aucune fausse alarme confirmée sur le témoin (40 exécutions, 5 fenêtres), pour les signaux par fenêtres. Le test
  canari, lui, a donné une fausse alarme (p = 0,039 sur le même modèle). Il y a 3 alertes
  brutes isolées (`output_tokens`, `latency_s`, MMD), que la persistance sur 2 fenêtres écarte.
- Changement de modèle : vu dès la 1re fenêtre et confirmé à la 2e (16 exécutions), et par le canari (p = 0,001). granite3.3:2b n'a appelé l'outil
  dans aucune des 40 exécutions : il répond de mémoire, en anglais, et se trompe 3 fois sur 4. Le premier outil, le
  nombre d'étapes, les jetons, la latence et le score de recherche changent tous.
- Passage à l'anglais : le MMD des plongements le voit dès la 1re fenêtre et le confirme à la 2e. Les trajectoires ne
  changent presque pas (S2 confirme `output_tokens` à la 4e fenêtre). L'exactitude ne baisse pas : c'est un
  changement des entrées, pas une panne.
- Perte de corpus : c'est le changement le plus coûteux pour l'utilisateur (exactitude 0,90 → 0,57) et le moins
  visible. Sur les 18 questions dont le système avait été retiré, l'agent n'a dit « absent du registre » que 3 fois ; les 15 autres fois, il a donné une réponse sans appui, le plus souvent fausse (exactitude 0,22 sur ces 18). La fidélité baisse (0,52 → 0,33 en moyenne)
  et le meilleur score de recherche aussi (18,6 → 14,2), mais avec 8 exécutions par fenêtre, dont la moitié seulement
  touchée, aucun test ne confirme. Il faut des fenêtres de 20 pour confirmer, soit 40 exécutions.
- La partie S2 générique (11 variables bilatérales, Benjamini-Hochberg) manque de puissance pour une baisse d'un seul
  score ; les tests unilatéraux ciblés réagissent plus tôt. Les deux sont utiles : S2 pour l'inattendu, les tests
  ciblés pour les pannes connues de la recherche.

## 6. Limites

- Taille : un seul jeu tabulaire, une ville, un changement réel ; 240 exécutions d'agent. Les intervalles sont
  larges (par exemple, fausses alarmes confirmées 0 sur 62 semaines : jusqu'à 5,8 % par semaine reste compatible).
- Partie 1 : le « changement connu » est un fait public bien daté, mais sa taille réelle sur le NO2 n'est pas
  connue d'avance ; on mesure l'écart à un modèle, pas l'effet causal du confinement. La baisse lente du NO2 d'une
  année sur l'autre (erreur moyenne −0,31 ppb en 2019) s'ajoute au changement.
- Partie 1 : la référence par période ne compte que 27 fenêtres ; ses seuils sont marqués non fiables par
  `AACalibrator`. Une référence plus longue (plus d'années) les resserrerait.
- Partie 1 : rejeu accéléré. Les points portent l'heure du rejeu et l'heure d'observation est un champ (`obs_ts`),
  pour garder une rétention finie dans InfluxDB.
- Partie 2 : les questions sont générées par gabarit, pas posées par de vrais utilisateurs. La notation par
  mots-clés est stricte et peut compter faux une bonne réponse formulée autrement (sigle, traduction libre).
- Partie 2 : la perte de corpus touche la moitié des systèmes interrogés ; une perte plus petite serait encore moins
  visible avec des fenêtres de 8.
- Partie 2 : le changement de modèle est massif (l'autre modèle n'utilise plus l'outil). Un changement plus subtil
  (autre quantification du même modèle) n'a pas été testé ici ; le laboratoire a montré qu'il demande environ 480
  réponses canari.
- Poste partagé : pendant un calcul lourd d'un autre lot, une complétion prenait plus de 5 minutes (au lieu de 5 à
  40 secondes) et dépassait le délai ; la campagne a été relancée sur machine au repos. Lors d'un essai préalable,
  Ollama 0.6.8 a continué de calculer au moins 10 minutes après l'arrêt du client, sur des requêtes sans plafond de
  jetons ; il a fallu décharger le modèle pour libérer le serveur. L'agent plafonne donc chaque réponse (`max_tokens=200`).
- Défaut trouvé dans pulsar-data-collection 0.11.0 : `FaithfulnessScorer.score` n'est pas sûr entre fils
  d'exécution. Appelé depuis 4 fils, il lève `RuntimeError: Already borrowed` (jeton de découpage Hugging Face
  partagé) dans 37 appels sur 40 ; en séquence, 0 sur 10. Dans la campagne (2 fils), 1 exécution sur 240 a échoué
  ainsi. Non corrigé ici (hors du périmètre de ce lot) ; correctif proposé : un verrou autour de l'appel au modèle, ou
  un découpeur par fil, et une mention dans la documentation.

## 7. Reproduire

Voir le [README](README.md). Durées sur ce poste : téléchargement 2 minutes ; rejeu 2 secondes ; analyse de la
partie 1 76 minutes (poste partagé) ; campagne de la partie 2 77 minutes pour les 240 exécutions plus
57 minutes pour l'ensemble canari ; analyse de la partie 2 moins de 2 minutes.
