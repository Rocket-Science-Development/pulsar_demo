# SC-07 Résultats et tableaux de bord à l'échelle : rapport (5 novembre 2026)

Dossier `C:/dev/pulsar-ml/wt/sc07/lab` (branche `fonction/sc07-tableaux-echelle`). Test : `lab/tests/sc_07.py`. Sondes de
variantes : `lab/tests/_sc07_probes.py`. Définition : `lab/registry_platform.yaml` (SC-07).

Sources :
- `lab/results/SC-07.before.json` et `lab/results/SC-07.before/` : tableaux tels que fournis par `main` (avant correctif) ;
- `lab/results/SC-07.after.json` et `lab/results/SC-07.after/` : mêmes mesures après les deux correctifs de
  `observability/build_dashboards.py`, et échelle de cardinalité (1000 à 20 000 modèles) ;
- `lab/results/SC-07/bench_quantiles.json`, `bench_modelvar_monitoring.json`, `bench_modelvar_ladder20k.json` : sondes ;
- résumé versionné : `lab/rapports/summary.sc07.json` (statuts, chiffres de ce rapport, bloc `environment`).

Dans chaque dossier de résultats : `load.json` (écriture), `direct_samples.json` et `direct_panels.json` (requêtes
directes à InfluxDB), `grafana_samples.json` et `grafana_panels.json` (Grafana, 1 puis 20 lecteurs).

## 0. En bref

- **Critères du registre atteints après correctif.** À 7 jours, le pire panneau des tableaux overview, drift et fairness
  met **0,13 s** au p95 en requête directe (un utilisateur) et **2,44 s** au p95 (max 2,93 s) avec **20 lecteurs
  simultanés** dans Grafana. La variable `model` liste **300 modèles**. Aucun timeout Flux, aucune erreur. Statut `passed`.
- **Avant correctif, le test échoue sur les erreurs, pas sur le temps** : 78 requêtes en erreur. Le panneau « Hourly
  quantiles of ${feature} » échoue dès que la variable `feature` désigne une variable catégorielle (`quantile:
  unsupported aggregate column type string`). La variable proposait les 20 variables, dont 4 catégorielles. Corrigé :
  elle ne liste plus que les variables numériques.
- **La limite attendue (cardinalité des tags) apparaît dans la variable `model`, pas dans les panneaux.** La négation
  `r.model_id != "permit_assistant"` dans le prédicat oblige InfluxDB à tester chaque série. Son temps croît avec la
  cardinalité : 0,21 s à 1000 modèles, 2,5 s à 10 000 et 3,8 s à 20 000 (p50). Le budget de 3 s est dépassé vers
  14 000 modèles. Sous 20 lecteurs, elle coûte déjà 0,73 à 0,88 s (p50) à 300 modèles. Corrigé : le filtre s'applique
  aux valeurs, après la recherche dans l'index. On mesure alors 0,10 s à 20 000 modèles et 0,2 s sous 20 lecteurs. Les
  panneaux par modèle restent plats : 0,065 s au plus jusqu'à 20 000 modèles (2,82 millions de séries).
- **Limite restante : le panneau des quantiles horaires à 30 jours sous charge.** Il met 0,46 s seul, mais **8,9 s au
  p95** avec 20 lecteurs, et le tableau drift entier 9,3 s. La cardinalité n'y est pour rien. `quantile` et `median`
  ne sont pas poussés vers le stockage dans InfluxDB 2.6 : chaque requête lit 107 550 points bruts. InfluxDB sature
  alors 14 des 16 fils du poste. Les variantes essayées (quantile exact, une requête au lieu de trois) n'y changent
  rien. La correction porte sur l'écriture : pré-agréger les quantiles horaires. Elle n'est pas faite ici.
- **Cardinalité mesurée** : 42 300 séries d'index pour `pulsar_results` (141 par modèle), 279 000 séries par champ
  (930 par modèle) et 42 610 séries pour tout le bucket. Le produit théorique modèle × métrique × variable × groupe
  (300 × 15 × 21 × 16 = 1,5 million) n'est jamais atteint : les combinaisons réelles sont creuses (2,8 % du produit).
- **Volume** : 5 066 400 points de résultats en 36 000 requêtes `ResultsSink.write_results`, 160 s, 31 600 points/s,
  0 erreur. Le bucket complet (résultats sur 30 jours, captures de 10 modèles, compteurs de 300 modèles) occupe
  **789 Mo** de TSM après compaction partielle (5 shards).
- **Environnement** : poste local, pas aws-db. La machine était partagée avec SC-04 (pannes réseau), qui tournait en
  même temps (conteneurs `sc04-influxdb`, `sc04-toxiproxy`). Le poste a atteint 99 % de CPU et 99,6 % de mémoire.

## 1. Environnement (bloc `environment` de `SC-07.after.json`)

Le test tourne en local, sur le poste de travail. Le registre prévoit aws-db, une r7i.2xlarge à 8 vCPU. Ici, le poste a
16 fils d'exécution et n'est pas dédié : SC-04 tournait en parallèle.

| Champ | Valeur |
|---|---|
| host | rocketscience (pas EC2) |
| cpu_model | AMD Ryzen 7 7730U with Radeon Graphics (8 cœurs / 16 fils) |
| ram_gib | 38,8 (Docker Desktop : 18,95 Gio pour les conteneurs) |
| os | Windows-11-10.0.26300-SP0 |
| python / venv | 3.12.10 / `C:/dev/pulsar-ml/validation_e2e/.venv` (les venvs de `pulsar_demo` et de `pulsar_data_collection` n'ont pas les deux paquets) |
| pulsar_data_collection | 0.13.0, commit 39f83c238e (`main`, installation éditable ; la dist-info indique 0.8.0) |
| pulsar_metrics | 0.14.0, commit 48657e80ec (`main`, installation éditable ; la dist-info indique 0.9.1) |
| pulsar_demo | branche `fonction/sc07-tableaux-echelle` |
| influxdb-client / pandas / numpy | 1.50.0 / 2.3.3 / 2.5.4 |
| docker | 28.0.4 |
| InfluxDB | 2.6.1, conteneur `sc07-influxdb`, port hôte 8107, `ulimit nofile` 65536, configuration par défaut |
| Grafana | 11.3.0, conteneur `sc07-grafana`, port hôte 3107, provisionné depuis `observability/grafana` du dossier (source de données vers `sc07-influxdb`, délai 60 s), lecteur anonyme |
| Autres conteneurs présents (non touchés) | `sc04-influxdb`, `sc04-toxiproxy` (SC-04), `pulsar-validation-influxdb`, `pulsar-grafana`, `pulsar-ollama` |

Piège rencontré sur ce poste : une requête vers `http://localhost:8107` met 21 s, à cause de l'essai IPv6 avant IPv4
sous Windows. Toutes les mesures passent donc par `127.0.0.1`. Un premier essai réduit, par `localhost`, a été
abandonné.

## 2. Données écrites

| Flux | Chemin d'écriture | Volume | Durée | Débit | Pertes |
|---|---|---|---|---|---|
| Résultats, 300 modèles × 120 fenêtres de 6 h (30 jours) | `ResultsSink.write_results`, 1 requête par modèle et par fenêtre (comme `analysis_job.py`), 8 processus | 5 066 400 points, 36 000 requêtes, 140,7 points/requête | 160,2 s | 31 628 points/s ; 33,3 ms par requête en moyenne | 0 erreur |
| Compteurs de capture, 300 modèles toutes les heures | `ResultsSink.write_stats_many` (un seul `StatsReporter` par processus) | 216 000 points | 23,7 s | — | 0 |
| Captures de 10 modèles, 150 lignes/h sur 720 h | `Pulse`, 20 variables, `PiiPolicy` sur un commentaire libre | 1 080 000 lignes | 122,7 s | — | 0 ligne écartée |

Les formes des résultats sont celles des vraies analyses, vérifiées une fois sur une fenêtre réelle de `pulsar_metrics`
0.14 (`DriftWatch`, `performance_results`, `fairness_results`, `FairnessMonitor.update`, `explain_results`). On
retrouve les mêmes tags, les mêmes champs et les mêmes chaînes `method`.

| Famille (par modèle et par fenêtre) | Points | Détail |
|---|---|---|
| Dérive | 20 | 16 variables numériques (`ks_2samp`), 4 catégorielles (`chi2`) ; `p_adjusted`, `confirmed` ; un tiers des modèles dérive sur `f00` les 10 derniers jours |
| Performance | 5 | accuracy, precision, recall, f1, label_coverage ; étiquettes absentes les 48 dernières heures |
| Équité, test fixe | 32 | langue (2 groupes), région (3), intersection langue × région (11), 2 critères ; groupes `other` des intersections retenus (< 30) ; un sixième des modèles a une règle biaisée sur `language=other` |
| Équité, test séquentiel | 2 | `sequential_demographic_parity`, `e_value` |
| Explications | 82 | NDCG et Jaccard, `importance_share` et `importance_share_difference` sur 20 variables, différence par groupe (2 groupes × 20) |
| **Total** | **141** | 120 fenêtres × 300 modèles = 36 000 écritures |

Les 30 jours sont écrits en entier : 120 fenêtres et 720 heures de captures se terminent à la dernière fenêtre close
(2026-11-05T12:00Z). La plage de 30 jours lit donc de vraies données, et la plage de 7 jours en est la dernière
semaine.

## 3. Cardinalité et disque (`metrics.cardinality`, `metrics.disk`)

| Mesure | Valeur |
|---|---|
| Séries d'index du bucket (`influxdb.cardinality`) | 42 610 |
| `pulsar_results` : séries d'index (mesure + jeu de tags) | 42 300, soit **141 par modèle** |
| `pulsar_results` : séries par champ (une par champ de chaque jeu de tags, ce que TSM stocke) | **279 000**, soit 930 par modèle |
| `pulsar_capture_stats` | 300 séries |
| Valeurs distinctes des tags de `pulsar_results` | `model_id` 300, `model_version` 1, `metric_type` 4, `metric_name` 15, `feature_name` 21, `group` 16, `reference_group` 3, `test_type` 2 |
| Champs de `pulsar_results` | 13 : `ci_high`, `ci_low`, `confirmed`, `e_value`, `method`, `metric_value`, `n`, `p_adjusted`, `status`, `suppressed`, `threshold`, `threshold_high`, `threshold_low` |
| Mesures du bucket | 12 (`pulsar_results`, `pulsar_capture_stats`, 10 mesures de captures) |
| TSM du bucket, juste après l'écriture | 769 Mo, 20 fichiers, 5 shards ; WAL 61 Mo |
| TSM du bucket, au second essai (environ 27 min plus tard) | **789 Mo**, 25 fichiers ; WAL 28 Ko ; moteur entier 822 Mo |

La cardinalité suit le nombre de résultats distincts par modèle (141), pas le produit des tags. Le modèle de données
reste donc linéaire en nombre de modèles : 141 séries d'index et 930 séries par champ pour chaque modèle ajouté.

## 4. Requêtes directes à InfluxDB (un utilisateur)

Méthode : on extrait chaque requête Flux de chaque panneau des 8 fichiers JSON provisionnés, soit 89 requêtes. Les
variables de gabarit sont d'abord exécutées dans l'ordre, comme Grafana le fait au chargement. On garde le modèle
choisi et la valeur par défaut, ou la valeur de rang `rep` si le défaut n'existe pas. Les macros `v.timeRangeStart` et
`v.timeRangeStop` sont remplacées par des instants absolus. Chaque requête est chronométrée sur `/api/v2/query`,
réponse CSV lue en entier. Il y a 5 répétitions par tableau, chacune sur un modèle différent parmi les 10 qui ont des
captures, aux plages de 7 et 30 jours. Ces chiffres sont ceux de la mesure après correctif. Avant correctif, les temps
sont les mêmes à ±0,04 s, mais 6 requêtes sont en erreur sur le panneau des quantiles.

| Tableau | Requêtes | 7 j : pire p95 (max) | Panneau | 30 j : pire p95 (max) | Panneau | Erreurs avant → après |
|---|---|---|---|---|---|---|
| **overview** | 12 | 0,09 s (0,10) | Latest verdict by kind | 0,06 s (0,06) | Latest verdict by kind | 0 → 0 |
| **drift** | 8 | 0,13 s (0,13) | Hourly quantiles (p90) | 0,46 s (0,46) | Hourly quantiles (p90) | 6 → 0 |
| **fairness** | 9 | 0,05 s (0,05) | Disparate impact ratio of intersectional groups | 0,05 s (0,05) | idem | 0 → 0 |
| performance | 9 | 0,03 s (0,03) | Accuracy (latest labelled window) | 0,05 s (0,05) | Hourly approval rate and mean score | 0 → 0 |
| explain | 6 | 0,11 s (0,12) | Importance by group (vs fr) | 0,10 s (0,10) | idem | 0 → 0 |
| privacy | 11 | 0,04 s (0,04) | Findings masked (period) | 0,04 s (0,04) | Findings masked per hour | 0 → 0 |
| capture | 12 | 0,04 s (0,04) | Rows not yet written | 0,04 s (0,04) | Storage available | 0 → 0 |
| agents | 22 | 0,04 s (0,04) | Errors by tool | 0,03 s (0,03) | Tool error rate and task success | 0 → 0 |

Variables de gabarit (requête directe, 5 chargements par tableau qui les utilise, après correctif) :

| Variable | Valeurs | p50 | max |
|---|---|---|---|
| `model` | **300** | 0,033 s (avant : 0,071 s) | 0,051 s (avant : 0,126 s) |
| `version` | 1 | 0,024 s | 0,030 s |
| `feature` (drift) | 16 (avant : 20, dont 4 catégorielles) | 0,032 s | 0,033 s |
| `attribute`, `group` (fairness) | 3 et 2 | 0,03 s | 0,032 s |
| `campaign` (agents) | 0 (pas de données d'agent) | 0,015 s | 0,066 s |

Verdict, à un utilisateur : tous les panneaux des 8 tableaux restent sous 0,5 s, à 7 comme à 30 jours.

## 5. Grafana, 1 puis 20 lecteurs simultanés (`/api/ds/query`)

Méthode : 20 fils d'exécution démarrent ensemble, chacun jouant un lecteur anonyme (rôle Viewer). Chaque lecteur
charge overview, drift puis fairness, d'abord à 7 jours puis à 30, et recommence 3 fois. À chaque chargement, il
exécute les variables dans l'ordre, puis tous les panneaux avec 6 requêtes en vol, comme un navigateur en HTTP/1.1. Le
modèle est tiré au hasard parmi les 10 qui ont des captures. Cela fait 3480 requêtes de panneaux et 1080 requêtes de
variables en 52 s. Le temps « tableau entier » va du premier envoi à la dernière réponse des panneaux. Il ne compte ni
les variables ni le rendu.

| Tableau | 7 j : pire panneau p95 (max) | Tableau entier p50 / p95 | 30 j : pire panneau p95 (max) | Tableau entier p50 / p95 | Erreurs avant → après |
|---|---|---|---|---|---|
| **overview** | 1,14 s (1,27) | 0,89 / 1,51 s | 1,01 s (1,41) | 0,88 / 1,26 s | 0 → 0 |
| **drift** | **2,44 s (2,93)**, quantiles | 2,09 / 2,57 s | **8,90 s (9,56)**, quantiles | 7,62 / 9,27 s | 69 → 0 |
| **fairness** | 0,72 s (0,84) | 0,66 / 1,07 s | 0,65 s (0,78) | 0,55 / 0,92 s | 0 → 0 |

- Tous panneaux confondus, à 7 jours : p50 0,36 s, p95 1,71 s, max 2,93 s, **0 requête sur 1740 au-dessus de 3 s**. À
  30 jours : p95 7,20 s, max 9,56 s, **189 requêtes sur 1740 au-dessus de 3 s**. Parmi elles, 180 viennent des trois
  requêtes de quantiles. Les 9 autres viennent de deux panneaux de drift, freinés par la saturation : « Confirmed
  drift » (p95 3,13 s) et « Windows with a confirmed drift » (p95 3,22 s).
- Un seul lecteur dans Grafana : pire panneau 0,17 s à 7 jours et 0,70 s à 30 jours (quantiles).
- Variable `model` via Grafana sous 20 lecteurs, à 300 modèles : p50 0,73 à 0,88 s, p95 jusqu'à 1,69 s et max 2,67 s
  avant correctif. Après correctif : p50 0,18 à 0,21 s, p95 0,38 à 0,71 s, max 1,03 s.
- Pic CPU des conteneurs (`docker stats`) : InfluxDB 1416 %, soit environ 14 des 16 fils ; Grafana 607 %. Poste : CPU
  99 %, mémoire 99,6 %, avec SC-04 en parallèle.
- 0 timeout Flux et 0 erreur après correctif. Le délai de la source de données (60 s) n'a jamais été approché.

## 6. Échelle de cardinalité (`metrics.cardinality_ladder`)

Pour trouver la limite attendue, un bucket séparé `sc07_ladder` reçoit N modèles d'une fenêtre chacun : 141 résultats
par modèle, avec les mêmes formes. À chaque palier, on chronomètre la variable `model` (5 fois, version provisionnée
et version d'avant SC-07) et quatre panneaux par modèle à 7 jours (3 fois) : Raw alerts, Latest window, Confirmed
drift (overview) et Latest verdict by kind.

| Modèles | Séries d'index | Écriture | `influxdb.cardinality()` | Variable `model` avant : p50 (max) | Variable `model` après : p50 (max) | 4 panneaux par modèle : pire p50 | Mémoire InfluxDB |
|---|---|---|---|---|---|---|---|
| 1000 | 141 000 | 3701 points/s | 9,1 s | 0,21 s (0,26) | 0,031 s (0,031) | 0,062 s | 909 Mio |
| 3000 | 423 000 | 3154 points/s | 26,4 s | 0,59 s (0,77) | 0,032 s (0,045) | 0,060 s | 1,15 Gio |
| 10 000 | 1 410 000* | 3310 points/s | 112,0 s | **2,50 s (3,37)** | 0,102 s (0,138) | 0,065 s | 1,28 Gio |
| 20 000 | 2 820 000* | 3380 points/s | 199,5 s | **3,83 s (4,10)** | 0,100 s (0,145) | 0,062 s | 1,68 Gio |

\* 141 × N. Au-delà de 60 s, la valeur renvoyée par `influxdb.cardinality()` n'est pas relue : seul son temps est
gardé. Aux paliers mesurés (1000 et 3000), la valeur relue est exactement 141 × N, comme à 300 modèles (42 300).

Variantes de la variable `model` à 20 000 modèles (`bench_modelvar_ladder20k.json`, 5 essais) :

| Requête | Valeurs | p50 | max |
|---|---|---|---|
| Avant SC-07 : `tagValues`, prédicat `_measurement == … and model_id != "permit_assistant"` | 20 000 | 3,67 s | 5,11 s |
| **Après SC-07** : `tagValues`, prédicat `_measurement == …`, puis `filter(r._value != "permit_assistant")` | 20 000 | **0,106 s** | 0,136 s |
| `schema.measurementTagValues` | 20 000 | 0,078 s | 0,092 s |
| `tagValues` sur une série par modèle (`performance` / `label_coverage`) | 20 000 | 0,075 s | 0,079 s |

Lecture : écrire une série nouvelle coûte environ 0,3 ms, soit 3300 points/s, contre 31 600 points/s dans des séries
existantes. Les panneaux par modèle ne dépendent pas du nombre de modèles. Le coût vient seulement de la variable
écrite avec une négation, et de `influxdb.cardinality()`, qui demande environ 70 µs par série.

## 7. Limites trouvées

### L1 (majeure, non corrigée) : quantiles horaires à 30 jours sous charge

- **Constat** : « Hourly quantiles of ${feature} » (drift, 3 requêtes) met 0,46 s seul à 30 jours, mais 8,9 s au p95
  (max 9,6 s) avec 20 lecteurs. Le tableau drift entier passe à 9,3 s au p95. À 7 jours, il reste sous le budget, mais
  avec peu de marge : 2,44 s au p95, 2,93 s au max.
- **Cause** : InfluxDB 2.6 ne pousse vers le stockage que `mean`, `sum`, `count`, `min`, `max`, `first` et `last`
  dans `aggregateWindow`, pas `quantile` ni `median`. Chaque requête fait donc remonter dans le moteur Flux les
  107 550 points bruts de la variable (150 lignes/h × 30 jours), et le panneau en fait trois. Une mesure ponctuelle
  sur la même série à 30 jours, non enregistrée dans un fichier, donne 17 à 35 ms avec `fn: mean` et 0,49 à 1,02 s
  avec `fn: median`. Avec 60 requêtes de ce type en même temps, le CPU sature. La cardinalité n'y est pour rien : le
  coût suit le volume de captures d'un modèle et la plage de temps.
- **Variantes essayées** (`bench_quantiles.json`, 20 requêtes simultanées) : à 30 jours, 3 requêtes t-digest donnent
  p50 7,68 s ; 3 requêtes `exact_mean` donnent 7,67 s ; une seule requête qui lit les captures une fois (`union`)
  donne 7,83 s en t-digest et 7,89 s en `exact_mean`. Aucune ne fait mieux : le coût est la remontée des points bruts.
  Aucun correctif sûr n'existe donc côté `build_dashboards.py`.
- **Proposition** : pré-agréger. `analysis_job.py` écrirait chaque heure p10, médiane et p90 de chaque variable
  numérique dans `pulsar_results`, avec `metric_type = "statistics"`, que l'overview exclut déjà. Une autre voie est
  une tâche InfluxDB qui alimente une mesure sous-échantillonnée. Le panneau lirait alors 720 points par quantile à
  30 jours, comme les autres panneaux de résultats, qui mettent 0,05 à 0,1 s ici. Cela touche le chemin d'écriture et
  reste à faire.

### L2 (majeure, corrigée) : variable `model` linéaire en cardinalité, la limite attendue

- **Constat** : `r.model_id != "permit_assistant"` dans le prédicat de `schema.tagValues` coûte 0,21 s à 1000 modèles,
  2,5 s à 10 000 et 3,8 s à 20 000 (p50 ; max 4,1 s). Le temps croît de façon linéaire, d'environ 1,4 à 1,8 µs par
  série d'index. Le budget de 3 s est atteint vers 14 000 modèles (interpolation entre 10 000 et 20 000). Sous 20 lecteurs, à 300 modèles
  seulement, la variable coûte déjà 0,73 à 0,88 s (p50) à chaque chargement de tableau.
- **Correctif** (`build_dashboards.py`) : `var_tag(..., exclude=("permit_assistant",))`. Le filtre porte sur les
  valeurs, après la recherche dans l'index :
  `schema.tagValues(bucket, tag: "model_id", predicate: (r) => r._measurement == "pulsar_results", start: -30d) |> filter(fn: (r) => r._value != "permit_assistant")`.
  Le résultat est identique (300, puis 20 000 valeurs). Après correctif : 0,10 s à 20 000 modèles, 0,033 s à 300
  modèles, et 0,2 s sous 20 lecteurs. Les 8 tableaux sont régénérés.

### L3 (mineure, corrigée) : variable `feature` du tableau drift et variables catégorielles

- **Constat** : la variable `feature` liste toutes les variables testées, `chi2` compris. Si l'utilisateur choisit une
  variable catégorielle, les 3 requêtes du panneau des quantiles échouent (HTTP 400). C'est la cause des 78 requêtes
  en erreur avant correctif : 6 en direct, 3 avec un lecteur, 69 avec 20 lecteurs.
- **Correctif** : le prédicat ajoute `and r.metric_name != "chi2"`. Le libellé devient « Feature (numeric) ». Dans la
  démo, la liste devient `income`, `age` et `household_size` (avant : avec `region` et `language`). Après correctif,
  il reste 0 erreur.

### L4 (mineure) : `influxdb.cardinality()` inutilisable sur un gros bucket

- 9 s à 141 000 séries, 112 s à 1,41 million, 200 s à 2,82 millions : au-delà de 10 000 modèles, cela dépasse le délai
  de 60 s de la source de données. Aucun tableau ne l'utilise, mais il ne faut pas en ajouter un qui le fasse. Pour un
  suivi régulier, il faut un compteur tenu à l'écriture, ou une métrique du serveur, à vérifier dans `/metrics` de
  la 2.6.

## 8. Verdict par tableau

| Tableau | 7 j, 1 utilisateur | 7 j, 20 lecteurs | 30 j, 1 utilisateur | 30 j, 20 lecteurs |
|---|---|---|---|---|
| overview | OK (0,09 s) | OK (1,14 s) | OK (0,06 s) | OK (1,01 s) |
| drift | OK (0,13 s ; erreurs avant correctif) | OK de justesse (2,44 s, max 2,93 s) | OK (0,46 s) | **Hors budget** (8,90 s, L1) |
| fairness | OK (0,05 s) | OK (0,72 s) | OK (0,05 s) | OK (0,65 s) |
| performance, explain, privacy, capture | OK (≤ 0,11 s) | non mesuré | OK (≤ 0,10 s) | non mesuré |
| agents | sans objet : pas de données d'agent, requêtes à vide | — | — | — |

Les pires p95 sont indiqués entre parenthèses. Le budget de 3 s ne porte que sur la plage de 7 jours (critère du
registre). La colonne 30 jours sous 20 lecteurs est un constat et n'entre pas dans les critères : c'est pourquoi le
JSON de résultat dit `passed` sans `limit`.

## 9. Limites de la mesure

- **Local, et non aws-db.** Le poste a 16 fils, contre 8 vCPU pour une r7i.2xlarge. Il était partagé avec SC-04 et
  saturé pendant la phase à 20 lecteurs. Sur lab-db, avec deux fois moins de fils, le drift à 7 jours sous 20 lecteurs
  a des chances de dépasser 3 s. Ce point n'est pas mesuré.
- **Résultats synthétiques, au vrai format.** Ils passent par le vrai chemin d'écriture (`ResultsSink`), mais leurs
  valeurs ne sont pas calculées par `pulsar_metrics` pour chacune des 36 000 fenêtres. Le coût des analyses est mesuré
  par SC-06.
- **Captures pour 10 modèles sur 300.** Un panneau de captures lit la mesure d'un seul modèle : son coût dépend du
  volume de ce modèle, pas du nombre de modèles. Les lecteurs choisissent parmi ces 10 modèles, pour que chaque
  panneau ait ses données. Les panneaux de résultats voient les 300 modèles dans le bucket.
- **Agents** : aucune donnée `permit_assistant`. Les temps du tableau agents et des tuiles d'agents de l'overview et
  de privacy sont des requêtes à vide, sans portée.
- **Grafana sans navigateur** : le chargement est émulé par `/api/ds/query`, 6 requêtes en vol par lecteur. Ni le
  rendu ni le cache du navigateur ne sont mesurés.
- **Compaction** : la première mesure suit l'écriture de quelques minutes (20 fichiers TSM). La seconde a lieu environ
  27 min plus tard (25 fichiers, WAL vidé). Les temps directs sont les mêmes à ±0,04 s.
- **Le statut de l'essai « before » est `failed`**, sur le seul critère `query_errors` (78). Celui de l'essai « after »
  est `passed`. L'échelle de cardinalité a tourné deux fois. Le premier passage s'est arrêté à 10 000 modèles, sur le
  délai de 60 s de `influxdb.cardinality()`. La boucle enregistre désormais un palier qui échoue au lieu d'interrompre
  le test. Le second passage est celui qui est rapporté.

## 10. Reproduire

```bash
# conteneurs dédiés (jetons générés dans lab/results/SC-07/*.env, jamais affichés)
docker network create sc07-net
docker run -d --name sc07-influxdb --network sc07-net -p 8107:8086 --ulimit nofile=65536:65536 \
  --env-file lab/results/SC-07/influx.env -v sc07-influxdb-data:/var/lib/influxdb2 influxdb:2.6.1
docker run -d --name sc07-grafana --network sc07-net -p 3107:3000 --env-file lab/results/SC-07/grafana.env \
  -v "$PWD/observability/grafana/provisioning:/etc/grafana/provisioning:ro" \
  -v "$PWD/observability/grafana/dashboards:/var/lib/grafana/dashboards:ro" -v sc07-grafana-data:/var/lib/grafana \
  grafana/grafana-oss:11.3.0
export LAB_INFLUX_ENV=lab/results/SC-07/influx.env LAB_INFLUX_URL=http://127.0.0.1:8107 LAB_GRAFANA_URL=http://127.0.0.1:3107
LAB_RUN_TAG=before python -m lab.harness SC-07                     # écriture + mesures (environ 8 min)
mkdir -p lab/results/SC-07.after && cp lab/results/SC-07.before/load.json lab/results/SC-07.after/  # chiffres d’écriture
LAB_RUN_TAG=after LAB_SC07_SKIP_LOAD=1 LAB_SC07_LADDER=1000,3000,10000,20000 python -m lab.harness SC-07
python -m lab.tests._sc07_probes quantiles lab/results/SC-07/bench_quantiles.json
python -m lab.tests._sc07_probes modelvar sc07_ladder lab/results/SC-07/bench_modelvar_ladder20k.json
```

Les conteneurs `sc07-influxdb` et `sc07-grafana`, les volumes `sc07-influxdb-data` et `sc07-grafana-data` et le réseau
`sc07-net` ont été supprimés après la mesure.
