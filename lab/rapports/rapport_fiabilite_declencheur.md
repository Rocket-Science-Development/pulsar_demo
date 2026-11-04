# Fiabilité du déclencheur « dérive mesurée » — rejeu de LR-03

Date : 4 novembre 2026
Source : `lab/results/LR-03/runs.jsonl` (1 800 exécutions de l'agent RAG sur le corpus PIAF, aucun nouvel appel à l'agent).
Script : `python -m lab.replay.declencheur_derive` (branche `fonction/fiabilite-declencheur` de pulsar_demo).
Résultats : `lab/rapports/fiabilite_declencheur.json` (résumés, versionné) et `lab/results/fiabilite_declencheur_pvalues.json` (p-valeurs brutes de chaque fenêtre, 3,8 Mo, non versionné : `--from-pvalues` le relit, le script le régénère).
Bibliothèque : pulsar_metrics, branche `fonction/fiabilite-declencheur` (`mmd_test(groups=)`).

## 1. Ce qu'on mesure

La couche de gouvernance réévalue un contrôle quand Pulsar signale une dérive. Une fausse alerte bascule un contrôle et dérange un relecteur. On mesure donc deux chiffres par signal :

- le taux de fausses alertes A/A, par fenêtre de 100 exécutions, et par semaine si l'on analyse une fenêtre par heure (168 fenêtres) ;
- le délai de détection, en fenêtres, à chaque niveau de perte du corpus (10 à 80 % des passages retirés, panne totale de l'index).

Une alerte n'est confirmée qu'après k fenêtres de suite en dérive (`DriftPersistence`, k = 1, 2 ou 3).

## 2. Protocole

- Référence : 300 exécutions tirées des 700 exécutions à 0 %. Fenêtre : 100 exécutions.
- A/A : 2 000 tirages. Chaque tirage prend une référence et 3 fenêtres disjointes parmi les 400 autres exécutions. C'est la situation d'un déploiement : une référence fixe, des fenêtres successives. Avec k, l'alerte part si les k premières fenêtres rejettent. Les tirages sont indépendants, l'intervalle de Wilson (95 %) est donc exact.
- Dégradation : 400 tirages par niveau (une référence, une fenêtre tirée dans les 200 exécutions du niveau). Le délai est simulé avec `DriftPersistence` sur des suites de verdicts tirés de ces fenêtres, supposées indépendantes.
- Par semaine : 168 × le taux par fenêtre. C'est une borne haute du nombre de nouvelles alertes (une longue série compte plusieurs fois).
- Les embeddings des 761 passages viennent de l'Ollama local (nomic-embed-text, granite-embedding:278m, CPU). Ils reproduisent ceux du labo : le cosinus question / passage recalculé égale celui de `runs.jsonl` à 10⁻⁴ près.

Signaux, tous unilatéraux dans le sens de la dégradation :

| Signal | Test |
|---|---|
| fidélité | Mann-Whitney, fenêtre plus basse (comme LR-03) |
| recherche vide | Fisher exact, fenêtre plus haute |
| pertinence (score) | Mann-Whitney sur le score BM25 du premier passage, plus bas |
| pertinence (cosinus) | Mann-Whitney sur le cosinus question / premier passage (granite-embedding), plus bas |
| MMD lignes | `mmd_test` sur tous les passages, permutés comme des lignes indépendantes (LR-03) |
| MMD groupes | mêmes passages, exécutions permutées en bloc (`groups=`) |
| MMD top 1 | une ligne par exécution (son premier passage) |
| déclencheur « score » | fidélité, recherche vide ou pertinence (score) : plus petite p-valeur × 3 (Bonferroni) |
| déclencheur « cosinus » | même chose avec la pertinence (cosinus) |

## 3. Diagnostic des trois constats de LR-03

**MMD à 42–44 % de fausses alertes : diagnostic confirmé.** Les 2 passages d'une exécution sont corrélés (même requête, souvent le même article). En les permutant comme des lignes indépendantes, la loi nulle est trop étroite. Rejeu : 40,8 % [38,6 ; 42,9] (nomic) et 43,0 % [40,9 ; 45,2] (granite). En permutant les exécutions entières : 5,9 % [5,0 ; 7,0] et 6,0 % [5,0 ; 7,1]. Un second jeu de 3 000 tirages indépendants donne 5,0 % [4,3 ; 5,8] avec `groups=` et 5,1 % [4,4 ; 6,0] en une ligne par exécution. Le niveau est tenu. Correctif : `mmd_test(..., groups=(groupes_reference, groupes_courant))`.

**Fidélité à 7 % et MMD top 1 à 6–12 % : diagnostic non confirmé.** Ces chiffres venaient de 100 à 200 tirages. Avec 2 000 tirages, la fidélité est à 4,5 % [3,7 ; 5,5] et la pertinence à 5,0–5,4 %. Le MMD top 1 est à 6,1–6,3 % [5,1 ; 7,4], puis à 5,1 % [4,4 ; 6,0] sur 3 000 autres tirages. Quand l'unité est l'exécution et que les exécutions sont échangeables, ces tests sont exacts : la p-valeur nominale suffit. Une calibration A/A n'apporte rien ici. Elle reste utile si le trafic réel a une dépendance temporelle (sessions, rafales de questions proches), ce que LR-03 ne contient pas (questions mélangées). `AACalibrator` sait déjà le faire pour un score par exécution : `DriftTestMetric("manwu", "faithfulness")` sur une table d'exécutions triée par le temps, `contiguous=True`. Sur les 700 exécutions de LR-03, il donne un alpha calibré de 0,046, proche du nominal, mais marqué non fiable (7 fenêtres ; il en faut au moins 10, 30 pour des seuils stables). Il ne calibre pas le MMD sur embeddings ; ce n'est pas nécessaire avec `groups=`. Aucun changement de code.

**La fidélité réagit tard : confirmé.** Puissance par fenêtre à 40 % de perte : 39 % [35 ; 44]. Le score de pertinence atteint 71 % [67 ; 75] et la recherche vide 81 % [77 ; 85]. pulsar-data-collection transporte déjà un tel score : `record_score(span, "retrieval_top", score)` écrit `pulsar.detector.retrieval_top` sur le span de l'outil, et l'enregistrement `agent_run` reçoit `det_retrieval_top` (maximum sur l'exécution). Aucun champ à ajouter.

## 4. Avant / après

Configuration actuelle de LR-03 : un signal seul, alpha 0,05, alerte dès la première fenêtre (k = 1).

| Configuration | Fausses alertes par fenêtre | Par semaine (168 fenêtres) | Puissance à 40 % | Délai médian à 40 % | Délai à 60 % et plus |
|---|---|---|---|---|---|
| MMD lignes, k = 1 (avant) | 40,8 % [38,6 ; 42,9] | 68 [65 ; 72] | 46 % | 2 | 1 |
| Fidélité seule, k = 1 (avant) | 4,5 % [3,7 ; 5,5] | 7,6 [6,2 ; 9,2] | 39 % | 2 | 1 |
| MMD groupes, k = 1 | 5,9 % [5,0 ; 7,0] | 9,9 [8,3 ; 11,8] | 3 % | 10 | 3 (60 %), 1 (80 %) |
| Déclencheur « score », k = 1 | 3,6 % [2,9 ; 4,5] | 6,1 [4,8 ; 7,6] | 79 % | 1 | 1 |
| **Déclencheur « score », k = 2** | **0,05 % [0,01 ; 0,28]** | **0,08 [0,02 ; 0,47]** | 79 % | **2** (p90 : 5) | **2** |
| Déclencheur « score », k = 3 | 0 % [0 ; 0,19] | 0 [0 ; 0,32] | 79 % | 4 (p90 : 9) | 3 |
| Déclencheur « cosinus », k = 2 | 0,40 % [0,20 ; 0,79] | 0,67 [0,34 ; 1,33] | 68 % | 3 (p90 : 7) | 2 |
| Déclencheur « score », k = 2, alpha 0,01 | 0 % [0 ; 0,19] | 0 [0 ; 0,32] | 46 % | 5 (p90 : 13) | 2 |

Le délai compte les fenêtres complètes après le changement ; avec une fenêtre par heure, ce sont des heures. Avec le déclencheur « score » et k = 2, 96 % des pertes de 40 % sont signalées en 6 fenêtres au plus.

Détection par niveau, déclencheur « score », alpha 0,05 :

| Perte du corpus | Puissance par fenêtre | k = 2 : délai médian | k = 2 : détecté en 6 fenêtres |
|---|---|---|---|
| 10 % | 0,5 % [0,1 ; 1,8] | non détecté | 0 % |
| 20 % | 21 % [18 ; 26] | 11 | 18 % |
| 40 % | 79 % [74 ; 82] | 2 | 96 % |
| 60 % | 100 % [99 ; 100] | 2 | 100 % |
| 80 % | 100 % [99 ; 100] | 2 | 100 % |
| panne totale | 100 % [99 ; 100] | 2 | 100 % |

Le MMD, même corrigé, ne voit pas la perte partielle : 2 à 24 % de puissance jusqu'à 60 %. Les passages de remplacement viennent du même corpus, souvent du même article. Il reste utile pour un changement de corpus (nouveaux documents), pas pour des documents manquants.

## 5. Configuration recommandée

- Signaux : fidélité, part de recherches vides, score de pertinence du premier passage (`det_retrieval_top`, écrit par l'application avec `record_score`). Une alerte si la plus petite des trois p-valeurs, multipliée par 3, passe sous 0,05.
- Fenêtre : 100 exécutions contre une référence de 300 exécutions au moins.
- Persistance : k = 2 (`DriftPersistence(windows=2)`).
- Calibration : p-valeurs nominales, valides quand l'unité est l'exécution. Si le trafic réel montre une dépendance temporelle, calibrer avec `AACalibrator(window_size=100, contiguous=True)` sur une référence d'au moins 1 000 exécutions triées par le temps (3 000 pour des seuils stables).
- MMD sur passages : hors du déclencheur. Si on le garde pour un changement de corpus, passer `groups=` (identifiant d'exécution) ou une ligne par exécution.
- Résultat attendu : environ 0,08 fausse alerte par semaine, au plus 0,47 (intervalle à 95 %) ; une perte de 40 % du corpus signalée en 2 heures en médiane, 5 heures au 90ᵉ centile.

Si l'application n'expose pas de score de recherche, le cosinus question / passage le remplace : 0,67 fausse alerte par semaine [0,34 ; 1,33], 3 heures de délai médian à 40 %.

## 6. Limites de cette mesure

- Les exécutions de LR-03 sont échangeables (questions tirées au hasard, posées une fois). Le trafic réel peut être plus corrélé dans le temps ; le taux de fausses alertes serait alors plus haut. À mesurer sur une période de référence réelle avant la mise en service.
- Le score BM25 dépend aussi des statistiques de l'index : il bouge quand on retire ou qu'on ajoute des documents. Pour la gouvernance, c'est un changement réel, mais pas toujours une dégradation.
- Le délai suppose des fenêtres indépendantes. Les réservoirs par niveau (200 exécutions) sont trop petits pour des suites de fenêtres disjointes.
- Le taux par semaine suppose 100 exécutions par heure. Avec moins de trafic, la fenêtre dure plus longtemps et le délai en heures s'allonge d'autant.
- Un seul corpus, un seul agent (llama3.2:3b), un seul type de dégradation (retrait de passages).
