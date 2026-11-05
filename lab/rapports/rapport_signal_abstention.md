# Signal d'abstention et réponses sans appui — rejeu de LR-03 et du Registre de l'IA

Date : 4 novembre 2026
Sources : `lab/results/LR-03/runs.jsonl` (1 800 exécutions, réponses en français) et `essai_donnees_gc/results/registre-gc-1/runs.jsonl` (240 exécutions sur le Registre de l'IA du gouvernement du Canada). Aucun nouvel appel à l'agent.
Script : `python -m lab.replay.signal_abstention --lr03 <chemin de LR-03/runs.jsonl> --nli` (branche `fonction/signal-abstention` de pulsar_demo, 4 min sur un processeur, plus 2,5 min pour le repli NLI).
Résultats : `lab/rapports/signal_abstention.json`. Étiquettes : `lab/replay/abstention_labels.json`.
Bibliothèques : pulsar_data_collection (`genai.abstention`) et pulsar_metrics (`rate_increase`), branches `fonction/signal-abstention`.

## 1. Ce qu'on mesure

Deux indicateurs par exécution, écrits sur l'enregistrement `agent_run` :

- `det_abstention` : la réponse dit que l'information est introuvable, absente des sources, ou que l'agent ne peut pas répondre. Une réponse qui dit cela puis répond quand même compte comme une réponse.
- `det_unsupported_answer` : l'agent a répondu alors que la recherche était vide, que le meilleur score de recherche était sous son seuil, ou que la fidélité était sous son seuil. Les seuils sont le 5ᵉ centile de la référence de chaque tirage, jamais de la fenêtre.

Le taux de réponses sans appui par fenêtre est l'indicateur de réponses inventées. Chaque taux est testé contre la référence par un test exact de Fisher unilatéral (`rate_increase`, alerte seulement si le taux monte). Le protocole et le code sont ceux de `declencheur_derive.py`, avec les mêmes graines : on retrouve les chiffres du déclencheur recommandé (fausses alertes à k = 2 : 0,05 % [0,01 ; 0,28] ; puissance à 40 % de perte : 79 % [74 ; 82]).

## 2. Les tests génériques ne suffisaient pas

On peut déjà tester `det_abstention` avec `manwu` (unilatéral). Deux défauts :

- une référence sans aucune abstention (cas du registre : 0 sur 79) rend le champ `constant_reference`, donc la preuve exportée est marquée non fiable ;
- sur une petite fenêtre, la p-valeur est approximative : 3 sur 8 contre 0 sur 80 donne 1,7·10⁻⁸ avec `manwu`, contre 5,1·10⁻⁴ avec le test exact.

`rate_increase` corrige les deux et place les deux taux dans `details` de la preuve (`pulsar.evidence/1`, schéma inchangé).

## 3. Le détecteur d'abstention

102 réponses tirées au hasard dans chaque groupe des deux campagnes (9 par niveau de perte pour LR-03, 5 à 10 par phase pour le registre), étiquetées à la lecture avant d'écrire les règles. Les règles ont été écrites sur les autres réponses.

| Détecteur | Abstentions (sur 33) | Fausses abstentions | Précision | Rappel |
|---|---|---|---|---|
| Règles | 33 | 1 | 0,97 [0,85 ; 0,99] | 1,00 [0,90 ; 1,00] |
| Règles + repli NLI | 33 | 3 | 0,92 [0,78 ; 0,97] | 1,00 [0,90 ; 1,00] |

L'erreur des règles est une supposition justifiée par le manque d'information (« probablement composée de couleurs neutres, car le passage n'a pas fourni d'informations »). Le repli NLI n'ajoute aucune abstention et deux fausses : il reste désactivé par défaut. L'échantillon ne contient que 2 abstentions du registre : la précision sur le registre seul n'est pas mesurable ici.

## 4. Fréquence des deux signaux

| Situation | Abstentions | Réponses sans appui | Questions touchées : réponses non signalées |
|---|---|---|---|
| LR-03, 0 % de perte (700) | 29,7 % [26,5 ; 33,2] | 7,9 % [6,1 ; 10,1] | — |
| LR-03, 40 % (200) | 38,0 % [31,6 ; 44,9] | 14,0 % [9,9 ; 19,5] | 25 sur 80 |
| LR-03, 80 % (200) | 56,5 % [49,6 ; 63,2] | 27,0 % [21,3 ; 33,5] | 12 sur 161 |
| Registre, référence (79) | 0 % [0 ; 4,6] | 17,7 % [10,9 ; 27,6] | — |
| Registre, perte de corpus (40) | 7,5 % [2,6 ; 19,9] | 12,5 % [5,5 ; 26,1] | 14 sur 18 |

Sur le registre, le détecteur retrouve les 3 réponses « absent du registre » sur les 18 questions touchées, comme le rapport de l'essai. Mais 14 des 15 réponses inventées ne sont pas signalées « sans appui » : la recherche BM25 renvoie toujours 3 fiches d'autres systèmes, avec un score et une fidélité dans la plage de la référence. Le critère « recherche sans résultat pertinent » ne voit pas qu'il s'agit du mauvais système.

## 5. Fausses alertes et délai

Alpha 0,05, persistance k = 2. LR-03 : fenêtres de 100 contre une référence de 300. Registre : fenêtres de 8 contre une référence de 80 ; ses tirages A/A réutilisent les mêmes 120 exécutions, leur intervalle est indicatif.

| Configuration | Fausses alertes par fenêtre, k = 2 | Puissance par fenêtre | Délai médian, k = 2 |
|---|---|---|---|
| **LR-03, 20 % de perte** | | | |
| déclencheur recommandé | 0,05 % [0,01 ; 0,28] | 21 % [18 ; 26] | 11 |
| abstention seule | 0,40 % [0,20 ; 0,79] | 7 % [5 ; 10] | non détecté |
| réponses sans appui seules | 0,55 % [0,31 ; 0,98] | 8 % [6 ; 11] | non détecté |
| recommandé + sans appui (× 4) | 0,05 % [0,01 ; 0,28] | 18 % [14 ; 22] | au-delà de 24 (47 % détectés en 24) |
| recommandé + les deux (× 5) | 0,15 % [0,05 ; 0,44] | 16 % [12 ; 19] | au-delà de 24 (41 % détectés en 24) |
| **LR-03, 40 % de perte** | | | |
| déclencheur recommandé | 0,05 % [0,01 ; 0,28] | 79 % [74 ; 82] | 2 (p90 : 5) |
| abstention seule | 0,40 % [0,20 ; 0,79] | 37 % [32 ; 42] | 7 |
| réponses sans appui seules | 0,55 % [0,31 ; 0,98] | 43 % [38 ; 48] | 6 |
| les deux (× 2) | 0,35 % [0,17 ; 0,72] | 48 % [43 ; 52] | 5 |
| recommandé + sans appui (× 4) | 0,05 % [0,01 ; 0,28] | 77 % [73 ; 81] | 2 (p90 : 5) |
| recommandé + les deux (× 5) | 0,15 % [0,05 ; 0,44] | 75 % [70 ; 79] | 2 (p90 : 6) |
| **Registre, perte de corpus, fenêtres de 8** | | | |
| déclencheur recommandé | 0,20 % [0,08 ; 0,51] | 37 % [32 ; 42] | 7 |
| abstention seule | 0 % [0 ; 0,19] | 8 % [6 ; 11] | non détecté |
| réponses sans appui seules | 0,15 % [0,05 ; 0,44] | 0,8 % [0,3 ; 2,2] | non détecté |
| recommandé + les deux (× 5) | 0,10 % [0,03 ; 0,36] | 31 % [27 ; 36] | 8 |
| contrôle du nom cité, seul (hypothèse) | 0,20 % [0,08 ; 0,51] | 40 % [35 ; 45] | 6 |
| recommandé + contrôle du nom (hypothèse) | 0,10 % [0,03 ; 0,36] | 42 % [37 ; 46] | 6 |

À partir de 60 % de perte (LR-03), tous les signaux détectent en 2 fenêtres, sauf les réponses sans appui seules à 60 % (puissance 64 %, délai 3).

Sur la campagne du registre telle qu'elle a tourné (5 fenêtres de 8 par phase), aucun signal, ancien ou nouveau, ne confirme la perte de corpus. Le témoin reste sans alerte. Le changement de modèle est confirmé à la 2ᵉ fenêtre par l'abstention (granite3.3:2b n'appelle pas l'outil et répond « not in the register » 22 fois sur 40), comme par la recherche vide. Avec des fenêtres de 20, l'abstention alerte à la 2ᵉ fenêtre sans confirmer ; la fidélité confirme à la 2ᵉ.

« Contrôle du nom cité » : l'application vérifie que le système nommé dans la question figure parmi les fiches renvoyées (`retrieval_relevant=False` sinon). Il est simulé avec `target_retrieved`, qui compare des identifiants : les noms du jeu de questions sont uniques, l'application peut faire la même vérification sur le nom, sans réponse connue. Il signale les 15 réponses inventées sur 15, mais aussi 20 % des réponses de la référence.

## 6. Conclusions

- **Écart du registre (fenêtres de 8) : non comblé.** Les deux signaux ne voient pas la perte de corpus : l'agent invente au lieu de s'abstenir (3 abstentions sur 18), et ses réponses inventées ont l'air appuyées. Même avec le contrôle du nom cité, la puissance par fenêtre de 8 reste à 42 % et le délai médian à 6 fenêtres (48 exécutions). La cause est la taille : 4 questions touchées par fenêtre en moyenne. Il faut des fenêtres de 20 ou plus.
- **Écart de LR-03 (fidélité tardive) : déjà comblé par le déclencheur recommandé, pas amélioré.** Seuls, les nouveaux signaux font à peu près comme la fidélité (37 à 48 % de puissance à 40 % de perte, contre 39 % pour la fidélité). Ajoutés au déclencheur, ils coûtent un peu de puissance (correction de Bonferroni sur 4 ou 5 signaux) sans réduire le délai.
- **Recommandation pour le déclencheur : inchangé.** Fidélité, part de recherches vides et score du premier passage, Bonferroni × 3, `DriftPersistence(windows=2)`.
- **Usage recommandé des deux signaux : indicateurs de gouvernance, hors du déclencheur.** Ils mesurent ce que l'utilisateur subit (taux de réponses inventées), et une preuve `rate_increase` est lisible par la couche de gouvernance. L'abstention détecte bien un changement de modèle qui ne cherche plus. Pour un agent qui répond sur des entités nommées, ajouter le contrôle du nom cité à `det_unsupported_answer`.

## 7. Limites

- Un seul juge humain pour les étiquettes, 102 réponses, deux modèles (llama3.2:3b, granite3.3:2b). Les règles sont écrites pour leurs formulations ; un autre modèle ou une autre langue demande une nouvelle vérification.
- LR-03 a un taux d'abstention de base élevé (30 %) : l'agent s'abstient déjà souvent avant toute perte, ce qui réduit la marge de hausse.
- Le registre est petit (240 exécutions) : ses tirages A/A ne sont pas indépendants, et la puissance sur la perte de corpus repose sur 40 exécutions.
- Le contrôle du nom cité est une hypothèse simulée, pas une mesure sur une application réelle.
