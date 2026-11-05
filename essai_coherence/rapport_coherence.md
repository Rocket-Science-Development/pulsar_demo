# Essai : cohérence des réponses d'un LLM (lot G)

*4 novembre 2026, poste de travail sans GPU. Campagne `coherence-1`.*

## Objet

Le défi fédéral demande de surveiller la « cohérence » des systèmes d'IA, et l'indice de confiance de la couche de
gouvernance s'appuie sur la stabilité d'une réponse demandée trois fois. Cet essai vérifie, sur des modèles locaux,
que la nouvelle mesure de Pulsar.ml (sonde `ConsistencyProbe` de pulsar-data-collection, métriques `consistency` et
`consistency_drift` de pulsar-metrics) distingue des conditions plus ou moins stables, et qu'elle ne donne pas de
fausse alerte quand rien ne change.

## Environnement

- Ollama local (conteneur `pulsar-ollama`), API compatible OpenAI sur `http://127.0.0.1:11434/v1`, processeur
  seulement (16 cœurs partagés avec d'autres travaux).
- Modèles : llama3.2:3b (Meta) et granite3.3:2b (IBM) ; plongements granite-embedding:278m (IBM) pour la vue en
  texte libre. Aucun modèle d'origine chinoise.
- Réglages : 2 appels simultanés au plus, réponse plafonnée à 48 jetons, délai client de 180 s, consigne système
  « répondre le plus brièvement possible ».
- Temps de calcul LLM : 572 s en tout (187 s, 138 s, 109 s et 138 s pour les quatre conditions), soit moins de
  10 minutes pour 450 appels, aucun échec ; médiane de 2 s par appel.

## Questions

30 questions synthétiques à réponse connue (`questions.py`), en français et en anglais : 10 calculs ou nombres
(montant annuel d'une prestation, hausse de loyer, nombre de provinces...), 10 questions par oui ou non (civisme et
géographie du Canada) et 10 questions à étiquette (capitales, provinces, océan, saison). Ce sont des faits publics :
aucune donnée personnelle. Les questions du Registre de l'IA du gouvernement du Canada n'ont pas été reprises : leurs
données ne sont pas présentes sur ce poste et ces questions demandent le registre comme contexte.

## Méthode

- Chaque question est posée plusieurs fois, tour après tour, dans un ordre aléatoire. La réponse finale est lue et
  normalisée (oui / non, nombre, étiquette). Deux réponses s'accordent si leurs réponses normalisées sont égales ; une
  réponse illisible ne s'accorde avec aucune autre.
- Par question : l'accord moyen sur les paires de réponses (3 paires pour 3 réponses) et l'accord complet (toutes
  les réponses identiques).
- Par fenêtre : la moyenne de l'accord sur les questions et la part des questions en accord complet, avec un
  intervalle bootstrap à 95 % sur les questions.
- Baisse de cohérence : test unilatéral par permutation (changement de signe des écarts par question, apparié sur les
  30 questions), seuil 5 %. La valeur est l'écart courant moins référence, avec son intervalle bootstrap.
- Référence : llama3.2:3b à la température 0,7, réponses 1 à 3. Témoin A/A : réponses 4 à 6 de la même série. Le taux
  de fausses alertes est aussi mesuré sur 500 partages aléatoires des 6 réponses de chaque question en deux moitiés
  de 3.

## Résultats

| Condition | Accord moyen | Accord complet | Bonnes réponses | Réponses illisibles |
|---|---|---|---|---|
| llama3.2:3b, T = 0,7, référence | 0,81 [0,69 ; 0,91] | 0,73 [0,57 ; 0,87] | 0,68 [0,52 ; 0,82] | 0 % |
| llama3.2:3b, T = 0,7, témoin A/A | 0,78 [0,64 ; 0,90] | 0,73 [0,57 ; 0,87] | 0,70 [0,54 ; 0,84] | 0 % |
| granite3.3:2b, T = 0,7 | 0,78 [0,64 ; 0,90] | 0,73 [0,57 ; 0,90] | 0,52 [0,36 ; 0,70] | 4 % |
| llama3.2:3b, T = 1,5 | 0,62 [0,48 ; 0,77] | 0,53 [0,37 ; 0,70] | 0,60 [0,46 ; 0,74] | 7 % |
| granite3.3:2b, T = 1,5 | 0,51 [0,36 ; 0,67] | 0,43 [0,27 ; 0,60] | 0,51 [0,36 ; 0,67] | 8 % |

| Comparaison avec la référence | Écart d'accord moyen | p | Alerte | Écart d'accord complet | p | Alerte |
|---|---|---|---|---|---|---|
| témoin A/A | -0,03 [-0,16 ; 0,09] | 0,37 | non | 0,00 [-0,17 ; 0,17] | 0,66 | non |
| granite3.3:2b, T = 0,7 | -0,03 [-0,19 ; 0,10] | 0,39 | non | 0,00 [-0,20 ; 0,17] | 0,63 | non |
| llama3.2:3b, T = 1,5 | -0,19 [-0,34 ; -0,04] | 0,016 | **oui** | -0,20 [-0,40 ; 0,00] | 0,055 | non |
| granite3.3:2b, T = 1,5 | -0,30 [-0,44 ; -0,17] | 0,001 | **oui** | -0,30 [-0,50 ; -0,13] | 0,006 | **oui** |

Sur les 500 partages A/A, le test d'accord moyen a alerté 4,4 % des fois [2,9 % ; 6,6 %] et celui de l'accord complet
2,6 % [1,5 % ; 4,4 %], pour un seuil nominal de 5 % : pas d'excès de fausses alertes.

Vue en texte libre (similarité cosinus des plongements granite-embedding:278m sur les réponses brutes) : 0,943
[0,914 ; 0,969] pour la référence, 0,945 pour le témoin A/A (p = 0,55), 0,927 pour granite à 0,7 (p = 0,21), 0,890
pour llama à 1,5 (écart -0,053 [-0,087 ; -0,019], p = 0,003) et 0,793 pour granite à 1,5 (écart -0,151 [-0,208 ;
-0,096], p < 0,001). Elle classe les conditions dans le même ordre, sur une échelle resserrée près de 1 : un seuil
absolu n'y a pas de sens, seule la comparaison à une référence en a.

Les preuves (`results/coherence-1/evidence.json`, 18 enregistrements) sont de la famille `genai`. Les tests de baisse
ont la direction `lower` sur la valeur p, statut `alert` ou `ok`, note `nominal_threshold` (seuil de 5 % non calibré
sur des tirages A/A) ; les métriques de fenêtre, sans seuil fourni, sont `no_verdict` avec la note `no_threshold`.
Toutes sont fiables (30 questions, 3 réponses chacune).

## Lecture

- La mesure sépare les conditions : la hausse de la température fait baisser l'accord de 0,19 pour llama3.2:3b et de
  0,30 pour granite3.3:2b, et le test le signale ; le témoin A/A ne déclenche rien.
- À la température 0,7, granite3.3:2b est aussi cohérent que llama3.2:3b (0,78 contre 0,81) mais se trompe plus souvent
  (52 % contre 68 % de bonnes réponses) : la cohérence n'est pas l'exactitude. Un modèle peut répéter la même erreur ;
  il faut suivre les deux.
- Les désaccords viennent surtout des calculs (prestation annuelle, hausse de loyer, pourcentage), puis de quelques
  questions de géographie ; les questions par oui ou non sont presque toujours stables.
- La part de questions en accord complet est plus grossière que l'accord moyen : l'écart de -0,20 de llama à 1,5
  n'atteint pas le seuil (p = 0,055). Avec 30 questions et 3 réponses, un écart d'environ 0,2 est à la limite de ce
  qu'on peut voir ; pour des écarts plus faibles, il faut plus de questions.

## Limites

- Une seule série par condition, sur un poste partagé ; les intervalles portent sur les questions, pas sur la
  variabilité d'une série à l'autre.
- Les questions sont courtes et à réponse fermée ; pour le texte libre, la vue par plongements n'a été évaluée que sur
  ces réponses courtes. Le seuil d'accord des textes libres (`agreement_threshold`) reste à ajuster sur des réponses
  connues pour concorder ou non.
- Une réponse qui nomme une ville hors de la liste est illisible et compte comme un désaccord.
- Sous Windows, « localhost » faisait attendre environ 20 s par appel (résolution IPv6 d'abord) : une première
  tentative a été arrêtée et relancée sur 127.0.0.1 ; la bibliothèque utilise maintenant 127.0.0.1 par défaut.

## Reproduire

```bash
python campaign.py --campaign coherence-1 --workers 2
python analyze.py --campaign coherence-1 --splits 500 --embeddings granite-embedding:278m
```
