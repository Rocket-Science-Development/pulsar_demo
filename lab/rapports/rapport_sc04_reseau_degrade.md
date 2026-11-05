# SC-04 : réseau dégradé entre Pulse et InfluxDB (5 novembre 2026)

Dossier `C:/dev/pulsar-ml/wt/sc04/lab` (branche `fonction/sc04-reseau-degrade`). Test : `lab/tests/sc_04.py`.

Sources :
- `lab/results/SC-04.default.json` et `lab/results/SC-04.tuned.json` (non versionnés), avec les cas détaillés
  `lab/results/SC-04.<profil>/cases_<profil>.json` et une trace par cas (`trace_<profil>_<cas>.json`, un point par seconde) ;
- résumé versionné : `lab/rapports/summary.sc04-reseau-degrade.json` (paramètres, environnement, critères, limites et
  toutes les mesures par cas des deux profils).

**Exécution locale, à taille réduite.** Le registre prévoit SC-04 sur `aws-scale` (45 min). Il a tourné ici sur un
portable, avec 10 modèles à 2000 lignes/s, et non avec des centaines de modèles. La machine était partagée avec
**SC-07** (temps des requêtes Flux), qui tournait en même temps sur ses propres conteneurs (`sc07-influxdb`,
`sc07-grafana`). Le CPU de l'hôte a atteint 99 % et la mémoire 93 à 99 % pendant les deux exécutions. Les latences
absolues sont donc plus hautes et plus bruitées que sur les machines du lab. Les comptes, eux, ne dépendent pas de la
charge.

## 0. En bref

- **L'appelant n'est jamais bloqué.** Sur 30 cas, aucune capture n'atteint 200 ms. Le p99 de capture sous défaut va de
  2,7 à 17,1 ms, et le maximum est de 131 ms. Le p99 de référence (sans défaut) vaut déjà 2,3 ms (défaut) et 3,6 ms
  (tuned) sur cette machine chargée : la cible de moins de 2 ms ne peut pas être jugée ici. Le pire p99 (17,1 ms,
  défaut, perte 20 %) suit un p99 de 23,4 ms dans la phase propre du même cas : c'est la charge de l'hôte, pas le réseau.
- **La mémoire reste bornée.** La hausse de RSS la plus forte d'un cas est de 108 Mo (défaut) et de 48 Mo (tuned),
  pour des bornes de 178 et 99 Mo. Sur toute l'exécution : +126 Mo et +55 Mo. Aucune file ne dépasse
  `max_queue_rows` (5000 lignes par Pulse).
- **Les comptes sont exacts partout** : 0 ligne non expliquée par les compteurs, 0 ligne écrite absente d'InfluxDB,
  0 ligne stockée non comptée, 0 doublon, dans les 30 cas.
- **Latence 200 ms et 1 s : réussi** dans les deux profils, avec et sans spool. Rien n'est perdu.
- **Perte et lien instable, avec spool : réussi.** 0 perte, 0 doublon. Tout est rejoué en moins d'une seconde après le
  retour du lien.
- **Perte et lien instable, sans spool : échec du critère « pertes seulement au-delà de la borne ».** Une requête dont
  la connexion est coupée après l'envoi est comptée incertaine, et n'est jamais réécrite, même avec de la place dans
  la file. Pertes comptées : 4,6 à 9,0 % du trafic à 5 % de perte, 27 à 28 % à 20 %, et 31 à 32 % pour le lien
  instable. InfluxDB n'avait stocké que 0 à 10 % de ces lignes incertaines.
- **Débit plafonné à 1 Mbit/s : effondrement des écritures.** Une requête de 5000 lignes (environ 557 o par ligne,
  2,8 Mo) demande environ 22 s à 1 Mbit/s, pour un `write_timeout_s` de 5 s : toutes les requêtes expirent. Pendant
  le défaut, 0 à 12 lignes/s sont écrites, pour 2000 offertes. Sans spool, rien n'est perdu au final : Toxiproxy
  finit de livrer les requêtes expirées, et les 77 000 lignes incertaines sont toutes stockées. Avec spool, tout est
  spoolé puis rejoué.
- **Avec `write_timeout_s=30`**, les écritures passent (46 à 57 lignes/s). Le lien reste plus lent que le trafic, donc
  les files se remplissent et les plus anciens lots sont écartés et comptés : 12 300 / 13 700 lignes sans spool, et
  aussi **10 550 / 1000 lignes avec spool**. Le stockage répond toujours, il n'est donc jamais jugé indisponible, et
  le spool ne prend pas le débordement de la file.
- **Aucun bogue trouvé dans `pulsar_data_collection` 0.13.0.** Les comportements observés sont ceux que la
  documentation décrit. Ce sont des limites de conception, documentées dans le README de dc (section « Capacity and
  degraded networks », branche `doc/plafond-capture-reseau`). Aucune version n'est changée.

## 1. Environnement (blocs `environment`)

Les deux fichiers ont le même bloc, au champ `collected_at` près.

| Champ | Valeur |
|---|---|
| hôte | portable Windows 11 Pro 10.0.26300 (`Windows-11-10.0.26300-SP0`), pas d'instance cloud |
| CPU | AMD Ryzen 7 7730U, 8 cœurs / 16 threads |
| RAM | 38,8 Gio |
| GPU | aucun |
| Python | 3.12.10 (venv `C:/dev/pulsar-ml/pulsar_data_collection/.venv`, `psutil` 7.2.2 ajouté par `PYTHONPATH` depuis une copie privée dans `lab/results/_pydeps` ; le venv de `pulsar_demo` n'existe pas) |
| pulsar_data_collection | 0.13.0, source `C:/dev/pulsar-ml/wt/doc-dc`, commit 39f83c2 (= `main`), non modifié |
| pulsar_metrics | non installé (inutile pour SC-04) |
| influxdb-client / pandas | 1.50.0 / 3.1.0 |
| Docker | 28.0.4 (Docker Desktop) |
| conteneurs du test | `sc04-influxdb` (influxdb:2.6.1, port hôte 8097) et `sc04-toxiproxy` (ghcr.io/shopify/toxiproxy:2.12.0, API sur 8475, proxy sur 8098), réseau `sc04-net` |
| autres conteneurs présents, non touchés | `sc07-influxdb`, `sc07-grafana` (SC-07, en cours), `pulsar-validation-influxdb`, `pulsar-grafana`, `pulsar-ollama` |
| pulsar_demo | `C:/dev/pulsar-ml/wt/sc04`, commit 6dec929, branche `fonction/sc04-reseau-degrade`, modifié (le test lui-même, non encore commité) |
| exécutions | défaut : 13:58:43 UTC, 994 s ; tuned : 14:15:28 UTC, 989 s |
| santé (harness) | CPU de l'hôte max 99,7 / 99,3 % ; mémoire max 93,5 / 99,2 % ; RSS max du processus 230 / 163 Mo ; 39 / 31 threads |

Le jeton d'administration d'InfluxDB a été généré dans `lab/results/sc04/influx.env` (non versionné). Il n'est ni
affiché ni commité.

## 2. Méthode

Chaque cas suit le même plan : 10 s sans défaut, 45 s sous défaut, 10 s sans défaut, puis vidage (`flush`, et attente
d'un spool vide, 240 s au plus). Puis les `seq` sont relus directement dans InfluxDB, sans passer par le proxy.
10 Pulse, lots de 50 lignes, 20 features flottantes, `seq` entier unique par ligne, 2000 lignes/s au total
(120 000 lignes/min). `max_queue_rows=5000` par Pulse, `write_timeout_s=5`, `max_retry_delay_s=60`, sonde `/ping`
toutes les secondes.

Profils, comme SC-03 : **défaut** = un écrivain privé par Pulse (10 connexions) ; **tuned** = un
`SharedWriter(concurrency=4)` pour les 10 Pulse, et `spool_compression="gzip"` avec spool. Chaque défaut est joué avec
`spool_dir` et sans. Un cas de référence sans défaut est joué sans spool. Cela fait 15 cas par profil.

### 2.1 Correspondance des défauts avec Toxiproxy

Toxiproxy est un proxy TCP : il ne perd pas de paquets. Les défauts sont donc traduits en ce que TCP montre à
l'application.

| Défaut du registre | Toxiproxy | Ce que cela représente, et ses limites |
|---|---|---|
| latence 200 ms / 1 s | `latency` sur le sens descendant (réponses), 200 / 1000 ms, sans gigue | +200 ms / +1 s par aller-retour de requête |
| perte 5 % / 20 % | 1) toutes les 250 ms, avec la probabilité p, un toxique `timeout` (timeout=0) dans les deux sens : les données sont retenues, puis libérées à la fin du créneau ; 2) `reset_peer` avec toxicity=p (chaque nouvelle connexion TCP a la probabilité p d'être réinitialisée 1 s après) | TCP retransmet les segments perdus : l'application voit des blocages, pas des octets perdus. Une suite de k créneaux perdus bloque k × 250 ms, sans le doublement du délai de retransmission de TCP : c'est optimiste au-delà de 2 pertes de suite. Les réinitialisations représentent les connexions qui ne survivent pas. Une connexion réinitialisée est remplacée, et la nouvelle refait le tirage : il y a eu 3 à 156 coupures par cas. Ce n'est pas un taux de perte de paquets au sens d'un `tc netem`. |
| plafond 1 Mbit/s | `bandwidth` dans les deux sens | Toxiproxy limite **chaque connexion**, pas le lien. Le débit par connexion est donc 125 Ko/s divisé par le nombre de connexions du profil : 12 Ko/s × 10 (défaut), 31 Ko/s × 4 (tuned), soit environ 1 Mbit/s au total. Une connexion inactive ne cède pas sa part : le plafond effectif est un peu en dessous de 1 Mbit/s. |
| plafond 1 Mbit/s, variante `bw1mbit_t30` | même toxique, `write_timeout_s=30` | atténuation testée : 30 s suffisent pour 5000 lignes à 1 Mbit/s |
| lien instable | proxy désactivé 5 s toutes les 10 s (5 coupures en 45 s) | Les connexions ouvertes sont coupées. Derrière la redirection de port de Docker Desktop, une nouvelle connexion est **acceptée puis fermée**, et non refusée : `dc` la compte donc comme coupée (incertaine), et non comme refusée (sûrement non stockée). C'est aussi ce que font beaucoup de répartiteurs de charge. |

### 2.2 Mesures et critères

- **Blocage de l'appelant** : latence de chaque `capture_data()` par phase (p50, p99, max). Une capture de 200 ms ou
  plus compte comme bloquée. Critère : 0.
- **Mémoire** : RSS échantillonnée toutes les 250 ms, comparée à une borne. La borne vaut
  3 × N × `max_queue_rows` × 184 o (formule de SC-05 : 23 colonnes de 8 o, ×3 pour pandas et ses copies), plus les
  requêtes en vol (`max_batch_rows` 5000 × concurrence × 1384 o × 2 : ligne de protocole et corps de requête), plus
  20 Mo. Bornes : 178 Mo (défaut, 10 écrivains) et 99 Mo (tuned, concurrence 4). Borne de la file seule : 46 Mo.
  Critère supplémentaire : aucune file ne dépasse `max_queue_rows` + un lot.
- **Disjoncteur** : chaque entrée en back-off (`_record_failure`) est comptée avec le type d'erreur. L'état de chaque
  destination (ouvert/fermé) est échantillonné toutes les 250 ms.
- **Comptes** : `captured = written + dropped + uncertain + rejected + spooled + queued_end` (0 non expliqué). Côté
  stockage : `written + replayed ≤ seq distincts stockés ≤ written + replayed + uncertain`, et 0 doublon
  (points − distincts).
- **Pertes seulement au-delà de la borne** : un Pulse ne peut écarter des lignes que si sa file a atteint
  `max_queue_rows` (à deux lots près, à cause de l'échantillonnage). Avec spool : aucune perte en plus des lignes
  écartées et comptées.

## 3. Résultats

Abréviations : p99 = p99 de capture pendant le défaut (avant / pendant / après) ; ΔRSS = hausse de RSS dans le cas ;
file = maximum des lignes en file (tous Pulse) ; vol = maximum des lignes en vol ; écrites/s = lignes écrites par
seconde pendant le défaut ; incert. (st.) = lignes incertaines (dont stockées quand même) ; pert. = lignes absentes
d'InfluxDB ; bo = entrées en back-off.

### 3.1 Profil défaut (10 écrivains privés)

| Cas | p99 ms (av./pend./ap.) | max ms | ΔRSS Mo | file / vol | écrites/s | écartées | incert. (st.) | spoolées = rejouées | pert. | bo (type) |
|---|---|---|---|---|---|---|---|---|---|---|
| référence | 2,33 / 2,25 / 4,23 | 7 | 2,0 | 2050 / 1300 | 2013 | 0 | 0 | — | 0 | 0 |
| lat200 | 5,21 / 4,63 / 4,63 | 30 | 8,4 | 2350 / 2250 | 1977 | 0 | 0 | — | 0 | 0 |
| lat200 spool | 6,39 / 8,11 / 5,66 | 119 | 8,5 | 2400 / 2600 | 1988 | 0 | 0 | 0 | 0 | 0 |
| lat1000 | 8,12 / 4,44 / 3,52 | 16 | 11,4 | 3150 / 4150 | 1934 | 0 | 0 | — | 0 | 0 |
| lat1000 spool | 4,72 / 3,66 / 6,91 | 82 | 11,7 | 3550 / 4200 | 1942 | 0 | 0 | 0 | 0 | 0 |
| loss5 | 5,42 / 3,83 / 3,29 | 6 | 9,8 | 2500 / 2250 | 1727 | 0 | 11 700 (0) | — | **11 700** | 54 (coupure) |
| loss5 spool | 2,99 / 3,94 / 13,02 | 20 | 9,0 | 2550 / 2050 | 1866 | 0 | 0 | 6050 | 0 | 20 (coupure) |
| loss20 | 23,39 / 17,08 / 3,82 | 131 | 10,7 | 3350 / 2600 | 1129 | 0 | 38 350 (1250) | — | **37 100** | 143 (coupure) |
| loss20 spool | 2,27 / 3,69 / 3,06 | 21 | 15,4 | 3050 / 2550 | 826 | 0 | 0 | 55 550 | 0 | 156 (coupure) |
| bw1mbit | 3,21 / 7,66 / 5,99 | 34 | 36,7 | 11 950 / 12 700 | **0** | 0 | 76 950 (76 950) | — | 0 | 70 (timeout) |
| bw1mbit spool | 6,32 / 7,60 / 3,87 | 63 | 104,2 | 11 350 / 12 300 | **12** | 0 | 0 | 79 000 | 0 | 65 (timeout) |
| bw1mbit_t30 | 3,78 / 3,82 / 3,26 | 87 | 108,1 | 50 000 / 50 000 | 46 | **12 300** | 20 650 (20 650) | — | 12 300 | 10 (timeout) |
| bw1mbit_t30 spool | 2,86 / 5,47 / 3,33 | 74 | 89,4 | 50 000 / 34 900 | 47 | **10 550** | 0 | 71 200 | 10 550 | 10 (timeout) |
| flap | 2,68 / 3,39 / 2,30 | 51 | 1,2 | 4100 / 2000 | 1072 | 0 | 41 450 (0) | — | **41 450** | 150 (coupure) |
| flap spool | 2,27 / 5,20 / 3,86 | 71 | 15,4 | 4200 / 3150 | 663 | 0 | 0 | 61 800 | 0 | 150 (coupure) |

Environ 130 000 lignes capturées par cas. Rejeu du spool : moins de 0,3 s après la fin de la phase « après » (le lien
est propre depuis 10 s). Spool max : 22 Mo (bw1mbit, gzip absent dans ce profil). Back-off max : 60 s (bw1mbit),
8 s au plus ailleurs.

### 3.2 Profil tuned (`SharedWriter(concurrency=4)`)

| Cas | p99 ms (av./pend./ap.) | max ms | ΔRSS Mo | file / vol | écrites/s | écartées | incert. (st.) | spoolées = rejouées | pert. | bo (type) |
|---|---|---|---|---|---|---|---|---|---|---|
| référence | 2,08 / 3,58 / 3,29 | 50 | 5,2 | 2000 / 2150 | 2019 | 0 | 0 | — | 0 | 0 |
| lat200 | 2,62 / 4,13 / 6,27 | 10 | 6,6 | 2050 / 2100 | 1992 | 0 | 0 | — | 0 | 0 |
| lat200 spool | 8,53 / 9,37 / 4,08 | 66 | 6,5 | 2000 / 2150 | 2010 | 0 | 0 | 0 | 0 | 0 |
| lat1000 | 3,95 / 4,38 / 5,35 | 52 | 10,5 | 2000 / 4100 | 1964 | 0 | 0 | — | 0 | 0 |
| lat1000 spool | 4,95 / 4,34 / 2,72 | 54 | 10,1 | 2250 / 4550 | 1968 | 0 | 0 | 0 | 0 | 0 |
| loss5 | 2,75 / 2,65 / 2,63 | 5 | 6,9 | 2450 / 2500 | 1882 | 0 | 6050 (0) | — | **6050** | 3 (coupure) |
| loss5 spool | 3,75 / 3,32 / 3,22 | 50 | 10,8 | 3950 / 4150 | 1808 | 0 | 0 | 10 450 | 0 | 3 (coupure) |
| loss20 | 4,03 / 3,64 / 4,62 | 14 | 23,8 | 14 050 / 10 000 | 869 | 0 | 39 650 (4000) | — | **35 650** | 17 (coupure) |
| loss20 spool | 3,68 / 4,06 / 5,92 | 16 | 12,2 | 4850 / 5000 | 538 | 0 | 0 | 67 900 | 0 | 24 (coupure) |
| bw1mbit | 4,77 / 4,13 / 4,99 | 57 | 33,9 | 8050 / 16 550 | **12** | 0 | 77 700 (77 700) | — | 0 | 23 (timeout) |
| bw1mbit spool | 3,86 / 3,89 / 3,20 | 53 | 23,0 | 10 550 / 10 550 | **6** | 0 | 0 | 86 100 | 0 | 11 (timeout) |
| bw1mbit_t30 | 3,61 / 7,76 / 27,66 | 41 | 48,4 | 50 000 / 20 000 | 0* | **13 700** | 8200 (8200) | — | 13 700 | 4 (timeout) |
| bw1mbit_t30 spool | 7,25 / 3,17 / 4,32 | 100 | 32,7 | 50 000 / 17 250 | 57 | **1000** | 0 | 68 500 | 1000 | 4 (timeout) |
| flap | 2,39 / 2,90 / 2,87 | 17 | 10,9 | 4350 / 4450 | 1077 | 0 | 39 900 (100) | — | **39 800** | 15 (coupure) |
| flap spool | 3,23 / 4,52 / 2,10 | 42 | 10,8 | 4900 / 5000 | 809 | 0 | 0 | 55 650 | 0 | 15 (coupure) |

\* Aucune des 4 requêtes de 5000 lignes en vol ne s'est terminée pendant les 45 s du défaut : à environ 31 Ko/s
par connexion, une requête met environ 90 s. Elles se terminent après le retour du lien.

Le `SharedWriter` regroupe les 10 Pulse sur une seule destination : il entre beaucoup moins souvent en back-off que
10 écrivains privés (3 à 24 entrées, contre 20 à 156), avec les mêmes défauts.

## 4. Verdict par défaut et par profil

Critère du registre : pas de blocage de l'appelant ; mémoire bornée ; pertes seulement au-delà de la borne de la
file, et comptées.

| Défaut | Profil | Sans spool | Avec spool |
|---|---|---|---|
| latence 200 ms | défaut / tuned | **réussi** / **réussi** (0 perte) | **réussi** / **réussi** |
| latence 1 s | défaut / tuned | **réussi** / **réussi** | **réussi** / **réussi** |
| perte 5 % | défaut / tuned | **échec** : 11 700 / 6050 lignes perdues alors que la file avait de la place (comptées incertaines) | **réussi** / **réussi** (0 perte, 6050 / 10 450 rejouées) |
| perte 20 % | défaut / tuned | **échec** : 37 100 / 35 650 lignes perdues (comptées) | **réussi** / **réussi** (0 perte) |
| 1 Mbit/s, timeout 5 s | défaut / tuned | **réussi, avec réserve** : 0 perte au final, mais 0 / 12 lignes/s écrites pendant le défaut et 77 000 lignes incertaines (stockées par chance) | **réussi, avec réserve** : 0 perte, mais tout est spoolé et rejoué après coup |
| 1 Mbit/s, timeout 30 s | défaut / tuned | **réussi** : 12 300 / 13 700 lignes écartées au-delà de la borne, comptées | **réussi au sens du critère** : 10 550 / 1000 lignes écartées au-delà de la borne, comptées, malgré le spool |
| lien instable | défaut / tuned | **échec** : 41 450 / 39 800 lignes perdues (comptées) | **réussi** / **réussi** (0 perte) |

Dans les 30 cas : 0 capture bloquée, RSS sous la borne, files sous `max_queue_rows`, comptes exacts, 0 doublon.

Statut du harness : `limit-found` pour les deux profils (5 limites mineures chacun, section 5). Les 9 critères chiffrés
du test sont tenus (`checks` tous vrais), après la correction du critère de spool (section 6).

## 5. Limites trouvées

### Majeures

**R1. Sans spool, une écriture coupée n'est jamais retentée, même avec de la place dans la file.** Les 6 cas perte et
lien instable sans spool perdent 4,6 à 32 % du trafic. Ces lignes sont comptées (`uncertain_rows`), et InfluxDB n'en
avait stocké que 0 à 10 % (1250 / 38 350 et 4000 / 39 650 à 20 % de perte, 100 / 39 900 pour le lien instable, 0
ailleurs). Le harness classe la limite « mineure », car le comportement est documenté (« never written again »). Pour
SC-04, elle est majeure : c'est elle qui fait échouer le critère du registre. Deux causes s'ajoutent :
- l'écriture InfluxDB est idempotente (mêmes mesure, tags et horodatage), et le spool rejoue déjà ces lignes sans
  doublon (0 doublon ici). Le même rejeu en mémoire, dans la borne de la file, ne créerait pas de doublon non plus ;
- un serveur arrêté derrière une redirection de port (Docker Desktop ici, souvent un répartiteur de charge) accepte la
  connexion puis la ferme. `dc` voit alors une coupure (incertaine), et non un refus (sûrement non stocké, remis en
  file depuis 0.12.1). Les 150 entrées en back-off du lien instable sont toutes des `StorageConnectionResetError`.

Proposition (à décider, non appliquée) : sans spool, remettre en file une fois, dans `max_queue_rows`, les lignes
d'une écriture incertaine, et ne compter en `uncertain_rows` que celles qui échouent encore ou qui sortent de la file.
En attendant, la documentation de dc recommande `spool_dir` sur tout lien qui peut couper des connexions.

**R2. Un plafond de débit fait expirer toutes les requêtes quand `max_batch_rows` × octets par ligne dépasse débit ×
`write_timeout_s`.** À 1 Mbit/s et 557 o par ligne, 5000 lignes demandent environ 22 s. Avec un timeout de 5 s, toutes
les requêtes expirent : 70 / 23 entrées en back-off, et 0 à 12 lignes/s écrites au lieu de 2000. La sonde `/ping`
répond, car elle est petite : le disjoncteur se referme à chaque seconde, et la requête suivante expire de nouveau.
Le back-off monte jusqu'à 60 s, mais la sonde le court-circuite.
Proposition : réduire la taille de lot après un timeout (diviser par deux, puis remonter peu à peu), ou documenter la
règle `max_batch_rows × octets/ligne < débit × write_timeout_s`. La documentation de dc la donne désormais.

### Mineures

**R3. Avec spool, un lien lent mais qui répond déborde quand même la file** (`bw1mbit_t30` spool : 10 550 lignes
écartées en défaut, 1000 en tuned, toutes comptées). Le spool ne reçoit que les lignes d'un stockage indisponible. Ici
le stockage répond lentement, il n'est jamais indisponible, et `_bound` écarte les plus anciens lots.
Proposition : avec `spool_dir`, spooler les lots au-delà de `max_queue_rows` au lieu de les écarter (dans la limite de
`spool_max_bytes`).

**R4. Les requêtes en vol sortent de la borne de la file.** Dans `bw1mbit_t30` défaut, il y a 50 000 lignes en file
et 50 000 lignes en vol (10 écrivains × 5000), avec leur protocole ligne. La RSS monte de 108 Mo, contre 46 Mo pour la
borne de la file seule. Elle reste sous la borne complète (178 Mo). Pour la mémoire, il faut donc compter
`max_batch_rows` × concurrence par écrivain, en plus de N × `max_queue_rows`. C'est documenté dans dc.

**R5. Sans spool, un plafond de débit avec timeout de 30 s écarte des lignes au-delà de la borne** (12 300 / 13 700).
C'est le comportement attendu de la borne, et ces lignes sont comptées. Il faut dimensionner `max_queue_rows` pour la
plus longue période dégradée, ou utiliser le spool (avec la limite R3).

## 6. Défauts du test corrigés pendant la campagne

- **Critère de spool.** La première version comptait comme « perte avec spool » (limite critique) toutes les lignes
  absentes, y compris les lignes écartées au-delà de la borne et comptées. Cela a donné à tort, en défaut, une limite
  critique « 10 550 rows lost » sur `bw1mbit_t30` spool. Le critère retient maintenant les pertes **en plus** des
  lignes écartées (`spool_lost_total` = 0 dans les deux profils) et une limite mineure pour l'écart (R3). Les deux
  résultats ont été réévalués sur leurs données enregistrées, sans nouvelle exécution :
  `python -m lab.tests.sc_04 reevaluate lab/results/SC-04.default.json lab/results/SC-04.tuned.json`. Le statut reste
  `limit-found`, et le champ `reevaluated_from` le trace dans chaque fichier.
- **Étiquette de profil des limites** lors d'une réévaluation : elle est maintenant prise dans chaque cas (`profile`),
  et non dans le profil du processus.
- **Plafond de débit par connexion.** Un essai à blanc avec 125 Ko/s par connexion n'a fait expirer aucune
  requête en défaut : avec 10 connexions, le plafond total était d'environ 10 Mbit/s. Le débit par connexion est maintenant divisé par le nombre de connexions du profil
  (section 2.1).

## 7. Ce qui n'a pas été fait

- Pas d'exécution sur `aws-scale` (consigne : local, sans AWS). La taille est réduite (10 modèles, 2000 lignes/s,
  45 s de défaut au lieu de 45 min de test).
- Pas de vraie perte de paquets (`tc netem` demanderait un hôte Linux). La perte est une approximation (section 2.1).
- La cible de p99 < 2 ms n'est pas jugeable sur cet hôte partagé : la référence est déjà à 2,3 / 3,6 ms.
- Une seule répétition par cas : les nombres de coupures dépendent du tirage (graine fixe par cas) et de la charge.

## 8. Reproduire

```bash
# conteneurs dédiés (le jeton est généré dans lab/results/sc04/influx.env, au format DOCKER_INFLUXDB_INIT_*, avec INFLUX_URL)
docker network create sc04-net
docker run -d --name sc04-influxdb --network sc04-net -p 8097:8086 --env-file lab/results/sc04/influx.env influxdb:2.6.1
docker run -d --name sc04-toxiproxy --network sc04-net -p 8475:8474 -p 8098:8098 ghcr.io/shopify/toxiproxy:2.12.0
export LAB_PLATFORM_INFLUX_ENV=lab/results/sc04/influx.env LAB_INFLUX_URL=http://127.0.0.1:8097
LAB_PROFILE=default LAB_RUN_TAG=default python -m lab.harness SC-04   # environ 17 min
LAB_PROFILE=tuned   LAB_RUN_TAG=tuned   python -m lab.harness SC-04   # environ 17 min
docker rm -f -v sc04-influxdb sc04-toxiproxy && docker network rm sc04-net
```

Démontage vérifié après la campagne : les deux conteneurs `sc04-*`, leurs deux volumes anonymes et le réseau
`sc04-net` sont supprimés. Les conteneurs de SC-07 et de la validation n'ont pas été touchés.
