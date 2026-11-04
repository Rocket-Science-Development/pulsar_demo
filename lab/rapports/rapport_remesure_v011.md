# Remesure du chemin d'écriture Pulsar.ml en 0.11 : rapport comparatif (4 novembre 2026)

Dossier `C:/dev/pulsar-ml/wt/lot-d/lab` (branche `fonction/environnement-exact`). Plan : `lab/rerun_v011.md`.

Sources :
- `lab/results/night_full_results.remesure-v011.json` : fusion non tronquée de `run_platform.remesure-v011.json` (clé `run:platform`) et `verify_platform.remesure-v011.json` (clé `verify:platform`) ;
- `lab/results/<id>.v011-default.json` et `<id>.v011-tuned.json` (0.11) ;
- `lab/results/<id>.default.json` et `<id>.tuned.json` (0.10, relance du 3 novembre) ;
- rapport de référence : `lab/results/rapport_relance_scale.md` ;
- statuts : `lab/results/summary.remesure-v011.json`.

Quatre colonnes partout, dans cet ordre :
- **0.10 défaut** et **0.10 tuned** : `pulsar_data_collection` 0.10.0, `pulsar_metrics` 0.11.0, pandas 3.1.0 (nuit du 2 au 3 novembre) ;
- **0.11 défaut** (`v011-default`) et **0.11 tuned** (`v011-tuned`) : `pulsar_data_collection` 0.11.0, `pulsar_metrics` 0.12.0, pandas 3.1.1.

Les profils sont les mêmes qu'en 0.10 : défaut = un écrivain privé par Pulse ; tuned = un `SharedWriter(concurrency=4)` partagé, `spool_compression="gzip"` (SC-03) et `durability="wal"` (SC-13). Les tailles de test sont les mêmes, sauf SC-03 (pannes de 60 s et 10 min seulement, sans la panne de 30 min).

## 0. En bref

- **Le gain de débit du `SharedWriter` se confirme sur les machines du lab.** À 100 modèles, il écrit 33,3k lignes/s à 2M lignes/min (0.10 : 13,3k) et 34,3k lignes/s dans SC-01 (0.10 : 13,7k). Le portable annonçait 24,8k lignes/s. La limite N1 de la relance est en grande partie corrigée : à 100 modèles, le profil tuned reste propre jusqu'à 2M lignes/min (0.10 : saturation dès 1M).
- **Le nouvel encodeur ne change pas le débit d'un Pulse seul.** Profil défaut, 1 modèle à 2M lignes/min : 24,9k lignes/s (0.10 : 25,0k), et 1 430 750 lignes écartées (0.10 : 1 409 550). La saturation reste à 2M lignes/min. En tuned, 1 modèle sature toujours à 4M lignes/min (24,1k lignes/s, 0.10 : 23,6k).
- **Le comptage des lignes incertaines tient.** Pour la première fois, des valeurs non nulles sont observées : 10 000 et 25 000 lignes incertaines dans SC-01 tuned (N = 300 et 500), 3350 dans SC-03 (panne de 60 s sans spool). Les comptes sont équilibrés à tous les niveaux : 0 perte non expliquée.
- **12 exécutions prévues : 7 exécutées, 5 sautées par le plan.** Statuts : 4 `limit-found` (SC-01 et SC-02 dans les deux profils), 4 `passed` (SC-03 tuned, SC-13 tuned, SC-05 défaut, SC-14 défaut).
- **Vérification : 15 verdicts, 15 limites confirmées, 0 réfutée.** La vérification juge 8 des 15 propositions inadaptées. Aucune limite critique.
- **Ce qui n'a pas changé** : le plafond côté appelant d'environ 2100 captures/s par processus en tuned (0.10 : environ 2200 à 2300) et le plafond d'environ 25k lignes/s d'un écrivain privé.
- **Nouvelle observation** : en tuned, sous les timeouts d'InfluxDB, le `SharedWriter` écarte les lots pris pendant le back-off, alors que la file en mémoire a encore de la place (10 000 lignes à N = 300, 5000 à N = 500).
- **Coût** : environ **6,1 USD** à 22:28 UTC, pour un plafond de **12 USD** ; environ 8,4 USD si les machines tournent jusqu'à l'arrêt automatique de 23:30 UTC. Démontage vérifié après ce rapport : les 2 instances (`i-07d601a69020c202e`, `i-09f5073c10678a227`) sont `terminated` et aucun volume `Project=pulsar-lab` ne reste ; l’instance `i-008454b3ee3fa822f` n’a pas été touchée.

## 1. Environnement (blocs `environment`)

Les blocs `environment` des 11 fichiers 0.11 (y compris le smoke, l'essai `bucket-collision` et l'essai `no-influx-env`) sont identiques, sauf le champ `collected_at`. Ils décrivent l'hôte lab-scale.

| Champ | Valeur (0.11, tous les fichiers) |
|---|---|
| host | ip-172-31-18-175 |
| cpu_model | Intel(R) Xeon(R) Platinum 8488C |
| logical_cpus / physical_cpus | 32 / 16 |
| ram_gib | 61.8 |
| gpus | [] |
| cloud.instanceType | c7i.8xlarge |
| cloud.region / availabilityZone | ca-central-1 / ca-central-1a |
| cloud.imageId / architecture | ami-0dccf39617797ec84 / x86_64 |
| os | Linux-7.0.0-1014-aws-x86_64-with-glibc2.39 |
| python / python_executable | 3.12.3 / /data/lab/venv/bin/python |
| pulsar_data_collection | 0.11.0 |
| pulsar_metrics | 0.12.0 |
| influxdb-client | 1.50.0 |
| pandas | 3.1.1 |
| numpy / scipy / scikit-learn | 2.5.4 / 1.18.1 / 1.9.1 |
| pydantic / psutil | 2.14.2 / 7.2.2 |
| opentelemetry-sdk, torch, transformers, gliner, sentence-transformers | null (non installés) |
| docker | 29.1.3 |
| containers | pulsar-lab-influx2 (influxdb:2.6.1) |
| services.influxdb | http://172.31.16.60:8087/health, reachable, v2.6.1 |
| services.grafana | http://localhost:3000/api/health, non joignable (URLError) |
| services.ollama | http://localhost:11434/api/tags, non joignable (URLError) |
| sources.pulsar_data_collection | /data/lab/pulsar_data_collection, commit e67491a5330f1d77ded4a81616c058cd0ffabb2a, branche HEAD, tag 0.11.0, dirty false |
| sources.pulsar_metrics | /data/lab/pulsar_metrics, commit cd18c94fca26061800eb1213516953837fe08c7a, branche HEAD, tag 0.12.0, dirty false |
| sources.pulsar_demo | /data/lab/pulsar_demo, commit fb07ef0b8c9d33497b2b9eae627e3a733550447c, branche fonction/environnement-exact, tag null, dirty false |

Hôte de stockage (d'après `results/provision.json`, pas d'après un bloc `environment`) : lab-db, r7i.2xlarge `i-09f5073c10678a227`, IP privée 172.31.16.60, InfluxDB 2.6.1 dans le conteneur `pulsar-lab-influx2` sur le port 8087, `ulimit nofile` 65536. lab-scale : `i-07d601a69020c202e`.

Pour 0.10, les fichiers n'ont pas de bloc `environment`. Ils indiquent l'hôte ip-172-31-16-44, Python 3.12.3, et dans `params.write` : pulsar-data-collection 0.10.0, pulsar-metrics 0.11.0, influxdb-client 1.50.0, pandas 3.1.0. Les types d'instance sont les mêmes (c7i.8xlarge et r7i.2xlarge), mais ce sont d'autres instances.

## 2. Statut par test

| Test | Profil | Statut 0.10 | Statut 0.11 | Durée 0.11 | Début 0.11 (UTC) | Chiffres clés 0.11 |
|---|---|---|---|---|---|---|
| SC-01 Nombreux modèles par processus | défaut | limit-found | limit-found | 399 s | 22:11 | p99 15,1 / 22,1 / 29,5 ms à N = 100 / 300 / 500 ; débit 78 / 21 / 10 % ; 534 threads, 500 TCP ; 0 perte |
| | tuned | limit-found | limit-found | 640 s | 20:02 | p99 ≤ 1,24 ms partout ; débit 99 / 71 / 41 % ; 34,3 à 37,7k lignes/s écrites ; 10 000 / 5000 lignes écartées (comptées) à N = 300 / 500 |
| SC-02 Ingestion soutenue | défaut | limit-found | limit-found | 2178 s | 20:21 | 1 modèle : saturation à 2M (24,9k lignes/s) ; 100 modèles : plafond appelant à 2,27M lignes/min, 0 drop |
| | tuned | limit-found | limit-found | 2329 s | 20:57 | 1 modèle : propre à 2M, saturation à 4M ; 100 modèles : propre à 2M, file 9,1M et retard 159 s à 4M, 0 drop |
| SC-03 Panne du stockage | défaut | limit-found | sauté (plan) | — | — | chiffres du 3 novembre conservés |
| | tuned | passed | **passed** | 2027 s | 21:36 | spool : 0 perte, 0 doublon ; p99 pendant la panne ≤ 1,40 ms ; 143 o/ligne |
| SC-13 kill -9 et redémarrage | défaut | limit-found | sauté (plan) | — | — | — |
| | tuned | passed | **passed** | 301 s | 20:16 | WAL : 3 512 500 lignes, 0 perte, 0 doublon, 0 `.tmp` |
| SC-05 Saturation de la file | défaut | passed (non valide) | **passed (valide)** | 165 s | 20:13 | écritures mesurées ; ΔRSS 195,5 Mo pour une borne de 523,5 Mo ; lot surdimensionné : 4000 écartées sur 5000, écriture max 1000 |
| | tuned | — | sauté (plan) | — | — | — |
| SC-14 Évolution de schéma | défaut | passed | **passed** | 55 s | 22:10 | 7 cas exacts ; 504 rejets ; 0 ligne valide perdue |
| | tuned | — | sauté (plan) | — | — | — |

Essais non retenus : `SC-01.v011-smoke` (N = 1 et 10, 19:55, passed) ; `SC-01.v011-default.bucket-collision` (19:56, écarté pour défaut de test, section 5) ; `SC-14.v011-default.no-influx-env` (20:16, erreur d'hôte, section 5).

## 3. Comparaison détaillée, test par test

### 3.1 SC-01 : 1 à 500 Pulse dans un processus (60 s par niveau, 10 lots/s de 50 lignes par modèle)

Valeurs dans l'ordre 0.10 défaut / 0.10 tuned / 0.11 défaut / 0.11 tuned.

| N | p99 capture (ms) | Captures/s atteintes (cible) | Lignes écrites/s | Threads | TCP | File max (lignes) | Flush (s) |
|---|---|---|---|---|---|---|---|
| 1 | 0,65 / 0,68 / 1,19 / 1,24 | 10 / 10 / 10 / 10 (10) | 500 partout | 35 / 36 / 35 / 36 | 1 partout | 450 / 450 / 500 / 500 | 0,03 / 0,02 / 0,00 / 0,02 |
| 10 | 0,68 / 0,65 / 1,27 / 1,20 | 100 partout (100) | 4,99k / 5,00k / 4,99k / 5,00k | 44 / 36 / 44 / 36 | 10 / 1 / 10 / 1 | 3350 / 4450 / 3700 / 4850 | 0,17 / 0,00 / 0,11 / 0,03 |
| 50 | 0,90 / 0,96 / 1,18 / 1,12 | 500 partout (500) | 24,7k / 19,1k / 24,8k / 25,0k | 84 / 39 / 84 / 39 | 50 / 4 / 50 / 2 | 23 900 / 424 000 / 17 800 / 3650 | 0,6 / 18,3 / 0,5 / 0,1 |
| 100 | 13,8 / 0,94 / **15,1** / 1,09 | 647 / 999 / **780** / 989 (1000) | 32,2k / 13,7k / 38,6k / **34,3k** | 134 / 39 / 134 / 39 | 100 / 2 / 100 / 4 | 38 000 / 2 466 750 / 55 050 / **1 485 600** | 0,25 / 159 / 0,6 / **26,7** |
| 300 | 21,3 / 0,88 / **22,1** / 1,01 | 441 / 2197 / **636** / 2144 (3000) | 22,0k / 18,3k / 31,7k / **37,7k** | 334 / 39 / 334 / 39 | 300 / 2 / 300 / 3 | 21 750 / 6 510 250 / 37 950 / 6 325 550 | 0,07 / 300 (expiré) / 0,17 / **110** |
| 500 | 75,0 / 0,88 / **29,5** / 0,99 | 247 / 2312 / **517** / 2062 (5000) | 12,2k / 19,2k / 25,5k / **36,6k** | 534 / 39 / 534 / 39 | 500 / 1 / 500 / 0 | 22 200 / 6 886 750 / 20 750 / 6 160 600 | 0,4 / 300 (expiré) / 0,9 / **108** |

- **Comptes** (0.11, stats lues après `close()`) : 0 perte et 0 ligne non expliquée à tous les niveaux et dans les deux profils. En 0.11 tuned : N = 300, 6 432 650 capturées, 6 417 650 stockées, 10 000 écartées, 10 000 incertaines, 200 erreurs d'écriture ; N = 500, 6 185 600 capturées, 6 155 600 stockées, 5000 écartées, 25 000 incertaines, 500 erreurs d'écriture. En 0.10 tuned, le test affichait 13 700 et 23 100 « perdues » (défaut de test corrigé depuis).
- **Démarrage à N = 500** : 1,88 / 0,013 / 1,07 / 0,012 s.
- **RSS pic à N = 500** : 375 / 1011 / 519 / 973 Mo. RSS par modèle ≤ 2,8 Mo au-delà de N = 1 dans tous les profils (critère < 30 Mo : OK).
- **CPU du processus à N = 500** : 132 / 99 / 127 / 98 % d'un cœur.
- **Latence à faible charge** : en 0.11, le p50 de capture à N = 1 vaut 1,09 et 1,14 ms (0.10 : 0,58 et 0,59 ms), et le p99 passe de 0,65 à 1,19 ms. L'écart disparaît au-delà de 50 modèles (p50 environ 0,34 ms dans les 4 colonnes). La cause n'est pas mesurée (code ou autre instance). Le critère de moins de 2 ms reste tenu.
- **Descripteurs InfluxDB** (conteneur) : pic de 1777 en 0.11 défaut et de 297 en 0.11 tuned, 0 « too many open files ». Le compteur s'accumule d'une exécution à l'autre (1170 dès N = 1 pour la dernière exécution, celle du profil défaut). Il n'est donc pas comparable entre colonnes.

Ce qui tient en 0.11 :
- **Débit du `SharedWriter`** : 13,7k → 34,3k lignes/s à N = 100, et 18,3 / 19,2k → 37,7 / 36,6k à N = 300 / 500. Le flush de fin passe de 159 s à 26,7 s à N = 100, et d'un dépassement à 300 s à 110 / 108 s à N = 300 / 500.
- **Profil défaut** : il gagne en débit appelant (780 / 636 / 517 captures/s, contre 647 / 441 / 247) et en p99 à N = 500 (29,5 ms contre 75,0). Il reste loin des cibles.

### 3.2 SC-02 : rampe de 10k à 4M lignes/min, 180 s par palier (au lieu de 15 min)

| Configuration | Point de saturation 0.10 défaut / 0.10 tuned / 0.11 défaut / 0.11 tuned |
|---|---|
| 1 modèle | 2M / 4M / 2M / 4M lignes/min |
| 100 modèles | 4M (plafond appelant 1,93M) / **1M** / 4M (plafond appelant 2,27M) / **4M** (retard seulement) |

| Palier | Mesure | 0.10 défaut | 0.10 tuned | 0.11 défaut | 0.11 tuned |
|---|---|---|---|---|---|
| 1 modèle, 1M/min | lignes écrites/s / retard | 16,7k / 0,15 s | 16,7k / 0,01 s | 16,7k / 0,15 s | 16,7k / 0,01 s |
| 1 modèle, 2M/min | écartées (comptées) | 1 409 550 | 0 | 1 430 750 | 0 |
| | lignes écrites/s | 25,0k | 33,3k | 24,9k | 33,3k |
| | retard / file max | 3,4 s / 100 000 | 0,05 s / 4700 | 3,4 s / 100 000 | 0,10 s / 4550 |
| | lignes par écriture | non mesuré | 2166 | 4999 | 1536 |
| 1 modèle, 4M/min | débit généré | 4,00M | 3,74M | 4,00M | 3,42M |
| | écartées (comptées) | 8 278 900 | 6 918 900 | 8 485 850 | 5 887 250 |
| | lignes écrites/s | 20,3k | 23,6k | 19,2k | 24,1k |
| | p99 capture | 5,6 ms | 11,5 ms | 6,0 ms | 13,4 ms |
| 100 modèles, 1M/min | retard / file max | 0,64 s / 16 600 | **44,2 s / 669 750** | 0,49 s / 10 650 | **0,01 s / 550** |
| | lignes écrites/s | 16,6k | 13,4k | 16,6k | 16,7k |
| 100 modèles, 2M/min | débit généré | 1,92M | 2,00M | 2,00M | 2,00M |
| | retard / file max | 1,17 s / 46 350 | **270 s / 4 174 250** | 0,83 s / 30 000 | **0,18 s / 4750** |
| | lignes écrites/s | 31,8k | 13,3k | 33,2k | **33,3k** |
| | p99 capture | 21,5 ms | 0,96 ms | 1,86 ms | 1,18 ms |
| 100 modèles, 4M/min | débit généré | 1,93M | 3,73M | 2,27M | 4,00M |
| | écartées | 0 | 180 550 | 0 | **0** |
| | retard / file max | 1,05 s / 58 700 | 654 s / 10 000 000 | 0,84 s / 61 350 | **159 s / 9 116 950** |
| | lignes écrites/s | 32,0k | 13,2k | 37,7k | 35,4k |
| | p99 capture | 20,7 ms | 0,95 ms | 16,2 ms | 1,04 ms |

- Dans les 4 colonnes et à tous les paliers : `lost_unaccounted` vaut 0, `uncertain` vaut 0, il n'y a aucune erreur d'écriture, et les lignes écartées sont comptées exactement.
- `rows_per_write_mean` est maintenant mesuré en profil défaut (0.10 : `None`).
- Heap InfluxDB maximal : 680 Mo (0.11 défaut) et 688 Mo (0.11 tuned).
- Ce qui tient en 0.11 :
  - **le `SharedWriter`** : à 100 modèles, il écrit 33,3k lignes/s à 2M (0.10 : 13,3k) ;
  - **l'encodeur** : aucun gain mesurable pour 1 modèle. Le défaut sature toujours à 2M lignes/min avec environ 25k lignes/s, et le tuned toujours à 4M avec environ 24k lignes/s.

### 3.3 SC-03 : pannes de 60 s et 10 min à 200k lignes/min (10 modèles), plus un trou noir de 60 s ; profil tuned seulement

| Cas | Mesure | 0.10 défaut | 0.10 tuned | 0.11 tuned |
|---|---|---|---|---|
| refus 60 s, spool | perdues / doublons | 0 / 0 | 0 / 0 | 0 / 0 |
| | en spool et rejouées | 204 750 | 201 000 | 206 500 |
| | spool max (Mo) / octets par ligne | 58,3 / 304 | 27,0 / 143 | 27,8 / 143 |
| | rejeu (s) | 9,4 | 10,2 | 11,2 |
| | p99 pendant / après (ms) | 1,17 / 1,37 | 0,78 / 1,04 | 1,37 / 1,40 |
| | lignes après le retour (horodatage / compteur) | 3350 / — | 3350 / — | 2900 / 3350 |
| refus 60 s, sans spool | écartées / incertaines | 203 400 / 0 | 203 600 / 0 | 203 500 / **3350** |
| | drops exacts | oui | oui | oui |
| refus 10 min, spool | perdues / doublons | 0 / 0 | 0 / 0 | 0 / 0 |
| | en spool et rejouées | 2 004 050 | 2 004 950 | 2 005 550 |
| | spool max (Mo) / octets par ligne | 578,8 / 303 | 273,8 / 144 | 272,5 / 143 |
| | rejeu (s) | 86,2 | 102,9 | 103,3 |
| | p99 pendant / après (ms) | 1,06 / 29,5 | 0,78 / 1,18 | 1,38 / 1,53 |
| | max après (ms) | 261 | 30 | 45 |
| refus 10 min, sans spool | écartées | 2 004 300 | 2 003 550 | 2 003 400 |
| trou noir 60 s, spool | perdues / doublons | 0 / 0 | 0 / 0 | 0 / 0 |
| | rejouées | 206 800 | 206 450 | 207 250 |
| | spool max (Mo) / octets par ligne | 55,6 / 308 | 27,7 / 143 | 27,8 / 155 |
| | p99 pendant / après (ms) | 1,02 / 1,31 | 0,89 / 1,02 | 1,40 / 1,42 |
| | lignes après le retour (horodatage / compteur) | 17 700 / — | 3350 / — | 4050 / 18 950 |

- Ce qui tient en 0.11 tuned : 0 perte et 0 doublon avec spool, gzip à 143 octets par ligne, et un p99 de capture sous 2 ms avant, pendant et après la panne.
- Le p99 est environ 0,5 ms plus haut qu'en 0.10 tuned, comme dans SC-01 à faible charge.
- En panne de 60 s sans spool, 3350 lignes sont maintenant marquées incertaines (connexion coupée) : c'est la nouvelle comptabilité 0.11. Elles ont été stockées (stockées = capturées − écartées).
- Le nouveau compteur « après le retour » (`fallback_rows_after_recovery_by_counter`) donne 18 950 lignes pour le trou noir, contre 4050 par horodatage de capture. Comme le notait la relance, la différence vient des lignes capturées pendant le blocage, qui attendent derrière le timeout de 5 s.
- Le module n'a exécuté le trou noir qu'à 60 s.

### 3.4 SC-13 : kill -9 en pleine rafale (50k lignes/s, spool actif, 3 scénarios × 3 répétitions)

| Mesure | 0.10 défaut | 0.10 tuned (WAL) | 0.11 tuned (WAL) |
|---|---|---|---|
| Lignes soumises | 3 575 000 | 3 392 500 | 3 512 500 |
| Perdues au total (par le crash) | 435 000 (308 000) | 0 (0) | **0 (0)** |
| storage_up : perte max / file au kill | 191 500 / 100 000 | 0 / 100 000 | 0 / 97 500 |
| storage_down : perte max / file au kill | 5000 / 5000 | 0 / 3500 | 0 / 4500 |
| kill_during_replay : perte max / file au kill | 6000 / 5000 | 0 / 5000 | 0 / 4000 |
| Doublons | 0 | 0 | 0 |
| `.tmp` orphelins | 0 | 0 | 0 |

Le WAL donne toujours 0 perte après les changements du chemin d'écriture. Le test sépare maintenant la perte au premier kill de la perte au kill pendant le rejeu (`lost_by_replay_kill_total` 0, `lost_after_restart_total` 0). Le profil défaut n'est pas remesuré : la fenêtre de perte M36 du défaut (jusqu'à 100 500 lignes) reste celle du 3 novembre.

### 3.5 SC-05 : max_queue_rows de 1k à 1M, rafale 10 fois le débit d'écriture (profil défaut)

| Niveau | Mesure | 0.10 défaut (test non valide) | 0.11 défaut |
|---|---|---|---|
| 1000 | capturées / écrites / écartées | 200 000 / 0 / 200 000 | 200 000 / 19 000 / 181 000 |
| 10 000 | file max / écrites | 7000 / 0 | 10 000 / 30 000 |
| 100 000 | file max / ΔRSS (borne) | 7000 / 17,7 Mo (70,4) | 100 000 / 31,7 Mo (70,4) |
| 1 000 000 | file max / ΔRSS (borne) / écrites | 7000 / 18,9 Mo (523,5) / 0 | 1 000 000 / 195,5 Mo (523,5) / 1 270 000 |
| | octets par ligne en file | 2827 (file presque vide) | 205 |
| lot de 5000 lignes avec q = 1000 | écartées / plus grosse écriture / reçues | 5000 / 0 / — | **4000 / 1000 / 1000** |
| | p99 capture max | 0,87 ms | 1,15 ms |

Le test a été corrigé depuis le 3 novembre : il injecte maintenant une fausse requête au niveau de la `Target` du `Channel`, avec le vrai encodage. Les écritures sont mesurées (`writes_measured_all` : vrai). Le résultat est donc valide. La limite m20 de la nuit est vérifiée et corrigée : un lot plus grand que `max_queue_rows` est tronqué à 1000 lignes, et les 4000 autres sont écartées et comptées. Les comptes sont exacts à tous les niveaux.

### 3.6 SC-14 : évolution de schéma (7 cas, profil défaut)

Les chiffres 0.11 sont identiques à ceux de 0.10 dans les 7 cas :
- optional 1000/1000 ;
- float_to_str 1000 stockées, 250 rejetées ;
- int_gets_nan 751 stockées = 751 écrites, 49 rejetées ;
- add_remove 750 ;
- bool_to_int 250 stockées, 150 rejetées ;
- same_write 250 stockées (100 écrites, 50 rejetées) ;
- mixed_rows 95 stockées (45 écrites, 5 rejetées).

Totaux : 504 rejets, 0 ligne valide perdue, 0 fichier de spool restant, 0 relance après inactivité. Le chemin 0.11 ne casse pas la comptabilité des 422.

## 4. Limites CONFIRMÉES (real=true), par sévérité

Les 15 verdicts sont tous confirmés. Aucune limite n'est critique : 14 sont majeures, 1 est mineure. Les limites de même nature sont regroupées. Quand `proposal_ok=false`, la proposition corrigée par la vérification est donnée.

### Majeures

**L1. SC-01 tuned : lignes écartées par le `SharedWriter` pendant le back-off, après des timeouts d'InfluxDB. 10 000 lignes à N = 300 et 5000 à N = 500.** (2 verdicts, proposition inadaptée)
- Preuve : N = 300 : 6 432 650 capturées, 6 417 650 stockées, 10 000 écartées, 10 000 incertaines, 200 erreurs d'écriture. N = 500 : 6 185 600 capturées, 6 155 600 stockées, 5000 écartées, 25 000 incertaines, 500 erreurs. Les comptes sont équilibrés.
- Dans le journal : des timeouts de lecture et des connexions coupées sur des écritures de 5000 lignes, avec une file au pic de 6,3M lignes et environ 37,7k lignes/s écrites pour environ 107k lignes/s capturées. Suivent des lignes « rows dropped (storage unavailable) », sans aucune ligne « queue full ».
- En 0.10 tuned, on voyait déjà 10 000 écartées et 100 erreurs à N = 500.
- Proposition du test (plus de concurrence côté écrivain, moins de modèles) : inadaptée, elle chargerait davantage InfluxDB. Proposition corrigée : pendant le back-off, garder les lignes en file (dans la limite de `max_queue_rows`) au lieu de les écarter, et/ou activer le spool par défaut. Dimensionner aussi le stockage.

**L2. SC-02 tuned, 1 modèle : saturation à 4M lignes/min.** (proposition inadaptée)
- Preuve : 3,42M lignes/min générées, 5 887 250 écartées à la borne de 100 000 lignes, 24,1k lignes/s écrites (33,3k au palier de 2M), p99 de capture 13,4 ms. En 0.10 tuned : 6 918 900 écartées, 23,6k lignes/s. L'encodeur 0.11 ne déplace pas ce point.
- Proposition du test (écritures en vol, gzip, lots plus gros) : sa prémisse est fausse, car ce profil a déjà 4 requêtes en vol et des lots de 5000 lignes. La cause probable est le CPU et le GIL (encodage et capture dans le même interpréteur), plus InfluxDB. La proposition ne vise ni l'un ni l'autre.

**L3. SC-02 défaut, 1 modèle : un Pulse sature à 2M lignes/min, environ 24,9k lignes/s.** (proposition validée)
- Preuve : 1 430 750 lignes écartées, file au maximum de 100 000, 24,9k lignes/s écrites avec des lots de 5000 et `write_concurrency=1`. En 0.10 : 1 409 550 écartées, 25,0k lignes/s. L'encodeur ne change rien.
- Proposition : plusieurs requêtes en vol. Elles existent déjà (`write_concurrency`, `SharedWriter`) : le profil tuned stocke les 6M lignes du même palier à 33,3k lignes/s. Il reste à en faire le défaut documenté. Le gzip et les lots adaptatifs ne sont pas testés.

**L4. SC-02 tuned, 100 modèles : à 4M lignes/min, file de 9 116 950 lignes et retard d'ingestion de 159 s (critère : 30 s), 0 drop.** (proposition inadaptée)
- Preuve : les 11 999 950 lignes sont stockées, avec 35,4k lignes/s écrites pour 66,7k offertes. En 0.10 tuned, au même palier : file de 10 000 000, retard de 654 s, 180 550 écartées.
- Ce plafond d'environ 35 à 38k lignes/s par processus se retrouve dans SC-01. Les journaux de SC-01 tuned montrent des timeouts côté serveur à ce débit : une partie du plafond vient d'InfluxDB.
- Proposition du test : sa prémisse est fausse en tuned, et le gzip ajoute du CPU client. Proposition corrigée : dimensionner ou partitionner InfluxDB, ou utiliser des clients multi-processus.

**L5. SC-01 tuned : plafond côté appelant d'environ 2100 captures/s par processus.** (2 verdicts, proposition validée)
- Preuve : 2144 captures/s à N = 300 (71 %) et 2062 à N = 500 (41 %), avec un CPU du processus à 99,2 et 98,1 % d'un cœur. En 0.10 : 2197 et 2312 captures/s.
- À la cible, 3000 captures/s × 0,336 ms = 1,01 cœur pour la capture seule, et 5000/s × 0,337 ms = 1,7 cœur : c'est impossible dans un seul processus Python.
- Proposition : documenter le plafond par processus et répartir la charge sur plusieurs processus. Construire les frames dans le thread d'écriture n'apporte qu'un gain marginal, puisque le CPU reste sous le même GIL.

**L6. SC-01 défaut : p99 de capture à 15,1 / 22,1 / 29,5 ms à N = 100 / 300 / 500 (cible : moins de 2 ms).** (3 verdicts, proposition validée)
- Preuve : p50 0,34 à 0,35 ms ; max 629,5 / 1020,2 / 1298,7 ms ; 134 / 334 / 534 threads ; CPU 109 / 118 / 127 % d'un cœur. En 0.10 : 13,8 / 21,3 / 75,0 ms. Le profil tuned, avec le même générateur, donne 1,09 / 1,01 / 0,99 ms avec 39 threads.
- Proposition : le `SharedWriter` existe déjà en 0.11. Il faut en faire le défaut, ou le documenter, pour les serveurs multi-modèles.

**L7. SC-01 défaut : débit de 780 / 636 / 517 captures/s à N = 100 / 300 / 500 (78 / 21 / 10 %).** (3 verdicts, proposition inadaptée)
- Preuve : mesuré. En 0.10 : 647 / 441 / 247 captures/s.
- L'explication du test (CPU de `prepare_frame`) est contredite par ses propres chiffres : un p50 de 0,34 ms donnerait environ 2900 captures/s. Le manque vient de la contention avec les N threads d'écriture. Le profil tuned atteint 989 / 2144 / 2062 captures/s.
- Proposition corrigée : le `SharedWriter`, puis la répartition sur plusieurs processus. Une capture moins coûteuse ne règle pas la cause principale.

**L8. SC-02 défaut, 100 modèles : à 4M lignes/min demandés, 2 272 816 lignes/min générées, 0 drop.** (proposition inadaptée)
- Preuve : 6 820 700 lignes stockées, retard de 0,84 s, 37,7k lignes/s écrites, p99 de capture 16,2 ms. Les écrivains suivent : c'est le générateur qui est affamé par 100 threads d'écriture, comme dans L7. En 0.10, le plafond était de 1,93M lignes/min.
- Proposition corrigée : le `SharedWriter` (le tuned génère 4,00M lignes/min). Le débit d'écriture n'était pas le goulot.

### Mineure

**L9. SC-01 défaut : un thread d'écriture et un pool HTTP par Pulse, soit 534 threads et 500 sockets TCP à N = 500, et un démarrage de 1,07 s.** (proposition validée)
- Preuve : la croissance est linéaire, de 35 threads et 1 socket à N = 1 jusqu'à 534 et 500 à N = 500. La cause du démarrage (« vérification du bucket, 2 ms/modèle ») est affirmée par le test, pas mesurée. En tuned : 39 threads, au plus 4 TCP, démarrage de 0,012 s.
- Proposition : `SharedWriter` par défaut, ou documenté, pour les serveurs multi-modèles.

### Bilan des limites de la relance du 3 novembre

| Limite | Sujet | Verdict 0.11 | Chiffres 0.10 → 0.11 |
|---|---|---|---|
| N1 | Débit du `SharedWriter` (13 à 19k lignes/s) | **Corrigée en grande partie** | 100 modèles à 2M : 13,3k → 33,3k lignes/s, retard 270 → 0,18 s ; à 4M : retard 654 → 159 s, écartées 180 550 → 0 ; SC-01 N = 100 : 13,7k → 34,3k lignes/s. Reste L4. |
| N2 | Plafond appelant en tuned | **Toujours présente** | 2197 / 2312 → 2144 / 2062 captures/s (N = 300 / 500) |
| N3 | 1 modèle tuned sature à 4M | **Toujours présente** | 6 918 900 → 5 887 250 écartées ; 23,6k → 24,1k lignes/s |
| M31 | Un thread et un client par Pulse (défaut) | **Toujours présente, un peu améliorée** | p99 à N = 500 : 75,0 → 29,5 ms ; captures/s à N = 500 : 247 → 517 ; démarrage : 1,88 → 1,07 s |
| M32 | Lignes incertaines | **Corrigée et maintenant exercée** | 0.10 : jamais de valeur non nulle ; 0.11 : 10 000, 25 000 et 3350 lignes incertaines, comptes équilibrés |
| M34 | Un Pulse sature à environ 25k lignes/s (défaut) | **Toujours présente** | 25,0k → 24,9k lignes/s ; 1 409 550 → 1 430 750 écartées à 2M |
| M36 | kill -9 (défaut) | Non remesurée (défaut sauté) ; WAL : 0 perte confirmée | tuned : 0 → 0 |
| m18 | Latence pendant le rejeu | Tuned : tient | p99 après une panne de 10 min : 1,18 → 1,53 ms |
| m19 | Taille du spool | Tient (gzip) | 143-144 → 143 octets par ligne |
| m20 | Lot plus grand que `max_queue_rows` | **Corrigée (vérifiable maintenant)** | 4000 écartées sur 5000, écriture max de 1000 |

## 5. Limites réfutées et défauts de test

Aucune limite n'est réfutée par la vérification (0 sur 15). Deux défauts de test ou d'hôte ont été corrigés pendant la remesure, sans modifier `pulsar_data_collection` ni `pulsar_metrics` :

| Essai | Constat | Pourquoi ce n'est pas une limite | Traitement |
|---|---|---|---|
| `SC-01.v011-default.bucket-collision` (19:56 UTC) | Une limite à N = 1 : « 2500 lignes comptées écartées mais stockées » (30 000 capturées, 32 500 stockées, comptes non équilibrés) | Le smoke de 19:55 avait écrit exactement 2500 lignes (N = 1) dans le même bucket, dont le nom avait été réutilisé. C'est un artefact du test. | Exécution refaite proprement à 22:11 UTC (N = 1 : 30 000 = 30 000, équilibré). La limite N = 1 est écartée. |
| `SC-14.v011-default.no-influx-env` (20:16 UTC) | `FileNotFoundError` sur `/home/ubuntu/.pulsar-lab/influx.env` | Fichier d'environnement absent sur l'hôte : un oubli de préparation de l'hôte | Corrigé sur l'hôte, sans changer le module. Exécution refaite à 22:10 UTC (passed). |

Les verdicts ont aussi relevé, dans des limites confirmées, des attributions de cause fausses dans le texte du test : le CPU de `prepare_frame` (L7), « le writer ne suit pas » (L1) et la prémisse de L2 / L4. Les limites restent réelles. Les propositions corrigées sont en section 4.

Les défauts de test listés le 3 novembre sont corrigés :
- SC-05 : la fausse écriture se fait au niveau de la `Target` ;
- SC-01 : les stats sont lues après `close()` ;
- SC-03 : ajout du compteur « après le retour » ;
- SC-13 : la perte est attribuée au kill qui la provoque.

`rows_per_write_mean` est aussi mesuré en défaut.

## 6. Tests non exécutés

| Test / profil | Raison |
|---|---|
| SC-03 défaut | Sauté par le plan : la 0.11 ne touche pas la détection des pannes ; les chiffres du 3 novembre restent valables |
| Panne de 30 min de SC-03 (tuned) | Hors plan (`LAB_OUTAGES=60,600`) |
| SC-13 défaut | Sauté par le plan : SC-13 tourne en tuned seulement |
| SC-05 tuned, SC-14 tuned | Sautés par le plan : ces tests tournent en défaut seulement |
| SC-06, SC-11, SC-12 | Hors périmètre de la remesure du chemin d'écriture |

Rien n'a été coupé pour cause de budget : la séquence est entrée dans le plafond de 12 USD.

## 7. Coût et santé

**Coût** (tarifs à la demande ca-central-1, d'après `results/provision.json` : 2,195 USD/h au total)

| Poste | Détail | USD à 22:28 UTC |
|---|---|---|
| lab-scale c7i.8xlarge | 1,5624 USD/h × 2,79 h (lancée à 19:41:12 UTC) | 4,36 |
| lab-db r7i.2xlarge | 0,5796 USD/h × 2,79 h (lancée à 19:41:22 UTC) | 1,62 |
| EBS gp3 | 4 volumes, 360 Go (30 + 100 + 30 + 200), 0,043 USD/h | 0,12 |
| IPv4 publiques | 2 × 0,005 USD/h | 0,03 |
| **Total** | | **≈ 6,1** |

- Budget approuvé : **12 USD**. Projection : environ 8,4 USD si les instances tournent jusqu'à l'arrêt automatique (timer systemd `pulsar-lab-stop` à 23:30 UTC). Le plan prévoyait environ 7,4 USD, et `provision.json` environ 8,2 USD.
- **État à 22:28 UTC** (`describe-instances` en lecture seule) : `i-07d601a69020c202e` (lab-scale) et `i-09f5073c10678a227` (lab-db) sont `running`, toutes deux tagguées `Project=pulsar-lab` et créées pour cette remesure. Aucune instance préexistante n'a été réutilisée. Les 4 volumes `Project=pulsar-lab` sont `in-use`, dont 2 volumes de données `LabVolume=data`. **Le démontage (terminate, volumes compris) reste à faire** ; il n'est pas couvert par ce rapport. Après l'arrêt automatique, seuls les volumes coûtent encore (environ 0,04 USD/h).

**Santé** (blocs `health` des fichiers 0.11)
- Exécutions : 7 exécutions retenues, toutes avec un statut final ; 1 erreur d'hôte (SC-14, refaite) ; 1 exécution refaite pour défaut de test (SC-01 défaut). Aucun plantage du harness.
- lab-scale (32 vCPU) : CPU de l'hôte ≤ 4,1 %, mémoire ≤ 3,8 %, disque ≤ 10,9 %. Les processus de test sont limités à un cœur (GIL), pas l'hôte.
- RSS maximal d'un processus de test : 1281 Mo (SC-02 tuned, file de 9,1M lignes), 973 Mo (SC-01 tuned), ≤ 519 Mo en défaut.
- Threads maximum : 534 en défaut (SC-01), 39 en tuned. Descripteurs du processus : ≤ 506.
- InfluxDB : pic de 1777 descripteurs dans le conteneur pour une limite de 65 536, 0 « too many open files », heap ≤ 688 Mo. Des timeouts de lecture côté serveur apparaissent dans SC-01 tuned au-delà d'environ 37k lignes/s (L1, L4).
- Les valeurs `rss_growth_mb_per_h` (jusqu'à 4996 Mo/h) sont extrapolées sur des exécutions de quelques minutes, avec des files qui se remplissent puis se vident. Elles n'indiquent pas de fuite.
- Rien n'a été commité, poussé ou fusionné. `pulsar_metrics` et `pulsar_data_collection` n'ont pas été modifiés.

## 8. Propositions pour `pulsar_data_collection` (à décider, non appliquées)

1. **Back-off du `SharedWriter` (L1)** : garder les lots en file pendant le back-off, dans la limite de `max_queue_rows`, au lieu de les écarter. Recommander `spool_dir` pour les serveurs multi-modèles.
2. **Profil par défaut (L3, L6, L7, L8, L9)** : le `SharedWriter` 0.11 tient maintenant environ 33 à 38k lignes/s à 100 modèles. On peut le recommander, ou l'activer par défaut, au-delà d'environ 50 modèles par processus.
3. **Plafonds à documenter (L2, L4, L5)** : environ 2100 captures/s par processus en tuned ; environ 25k lignes/s par écrivain privé ; environ 35 à 38k lignes/s par processus vers un InfluxDB r7i.2xlarge. Au-delà : plusieurs processus, et dimensionnement ou partitionnement d'InfluxDB.
4. **Latence à faible charge** : le p50 de capture à N = 1 passe de 0,58 à 1,09 ms entre 0.10 et 0.11. Le mesurer sur une même instance, avec les deux versions, pour séparer l'effet du code de celui de la machine.
5. **Profils sautés** : remesurer SC-03 et SC-13 en défaut en 0.11 à la prochaine relance, pour confirmer M36 et m18 sur la version livrée.
