reference of 80 rows holds 4.0 windows of 20: at least 10 are needed (200 rows; 30 for stable thresholds). Pass allow_short=True to calibrate anyway: thresholds are then flagged reliable=False

Retrieval quality, one-sided Mann-Whitney (windows flagged, first flag, first confirmed):

| Phase | Faithfulness | Best retrieval score |
|---|---|---|
| holdout | 0 / 2, -, - | 0 / 2, -, - |
| corpus_loss | 2 / 2, 1, 2 | 1 / 2, 2, - |
| language_en | 0 / 2, -, - | 0 / 2, -, - |
| model_granite | 0 / 2, -, - | 2 / 2, 1, 2 |

MMD of the request embeddings (windows flagged, first flag, first confirmed):

| Phase | Windows flagged | First flag | First confirmed |
|---|---|---|---|
| holdout | 0 / 2 | - | - |
| corpus_loss | 0 / 2 | - | - |
| language_en | 2 / 2 | 1 | 2 |
| model_granite | 0 / 2 | - | - |

Canary model equality: same p = 0.039, granite p = 0.001

| Phase | Runs | Accuracy [95 % CI] | Target retrieved | Mean s/run |
|---|---|---|---|---|
| reference | 80 | 0.90 [0.81, 0.96] | 0.88 | 45.2 |
| holdout | 40 | 0.88 [0.73, 0.96] | 0.88 | 41.4 |
| corpus_loss | 40 | 0.57 [0.41, 0.73] | 0.53 | 46.2 |
| language_en | 40 | 0.90 [0.76, 0.97] | 0.95 | 37.4 |
| model_granite | 40 | 0.25 [0.13, 0.41] | 0.00 | 7.6 |
