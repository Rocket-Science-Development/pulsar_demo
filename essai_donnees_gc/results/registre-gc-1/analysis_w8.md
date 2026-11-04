Windows of 8 agent runs; reference: 80 runs; FDR 0.05; confirmation over 2 windows.

| Phase | Runs | Windows | With alert | Confirmed | First alert | First confirmed | Confirmed features | Any alert |
|---|---|---|---|---|---|---|---|---|
| holdout | 40 | 5 | 2 | 0 | 1 | - | - | latency_s, output_tokens |
| corpus_loss | 40 | 5 | 1 | 0 | 5 | - | - | det_retrieval_top |
| language_en | 40 | 5 | 3 | 2 | 3 | 4 | output_tokens | input_tokens, latency_s, output_tokens |
| model_granite | 40 | 5 | 5 | 4 | 1 | 2 | det_faithfulness, det_retrieval_top, first_tool, input_tokens, latency_s, output_tokens, steps, task_success, tool_calls | det_faithfulness, det_retrieval_top, first_tool, input_tokens, latency_s, output_tokens, steps, task_success, tool_calls |


Retrieval quality, one-sided Mann-Whitney (windows flagged, first flag, first confirmed):

| Phase | Faithfulness | Best retrieval score |
|---|---|---|
| holdout | 0 / 5, -, - | 0 / 5, -, - |
| corpus_loss | 2 / 5, 1, - | 2 / 5, 2, - |
| language_en | 0 / 5, -, - | 0 / 5, -, - |
| model_granite | 0 / 5, -, - | 5 / 5, 1, 2 |

MMD of the request embeddings (windows flagged, first flag, first confirmed):

| Phase | Windows flagged | First flag | First confirmed |
|---|---|---|---|
| holdout | 1 / 5 | 2 | - |
| corpus_loss | 0 / 5 | - | - |
| language_en | 5 / 5 | 1 | 2 |
| model_granite | 0 / 5 | - | - |

Canary model equality: same p = 0.039, granite p = 0.001

| Phase | Runs | Accuracy [95 % CI] | Target retrieved | Mean s/run |
|---|---|---|---|---|
| reference | 80 | 0.90 [0.81, 0.96] | 0.88 | 45.2 |
| holdout | 40 | 0.88 [0.73, 0.96] | 0.88 | 41.4 |
| corpus_loss | 40 | 0.57 [0.41, 0.73] | 0.53 | 46.2 |
| language_en | 40 | 0.90 [0.76, 0.97] | 0.95 | 37.4 |
| model_granite | 40 | 0.25 [0.13, 0.41] | 0.00 | 7.6 |
