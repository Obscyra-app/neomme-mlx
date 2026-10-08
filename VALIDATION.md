# Validation — 2026-10-08

Source: Hcompany/NeoMME-260M-Retriever, revision
`481083153ac22931afbd614e672212bffbcb666e`.
162 tensors; original BF16 weights are 526,156,876 bytes. No training.
Runtime loads all weights with exact name/shape validation and expands to FP32.

## Numerical parity

Official Transformers 5.19.0 / Torch 2.14.1 CPU FP32 oracle versus MLX 0.32.3.
Six development groups (23 inputs) and three separate control groups (12 inputs):
English/Russian text, multilingual/empty queries, four synthetic document pages,
mixed padded batches, RGB/gray/alpha input, resized pages and 1202-token text.

All FP32 embedding gates passed: max absolute error <=2e-4 and per-vector
cosine >=0.99999 for dense and unpadded token vectors. Token IDs, masks and
spatial positions matched exactly. Pixel differences <=1e-6.
24 additional randomized image processor comparisons passed against the official
processor, along with text formatting, error handling and scorer identities.
Batch versus singleton and dense dimensions 128/256/512/1024 also passed.

Independent NumPy float64 replay checked 184 pairwise scores across dense and
MeanMaxSim retrieval. Maximum score difference was <1.1e-6; all 40 query-ranking
comparisons matched completely. Original eight source files re-hashed unchanged.

Parity is not accuracy. One new Russian electricity query selected the wrong
page in dense mode in both implementations; MeanMaxSim selected the intended
page. On a separate short text control set, both implementations selected an
incorrect document for that query. These tiny synthetic tests are not a general
retrieval benchmark or evidence of uniformly strong Russian retrieval.

## Rejected candidate and repaired issue

FP16 did not pass the frozen thresholds on page and long-text token vectors
(long-text minimum cosine 0.98281, below 0.999). It is not the deployment default.
The first resize implementation used float intermediates and differed from
Torch uint8 resize. It was replaced with int16 coefficients and per-pass rounding;
original failure artifacts are retained in the private development workspace.

## Matched warm timings

Apple Silicon Mac, FP32, same saved inputs, three warmups and nine measurements
per workload. CPU uses four Torch threads; MLX uses the GPU. Model forward and
embedding heads only; processor, PDF rendering and file I/O excluded.

| Workload | Torch CPU p50 | MLX GPU p50 |
|---|---:|---:|
| Six queries, padded to 23 tokens | 83.42 ms | 109.66 ms |
| Four pages, 210 tokens each | 334.40 ms | 285.45 ms |
| One text, 1202 tokens | 575.46 ms | 449.37 ms |

One process per backend, not a broad or repeated hardware benchmark. MLX was
slower on short queries in this measurement. No universal acceleration claim.
Direct MLX validation sampled process footprint peaked around 1.54 GB (decimal),
not an exact peak or a guarantee for arbitrary inputs. All completed guarded
model runs showed zero positive growth in system swap.

## Functional scope

Local CLI indexes image files, UTF-8 TXT/MD chunks and rendered PDF pages,
then searches with dense cosine or MeanMaxSim. Six actual CLI searches selected
the expected synthetic PDF page. Runtime import checks exclude Torch/Transformers.
No production dataset, training, cloud GPU, publication or external messaging.
16K context, large corpora and iOS have not been validated.
