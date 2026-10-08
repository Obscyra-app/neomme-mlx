# NeoMME retrieval on MLX

Unofficial native Python MLX inference for H Company's NeoMME-260M-Retriever.
Text and document images share one encoder. Returns 1024-dimensional dense
vectors and 128-dimensional token vectors. Scores use cosine or **MeanMaxSim**.
This is a retrieval encoder, not a chatbot or OCR transcription engine.

## Install and run

Requires Apple Silicon and Python 3.12. After downloading the model package (see below):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m neomme_mlx --model . index document.pdf notes.txt image.png --output my-index
.venv/bin/python -m neomme_mlx --model . search --index my-index --query 'Когда отправляется поезд?' --mode late
```

The workspace version uses `experiments.neomme_mlx` as the module name.
Indexing and searching are offline. Existing indexes are never overwritten.
PDFs are rendered page-by-page with PDFium; text files are chunked by paragraphs.
The default image longest side is 1024 pixels; `index --max-side 512` uses less
compute but may miss small text. `--max-pages` defaults to 100 and rejects larger
PDFs before encoding. PNG/JPEG/WebP/BMP/TIFF, UTF-8 TXT/MD and PDF are supported.
PDF rendering preserves page appearance; no OCR or text-layer shortcut is used.

## Python

```python
from neomme_mlx import load_model
from neomme_mlx.processor import Processor
from neomme_mlx.retrieval import mean_maxsim
import numpy as np

processor = Processor('.', max_side=512)
model = load_model('.', dtype='float32')
q = model(**processor(['Find the invoice total.'], task='query'))
d = model(**processor([{'image': 'invoice.png'}], task='document'))
score = mean_maxsim(np.array(q['multi'][0]), np.array(d['multi'][0]))
```

For padded batches, remove rows where `attention_mask == 0` before passing
vectors to `mean_maxsim`. Query formatting adds `<query>` and ten learned mask
tokens; documents use `<doc>`. Image documents include the image header, patch
tokens and row markers. Do not manually add these tokens to ordinary text.
`dense_dim=128/256/512/1024` truncates pooled features before normalization.

## Precision and limits

FP32 is the validated deployment precision. Source BF16 tensors load directly
and expand to FP32 in memory; no training or changes to source weights occur.
FP16 is experimental and failed the frozen token-vector parity thresholds on
page and long-text fixtures. The CLI therefore only exposes FP32.

The processor rejects inputs above 4096 tokens by default instead of silently
truncating image layouts or query expansion tokens. The architecture supports
16384 positions, but this release does not claim validation at that length.
Images are RGB-converted as in the official processor. EXIF orientation is not
automatically applied; normalize orientation before indexing if needed.
Large indexes are currently loaded into RAM for exact search; this is a small
local corpus tool, not an ANN database or a browser application.

## Provenance

- Source: https://huggingface.co/Hcompany/NeoMME-260M-Retriever
- Revision: `481083153ac22931afbd614e672212bffbcb666e`
- Original model and reference implementation: H Company / Hugging Face,
  Apache-2.0. See LICENSE and NOTICE.
- No official affiliation. No claim of first implementation, general retrieval
  quality, or universal performance improvement. See VALIDATION.md.

Manifest records all packaged file hashes. `python verify_package.py` verifies
them without loading the model. Development oracle and audit scripts are
separate from the runtime, which does not import Torch or Transformers.

## Download

Download all files from this repository with `huggingface_hub.snapshot_download`,
then follow the install instructions above in that directory.

```python
from huggingface_hub import snapshot_download
snapshot_download("Nurymanau/NeoMME-260M-Retriever-MLX", local_dir="neomme-mlx")
```

[Source code](https://github.com/Obscyra-app/neomme-mlx).
Validated local release v0.1.0. The weights are original BF16 safetensors;
this native runtime expands them to FP32. This is not an MLX-LM chat model.

## Source checkout

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt huggingface_hub==1.33.0
.venv/bin/python download_model.py --output model
.venv/bin/python -m neomme_mlx --model model index document.pdf --output my-index
.venv/bin/python -m neomme_mlx --model model search --index my-index --query 'Find the invoice total'
```

The downloader pins the exact published model revision and verifies its manifest.
Python 3.12 on Apple Silicon is required. See VALIDATION.md for measured limits.
