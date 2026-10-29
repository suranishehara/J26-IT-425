# Component 1 — Virtual Try-On and Draping

**New laptop setup:** [Installation.md](Installation.md).

Component 1 generates an experimental image of a person wearing a supplied garment reference. The current application uses pretrained **CatVTON**, automatic **SegFormer clothing masks**, user mask corrections and outside-mask compositing. A separate **VITON-HD benchmark pipeline** is retained.

## Contents

- [1. Objectives and current status](#1-objectives-and-current-status)
- [2. Start the application](#2-start-the-application)
- [3. End-to-end workflow](#3-end-to-end-workflow)
- [4. Models and techniques](#4-models-and-techniques)
- [5. Automatic masking and corrections](#5-automatic-masking-and-corrections)
- [6. Datasets and demonstration images](#6-datasets-and-demonstration-images)
- [7. VITON-HD benchmark pipeline](#7-viton-hd-benchmark-pipeline)
- [8. Saved outputs and recorded results](#8-saved-outputs-and-recorded-results)
- [9. Tests and manual checks](#9-tests-and-manual-checks)
- [10. Research evaluation](#10-research-evaluation)
- [11. API and implementation structure](#11-api-and-implementation-structure)
- [12. Troubleshooting and performance](#12-troubleshooting-and-performance)
- [13. Privacy, permissions and reproducibility](#13-privacy-permissions-and-reproducibility)
- [14. Remaining work and resume point](#14-remaining-work-and-resume-point)
- [15. Supporting documentation and sources](#15-supporting-documentation-and-sources)

## 1. Objectives and current status

The research objective is to investigate whether adapting a pretrained virtual try-on model using permitted Sri Lankan garment examples improves garment appearance and traditional draping fidelity compared with an unchanged baseline. The current work establishes inference, masking, a usable interface and evaluation scaffolding; it does not yet establish that research improvement.

| Area | Completed implementation | Boundary / outstanding work |
| --- | --- | --- |
| Browser application | Three-column desktop workspace: inputs, editable mask, result; PNG download | Local development application, not a deployed service |
| Uploaded-image inference | Real CatVTON image generation on an NVIDIA RTX 4050 | Pretrained model; no local fine-tuning |
| Automatic masks | CPU SegFormer parsing, category-dependent mask construction | Approximate regions; always review |
| Mask editing | Paint, erase, brush size and clear controls | User must include intended replacement area |
| Quality modes | Standard 384×512 / 30 steps and Detailed 768×1024 / 50 steps | Higher resolution does not guarantee accurate sleeves, prints or drape |
| Job execution | Serialized GPU jobs, status polling, output and log storage | No persistent job-status database or cancellation control |
| Benchmark | VITON-HD assets/configuration, verification and six-pair pilot outputs | Not a full-dataset quantitative evaluation |
| Traditional categories | Saree and Kandyan/Osariya options with full-outfit masks | Category selection is manual metadata, not a trained local adapter |
| Research records | Dataset template, protocol and evaluation CSV exporter | Independent human ratings and baseline/adapted study remain unfinished |

No percentage-complete estimate is claimed: a working application and a completed research study are different milestones.

## 2. Start the application

### Existing prepared environment

Open PowerShell inside `component-1-virtual-tryon`:

```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe backend\app.py
```

Open [http://localhost:8000](http://localhost:8000). Keep the terminal running. Stop it with `Ctrl+C` when finished.

Activation is optional when using the explicit `.venv` Python command. If PowerShell blocks activation, run the second command directly rather than changing machine-wide execution policy. Component 1 uses port **8000**; Component 2 uses **8001**.

### Preparing a new environment

The existing environment uses Python 3.12.14. Its installed package metadata records PyTorch 2.14.1+cu126, torchvision 0.29.1+cu126, Diffusers 0.32.2 and Transformers 4.46.3. These are the local snapshot, not a promise that installing unpinned packages later reproduces it exactly.

For a fresh Windows environment, install a compatible NVIDIA driver and CUDA-enabled PyTorch build. The existing project uses CUDA 12.6 wheels:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements-catvton.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-viton.txt
```

The second requirements file adds packages needed for the retained VITON-HD pipeline, including NumPy, OpenCV and torchgeometry. Do not recreate or replace an existing working `.venv` just to follow these fresh-install instructions.

Check CUDA without generating an image:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

The server needs Pillow even before generation. CatVTON generation requires CUDA in the current worker; CPU generation is not implemented. SegFormer deliberately runs on CPU to preserve GPU memory.

`external/CatVTON` and `external/VITON-HD` contain the upstream source used by the wrappers. Requirements installation does not recreate missing source directories or download the VITON-HD dataset/checkpoints. Do not substitute the full upstream CatVTON application/Detectron2 dependency stack for this project's narrower inference setup without reviewing compatibility.

### Model downloads

On first use, Hugging Face models download to `runtime/models` unless an existing `HF_HOME` environment variable overrides that location. This needs network access, disk space and compliance with upstream access terms. Uploads are not sent to a remote image-generation API. Later requests can reuse cached files, but the current processes still reload models and may contact the model host for metadata.

## 3. End-to-end workflow

```text
Person photo ──> browser crop to 768×1024 ──> SegFormer CPU parsing
                                                    │
Manual category ──> mask policy ──────────────────────┘
                                                    │
                                         automatic clothing mask
                                                    │
                                         user paint / erase review
                                                    │
Garment reference ──> browser fit on white 768×1024 canvas
                                                    │
Person + garment + reviewed mask + seed + quality ────┘
                         │
                validated local API request
                         │
                saved job → serialized GPU worker
                         │
                pretrained CatVTON diffusion
                         │
             raw image → outside-mask compositing
                         │
              saved PNG + metrics + inference log
                         │
                 browser result and download
```

### What the user does

1. Upload a clear person photo. Use a full-body view for dress/saree experiments, even lighting and enough space around the clothing.
2. Upload a clean garment reference. A reference showing the intended drape may help experimentation but does not guarantee the model reproduces it.
3. Choose Shirt / blouse / kurta, Dress, Saree or Kandyan saree / Osariya. Person upload and category changes trigger automatic clothing detection.
4. Review the highlighted mask. Paint missing garment areas; erase face, hair and unwanted areas. Long kurtas or extended saree fabric can need manual expansion.
5. Select quality and seed. The browser defaults to **Detailed** and seed **42**.
6. Select Generate try-on. The page polls every 2.5 seconds until completion or failure.
7. Inspect the result for identity, garment colour/print, sleeve length, boundaries and drape before downloading or rating it.

### What happens internally

- The browser centre-crops the person to a 768×1024 canvas. It fits the garment inside the same canvas with white padding, preserving aspect ratio.
- The API receives PNG data URLs for person, garment and mask. All three must have the same dimensions: either 384×512 or 768×1024.
- The mask is thresholded at 128. Accepted mask coverage is strictly greater than 1% and less than 90%.
- The request validates category, quality and seed, then saves the inputs in a UUID-named job directory.
- A background thread waits for a single generation lock. Each job launches a new CatVTON subprocess with a 30-minute execution timeout after acquiring the lock.
- The worker resizes inputs to the selected inference resolution and performs seeded diffusion. It saves the raw generated image and then composites it with the normalized person image.
- Outside-mask pixels are preserved relative to that normalized/resized input, **not the untouched original upload**. Feathering is inward only, using a Gaussian-blurred mask limited by the original mask.

Changing seed, mask or quality is inference configuration, not model training.

## 4. Models and techniques

| Role | Model / implementation | Actual use |
| --- | --- | --- |
| Main generator | `zhengchong/CatVTON`, `mix` attention checkpoint (`mix-48k-1024`) | Garment-conditioned latent diffusion inpainting |
| Base inpainting network and scheduler | `booksforcharlie/stable-diffusion-inpainting` | UNet and DDIM scheduler loaded by CatVTON |
| Variational autoencoder | `stabilityai/sd-vae-ft-mse` | Image/latent encoding and decoding |
| Safety checker | Stable Diffusion safety checker and CLIP image processor from the base checkpoint | Enabled by the pipeline's default; not a garment classifier |
| Automatic clothing parsing | `mattmdjaga/segformer_b2_clothes` | ATR clothing-label segmentation on CPU |
| Independent benchmark | Official VITON-HD | Segmentation, geometric garment warping and ALIAS synthesis |
| Final preservation | Pillow / NumPy compositing | Restore normalized source pixels outside the selected mask |

CatVTON uses visual conditioning from the person, replacement mask and garment. Its pipeline concatenates person/garment latent representations and applies pretrained attention weights in a diffusion UNet; this application does not use a text prompt to specify the clothing. The worker uses float16 CUDA inference, VAE slicing/tiling and the pipeline's default guidance scale of 2.5.

The application does **not** use the vendored CatVTON DensePose/SCHP auto-mask stack, FLUX variant or mask-free variant. Those source files existing under `external/` do not mean they are active. Our auto-mask path is SegFormer plus explicit mask rules.

### Quality settings

All resolutions below are width × height.

| UI option | API value | Model resolution | Diffusion steps |
| --- | --- | --- | --- |
| Standard | `balanced` | 384×512 | 30 |
| Detailed — browser default | `high` | 768×1024 | 50 |

An API request omitting `quality` defaults to `balanced`, unlike the browser's explicit `high` selection. Current browser inputs retain a 768×1024 canvas; the old statement that every upload is reduced to 384×512 no longer describes this implementation. Previously saved low-resolution smoke inputs still cannot regain detail merely by enlargement.

### Inference versus training

No model was trained from scratch or locally fine-tuned in Component 1. We downloaded and integrated pretrained weights, built preprocessing/masking, added inference controls, implemented output compositing and tested execution. No trained Sri Lankan drape adapter, local LoRA, automatic garment classifier or physical cloth simulation has been completed.

## 5. Automatic masking and corrections

`scripts/detect_clothing.py` loads SegFormer in a separate CPU process for each detection request. It interpolates model logits to the person canvas and chooses the highest-scoring semantic label per pixel.

Mask policies use this parser's ATR IDs:

| Selection | Labels initially included | Additional handling |
| --- | --- | --- |
| Upper body | Upper garment 4, dress 7, scarf 17 | For a one-piece dress, approximate bodice boundary at 45% of its vertical extent; protect its lower portion |
| Dress / Saree / Kandyan-Osariya | Upper garment 4, skirt 5, trousers 6, dress 7, belt 8, scarf 17 | Include lower garments and horizontally fill between selected boundaries over lower-clothing rows |

The full-outfit gap filling was added because a source person wearing trousers otherwise leaves the background between the legs locked outside the mask. A continuous skirt needs that region available for synthesis. This is a geometric heuristic, not traditional pleat recognition.

The code closes small gaps, fills holes and dilates the selected area. It then removes protected parser regions: hat, hair, sunglasses, shoes, face and bag. Upper-body mode additionally protects lower garments, belt and legs. Protection depends on parser correctness; review the actual mask rather than assuming these areas are always safe.

Dress, saree and Kandyan/Osariya currently share the same full-outfit mask policy. Their names do not select different learned model weights. Traditional draping landmarks, folds and pleats are not explicitly estimated.

Detection uses a serialized lock, temporary files under `runtime/uploads` and a 10-minute subprocess timeout. Temporary detection files are cleaned on normal context exit. If detection fails, the user can still paint a mask manually. A new detection can replace manual corrections, so review again after changing category or re-detecting.

## 6. Datasets and demonstration images

### VITON-HD benchmark assets

The local project contains a small official benchmark pilot, not evidence that the entire training/test dataset has been downloaded:

```text
external/VITON-HD/
  checkpoints/
    seg_final.pth
    gmm_final.pth
    alias_final.pth
  datasets/
    test_pairs.txt
    test_pairs_pilot.txt
    test/
      image/
      cloth/
      cloth-mask/
      image-parse/
      openpose-img/
      openpose-json/
```

At this snapshot, the pilot list has **six person/garment pairs**. The local test folders contain six person images and six files each for parsing and OpenPose inputs, with twelve cloth images/masks. Folder presence is not a full-dataset completeness check. The upstream benchmark is frontal upper-body clothing, not a Sri Lankan traditional-drape dataset.

The CatVTON smoke test reuses the first available VITON-HD pilot pair and its indexed human-parsing mask. That benchmark parsing uses different label IDs from the SegFormer auto-mask path; the smoke script selects labels 5/6/7 specifically for that benchmark representation. Do not interchange the label schemas.

### Synthetic demonstration collection

The project also has separately generated references for demonstrations:

| Location under `data/` | Contents at this snapshot |
| --- | --- |
| `garment_references/shirts` | 5 synthetic references |
| `garment_references/blouses` | 5 synthetic references |
| `garment_references/kurtas` | 5 synthetic references |
| `garment_references/dresses` | 5 synthetic references |
| `garment_references/sarees` | 5 synthetic references |
| `garment_references/kandyan_osariya` | 10 synthetic references in nested folders |
| `person_references` | 10 synthetic person PNGs |

ZIP bundles and prompt records are also stored locally. These are **not paired real training data**, verified drape labels, identity-preservation ground truth or evidence of try-on accuracy. The seven-item `data/garments.json` is a legacy catalogue for the earlier routing prototype; the current page accepts uploaded references instead.

### Local Sri Lankan adaptation data — planned, not completed

`data/local_sri_lanka` contains collection guidance and an annotation template. A supervised example needs a permitted person photo, garment reference and reference photo of that same person wearing that garment, plus labels and source/consent records.

Suggested fields include sample/subject IDs, all three image paths, garment family/type, drape/pleat/front-fall labels, pattern, colour, fabric, permission reference and split. Split by **person and physical garment**, keeping duplicates and alternate views in the same split. Do not infer that Component 2's unpaired classification images satisfy this paired try-on requirement.

Dress Code is mentioned in the dataset manifest as a possible separately requested comparison dataset; it is not a completed local training/evaluation dataset here. Upstream training of CatVTON is separate from this project's use of pretrained weights.

## 7. VITON-HD benchmark pipeline

This is an independent command-line path, not the model serving the current browser upload form.

`config/viton_hd.json` sets batch size 1, test mode, the pilot pair list and **768×1024 native input**. The three released networks provide segmentation (`seg_final.pth`), geometric matching/warping (`gmm_final.pth`) and ALIAS synthesis (`alias_final.pth`).

With permitted source, checkpoints and preprocessed data in place:

```powershell
.\.venv\Scripts\python.exe scripts\verify_viton_assets.py
.\.venv\Scripts\python.exe scripts\run_viton_inference.py
```

The asset verifier checks required checkpoint files, preprocessing directories, pair-list presence and CUDA availability. It does not validate every sample file or provide checksum-based provenance.

`scripts/create_pilot_pairs.py` can create up to ten pairs from the available `test_pairs.txt`; it **rewrites** `test_pairs_pilot.txt`. Do not rerun it casually if preserving an experimental pair list. The current saved list contains six pairs.

The runner names new results `runtime/outputs/viton-hd/baseline_native_1024x768`. Six existing pilot outputs and a contact sheet reside under the historical directory `runtime/outputs/viton-hd/baseline_512`; its name is not proof of the actual synthesis resolution. Current configuration requires native dimensions because the released GMM has fixed-size layers; reducing them can cause checkpoint shape mismatch.

On the 6 GB RTX 4050, close other GPU-heavy processes before running. A passing pilot establishes execution, not full-benchmark or traditional-garment performance. The original research protocol names VITON-HD as the baseline, while the interactive app subsequently uses CatVTON; a future study must explicitly choose and freeze its comparison model rather than silently mixing these conditions.

## 8. Saved outputs and recorded results

### CatVTON job record

```text
runtime/outputs/jobs/<32-character-job-id>/
  person.png       normalized submitted input
  garment.png      normalized submitted reference
  mask.png         thresholded replacement mask
  request.json     seed, category and quality
  inference.log    worker output and errors
  generated.png    raw model output
  result.png       composited user-visible output
  metrics.json     model, resolution, steps, seed, category, peak GPU bytes
```

The generated images may use a different resolution from the stored input PNGs if Standard was selected. Metrics record synthesis resolution. The `adaptation` string in these records describes mask/compositing behaviour; it is **not evidence of learned model adaptation**.

| Saved example | Resolution / steps | Peak allocated GPU memory |
| --- | --- | --- |
| `runtime/outputs/catvton_smoke/metrics.json` | 384×512 / 30 | 2,733,531,136 bytes, approximately 2.55 GiB |
| `runtime/outputs/quality_eaca45fcf5aa4d6e9a33b2422762400f/metrics.json` | 768×1024 / 50 | 3,349,948,416 bytes, approximately 3.12 GiB |

These are individual recorded runs on the RTX 4050, not average benchmarks, quality scores or guaranteed VRAM limits. The counter is reset after pipeline initialization; it is PyTorch allocated-memory accounting, not total system/GPU usage. Existing metrics do not record end-to-end duration.

Earlier visual inspection reported sharper torso detail at higher resolution, but sleeve-length errors and boundary artifacts remained. A trial expanded-arm mask produced worse halos and was not adopted. The user subsequently confirmed masking/generation worked after the full-outfit trouser-gap correction, while reporting longer generation time. No validated speed or quality improvement is claimed from that feedback alone.

Jobs are serialized, but status is kept only in the server's in-memory `STATES` dictionary. After restart, saved files remain while polling old IDs returns `unknown`. There is no automatic output cleanup or job-resume mechanism.

## 9. Tests and manual checks

### Fast automated tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

**Verified on 18 October 2026: 10 tests passed.** The tests cover mask protections, upper-only dress handling, full-outfit trouser-gap filling, invalid input rejection, native input/quality preservation and legacy routing contracts. GPU execution is not required for these unit tests; they do not establish image quality or cultural correctness.

### Manual browser acceptance check

1. Start the server and open the root page, not the legacy `/index.html`.
2. Upload a person and garment, then confirm an automatic mask appears or a clear error permits manual painting.
3. Check that face/hair remain outside the mask. Switch full-outfit categories and verify lower clothing and the gap between trouser legs are editable.
4. Paint, erase and clear; ensure the preview reflects corrections.
5. Run Standard first. Confirm queued/running/completed status, rendered output and PNG download.
6. Inspect the matching job directory and its logs/metrics.
7. If needed, repeat Detailed with the same inputs, seed and mask. Check metrics show 768×1024 and 50 steps; compare quality without assuming improvement.
8. Check failure cases such as an empty mask. Do not use a fully selected image as a valid clothing mask.

### Model/API smoke scripts — optional GPU work

```powershell
# Requires the benchmark pilot assets; writes/reuses catvton_smoke outputs.
.\.venv\Scripts\python.exe scripts\smoke_catvton.py

# Requires the running server and smoke input files; creates a new GPU job.
.\.venv\Scripts\python.exe scripts\smoke_api.py

# Requires the running server and benchmark assets; detects a mask and runs Detailed.
.\.venv\Scripts\python.exe scripts\smoke_detailed_api.py
```

The API smoke scripts have their own shorter polling deadlines (six and five minutes), so a client timeout does not necessarily stop the server worker. Avoid launching repeated GPU tests unnecessarily. These scripts generate outputs; they are not read-only health checks.

`scripts/compare_quality.py <existing-job-id>` copies saved inputs into a `quality_<id>` directory and reruns Detailed. A second positional argument supplies a replacement mask and uses a `_refined` directory. The original job is preserved, but repeating the same comparison overwrites files in its comparison directory.

## 10. Research evaluation

The planned study compares an unchanged pretrained model against a genuinely trained local adaptation using held-out global and traditional samples. This study is **not complete**. Mask settings and higher synthesis resolution are not a substitute for training an adapted model.

Prepared human criteria use 1–5 ratings:

- Visual quality.
- Identity retention.
- Garment colour and print preservation.
- Traditional drape/pleat/front-fall fidelity, where applicable.
- Cultural appropriateness, where applicable and judged by suitable reviewers.

The exporter creates prefilled technical records while leaving subjective ratings blank:

```powershell
.\.venv\Scripts\python.exe scripts\export_evaluation_records.py --output data\evaluation_unscored_2026_10_18.csv --reviewer reviewer_01
```

Use a new filename or first back up reviewed files: the exporter opens its output in write mode and **overwrites existing scores at that path**. Its default output is `data/evaluation_results_prefilled.csv`. `scripts/create_evaluation_template.py` similarly rewrites its fixed template file.

Only completed job folders with readable request/metrics records are exported. Sample IDs are assigned during export and are not stable when the set/order changes; retain `job_id` as the durable linkage. Category-to-family mappings in the exporter are coarse UI-derived labels, not expert garment annotations.

No dataset-level FID, LPIPS, SSIM, independent draping score, significance test or baseline-versus-adapted improvement is established by the current artifacts. Unpaired benchmark outputs cannot be treated as paired ground truth without checking the experimental pairing. Report sample counts, uncertainty and failure examples, not just attractive outputs. See [RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md).

## 11. API and implementation structure

### Routes

| Method / path | Purpose |
| --- | --- |
| `GET /` | Current `frontend/local.html` application |
| `POST /api/detect-mask` | Person PNG + category → mask PNG data URL and notice |
| `POST /api/try-on` | Person, garment, mask PNGs + category, seed, quality → queued job |
| `GET /api/jobs/<id>` | Current in-memory job status, result URL/metrics or error |
| `GET /results/<id>/result.png` | Download final saved result |
| `GET /api/garments` | Legacy seven-item catalogue |
| `GET /index.html` | Historical overlay prototype, not compatible with the current generation request |

The server accepts POST bodies up to 8,000,000 bytes, including JSON and base64 overhead. Accepted categories are `upper-body`, `dress`, `saree`, `kandyan-osariya`; quality is `balanced` or `high`; seed range is 0 through 4,294,967,295. POST success returns HTTP 201. The default seed is 42.

Example request structure (image placeholders must be replaced with actual PNG data URLs):

```json
{
  "person": "data:image/png;base64,...",
  "garment": "data:image/png;base64,...",
  "mask": "data:image/png;base64,...",
  "category": "saree",
  "quality": "high",
  "seed": 42
}
```

The backend is Python's `ThreadingHTTPServer`, serving a no-build HTML/CSS/JavaScript frontend. It binds only to `127.0.0.1:8000`, sends `Cache-Control: no-store`, and uses exclusive Windows port binding to avoid two versions sharing the port. It is a local research server without authentication, production hardening or persistent queue management.

```text
component-1-virtual-tryon/
  README.md
  backend/
    app.py                       server and API routes
    services/generation.py       validated uploads and serialized GPU jobs
    services/mask_detection.py   temporary detection subprocess
    services/tryon_service.py    retained legacy routing prototype
    services/catalogue.py       legacy catalogue loader
  frontend/
    local.html / local.js / local.css   current application
    index.html                         historical prototype
  config/viton_hd.json            independent VITON-HD run configuration
  data/
    garments.json                 legacy catalogue
    garment_references/           synthetic demonstration assets
    person_references/            synthetic demonstration people
    local_sri_lanka/               paired-data collection template
    DATASET_MANIFEST.md
  docs/                           protocol, handoff and saved stop point
  scripts/                        inference, masks, smoke tests and CSV tools
  tests/                          10 current unit tests
  external/CatVTON/                vendored generator source
  external/VITON-HD/               vendored benchmark source and local assets
  runtime/models/                 downloaded model cache
  runtime/uploads/                temporary mask-detection files
  runtime/outputs/                saved jobs, comparisons and benchmark outputs
```

## 12. Troubleshooting and performance

### Generation takes longer

This is expected in several current conditions:

- Detailed uses four times the pixel count of Standard and 50 rather than 30 denoising steps.
- Every generation launches a fresh Python worker and reloads the UNet, VAE, attention weights and safety checker. Cached downloads do not eliminate loading time.
- First use may download large models; CPU mask detection has its own model load/download cost.
- GPU jobs wait for earlier jobs because execution is serialized.
- VAE tiling/slicing trades some execution overhead for reduced memory pressure; other GPU processes can also affect performance.

Use Standard for quick experiments, inspect the active job's `inference.log`, and avoid duplicate submissions. Persistent model workers, timing instrumentation and safe inference optimizations remain future work; no such speed fix is claimed as implemented.

### Failed to fetch / no connection

Ensure the server terminal is still running and the page is opened at `http://localhost:8000`, not directly from disk or on Component 2's port. Read terminal exceptions. If the port is occupied, identify the existing process before stopping anything; do not terminate unrelated work.

### Automatic mask is poor or missing

Use a clearer photo, select the intended category and paint/erase manually. Full-outfit masks need extra draped fabric regions; upper-only long kurtas can extend below the detected top. The parser is not trained here to detect traditional drape boundaries. Detection failures do not imply the generator weights are invalid.

### CUDA or model failure

Use this component's `.venv` executable, check CUDA availability, close unnecessary GPU applications and inspect `runtime/outputs/jobs/<id>/inference.log`. Network/download failures, missing source files, incompatible packages and insufficient memory are distinct causes. Do not solve VITON-HD shape errors by reducing its configured native dimensions.

### Incorrect sleeves, print, identity or drape

Review mask coverage, reference visibility and crop alignment. Outside-mask preservation cannot repair a wrong garment synthesis inside the mask. Identity outside the mask may be retained while other features still change. These are research limitations, not proof that the application failed to run.

### Job disappears after restart

Polling state is in memory only. Existing files remain in the job directory, but the restarted server cannot reconstruct prior status. There is no auto-resume or persistent job browser.

## 13. Privacy, permissions and reproducibility

- Process only photos you have permission to use and follow university ethics requirements. Store consent records separately and use pseudonymous identifiers.
- Generation uploads and outputs **are saved locally** in job directories, unlike the normally temporary detection files. There is no automatic retention/deletion policy. Do not promise that closing the browser deletes them.
- Keep identifiable images, model weights and restricted datasets out of public repositories and shared bundles. Inspect files before publishing evaluation exports.
- CatVTON's vendored documentation identifies CC BY-NC-SA 4.0; VITON-HD identifies CC BY-NC 4.0. Base networks, VAE, parser, data and other dependencies retain their own terms. Check the actual provider terms for intended use; educational use does not remove attribution or other conditions.
- The local wrapper does not pin Hugging Face revisions or record downloaded model checksums automatically. Observed cache snapshot IDs are not equivalent to enforced pins. Future reproducible studies should freeze revisions, hash weights and save the package environment, source changes and hardware details.
- Seeds aid repeatability, but identical results across different library/GPU versions are not guaranteed. Record exact input files, masks, resolution, steps and configuration.
- Saved metrics do not constitute accuracy measurements. Synthetic references must remain labelled synthetic and separate from real-data research claims.

## 14. Remaining work and resume point

When the user explicitly resumes Component 1:

1. Preserve representative outputs and reviewed score files; document current failure cases.
2. Improve the evaluation workflow so re-export does not overwrite scored data; add stable identifiers and stronger run provenance.
3. Agree on the research baseline and collect permitted paired local data with reliable drape labels, subject/garment-separated splits and an independent test set.
4. Implement actual traditional-garment adaptation and train only on training data. Reassess architecture suitability for full saree/Osariya draping rather than assuming an upper-body baseline is sufficient.
5. Compare frozen baseline and adaptation on held-out global/traditional examples with independent blinded ratings and appropriate statistical analysis.
6. Investigate performance with recorded stage timings and a persistent worker design, preserving output correctness and memory safety.
7. Consider recommendation-to-try-on integration only as a separately agreed task.

Not implemented: a specialised Sri Lankan drape model, automatic garment/drape classification in this application, 3D cloth physics, verified fit prediction, video try-on, full quantitative research evaluation or production deployment.

## 15. Supporting documentation and sources

### Local guides

- [Saved stop point](docs/STOP_POINT.md)
- [Research protocol](docs/RESEARCH_PROTOCOL.md)
- [Model/data handoff](docs/MODEL_AND_DATA_HANDOFF.md) — early planning document; its renderer-replacement proposal predates the current CatVTON integration.
- [Dataset manifest](data/DATASET_MANIFEST.md)
- [Paired Sri Lankan dataset collection guide](data/local_sri_lanka/README.md)
- [Synthetic garment reference guide](data/garment_references/README.md)
- [Garment prompts](data/garment_references/PROMPTS.md), [person prompts](data/person_references/PROMPTS.md)
- [Vendored CatVTON documentation](external/CatVTON/README.md)
- [Vendored VITON-HD documentation](external/VITON-HD/README.md)

Older guides describe their writing-time milestones. Current application code, configuration and saved run records take precedence for claims about what actually executes.

### Primary project/model references

- [CatVTON source](https://github.com/Zheng-Chong/CatVTON), [paper](https://arxiv.org/abs/2407.15886), [attention weights](https://huggingface.co/zhengchong/CatVTON)
- [VITON-HD source](https://github.com/shadow2496/VITON-HD), [paper](https://arxiv.org/abs/2103.16874)
- [SegFormer clothes checkpoint](https://huggingface.co/mattmdjaga/segformer_b2_clothes)
- [Base inpainting checkpoint](https://huggingface.co/booksforcharlie/stable-diffusion-inpainting)
- [VAE checkpoint](https://huggingface.co/stabilityai/sd-vae-ft-mse)
