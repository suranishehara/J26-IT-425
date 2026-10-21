# SIMMC 2.1 — official download and fashion-only preparation

Intake completed 19 October 2026. This intake milestone downloaded and audited a published dataset without training or activating a model. **Subsequent native-act training and assessment are now completed**, documented in [SIMMC_NLP_EXPERIMENT.md](SIMMC_NLP_EXPERIMENT.md). The chatbot's current rule parser, knowledge base, Ollama generation and volunteer records are unchanged.

## Source, licence and attribution

- Publisher: [facebookresearch/simmc2](https://github.com/facebookresearch/simmc2), SIMMC 2.1 release.
- Pinned commit: `6e3de4e3d11e2914edc3ab9123e6b0eb88933237`.
- Licence declared by repository: **CC-BY-NC-SA-4.0**. Retain attribution, non-commercial and share-alike conditions; preserve the downloaded full licence. The licence of a pretrained encoder is a separate matter.
- Cite: Kottur, Moon, Geramifard and Damavandi, **SIMMC 2.0: A Task-oriented Dialog Dataset for Immersive Multimodal Conversations**, EMNLP 2021, pp. 4903–4912, [DOI 10.18653/v1/2021.emnlp-main.401](https://aclanthology.org/2021.emnlp-main.401/), and identify the [2.1 release](https://github.com/facebookresearch/simmc2/blob/main/data/README.md) used.

The 2.1 filenames contain data whose top-level `version` still says `simmc_2_dstc10`. This upstream header was retained and recorded, not rewritten. Version attribution is anchored in the publisher's release documentation, filenames and pinned commit.

## Downloaded files

Raw directory: `runtime/datasets/raw/simmc21_official_01/`.

```text
download_manifest.json
LICENSE
README.md
data/
  README.md
  simmc2.1_dials_dstc11_train.json
  simmc2.1_dials_dstc11_dev.json
  simmc2.1_dials_dstc11_devtest.json
  fashion_prefab_metadata_all.json
```

Downloaded source bytes total **79,439,985** (approximately 79.44 decimal MB), excluding the generated manifest. Every downloaded file has a recorded URL, byte count and SHA-256. The metadata's Git-LFS pointer was resolved; its content was checked against the upstream LFS SHA-256 and expected size. Actual metadata has **288 catalogue entries**. Other files are ordinary pinned HTTPS downloads with locally recorded hashes, not independently signed artifacts.

No scene images, scene JSON archives, furniture metadata, mini dialogues, model checkpoints or official `teststd` final-test file were downloaded. The raw mixed-domain dialogue files remain unmodified.

## Prepared fashion subset

Output: `runtime/datasets/simmc21_fashion_01/`.

```text
train.fashion.json
dev.fashion.json
devtest.fashion.json
audit.json
```

The sole content filter is `dialogue.domain == 'fashion'`. Native dialogue IDs, turn order, user/assistant text, act annotations, slot values, requested slots, ambiguity labels/candidates and scene references are preserved. Furniture dialogues are excluded from the derived files, not deleted from the original files. No conversations are randomly redistributed.

| Provider split | Raw dialogues, both domains | Fashion dialogues | Fashion user/assistant turn pairs | Fashion utterances |
|---|---:|---:|---:|---:|
| train | 7,307 | 4,667 | 27,567 | 55,134 |
| dev | 563 | 563 | 3,494 | 6,988 |
| devtest | 1,687 | 1,172 | 6,549 | 13,098 |
| **Total** | **9,557** | **6,402** | **37,610** | **75,220** |

A turn pair consists of one user and one assistant utterance. Thus 75,220 utterances are **not 75,220 independent conversations or participants**. This fetched development file happens to contain fashion only; its mixed-domain top-level header was not used to infer per-dialogue domains.

Partition policy:

- **train:** only partition for fitting model weights.
- **dev:** development selection and error analysis.
- **devtest:** preserve for held-out model assessment; do not fit or tune on it. Intake already inspects schema, counts and label distributions, so it is not a wholly unopened research final test.
- **Official teststd:** not downloaded, inspected, predicted or evaluated. This milestone does not claim to provide its results.

## Labels available for future NLP work

All retained user turns have a native dialogue-act label. Seven native acts occur in training:

| Native act | Training user turns |
|---|---:|
| REQUEST:GET | 9,244 |
| REQUEST:ADD_TO_CART | 4,146 |
| ASK:GET | 3,165 |
| INFORM:REFINE | 2,907 |
| INFORM:DISAMBIGUATE | 2,869 |
| INFORM:GET | 2,700 |
| REQUEST:COMPARE | 2,536 |

Training contains **3,165 ambiguity-labelled user turns**. Slot keys include type, colour (`color`), pattern, brand, size, available sizes, sleeve length, price and customer review. Complete split statistics and label frequencies are in `audit.json`.

**These seven acts are not the app's seven intents.** In particular, this dataset does not automatically provide labels for Sri Lankan occasion advice, greeting, explanation or fabric-property verification. A task definition and reviewed mapping are needed before integrating a learned classifier. Annotated slot values are not automatically token-level NER spans; do not report token F1 without valid span labels and a defined scorer.

## Audit findings and limits

- Duplicate source dialogue IDs within or across downloaded partitions: **zero**.
- Shared normalized fashion user text counts: train/dev **257**, train/devtest **423**, dev/devtest **126**. Normalization is Unicode NFKC, case folding and whitespace normalization; punctuation remains. These are shared distinct texts, not counts of duplicate whole conversations. Short repeated utterances are expected. The provider partitions were preserved, and this is not proof of source/speaker/template independence.
- Dataset flows originate from a simulator with manual paraphrasing. Do not call all records independently volunteered natural conversations.
- Some catalogue attributes are synthetically generated. Prices, brands and reviews are benchmark context, not current stock, buying advice or cultural evidence.
- This is a text-only adaptation. Scene-dependent references remain in the data, but we have not supplied the visual context needed to evaluate the full multimodal tasks. Begin with clearly defined text/dialogue-act tasks and disclose that scope.
- SIMMC is not a Sri Lankan etiquette dataset. It cannot replace expert-reviewed local knowledge or independent Sri Lankan evaluation questions.
- No benchmark accuracy, improvement over rules or cultural-suitability score has been measured yet.

## Reproducibility and checks

Intake script: `scripts/prepare_simmc.py`. It refuses existing output destinations, pins the source commit, validates JSON and metadata LFS integrity, rechecks source hashes before preparation, validates domain/ID/turn structure, and exports unchanged native annotations. A partial failed download may leave a raw folder; do not blindly overwrite it. Inspect it and choose a new versioned destination for a retry.

Re-prepare from verified local files into a fresh directory:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_simmc.py --output runtime\datasets\simmc21_fashion_02
```

Fresh official re-download (new destination names):

```powershell
.\.venv\Scripts\python.exe scripts\prepare_simmc.py --download --raw runtime\datasets\raw\simmc21_official_02 --output runtime\datasets\simmc21_fashion_02
```

**80 automated tests pass**, including four intake tests for filtering/native label preservation, duplicate/invalid record rejection, LFS pointer parsing, source hash guards and preservation of the held-out partition. These are software checks, not NLP accuracy.

## Next major task

The planned native dialogue-act benchmark is now completed; see [the experiment report](SIMMC_NLP_EXPERIMENT.md). Keep native-task results separate from the app's own intents and from Sri Lankan cultural explanation evaluation. Next, define app compatibility and an integration plan before activating any candidate. Importing these dialogues into the RAG knowledge base as cultural facts is not the next step.
