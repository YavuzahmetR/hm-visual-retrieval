# H&M Visual Retrieval

Find different variants of the same product family using a catalog item or an
uploaded photo. A **frozen ImageNet ResNet18** produces normalized 512D features;
a learned **512?128 projection head** adapts them to family retrieval. Ranking
uses cosine similarity.

The selected model is an **augmentation-trained continuation of the epoch-15
head**. The CNN remains frozen. The Streamlit demo defaults to this head and
keeps the original head and frozen baseline available for comparison.

## Results

All 8,065 test images serve as both queries and gallery items. Each query's own
article is excluded; relevance means **same `product_code`, different
`article_id`**.

| Test metric | Frozen ResNet18 | Epoch-15 head | Selected augmented head |
| --- | ---: | ---: | ---: |
| mAP@10 | 0.262773 | 0.409823 | **0.417074** |
| Recall@10 | 0.341722 | 0.513853 | **0.523485** |
| HitRate@10 | 0.615995 | 0.775945 | **0.788841** |

The selected head improves mAP@10 by **1.77% relative to epoch 15** and by
**58.72% relative to the frozen baseline**. The top ten contain another
same-family variant for **6,362 / 8,065** queries, compared with 6,258 for epoch 15.

Checkpoint selection used validation only. The original test results had
already been evaluated and published before augmented continuation. The later
comparison reuses that test split; it is **not a new blind test**. Test scores
were not used to choose the continuation epoch or tune its hyperparameters.

Original reports remain in [reports/test](reports/test).
Continuation reports are in [reports/robust_head](reports/robust_head).

## Demo

Run from the project root in Windows Git Bash:

```bash
./.venv/Scripts/python.exe -m streamlit run ./scripts/demo_app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open [http://localhost:8501](http://localhost:8501). If the app is already running
when the code changes, stop it with Ctrl+C and restart it.

- Search 8,065 catalog products by recorded name, colour, pattern or article ID.
- Upload a JPG/PNG, then inspect the five highest-ranked products.
- Switch between the **selected augmented head** and the **original epoch-15
  head**. Compare the chosen head with the frozen 512D baseline.
- Product cards display names, variant attributes, cosine scores and details.
  Catalog queries exclude their own article ID, preserving leading zeros.

CPU is the default. CUDA accelerates uploaded-photo encoding when available;
catalog search uses saved embeddings. Models and galleries are cached per
device and head version. Uploaded-query caches also distinguish the selected
head, so switching models cannot reuse the other head's query embedding.

Uploads accept JPEG/PNG up to **10 MB and 20 million pixels**. Images are decoded
in memory, EXIF orientation is applied, and the original deterministic ImageNet
preprocessing is used. Unreadable images produce an error. Cosine similarity
is a ranking score, **not a probability of correctness**.

### Catalog preview

These original epoch-15 screenshots show **Mom Shorts ? Grey ? Denim**
(`0468480002`, family `0468480`). They document the earlier demo; the current
default is the augmented head. Both original models find two same-family
variants in this selected example. Aggregate metrics describe overall results.

![Catalog query and search controls](reports/demo/catalog_query.png)

<details>
<summary>Original epoch-15 and frozen-baseline results</summary>

![Epoch-15 head results](reports/demo/projection_results.png)

![Frozen baseline results](reports/demo/frozen_results.png)

</details>

### External upload preview

Both heads processed the **same 13 unlabelled photos**, with no decoding errors.
The images' product families were not verified, so **no upload mAP, Recall or
HitRate was calculated**. Nine queries had the same first result under both
heads; this is ranking agreement, not accuracy. These examples illustrate
behavior and limitations, not measured general fashion similarity.

The screenshots below come from the selected head's saved HTML preview and show
its top five results. Long source filenames are hidden; product IDs remain
available in expandable details.

**Denim shirt: visually related shirts appear near the top.**

![Denim shirt upload and the selected head's top-five results](reports/uploads/denim_shirt.png)

**Failure example: a brown jacket retrieves hoodies and unrelated outerwear.**

![Brown jacket upload with mismatched retrieved products](reports/uploads/brown_jacket_failure.png)

**Mixed example: a red dress retrieves a red dress detail at rank three,
alongside different colours and designs.**

![Red dress upload with mixed results](reports/uploads/red_dress.png)

[Full selected-head HTML preview](reports/uploads/preview.html) ?
[Qualitative summary](reports/uploads/summary.json)

Download the HTML file and open it locally; its thumbnails are embedded and it
needs no server. GitHub shows HTML as source rather than embedding this app in
the README. Original photo-to-result mappings remain in local prediction CSVs.

## Dataset and evaluation

Source: Kaggle's H&M Personalized Fashion Recommendations dataset. The subset
contains **23,059 families**, each with at least two available article images.

| Split | Images | Families |
| --- | ---: | ---: |
| Train | 64,912 | 18,447 |
| Validation | 8,110 | 2,306 |
| Test | 8,065 | 2,306 |

Families were split with seed 42; each `product_code` belongs to exactly one
split. Validation/test families do not update model weights. Full catalog
validation uses all 8,110 images; full test uses all 8,065 images.

- **Recall@K:** retrieved relevant images divided by all relevant gallery
  images, averaged across queries.
- **HitRate@K:** queries with at least one relevant result in the first K.
- **mAP@K:** mean average precision, accounting for ranking positions. The AP
  denominator is `min(K, number of relevant gallery images)`.

This is a **product-family variant retrieval proxy**, not personalized
recommendation or human-labelled general fashion similarity. mAP is not
classification accuracy.

## Training and robustness

### Original head

ResNet18 uses ImageNet1K V1 weights, `fc = Identity`, RGB input, shorter-edge
resize to 256, center crop to 224?224 and ImageNet normalization. Its normalized
512D output feeds a bias-free linear 512?128 layer and L2 normalization:
**65,536 trainable parameters**.

The original head trains on cached train features, with up to 64 families per
batch and two distinct variants per family. Positives share a family; negatives
are the closest other-family item in the batch. Training uses triplet margin
loss (`margin=0.2`, `p=2`), AdamW (`lr=1e-3`, `weight_decay=1e-4`). Every epoch
visits each family once, sampling two articles; it does not visit every article.

After five initial epochs, a new AdamW optimizer continued from the learned
weights using sampling seed 47. This was not exact optimizer resume. With a
maximum of 20 epochs and patience three, training stopped at epoch 18;
validation mAP@10 selected epoch 15. This stage used no random augmentation.

![Original head learning curve](reports/validation/learning_curve.png)

[Original validation retrieval examples](reports/validation/validation_comparison.png)

### Controlled robustness benchmark

One source article from each of the 2,306 validation families is evaluated
against the unchanged 8,110-item validation gallery. The source article is
excluded by ID. Five mild changes are applied independently and reproducibly:
lighting, rotation, crop, blur and perspective.

| Query condition | Frozen mAP@10 | Epoch-15 head | Selected head |
| --- | ---: | ---: | ---: |
| Clean | 0.277479 | 0.443098 | 0.451102 |
| Lighting | 0.267721 | 0.406135 | 0.421443 |
| Rotation | 0.252059 | 0.381119 | 0.399355 |
| Crop | 0.256421 | 0.405617 | 0.419478 |
| Blur | 0.246886 | 0.395667 | 0.411619 |
| Perspective | 0.267296 | 0.426168 | 0.432527 |

These are **synthetic validation robustness results**, not real-world upload
accuracy. The family-balanced clean score differs from all-image validation
because its query set contains one article per family. Views of the same source
are correlated.

### Augmented head continuation

The selected experiment starts from epoch 15 with a new AdamW optimizer
(`lr=1e-4`, `weight_decay=1e-4`). Only the 128D head trains. ResNet18 stays frozen
and in evaluation mode, including BatchNorm running statistics.

Training loads images online in batches of **64 families / 128 images**,
using float32. Thirty percent of views stay clean; the others receive a mild
benchmark transformation. A quarter of augmented views also receive a second
distinct transformation. Sampling changes each epoch; hue is preserved.
Existing frozen validation features are reused, but every candidate gets its
own newly projected 128D query/gallery features.

Selection maximizes mean mAP@10 over the five transformed conditions, subject
to clean all-image validation mAP@10 staying at least **0.416447**. Five additional
epochs ran; the fifth was selected. Clean validation mAP@10 became **0.422918**;
transformed mean mAP@10 rose from **0.402941 to 0.416884** (+3.46% relative).

There is no unaugmented continuation control. The observed gain describes
**augmented continuation**, not the isolated causal effect of augmentation.
The published inference checkpoint contains the selected weights and metadata;
full optimizer/RNG states remain in the local experiment. **CNN fine-tuning has
not been performed.**

## Local assets and setup

Tested with Windows, Python 3.12.13, PyTorch 2.6.0+cu124,
Torchvision 0.21.0+cu124, Streamlit 1.65.0 and an RTX 3050 Laptop GPU (4 GB).
The existing training/extraction scripts require CUDA; demo inference also
supports CPU.

For a new environment:

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
./.venv/Scripts/python.exe -m pip install -r requirements.txt
cp project_config.example.json project_config.json
```

Point `data_root` to the existing images, laid out as
`<data_root>/images/010/0108775015.jpg`. For this completed local project, keep the
current environment and extracted data; raw CSVs and the ZIP need not be
downloaded again. `requirements-lock.txt` records the development environment,
including notebook and test tools beyond the runtime requirements.

The demo needs these local assets:

- `artifacts/subset_manifest.csv` and the configured image directory.
- `model_cache/hub/checkpoints/resnet18-f37072fd.pth`.
- `models/projection_head_robust.pt` and, for comparison,
  `models/projection_head_epoch15.pt`.
- `results/test_evaluation/`: frozen 512D features, original head's 128D
  features, row CSV, cache metadata and original metrics.
- `results/robust_head_v1/test_evaluation/`: selected head's 128D features,
  row CSV and completed metrics.

Hashes, dimensions, ID order and encoder/preprocessing metadata are checked.
The demo does not download weights or rebuild missing galleries. Dataset images,
manifest and large caches are excluded from Git; a fresh checkout alone cannot
run the local demo.

For data preparation on a new machine, run only code cells 2 and 4 of
`notebooks/kaggle_data_preparation.ipynb` on Kaggle with the competition data.
Download `subset_manifest.csv` into `artifacts/`. Obtain the official local
ResNet weights using:

```bash
./.venv/Scripts/python.exe -c "import torch; from torchvision.models import ResNet18_Weights; torch.hub.set_dir('model_cache/hub'); ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)"
```

## Commands for saved experiments

Completed results should be read from reports and caches rather than rerun.
Reproduction requires the local assets above; training outputs must be new
directories. These commands are provided for reproducibility, not executed by
the demo.

```bash
# Original cached-feature experiment (fresh workspace)
./.venv/Scripts/python.exe ./scripts/resnet_baseline.py
./.venv/Scripts/python.exe ./scripts/cache_train_embeddings.py --build --workers 2
./.venv/Scripts/python.exe ./scripts/train_projection_head.py
./.venv/Scripts/python.exe ./scripts/continue_projection_head.py
./.venv/Scripts/python.exe ./scripts/report_validation.py
./.venv/Scripts/python.exe ./scripts/evaluate_test.py

# Controlled validation robustness
./.venv/Scripts/python.exe ./scripts/evaluate_query_robustness.py \
  --families 2306 --device cuda --batch-size 16 \
  --output results/query_robustness_full_v1

# Bounded augmented continuation, then its final test comparison
./.venv/Scripts/python.exe ./scripts/train_robust_head.py \
  --families-per-batch 64 --epochs 5 --output results/robust_head_v1
./.venv/Scripts/python.exe ./scripts/evaluate_robust_head_test.py --device cuda
```

The two test evaluators display saved results when already completed. The
robust-head evaluator uses the original frozen cache, computes only the new
head's results, and preserves the original test reports.

To search a folder with the selected head:

```bash
./.venv/Scripts/python.exe ./scripts/evaluate_bulk_upload.py \
  --input results/upload_photos_v1 \
  --checkpoint models/projection_head_robust.pt \
  --device cuda --top-k 10 --output results/upload_selected_v2
```

Use `--device cpu` if needed or `--check-only` for a preflight. Outputs include
`preview.html`, `predictions.csv`, `summary.json` and `errors.csv`. Existing
nonempty output directories are protected. The bulk script's default checkpoint
remains epoch 15; the command above explicitly selects the published robust head.

Verified labels can be supplied with `--labels labels.csv`; without them, no
family-retrieval metrics are calculated:

```csv
filename,product_code,source_article_id
known_product_photo.jpg,0468480,0468480002
```

This is a schema example, not a label for an arbitrary photo. The optional
source article must belong to the labelled family and be in the gallery; it is
excluded when supplied. Every photo needs a verified family present in the
gallery. Visual resemblance or a category label is insufficient.

## Project layout

| Directory | Purpose |
| --- | --- |
| `scripts/` | Data helpers, demo, training, metric functions and evaluations |
| `models/` | Original epoch-15 and selected augmented inference checkpoints |
| `reports/validation`, `reports/test` | Preserved original experiment reports |
| `reports/robust_head` | Continuation config, history, validation and final test reports |
| `reports/uploads` | Clean HTML preview, screenshots and qualitative summary |
| `reports/classification` | Earlier SmallCNN/majority-baseline results, kept separately |
| `notebooks/` | Original Kaggle data preparation and classification exploration |
| `artifacts/`, `results/`, `model_cache/` | Local manifest, experiment caches and weights; ignored by Git |

SmallCNN was a separate classification exploration. Its weights are not used
by retrieval, and its accuracy/F1 are not directly compared with retrieval mAP.

## Limitations and future work

Results use one family split and bounded training runs. No uncertainty estimate
or verified external-photo benchmark is provided. Catalog backgrounds, detail
crops, garment category and colour can affect rankings; wrong matches remain.
External uploads are exploratory, and improvement on every photo is not
expected.

Final-block CNN fine-tuning, alternative head dimensions and a labelled external
query set are possible future experiments. They were not used to select this
model. Deployment and an independent API are outside the current local demo.
