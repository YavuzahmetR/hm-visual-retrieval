# Completed experiments

### Original head

ResNet18 uses ImageNet1K V1 weights, `fc = Identity`, RGB input, shorter-edge
resize to 256, center crop to 224 × 224 and ImageNet normalization. Its normalized
512D output feeds a bias-free linear 512 → 128 layer and L2 normalization:
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

![Original head learning curve](../reports/validation/learning_curve.png)

[Original validation retrieval examples](../reports/validation/validation_comparison.png)

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
