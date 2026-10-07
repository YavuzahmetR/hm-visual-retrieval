# H&M Visual Retrieval

A visual search project that retrieves different variants of the same
product family from H&M product images.

A frozen, ImageNet-pretrained ResNet18 extracts visual embeddings.
A lightweight projection head trained with triplet loss adapts these
embeddings to product-family retrieval. Products are ranked by cosine
similarity between normalized embeddings.

On the held-out test split, mAP@10 improved from **0.2628 to 0.4098**,
a **56% relative improvement** over the frozen ResNet18 baseline.

## Final Results

| Test metric | Frozen ResNet18 | Projection head | Improvement |
|---|---:|---:|---:|
| mAP@10 | 0.2628 | **0.4098** | +0.1470 |
| Recall@10 | 0.3417 | **0.5139** | +0.1721 |
| HitRate@10 | 0.6160 | **0.7759** | +0.1600 |

For **6,258 of 8,065 test queries**, the top ten results contained
at least one other variant from the same product family.

The model was selected using validation mAP@10:
**epoch 15**, with a validation score of **0.4164**.
Test results were not used for model selection or training decisions.

Full results are available in `reports/test/metrics.json`
and `reports/test/comparison.csv`.

## Dataset and Splits

Dataset: Kaggle — H&M Personalized Fashion Recommendations.

The experiment uses 23,059 product families with at least two
variants that have available images.

| Split | Images | Product families |
|---|---:|---:|
| Train | 64,912 | 18,447 |
| Validation | 8,110 | 2,306 |
| Test | 8,065 | 2,306 |

The split was created at the product-family level with seed 42.
Each `product_code` appears in exactly one split.
Validation and test families are therefore unseen during training.

A retrieved product is considered relevant when it has the same
`product_code` as the query but a different `article_id`.

This is a proxy task for retrieving product-family variants.
It does not measure human-rated fashion similarity or personalized
recommendation quality.

## Method

1. Freeze ResNet18 ImageNet1K V1 weights and remove its classification layer.
2. Convert images to RGB and apply the pretrained weights' standard
   preprocessing: resize the shorter edge to 256, center crop to
   224×224, and apply ImageNet normalization.
3. Extract and cache L2-normalized, 512-dimensional embeddings.
4. Train a bias-free 512→128 linear projection followed by L2 normalization.
   The head contains **65,536 trainable parameters**.
5. Sample up to 64 product families per batch, with two distinct variants
   from each family. The other same-family variant is the positive;
   the nearest different-family item in the batch is the negative.
6. Optimize triplet margin loss with margin 0.2, AdamW learning rate
   0.001, and weight decay 0.0001.

Each epoch visits every training family once and samples two variants.
It does not process all 64,912 training images in every epoch.

The initial training run lasted five epochs. Training then continued
from the learned weights with a new AdamW optimizer and sampling seed 47.
This continuation did not preserve the original optimizer state.

The maximum epoch was 20, with early-stopping patience set to three.
Training stopped at epoch 18, and validation selected epoch 15.

No CNN fine-tuning or random image augmentation was applied.

## Evaluation Protocol

Validation and test queries are evaluated against their respective
complete galleries. The query image itself is excluded from retrieval.

- **Recall@K:** The fraction of relevant gallery images retrieved
  in the top K, averaged across queries.
- **HitRate@K:** The fraction of queries with at least one relevant
  result in the top K.
- **mAP@K:** Mean average precision, accounting for the positions
  of relevant results. The AP denominator is
  `min(K, number of relevant gallery images)`.

mAP is not classification accuracy.

## Learning Curve and Retrieval Examples

![Learning curve](reports/validation/learning_curve.png)

![Frozen baseline and projection head on the same queries](reports/validation/validation_comparison.png)

The visual comparison uses the same three validation queries for both models.
These examples illustrate retrieval behavior; aggregate metrics measure
overall performance. Incorrect matches remain.

The `test_evaluated: false` field in
`reports/validation/projection_metrics.json` describes the report
at training time. The later, completed final test evaluation is recorded
in `reports/test/metrics.json`.

## Project Files

| File or directory | Purpose |
|---|---|
| `data.py` | Validate the manifest and resolve local image paths |
| `resnet_smoke.py` | Check CUDA and embedding extraction using one image |
| `resnet_baseline.py` | Evaluate the frozen validation baseline |
| `cache_train_embeddings.py` | Build the training embedding cache |
| `train_projection_head.py` | Train the initial five epochs |
| `continue_projection_head.py` | Continue training with early stopping |
| `report_validation.py` | Generate the learning curve and retrieval comparison |
| `evaluate_test.py` | Evaluate the selected checkpoint on the test split |
| `retrieval_metrics.py` | Shared retrieval metric implementation |
| `test_retrieval.py` | Metric tests using small, controlled galleries |
| `models/projection_head_epoch15.pt` | Selected projection head checkpoint |
| `reports/` | Saved metrics, training history, and figures |
| `notebooks/kaggle_data_preparation.ipynb` | Original data preparation and SmallCNN exploration |

SmallCNN was an earlier category-classification experiment.
Its weights are not used by the retrieval model, and its classification
accuracy is not directly comparable to retrieval metrics.

## Running the Project

Tested environment: Windows, Python 3.12, PyTorch 2.6.0+cu124,
Torchvision 0.21.0+cu124, and an NVIDIA RTX 3050 Laptop GPU.

The current model computation scripts require CUDA.

### Environment Setup

Windows Git Bash:

```bash
python -m venv .venv

./.venv/Scripts/python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
./.venv/Scripts/python.exe -m pip install -r requirements.txt

cp project_config.example.json project_config.json
```

Update `data_root` in `project_config.json` to point to your local
dataset directory.

Expected image layout:

```text
<data_root>/images/010/0108775015.jpg
```

Raw data and embedding caches are not included in the repository.
Obtain the images and `articles.csv` from the Kaggle competition.

To generate the original manifest, upload the data preparation notebook
to Kaggle, attach the H&M competition data, and run only
**code cells 2 and 4**. These cells match articles to available images
and create the product-family split.

Download `/kaggle/working/hm_visual/subset_manifest.csv`
and place it at `artifacts/subset_manifest.csv` locally.

### Pretrained ResNet Weights

Download the official ResNet18 weights into the project's model cache:

```bash
./.venv/Scripts/python.exe -c "import torch; from torchvision.models import ResNet18_Weights; torch.hub.set_dir('model_cache/hub'); ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)"
```

### Evaluate the Saved Model

Check the dataset paths, then copy the selected checkpoint to the location
expected by the evaluation script:

```bash
./.venv/Scripts/python.exe ./data.py

mkdir -p results/projection_head_v2
cp models/projection_head_epoch15.pt results/projection_head_v2/best.pt

./.venv/Scripts/python.exe ./evaluate_test.py
```

This workflow extracts test embeddings and computes retrieval metrics.
It does not train the model.

If a completed local test report already exists for the same checkpoint,
the script displays the saved results.

### Reproduce Training

Use a separate, fresh working directory for this workflow:

```bash
./.venv/Scripts/python.exe ./data.py
./.venv/Scripts/python.exe ./resnet_smoke.py
./.venv/Scripts/python.exe ./resnet_baseline.py
./.venv/Scripts/python.exe ./cache_train_embeddings.py --build --workers 2
./.venv/Scripts/python.exe ./train_projection_head.py
./.venv/Scripts/python.exe ./continue_projection_head.py
./.venv/Scripts/python.exe ./report_validation.py
```

Training scripts refuse to overwrite existing experiment directories.
Finalize model selection using validation before evaluating the test split.

Different hardware or package versions may produce different numerical results.

### Metric Tests

```bash
./.venv/Scripts/python.exe -m pip install pytest
./.venv/Scripts/python.exe -m pytest test_retrieval.py -q
```

## Limitations and Future Experiments

Results come from one data split and one training run.
Each evaluation gallery contains only families in that split
with at least two available images.

Possible future experiments include:

- Extracting new training representations with image augmentation.
- Fine-tuning the final ResNet block.
- Analyzing retrieval errors by category.
- Adding an image-upload search demo or API.

These extensions are not part of the reported final results.