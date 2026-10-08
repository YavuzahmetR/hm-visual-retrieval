# Demo and output review — 2026-10-09

## Available interface

This repository has a local Streamlit application and a bulk-upload CLI. It has no independent HTTP/FastAPI routes, API request model or API response contract.

The Streamlit entry point is `scripts/demo_app.py`. It offers catalog queries, JPEG/PNG uploads, CPU/available CUDA, robust/epoch15 head and frozen/projected comparison. Its documented server address is localhost.

## Measured preservation

The completed local gallery and actual published heads were loaded with old and new source on CPU. Both head versions retained exact embedding and ranking outputs for 24 selected catalog queries/head and 13 saved unlabelled upload photos/head. Catalog results retain ID, family, product metadata, rank and cosine; source IDs are excluded. This is bounded behavior verification, not a fresh accuracy result.

Empty, invalid and oversized uploaded bytes retain the same controlled `ValueError` messages. The existing decoder also rejects non-JPEG/PNG images and excessive pixel count, applies EXIF orientation and preserves deterministic preprocessing. The latter behaviors are source-preservation observations unless specifically present in the machine-readable run record; no new broad coverage claim is made.

## Consumer-facing gaps and decisions

| Item | Actual behavior / next step |
| --- | --- |
| Fresh clone assets | Committed heads/reports are available; images, manifest, encoder weights and large galleries are local and absent from Git. A reviewer must follow `SETUP.md` and supply the exact required assets. This is the main obstacle to immediately running a clone. |
| Model default | Streamlit defaults to robust; bulk CLI defaults to epoch15. The chosen model must be explicit in a shared bulk command. Both behaviors are preserved and documented. |
| Output confidence | Cosine and rank are emitted; no calibrated confidence/probability is claimed. |
| External labels | Unlabelled photo ranking is qualitative. Only verified family/source labels enable Recall/HitRate/mAP in the bulk summary. |
| Output directory | Bulk output must be new or empty; existing results are protected. `--check-only` validates before inference and creates no output. |
| Image failures | Streamlit displays a clear error; bulk records decoding failures in `errors.csv` and records valid/skipped/coverage in `summary.json`. Some images may fail while others succeed. |
| All photos fail | A summary and error CSV can exist without predictions/HTML. Consumers should inspect valid/skipped counts instead of assuming `predictions.csv` always exists. |
| Cache mismatch | Missing/changed/foreign row order, model, encoder metadata or cache signature stops the loader; the demo does not silently rebuild data. |
| Historical environment | The saved cache protocol checks the encoder/preprocessing/runtime signature. A different torch build may require a separately recorded cache reproduction, rather than bypassing the check. |
| HTTP API | Not implemented; response-status integration cannot be claimed or checked here. Current UI/CLI scope is retained. |

## Verification boundary

The recorded comparison does not repeat all 8,065 test queries, training, full CUDA behavior or an interactive browser journey. It checks source/CLI imports and bounded actual CPU inference with unchanged assets. New automated test suites and Docker packaging were not added; ML test-split evaluators and recorded reports remain because they are scientific outputs.

For a reviewer: start the local Streamlit command after the asset checklist, choose a catalog item, compare models, then upload a valid photo and an unreadable file. That manual workflow exposes the actual scope without requiring a software test runner.
