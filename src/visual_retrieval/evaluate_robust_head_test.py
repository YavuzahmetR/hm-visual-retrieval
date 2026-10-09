"""Evaluate the validation-selected robust head once on the existing test split."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from visual_retrieval.data import PROJECT_DIR
from visual_retrieval.demo_retrieval import HEAD_SHA256, _sha256, validate_embeddings
from visual_retrieval.evaluate_bulk_upload import check_assets
from visual_retrieval.retrieval_metrics import evaluate
from visual_retrieval.train_projection_head import ProjectionHead, project_validation

TEST_CACHE = PROJECT_DIR / "results/test_evaluation"


def read_inputs(run: Path) -> tuple:
    """Seçilmiş head'i ve mevcut cache'i kontrol eder; model forward çalıştırmaz."""
    checkpoint_path = run / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    selection = json.loads((run / "metrics.json").read_text("utf-8"))
    config = checkpoint["config"]
    if (
        selection.get("pilot") is not False
        or selection.get("keep_epoch15") is not False
        or selection.get("selected_additional_epoch") != checkpoint["epoch"]
        or config.get("pilot") is not False
        or config.get("encoder_frozen") is not True
        or config.get("head_dim") != 128
        or config.get("parent_sha256") != HEAD_SHA256
    ):
        raise ValueError("Expected a completed, validation-selected 128D robust head.")

    # Upload ve test değerlendirmesi aynı encoder/cache kontrollerini kullanır.
    catalog = check_assets(checkpoint_path)
    baseline = json.loads((TEST_CACHE / "metrics.json").read_text("utf-8"))
    if (
        baseline.get("split") != "test"
        or baseline.get("queries") != len(catalog)
        or baseline.get("product_families") != catalog["product_code"].nunique()
        or baseline.get("selection", {}).get("epoch") != 15
        or baseline["selection"].get("checkpoint_sha256") != HEAD_SHA256
    ):
        raise ValueError(
            "Saved test report does not match the published epoch-15 model."
        )

    clean_map = checkpoint["validation_metrics"]["map@10"]
    robustness_mean = checkpoint["robustness_mean_map10"]
    if (
        not np.isfinite([clean_map, robustness_mean]).all()
        or clean_map < baseline["selection"]["validation_map@10"]
        or clean_map < selection["minimum_clean_map10"]
        or robustness_mean != selection["best_robustness_mean_map10"]
    ):
        raise ValueError("Checkpoint does not match the recorded validation selection.")

    frozen = np.load(TEST_CACHE / "frozen_embeddings.npy", allow_pickle=False)
    return catalog, frozen, checkpoint, baseline, _sha256(checkpoint_path)


def make_comparison(baseline: dict, candidate: dict) -> pd.DataFrame:
    """Eski iki modelin skorlarını rapordan alır; yalnızca yeni head değerlendirilir."""
    previous = baseline["projection_metrics"]
    table = pd.DataFrame(
        {
            "frozen_resnet18": baseline["frozen_metrics"],
            "epoch15_head": previous,
            "robust_head": candidate,
            "delta_vs_epoch15": {
                key: candidate[key] - previous[key] for key in previous
            },
        }
    )
    table.index.name = "metric"
    return table


def print_comparison(table: pd.DataFrame) -> None:
    """Ana metrikleri terminalde gösterir; dosyada @1, @5 ve @10 bulunur."""
    print("\n" + table.loc[["map@10", "recall@10", "hit_rate@10"]].round(6).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=PROJECT_DIR / "results/robust_head_v1"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate files; no inference or output.",
    )
    args = parser.parse_args()
    run = args.run.resolve()
    output = run / "test_evaluation"
    report_path = output / "metrics.json"
    catalog, frozen, checkpoint, baseline, checkpoint_hash = read_inputs(run)
    print(
        f"Test: {len(catalog):,} articles / {catalog['product_code'].nunique():,} families; "
        f"selected additional epoch: {checkpoint['epoch']}."
    )
    print("Previously evaluated test split; not a new blind test. No training.")

    if args.check_only:
        print("Checkpoint, validation selection and original test cache checks passed.")
        return

    # Tamamlanmış sonuç tekrar hesaplanmaz. Farklı checkpoint'e ait sonuç da kullanılmaz.
    if report_path.exists():
        report = json.loads(report_path.read_text("utf-8"))
        if (
            report["selection"]["checkpoint_sha256"] != checkpoint_hash
            or report["cache_signature"]
            != json.loads((TEST_CACHE / "cache_metadata.json").read_text("utf-8"))[
                "signature"
            ]
        ):
            raise ValueError("Saved candidate test result belongs to different assets.")
        print("Using the completed candidate test report; no recalculation.")
        print_comparison(make_comparison(baseline, report["candidate_metrics"]))
        return

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Incomplete output exists; no overwrite: {output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu.")

    started = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    head = ProjectionHead().to(args.device)
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    head.requires_grad_(False).eval()

    # CNN değişmedi: hazır 512D vektörleri yalnızca yeni head'den geçiriyoruz.
    # Eski head'in 128D galerisi bu aday için kullanılmaz.
    print("Projecting the saved 512D test cache with the selected head...", flush=True)
    projected = project_validation(head, torch.from_numpy(frozen).to(args.device))
    validate_embeddings(projected, len(catalog), 128)
    print("Evaluating only the new head; every query excludes itself...", flush=True)
    candidate, rankings = evaluate(projected, catalog["product_code"].to_numpy())
    comparison = make_comparison(baseline, candidate)

    report = {
        "split": "test",
        "queries": len(catalog),
        "product_families": int(catalog["product_code"].nunique()),
        "gallery": "complete test split; query itself excluded",
        "relevance": "same product_code, different article_id",
        "ap_denominator": "min(K, number of relevant gallery images)",
        "test_previously_evaluated": True,
        "note": "Final comparison after validation selection; do not tune from test scores.",
        "selection": {
            "run": run.name,
            "parent_epoch": 15,
            "additional_epoch": checkpoint["epoch"],
            "criterion": checkpoint["config"]["selection"],
            "validation_map@10": checkpoint["validation_metrics"]["map@10"],
            "validation_robustness_mean_map@10": checkpoint["robustness_mean_map10"],
            "checkpoint_sha256": checkpoint_hash,
        },
        "cache_signature": json.loads(
            (TEST_CACHE / "cache_metadata.json").read_text("utf-8")
        )["signature"],
        "frozen_metrics": baseline["frozen_metrics"],
        "epoch15_metrics": baseline["projection_metrics"],
        "candidate_metrics": candidate,
        "metric_deltas_vs_epoch15": comparison["delta_vs_epoch15"].to_dict(),
        "torch_version": str(torch.__version__),
        "projection_device": args.device,
        "seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "projection_embeddings.npy", projected, allow_pickle=False)
    np.save(output / "projection_top10_indices.npy", rankings, allow_pickle=False)
    catalog[["article_id", "product_code"]].to_csv(
        output / "embedding_rows.csv", index=False
    )
    comparison.to_csv(output / "comparison.csv")
    temporary_report = output / "metrics.tmp.json"
    temporary_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary_report.replace(report_path)
    print_comparison(comparison)
    print("\nSaved:", output)
    print("Published epoch-15 model, demo and original test reports are unchanged.")


if __name__ == "__main__":
    main()
