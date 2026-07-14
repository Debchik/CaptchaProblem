from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def build_notebook(out_path: str | Path = "notebooks/complex_features.ipynb") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Complex CAPTCHA Features\n\n"
            "## Hypothesis\n\n"
            "Template-mining features from blocks **D1-D4** and SSL sequence embeddings from blocks **E1-E8** "
            "should improve separation between human and bot sessions beyond the original `competition_main.ipynb` "
            "feature family. The primary validation metric for model comparison and ensemble weighting is "
            "`roc_auc_score(y_true, y_pred, max_fpr=0.1)`."
        ),
        nbf.v4.new_markdown_cell(
            "## Notes\n\n"
            "- The notebook keeps the orchestration readable and delegates heavy reusable logic into "
            "`src/complex_feature_pipeline.py`.\n"
            "- The final artifact is written to `outputs/submit_complex.csv`.\n"
            "- Intermediate caches are stored under `outputs/complex_cache/` so repeated runs can resume quickly."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "import torch\n"
            "\n"
            "from src.complex_feature_pipeline import run_full_complex_pipeline"
        ),
        nbf.v4.new_code_cell(
            "ROOT = Path.cwd().resolve()\n"
            "if not (ROOT / 'data').exists():\n"
            "    ROOT = ROOT.parent\n"
            "\n"
            "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
            "TRAIN_PATH = ROOT / 'data' / 'train.parquet'\n"
            "TEST_PATH = ROOT / 'data' / 'test.parquet'\n"
            "UNLABELED_PATH = ROOT / 'data' / 'unlabelled.parquet'\n"
            "CACHE_DIR = ROOT / 'outputs' / 'complex_cache'\n"
            "SUBMISSION_PATH = ROOT / 'outputs' / 'submit_complex.csv'\n"
            "METRICS_PATH = ROOT / 'outputs' / 'complex_metrics.json'\n"
            "\n"
            "print({'device': DEVICE, 'submission_path': str(SUBMISSION_PATH), 'metrics_path': str(METRICS_PATH)})"
        ),
        nbf.v4.new_code_cell(
            "results = run_full_complex_pipeline(\n"
            "    train_path=TRAIN_PATH,\n"
            "    test_path=TEST_PATH,\n"
            "    unlabeled_path=UNLABELED_PATH,\n"
            "    cache_dir=CACHE_DIR,\n"
            "    submission_path=SUBMISSION_PATH,\n"
            "    metrics_path=METRICS_PATH,\n"
            "    device=DEVICE,\n"
            ")\n"
            "results"
        ),
        nbf.v4.new_code_cell(
            "metrics = json.loads(METRICS_PATH.read_text(encoding='utf-8'))\n"
            "pd.DataFrame(metrics['model_scores']).T.sort_values('pauc_01', ascending=False)"
        ),
        nbf.v4.new_code_cell(
            "pd.Series(metrics['weights']).sort_values(ascending=False)"
        ),
        nbf.v4.new_markdown_cell(
            "## Conclusion\n\n"
            "After execution, use the tables above as the experiment summary. The key number to compare across "
            "models is **pAUC@0.1** (`roc_auc_score(..., max_fpr=0.1)`)."
        ),
    ]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
    nbf.write(nb, out_path)
    return out_path


if __name__ == "__main__":
    path = build_notebook()
    print(path)
