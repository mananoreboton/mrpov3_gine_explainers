# mprov3-ui

Zero-dependency HTTP server for browsing GINE and Explainer result sites
locally.

## Usage

```bash
uv run mprov3-ui
uv run mprov3-ui --port 9090
uv run mprov3-ui --no-browser
```

## Routes

| Route | Description |
|-------|-------------|
| `/` | Landing page (dynamically lists available folds) |
| `/gine/` | `mprov3_gine/results/visualizations/` |
| `/classifications/` | `mprov3_gine/results/classifications/` |
| `/explainer/` | Global cross-fold index (`index.html`) or single-fold fallback |
| `/explainer/explainer_summary.html` | Explainer summary — primary comparative view across folds |
| `/explainer/fold_K/` | Per-fold root (redirects to `explanation_web_report/`) |
| `/explainer/fold_K/<path>` | Serves files from `results/folds/fold_K/<path>` |
