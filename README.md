# SignalGap — AI + Spectral Market Intelligence (Public Research App)

**A public, read-only Streamlit showcase of historical financial ML research.**

The app explores a genuine question: do random-matrix spectral features add information beyond a conventional price-only model? It compares 5-day and 20-day benchmark-relative stock-direction forecasts using the project's historical walk-forward outputs.

The answer from the current dataset is **no demonstrated directional advantage**. This is presented openly rather than dressing up a speculative market claim.

## Publish a public URL in 4 steps (no Ubuntu required)

1. Extract the ZIP. Open the extracted folder until you see `app.py`, `analytics.py`, `requirements.txt`, and the `data` folder **together**.
2. Create a **new Public GitHub repository**, preferably `signalgap-market-intelligence`, at https://github.com/new. Do not initialize it with a README or .gitignore since the package already contains files.
3. In the new GitHub repo choose **uploading an existing file** or **Add file → Upload files**. Drag **all the contents** of the extracted folder, including `data/` with its six included files. Commit. Do not upload only the ZIP or the outer folder. Check that `app.py` is at the root and `data/validation_summary.json` is accessible.
4. At https://share.streamlit.io/ choose **Create app** → **Yup, I have an app**, select the repository, `main` branch and `app.py`. Choose a public subdomain, e.g. `signalgap-research` if available, and deploy. Open the resulting `*.streamlit.app` link to test it, then post that exact URL on LinkedIn.

See Streamlit's official deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

### Expected GitHub structure

```text
signalgap-market-intelligence/
├── app.py                         ← Required entrypoint
├── analytics.py                   ← Tested calculation helpers
├── requirements.txt              ← Required dependencies
├── README.md
├── .gitignore
├── .streamlit/config.toml         ← Optional dark theme
├── data/
│   ├── validation_summary.json
│   ├── direction_predictions_5d.csv
│   ├── direction_predictions_20d.csv
│   ├── regime_predictions_5d.csv
│   ├── regime_predictions_20d.csv
│   └── spectral_features_six_asset_reference.csv
└── tests/test_snapshot.py
```

**Avoid repeating the previous missing-file issue:** Don't publish if the `data/` directory is missing.

### Test before publishing

Inside the extracted directory, if Python is available:

```bash
python tests/test_snapshot.py
```

This checks that the on-disk data exactly reproduces the published direction and risk metrics.

To run the app locally with Python and pip:

```bash
pip install -r requirements.txt
streamlit run app.py
```

No Tinker account, external API key, GPU, WSL or local server is needed for Streamlit Cloud hosting.

## What viewers can do

- Switch 5-day versus 20-day evaluation horizons and filter five covered equities.
- Compare stock-direction Brier scores and accuracy against a price-only baseline.
- Explore an asset-by-asset comparison of predictive reliability.
- Adjust confidence thresholds to see historical overconfidence and actual accuracy.
- View historical spectral concentration in a separately labeled six-asset exploratory series.
- Examine the methodology and explicit limitations of the research.
- Download the filtered *historical prediction table* to reproduce the score calculations.

## Why this is not a stock-prediction or investment application

- **No live ANT or other stock forecasts** are generated. Historical data are frozen as of November 20, 2026.
- The directional backtest uses five stocks (AMZN, GOOGL, MSFT, NVDA, ORCL), with SPY outside the feature basket as the benchmark.
- The separate historical spectral visual includes six assets (the five plus SPY) and **was not** the feature set used by the direction models. It is labeled accordingly.
- Tinker / Inkling LoRA training and event extraction are earlier experiments. **They were not inputs to the historical stock-direction results in this release.**
- Sample selection may exhibit survivorship bias. Return labels overlap in time and across stocks; past inference accuracy is not evidence of future alpha.
- No licensed raw price time series is published; bundled tables are project-generated derived evaluation outputs. Confirm rights to publicly redistribute all datasets before publishing.
- Research-only. Not investment advice.

## Security / privacy

The public build contains **no API keys or secrets**, does not call model providers, does not execute orders, and does not allow external uploads. Keep Tinker tokens and private training credentials out of the repository. If you later add APIs, use Streamlit Cloud Secrets rather than storing keys in code.

## Sample LinkedIn text

> I built SignalGap, an experimental market-intelligence research lab combining random-matrix spectral analysis, AI fine-tuning experiments, and walk-forward financial ML. The most interesting finding? Spectral features did **not** improve directional forecasts in my first real-market benchmark. Rather than hide that outcome, the app makes model comparisons, confidence calibration, data provenance, and failure modes interactive. It's a public demonstration of rigorous AI engineering—not a stock-tip engine. [Your deployed URL]

## Run status

The Python module and data-reconciliation tests have passed in the development environment. Full Streamlit UI deployment must be verified on your own Streamlit Cloud instance before sharing publicly.
