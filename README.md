# Factory Reallocation & Shipping Optimization Recommendation System — Nassau Candy Distributor

## Overview

This project converts historical Nassau Candy Distributor order data into a decision-support system for evaluating product-to-factory reassignment scenarios.

The system:
- predicts shipping lead time,
- compares Linear Regression, Random Forest and Gradient Boosting,
- evaluates route performance,
- simulates each product across five factories,
- ranks factory alternatives,
- exposes the analysis through Streamlit,
- separates modelled operational improvement from financial-risk proxies.

> **Important:** The supplied Order Date and Ship Date fields produce unusually large lead times (904–1,642 days). The project preserves the source data and flags this issue instead of silently correcting it.

## Dataset

- Records: 10,194
- Unique orders: 8,549
- Products: 15
- Factories: 5
- Customer regions: 4
- Ship modes: 4
- Total sales: $141,783.63
- Total gross profit: $93,442.80

Gross Profit reconciles to Sales minus Cost for the supplied rows.

## Factories

| Factory | Latitude | Longitude |
|---|---:|---:|
| Lot's O' Nuts | 32.881893 | -111.768036 |
| Wicked Choccy's | 32.076176 | -81.088371 |
| Sugar Shack | 48.119140 | -96.181150 |
| Secret Factory | 41.446333 | -90.565487 |
| The Other Factory | 35.117500 | -89.971107 |

## Methodology

1. Parse order and shipment dates.
2. Calculate `LeadTimeDays`.
3. Map each product to its supplied current factory.
4. Add approximate factory-to-destination distance using state/province centroids.
5. Encode categorical features and normalize numerical variables.
6. Compare:
   - Linear Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
7. Evaluate with RMSE, MAE and R².
8. Cluster factory-region routes.
9. Simulate all five factories for every product.
10. Rank alternatives using lead-time improvement plus a gross-margin risk proxy.

### Model results

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Linear Regression | 182.8 | 181.6 | 0.524 |
| Random Forest | 186.2 | 175.6 | 0.506 |
| Gradient Boosting | 188.2 | 178.5 | 0.496 |

The selected model is **Linear Regression**. Recommendation coverage is **86.7%** of products, meaning the engine identifies at least one alternative with a lower modelled lead time for that share of products. The average reduction among positive scenarios is **74.2 days**.

## Streamlit Modules

### 1. Overview
- order and lead-time KPIs
- ship-mode comparison
- region comparison
- factory-region operational table

### 2. Factory Optimization Simulator
- product selector
- predicted lead time under all factories
- current vs alternative factory comparison
- speed-priority control

### 3. Recommendation Dashboard
- ranked product-level factory alternatives
- lead-time reduction
- risk flag
- recommendation coverage

### 4. Risk & Model
- model comparison
- RMSE, MAE and R²
- recommendation coverage
- profit-impact limitation

## Files

- `streamlit_app.py` — Streamlit application
- `Nassau_Candy_cleaned.csv` — cleaned/enriched dataset
- `model_metrics.csv` — model evaluation results
- `route_clusters.csv` — route cluster output
- `factory_recommendations.csv` — scenario recommendations
- `Nassau_Candy_Research_Paper.docx` — research paper
- `Nassau_Candy_Executive_Summary.docx` — executive summary
- `Project_Feedback_Video_Script.txt` — video script
- `requirements.txt` — Python dependencies

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The dashboard loads `Nassau_Candy_cleaned.csv` by default and also supports uploading a CSV through the sidebar.

## Important Interpretation Notes

- This is a historical/descriptive and scenario-prediction project, not a guaranteed optimization engine.
- The model error is substantial; recommendations should be validated operationally.
- The source date fields generate unusually long lead times.
- State/province centroid distances are approximate, not carrier route distances.
- No factory-specific freight cost, capacity, inventory, or service-level data is available.
- Therefore exact profit impact cannot be calculated. Gross margin is used only as a risk/exposure proxy.
- A product-level recommendation should not be executed without checking factory capacity, freight contracts, inventory positioning and service-level commitments.

## Suggested GitHub Repository Name

`nassau-candy-factory-optimization`
