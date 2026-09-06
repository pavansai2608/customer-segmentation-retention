# Customer Segmentation & Retention — Project Deep Dive

> A complete technical audit of this repository. Every design choice, formula, hyperparameter,
> workaround and known weakness is documented here. Nothing is left for the reader to infer.
>
> Audit performed against commit `4890e9d`. All numbers in this document were recomputed from
> the actual data files in the repo, not copied from the README — where they disagree with the
> README, that is called out explicitly.

---

## 1. Executive Summary

Businesses lose revenue when valuable customers stop buying without ever complaining, and
retention budgets get spread evenly across a customer base where most of the value sits in a
small minority. This project answers a single operational question — **"which customers are
worth spending retention money on, and which are not?"** — by combining a *value* estimate
with a *risk* estimate rather than using either alone.

The pipeline takes ~542k raw UK e-commerce transactions, cleans them to ~391k, collapses them
into 4,334 customer-level RFM profiles, segments those with K-Means, predicts each customer's
90-day forward revenue with a BG/NBD + Gamma-Gamma pair, predicts churn probability with
XGBoost, and crosses the two into a four-quadrant action matrix (Retain Immediately / Nurture /
Let Go / Monitor). The result is served by a FastAPI backend and a React dashboard, both
deployed on Render, with DVC + DagsHub for data versioning, MLflow for experiment tracking, and
a locally-hosted Jenkins pipeline for CI.

The intended audience is a retention or CRM analyst who needs a ranked, justified call list —
not a model score. The headline output is a list of **229 customers** who are simultaneously
high-value and high-risk, carrying **£16.5K of predicted 90-day lifetime value** between them.

---

## 2. Dataset

### 2.1 Source and shape

| Property | Value |
|---|---|
| File | `data/raw/online_retail_II.csv` |
| Size on disk | 42 MB |
| Rows (excluding header) | **541,910** |
| Columns | 8 |
| Encoding | `ISO-8859-1` (not UTF-8 — the file contains Latin-1 product descriptions) |
| Date range | **2010-12-01 08:26:00 → 2011-12-09 12:50:00** (~12.3 months) |
| Unique customers (non-blank IDs) | 4,372 |
| Unique invoices | 25,900 |
| Countries | 38 |
| Currency | GBP (£) — single currency, no FX conversion needed |

Provenance: the UCI Machine Learning Repository "Online Retail" family — transactions from a
UK-based, registered, non-store online giftware retailer.

> **⚠️ Naming discrepancy worth knowing before an interview.** The file is named
> `online_retail_II.csv`, but the genuine *Online Retail II* dataset spans **2009-12-01 →
> 2011-12-09** and contains roughly **1,067,371** rows across two sheets. This file has
> **541,910 rows covering only 2010-12-01 → 2011-12-09**, which matches the *original* Online
> Retail (I) dataset. In other words: **the filename says II, the contents are I.** If asked
> "how much data did you train on", the honest answer is ~542k transactions over ~12 months,
> not the ~1M over 24 months that the name implies. Nothing downstream is wrong — the pipeline
> is correct for the data it was given — but the name is misleading.

### 2.2 Columns

| Column | Type | Meaning | Notes |
|---|---|---|---|
| `Invoice` | string | Order/transaction ID | A `C` prefix marks a **cancellation** |
| `StockCode` | string | Product code | Includes non-product codes (postage, fees, manual adjustments) |
| `Description` | string | Product name | 1,454 blanks |
| `Quantity` | int | Units on the line | Negative on returns/cancellations |
| `InvoiceDate` | string | Timestamp, format `%m/%d/%y %H:%M` | Parsed explicitly, not inferred |
| `Price` | float | Unit price in GBP | Zero and negative values present |
| `Customer ID` | float | Customer identifier | **135,080 blanks** — anonymous/guest checkouts |
| `Country` | string | Ship-to country | 38 values, heavily UK-dominated |

### 2.3 Raw-data problems, measured

Every count below was computed directly from the raw file:

| Problem | Rows affected | % of raw |
|---|---:|---:|
| Missing `Customer ID` | 135,080 | 24.93% |
| Missing `Description` | 1,454 | 0.27% |
| Cancellation invoices (`Invoice` starts with `C`) | 9,288 | 1.71% |
| `Quantity` ≤ 0 (returns, adjustments) | 10,624 | 1.96% |
| `Price` ≤ 0 (freebies, bad adjustments) | 2,517 | 0.46% |
| Non-product `StockCode` (postage/fees/manual) | 2,914 | 0.54% |
| `gift_`-prefixed `StockCode` (gift-card top-ups) | 34 | 0.01% |
| Exact duplicate rows | 5,268 | 0.97% |

### 2.4 Exactly how `src/build_decision_matrix.py` cleans each one

All cleaning lives in `load_and_clean_data()`. In order:

```python
df = pd.read_csv(RAW_DATA_PATH, encoding="ISO-8859-1")
```
`ISO-8859-1` is mandatory — UTF-8 decoding raises on the Latin-1 product descriptions.

**1. Drop anonymous transactions**
```python
df = df.dropna(subset=["Customer ID"])
```
Removes all 135,080 blank-ID rows. *Why:* every downstream artefact — RFM, BG/NBD, the churn
label — is keyed on a customer. A transaction with no customer cannot contribute to a
customer-level feature and would silently corrupt aggregate revenue if kept.

**2. Coerce `Quantity` to numeric, then drop non-positive**
```python
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df = df[df["Quantity"] > 0]
```
`errors="coerce"` turns any misaligned/text value into `NaN` rather than raising; the `> 0`
comparison then drops both `NaN` and negatives in one step. This is also what removes
**cancellations** — a `C`-invoice line always carries a negative `Quantity`, so filtering on
quantity handles cancellations implicitly. **There is no explicit `Invoice.startswith("C")`
filter anywhere in the codebase.**

> **Design consequence:** returns are *deleted*, not *netted off*. A customer who bought £1,000
> and returned £900 is recorded as having spent £1,000. Monetary value is therefore **gross
> revenue, not net**. Netting returns would be more accurate, and is a legitimate "what would
> you improve" answer.

**3. Drop non-positive prices**
```python
df = df[df["Price"] > 0]
```
Removes free samples and bad adjustments that would deflate average order value.

**4. Drop non-product stock codes**
```python
JUNK_STOCK_CODES = ["POST","DOT","M","m","D","C2","S","BANK CHARGES","AMAZONFEE","CRUK","PADS"]
df = df[~df["StockCode"].astype(str).str.upper().isin([c.upper() for c in JUNK_STOCK_CODES])]
df = df[~df["StockCode"].astype(str).str.startswith("gift_", na=False)]
```
`POST`/`DOT` = postage, `M`/`m` = manual, `BANK CHARGES`/`AMAZONFEE` = fees, `CRUK` = charity,
`S` = samples, `PADS` = packaging, `D` = discount. Uppercasing both sides makes the match
case-insensitive (which is why both `"M"` and `"m"` in the list are redundant but harmless).
`na=False` on `startswith` prevents `NaN` from propagating as a match. *Why:* postage and bank
fees are not purchase behaviour; leaving them in inflates Frequency and Monetary for customers
who simply paid more shipping.

**5. Drop exact duplicates**
```python
df = df.drop_duplicates()
```
Removes 5,268 fully-identical rows (double-submitted line items).

**6. Parse dates and derive line revenue**
```python
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["TotalPrice"] = df["Quantity"] * df["Price"]
```

> **Divergence from the notebook.** `notebooks/Customer_Segmentation.ipynb` cell 8 uses an
> explicit format: `pd.to_datetime(df["InvoiceDate"], format="%m/%d/%y %H:%M")`. The script
> omits the format and lets pandas infer. Inference on 391k mixed strings is slow and, on
> ambiguous days (`03/04/11`), can silently flip day/month. **Adding the explicit format back
> to the script is a free correctness and speed win.**

**7. Persist**
```python
CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CLEANED_DATA_PATH, index=False)
```

### 2.5 Cleaning outcome, measured

| Stage | Rows | Customers |
|---|---:|---:|
| Raw | 541,910 | 4,372 |
| Cleaned (`data/processed/cleaned_retail.csv`) | **391,150** | **4,334** |
| Removed | 150,760 (**27.82%**) | 38 |

Total cleaned revenue: **£8,737,227.64**.

The 38 lost customers are those whose *entire* transaction history consisted of cancellations,
zero-price lines, or junk stock codes.

---

## 3. Pipeline Walkthrough, Stage by Stage

The canonical implementation is `src/build_decision_matrix.py`, which reproduces the Colab
notebook so the project does not depend on manually re-running notebook cells. Run with:

```bash
python src/build_decision_matrix.py
```

### 3.1 RFM feature construction

**What:** collapse 391k transactions into one row per customer with three behavioural features.

**How** (`compute_rfm()`):
```python
snapshot_date = df["InvoiceDate"].max() + dt.timedelta(days=1)

rfm = df.groupby("Customer ID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,   # Recency
    "Invoice":     "nunique",                                   # Frequency
    "TotalPrice":  "sum",                                       # Monetary
})
rfm.columns = ["Recency", "Frequency", "Monetary"]
```

| Feature | Formula | Units |
|---|---|---|
| Recency | `(snapshot_date − customer's last InvoiceDate).days` | days (lower = better) |
| Frequency | `nunique(Invoice)` | distinct **orders**, not line items |
| Monetary | `sum(Quantity × Price)` | £ gross |

**Reference-date logic — the important bit.** The data ends 2011-12-09. Using the real "today"
would make every customer's recency ~14 years and every customer identically dead. So the
pipeline defines `snapshot_date = max(InvoiceDate) + 1 day = 2011-12-10`, i.e. it pretends
"today" is the day after the dataset ends. The `+1 day` guarantees the most recent customer
gets `Recency = 1` rather than `0`, which keeps the value strictly positive for `qcut` binning
and for any later log transform.

**Why `nunique(Invoice)` and not `len(rows)`:** a single basket of 20 items is one purchase
decision, not 20. Counting rows would reward large baskets as if they were loyalty.

**RFM quintile scores** (carried into the output CSV but **not used as model features**):
```python
rfm["R_Score"] = pd.qcut(rfm["Recency"], q=5, labels=[5,4,3,2,1])
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5, labels=[1,2,3,4,5])
rfm["M_Score"] = pd.qcut(rfm["Monetary"], q=5, labels=[1,2,3,4,5])
rfm["RFM_Score"] = R_Score.astype(str) + F_Score.astype(str) + M_Score.astype(str)
```
- Recency labels are **reversed** (`[5,4,3,2,1]`) because low recency is good.
- Frequency uses `.rank(method="first")` **before** `qcut`. *Why:* Frequency is a small-integer
  distribution with massive ties (thousands of customers have exactly 1 order), and `qcut` on
  tied values raises `ValueError: Bin edges must be unique`. Ranking with `method="first"`
  breaks ties by position, producing 5 evenly-sized bins. This is a real, non-obvious
  workaround and a good interview talking point.
- `RFM_Score` is a 3-char string like `"555"` — kept for human interpretability only.

### 3.2 K-Means clustering

**What:** discover natural customer groups instead of hand-writing RFM score rules.

**How** (`fit_segments()`):
```python
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["Cluster"] = km_final.fit_predict(rfm_scaled)
```

**Why scale first:** K-Means minimises Euclidean distance. Recency spans ~1–374 (days),
Monetary spans ~£3–£280,000. Unscaled, Monetary would be the *only* feature that matters —
the clustering would degenerate into a 1-D split on spend. `StandardScaler` puts all three on
mean 0 / sd 1.

**How K was chosen:** the **Elbow Method**, in notebook cell 25 — fit K = 2…10, record
`km.inertia_` (within-cluster sum of squares), plot, and pick the point where the improvement
flattens. K = 4 was selected. The elbow sweep lives only in the notebook; the script hardcodes
`n_clusters=4` as the settled result.

`n_init=10` runs 10 random initialisations and keeps the best inertia, guarding against a bad
seed. `random_state=42` makes the assignment reproducible.

**Mapping clusters → business labels — the key design choice.**

K-Means cluster *indices are arbitrary and unstable*. The notebook (cell 27) hardcoded the map:
```python
cluster_labels = {2: "Champions", 3: "Loyal Customers", 0: "At Risk", 1: "Hibernating"}
```
This is fragile — any change to the data, the seed, or the sklearn version reshuffles the
indices and silently mislabels every customer. The script fixes this by **deriving the labels
from cluster behaviour**:

```python
cluster_summary = rfm.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean()

rank_score = (
    cluster_summary["Frequency"].rank()
    + cluster_summary["Monetary"].rank()
    - cluster_summary["Recency"].rank()
)
ordered_clusters = rank_score.sort_values(ascending=False).index.tolist()
labels_in_order = ["Champions", "Loyal Customers", "At Risk", "Hibernating"]
cluster_labels = dict(zip(ordered_clusters, labels_in_order))

rfm["Segment"] = rfm["Cluster"].map(cluster_labels)
rfm["Segment"] = rfm["Segment"].fillna("Unclassified")
```

Frequency and Monetary ranks are **added** (higher is better) and Recency rank is **subtracted**
(lower is better), giving a single "goodness" score per cluster. Sorting descending and zipping
against the label list assigns names by behaviour, not by index. This is the single most
defensible engineering improvement in the pipeline over the notebook.

**Actual cluster means from the shipped `final_decision_matrix.csv`:**

| Cluster | Recency (days) | Frequency (orders) | Monetary (£) | rank_score | → Label | Customers |
|---:|---:|---:|---:|---:|---|---:|
| 2 | 7.38 | 81.77 | 125,712.09 | 4+4−1 = **7** | Champions | 13 |
| 3 | 15.40 | 21.97 | 12,240.55 | 3+3−2 = **4** | Loyal Customers | 210 |
| 0 | 43.84 | 3.64 | 1,322.51 | 2+2−3 = **1** | At Risk | 3,045 |
| 1 | 248.54 | 1.54 | 474.12 | 1+1−4 = **−2** | Hibernating | 1,066 |

The ranking reproduces the notebook's hardcoded map exactly — but now it will keep doing so
after a retrain.

> **⚠️ Honest weakness.** These clusters are severely imbalanced: **70.3%** of customers land in
> one cluster ("At Risk"), while "Champions" holds **13 people (0.3%)**. That is the classic
> signature of K-Means on heavy-tailed, right-skewed monetary data — the algorithm spends its
> cluster budget isolating a handful of whale customers instead of partitioning the mass.
> A log or Box-Cox transform on Frequency and Monetary before scaling, or switching to
> Gaussian Mixture / DBSCAN, would produce far more balanced and actionable segments.
> Also note "At Risk" here is a *cluster name*, not a churn statement — a customer can be
> labelled "At Risk" and still have a low churn probability.

### 3.3 BG/NBD model (transaction frequency)

**What it predicts:** how many purchases a customer will make in a future window, and the
probability they are still an active customer at all.

**Library:** `lifetimes.BetaGeoFitter`.

**How** (`fit_ltv_models()`):
```python
bgf_data = summary_data_from_transaction_data(
    df,
    customer_id_col="Customer ID",
    datetime_col="InvoiceDate",
    monetary_value_col="TotalPrice",
    observation_period_end=df["InvoiceDate"].max(),
)

bgf = BetaGeoFitter(penalizer_coef=0.5)
bgf.fit(bgf_data["frequency"], bgf_data["recency"], bgf_data["T"])
```

`summary_data_from_transaction_data` reshapes transactions into the RFM-T form the model needs:

| Field | Meaning (⚠️ *not* the same as the RFM table above) |
|---|---|
| `frequency` | number of **repeat** purchases = total orders − 1 |
| `recency` | age of the customer **at their last purchase**, in days |
| `T` | age of the customer at the end of the observation window, in days |
| `monetary_value` | average spend **per repeat transaction** |

> **Critical distinction for interviews.** `lifetimes` "recency" is *not* "days since last
> purchase" — it is "days between first and last purchase". And `lifetimes` "frequency" excludes
> the first purchase. These differ from the RFM table's `Recency`/`Frequency`, and both live in
> this codebase simultaneously. `backend/main.py` handles the translation explicitly
> (see §4.6).

**Assumptions of BG/NBD (Beta-Geometric / Negative Binomial Distribution):**
1. While active, a customer purchases at a Poisson rate λ.
2. λ is Gamma-distributed across the population (heterogeneity).
3. After every purchase, a customer "dies" with probability p.
4. p is Beta-distributed across the population.
5. λ and p are independent across customers.
6. It models **non-contractual, continuous-time** purchasing — exactly right for retail, where
   nobody cancels a subscription; they just stop coming.

**`penalizer_coef=0.5` — a real workaround.** The notebook comment (cell 33) reads:
> `# Increased penalizer_coef from 0.01 to 0.5 to fix convergence error`

The default L2 penalty of 0.01 caused the maximum-likelihood optimiser to fail to converge on
this dataset (heavy tails + many one-time buyers push the Gamma/Beta parameters toward
degenerate values). Raising the penalty regularises the fit into convergence at the cost of
slightly shrunken parameters.

**Fit on:** *all* customers, including one-time buyers (`frequency = 0`). This is correct —
BG/NBD explicitly models the "bought once, never returned" population.

### 3.4 Gamma-Gamma model (monetary value)

**What it predicts:** a customer's expected *average spend per transaction*, shrunk toward the
population mean for customers with little data.

**How:**
```python
returning_customers = bgf_data[bgf_data["frequency"] > 0]
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(returning_customers["frequency"], returning_customers["monetary_value"])
```

**Why filter `frequency > 0`:** Gamma-Gamma estimates the *distribution* of a customer's order
values. A customer with exactly one order has no observable spread, and `lifetimes` will raise
if `monetary_value` contains zeros (which one-time buyers get). The filter is mandatory, not
optional.

**The core assumption — and the one you will be asked about:**
> **Monetary value is independent of purchase frequency.**

That is, how *often* someone buys carries no information about how *much* they spend per order.
The notebook does not test this. The standard check is
`returning_customers[["frequency","monetary_value"]].corr()` — a Pearson correlation below
roughly |0.1| is conventionally treated as acceptable. **This validation step is missing from
this project and is an honest gap to acknowledge.**

**Combining the two into `predicted_ltv`:**
```python
ltv = ggf.customer_lifetime_value(
    bgf,
    returning_customers["frequency"],
    returning_customers["recency"],
    returning_customers["T"],
    returning_customers["monetary_value"],
    time=3,               # 3 months
    freq="D",             # recency/T are expressed in days
    discount_rate=0.01,   # ~1% monthly discount
)
```

Internally this is a discounted expected-revenue calculation: for each future month *t*,
BG/NBD supplies the expected number of transactions, Gamma-Gamma supplies the expected value of
each, and the product is discounted by `(1 + discount_rate)^t` and summed.

- `time=3` is measured in **months** → a **90-day forward LTV horizon**.
- `freq="D"` tells `lifetimes` that the `recency`/`T` columns it was handed are in days.
- `discount_rate=0.01` applies ~1% monthly time-value-of-money discounting.

**Merging back, and the zero-fill:**
```python
rfm_ltv = rfm.reset_index().merge(ltv_df, on="Customer ID", how="left")
rfm_ltv["predicted_ltv"] = rfm_ltv["predicted_ltv"].fillna(0.0)
```
A `left` join keeps all 4,334 customers; the 1,558 one-time buyers excluded from Gamma-Gamma
come back as `NaN` and are filled with **0.0**.

> **⚠️ Consequence that matters downstream.** **1,558 of 4,334 customers (35.9%) have
> `predicted_ltv = 0.00` exactly.** The LTV median is **£156.66**, so every one of those 1,558
> is automatically classified "Low LTV" by the decision matrix and routed to *Let Go* or
> *Monitor*. Assigning zero value to a one-time buyer is a strong business claim — BG/NBD alone
> could have given them a non-zero expected purchase count. This single `fillna(0.0)` drives a
> third of the final recommendations.

### 3.5 Churn label definition

**There is no churn column in the source data.** The label is *derived* (`add_churn_label()`):

```python
last_date = df["InvoiceDate"].max()                       # 2011-12-09
last_purchase = df.groupby("Customer ID")["InvoiceDate"].max().reset_index()
last_purchase.columns = ["Customer ID", "last_purchase_date"]

rfm_ltv = rfm_ltv.merge(last_purchase, on="Customer ID", how="left")
rfm_ltv["churned"] = ((last_date - rfm_ltv["last_purchase_date"]).dt.days > 90).astype(int)
```

> **A customer is `churned = 1` if they have not purchased in the 90 days before the dataset
> ends** (i.e. no purchase since ~2011-09-10).

**Why 90 days:** it is a conventional retail inactivity window, and it matches the 90-day LTV
prediction horizon so that "value at stake" and "risk of losing it" cover the same period.
It is nonetheless an **arbitrary threshold** — 60 or 180 days would produce a different label,
a different class balance, and different model metrics.

**Measured result: churn rate = 33.34%** (1,445 churned / 2,889 active of 4,334).

> **⚠️ This is a snapshot label, not a survival label.** It cannot distinguish "left forever"
> from "buys twice a year and is due next month". A customer with a genuine 120-day purchase
> cycle is labelled churned every single time. Proper treatment would be survival analysis
> (Cox PH / Kaplan-Meier) or an individualised threshold based on each customer's own
> inter-purchase interval.

**Data-leakage control.** Because `churned` is defined *entirely* from recency, **`Recency` is
deliberately excluded from the feature set.** Notebook cell 40 states it outright:
```python
# Remove Recency — it directly encodes our churn definition (data leakage)
features = ["Frequency", "Monetary", "predicted_ltv"]
```
Leaving `Recency` in would produce a near-perfect ROC-AUC that means nothing — the model would
simply be re-deriving the label. **This is the single most important modelling decision in the
project and the best thing to lead with in an interview.**

### 3.6 Class imbalance handling (SMOTE)

**Balance:** 2,889 not-churned vs 1,445 churned ≈ **2:1**. Not catastrophic, but enough to bias
a classifier toward the majority.

**How** (`train_models()`):
```python
X = rfm_ltv[FEATURE_COLUMNS]
y = rfm_ltv["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

> ✅ **Confirmed from the code: SMOTE is applied strictly AFTER the train/test split, and only
> to `X_train`/`y_train`.** `X_test`/`y_test` remain untouched and retain the true 2:1
> distribution.

This ordering is non-negotiable and is the classic interview trap. SMOTE creates synthetic
minority points by interpolating between a sample and its k-nearest neighbours. If applied
*before* splitting, synthetic points derived from a training row can land in the test set —
the model has effectively seen the test data, and the reported score is inflated and
meaningless. Both `src/build_decision_matrix.py` and notebook cell 40 get this right.

**Also note:** the test set stays imbalanced *on purpose*. Evaluation must reflect the real
2:1 world, not the artificially balanced training world.

`stratify=y` preserves the 33.34% churn rate in both split halves.

### 3.7 XGBoost model (the primary classifier)

```python
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss",
)
xgb_model.fit(X_train_balanced, y_train_balanced)
```

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 100 | Boosting rounds; ample for 3 features / ~4.6k synthetic rows |
| `max_depth` | 4 | Deliberately shallow — with only 3 features, deeper trees memorise |
| `learning_rate` | 0.1 | Standard shrinkage, pairs conventionally with ~100 rounds |
| `random_state` | 42 | Reproducibility |
| `eval_metric` | `"logloss"` | Silences the XGBoost ≥1.3 default-metric warning; proper scoring rule for probabilities |

**Feature contract — exact list and order:**
```python
FEATURE_COLUMNS = ["Frequency", "Monetary", "predicted_ltv"]
```
> **Order is part of the contract.** XGBoost validates by position/name at predict time. Any
> code building an inference DataFrame must reproduce these three columns with these exact
> names. `backend/main.py` hardcodes the same list in `/customer/{id}` and constructs the same
> three keys in `/predict`. **Changing `FEATURE_COLUMNS` in the training script without
> changing the backend breaks inference at runtime, not at import.**

**Split strategy:** 80/20, `random_state=42`, `stratify=y`.

**No hyperparameter search was performed.** There is no `GridSearchCV`/`RandomizedSearchCV`/
Optuna anywhere in the repo. These are sensible defaults, not tuned values — say so honestly.

### 3.8 Logistic Regression (baseline)

```python
lr_model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=1000))
lr_model.fit(X_train_balanced, y_train_balanced)
```

> **Evolution worth noting.** Notebook cell 40 fits a **bare** `LogisticRegression` with no
> scaling. The script wraps it in `make_pipeline(StandardScaler(), ...)`. This matters:
> Monetary spans ~£3–£280,000 while Frequency spans 1–200, so an unscaled LR is dominated by
> Monetary's magnitude and converges poorly. Bundling the scaler *inside* the pipeline also
> means the scaler is fit on training folds only and travels with the pickled model — no
> train/serve skew. This is a genuine improvement made when the notebook was productionised.

`max_iter=1000` (default 100) is needed for lbfgs to converge on this data.

**Additional 5-fold cross-validation, LR only:**
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for train_idx, valid_idx in cv.split(X, y):
    ...
    cv_model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=1000))
    cv_model.fit(X_train_cv, y_train_cv)
    cv_scores.append(roc_auc_score(y_valid_cv, cv_model.predict_proba(X_valid_cv)[:, 1]))
```
> **⚠️ Note:** this CV loop fits on the **raw, unbalanced** `X`/`y` — **SMOTE is not applied
> inside the folds.** So the CV number is not directly comparable to the SMOTE-trained holdout
> score. The statistically correct construction is an `imblearn.pipeline.Pipeline` containing
> SMOTE so that resampling happens inside each fold. Another honest, concrete improvement.

**Why both models were kept — and the surprising result:**

| Model | Test ROC-AUC |
|---|---:|
| Logistic Regression | **0.788** |
| XGBoost | **0.7798** |

**The linear baseline beat the gradient-boosted model.** That is the interesting finding, and
the reason both were retained. With only three features — two of which (`Monetary` and
`predicted_ltv`) are strongly collinear by construction — there is very little non-linear
structure for XGBoost to exploit, so it mostly adds variance. Keeping the baseline is what
*proves* the complex model was not needed; discarding it would have hidden that.

XGBoost is nonetheless the model **deployed** (`xgb_churn_model.pkl` is what
`backend/main.py` loads). `lr_churn_model.pkl` is saved and versioned but never served.

> **Interview-ready framing:** "My baseline outperformed my boosted model by 0.008 AUC. That
> told me the signal in these three features is essentially linear. I kept both, shipped
> XGBoost for headroom as features grow, and treated the near-tie as evidence that the next win
> was in feature engineering, not model complexity."

### 3.9 SHAP (interpretability)

**What it was used for:** explaining *which* features drive the churn prediction, both globally
and per-customer, so a retention analyst can justify a call.

**How** (notebook cell 43 — notebook only; **not in the productionised script**):
```python
import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)
```
`TreeExplainer` is the exact, polynomial-time SHAP algorithm for tree ensembles.

**Result: `predicted_ltv` is the #1 most important feature** (per the README's stated finding).

**Why that makes intuitive business sense — and why it is also a caution:**

*The business reading:* `predicted_ltv` is the only feature carrying **timing** information.
`Frequency` and `Monetary` are undirected lifetime totals — they tell you *how much* someone has
bought but not *when* or *whether they are still alive*. `predicted_ltv` is manufactured by
BG/NBD, whose entire job is estimating P(still active); a collapsing purchase cadence drives LTV
toward zero. So the strongest churn signal is "the model that specialises in aliveness thinks
this person is dying" — which is exactly what you would hope. A high-value customer whose
predicted forward revenue has fallen off a cliff is the textbook retention target.

*The caution:* `predicted_ltv` is a **model output used as a model input**. Errors in BG/NBD
propagate into XGBoost, the two models share the same underlying transaction data, and the
"feature" partially encodes the recency information deliberately excluded in §3.5 (BG/NBD's
`recency` and `T` are recency-derived). So SHAP ranking `predicted_ltv` first is partly a
statement about **model stacking**, not purely about customer behaviour. Worth saying out loud
before someone else points it out.

### 3.10 The decision matrix

**The rule.** In `src/build_decision_matrix.py` → `build_action_column()`:
```python
ltv_median = rfm_ltv["predicted_ltv"].median()

def recommend(row):
    high_risk = row["churn_probability"] > 0.5
    high_ltv  = row["predicted_ltv"] > ltv_median
    if     high_risk and     high_ltv: return {"code": "retain",  "label": "🔴 Retain Immediately"}
    elif   high_risk and not high_ltv: return {"code": "let_go",  "label": "⚪ Let Go"}
    elif not high_risk and     high_ltv: return {"code": "nurture", "label": "🟢 Nurture"}
    else:                                return {"code": "monitor", "label": "🔵 Monitor"}
```

| | **High LTV** (`> median`) | **Low LTV** (`≤ median`) |
|---|---|---|
| **High risk** (`churn_probability > 0.5`) | 🔴 **Retain Immediately** — `retain` | ⚪ **Let Go** — `let_go` |
| **Low risk** (`≤ 0.5`) | 🟢 **Nurture** — `nurture` | 🔵 **Monitor** — `monitor` |

Three columns are written: `action_code` (machine-readable), `action_label` (display), and
`action` (a duplicate of `action_label`, kept for backward compatibility with earlier API
consumers).

**Thresholds:** the risk cut is an absolute **0.5**; the value cut is the **median
`predicted_ltv` = £156.66**, which is *relative* — it guarantees a ~50/50 value split by
construction. Because 1,558 customers sit at exactly `0.0`, the median is pulled well below the
mean.

**Measured distribution:**

| Action | Count | % |
|---|---:|---:|
| ⚪ Let Go | 2,039 | 47.05% |
| 🟢 Nurture | 1,938 | 44.71% |
| 🔴 Retain Immediately | **229** | 5.28% |
| 🔵 Monitor | 128 | 2.95% |

### 3.11 ⚠️ THE DUPLICATED LOGIC — read this before changing anything

**The decision-matrix rule exists in TWO places and is NOT shared:**

| Location | Function | Used for |
|---|---|---|
| `src/build_decision_matrix.py` | `build_action_column()` | Batch scoring of all 4,334 customers → `final_decision_matrix.csv` |
| `backend/main.py` | `recommend_action(churn_probability, predicted_ltv)` | Live scoring of a brand-new customer via `POST /predict` |

Both implement `churn_probability > 0.5` × `predicted_ltv > median` → the same four outcomes,
with the same emoji labels and the same action codes. **There is no shared module. There is no
test asserting they agree.**

They also source the median differently:

```python
# src/build_decision_matrix.py — median of the training population
ltv_median = rfm_ltv["predicted_ltv"].median()

# backend/main.py — median of whatever CSV happens to be loaded, computed at import time
LTV_MEDIAN = float(df["predicted_ltv"].median()) if "predicted_ltv" in df.columns and not df.empty else 0.0
```

**Failure modes:**
1. Change the threshold in one file → `/predict` silently disagrees with `/retain` for the same
   customer profile. No test catches it.
2. If model artefacts fail to load, `LTV_MEDIAN` falls back to **`0.0`**, at which point *every*
   customer has `predicted_ltv > 0` → "High LTV" → every high-risk customer becomes
   *Retain Immediately*. (In practice `MODELS_READY` gates the endpoint first, so this is
   latent rather than live — but it is one refactor away from being real.)
3. The emoji labels are duplicated as string literals in three places (both Python files and
   `ACTION_CODE_MAP` in the backend), so any label edit needs three coordinated changes.

**Fix:** extract a single `recommend_action(churn_probability, predicted_ltv, ltv_median)` into
a shared module imported by both, persist `ltv_median` alongside the model as part of the
artefact, and add a test asserting the two paths agree across a grid of inputs.

---

## 4. Backend (FastAPI)

`backend/main.py` — 287 lines.

### 4.1 Startup and artefact loading

Paths are resolved **relative to the file**, not the working directory:
```python
BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "xgb_churn_model.pkl"
DATA_PATH  = BASE_DIR.parent / "models" / "final_decision_matrix.csv"
BGF_PATH   = BASE_DIR.parent / "models" / "bgf_model.pkl"
GGF_PATH   = BASE_DIR.parent / "models" / "ggf_model.pkl"
```
This is why `backend/` and `models/` must sit **side by side** inside the Docker image
(see §8.3). It also means `uvicorn` can be started from any directory.

`load_artifacts()` loads each artefact in its own `try/except`, appending failures to an
`errors` list rather than raising:
```python
model, df, bgf, ggf, load_errors = load_artifacts()
ESSENTIAL_READY     = model is not None and not df.empty
MODELS_READY        = ESSENTIAL_READY
LTV_MEDIAN          = float(df["predicted_ltv"].median()) if ... else 0.0
USING_LTV_FALLBACK  = bgf is None or ggf is None
```

Two tiers of readiness:
- **Essential** = XGBoost model + decision-matrix CSV. Without these the data endpoints 503.
- **Optional** = `bgf`/`ggf`. Without these `/predict` still works via a heuristic fallback.

All of this executes at **module import time**, which is why `backend/test_main.py` needs the
model files present, and why a missing artefact surfaces on the very first request rather than
at startup.

### 4.2 Endpoint reference

| Method | Path | Returns | Failure behaviour |
|---|---|---|---|
| `GET` | `/` | `{"status": "API is running"}` | never fails |
| `GET` | `/health` | `{status, models_loaded, using_ltv_fallback, load_errors}` | always 200; `status` is `"ok"` or `"degraded"` |
| `HEAD` | `/health` | empty body | added in `31bbddc` for UptimeRobot |
| `GET` | `/segments` | `[{segment, count}]` from `df["Segment"].value_counts()` | **503** if not ready |
| `GET` | `/actions` | `[{action, count}]` from `df["action"].value_counts()` | **503** if not ready |
| `GET` | `/retain` | Top **20** retain-list customers | **503** if not ready |
| `GET` | `/customer/{customer_id}` | Full row + **live-recomputed** churn probability | **503** if not ready, **404** if no match |
| `POST` | `/predict` | Live churn + LTV + action for an unseen customer | **400** validation, **503** not ready, **429** rate limit |

**`/health`** is the diagnostic surface and reports real detail:
```json
{"status":"ok","models_loaded":true,"using_ltv_fallback":false,"load_errors":[]}
```
When artefacts are missing it returns `"status":"degraded"` with the exact `FileNotFoundError`
strings in `load_errors` — this is precisely how the Render deployment failure was diagnosed.

**`HEAD /health`** exists solely because UptimeRobot issues HEAD requests; without it the
monitor received 405 and the keep-warm ping was useless.

**`/retain`**:
```python
retain = df[df["action"].str.contains("Retain", na=False)]
retain = retain.sort_values("predicted_ltv", ascending=False).head(20)
result = retain[["Customer ID","Frequency","Monetary","predicted_ltv",
                 "churn_probability","Segment","action"]].copy()
result["action_code"] = result["action"].map(ACTION_CODE_MAP)
```
A **substring** match on `"Retain"` rather than an equality check on `action_code` — resilient
to emoji/label drift but fragile if a future label ever contains the word "Retain" in another
sense. Capped at **20** rows. `ACTION_CODE_MAP` back-fills `action_code` because older CSVs
lacked the column.

**`/customer/{customer_id}`** does **not** just look up the stored score — it re-runs inference:
```python
row = df[df["Customer ID"] == int(customer_id)]
if row.empty: return JSONResponse(status_code=404, content={"error": "Customer not found"})

FEATURE_COLUMNS = ["Frequency", "Monetary", "predicted_ltv"]
features = row[FEATURE_COLUMNS]
live_churn_prob = float(model.predict_proba(features)[0][1])

result = row.iloc[0].to_dict()
result["churn_probability"] = live_churn_prob
```
The stored `churn_probability` is overwritten by a fresh prediction, so swapping in a retrained
`.pkl` changes the API's answer immediately without regenerating the CSV. Note `Customer ID` is
stored as a **float** (`13093.0`) in the CSV; the `int(customer_id)` comparison works because
pandas compares numerically.

### 4.3 Rate limiting on `/predict`

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("10/minute")
def predict_customer(input: CustomerInput, request: Request):
```
**10 requests per minute, keyed on client IP**, returning **429** on exceed. `/predict` is the
only endpoint doing real model work on user-supplied input, so it is the only abuse surface.

> **Mechanical gotcha:** `slowapi` requires the endpoint signature to contain a parameter
> literally named `request: Request`. Remove or rename it and the decorator raises at runtime.
> This is why `request` appears unused in the function body.

> **Deployment caveat:** behind Render's proxy, `get_remote_address` sees the proxy IP unless
> `X-Forwarded-For` is honoured — so in production the limit may apply globally rather than
> per-client.

### 4.4 Validation in `/predict`

Order of checks is deliberate:

```python
try:
    first = datetime.strptime(input.first_purchase_date, "%Y-%m-%d")
    last  = datetime.strptime(input.last_purchase_date,  "%Y-%m-%d")
except ValueError:
    return JSONResponse(400, {"error": "Dates must be in YYYY-MM-DD format"})

if not MODELS_READY:
    return JSONResponse(503, {"error": "Model artifacts are not available"})

today = datetime.now()
if last < first:            return 400 "last_purchase_date cannot be before first_purchase_date"
if last > today:            return 400 "last_purchase_date cannot be in the future"
if input.total_orders <= 0: return 400 "total_orders must be greater than 0"
if input.total_spent < 0:   return 400 "total_spent cannot be negative"
```

> **Note the ordering:** the date-format check runs **before** the `MODELS_READY` gate. That is
> what allows `test_predict_rejects_bad_date_format` to pass in a CI environment with no model
> files. Deliberate or not, it is load-bearing for the test suite.

Pydantic (`CustomerInput`) handles type coercion; the dates are typed `str` and parsed manually
so the API can return a friendly message instead of Pydantic's 422.

### 4.5 Graceful degradation

**Tier 1 — essential artefacts missing.** Every data endpoint returns:
```python
return JSONResponse(status_code=503, content={"error": "Model artifacts are not available"})
```
The process still starts and `/health` still answers — so an orchestrator can distinguish
"container is dead" from "container is up but unusable". This is exactly the state the Render
deployment was found in.

**Tier 2 — `bgf`/`ggf` missing.** `/predict` falls back to a hand-rolled heuristic:
```python
def estimate_predicted_ltv(first_purchase_date, last_purchase_date, total_orders, total_spent, today):
    if total_orders <= 0 or total_spent < 0:
        return 0.0
    avg_order_value = total_spent / total_orders
    frequency       = max(total_orders - 1, 0)
    recency_days    = max((last_purchase_date - first_purchase_date).days, 0)
    age_days        = max((today - first_purchase_date).days, 1)

    base_ltv        = avg_order_value * (frequency + 1) * 1.2
    recency_factor  = max(0.0, 1 - (recency_days / age_days))
    return round(max(base_ltv * (0.6 + recency_factor * 0.4), 0.0), 2)
```
A crude "average order value × repeat count × 1.2, scaled 0.6–1.0 by how recently they bought".
The `max(..., 1)` on `age_days` prevents division by zero for a same-day customer.
`/health` advertises the degraded state via `using_ltv_fallback: true`.

The BG/NBD path is *also* wrapped so a runtime failure degrades rather than 500s:
```python
try:
    predicted_ltv = float(ggf.customer_lifetime_value(...).iloc[0])
except Exception as exc:
    logger.warning("LTV calculation failed, using fallback: %s", exc)
    predicted_ltv = estimate_predicted_ltv(...)
```
And BG/NBD is skipped entirely when `lifetimes_frequency == 0`, because Gamma-Gamma is
undefined for a customer with no repeat purchase (§3.4).

### 4.6 How `/predict` builds its inference DataFrame

This is where the two different "frequency"/"recency" conventions are reconciled:

```python
frequency_rfm  = input.total_orders            # RFM Frequency  = total orders
monetary_rfm   = input.total_spent             # RFM Monetary   = total spend

lifetimes_frequency = max(input.total_orders - 1, 0)      # repeat purchases only
lifetimes_recency   = (last - first).days                 # first→last span, NOT days-since
T                   = (today - first).days                # customer age
avg_order_value     = input.total_spent / input.total_orders

features = pd.DataFrame([{
    "Frequency":     frequency_rfm,
    "Monetary":      monetary_rfm,
    "predicted_ltv": predicted_ltv,
}])

churn_prob = float(model.predict_proba(features)[0][1])
action     = recommend_action(churn_prob, predicted_ltv)
```

A **single-row DataFrame with exactly the three trained column names, in order** — matching
`FEATURE_COLUMNS` from §3.7. Passing a bare list or a dict with different keys would either
throw or, worse, silently reorder features.

> **⚠️ Real modelling bug in live inference.** `T = (today - first).days` uses
> `datetime.now()` — the *real* current date. The BG/NBD model was fit with `T` measured
> against **2011-12-09**. In 2026, a customer whose first purchase was "one year ago" gets a
> plausible `T`, but the model's learned Gamma/Beta parameters were calibrated on a population
> whose `T` maxed out at ~374 days. Any input with a multi-year history produces a `T` far
> outside the training range, and BG/NBD will confidently declare the customer dead. This is
> unflagged covariate shift between training and serving, and it is the most substantive
> technical criticism available of the live `/predict` path.

Response payload:
```json
{"frequency": 9, "monetary": 4820.0, "predicted_ltv": 169.0,
 "churn_probability": 0.213, "action": {"code": "nurture", "label": "🟢 Nurture"},
 "action_code": "nurture", "action_label": "🟢 Nurture"}
```
`action` is returned as a nested object *and* flattened into `action_code`/`action_label` —
redundancy retained for older frontend versions.

### 4.7 Why `dill` instead of `joblib` for bgf/ggf

```python
joblib.dump(xgb_model, MODELS_DIR / "xgb_churn_model.pkl")   # fine
joblib.dump(lr_model,  MODELS_DIR / "lr_churn_model.pkl")    # fine

# bgf/ggf contain lambdas from the lifetimes library's internal optimizer,
# which plain pickle/joblib cannot serialize — use dill instead.
with open(MODELS_DIR / "bgf_model.pkl", "wb") as f: dill.dump(bgf, f)
with open(MODELS_DIR / "ggf_model.pkl", "wb") as f: dill.dump(ggf, f)
```

`lifetimes` fitters retain references to **lambda functions** created during
`scipy.optimize.minimize` (the negative-log-likelihood closure). Python's `pickle` — and
therefore `joblib`, which builds on it — serialises functions **by qualified name**, and a
lambda has no importable name, producing:
> `PicklingError: Can't pickle <function <lambda> at 0x...>`

`dill` serialises function **bytecode**, so closures and lambdas survive. The backend mirrors
this asymmetry on load:
```python
model = joblib.load(MODEL_PATH)                                   # XGBoost
with open(artifact_path, "rb") as f: loaded = dill.load(f)        # bgf / ggf
```
**Consequence:** `dill` is a hard runtime dependency of the backend, pinned as `dill==0.3.9`.
Because dill embeds bytecode, these two artefacts are also the most Python-version- and
library-version-sensitive files in the project — which is exactly what commits `5db7f1b`
("match pandas/numpy to training environment"), `4d91504`, `6fa68d6` and `1263f69`
("re-save bgf/ggf models with compatible library versions") were fighting.

### 4.8 CORS

```python
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://127.0.0.1:3000,http://localhost:3000"
).split(",")

app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])
```
Comma-separated env var, defaulting to local CRA dev. **The deployed frontend's exact origin
must be listed or every browser request is blocked** — note that an origin includes scheme and
has no trailing slash.

### 4.9 Logging

`logging.basicConfig(level=logging.INFO)` with a named logger. Successful lookups and
predictions log at INFO; rejected input and artefact-load failures log at WARNING. Notably the
prediction log records *inputs and outcome* but never a customer identifier for `/predict`,
which is reasonable from a PII standpoint.

---

## 5. Frontend (React)

`frontend/my-app/` — Create React App, **React 19.2.7**, `react-scripts` 5.0.1.

### 5.1 Structure

- **`src/App.js` — 939 lines, one default-exported `App()` component.**
  There is **no router**, no `src/components/` directory, no state library, no `src/api/`
  module. Presentational sub-components (`Tag`, `Stat`, `Panel`, `Field`, `RiskReadout`,
  `Readout`, `CategoryChart`, `ChartTooltip`, `EmptyState`, `ErrorNote`, `Skeletons`, and an
  `Icon` map of inline SVGs) are declared in the same file above `App()`.
- **`src/App.css` — 808 lines**, a CSS-variable design system: `:root` holds the light-mode
  token set, `[data-theme="dark"]` overrides the same names. No CSS framework, no CSS-in-JS.
- `src/index.js` — `ReactDOM.createRoot` + `<React.StrictMode>`.
- `src/index.css` — font loading fallbacks and global resets.
- `src/__tests__/App.test.js` — 47 lines, 2 tests.

**Dependencies:** `axios` (HTTP), `recharts` 3.9.2 (charts), `@testing-library/*` (tests),
`web-vitals`. No Redux, no React Query, no form library.

### 5.2 What it renders

Sections, in DOM order, all mounted simultaneously in one scroll container:

1. **Sidebar rail** — brand, five nav items (Overview / Distribution / Lookup / Predict /
   Retain list), and a model-status indicator driven by `loadError`.
2. **Hero (`#overview`)** — the retain count as an oversized figure, plus Revenue exposure,
   Customers scored, and Avg. churn probability.
3. **Distribution (`#analysis`)** — two horizontal Recharts `BarChart`s (`layout="vertical"`)
   for `/segments` and `/actions`, with `Cell` fills mapped per category and `LabelList` values.
4. **Customer lookup (`#lookup`)** — numeric input + Search button; renders a churn readout
   (percentage, Low/Medium/High meter) and a detail list.
5. **Score a new customer (`#predict`)** — four inputs (two dates, orders, spend) + submit;
   renders the same readout shape from the `/predict` response.
6. **Retain list (`#retain`)** — a scrollable table of the 20 rows from `/retain`.

**State and data flow** — all `useState`/`useEffect` in `App()`, no external store:
- One `useEffect` on mount runs `Promise.all([...segments, ...actions, ...retain])`; any
  rejection sets `loadError`.
- A second `useEffect` debounces customer lookup by **300 ms** and uses an `AbortController`
  (held in `lookupAbortRef`) to cancel in-flight requests, guarding against out-of-order
  responses.
- A third `useEffect` drives an `IntersectionObserver` scrollspy that highlights the active nav
  item. It is guarded with `typeof IntersectionObserver === "undefined"` so it no-ops under
  jsdom in tests.

**Theme toggle:** `data-theme` is written to `document.documentElement` and persisted to
`localStorage` under the key **`csr-theme`**; it defaults to `"dark"`.

**Form defaults:** the lookup field is pre-filled with customer **`13093`** (the top of the
retain list) and the prediction form with a one-year-ago → today date range, 1 order and £0
spend, so both tools return output on first use.

### 5.3 `REACT_APP_API_URL` — build-time only

```javascript
const API = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";
```

**Create React App inlines `REACT_APP_*` variables into the bundle at build time.** Webpack's
`DefinePlugin` performs a literal text substitution during `npm run build`; the string
`process.env.REACT_APP_API_URL` does not survive into the shipped JavaScript at all — it is
replaced by the value that was set when the build ran.

Consequences:
- **Setting it at runtime does nothing.** There is no `process` object in a browser. Changing
  the env var on the Render service and restarting the container has **zero effect** on an
  already-built static bundle; the service must be **rebuilt**.
- It must be supplied as a Docker **build arg** (`docker-compose.yml` does exactly this) or as
  a build-time environment variable in Render's build step.
- The value must be reachable **from the user's browser**, not from inside the Docker network —
  which is why `docker-compose.yml` passes `http://localhost:8000` and not `http://backend:8000`.
  This is called out in a comment in the compose file.

> **⚠️ Live configuration bug found during this audit.** `REACT_APP_API_URL` is defined in
> **`frontend/.env`**, but CRA only reads `.env` files from the **application root**, which is
> `frontend/my-app/`. **`frontend/.env` is never read.** `frontend/my-app/.env` contains only
> `PORT=3000`. The app therefore falls back to the hardcoded `http://127.0.0.1:8000` for any
> local `npm start`. This is silent and was masked for a long time because the fallback happened
> to be correct. **Fix: move the `REACT_APP_API_URL` line into `frontend/my-app/.env`.**
> (`frontend/my-app/.env.example` already documents the correct value.)

---

## 6. MLOps: DVC + DagsHub + MLflow

### 6.1 Why large files are not in git

Git stores a full compressed copy of **every version** of every file. Committing an 42 MB CSV
five times means a permanently ~200 MB repository — every clone pays for it forever, and GitHub
warns above 50 MB / hard-rejects above 100 MB per file. DVC stores only a small text pointer in
git and pushes the actual bytes to external storage.

**Remote** (`.dvc/config`, committed):
```ini
[core]
    remote = origin
['remote "origin"']
    url = https://dagshub.com/pavansai2608/customer-segmentation-retention.dvc
```
**Credentials** (`.dvc/config.local`, gitignored, never committed):
```ini
['remote "origin"']
    auth = basic
    user = <dagshub username>
    password = <dagshub token>
```

A `.dvc` pointer looks like:
```yaml
outs:
- md5: 39f6e4ec2a09d6aca27ee2dcb906690f
  size: 44002285
  path: online_retail_II.csv
```
DVC also auto-writes a `.gitignore` in each tracked directory so git ignores the real file.

### 6.2 Current tracking state (as of `4890e9d`)

| Path | Tracked by |
|---|---|
| `data/raw/online_retail_II.csv` (42 MB) | **DVC** |
| `data/processed/cleaned_retail.csv` (37 MB) | **DVC** |
| `models/*.pkl`, `models/final_decision_matrix.csv` (920 KB total) | **git** (changed in `4890e9d`) |

> **This changed during the audit period.** The five serving artefacts were originally
> DVC-tracked and pulled at container start. Because they total under 1 MB and a failed pull
> silently produced a modelless production service (§8.5), they were moved into git and are now
> baked into the Docker image. **DVC now covers the 79 MB of data only.** The `dvc pull` in
> `entrypoint.sh` is retained as a fallback for the case where an artefact is genuinely absent.

### 6.3 The retraining sequence — all four steps are mandatory

```bash
python src/build_decision_matrix.py     # 1. regenerate artefacts

dvc add data/processed/cleaned_retail.csv   # 2. update the pointer + local cache
dvc push                                    # 3. UPLOAD THE ACTUAL BYTES to DagsHub
git add data/processed/cleaned_retail.csv.dvc
git commit -m "retrain: ..."                # 4. commit the pointer
git push
```

### 6.4 The exact failure mode when `dvc push` is skipped

This is the single most valuable operational lesson in the repo, and it **actually happened**:

1. `dvc add` computes a new MD5, rewrites the `.dvc` pointer, and copies the file into the
   **local** `.dvc/cache`.
2. You `git commit` + `git push` the pointer. **Everything looks correct locally** — `dvc
   status` is clean, the app runs, tests pass, because your local cache has the blob.
3. Deployment clones the repo, reads the pointer, asks DagsHub for hash `39f6e4…`, and DagsHub
   has never seen it.
4. `dvc pull` fails with a **`missing-files`** error.
5. `entrypoint.sh` swallows it (`|| echo "WARNING: dvc pull failed"`) and starts uvicorn anyway.
6. The container comes up **healthy but empty**: `/health` returns
   `{"status":"degraded","models_loaded":false,"load_errors":["model: [Errno 2] No such file..."]}`
   and every data endpoint returns **503**.
7. The frontend's `Promise.all(...).catch()` cannot distinguish 503 from a network failure and
   shows *"Couldn't reach the API. Is the backend running?"* — pointing the operator at the
   wrong problem entirely.

**Diagnostic command:** `dvc status --cloud`. Any file listed as `new:` exists locally but is
**not on the remote** — that is the smoking gun.

> **⚠️ Current outstanding issue.** `dvc status --cloud` currently reports all 7 tracked files
> as `new:`, and `dvc push` fails with **HTTP 401 Unauthorized**; the DagsHub API returns
> **403** for the stored token. **The DagsHub token in `.dvc/config.local` is revoked or
> expired.** Model serving is unaffected (artefacts now ship in the image), but no data can be
> pushed or pulled until a new token is generated.

### 6.5 ⚠️ SECURITY: a live credential is committed to this repository

**`notebooks/Customer_Segmentation.ipynb`, cell 15, contains a hardcoded DagsHub token in
plaintext:**
```python
!dvc remote modify origin --local user pavansai2608
!dvc remote modify origin --local password 6d31874d…   # 40-char token, in the committed notebook
```
It is present in the current `HEAD` and in history via commits `302d24f`
("docs: add notebook with RFM, LTV, and churn modeling pipeline") and `fbc5b43`
("updated notebook"). Commit `520ee5d` — *"fix: remove hardcoded DagsHub credentials from
Jenkinsfile"* — shows the problem was recognised in the Jenkinsfile but **the notebook was
missed**. This is very likely why the token now returns 403: automated secret scanning revokes
exposed credentials.

**Required remediation:**
1. Revoke the token in DagsHub (assume it is compromised regardless of the 403).
2. Strip it from the notebook and replace with an env-var read.
3. Purge it from git history (`git filter-repo` or BFG) — deleting it in a new commit does
   **not** remove it from history.
4. Add a pre-commit hook (`detect-secrets`, `gitleaks`) so it cannot recur.
5. Consider `nbstripout` so notebook outputs and secrets stop entering git at all.

### 6.6 MLflow experiment tracking

Notebook cell 46 logs runs to DagsHub's hosted MLflow server:
```python
import mlflow, dagshub
dagshub.init(repo_owner="pavansai2608", repo_name="customer-segmentation-retention", mlflow=True)
mlflow.set_experiment("churn-prediction")

with mlflow.start_run(run_name="Logistic Regression Baseline"):
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("features", "Frequency, Monetary, predicted_ltv")
    mlflow.log_param("smote", True)
    mlflow.log_metric("roc_auc", 0.788)
    mlflow.log_metric("accuracy", 0.71)
    mlflow.log_metric("churn_recall", 0.80)
    mlflow.sklearn.log_model(lr_model, "logistic_regression_model")
```
`dagshub.init(mlflow=True)` points `MLFLOW_TRACKING_URI` at the DagsHub-hosted server, so no
MLflow server needs to be run locally. Params, metrics, and the serialised model are logged.

Experiments: <https://dagshub.com/pavansai2608/customer-segmentation-retention/experiments>

> **⚠️ Note:** the metrics above are **hardcoded literals**, not computed variables — e.g.
> `mlflow.log_metric("roc_auc", 0.788)` rather than `roc_auc_score(y_test, lr_proba)`. If the
> model is retrained and the score changes, MLflow will keep logging 0.788. Also, MLflow logging
> exists **only in the notebook** — `src/build_decision_matrix.py` prints metrics to stdout and
> logs nothing. Wiring MLflow into the script would make the tracking real rather than
> decorative.

---

## 7. CI/CD (Jenkins)

### 7.1 The trigger chain

```
git push  →  GitHub webhook  →  ngrok tunnel  →  local Jenkins  →  pipeline runs
```

1. Push to `main` on GitHub.
2. GitHub fires a webhook POST to `https://<ngrok-domain>/github-webhook/`
   (content type `application/json`).
3. A persistent ngrok tunnel with a **fixed dev domain** forwards it to `localhost:8080`:
   ```bash
   nohup ngrok http --url=https://porthole-unvocal-upstate.ngrok-free.dev 8080 > ngrok.log 2>&1 &
   ```
   The fixed domain is the important detail — a random ngrok URL changes on every restart and
   the GitHub webhook would need reconfiguring each time. `nohup … &` detaches it so it survives
   closing the terminal (it only stops on a full machine restart).
4. The Jenkins job has **"GitHub hook trigger for GITScm polling"** enabled and runs the
   `Jenkinsfile`.

### 7.2 Why Jenkins locally instead of GitHub Actions

Both the git history and the pipeline design point to one reason: **the CI needs the real model
and data files, and they are not in the repo.** GitHub Actions runs on ephemeral cloud runners
that would have to `dvc pull` ~80 MB from DagsHub on every single run — slow, dependent on
DagsHub uptime, and requiring credentials in CI secrets. The local Jenkins host already has a
populated checkout on disk, so the files can simply be **mounted**. Secondary benefits: no
minute quotas, and hands-on experience with a tool still dominant in enterprise environments.

### 7.3 Pipeline stages

```groovy
environment {
    DAGSHUB_CREDS = credentials('dagshub-token')
    LOCAL_REPO_WITH_DATA = "${HOME}/Desktop/customer-segmentation-retention"
}
options { timestamps(); disableConcurrentBuilds() }
```
`disableConcurrentBuilds()` matters because both stages bind-mount the *same* host directory —
concurrent runs would race on it. `credentials('dagshub-token')` pulls the token from Jenkins'
credential store rather than the file (the fix in `520ee5d`).

**Stage 1 — Backend: install & test**
```groovy
docker run --rm -u root \
    -v "$WORKSPACE":/workspace \
    -v "$LOCAL_REPO_WITH_DATA/models":/workspace/models \
    -v "$LOCAL_REPO_WITH_DATA/data":/workspace/data \
    -w /workspace \
    python:3.12-slim ./run_backend_tests.sh
```
where the generated script is:
```sh
pip install -r requirements.txt
cd backend
pytest -v --junitxml=test-results.xml
```
- `python:3.12-slim` — note this differs from the **`python:3.11-slim`** in
  `backend/Dockerfile`. **CI tests on 3.12; production runs 3.11.**
- `-u root` avoids permission errors writing into the mounted workspace.
- `models/` and `data/` are mounted from the host's pre-populated checkout, overlaying the
  workspace copies.
- `--junitxml` output is consumed by `post { always { junit 'backend/test-results.xml' } }` so
  Jenkins renders per-test results even on failure.

**Stage 2 — Frontend: install, test & build**
```groovy
environment { REACT_APP_API_URL = 'http://127.0.0.1:8000' }
docker run --rm -e REACT_APP_API_URL -v "$WORKSPACE":/workspace -w /workspace \
    node:20-alpine ./run_frontend.sh
```
```sh
cd frontend/my-app
npm ci
CI=true npm test -- --watchAll=false
npm run build
```
- **`npm ci`** (not `npm install`) — installs strictly from `package-lock.json` for a
  reproducible tree, and fails if lockfile and manifest disagree. Commits `56a6119`
  ("Regenerate frontend lockfile for npm ci") and `1f29fe7` exist precisely because of this.
- `CI=true` makes CRA's Jest run once instead of entering watch mode, **and promotes warnings
  to errors** during `npm run build` — so an unused import fails the build.
- `REACT_APP_API_URL` is passed because the build bakes it in (§5.3), even though the CI build
  artefact is discarded.

**Cleanup:** `post { always { sh 'rm -f run_backend_tests.sh run_frontend.sh' } }`.

### 7.4 Why the backend stage avoids a network `dvc pull`

The `Jenkinsfile` comment states it directly:
> *"This is where models/ and data/ actually have real file content (pulled via `dvc pull` once,
> manually, outside of CI). We mount those directories straight into the container so tests
> don't depend on a network dvc pull succeeding during every single CI run."*

A network pull per run would make CI slower, non-deterministic, dependent on DagsHub
availability, and would require the DagsHub token inside every build. Given `backend/main.py`
loads artefacts at import time (§4.1), a failed pull would fail the entire suite for reasons
unrelated to the code under test.

> **⚠️ The trade-off, honestly stated.** `LOCAL_REPO_WITH_DATA` is a **hardcoded absolute path
> to one developer's Desktop**. The pipeline is unrunnable on any other machine, cannot be
> reproduced by a collaborator, and CI silently tests against whatever happens to be sitting in
> that directory — including uncommitted local changes. This is the weakest link in the CI
> design.

---

## 8. Docker & Deployment

### 8.1 `docker-compose.yml` vs `docker-compose.dev.yml`

| | `docker-compose.yml` (prod-like) | `docker-compose.dev.yml` (development) |
|---|---|---|
| Frontend Dockerfile | `Dockerfile` (multi-stage → **nginx**) | `Dockerfile.dev` (CRA dev server) |
| Frontend port | `3000:80` (nginx on 80) | `3000:3000` (webpack dev server) |
| `REACT_APP_API_URL` | passed as a **build arg** (baked in) | passed as a **runtime env var** (dev server reads it live) |
| Source code | **copied** into the image at build | **bind-mounted** (`./src`, `./public`) for hot reload |
| Backend `models/` | inside the image | bind-mounted from host |
| Backend command | `entrypoint.sh` | `uvicorn … --reload` |
| Healthcheck | yes, on `/health` | none |
| `depends_on` | `condition: service_healthy` | plain `depends_on` |
| `restart` | `unless-stopped` | none |

The prod compose gates the frontend on the backend actually passing its healthcheck:
```yaml
depends_on:
  backend:
    condition: service_healthy
```
so the UI never comes up against a backend that is still loading artefacts.

### 8.2 nginx configuration (`frontend/my-app/nginx.conf`)

```nginx
gzip on;
gzip_types text/plain text/css application/javascript application/json image/svg+xml;
gzip_min_length 1024;

location /static/ { expires 1y; add_header Cache-Control "public, immutable"; }
location /       { try_files $uri $uri/ /index.html; }
```
`/static/` is cached for a year because CRA content-hashes those filenames. The `try_files`
fallback to `index.html` is the standard SPA rewrite so deep links do not 404.

### 8.3 `backend/Dockerfile` and its path assumptions

```dockerfile
# Build context: project root. Run as: docker build -f backend/Dockerfile .
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt dvc

# main.py resolves paths as BASE_DIR.parent / "models", so backend/ and
# models/ must sit side by side inside the image, same as on disk.
COPY backend/ ./backend/
COPY models/ ./models/
COPY .dvc/ ./.dvc/
COPY .dvcignore ./

WORKDIR /app/backend
COPY backend/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/health')" || exit 1
CMD ["./entrypoint.sh"]
```

Key points:
- **Build context must be the repo root** (`docker build -f backend/Dockerfile .`), because the
  Dockerfile copies `requirements.txt`, `models/`, and `.dvc/` from above `backend/`.
- The resulting layout is `/app/backend/` + `/app/models/`, with the working directory
  `/app/backend`, so `BASE_DIR.parent / "models"` resolves to `/app/models` — matching §4.1.
- **`libgomp1`** is the GNU OpenMP runtime, a hard shared-library dependency of the XGBoost and
  scikit-learn wheels. Without it, `import xgboost` fails on slim images with
  `libgomp.so.1: cannot open shared object file`. Non-obvious and easy to miss.
- **`git`** is installed because DVC requires a git repository to operate.
- The `HEALTHCHECK` uses stdlib `urllib` rather than `curl`, which is not present in slim
  images.
- `.dockerignore` excludes `data/`, `notebooks/`, `src/`, `frontend/`, `.dvc/cache/`,
  `__pycache__/` and `.git` — keeping the image small and, importantly, keeping the 79 MB of
  training data out of it.

### 8.4 What `entrypoint.sh` does at container start

```sh
#!/bin/sh
set -e

if [ -f ../models/xgb_churn_model.pkl ] && [ -f ../models/final_decision_matrix.csv ]; then
    echo "Model artifacts present in image — skipping dvc pull."
elif [ -n "$DAGSHUB_USER" ] && [ -n "$DAGSHUB_TOKEN" ]; then
    dvc remote modify origin --local auth basic
    dvc remote modify origin --local user "$DAGSHUB_USER"
    dvc remote modify origin --local password "$DAGSHUB_TOKEN"

    if [ ! -d ../.git ]; then
        cd ..
        git init -q
        git config user.email "deploy@render.local"
        git config user.name "Render Deploy"
        git add .dvc models/*.dvc .dvcignore 2>/dev/null || true
        git commit -q -m "init for dvc" --allow-empty
        cd backend
    fi

    dvc pull -v || echo "WARNING: dvc pull failed — models may be missing"
else
    echo "WARNING: DAGSHUB_USER/DAGSHUB_TOKEN not set — skipping dvc pull"
fi

exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
```

1. **Artefact check first** (added in `4890e9d`) — if the models are already baked into the
   image, skip DVC entirely. This makes startup network-independent.
2. **Credential injection** — writes `DAGSHUB_USER`/`DAGSHUB_TOKEN` into `.dvc/config.local` at
   runtime so the secret is never committed or baked into the image.
3. **`git init` inside the container** — `.dockerignore` excludes `.git`, but **DVC refuses to
   operate outside a git repository**. So a throwaway repo is created with an empty commit
   purely to satisfy DVC. This is commit `7c592a9`, *"fix: initialize git repo in container so
   dvc pull works on Render"* — a genuinely non-obvious deployment workaround.
4. **`|| echo`** so a failed pull warns rather than killing the container (`set -e` would
   otherwise abort). This is what allowed the "healthy but modelless" state in §6.4.
5. **`exec uvicorn`** — replaces the shell as PID 1 so SIGTERM reaches uvicorn and Render can
   shut the container down cleanly.
6. **`${PORT:-8000}`** — Render injects `$PORT` and the service must bind it.

### 8.5 Render.com deployment

| Service | URL |
|---|---|
| Frontend (static site) | `https://customer-segmentation-retention-1-x35i.onrender.com` |
| Backend (FastAPI) | `https://customer-segmentation-retention-lfyh.onrender.com` |
| Swagger docs | `…-lfyh.onrender.com/docs` |

- **Free tier**, both services.
- **Cold starts:** the instance spins down after ~15 minutes idle; the next request takes up to
  **50 seconds**.
- **UptimeRobot** pings `/health` every **5 minutes** to keep it warm — which is why
  `HEAD /health` had to be added (`31bbddc`).
- **CORS:** `ALLOWED_ORIGINS` on the backend service must contain the frontend's exact origin.
- **`REACT_APP_API_URL`** must be set at frontend **build** time (§5.3).

> **Gotcha:** **renaming a Render service changes its display name but *not* its
> `.onrender.com` subdomain**, which is fixed at creation. The random-looking suffixes
> (`-1-x35i`, `-lfyh`) are permanent unless the service is deleted and recreated. Any hardcoded
> URL keeps working after a rename — and any assumption that the URL follows the name is wrong.

### 8.6 Deployment incident recorded in this repo

Verified live during this audit and preserved as a case study:

```
GET /health → {"status":"degraded","models_loaded":false,
               "load_errors":["model: [Errno 2] No such file or directory:
               '/app/models/xgb_churn_model.pkl'", …]}
GET /segments → 503 {"error":"Model artifacts are not available"}
```
Frontend, API URL, and CORS were all correct; the backend was up and answering. Root cause:
`dvc pull` failed (revoked token, §6.5), so the container had no models. **Resolution
(`4890e9d`):** the five artefacts (920 KB) were moved into git and are now copied into the
image, `entrypoint.sh` gained the artefact check, and the live backend returned
`models_loaded: true` ~75 seconds after the push.

---

## 9. Testing

### 9.1 Backend — `backend/test_main.py` (58 lines, 4 tests)

**Import mechanism** — `main.py` is loaded by file path, not by package import:
```python
BACKEND_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("main", BACKEND_DIR / "main.py")
main = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(main)
```
There is no `__init__.py` and `backend/` is not an installed package, so a plain `import main`
would depend on `sys.path` and break under different pytest invocations. This makes the suite
runnable from anywhere. **Side effect:** executing the module runs `load_artifacts()`, so the
model files must be present — which is why CI mounts them (§7.3).

```python
@pytest.fixture
def client():
    return TestClient(main.app)      # starlette TestClient, requires httpx
```
(`httpx==0.28.1` is in `requirements.txt` solely for this — commit `0f7987b`.)

| # | Test | Asserts |
|---|---|---|
| 1 | `test_predict_rejects_bad_date_format` | `POST /predict` with `"bad-date"` → **400** and the exact message `"Dates must be in YYYY-MM-DD format"` |
| 2 | `test_health_reports_status` | `GET /health` → **200** and the body contains a `status` key |
| 3 | `test_segments_endpoint_returns_data` | `GET /segments` → **200**, payload is a **non-empty list** |
| 4 | `test_predict_accepts_valid_input` | `POST /predict` with a valid profile → **200**, and `action_code ∈ {retain, let_go, nurture, monitor}` |

Test 1 passes even without model files (§4.4). Tests 2–4 require the artefacts.

> **Coverage gaps, honestly:** nothing covers `/actions`, `/retain`, `/customer/{id}` (neither
> the 200 nor the 404 path), the rate limiter, the `estimate_predicted_ltv` fallback, the
> 503-when-not-ready branch, or — most importantly — **that `recommend_action()` in the backend
> agrees with `build_action_column()` in the pipeline** (§3.11). There are no unit tests at all
> for `src/build_decision_matrix.py`.

### 9.2 Frontend — `src/__tests__/App.test.js` (47 lines, 2 tests)

`axios` is fully mocked; `window.matchMedia` is stubbed in `beforeAll` (jsdom does not implement
it and the theme initialiser calls it); `localStorage` is cleared.

| # | Test | Asserts |
|---|---|---|
| 1 | `renders the dashboard title` | `findByText(/Segmentation & Retention/i)` is in the document |
| 2 | `looks up a customer when the search button is clicked` | With `/customer/42` mocked, typing `42` and clicking **Search** renders `/At Risk/i` |

A **smoke test**: it proves the app mounts, fetches, and completes one user journey. It does not
assert chart contents, theme switching, error states, or responsive behaviour.

> **Note for future edits:** test 2 queries `/At Risk/i` with `findByText`, which throws if
> **more than one** element matches. Any new copy containing the words "at risk" anywhere in the
> rendered DOM will break it. This has already happened once.

**Run:** `CI=true npm test -- --watchAll=false`

---

## 10. Key Results & Numbers

Every concrete figure in the codebase, recomputed from the shipped artefacts.

### Data

| Metric | Value | Source |
|---|---:|---|
| Raw transactions | 541,910 | `data/raw/online_retail_II.csv` |
| Cleaned transactions | 391,150 | `data/processed/cleaned_retail.csv` |
| Rows removed by cleaning | 150,760 (27.82%) | computed |
| Date range | 2010-12-01 → 2011-12-09 | computed |
| Raw unique customers | 4,372 | computed |
| **Final customers modelled** | **4,334** | `final_decision_matrix.csv` |
| Unique invoices (raw) | 25,900 | computed |
| Countries | 38 | computed |
| Total cleaned revenue | **£8,737,227.64** | computed |
| Missing `Customer ID` rows | 135,080 (24.93%) | computed |
| Cancellation rows (`C` invoices) | 9,288 | computed |
| `Quantity ≤ 0` rows | 10,624 | computed |
| `Price ≤ 0` rows | 2,517 | computed |
| Duplicate rows | 5,268 | computed |
| Junk stock-code rows | 2,914 (+34 `gift_`) | computed |

### Segments (K-Means, K = 4)

| Segment | Customers | % | Mean Recency | Mean Frequency | Mean Monetary |
|---|---:|---:|---:|---:|---:|
| At Risk | 3,045 | 70.26% | 43.84 d | 3.64 | £1,322.51 |
| Hibernating | 1,066 | 24.60% | 248.54 d | 1.54 | £474.12 |
| Loyal Customers | 210 | 4.85% | 15.40 d | 21.97 | £12,240.55 |
| Champions | 13 | 0.30% | 7.38 d | 81.77 | £125,712.09 |

### Modelling

| Metric | Value |
|---|---|
| Churn definition | no purchase in the **90 days** before 2011-12-09 |
| **Churn rate** | **33.34%** (1,445 churned / 2,889 active) |
| Features (exact order) | `Frequency`, `Monetary`, `predicted_ltv` |
| Excluded feature | `Recency` (leakage — encodes the label) |
| Train/test split | 80/20, `random_state=42`, `stratify=y` |
| Imbalance handling | SMOTE, `random_state=42`, **train set only** |
| **Logistic Regression ROC-AUC** | **0.788** |
| **XGBoost ROC-AUC** | **0.7798** |
| LR accuracy (MLflow) | 0.71 |
| LR churn recall (MLflow) | 0.80 |
| XGBoost params | `n_estimators=100`, `max_depth=4`, `learning_rate=0.1`, `eval_metric="logloss"` |
| LR params | `StandardScaler` + `max_iter=1000` |
| Cross-validation | 5-fold `StratifiedKFold` (LR only, **no SMOTE in folds**) |
| Top SHAP feature | **`predicted_ltv`** |
| BG/NBD penalizer | **0.5** (raised from 0.01 to fix convergence) |
| Gamma-Gamma penalizer | 0.01 |
| LTV horizon | `time=3` months (**90 days**), `discount_rate=0.01`, `freq="D"` |
| Model deployed | **XGBoost** (`xgb_churn_model.pkl`) |

### Decision matrix

| Metric | Value |
|---|---|
| Risk threshold | `churn_probability > 0.5` |
| Value threshold | `predicted_ltv > median` |
| **Median `predicted_ltv`** | **£156.66** |
| Customers with `predicted_ltv = 0` | **1,558 (35.95%)** |
| `churn_probability` min / mean / max | 0.0022 / 0.4338 / 0.9392 |
| 🔴 Retain Immediately | **229** (5.28%) |
| 🟢 Nurture | 1,938 (44.71%) |
| ⚪ Let Go | 2,039 (47.05%) |
| 🔵 Monitor | 128 (2.95%) |
| Retain-list revenue exposure | **£16.5K** predicted LTV |
| `/retain` endpoint cap | top **20** by `predicted_ltv` |

> **⚠️ README discrepancy:** the README states **"226 customers flagged as Retain Immediately"**.
> The shipped `final_decision_matrix.csv` contains **229**. The README figure predates the last
> retrain (commit `a7f17bf`, *"Improve action codes and retrain pipeline"*) and was never
> updated. **Quote 229, not 226.**

### Code size

| File | Lines |
|---|---:|
| `frontend/my-app/src/App.js` | 939 |
| `frontend/my-app/src/App.css` | 808 |
| `backend/main.py` | 287 |
| `src/build_decision_matrix.py` | 282 |
| `Jenkinsfile` | 79 |
| `backend/test_main.py` | 58 |
| `frontend/my-app/src/__tests__/App.test.js` | 47 |
| `notebooks/Customer_Segmentation.ipynb` | 49 cells |

---

## 11. Known Limitations & What I'd Improve

Ordered roughly by severity. These are the honest answers to *"what would you do differently?"*

### Critical

1. **A live credential is committed to the repository.** A DagsHub token sits in plaintext in
   `notebooks/Customer_Segmentation.ipynb` cell 15 and in git history (§6.5). It must be
   revoked, stripped, purged from history, and prevented with a `gitleaks`/`detect-secrets`
   pre-commit hook. The token now returns 403, which is consistent with automated revocation
   after exposure.

2. **The decision-matrix rule is duplicated and unsynchronised** (§3.11). `build_action_column()`
   and `recommend_action()` implement the same logic in two files with no shared module and no
   test asserting agreement. *Fix:* one shared function, `ltv_median` persisted as part of the
   model artefact, and a property test comparing both paths across a grid.

3. **Training/serving skew in `/predict`'s `T`.** `T = (today - first).days` uses the real
   current date, while BG/NBD was fit against a 2011 horizon where `T` never exceeded ~374 days
   (§4.6). Live predictions run far outside the training distribution. *Fix:* anchor inference
   to the training snapshot date, or retrain on recent data.

### High

4. **`predicted_ltv = 0` for 36% of customers** by `fillna(0.0)` (§3.4). One line drives a third
   of all recommendations. *Fix:* use BG/NBD's expected purchases × population-mean order value
   for one-time buyers instead of zero, or model them as an explicit segment.

5. **Severely imbalanced clusters** — 70% in one segment, 13 people in another (§3.2). *Fix:*
   log-transform Frequency/Monetary before scaling, or use a Gaussian Mixture; validate with a
   silhouette score, which is currently never computed.

6. **The churn label is arbitrary and non-survival.** A fixed 90-day window cannot distinguish
   a lost customer from a slow one (§3.5). *Fix:* survival analysis, or a per-customer threshold
   derived from their own inter-purchase interval.

7. **Returns are dropped rather than netted.** Monetary is gross revenue; a customer who
   returned everything still looks valuable (§2.4). *Fix:* match `C` invoices to originals and
   subtract.

8. **Gamma-Gamma's independence assumption is never tested** (§3.4). One `.corr()` call would
   validate or invalidate the whole LTV layer.

9. **The `churn_probability` stored in the decision matrix is in-sample.** The script scores the
   full dataset with a model trained on 80% of it — the code says so explicitly in a comment.
   It is fine for a business-facing ranking but must never be quoted as a performance metric.

10. **CI depends on a hardcoded path to one developer's Desktop** (`LOCAL_REPO_WITH_DATA`,
    §7.4). The pipeline is unreproducible anywhere else and can test against uncommitted local
    files. *Fix:* a small committed fixture dataset, or a DVC pull with a cached remote.

### Medium

11. **`REACT_APP_API_URL` lives in a file CRA never reads** (§5.3) — `frontend/.env` instead of
    `frontend/my-app/.env`. Silent, and masked only because the hardcoded fallback happened to
    match.

12. **Single 939-line frontend component with no router** (§5.1). No deep-linking to a section,
    no code splitting, everything re-renders together, and the file is hard to test in units.
    *Fix:* split into `components/`, add React Router, extract an `api.js`.

13. **Free-tier cold starts** — up to 50 s on first request after 15 minutes idle (§8.5).
    UptimeRobot mitigates but does not eliminate it. *Fix:* a paid instance, or a frontend
    skeleton/warming state that sets expectations.

14. **The frontend cannot distinguish 503 from a network failure.** A backend answering
    "models unavailable" is reported to the user as *"Couldn't reach the API. Is the backend
    running?"*, which sends the operator down the wrong path — as happened in §8.6. *Fix:*
    branch on `error.response?.status`.

15. **MLflow metrics are hardcoded literals**, and MLflow is wired only into the notebook, not
    the script (§6.6). The tracking is decorative rather than functional.

16. **No hyperparameter tuning anywhere** (§3.7). Defaults only — worth stating plainly rather
    than implying the values were optimised.

17. **CV is run without SMOTE inside the folds** (§3.8), so the CV score is not comparable to
    the holdout score. *Fix:* `imblearn.pipeline.Pipeline`.

18. **Python version drift:** CI tests on `python:3.12-slim`, production runs `python:3.11-slim`
    (§7.3). Given that `dill` artefacts are bytecode-sensitive, this is a real risk.

### Low

19. **Test coverage is thin** — 4 backend tests, 2 frontend tests, zero tests for the pipeline
    (§9). No coverage of `/retain`, `/actions`, `/customer/{id}`, the rate limiter, or the LTV
    fallback.
20. **The dataset filename misrepresents its contents** (§2.1) — named *II*, contains *I*.
21. **README numbers have drifted** — "226 Retain Immediately" vs the actual 229 (§10).
22. **Emoji are embedded in API payloads** (`"🔴 Retain Immediately"`). Presentation leaking into
    the data layer; the frontend has to strip them for display.
23. **`action` duplicates `action_label`** in the CSV and API responses — dead weight kept for
    backward compatibility.
24. **`src/build_decision_matrix.py` needs `imbalanced-learn`, which is not in
    `requirements.txt`** — that file covers only the serving path, so a clean environment cannot
    retrain without a separate install.
25. **No model monitoring** — nothing detects drift, and nothing triggers a retrain.
26. **`Customer ID` is stored as a float** (`13093.0`) throughout the CSV.
27. **Notebook outputs are committed**, inflating diffs; `nbstripout` would help.
28. **`build_log.txt` and `ngrok.log` are committed** to the repository root as build artefacts.

---

## 12. File-by-File Map

### Machine learning

| Path | Purpose |
|---|---|
| `src/build_decision_matrix.py` | **Canonical pipeline.** Cleans raw data → RFM → K-Means → BG/NBD + Gamma-Gamma → churn label → SMOTE → LR + XGBoost → decision matrix. Writes all five model artefacts and the final CSV. |
| `notebooks/Customer_Segmentation.ipynb` | 49-cell Colab notebook the pipeline was developed in. Contains the K-elbow sweep, EDA plots, SHAP analysis and MLflow logging that the script omits. ⚠️ contains a committed secret. |
| `models/xgb_churn_model.pkl` | Deployed churn classifier (joblib). |
| `models/lr_churn_model.pkl` | Logistic Regression baseline (joblib). Versioned but never served. |
| `models/bgf_model.pkl` | Fitted BG/NBD `BetaGeoFitter` (**dill**). |
| `models/ggf_model.pkl` | Fitted Gamma-Gamma `GammaGammaFitter` (**dill**). |
| `models/final_decision_matrix.csv` | 4,334 × 17 scored customer table — the backend's entire read-only dataset. |
| `data/raw/online_retail_II.csv` | 42 MB raw transactions (DVC-tracked). |
| `data/processed/cleaned_retail.csv` | 37 MB cleaned transactions (DVC-tracked). |
| `data/*/*.dvc` | DVC pointer files (MD5 + size + path) committed in place of the data. |

### Backend

| Path | Purpose |
|---|---|
| `backend/main.py` | FastAPI app: artefact loading, 6 endpoints, CORS, rate limiting, LTV fallback, the duplicated `recommend_action()`. |
| `backend/test_main.py` | 4 pytest tests using `TestClient`, importing `main.py` by file path. |
| `backend/Dockerfile` | `python:3.11-slim` image; installs `libgomp1` + `git`; copies `backend/` and `models/` side by side; healthcheck on `/health`. |
| `backend/entrypoint.sh` | Container start: skip DVC if artefacts present, else inject credentials, `git init`, `dvc pull`, then `exec uvicorn`. |
| `requirements.txt` | Pinned **serving** dependencies. ⚠️ omits `imbalanced-learn`, needed for retraining. |

### Frontend

| Path | Purpose |
|---|---|
| `frontend/my-app/src/App.js` | 939-line single-component dashboard: shell, hero, two Recharts charts, lookup, prediction form, retain table, theme toggle, scrollspy. |
| `frontend/my-app/src/App.css` | 808-line CSS-variable design system with light/dark token sets. |
| `frontend/my-app/src/index.js` | React 19 root render inside `StrictMode`. |
| `frontend/my-app/src/index.css` | Font stacks and global resets. |
| `frontend/my-app/src/__tests__/App.test.js` | 2 smoke tests with `axios` and `matchMedia` mocked. |
| `frontend/my-app/public/index.html` | HTML shell: title, theme-color, Google Fonts links. |
| `frontend/my-app/public/manifest.json` | PWA manifest. ⚠️ references `logo192.png`/`favicon.ico`, which are absent → 404 in console. |
| `frontend/my-app/package.json` | React 19, Recharts 3, axios, `react-scripts` 5. |
| `frontend/my-app/nginx.conf` | Production static serving: gzip, 1-year `/static/` cache, SPA fallback. |
| `frontend/my-app/Dockerfile` | Multi-stage build → nginx. |
| `frontend/my-app/Dockerfile.dev` | Dev-server image for hot reload. |
| `frontend/my-app/.env` | `PORT=3000` only. |
| `frontend/my-app/.env.example` | Documents `REACT_APP_API_URL`. |
| `frontend/.env` | ⚠️ **Never read by CRA** — wrong directory (§5.3). |

### Infrastructure

| Path | Purpose |
|---|---|
| `Jenkinsfile` | Two-stage declarative pipeline: backend pytest in `python:3.12-slim` with host-mounted `models/`+`data/`; frontend `npm ci`/test/build in `node:20-alpine`. |
| `docker-compose.yml` | Production-like stack: nginx frontend on `3000:80`, backend healthcheck-gated, `REACT_APP_API_URL` as build arg. |
| `docker-compose.dev.yml` | Dev stack: bind-mounted source, `uvicorn --reload`, CRA dev server. |
| `.dvc/config` | Committed DVC remote definition (DagsHub URL). |
| `.dvc/config.local` | ⚠️ Gitignored credentials. Token currently returns 403. |
| `.dvcignore` | Paths DVC should not scan. |
| `.dockerignore` | Excludes `data/`, `notebooks/`, `src/`, `frontend/`, `.git`, `.dvc/cache` from the backend image. |
| `.gitignore` | OS, Python, Node, env and build-output exclusions. |

### Documentation

| Path | Purpose |
|---|---|
| `README.md` | Project overview, stack table, live URLs, local/Docker run instructions, DVC and Jenkins setup, API list, deployment quirks. ⚠️ "226" is stale. |
| `CLAUDE.md` | Repo guide for AI coding agents: commands, architecture, coupling warnings, deployment constraints. |
| `PROJECT_DEEP_DIVE.md` | This document. |
| `build_log.txt` | Committed frontend build diagnostics (artefact; could be removed). |
| `ngrok.log` | Committed ngrok output (artefact; could be removed). |

---

## Appendix: How the pipeline evolved (`git log`)

Reading the history tells the story of a project moving from notebook to production:

| Commit | Date | What it established |
|---|---|---|
| `56199df` | 2026-07-08 | First FastAPI backend with churn endpoints and the decision matrix. |
| `7d7053c` | 2026-07-10 | Paths resolved relative to `__file__`; CORS moved to an env var. **Deployment-awareness begins.** |
| `90ae27d` | 2026-07-12 | `/predict` added (+99 lines) — live inference for unseen customers. |
| `56ddfc2` | 2026-07-12 | Date and order-count validation added to `/predict`. |
| `ac12508` | 2026-07-12 | `/predict` output bucketed into the four retention actions. |
| `3adba7e` | 2026-07-12 | **`src/build_decision_matrix.py` created (+267 lines)** — the notebook is productionised. |
| `6989b41` | 2026-07-15 | Backend/frontend visualisation tweaks. |
| `a7f17bf` | 2026-07-15 | Action codes improved and the pipeline retrained — this is where **ranking-based cluster labelling** replaced the notebook's hardcoded map. |
| `8316b78` | 2026-07-15 | API hardening (+101 lines): logging, structured errors, reliability. |
| `c93791b` | 2026-07-16 | **503 fallback for artefact loading** (+70 lines) — graceful degradation. |
| `db03ce3` → `7ae793e` | 2026-07 | Jenkins pipeline added and debugged; `npm ci` lockfile fixes; `httpx` added for the test client. |
| `520ee5d` | 2026-07 | Hardcoded DagsHub credentials removed **from the Jenkinsfile** (the notebook was missed). |
| `2908774` → `74f58da` | 2026-07 | DVC pull at container startup; `git init` workaround for Render. |
| `6fa68d6` → `1263f69` | 2026-07 | Four consecutive commits pinning numpy/pandas and re-saving bgf/ggf — **the dill version-sensitivity tax**. |
| `31bbddc` | 2026-07-24 | `HEAD /health` for UptimeRobot. |
| `fa9ea77` | 2026-09-06 | Dashboard UI redesign. |
| `4890e9d` | 2026-09-06 | **Serving artefacts moved into the image**, ending the startup dependency on DagsHub. |

The arc: *notebook → script → API → validation → containerisation → CI → graceful degradation →
removing the last runtime network dependency.* That progression is itself the most interviewable
thing about the project.
