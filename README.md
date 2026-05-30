## Credit Scoring Business Understanding

### 1. Basel II and Model Interpretability

The Basel II Capital Accord requires financial institutions to
maintain documented, interpretable, and auditable risk models.
Specifically, Basel II's Internal Ratings-Based (IRB) approach
mandates that banks must be able to explain every credit
decision to regulators, demonstrate that their models are
validated against historical data, and show that risk estimates
are conservative and well-documented.

This directly influences modeling choices in three ways:

**Interpretability over complexity:** A Logistic Regression
model with Weight of Evidence (WoE) encoded features produces
coefficients that directly explain each variable's contribution
to default probability. A gradient boosting model may achieve
higher AUC but cannot natively explain why a specific applicant
was denied — requiring additional tools like SHAP.

**Documentation requirements:** Every transformation, feature
selection decision, and threshold choice must be documented and
justified. This project maintains that audit trail through
versioned code, MLflow experiment logs, and this README.

**Model validation:** Basel II requires models to be validated
on out-of-sample data and periodically re-validated as
population behavior shifts. Our train/test split and MLflow
tracking directly support this requirement.

---

### 2. Proxy Variable — Necessity and Business Risks

The raw Xente eCommerce dataset contains no historical default
label — no customer has previously taken a loan through this
platform. Without a direct target variable, supervised learning
cannot be applied directly.

**Why a proxy is necessary:**
We engineer a proxy target variable using RFM (Recency,
Frequency, Monetary) behavioral analysis. The underlying
assumption is that a customer's transaction behavior is
correlated with their financial reliability:

- **High engagement** (frequent, recent, high-value transactions)
  → financially active → lower default risk
- **Low engagement** (infrequent, old, low-value transactions)
  → potentially financially stressed → higher default risk

K-Means clustering segments customers into behavioral groups,
and the least engaged cluster is labeled `is_high_risk = 1`.

**Business risks introduced by proxy-based prediction:**

| Risk | Description |
|---|---|
| Label noise | Disengaged shoppers are not necessarily bad borrowers |
| Selection bias | eCommerce behavior may not represent creditworthiness |
| False positives | Good borrowers denied loans due to low shopping activity |
| False negatives | Bad borrowers approved due to high shopping activity |
| Regulatory risk | Proxy must be justified and disclosed under Basel II |

This proxy is a modeling assumption, not ground truth. All
predictions and model outputs must be communicated with this
limitation explicitly stated.

---

### 3. Interpretable vs High-Performance Models

| Dimension | Logistic Regression + WoE | Gradient Boosting (XGBoost) |
|---|---|---|
| Predictive accuracy | Moderate | High |
| Interpretability | High — coefficients are human-readable | Low — black box |
| Basel II compliance | Native — each feature's contribution is explicit | Requires SHAP or LIME |
| Development speed | Fast | Slower — requires tuning |
| Regulatory approval | Straightforward | More documentation burden |
| Overfitting risk | Low | Higher without careful tuning |
| Best for | Regulatory-first environments | Performance-first environments |

**Our approach:** We train both interpretable (Logistic
Regression) and high-performance (XGBoost, LightGBM) models,
compare them on AUC and F1, and use SHAP to provide
post-hoc interpretability for the gradient boosting model.
The final production model balances performance with
Basel II compliance requirements.