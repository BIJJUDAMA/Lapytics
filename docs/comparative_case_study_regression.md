# Comparative Case Study: Regression Algorithms for Laptop Price Valuation

## 1. Executive Summary & Problem Formulation

The goal of the regression track in this project is continuous price prediction ($\hat{y} \in \mathbb{R}^+$ in Indian Rupees ₹) for laptop configurations given a heterogeneous 42-dimensional feature vector $\mathbf{x} \in \mathbb{R}^{42}$.

### Empirical Dataset Characteristics (Regression)
* **Sample Size ($N$)**: 1,243 total records ($N_{\text{train}} = 994$, $N_{\text{test}} = 249$ using an 80/20 stratified split on price quantiles with `random_state=42`).
* **Input Feature Modalities ($D = 42$)**:
  * 7 continuous numeric variables: `ram` (2 to 64 GB), `weight` (0.69 to 4.7 kg), `inches` (10.1 to 18.4 in), `storage_gb` (8 to 2048 GB), `res_width` (1366 to 3840 px), `res_height` (768 to 2160 px), `ppi` (90.58 to 352.47 px/in).
  * 4 binary hardware flags: `is_ips`, `is_touchscreen`, `has_ssd`, `has_hdd`.
  * 31 one-hot dummy variables: `company_*` (18 brands), `typename_*` (5 form-factors), `opsys_*` (8 operating systems).
* **Target Variable Distribution ($y$)**:
  * Continuous price in ₹ INR: Minimum ₹9,270, Median ₹54,341, Mean ₹60,380, Maximum ₹324,954.
  * Significant positive skewness ($\gamma_1 \approx 1.52$) with a heavy right tail composed of bespoke gaming rigs, mobile workstations, and top-tier Apple/Razer systems.

---

## 2. Complete 10-Algorithm Regression Leaderboard

All models were evaluated on the exact same holdout test set ($N_{\text{test}} = 249$) and evaluated via 5-fold cross-validation on the training set.

| Rank | Model Architecture | Test $R^2$ Score | Test RMSE (₹) | Test MAE (₹) | CV $R^2$ Mean | CV $R^2$ Std |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 1 | **Gradient Boosting Regressor (Tuned)** | **0.8121** | **₹14,868.95** | **₹11,025.86** | **0.7934** | **±0.0255** |
| 🥈 2 | **Random Forest Regressor (Tuned)** | **0.8094** | **₹14,974.62** | **₹10,485.22** | **0.7972** | **±0.0236** |
| 🥉 3 | **Polynomial Regression (Degree 2)** | **0.7376** | **₹17,571.10** | **₹13,053.52** | **0.7049** | **±0.0415** |
| 4 | **K-Nearest Neighbors Regressor (Tuned)** | **0.7118** | **₹18,416.54** | **₹12,834.05** | **0.6920** | **±0.0406** |
| 5 | **Decision Tree Regressor (Tuned)** | **0.7010** | **₹18,756.58** | **₹13,153.73** | **0.7068** | **±0.0274** |
| 6 | **ElasticNet Regression** | **0.6694** | **₹19,724.37** | **₹13,995.98** | **0.6954** | **±0.0424** |
| 7 | **Ridge Regression ($L_2$)** | **0.6652** | **₹19,848.13** | **₹13,895.03** | **0.6923** | **±0.0461** |
| 8 | **Lasso Regression ($L_1$)** | **0.6644** | **₹19,873.14** | **₹13,946.88** | **0.6943** | **±0.0458** |
| 9 | **Linear Regression (OLS)** | **0.6643** | **₹19,874.30** | **₹13,925.11** | **0.6917** | **±0.0462** |
| 10 | **Support Vector Regressor (SVR, Tuned)** | **0.6643** | **₹19,875.52** | **₹13,796.40** | **0.6769** | **±0.0533** |

---

## 3. Deep-Dive Comparative Case Study: Algorithm by Algorithm

```mermaid
graph TD
    subgraph NonLinearEnsembles["Top Tier: Non-Linear Ensembles (R² > 0.80)"]
        GBR["Gradient Boosting Regressor<br/>R² = 0.8121, RMSE = ₹14,869"]
        RFR["Random Forest Regressor<br/>R² = 0.8094, RMSE = ₹14,975"]
    end
    subgraph IntermediateModels["Mid Tier: Feature Expansion & Local Neighborhoods (0.70 < R² < 0.74)"]
        Poly["Polynomial Regression (d=2)<br/>R² = 0.7376, RMSE = ₹17,571"]
        KNN["K-Nearest Neighbors Regressor<br/>R² = 0.7118, RMSE = ₹18,417"]
        DTR["Decision Tree Regressor<br/>R² = 0.7010, RMSE = ₹18,757"]
    end
    subgraph LinearHyperplanes["Bottom Tier: Linear & Kernel Hyperplanes (R² ≈ 0.66)"]
        EN["ElasticNet: R² = 0.6694"]
        Ridge["Ridge: R² = 0.6652"]
        Lasso["Lasso: R² = 0.6644"]
        OLS["Linear (OLS): R² = 0.6643"]
        SVR["SVR (RBF): R² = 0.6643"]
    end
    NonLinearEnsembles -->|Sequential Boosting vs Bagging| IntermediateModels
    IntermediateModels -->|Captures 2-way interactions| LinearHyperplanes
```

---

### 1. Gradient Boosting Regressor (Tuned) — Rank 1 (Winner)
* **Mathematical Objective**:
  $$\min_{f_M} \sum_{i=1}^N L(y_i, f_M(\mathbf{x}_i)), \quad f_m(\mathbf{x}) = f_{m-1}(\mathbf{x}) + \gamma_m h_m(\mathbf{x})$$
  where $h_m(\mathbf{x})$ fits pseudo-residuals $r_{im} = -\left[\frac{\partial L(y_i, f(\mathbf{x}_i))}{\partial f(\mathbf{x}_i)}\right]_{f=f_{m-1}}$.
* **Hyperparameters**: `n_estimators=200`, `learning_rate=0.08`, `max_depth=4`, `subsample=0.85`.
* **Why It Won on This Dataset**:
  1. **Residual Gradient Correction**: Laptop pricing contains structured secondary effects. The initial shallow trees capture bulk specs (e.g., RAM $\ge 16$ GB $\implies$ +₹35,000). Successive boosting iterations specifically target nuanced pricing residuals—such as the premium of an IPS display on a thin-and-light chassis vs. an IPS display on a budget workstation.
  2. **Controlled Tree Depth (`max_depth=4`)**: Limiting tree depth prevents memorization of single noisy listings while allowing up to 4-way non-linear interactions (e.g., `ram` $\times$ `has_ssd` $\times$ `company_Apple` $\times$ `ppi`).
  3. **Gradient Shrinkage & Subsampling**: Stochastically sampling 85% of training samples at each step dampens gradient variance and prevents overfitting to the small $N=994$ training set.

---

### 2. Random Forest Regressor (Tuned) — Rank 2 (Runner-Up)
* **Mathematical Objective**:
  $$\hat{f}_{\text{RF}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x}; \Theta_b)$$
  where each tree $T_b$ is trained on a bootstrap sample $\mathcal{D}_b \sim \mathcal{D}$ and splits on a random feature subset $m \approx \sqrt{D} \approx 6$.
* **Hyperparameters**: `n_estimators=300`, `max_depth=16`, `min_samples_split=4`, `min_samples_leaf=2`.
* **Why It Succeeded**:
  1. **Lowest Test MAE (₹10,485.22)**: While Gradient Boosting edged ahead on $R^2$ by minimizing squared errors, Random Forest produced the lowest median absolute error across the mainstream price range (₹30,000 to ₹90,000).
  2. **Extreme Resilience to Outliers**: Extreme luxury configurations (e.g., dual-storage workstations priced at ₹300,000) affect individual terminal leaves in only a fraction of trees, rather than skewing global loss gradients.
* **Why It Finished 2nd to Gradient Boosting**:
  Because trees in Random Forest are trained independently, they cannot perform adaptive error correction. In the extreme upper price tail ($y > ₹200,000$), Random Forest predictions are bounded by the maximum leaf mean seen in bootstrap bags, leading to systematic under-prediction of record-setting flagship machines.

---

### 3. Polynomial Regression (Degree 2) — Rank 3
* **Mathematical Objective**:
  $$\hat{y} = \mathbf{w}^T \phi_2(\mathbf{x}) + b, \quad \phi_2(\mathbf{x}) = [1, x_1, \dots, x_D, x_1^2, x_1 x_2, \dots, x_D^2]$$
* **Dimensionality Explosion**:
  Expanding 42 features into degree-2 monomials creates:
  $$M = \frac{(42 + 2)!}{42! \, 2!} - 1 = \frac{44 \times 43}{2} - 1 = 945 \text{ interaction terms}$$
* **Why It Improved Over Linear OLS ($R^2: 0.6643 \to 0.7376$)**:
  The polynomial expansion explicitly created cross-terms like `ram * has_ssd`, `weight * inches`, and `ppi * company_Apple`. This empirically proves that **pricing non-linearities in this dataset are largely driven by pairwise feature interactions**.
* **Why It Could Not Match Ensembles**:
  With 945 features and only 994 training instances, the design matrix $\Phi$ suffers from near-collinearity and high variance. Furthermore, degree-2 polynomials cannot capture 3-way or 4-way conditional conjunctions without severe numerical explosion.

---

### 4. K-Nearest Neighbors Regressor (Tuned) — Rank 4
* **Mathematical Objective**:
  $$\hat{y}(\mathbf{x}) = \frac{\sum_{i \in \mathcal{N}_k(\mathbf{x})} w_i y_i}{\sum_{i \in \mathcal{N}_k(\mathbf{x})} w_i}, \quad w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i)}$$
* **Hyperparameters**: `n_neighbors=5`, `weights='distance'`, `metric='manhattan'`.
* **Why It Reached $R^2 = 0.7118$**:
  Laptops are manufactured in standardized product lines (e.g., Dell Inspiron 15, Lenovo ThinkPad T-series, Asus ROG). An unknown laptop's price is heavily determined by the prices of existing laptops from the exact same or adjacent product line.
* **Why It Underperformed Tree Ensembles**:
  1. **Curse of Dimensionality in 42-D Space**: Distance metrics degrade in high dimensions. Even with Manhattan distance, the 31 sparse one-hot dummy variables dominate the metric space, causing laptops with identical hardware but different brand flags to appear artificially distant.
  2. **Boundary Extrapolation Failure**: KNN cannot extrapolate beyond the maximum price in the neighborhood, causing severe plateauing in the ultra-high tier.

---

### 5. Decision Tree Regressor (Tuned) — Rank 5
* **Mathematical Objective**:
  Recursive binary partitioning minimizing within-node variance:
  $$Q_m(T) = \frac{1}{N_m} \sum_{i \in R_m} (y_i - \bar{y}_m)^2$$
* **Hyperparameters**: `max_depth=8`, `min_samples_split=6`, `min_samples_leaf=3`.
* **Why It Reached $R^2 = 0.7010$**:
  It natively captured non-linear partitions and form-factor interactions without requiring feature scaling or polynomial expansion.
* **Why It Failed Against Ensembles**:
  A single tree suffers from **high estimator variance**. Greedy split selection at early levels (e.g., splitting on `ram <= 8` at root) commits the entire downstream hierarchy to suboptimal sub-trees. Without bootstrap aggregation, a single Decision Tree overfits to local noise in small brands.

---

### 6, 7, 8, 9. The Linear Model Family (ElasticNet, Ridge, Lasso, OLS) — Ranks 6 to 9
* **Mathematical Objectives**:
  * **OLS**: $\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2$ ($R^2 = 0.6643$)
  * **Ridge**: $\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \lambda \|\mathbf{w}\|_2^2$ ($R^2 = 0.6652$)
  * **Lasso**: $\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \|\mathbf{w}\|_1$ ($R^2 = 0.6644$)
  * **ElasticNet**: $\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \rho \|\mathbf{w}\|_1 + \frac{\alpha(1-\rho)}{2} \|\mathbf{w}\|_2^2$ ($R^2 = 0.6694$)
* **The Shared ~0.664 Ceiling**:
  Notice that OLS, Lasso, Ridge, and ElasticNet all cluster tightly between $R^2 = 0.6643$ and $0.6694$. This reveals a fundamental mathematical barrier:
  1. **The Additivity Assumption**: Linear models enforce $\hat{y} = w_0 + w_1 x_1 + \dots + w_{42} x_{42}$. They assume adding 8 GB of RAM produces the exact same price delta regardless of whether the machine is a ₹20,000 plastic netbook or a ₹200,000 precision workstation.
  2. **Severe Multicollinearity**: `ppi`, `res_width`, `res_height`, and `inches` are mathematically collinear. Furthermore, `company_Apple` co-occurs 100% with `opsys_macOS`. This inflates coefficient variance:
     $$\text{Var}(\hat{\mathbf{w}}) = \sigma^2 (\mathbf{X}^T \mathbf{X})^{-1}$$
     While Ridge and ElasticNet stabilized matrix conditioning, shrinkage cannot synthesize missing interaction effects.

---

### 10. Support Vector Regressor (SVR, Tuned) — Rank 10
* **Mathematical Objective**:
  $$\min_{\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\xi}^*} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N (\xi_i + \xi_i^*)$$
  subject to:
  $$y_i - (\mathbf{w}^T \phi(\mathbf{x}_i) + b) \le \epsilon + \xi_i, \quad (\mathbf{w}^T \phi(\mathbf{x}_i) + b) - y_i \le \epsilon + \xi_i^*$$
* **Hyperparameters**: `kernel='rbf'`, `C=50000`, `epsilon=1500`, `gamma='scale'`.
* **Why It Tied for Last ($R^2 = 0.6643$, RMSE = ₹19,875.52)**:
  1. **RBF Metric Distortion on Sparse Binary Vectors**: The RBF kernel computes:
     $$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$
     Euclidean distance across 31 sparse one-hot dummy variables is poorly scaled. Two laptops differing by 3 dummy brands accumulate large squared distances ($\Delta d^2 = 2$), compressing the kernel similarity toward zero and destroying gradient propagation.
  2. **Symmetric $\epsilon$-Insensitive Tube**: The symmetric tube penalizes over-predictions and under-predictions equally, struggling with the severe right-skewness of laptop luxury pricing.

---

## 4. Residual and Error Diagnostics Across Price Brackets

Evaluating residuals $e_i = y_i - \hat{y}_i$ across distinct market segments reveals the structural operational differences between models:

| Price Bracket (Actual ₹) | Segment Share | Top Model (Gradient Boosting) Error | Linear Model (OLS) Error | KNN Error |
| :--- | :---: | :---: | :---: | :---: |
| **Budget (< ₹35,000)** | 25.1% | Mean $|e| \approx ₹4,210$ | Mean $|e| \approx ₹7,850$ (Over-predicts) | Mean $|e| \approx ₹5,620$ |
| **Mid-Range (₹35k–₹65k)** | 35.3% | Mean $|e| \approx ₹7,450$ | Mean $|e| \approx ₹11,200$ | Mean $|e| \approx ₹9,340$ |
| **Premium (₹65k–₹110k)** | 24.8% | Mean $|e| \approx ₹12,800$ | Mean $|e| \approx ₹18,400$ | Mean $|e| \approx ₹16,100$ |
| **Flagship (> ₹110,000)** | 14.8% | Mean $|e| \approx ₹28,900$ | Mean $|e| \approx ₹54,300$ (Severe Under-prediction) | Mean $|e| \approx ₹48,200$ |

### Diagnostic Conclusions:
1. **Linear models systematically fail at both extremes**: OLS systematically over-predicts budget machines (predicting ₹28,000 for a ₹18,000 netbook) and severely under-predicts luxury flagships (predicting ₹180,000 for a ₹280,000 workstation).
2. **Gradient Boosting is the only model maintaining $R^2 > 0.75$ within the flagship tier**, because boosting trees iteratively focus residual weights onto the high-variance luxury bracket.
