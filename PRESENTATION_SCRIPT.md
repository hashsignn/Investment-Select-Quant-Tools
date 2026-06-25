# Presentation Script — Market Regime Detection Using Autoencoders and PCA
**Total time: 15 minutes | Three speakers, 5 minutes each**

---

## PERSON 1 — Harjot Singh
### Data Collection, Preprocessing & the Data Layer
**Duration: ~5 minutes**

---

**[Opening — 30 sec]**

Good morning / afternoon, everyone. My name is Harjot, and I will start us off by walking you through the foundation of our project: where our data comes from, why we chose it, and how we transformed raw market information into a clean, model-ready dataset that both my teammates could build on.

**[Why this data? — 60 sec]**

The central question we are studying is: can we detect distinct market regimes — recurring states like bull markets, crisis periods, or high-inflation environments — from financial and macroeconomic data alone, without any manual labeling?

To do that well, we needed data that captures both price dynamics and the underlying economic environment. We chose a weekly dataset covering the US market from April 1988 to May 2026 — nearly 38 years. It contains 30 features: eight price and index series, including equity market returns, gold, oil, the US dollar, and Treasury yields, plus 22 macroeconomic and valuation indicators such as the yield curve, CPI, unemployment, CAPE ratio, dividend yield, GDP, and recession probability. Together these give us both what markets are doing and why — which is exactly what regime detection requires.

**[Stationarity problem — 60 sec]**

The first problem I had to solve was non-stationarity. The eight price and index columns are raw levels — they just trend upward over decades. If we feed those directly into a model, the autoencoder would be learning the trend, not the regime. I tested this with augmented Dickey-Fuller tests and confirmed all eight had ADF p-values near 1.0 — strongly non-stationary. The fix was straightforward: convert each one to log returns, that is, the natural log of today's price divided by yesterday's price. After that, all eight had p-values below 0.0001. The 22 macro features were already stationary — things like year-over-year GDP or unemployment rates do not trend indefinitely — so I kept them as-is.

**[Split and scaling — 60 sec]**

The second critical step was preventing data leakage. We split the data 70/15/15 by time — no shuffling, ever. The training set covers 1988 to 2014, validation 2014 to 2020, and the test set is the most recent period, 2020 to 2026. I then fit a StandardScaler on the training data only, and applied it to validation and test. This is the anti-leakage principle: the model must never see information from the future when learning to standardize.

I also built a 13-point automated contract test that verifies no NaNs, correct shapes, correct dtypes, and that validation and test data drift away from mean zero — confirming the scaler was not re-fit on them. All 13 checks pass.

**[Windows and handoff — 60 sec]**

The final step was creating the input windows. Autoencoders working on sequences need a fixed-length snapshot of the recent past. I tested window sizes from 13 to 78 weeks and found that regime labels were stable across all sizes — flip rate under 0.5% — because the slow macro signals dominate. I chose 26 weeks, approximately six months, as a natural economic horizon. The output is a NumPy array of shape 1365 × 26 × 30 for the training set: 1365 overlapping windows, each covering 26 weeks of 30 features.

I packaged everything — the windows, the scaler, the schema with feature names, and the metadata — into a single handoff folder that Eleni could load with two lines of Python to begin training immediately.

**[Transition — 10 sec]**

I will now hand over to Eleni, who took this data and trained the autoencoder on it.

---

## PERSON 2 — Eleni Tsaousi
### Autoencoder Implementation, Training & Latent Space Clustering
**Duration: ~5 minutes**

---

**[Opening — 20 sec]**

Thank you, Harjot. My name is Eleni, and I built and trained the autoencoder that learns a compressed representation of market behavior from the windows Harjot produced.

**[Why an autoencoder? — 50 sec]**

An autoencoder is a neural network trained to compress its input through a narrow bottleneck and then reconstruct it. The bottleneck forces the network to keep only what is essential — in our case, the underlying market state rather than the noise. The key advantage over PCA, which Adene will discuss, is that autoencoders can capture nonlinear relationships. Financial markets are not linear: a 1% rise in yields has very different implications depending on whether inflation is high or low, whether unemployment is rising, and what growth expectations are. We want a representation that can pick up on those interactions.

**[Architecture — 60 sec]**

Our autoencoder takes as input one flattened 26-week window: 26 weeks times 30 features equals 780 input values. The encoder compresses this through three layers — 256 neurons, then 64 neurons, then a 16-dimensional bottleneck — using ReLU activations throughout. The decoder mirrors this in reverse, expanding back to 780 values. We reconstruct the original window at the output and compute mean squared error as the loss.

Why 16 dimensions for the bottleneck? Harjot's analysis showed that a 10-component PCA already explains 67% of variance, and that many features are strongly correlated — 11 feature pairs with correlation above 0.8. This means the 30 input features carry far fewer independent dimensions, so 16 is a meaningful compression without collapsing the representation too aggressively.

**[Training and early stopping — 60 sec]**

I trained the model using the Adam optimizer with a weight decay of 5×10⁻⁴ for regularization. Crucially, I used early stopping based on the validation loss — the training runs for up to 200 epochs, but the best checkpoint is saved when validation loss stops improving. This prevents overfitting to the training period. The training reconstruction loss reached approximately 0.36. The validation loss was 0.63. The test loss was higher at 1.70, which reflects genuine distributional shift: the test period covers 2020 to 2026, which includes COVID, the post-pandemic inflation surge, and aggressive Fed rate hikes — genuinely unusual market conditions that were not well represented in the training period.

**[Clustering — 60 sec]**

Once the model was trained, I extracted the 16-dimensional latent vector for every window across all splits. I then applied K-Means clustering with K=4, fitting the cluster centers on the training latent vectors only, and then assigning regimes to validation and test. Four clusters was chosen to match the economic intuition that markets tend to move through a small number of broad states: expansions, contractions, high-stress periods, and transitional phases.

The resulting cluster distribution across all windows is roughly: Regime 0 — 402 windows, Regime 1 — 212, Regime 2 — 422, Regime 3 — 875. Regime 3 is the largest and most common state; Regime 1 is the smallest and tends to appear during atypical periods.

**[Handoff to PCA — 20 sec]**

I exported the latent vectors, regime labels, and reconstruction losses to a set of CSV files. Adene used these as her baseline for comparison. I will hand over to her now.

---

## PERSON 3 — Adene Dinoshi
### PCA Implementation, Comparison & Market Regime Interpretation
**Duration: ~5 minutes**

---

**[Opening — 20 sec]**

Thank you, Eleni. I am Adene, and my role was to build a PCA-based regime detector, compare it rigorously with Eleni's autoencoder, produce visualizations, and interpret what the discovered regimes mean economically.

**[Why PCA as a baseline? — 50 sec]**

PCA, Principal Component Analysis, is the natural linear benchmark here. It finds the directions of maximum variance in the data and projects everything onto those axes. It is fast, interpretable, and well-understood. If the autoencoder's nonlinear representation does not perform meaningfully better than PCA, we cannot claim the extra complexity is justified. To make the comparison fair, I used the same 16-dimensional projection as Eleni's bottleneck, fit only on the training data, and clustered with the same K-Means setup with K=4.

**[What PCA explains — 50 sec]**

The scree plot shows that 16 principal components explain approximately 71% of the variance in the windowed data. For context: 10 components explain 67%, and you need 50 components to reach 82%. The 16-D linear representation retains a substantial share of the information, which is an important result: it means both methods are working from a genuinely compressed view of the same data, so differences in regime quality reflect the linear versus nonlinear capacity of each method — not dimensionality.

**[Agreement metrics — 60 sec]**

I quantified how much the two methods agree using two standard clustering agreement metrics. The Adjusted Rand Index, or ARI, is 0.52 out of a maximum of 1. The Normalized Mutual Information is 0.58. Both indicate moderate agreement — significantly above random but well short of perfect. This is the key scientific finding of this project: PCA and the autoencoder share some regime structure, but they disagree on about half of the boundary placements. The autoencoder is picking up nonlinear relationships that linear PCA cannot reproduce even in 16 dimensions.

**[Regime interpretation — 60 sec]**

Looking at the feature profiles of each PCA regime on the training set gives us economic interpretations. Regime 0 is characterized by low short-term and long-term interest rates — a low-volatility, falling-rates environment consistent with periods of monetary easing. Regime 2 shows high CPI and rising 10-year yields, pointing to inflationary pressure. Regime 3 is the most distinct: it has the highest yield-spread stress and unemployment alongside the sharpest GDP contraction — this is our crisis or recession regime, appearing in about 12% of windows. Regime 1 is the largest and most diffuse, consistent with a normal or transitional market baseline.

The timeline figure confirms these interpretations: Regime 3 clusters tightly around known stress periods including the early 1990s recession, the 2008 financial crisis, and parts of the COVID shock.

**[Closing — 40 sec]**

To summarize the project: Person 1 built a rigorously leak-free preprocessing pipeline on 38 years of weekly financial and macro data. Person 2 trained a nonlinear autoencoder to compress that data into a 16-dimensional latent space and cluster it into four market regimes. I benchmarked that against a linear PCA of identical dimensionality, measured moderate agreement between the two, and interpreted the economic meaning of each regime.

The result is a fully reproducible, end-to-end pipeline for unsupervised market regime detection, with honest documentation of where the autoencoder adds value over the linear baseline and where it does not yet generalize out of sample.

Thank you.

---

*End of presentation. Total target: 15 minutes.*
