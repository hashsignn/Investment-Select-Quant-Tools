# Potential Q&A — Market Regime Detection Using Autoencoders and PCA
**Prepared by topic per presenter. Each person should know their own section and have a basic grasp of the others.**

---

## Person 1 — Data Collection & Preprocessing

**Q1: Why did you choose weekly frequency rather than monthly or daily?**
Weekly data gives enough time-resolution to catch regime transitions as they develop (a monthly frequency might already be mid-transition), while being free from the high-frequency microstructure noise in daily data. Macro features like CPI and GDP are also available monthly or quarterly, so we align everything to weekly by forward-filling where needed, which is defensible for slow-moving macro signals.

**Q2: Why log returns for the price columns and not percentage returns?**
Log returns are additive over time and have better statistical properties: they are approximately normally distributed for small moves, and the transformation is strictly monotone so no information is lost. The difference is negligible at small scales, but log returns are the standard for financial econometrics and make the ADF test results cleaner.

**Q3: You split 70/15/15 by time. Why not cross-validate?**
Standard k-fold cross-validation would shuffle time, meaning the model would train on future data and validate on the past — that is forward-looking bias, which invalidates any out-of-sample performance claim. Financial data is time-ordered by construction. We use forward chaining (train on oldest, test on newest) to honestly simulate the production setting.

**Q4: Why 26-week windows? Could a shorter or longer window detect different regimes?**
We tested 13, 26, 39, 52, 65, and 78-week windows and measured how often regime labels changed between adjacent windows (the regime flip rate). The flip rate was around 0.5% across all sizes, because the dominant signals are the slow macro variables that change on a monthly or quarterly basis. We chose 26 weeks as a meaningful economic horizon — roughly one fiscal half-year — rather than over-fitting the window size to the data.

**Q5: You mention 13 contract tests. What would fail if someone introduced leakage?**
The key check is that validation and test means drift away from zero. If the scaler had been fit on the full dataset, all three splits would have mean approximately zero and standard deviation approximately one. The test explicitly checks that the validation mean is NOT near zero, confirming that only training data was used to fit the scaler.

**Q6: What are the 30 features exactly?**
Eight are price or index series converted to log returns: equity market (_MKT), gold (_AU), US dollar index (_DXY), long-term corporate bonds (_LCP), 10-year Treasury yield (_TY), oil (_OIL), value factor (_VA), and growth factor (_GR). The remaining 22 are macroeconomic and valuation signals kept in their original form: employment, PE ratio, CAPE, dividend yield, Rho, realized volatility (MOV), nominal interest rate, real rate, 2-year yield, 10-year yield, yield spread, money supply, GDP, CPI, consumer confidence, recession probability, and others.

**Q7: Could you have used more features?**
Yes. We deliberately kept the set to features available continuously over the full 1988–2026 window to avoid survivorship bias from newer data series. Adding more features would also increase the input dimensionality and potentially make the autoencoder's compression task harder without adding regime-relevant information.

---

## Person 2 — Autoencoder Implementation & Clustering

**Q1: Why a fully connected autoencoder rather than an LSTM or Transformer?**
Our input is a fixed-length 26-week window flattened to 780 values. Within that window, the temporal ordering matters less than the overall distributional state of the market — we are looking for what regime we are in, not predicting the next price. A fully connected architecture is a natural and computationally efficient baseline here. An LSTM or Transformer would add complexity that we have not yet shown is necessary to beat this baseline.

**Q2: Why 16 latent dimensions? How was that chosen?**
Two reasons: First, Harjot's PCA analysis showed that a 10-component PCA already explains 67% of variance, and 16 components explain about 71%. This implies the effective dimensionality of the data is well below 30. Second, we compared against a 3-dimensional bottleneck and observed mode collapse — the encoder pushed all inputs toward the same region of latent space, producing unstable cluster assignments. Widening to 16 solved this without losing interpretability. Choosing the same number as the PCA comparison also makes the benchmark fair.

**Q3: What is early stopping and why does it matter here?**
Early stopping monitors the validation loss at each epoch and saves the model checkpoint whenever a new minimum is reached. Training stops when no improvement has been seen for 40 consecutive epochs. Without this, the model would continue fitting the training set until it memorized it — the training loss would drop but the validation loss would rise (overfitting). Since we are interested in regime structure that generalizes across time, we need to select the model that performs best on held-out data.

**Q4: Why does the test reconstruction loss jump to 1.70 when training is only 0.36?**
The test period is 2020 to 2026. This includes COVID-19, the fastest post-pandemic inflation in 40 years, and the Federal Reserve raising rates by 525 basis points in 18 months. These are combinations of macro conditions the training data (1988–2014) had never seen simultaneously. A high reconstruction loss means the model has to work harder to compress these windows — they are genuinely novel relative to the training distribution. We interpret this as evidence of a regime shift rather than model failure.

**Q5: Why K-Means with K=4? Did you try other values of K?**
K=4 reflects our prior economic intuition that financial markets tend to cycle through a small number of broad states. We also evaluated the Elbow method on within-cluster sum of squares across K values from 2 to 8, and K=4 gave a clear bend. More practically, K=4 gives interpretable regimes that map to recognizable economic periods: expansion, contraction, stress, and transition. Larger K produces fragmented regimes that are harder to interpret and less stable out of sample.

**Q6: Why do you cluster only on training latent vectors and then predict for val and test?**
The same reason we scaled only on training data: we cannot use future information to define the regime structure. If we clustered on all data at once, the cluster centroids would be influenced by validation and test observations, leaking information about the future into the training regime assignments. By fitting KMeans only on training, the regime labels for val and test are true out-of-sample predictions.

**Q7: The regime distribution shows Regime 3 has 875 windows — twice any other. Is that a problem?**
Not necessarily. Regime 3 corresponds to the most common "baseline" market state — normal conditions without strong inflation, stress, or monetary extremes. Markets spend most of their time in normal conditions, so a large cluster for the baseline regime is economically sensible. What matters more is whether the smaller, more distinctive regimes — like the crisis regime — are coherent and stable over time, which they are.

**Q8: Could this be used for live trading signals?**
In its current form, no. The backtest shows the strategy is profitable and economically sensible in sample but fails to generalize to the test period. The test Sharpe is negative and the IC is indistinguishable from zero. This is consistent with the high test reconstruction loss — the model is encountering regime conditions it has not seen. Before using this for live trading, one would need to address the distribution shift problem, potentially with online learning or periodic retraining.

---

## Person 3 — PCA, Comparison & Regime Interpretation

**Q1: What exactly does "Adjusted Rand Index of 0.52" mean in plain English?**
The Adjusted Rand Index measures how often the two methods agree on whether any two windows belong to the same regime or different regimes, after correcting for chance agreement. A value of 1 means perfect agreement, 0 means the methods agree no more often than random chance. At 0.52, the methods share meaningful regime structure — they are not random — but they disagree on about half the cases. In practical terms: for the windows that both methods agree on, they are capturing the same economic signal; for the rest, the autoencoder's nonlinear capacity is grouping things differently than linear PCA does.

**Q2: PCA explains 71% of variance in 16 dimensions. Is that enough?**
It depends on the question. For regime detection, it appears sufficient to identify the broad economic states. The heatmap shows that each PCA regime has a clear majority overlap with a corresponding AE regime, which means the two share the dominant structure. The 29% of variance PCA discards as noise may contain the nonlinear interactions the autoencoder uses to refine the boundaries. Whether that refinement is economically meaningful is an open question — it is why we include the ARI and NMI as metrics rather than just claiming one method is better.

**Q3: How do you know the regime labels actually map to real economic states?**
We look at the mean standardized value of each feature across windows in each regime. If a regime has high CPI and high 10-year yields, and those windows cluster around the early 1980s and 2021–2023 in the timeline, that is consistent with known inflationary periods. We cross-checked the crisis regime (Regime 3) against NBER recession dates and found significant overlap. The interpretations are educated inferences, not ground truth — no ground truth exists for unlabeled market regimes.

**Q4: Why does the regime timeline show the test period dominated by one or two regimes?**
The test period covers 2020–2026, which is an exceptional stretch: the COVID shock, the reopening boom, and the most aggressive rate-hiking cycle in modern history. The training data, which spans 1988–2014, included some of these macro configurations separately but never in this sequence. The regime assignments in the test period may be reflecting mode collapse where the model pushes novel windows into the nearest training-era cluster rather than identifying a truly new state.

**Q5: Why did you use t-SNE for the 2D and 3D visualizations instead of just plotting the first two or three components?**
Both PCA and the autoencoder produce 16-dimensional representations. Directly plotting only PC1 vs PC2 would discard 14 dimensions of information, potentially making regimes look more separated or more mixed than they truly are. t-SNE reduces all 16 dimensions to 2 or 3 while preserving local neighborhood structure, giving a more honest picture of how the regimes are actually distributed in the full latent space.

**Q6: Why do your regime interpretations use the last week of each window rather than the average over the window?**
The last week is the most current observation in the window — it is the point in time to which we assign the regime label. Using the full 26-week average would smooth over transient events at the start of the window that are no longer relevant at the window's endpoint. The last-week snapshot captures the market state at the moment of classification.

**Q7: If you had to pick one result from this project as the key takeaway, what would it be?**
The ARI of 0.52 between PCA and the autoencoder. It shows that nonlinear representation learning does identify different regime structure than the linear baseline, which validates the core hypothesis of the project. But 0.52 also means the methods substantially overlap, which tells us that a large share of the regime signal is linear in nature and does not require a neural network to detect. The honest answer is: the autoencoder adds something, but not everything, over PCA in this setting.

**Q8: Your feature profiles show Regime 1 as "mixed / transitional." How useful is a regime you can't interpret?**
It is actually useful as a null category. In a 4-cluster solution, if you force every observation into a named regime, you risk over-interpreting noise. Having a diffuse "transitional" regime means the other three clusters are sharper and more coherent. The transitional windows are likely periods where the market is moving between states — they are real and meaningful, just not in a way that maps to a simple economic label.

---

## General / Cross-Person Questions

**Q: How does your work connect to actual investment decisions?**
We included a simple backtest as a proof-of-concept. The rule assigns long, short, or flat positions based on which regime has historically produced the best or worst forward returns. The strategy is profitable in sample but does not generalize out of sample, with a negative test Sharpe. We are transparent about this: regime detection at this stage is a research tool, not a tradeable signal.

**Q: Why didn't you use Hidden Markov Models, which are standard for regime detection?**
HMMs assume a specific Markov structure (the next regime depends only on the current one) and typically require the number of regimes and the emission distributions to be specified explicitly. Autoencoders learn the representation purely from data without those assumptions, which is the comparison we were asked to make in this project. Comparing against HMMs would be a natural extension.

**Q: Is the 38-year dataset long enough to trust these results?**
Thirty-eight years covers multiple full business cycles, two major equity bear markets (2000–2002, 2008–2009), two inflationary episodes (late 1980s and 2021–2023), and several credit stress events. This is a reasonable length for studying macro regimes, which operate on multi-year cycles. It is not, however, long enough to make strong statistical claims about very rare tail events.
