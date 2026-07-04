# 📈 Portfolio Optimization & Risk Parity Strategy

This project develops a quantitative finance pipeline to build, optimize, and project the performance of a multi-sector stock portfolio. The system aims to construct a diversified portfolio of $50,000 that minimizes downside risk while outperforming traditional market benchmarks (S&P 500).

**Authors**
* Elena Maggiore
* Marco Rosato

### Challenge & Solution:

A primary challenge in portfolio optimization is that financial assets are highly correlated, and traditional Mean-Variance optimization is sensitive to daily market noise and estimation errors in the covariance matrix.

To solve this, we implemented this strategy:
* **Unsupervised Diversification:** We used a correlation-based distance metric combined with **Hierarchical Clustering (Ward's method)** to group assets by their true underlying market behavior, ensuring we pick assets that do not move together.
* **Risk Parity & Ledoit-Wolf Shrinkage:** Instead of dividing capital equally, we allocated capital to ensure an **equal risk contribution** from each asset. To prevent the optimizer from overreacting to extreme daily volatility, we applied **Ledoit-Wolf shrinkage** to estimate a more robust covariance matrix.
* **Frequency Filtering:** We tackled market microstructure noise (e.g., algorithmic trading anomalies) by comparing Daily, Weekly, and Monthly sampling frequencies, isolating the true macroeconomic signals.

### Dataset
The project utilizes historical market data downloaded via the `yfinance` API.
* **Universe:** 92 tickers representing a broad range of 11 sectors (Technology, Healthcare, Financials, Energy, Defense, Precious Metals, sector ETFs, etc.) plus the **S&P 500 (^GSPC)** as the benchmark.
* **Timeframes:** 
  * *Training Set:* 2012-01-01 to 2019-12-31 (used for clustering and optimization).
  * *Testing Set (Out-of-Sample):* 2022-01-01 to 2025-12-31 (used for performance projection, skipping the 2020-2021 COVID-19 anomaly).

### Project Workflow
The project is structured into several key stages:

1. **Data Engineering & Financial Metrics**
   * Preprocessing included removing assets listed post-2012 to ensure a complete matrix, forward-filling missing values, and computing **logarithmic returns**.
   * Calculated risk-adjusted metrics for each asset: Drawdowns, Beta, VaR (95%), CVaR (95%), Semi-standard deviation, Sharpe, Treynor, Sortino, and Calmar ratios.

2. **Asset Clustering & Selection**
   * Converted the correlation matrix into a distance matrix ($distance = \sqrt{0.5 \times (1 - corr)}$).
   * Evaluated K-Means, Complete Linkage, and Ward's method using Silhouette Scores. **Ward's method** was selected, automatically discovering 34 distinct behavioral market clusters (for the daily case).
   * The single best-performing asset from each cluster was selected using risk-adjusted metrics.

3. **Risk Parity Optimization**
   * Implemented a custom objective function optimized via Sequential Least Squares Programming (SLSQP) to find the exact weights that equalize the marginal risk contribution of each selected asset.

4. **Multi-Frequency & Strategy Analysis**
   * **Frequency Analysis:** Evaluated how the optimal number of clusters and overall portfolio performance changes when using Daily, Weekly, or Monthly data.
   * **Metric Comparison:** Conducted a final ablation study on the monthly data, comparing asset selection via the **Sortino Ratio** (focusing on downside volatility) versus the **Calmar Ratio** (focusing on maximum drawdown).

### Key Results
* **Outperforming the Benchmark:** The base Daily Risk Parity portfolio successfully beat the S&P 500 out-of-sample, generating a 54.00% return vs the benchmark's 44.69%, while maintaining a lower maximum drawdown (-19.54% vs -25.43%).
* **The Power of Monthly Data:** Filtering out short-term market noise by aggregating data monthly reduced the optimal cluster count to 32 and significantly boosted performance, raising the Sharpe Ratio to 0.90.
* **Calmar vs. Sortino:** The final optimization revealed that selecting assets based on the **Calmar Ratio** on a Monthly timeframe was the best strategy. It achieved an **85.11% Total Return** (Sharpe Ratio: 0.99), outperforming the Sortino strategy by successfully allocating capital to resilient sectors (like the SMH Semiconductor ETF) and avoiding assets.
