# Trader Sentiment Insights

## Overview
Trader Sentiment Insights is a comprehensive analytics project that explores the relationship between market sentiment (using the Fear & Greed Index) and trader performance. By merging historical trading data with sentiment indicators, the project uncovers patterns, tests statistical hypotheses, and applies machine learning to classify and segment trader behavior. The goal is to provide actionable insights for risk management, strategy development, and further research in quantitative trading.

## Project Goals
- Quantify how market sentiment affects trader performance metrics (PnL, win rate, trade size, etc.)
- Identify statistically significant differences between sentiment regimes
- Build predictive models to classify sentiment based on trading features
- Segment traders into behavioral clusters
- Provide visual and statistical outputs for decision support

---

## Data Dictionary

### `data/historical_data.csv`
| Column             | Description                                      |
|--------------------|--------------------------------------------------|
| Account            | Trader's unique account address                  |
| Coin               | Traded asset identifier                          |
| Execution Price    | Price at which the trade was executed            |
| Size Tokens        | Trade size in tokens                             |
| Size USD           | Trade size in USD                                |
| Side               | BUY/SELL indicator                               |
| Timestamp IST      | Trade timestamp (local time)                     |
| Start Position     | Position size before trade                       |
| Direction          | Trade direction (Buy/Sell)                       |
| Closed PnL         | Profit and Loss realized on trade                |
| Transaction Hash   | Blockchain transaction hash                      |
| Order ID           | Unique order identifier                          |
| Crossed            | Whether the order was crossed (TRUE/FALSE)       |
| Fee                | Trading fee paid                                 |
| Trade ID           | Unique trade identifier                          |
| Timestamp          | Trade timestamp (UNIX, optional)                 |

### `data/fear_greed_index.csv`
| Column         | Description                                 |
|----------------|---------------------------------------------|
| timestamp      | UNIX timestamp                              |
| value          | Fear & Greed Index value (0-100)            |
| classification | Sentiment label (e.g., Fear, Greed, Neutral)|
| date           | Date (YYYY-MM-DD)                           |

---

## Workflow
1. **Data Loading & Preprocessing**
   - Cleans and standardizes both datasets
   - Handles missing values and type conversions
2. **Feature Engineering**
   - Aggregates trade data by account, date, and sentiment
   - Computes rolling averages, win rates, and cumulative PnL
3. **Exploratory Data Analysis (EDA)**
   - Prints summary statistics and group means
   - Visualizes distributions and correlations
4. **Statistical Testing**
   - Performs t-tests and ANOVA to compare sentiment groups
5. **Machine Learning Modeling**
   - Trains a Random Forest classifier to predict sentiment
   - Evaluates with classification report and confusion matrix
   - Plots feature importances
6. **Clustering/Segmentation**
   - Applies KMeans clustering to segment trader behaviors
   - Reports cluster counts
7. **Visualization**
   - Saves and displays multiple plots (see below)
8. **Insights & Recommendations**
   - Prints actionable findings and next steps

---

## Output Visualizations
- `closedPnL_by_sentiment.png` — Bar plot of average closed PnL by sentiment
- `correlation_matrix.png` — Correlation heatmap of features
- `kde_closedPnL_by_sentiment.png` — KDE plot of closed PnL by sentiment
- `timeseries_closedPnL_by_sentiment.png` — Time series of closed PnL by sentiment
- `total_trades_by_sentiment.png` — Bar plot of total trades by sentiment
- `rf_confusion_matrix.png` — Random Forest confusion matrix (if enough data)
- `rf_feature_importance.png` — Random Forest feature importance (if enough data)

---

## Customizing the Analysis
- **Add new features:** Edit `simple_analysis.py` to include more columns or engineered features in the `summary` DataFrame.
- **Change clustering:** Adjust the number of clusters in the KMeans section for finer or coarser segmentation.
- **Try new models:** Swap out Random Forest for other classifiers (e.g., Logistic Regression, XGBoost) in the ML section.
- **Visualizations:** Add or modify plots to explore other relationships or time periods.
- **Statistical tests:** Add more tests (e.g., Mann-Whitney U, Kruskal-Wallis) for non-parametric analysis.

---

## Troubleshooting & Tips
- **No plots appear?** Make sure you have a GUI environment. If running on a server, use Jupyter or save plots and download them.
- **Missing columns:** Check your CSVs for typos or missing headers. Column names are case-insensitive but must match those in the script.
- **Not enough data for ML/clustering:** The script requires a minimum number of samples for some analyses. Add more data for robust results.
- **Warnings about KDE/skipped plots:** These occur if a group has only one value. This is normal and does not affect other results.
- **Dependency errors:** Install all required packages (see below).

---

## Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy

Install dependencies with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

---

## Example Usage
```bash
python simple_analysis.py
```

---

## FAQ
**Q: Can I use my own trading or sentiment data?**  
A: Yes! Just format your CSVs to match the data dictionary above.

**Q: How do I add more visualizations?**  
A: Edit the Visualization section in `simple_analysis.py` and use matplotlib/seaborn as needed.

**Q: Can I run this on a cloud server?**  
A: Yes, but interactive plots may not display. Use `plt.savefig()` and download the images.

**Q: How do I interpret the clustering results?**  
A: Each cluster groups accounts with similar trading behaviors. Use the summary stats to profile each group.

---

## Contributing & Contact
- Pull requests and suggestions are welcome!
- For questions or collaboration, contact Vishal Deep at vishalyep1022@gmail.com
- Please cite this project if used in academic or professional work.

---

## License
This project is for educational and research purposes. 
