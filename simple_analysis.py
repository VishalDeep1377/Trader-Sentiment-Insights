# -----------------------------------
# 1. Import Libraries
# -----------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans

sns.set(style='whitegrid', palette='muted', font_scale=1.1)

# -----------------------------------
# 2. Load and Preprocess Data
# -----------------------------------
fear_greed_path = 'data/fear_greed_index.csv'
trades_path = 'data/historical_data.csv'

fear_greed = pd.read_csv(fear_greed_path, parse_dates=['date'])
trades = pd.read_csv(trades_path)
trades.columns = [col.strip().lower().replace(' ', '_') for col in trades.columns]

if 'timestamp_ist' in trades.columns:
    trades['date'] = pd.to_datetime(trades['timestamp_ist'], errors='coerce')
elif 'timestamp' in trades.columns:
    trades['date'] = pd.to_datetime(trades['timestamp'], errors='coerce', unit='s')
else:
    raise ValueError('No timestamp_ist or timestamp column found in trades data.')

fear_greed = fear_greed.dropna(subset=['date', 'classification'])
trades = trades.dropna(subset=['date', 'account'])
if 'closed_pnl' in trades.columns:
    trades['closed_pnl'] = pd.to_numeric(trades['closed_pnl'], errors='coerce').fillna(0)
if 'size_tokens' in trades.columns:
    trades['size_tokens'] = pd.to_numeric(trades['size_tokens'], errors='coerce').fillna(0)

# -----------------------------------
# 3. Merge Data
# -----------------------------------
merged = pd.merge(trades, fear_greed[['date', 'classification']], on='date', how='inner')

# -----------------------------------
# 4. Feature Engineering
# -----------------------------------
def win_rate(x):
    return (x > 0).sum() / len(x) if len(x) > 0 else 0

summary = merged.groupby(['account', 'date', 'classification']).agg(
    avg_closed_pnl=('closed_pnl', 'mean'),
    std_closed_pnl=('closed_pnl', 'std'),
    total_trades=('account', 'count'),
    avg_trade_size=('size_tokens', 'mean'),
    win_rate=('closed_pnl', win_rate),
    cumulative_pnl=('closed_pnl', 'sum')
).reset_index()

summary = summary.sort_values(['account', 'date'])
summary['rolling_7d_pnl'] = summary.groupby('account')['avg_closed_pnl'].transform(lambda x: x.rolling(7, min_periods=1).mean())

print('Sample of engineered features:')
print(summary.head())

# -----------------------------------
# 5. Exploratory Data Analysis (EDA)
# -----------------------------------
print("\nMean closed PnL by sentiment:")
print(summary.groupby('classification')['avg_closed_pnl'].mean())

print("\nMean total trades by sentiment:")
print(summary.groupby('classification')['total_trades'].mean())

# Correlation matrix
corr = summary[['avg_closed_pnl', 'std_closed_pnl', 'total_trades', 'avg_trade_size', 'win_rate', 'cumulative_pnl', 'rolling_7d_pnl']].corr()
print('\nCorrelation matrix:')
print(corr)

# -----------------------------------
# 6. Visualization
# -----------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(x='classification', y='avg_closed_pnl', data=summary, errorbar=None)
plt.title('Average Closed PnL by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Average Closed PnL')
plt.tight_layout()
plt.savefig('closedPnL_by_sentiment.png')
plt.show()
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
for sentiment in summary['classification'].unique():
    data = summary[summary['classification'] == sentiment]['avg_closed_pnl']
    if data.nunique() > 1:
        sns.kdeplot(data, label=sentiment, fill=True)
plt.title('KDE of Average Closed PnL by Sentiment')
plt.xlabel('Average Closed PnL')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('kde_closedPnL_by_sentiment.png')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
for sentiment in summary['classification'].unique():
    subset = summary[summary['classification'] == sentiment]
    plt.plot(subset['date'], subset['avg_closed_pnl'], marker='o', linestyle='-', label=sentiment)
plt.title('Time Series of Average Closed PnL by Sentiment')
plt.xlabel('Date')
plt.ylabel('Average Closed PnL')
plt.legend()
plt.tight_layout()
plt.savefig('timeseries_closedPnL_by_sentiment.png')
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(x='classification', y='total_trades', data=summary, errorbar=None)
plt.title('Total Trades by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Total Trades')
plt.tight_layout()
plt.savefig('total_trades_by_sentiment.png')
plt.show()
plt.close()

# -----------------------------------
# 7. Statistical Testing
# -----------------------------------
fear_pnl = summary[summary['classification'] == 'Fear']['avg_closed_pnl']
greed_pnl = summary[summary['classification'] == 'Greed']['avg_closed_pnl']
if len(fear_pnl) > 1 and len(greed_pnl) > 1:
    t_stat, p_val = stats.ttest_ind(fear_pnl.dropna(), greed_pnl.dropna(), equal_var=False)
    print(f"\nT-test for avg_closed_pnl (Fear vs Greed): t={t_stat:.2f}, p={p_val:.4f}")

if summary['classification'].nunique() > 1 and all(summary.groupby('classification').size() > 1):
    anova = stats.f_oneway(
        *[summary[summary['classification'] == c]['avg_closed_pnl'].dropna() for c in summary['classification'].unique()]
    )
    print(f"ANOVA for avg_closed_pnl across sentiments: F={anova.statistic:.2f}, p={anova.pvalue:.4f}")

# -----------------------------------
# 8. Machine Learning Modeling
# -----------------------------------
clf_data = summary[summary['classification'].isin(['Fear', 'Greed'])].copy()
clf_data['target'] = clf_data['classification'].map({'Fear': 0, 'Greed': 1})
features = ['avg_closed_pnl', 'std_closed_pnl', 'total_trades', 'avg_trade_size', 'win_rate', 'cumulative_pnl', 'rolling_7d_pnl']
X = clf_data[features].fillna(0)
y = clf_data['target']

if len(clf_data['target'].unique()) == 2 and len(clf_data) > 10:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print('\nRandomForest Classification Report:')
    print(classification_report(y_test, y_pred, target_names=['Fear', 'Greed']))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    print('\nFeature Importances (RandomForest):')
    print(importances)
    # Confusion Matrix Plot
    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fear', 'Greed'], yticklabels=['Fear', 'Greed'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Random Forest Confusion Matrix')
    plt.tight_layout()
    plt.savefig('rf_confusion_matrix.png')
    plt.show()
    plt.close()
    # Feature Importance Plot
    plt.figure(figsize=(8, 5))
    importances.sort_values().plot(kind='barh')
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png')
    plt.show()
    plt.close()

# -----------------------------------
# 9. Account Segmentation (Clustering)
# -----------------------------------
clustering_features = ['avg_closed_pnl', 'total_trades', 'win_rate', 'avg_trade_size']
clustering_data = summary[clustering_features].fillna(0)
if len(summary) >= 3:
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    summary['cluster'] = kmeans.fit_predict(clustering_data)
    print("\nCluster counts:")
    print(summary['cluster'].value_counts())
else:
    print("\nNot enough data for clustering.")

# -----------------------------------
# 10. Insights and Recommendations
# -----------------------------------
print("""
Key Insights:
- See above for group means, correlations, and (if enough data) statistical and ML results.
- Clustering (if enough data) reveals distinct trader behavior groups.

Recommendations:
- Monitor key features (PnL, win rate, trade size) as sentiment signals.
- Use model outputs to inform risk management or strategy adjustments.
- Further research: try more features, time-series models, or external data.
""")