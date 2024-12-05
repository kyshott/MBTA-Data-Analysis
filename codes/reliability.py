import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

"""
A few notes about this analysis; the question to be answered is regarding that of 
whether or not the year can affect the reliability of an MBTA Commuter
Rail line. Specifically, this is done by using a machine learning model w/linear regression.
For higher R^2 values, the line is more affected over time. For lower or 0 R^2 values, the
passage of time has no effect at all. The MAE will demonstrate the accuracy of the predictions based on the
data given. This also provides insight into some patterns that riders can look out for with specific lines.
"""

def reliabilityScores():
    """
    Display the reliability scores for each MBTA Commuter Rail line based on
    the data provided from the CSV file; generate a descending bar chart.

    Args:
        None
    Returns:
        None
    """
    data = pd.read_csv("/workspaces/MBTA-Data-Analysis/data/commuterreliability.csv")

    print(data)

    data['reliability_score'] = data['otp_numerator'] / data['otp_denominator']

    aggregated_scores = data.groupby('gtfs_route_long_name')['reliability_score'].mean()

    aggregated_scores = aggregated_scores.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    aggregated_scores.plot(kind='bar', color='#80276c')
    plt.title('Aggregated Reliability Scores for MBTA Commuter Rail Lines')
    plt.xlabel('Commuter Rail Line')
    plt.ylabel('Reliability Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    #plt.savefig("reliabilityscores") Unused, as visualization is already complete; uncomment to use

    return aggregated_scores

def reliability_trend_by_year():
    """
    Generate a prediction for the reliability of each rail line for 2025 based on the
    historical data. Use linear regression to make this prediction.

    Args:
        None
    Returns:
        None
    """
    data = pd.read_csv("/workspaces/MBTA-Data-Analysis/data/commuterreliability.csv")
    data['service_date'] = pd.to_datetime(data['service_date'])
    data['year'] = data['service_date'].dt.year

    data['reliability_score'] = data['otp_numerator'] / data['otp_denominator']
    data = data[(data['reliability_score'].notna()) & (data['otp_denominator'] > 0)]
    yearly_data = data.groupby(['year', 'gtfs_route_long_name']).agg(
        reliability_score=('reliability_score', 'mean')
    ).reset_index()

    plt.figure(figsize=(12, 6))
    lines = yearly_data['gtfs_route_long_name'].unique()
    colormap = plt.cm.get_cmap('tab20', len(lines))
    colors = colormap.colors[:len(lines)]

    legend_entries = []

    for i, line in enumerate(lines):
        line_data = yearly_data[yearly_data['gtfs_route_long_name'] == line]

        X = line_data[['year']]
        y = line_data['reliability_score']
        model = LinearRegression()
        model.fit(X, y)

        future_years = pd.DataFrame({'year': np.arange(line_data['year'].min(), 2026)})
        predictions = model.predict(future_years)

        y_pred_train = model.predict(X)
        r2 = r2_score(y, y_pred_train)
        mae = mean_absolute_error(y, y_pred_train)

        plt.plot(
            future_years['year'], predictions, 
            color=colors[i], label=f"{line} (R²={r2:.2f}, MAE={mae:.2f})", 
            marker='o', markersize=5
        )

    plt.title("Predicted MBTA Commuter Rail Reliability by Year", fontsize=16)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Reliability Score", fontsize=12)
    plt.xticks(np.arange(yearly_data['year'].min(), 2026, 1))
    plt.legend(title="Commuter Rail Lines", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.4)
    plt.tight_layout()

    plt.savefig("reliabilitypredictionswith2025")

if __name__ == "__main__":

    reliabilityScores()

    reliability_trend_by_year()