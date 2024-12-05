import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

"""
A few notes about this analysis; the question to be answered is regarding that of 
whether or not the time of the year can affect the reliability of an MBTA Commuter
Rail line. Specifically, this is done by using a machine learning model w/linear regression.
For higher R^2 values, the line is more affected by the month. For lower or 0 R^2 values, the
month has no effect at all. The MAE will demonstrate the accuracy of the predictions based on the
data given. This also provides insight into some patterns that riders can look out for with specific lines.
For example, the Fitchburg line seems to decline throughout the year, which can give riders insight into what months
the train will run on time. However, there is a lot of assumption to be made based off of the prediction model alone,
which is definitely a large shortcoming.
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

def reliabilityRegression():
    """
    Create a linear regression model for predicting the reliability scores for each
    commuter rail line based on month. This will determine if the month of the year has
    any major effect on any of the lines.

    Args:
        None
    Returns:
        None
    """
    data = pd.read_csv("/workspaces/MBTA-Data-Analysis/data/commuterreliability.csv")

    data['service_date'] = pd.to_datetime(data['service_date'])
    data['year_month'] = data['service_date'].dt.to_period('M')
    data['reliability_score'] = data['otp_numerator'] / data['otp_denominator']

    data = data[(data['reliability_score'].notna()) & (data['otp_denominator'] > 0)]
    data['month'] = data['service_date'].dt.month
    data['year'] = data['service_date'].dt.year

    monthly_data = data.groupby(['month', 'gtfs_route_long_name']).agg(
        reliability_score=('reliability_score', 'mean')
    ).reset_index()

    predictions = pd.DataFrame()
    lines = monthly_data['gtfs_route_long_name'].unique()

    colormap = cm.get_cmap('tab20', len(lines))
    colors = {line: colormap(i) for i, line in enumerate(lines)}

    plt.figure(figsize=(12, 8))
    for line in lines:
        line_data = monthly_data[monthly_data['gtfs_route_long_name'] == line].copy()

        X = line_data[['month']]
        y = line_data['reliability_score']

        model = LinearRegression()
        model.fit(X, y)

        line_data['predicted_reliability'] = model.predict(X)

        r2 = model.score(X, y)
        mae = mean_absolute_error(y, line_data['predicted_reliability'])

        predictions = pd.concat([predictions, line_data])

        plt.plot(
            line_data['month'],
            line_data['predicted_reliability'],
            label=f"{line} (R²: {r2:.2f}, MAE: {mae:.2f})",
            marker='o',
            color=colors[line]
        )

    plt.xlabel('Month')
    plt.ylabel('Predicted Reliability Score')
    plt.title('Predicted Reliability by Train Line (Aggregated by Month)')
    plt.xticks(
        ticks=range(1, 13),
        labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    )

    plt.legend(loc='lower left', fontsize=8)
    plt.tight_layout()
    plt.savefig("reliabilitypredictions")

if __name__ == "__main__":
    
    comparetimes()

    reliabilityRegression()