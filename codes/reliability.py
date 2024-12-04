import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def comparetimes():
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

    #plt.savefig("reliabilityscores")

    return aggregated_scores

def reliabilityRegression():
    data = pd.read_csv("/workspaces/MBTA-Data-Analysis/data/commuterreliability.csv")

    data['service_date'] = pd.to_datetime(data['service_date'])
    data['year_month'] = data['service_date'].dt.to_period('M')

    data['reliability_score'] = data['otp_numerator'] / data['otp_denominator']

    print(data, "\n")

    data = data[(data['reliability_score'].notna()) & (data['otp_denominator'] > 0)]

    monthly_data = data.groupby(['year_month', 'gtfs_route_long_name']).agg(
        reliability_score=('reliability_score', 'mean')
    ).reset_index()

    monthly_data['month_number'] = monthly_data['year_month'].dt.month

    print(monthly_data)

    train_data = monthly_data[monthly_data['year_month'] < '2024-01']
    test_data = monthly_data[monthly_data['year_month'] >= '2024-01']

    model = LinearRegression()
    X_train = train_data[['month_number']]
    y_train = train_data['reliability_score'] 
    model.fit(X_train, y_train)

    X_test = test_data[['month_number']]
    predictions = model.predict(X_test)

    print(predictions)

if __name__ == "__main__":
    
    comparetimes()

    reliabilityRegression()