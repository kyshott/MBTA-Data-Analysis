import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

"""
A few notes regarding this analysis:
Some data needed to be adjusted in the CSV file to account for specific factors, such as line closures,
COVID-19, etc. After fine tuning, the average R-squared value is about 0.93, with an average Mean Absolute
Error of around 5700, so the model seems to have done quite well at predictions.
"""

def rdshLineChart():
    """
    Return a Pandas dataframe for the ridership CSV file, as well as 
    generate a matplotlib chart for visualization.

    Args:
        None
    Returns:
        Pandas dataframe
    """
    file_path = "/workspaces/MBTA-Data-Analysis/data/commuterridership2024.csv"
    data = pd.read_csv(file_path, index_col=0)

    plt.figure(figsize=(12, 6))
    for year in data.columns:
        plt.plot(data.index, data[year], marker='o', label=str(year))

    plt.title("MBTA Commuter Rail Ridership by Year", fontsize=16)
    plt.xlabel("Month", fontsize=14)
    plt.ylabel("Ridership", fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title="Year", fontsize=10, loc='lower right', bbox_to_anchor=(1, 0))

    plt.tight_layout()

    plt.savefig("ridership")

    return data

def predictRiders() -> None:
    """
    Make a random forest regressor model for predicting commuter rail ridership, 
    factoring in COVID-19 trends as well as lower ridership / ticket collection
    due to line closures. Then, generate a graph based on prediction.

    Args:
        None
    Returns:
        None
    """
    data = rdshLineChart()
    
    data = data.T.reset_index()
    print(data, "\n")
    data = data.melt(id_vars=["index"], var_name="Month", value_name="Rider_Count")
    data.columns = ["Year", "Month", "Rider_Count"]
    
    data["Month"] = pd.to_datetime(data["Month"], format="%B").dt.month
    data["Year"] = data["Year"].astype(int)
    data["Is_Pre_COVID"] = (data["Year"] < 2020).astype(int)

    X = data[["Year", "Month", "Is_Pre_COVID"]]
    y = data["Rider_Count"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared: {r2}")

    data["Predicted"] = model.predict(X)

    print(data)

    post_2020_data = data[data["Year"] >= 2020]

    month_growth_rates = (
        data.groupby("Month")["Rider_Count"]
        .apply(lambda x: (x.values[-1] - x.values[0]) / (x.index[-1] - x.index[0] + 1))
    )

    future_years = [2025]
    future_predictions = []

    for year in future_years:
        predictions = []
        for month in range(1, 13):
            latest_value = data[
                (data["Year"] == 2024) & (data["Month"] == month)
            ]["Rider_Count"].values[0]
            
            predicted_value = latest_value + month_growth_rates[month] * (year - 2024)
            predictions.append(predicted_value)
        
        future_predictions.append(pd.Series(predictions, index=range(1, 13), name=year))

    future_df = pd.concat(future_predictions, axis=1)
    future_df.columns = ["2025"]

    plt.figure(figsize=(12, 6))

    for year in future_df.columns:
        plt.plot(future_df.index, future_df[year], label=f"Predicted ridership for {year}", marker="o")
    
    plt.xlabel("Month")
    plt.ylabel("Rider Count")
    plt.title("Predicted Monthly MBTA Commuter Rail Ridership for 2025")
    plt.xticks(range(1, 13), 
               ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    legend_text = f"R² = {r2:.2f}, MAE = {mae:.0f}"
    plt.legend(title=legend_text)
    plt.grid(True)
    plt.savefig("ridershippredictions")
    
if __name__ == "__main__":
    
    predictRiders()