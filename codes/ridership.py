import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def rdshLineChart():
    """
    Return a Pandas dataframe for the ridership CSV file, as well as 
    generate a matplotlib chart for visualization.

    Args:
        None
    Returns:
        Pandas dataframe
    """
    file_path = "/workspaces/MBTA-Data-Analysis/data/commuterridership.csv"
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

def predictRiders():
    """
    Make a predictive model for the monthly riders.

    Args:
        None
    Returns:
        None
    """
    file_path = "/workspaces/MBTA-Data-Analysis/data/commuterridership.csv"
    data = pd.read_csv(file_path, index_col=0)
    
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
    
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolor='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    plt.title("Predicted vs. Actual Rider Counts")
    plt.xlabel("Actual Rider Counts")
    plt.ylabel("Predicted Rider Counts")
    plt.grid(alpha=0.3)
    plt.savefig("scattered")

    data["Predicted"] = model.predict(X)

    print(data)
    
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data["Rider_Count"], label="Actual", marker='o')
    plt.plot(data["Predicted"], label="Predicted", linestyle='--', marker='x')
    plt.legend()
    plt.title("Actual vs. Predicted Rider Counts Over Time")
    plt.xlabel("Time (Year-Month)")
    plt.ylabel("Rider Counts")
    plt.grid(alpha=0.3)
    plt.xticks(np.arange(0, len(data), step=12), data["Year"].unique(), rotation=45)
    plt.tight_layout()
    plt.savefig("monthly_ridership_predictions")
    plt.show()
    """


if __name__ == "__main__":
    
    predictRiders()