import matplotlib.pyplot as plt
import pandas as pd

def rdshLineChart():

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

if __name__ == "__main__":
    
    rdshLineChart()