import os
import pymbta3 as mbta
import matplotlib.pyplot as plt

def retrievestops() -> dict:
    """
    A function that retrieves and munges line/stop data into a readable
    dictionary.

    Args:
        None
    Returns:
        stops_dict (dict): The data parsed into a dictionary.
    """
    api_key = os.getenv('MBTA_API_KEY') 
    if not api_key:
        raise ValueError("API key not found. Please set the 'MBTA_API_KEY' environment variable.")

    stops_dict = {}

    rts = mbta.Routes(key=api_key)
    stp = mbta.Stops(key=api_key)

    routes = rts.get(type='2')

    for route in routes['data']:
        line_name = route['attributes']['long_name']
        
        stops = stp.get(route=route['id'])
        
        stops_dict[line_name] = len(stops['data'])
    
    return stops_dict


def df_to_image(stops_dict) -> None:
    """
    Converts a dictionary into a formatted image via matplotlib.

    Args:
        stops_dict (dict): A dictionary to convert to an image.
    Returns:
        None -> Generates visualization and outputs as a .png file into the 
        working directory.
    """
    sorted_stops = sorted(stops_dict.items(), key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(5, len(sorted_stops) * 0.5))
    ax.axis('off')

    plt.title("MBTA Commuter Rail Stop Counts", fontsize=14)

    table = ax.table(cellText=sorted_stops, colLabels=["Line", "Stop Count"], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.55, 1.55) 

    table[0, 0].set_facecolor('#D3D3D3')
    table[0, 1].set_facecolor('#D3D3D3') 

    fig.patch.set_facecolor('#FFF5E1')

    plt.subplots_adjust(top=0.9, bottom=0.4)

    plt.savefig("stopcountvisualization", bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    
    stops_dict = retrievestops()

    df_to_image(stops_dict)



