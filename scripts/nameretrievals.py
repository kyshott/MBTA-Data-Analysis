import requests

def retrieveroutesprint() -> None:
    """
    Retrieve the names of all routes; print the parsed json data.
    
    Args:
        None
    Returns:
        None
    """
    url = "https://api-v3.mbta.com/routes"

    response = requests.get(url)
    data = response.json()

    routes = [(route['attributes']['long_name'], route['id']) for route in data['data']]

    for route_name, route_id in routes:
        print(f"Route Name: {route_name}, Route ID: {route_id}")


def retrieverouteswrite() -> None:
    """
    Retrieve the names of all routes; write the parsed json data to a .txt file.

    Args:
        None
    Returns:
        None
    """
    url = "https://api-v3.mbta.com/routes"

    response = requests.get(url)
    data = response.json()

    routes = [(route['attributes']['long_name'], route['id']) for route in data['data']]

    with open("routenames.txt", "w") as file:
        for route_name, route_id in routes:
            file.write(f"Route Name: {route_name}, Route ID: {route_id}\n")


def retrievestopsprint() -> None:
    """
    Retrieve the names of all stops; print the parsed json data.

    Args:
        None
    Returns:
        None
    """
    url = "https://api-v3.mbta.com/stops"
    
    response = requests.get(url)
    data = response.json()
    
    stops = [(stop['attributes']['name'], stop['id']) for stop in data['data']]
        
    for stop_name, stop_id in stops:
        print(f"Stop Name: {stop_name}, Stop ID: {stop_id}")


def retrievestopswrite() -> None:
    """
    Retrieve the names of all stops; write the parsed json data to a .txt file.

    Args:
        None
    Returns:
        None
    """
    url = "https://api-v3.mbta.com/stops"
    
    response = requests.get(url)
    data = response.json()
    
    stops = [(stop['attributes']['name'], stop['id']) for stop in data['data']]

    with open("stopnames.txt", "w") as file:
        for stop_name, stop_id in stops:
            file.write(f"Stop Name: {stop_name}, Stop ID: {stop_id}\n")
            

if __name__ == "__main__":
    retrieveroutesprint()
    print("\n")
    retrievestopsprint()
