import os
import requests
import pymbta3 as mbta

def comparetimes() -> None:
    api_key = os.getenv('MBTA_API_KEY')

    sch = mbta.Schedules(key=api_key)

    commsch = sch.get(type='2')

if __name__ == "__main__":
    
    comparetimes()