import pymbta3 as mbta 

ats = mbta.Alerts(key='8f8266c4357e43d6ada8e3f193eb58ec')

def alert_counts():
    alerts = ats.get(stop='CR-Kingston')

    for alert in alerts['data']:
        print(alert['attributes']['short_header'])

if __name__ == "__main__":
    alert_counts()
