# alerts.py

def check_alerts(data):

    alerts = []

    status = "Healthy"

    if data["cpu"] > 80:

        alerts.append("🔥 High CPU Usage")

        status = "Warning"

    if data["memory"] > 80:

        alerts.append("⚠️ High Memory Usage")

        status = "Warning"

    if data["disk"] > 90:

        alerts.append("🚨 Disk Almost Full")

        status = "Critical"

    if data["cpu_temp"] != "N/A":

        if data["cpu_temp"] > 80:

            alerts.append("🌡️ High CPU Temperature")

            status = "Critical"

    return alerts, status