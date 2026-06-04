import psutil
import platform
import time

def get_system_health():

    # UPTIME
    boot_time = psutil.boot_time()

    uptime = int(time.time() - boot_time)

    # DISK USAGE
    try:

        disk = psutil.disk_usage("C:/").percent

    except:

        disk = 0

    # CPU TEMP
    cpu_temp = "N/A"

    try:

        temps = psutil.sensors_temperatures()

        if temps:

            for name, entries in temps.items():

                if len(entries) > 0:

                    cpu_temp = round(
                        entries[0].current,
                        1
                    )

                    break

    except:

        cpu_temp = "N/A"

    # FINAL DATA
    data = {

        "cpu": psutil.cpu_percent(interval=1),

        "memory": psutil.virtual_memory().percent,

        "disk": disk,

        "processes": len(psutil.pids()),

        "net_sent": round(
            psutil.net_io_counters().bytes_sent / (1024 * 1024),
            2
        ),

        "net_recv": round(
            psutil.net_io_counters().bytes_recv / (1024 * 1024),
            2
        ),

        "uptime": uptime,

        "system": platform.system(),

        "cpu_temp": cpu_temp
    }

    return data