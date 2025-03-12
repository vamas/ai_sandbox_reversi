import asyncio
import socket
from datetime import datetime, timezone
from influxdb import InfluxDBClient

chart_colors = ["RED","YELLOW","GREEN","BLUE","PURPLE","CYAN","WHITE"]

training_stats = {
    "learner": "Default",
    "color": "WHITE",
    "run_id": None,
    "epoch": 0,
    "episode": 0,
    "loss": 0.0,
    "accuracy": 0.0,
    "score": 0.0,
    "avg_q_value_change": 0.0,
    "epsilon": 0.0,
    "learning_rate": 0.0,
    "training_time": 0.0
}

metrics_buffer_size = 100
metrics_buffer = []

influx_client = InfluxDBClient(host="192.168.0.181", port=8086, database="AIMLTraining")

async def periodic_logger_task(interval):
    while True:
        # Use the stats for any purpose, e.g., logging, sending to a dashboard, etc.
        # print("Periodic task reading training stats:", training_stats)
        try:
            send_metrics()
            await asyncio.sleep(interval)
        except Exception as e:
            raise e

def send_metrics():
    # Append the current metric first
    metrics_buffer.append((datetime.now(timezone.utc), training_stats.copy()))
    # Then check if the buffer is full
    if len(metrics_buffer) >= metrics_buffer_size:
        # Flush the buffer asynchronously (using a copy)
        write_points(metrics_buffer.copy())
        metrics_buffer.clear()

def write_points(buffer):
    data_points = [ {
                        "measurement": e[1]["learner"],
                        "tags": {
                            "run": e[1]["run_id"],
                            "host": socket.gethostname(),
                            "color": e[1]["color"]
                        },
                        "time": e[0],
                        "fields": {
                            "epoch": e[1]["epoch"],
                            "loss": e[1]["loss"],
                            "accuracy": e[1]["accuracy"],
                            "avg_q_value_change": e[1]["avg_q_value_change"],
                            "epsilon": e[1]["epsilon"],
                            "learning_rate": e[1]["learning_rate"],
                            "score": e[1]["score"],
                            "episode": e[1]["episode"],
                            "training_time": e[1]["training_time"]
                        }
                }  for e in buffer ]
    influx_client.write_points(data_points)
    influx_client.close()
