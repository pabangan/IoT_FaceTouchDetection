# installing InfluxDB
# For Mac and Windows, you can find the downloads at https://portal.influxdata.com/downloads/.
# Click the v1.7.10 button under InfluxDB to find the download for your system.

# • If on Windows:
#   - Extract the downloaded zip folder to a location of your choosing and then you may wish to add it to your
#   - system PATH (e.g., C:\influxdb-1.7.10-1).
#   - Open a command line and start the Influx service by running the command “influxd.exe” <--------------------------
#   - Leave that command window open and then open another new command line window

# To open the InfluxDB CLI, enter the following:
#   influx -precision=rfc3339  <---------------------------------------------------------------------------------------

#       create database IoTProjectDB
#       use IoTProjectDB

# pip install influxdb

# While running your Python program, you can check that points are being written into your database by
# using the Influx command line interface (CLI) from another command / terminal window:
#  >use lab4db
#  >select * from cpu1

# installing Grafana:
# For Mac and Windows, you can find the downloads at https://grafana.com/grafana/download.
# # • If on Windows:
#   - Windows install guide: https://grafana.com/docs/grafana/latest/installation/windows/
#   - After installing, Grafana should start running in the background

# localhost:3000

# !/usr/bin/env python3
import paho.mqtt.client as mqtt
import json
import time
from influxdb import InfluxDBClient
import pprint  # pretty printer to print dictionary
from datetime import datetime


# def on_connect(client, userdata, flags, rc):
#     print("Connected with result code " + str(rc))
#     client.subscribe("uiowa/iot/lab4/#")

# python app2.py --ip localhost --port 8000

def on_message(val):
    influx_client = InfluxDBClient(host='localhost', port=8086, username='root', password='root',
                                   database='IoTProjectDB')
    point_data = {
        "measurement": "camerastream",
        "fields": {
            "touched_face": val
        }
    }
    influx_client.write_points([point_data])

# Initialize the MQTT client that should connect to the Mosquitto broker
# mqtt_client = mqtt.Client()
# mqtt_client.on_connect = on_connect
# mqtt_client.on_message = on_message
# mqtt_client.on_client = on_client
# connOK = False
# while (connOK == False):
#     try:
#         mqtt_client.connect("broker.hivemq.com", 1883, 60)
#         connOK = True
#     except:
#         connOK = False
#     time.sleep(1)
#
# # Blocking loop to the Mosquitto broker
# mqtt_client.loop_forever()
