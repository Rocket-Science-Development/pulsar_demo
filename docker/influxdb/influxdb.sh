#!/bin/bash
set -e

influx -execute "CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASSWORD}' WITH ALL PRIVILEGES"
influx -execute "CREATE DATABASE PULSAR_DATA_COLLECTION"


cat >> /etc/influxdb/influxdb.conf <[http]
  auth-enabled = true
EOD