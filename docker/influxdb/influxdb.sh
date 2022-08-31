#!/bin/bash
set -e

influx -execute "CREATE USER ${INFLUXDB_USER} WITH PASSWORD '${INFLUXDB_USER_PASSWORD}' WITH ALL PRIVILEGES"

cat >> /etc/influxdb/influxdb.conf <[http]
  auth-enabled = true
EOD