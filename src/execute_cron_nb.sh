#!/bin/bash

#Running jupyter lab
jupyter lab --ip 0.0.0.0 --allow-root --no-browser

#Cron execution
cron -f