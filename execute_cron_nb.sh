#!/bin/bash

#Running jupyter lab
jupyter lab --ip 0.0.0.0 --allow-root --no-browser

#Running FastAPI
uvicorn main:app --reload --host 0.0.0.0 --port 8000