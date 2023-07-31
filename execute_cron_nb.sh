#!/bin/bash

RUN pip install jupyterlab

# Expose the port on which Jupyter Notebook will run
EXPOSE 8888
#Running jupyter lab
jupyter lab --ip 0.0.0.0 --allow-root --no-browser

# Expose the port that the FastAPI application will run
EXPOSE 8000
#Running FastAPI
uvicorn main:app --reload --host 0.0.0.0 --port 8000