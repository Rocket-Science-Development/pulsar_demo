FROM amd64/python:3.9.6-slim-buster

RUN apt-get update && apt-get install -y git gcc cron

# Add crontab file in the cron directory
ADD ./src/crontab/crontab /etc/cron.d/crontab

# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/crontab

# Create the log file to be able to run tail
RUN touch /var/log/cron.log

WORKDIR /app

COPY requirements.txt requirements.txt
COPY src/compute_metrics_script.py compute_metrics_script.py
COPY src/gen_data_and_simulate_drift.py /app/gen_data_and_simulate_drift.py
COPY src/fake_data_script.py /app/fake_data_script.py
COPY src/training_script.py /app/training_script.py
COPY src/main.py .
COPY datasets/ datasets/
COPY models/ models/
COPY notebooks/pulsar_data_collection-0.0.0-py3-none-any.whl /app/
COPY execute_cron_nb.sh /app/execute_cron_nb.sh

RUN pip install python-multipart
RUN pip install influxdb-client
RUN pip install /app/pulsar_data_collection-0.0.0-py3-none-any.whl
RUN pip install python-dotenv
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

RUN /usr/bin/crontab /etc/cron.d/crontab

RUN chmod 0644 /app/execute_cron_nb.sh
RUN pip install jupyterlab

# Expose the port on which Jupyter Notebook will run
EXPOSE 8888

# Expose the port that the FastAPI application will run
EXPOSE 8000

# # Start the FastAPI app when the container starts
# CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]

# # Run Jupyter Notebook when the container launches
RUN pip install jupyterlab
CMD ["jupyter", "lab", "--ip", "0.0.0.0", "--allow-root", "--no-browser"]

# CMD ["bash", "execute_cron_nb.sh"]


# RUN adduser app && chown -R app /app
# USER app
#
# RUN python -m venv /app/.venv
# ENV PATH="/app/.venv/bin:$PATH"
#
# COPY requirements.txt /app/requirements.txt
# COPY main.py /app/main.py
# COPY kidney_disease.pkl /app/kidney_disease.pkl
#
# RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
#
# EXPOSE 8000
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["while true; do sleep 15; done"]
