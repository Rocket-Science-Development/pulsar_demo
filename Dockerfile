FROM python:3.9.6-slim-buster

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
COPY src/gen_data_and_simulate_drift.py gen_data_and_simulate_drift.py
COPY src/fake_data_script.py fake_data_script.py
COPY src/training_script.py training_script.py
COPY datasets/ datasets/
COPY models/ models/

RUN pip install --no-cache-dir -r requirements.txt

RUN /usr/bin/crontab /etc/cron.d/crontab

# Run the command on container startup
CMD ["cron", "-f"]


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
