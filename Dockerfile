FROM python:3.9.6-slim-buster

# Install dependencies
RUN apt-get update && apt-get install -y git gcc cron procps

# Set working directory
WORKDIR /app

# Copy application files
COPY requirements.txt requirements.txt
COPY src/compute_metrics_script.py compute_metrics_script.py
COPY src/gen_data_and_simulate_drift.py gen_data_and_simulate_drift.py
COPY src/fake_data_script.py fake_data_script.py
COPY src/training_script.py training_script.py
COPY src/roi.py roi.py
COPY src/class_Legendary_model.pkl class_Legendary_model.pkl
COPY datasets/ datasets/
COPY models/ models/

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy and set up cron job
COPY src/crontab/crontab /etc/cron.d/my-cron-job
RUN sed -i 's/\r$//' /etc/cron.d/my-cron-job && \
    chmod 0644 /etc/cron.d/my-cron-job

# Create log files for cron output
RUN touch /var/log/cron.log /var/log/cron_env.log

# Ensure cron picks up the job file in /etc/cron.d/
# Start cron and follow the log file
CMD ["cron", "-f"]