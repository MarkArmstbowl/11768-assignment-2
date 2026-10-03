FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

# Install pinned agent/verifier dependencies at BUILD time: grading is offline.
RUN pip install --no-cache-dir \
      matplotlib==3.10.5 \
      numpy==2.3.2 \
      pandas==2.3.1 \
      pillow==11.3.0 \
      pytest==8.4.1 \
      pytest-json-ctrf==0.3.5

# Use the same headless backend for the solution and the verifier's replay.
ENV MPLBACKEND=Agg
RUN python -c "import matplotlib.pyplot as plt; f=plt.figure(); f.savefig('/tmp/warm.png')" \
 && rm -f /tmp/warm.png

WORKDIR /app
COPY cohorts.csv /app/cohorts.csv
