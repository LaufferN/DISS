# Start from a core stack version
FROM jupyter/scipy-notebook:latest
USER root
RUN apt-get update && apt-get install -y \
    zlib1g-dev \
    graphviz
USER jovyan    
# Install from requirements.txt file
COPY requirements.txt /tmp/
COPY experiment.ipynb /home/jovyan/
RUN pip install wheel
RUN pip install --requirement /tmp/requirements.txt