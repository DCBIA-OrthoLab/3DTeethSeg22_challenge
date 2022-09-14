FROM python:3.9-bullseye

RUN apt-get update && \
    apt-get install -y git \
    wget\
    -y unzip

RUN apt install -y libgl1-mesa-glx
######


RUN apt-get install -y gcc


RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm


RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output
USER algorithm


WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

# Copy all required files such that they are available within the docker image (code, weights, ...)
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm checkpoints/ /opt/algorithm/checkpoints/
COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm utils/ /opt/algorithm/utils/

# Install required python packages via pip - you may adapt the requirements.txt to your needs
# RUN pip install --no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113


RUN python -m pip install --user -r requirements.txt


# RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html
RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1120/download.html


RUN pip install requests
# Entrypoint to your python code - executes process.py as a script
ENTRYPOINT python -m process $0 $@













