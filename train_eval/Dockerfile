ARG PYTHON_VERSION="3.9.16-slim"
FROM python:$PYTHON_VERSION as base

ENV VIRTUALENV=/opt/venv
RUN python3 -m venv $VIRTUALENV
ENV PATH="$VIRTUALENV/bin:$PATH"
COPY requirements.txt ./tmp/requirements.txt
RUN pip3 install --upgrade pip setuptools && \
    pip3 install --no-cache-dir -r ./tmp/requirements.txt

FROM python:3.9.16-slim as train

WORKDIR /container_files
ARG USERNAME="manul"
RUN useradd --no-log-init -u 1001 -r -m -g users $USERNAME
RUN chown -R $USERNAME:users /container_files
USER $USERNAME

COPY --from=base /opt/venv /opt/venv
COPY net.py train.py ./

ENV VIRTUALENV=/opt/venv
ENV PATH="$VIRTUALENV/bin:$PATH"
CMD ["python3", "train.py"]

FROM train as eval

WORKDIR /container_files

COPY eval.py ./
CMD ["python3", "eval.py"]