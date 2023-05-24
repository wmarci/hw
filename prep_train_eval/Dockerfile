ARG PYTHON_VERSION="3.9.16-slim"
FROM python:$PYTHON_VERSION as base

ENV VIRTUALENV=/opt/venv
RUN python3 -m venv $VIRTUALENV
ENV PATH="$VIRTUALENV/bin:$PATH"
COPY ./prep_train_eval/requirements.txt ./tmp/requirements.txt
RUN pip3 install --upgrade pip setuptools && \
    pip3 install --no-cache-dir -r ./tmp/requirements.txt

FROM python:3.9.16-slim as prep

WORKDIR /container_files
ARG USERNAME="manul"
RUN useradd --no-log-init -u 1001 -r -m -g users $USERNAME
RUN chown -R $USERNAME:users /container_files
USER $USERNAME

COPY --from=base /opt/venv /opt/venv
COPY ./prep_train_eval/net.py ./prep_train_eval/prep.py ./
COPY ./common ./common

ENV STAGE="data_prep"
ENV VIRTUALENV=/opt/venv
ENV PATH="$VIRTUALENV/bin:$PATH"
CMD ["python3", "prep.py"]

FROM prep as train

WORKDIR /container_files

ENV STAGE="train"
COPY ./prep_train_eval/train.py ./
CMD ["python3", "train.py"]

FROM prep as eval

WORKDIR /container_files

ENV STAGE="eval"
COPY ./prep_train_eval/eval.py ./
CMD ["python3", "eval.py"]