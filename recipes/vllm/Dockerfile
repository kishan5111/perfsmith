ARG VLLM_IMAGE=vllm/vllm-openai:v0.17.1
FROM ${VLLM_IMAGE}

WORKDIR /workspace/perfsmith

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    VLLM_NO_USAGE_STATS=1

COPY pyproject.toml README.md ./
COPY perfsmith ./perfsmith
COPY fixtures ./fixtures
COPY recipes ./recipes

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -e .

ENTRYPOINT ["/bin/bash"]
