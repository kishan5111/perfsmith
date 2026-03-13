# Day-0 image: official vLLM runtime plus the local Perfsmith CLI.
# Day-0 image is pinned to vLLM 0.16.0 by default.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.16.0
FROM ${VLLM_IMAGE}

WORKDIR /workspace/perfsmith

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    VLLM_NO_USAGE_STATS=1

COPY pyproject.toml README.md ./
COPY perfsmith ./perfsmith
COPY fixtures ./fixtures
COPY day_0 ./day_0

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -e .

ENTRYPOINT ["/bin/bash"]
