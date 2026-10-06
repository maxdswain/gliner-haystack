FROM vllm/vllm-openai:v0.20.0

# vLLM Factory's GLiNER2.5 plugin lives in the source tree, not the older PyPI release.
COPY vendor/vllm-factory-fstn /opt/vllm-factory-fstn
RUN pip install --no-cache-dir -e /opt/vllm-factory-fstn \
    && pip install --no-cache-dir \
       'gliner2 @ git+https://github.com/fastino-ai/GLiNER2.git@af36b41cf948995b885cf6a33cc0bc4730be9620' \
       'gliner>=0.2.26'
