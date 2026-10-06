# gliner-haystack

Serve GLiNER2.5 with vLLM Factory using Docker Compose. GPU and CPU images are available:

| Image | Example Compose file | Requirements |
| --- | --- | --- |
| `ghcr.io/maxdswain/vllm-gliner:latest` (GPU) | [compose.yaml](compose.yaml) | NVIDIA GPU and NVIDIA Container Toolkit |
| `ghcr.io/maxdswain/vllm-gliner:latest-cpu` (x86-64 CPU) | [compose.cpu.yaml](compose.cpu.yaml) | x86-64 CPU |

## Quick start

Install Docker with Compose, then:

```sh
git clone https://github.com/maxdswain/gliner-haystack.git
cd gliner-haystack

# GPU (default)
docker compose up

# Or CPU
docker compose -f compose.cpu.yaml up
```

The first start downloads the model. Once the server is ready, try:

```sh
curl http://localhost:8000/pooling \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/current","data":{"text":"Alice works at Acme in Paris.","schema":{"entities":{"person":"","organization":"","location":""}},"runtime":"classic","include_spans":true}}'
```
