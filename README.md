# Transcriptor local (FastAPI + faster-whisper)

## Requisitos
- Docker (recomendado) o Python 3.11+
- Si NO usas Docker, instala `ffmpeg` en tu sistema

## Arranque rápido con Docker
```bash
# 1) Construir imagen
docker build -t transcriptor .

# 2) Ejecutar (CPU)
docker run --rm -p 8000:8000 --name transcriptor   -e ASR_MODEL_NAME=medium   -e ASR_DEVICE=cpu   -e ASR_VAD=true   transcriptor
```

Abre http://localhost:8000 y sube tu `.m4a`.

### Ejecutar con GPU (opcional, NVIDIA)
Requiere NVIDIA Container Toolkit previamente configurado.
```bash
docker run --rm -p 8000:8000 --name transcriptor   --gpus all   -e ASR_MODEL_NAME=large-v3   -e ASR_DEVICE=cuda   -e ASR_COMPUTE_TYPE=float16   -e ASR_VAD=true   transcriptor
```

## Ejecutar sin Docker (opcional)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ASR_MODEL_NAME=medium
export ASR_DEVICE=auto
export ASR_VAD=true
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Consejos
- `medium` en CPU = buen equilibrio.
- `large-v3` + GPU = mejor para suizo-alemán y entornos ruidosos.
- La detección de idioma es automática.
