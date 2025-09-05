import os
from typing import List, Tuple

from faster_whisper import WhisperModel

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model_name = os.getenv("ASR_MODEL_NAME", "large-v3")
    device_env = os.getenv("ASR_DEVICE", "auto").lower()
    compute_type = os.getenv("ASR_COMPUTE_TYPE", "default")

    # Selección de dispositivo
    if device_env == "cpu":
        device = "cpu"
    elif device_env == "cuda":
        device = "cuda"
    else:
        # auto: intenta cuda si está disponible
        try:
            import torch  # optional, solo para detectar cuda
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    _MODEL = WhisperModel(model_name, device=device, compute_type=compute_type)
    return _MODEL


def transcribe_audio(filepath: str, vad: bool = True) -> Tuple[str, List[Tuple[float, float, str]]]:
    """Transcribe el archivo y devuelve (texto, segmentos[(start, end, text)])."""
    model = get_model()

    segments, info = model.transcribe(
        filepath,
        task="transcribe",           # no traducir, solo transcribir
        vad_filter=vad,
        beam_size=5,
        best_of=5,
        condition_on_previous_text=True,
        temperature=0.0,             # estable
        no_speech_threshold=0.45,
    )

    out_text = []
    out_segments = []
    for seg in segments:
        out_text.append(seg.text.strip())
        out_segments.append((seg.start, seg.end, seg.text.strip()))

    return " ".join(out_text).strip(), out_segments


def to_srt(segments: List[Tuple[float, float, str]]) -> str:
    def fmt(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    lines = []
    for i, (start, end, text) in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{fmt(start)} --> {fmt(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)
