import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

from .utils import transcribe_audio, to_srt

app = FastAPI(title="Transcriptor local", version="0.1.0")

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


@app.post("/transcribe", response_class=HTMLResponse)
async def transcribe_page(request: Request, file: UploadFile = File(...), want_srt: Optional[bool] = Form(False)):
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / file.filename
        content = await file.read()
        tmp_path.write_bytes(content)

        text, segments = transcribe_audio(str(tmp_path), vad=os.getenv("ASR_VAD", "true").lower() == "true")

        out_txt = Path(td) / (Path(file.filename).stem + ".txt")
        out_txt.write_text(text, encoding="utf-8")

        srt_name = None
        srt_text = None
        if want_srt:
            out_srt = Path(td) / (Path(file.filename).stem + ".srt")
            srt_text = to_srt(segments)
            out_srt.write_text(srt_text, encoding="utf-8")
            srt_name = out_srt.name

        return TEMPLATES.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result_text": text,
                "download_name": out_txt.name,
                "download_text": out_txt.read_text(encoding="utf-8"),
                "srt_name": srt_name,
                "srt_text": srt_text,
            },
        )


@app.post("/api/transcribe")
async def transcribe_api(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / file.filename
        content = await file.read()
        tmp_path.write_bytes(content)

        text, segments = transcribe_audio(str(tmp_path), vad=os.getenv("ASR_VAD", "true").lower() == "true")

        return JSONResponse({
            "text": text,
            "segments": [
                {"start": float(s), "end": float(e), "text": t} for s, e, t in segments
            ],
        })
