import os
import io
import json
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tutor import EnglishTutor
from memory import TutorMemory
from voice_service import transcribe_audio_file, generate_speech_audio

app = FastAPI(title="Elena - AI Spoken English Coach")

# Initialize Tutor and Memory
tutor = EnglishTutor(model_name="gpt-oss:120b-cloud")
memory = TutorMemory()

# Ensure static folder exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

class TextMessageRequest(BaseModel):
    message: str
    topic: Optional[str] = None
    voice: Optional[str] = "af_bella"
    speed: Optional[float] = 1.0

@app.post("/api/chat/text")
async def chat_text(req: TextMessageRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    # 1. Get tutor analysis & response from Ollama
    result = tutor.chat(user_input=req.message, topic=req.topic)
    
    # 2. Generate Kokoro voice audio
    spoken_text = result.get("spoken_response", "")
    audio_bytes = generate_speech_audio(spoken_text, voice=req.voice or "af_bella", speed=req.speed or 1.0)
    has_audio = audio_bytes is not None

    return {
        "user_text": req.message,
        "tutor_feedback": result,
        "has_audio": has_audio
    }

@app.post("/api/chat/audio")
async def chat_audio(
    audio: UploadFile = File(...),
    topic: Optional[str] = Form(None),
    voice: Optional[str] = Form("af_bella"),
    speed: Optional[float] = Form(1.0)
):
    # 1. Read uploaded audio
    audio_content = await audio.read()
    if not audio_content:
        raise HTTPException(status_code=400, detail="No audio received")

    # 2. STT via faster-whisper (base.en)
    try:
        user_transcript = transcribe_audio_file(audio_content)
    except Exception as e:
        print(f"[STT Error] {e}")
        user_transcript = ""

    if not user_transcript.strip():
        return JSONResponse(
            status_code=400,
            content={"detail": "Could not understand audio. Please try speaking clearly."}
        )

    # 3. AI Tutor via Ollama
    result = tutor.chat(user_input=user_transcript, topic=topic)

    # 4. Return transcript and feedback
    return {
        "user_text": user_transcript,
        "tutor_feedback": result,
        "has_audio": True
    }

@app.get("/api/tts")
async def get_tts_audio(text: str, voice: str = "af_bella", speed: float = 1.0):
    audio_bytes = generate_speech_audio(text, voice=voice, speed=speed)
    if audio_bytes:
        return Response(content=audio_bytes, media_type="audio/wav")
    else:
        # 404 or empty fallback
        raise HTTPException(status_code=500, detail="Could not generate TTS audio")

@app.get("/api/memory")
async def get_memory():
    profile = memory.get_all_profile()
    mistakes = memory.get_common_mistakes(limit=15)
    vocab = memory.get_vocabulary_list(limit=30)
    history = memory.get_recent_history(limit=20)
    return {
        "profile": profile,
        "mistakes": mistakes,
        "vocabulary": vocab,
        "history": history
    }

@app.post("/api/memory/profile")
async def update_profile(level: str = Form(...), goal: str = Form(...), interests: str = Form(...)):
    memory.set_profile_field("proficiency_level", level)
    memory.set_profile_field("learning_goal", goal)
    memory.set_profile_field("interests", interests)
    return {"status": "success"}

@app.get("/api/models")
async def get_models():
    models = tutor.get_available_models()
    return {"models": models, "current": tutor.model_name}

@app.post("/api/model/select")
async def select_model(model_name: str = Form(...)):
    tutor.model_name = model_name
    return {"status": "success", "current": tutor.model_name}

@app.get("/api/session/analyze")
async def get_session_analysis():
    from report_generator import analyze_session
    transcript = memory.get_recent_history(limit=25)
    mistakes = memory.get_common_mistakes(limit=15)
    profile = memory.get_all_profile()
    
    analysis = analyze_session(
        transcript=transcript,
        mistakes=mistakes,
        profile=profile,
        model_name=tutor.model_name
    )
    return analysis

@app.get("/api/session/report/pdf")
async def download_session_pdf(student_name: str = "Student"):
    from report_generator import analyze_session, build_pdf_report
    transcript = memory.get_recent_history(limit=25)
    mistakes = memory.get_common_mistakes(limit=15)
    profile = memory.get_all_profile()
    
    analysis = analyze_session(
        transcript=transcript,
        mistakes=mistakes,
        profile=profile,
        model_name=tutor.model_name
    )
    
    pdf_bytes = build_pdf_report(analysis, student_name=student_name)
    filename = f"Elena_Spoken_English_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
