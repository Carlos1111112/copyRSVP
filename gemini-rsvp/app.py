import os
import json
import uuid
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# --- Gemini ---
import google.generativeai as genai

# =========================
# Config
# =========================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Falta GEMINI_API_KEY en .env")

genai.configure(api_key=GEMINI_API_KEY)
# Modelos buenos para texto estructurado: "gemini-1.5-flash" (rápido) o "gemini-1.5-pro" (mejor)
MODEL_NAME = "gemini-1.5-flash"

# =========================
# Schemas (tu formato)
# =========================

class QuizItem(BaseModel):
    # Formato EXACTO como tus archivos (ver learned_questions*.json)
    # pregunta: enunciado; opciones: lista; respuesta: índice correcto; explanation: texto
    pregunta: str
    opciones: List[str]
    respuesta: int
    explanation: str = Field(..., description="Breve justificación de la respuesta")

class QuizPayload(BaseModel):
    texto: str
    preguntas: List[QuizItem]

class RSVPIn(BaseModel):
    topic: str = Field(..., example="física cuántica")
    palabras_objetivo: int = Field(250, ge=80, le=800, description="Largo aproximado del texto")
    num_preguntas: int = Field(5, ge=3, le=12)

class RSVPOut(BaseModel):
    id: str
    texto: str
    palabras: List[str]

class QuizIn(BaseModel):
    texto: str = Field(..., description="Texto base para generar preguntas")
    num_preguntas: int = Field(5, ge=3, le=12)

class QuizOut(BaseModel):
    rsvp_session_id: str
    preguntas: List[QuizItem]

# =========================
# FastAPI
# =========================
app = FastAPI(title="FastAPI (Gemini) RSVP", version="0.1.0")

# =========================
# Prompts
# =========================

PROMPT_TEXTO = """Eres un redactor claro y riguroso.
Escribe un texto informativo en español sobre el tema: "{topic}".
Requisitos:
- Extensión aproximada: {palabras_objetivo} palabras (±10%).
- Lenguaje sencillo pero preciso; párrafos cortos.
- Incluye contexto, idea principal y 2–3 detalles clave.

Responde SOLO con el texto, sin prefijos, sin comillas.
"""

PROMPT_QUIZ = """Eres un generador de preguntas tipo test (opción múltiple) en español.
Debes devolver JSON **VÁLIDO** con este esquema EXACTO:

{{
  "texto": "<resumen de 1-2 frases del tema>",
  "preguntas": [
    {{
      "pregunta": "<enunciado claro>",
      "opciones": ["A", "B", "C", "D"],
      "respuesta": 0,   // índice de la opción correcta
      "explanation": "justificación breve y verificable"
    }}
  ]
}}

Reglas:
- Genera exactamente {num_preguntas} preguntas.
- Una sola respuesta correcta por pregunta.
- Las opciones deben ser plausibles pero sólo una correcta.
- Las preguntas deben poder resolverse leyendo el texto dado.
- El JSON debe parsear sin errores.

TEXTO BASE (úsa-lo como fuente):
\"\"\"{texto_base}\"\"\"
"""

# =========================
# Gemini helpers
# =========================

def call_gemini(prompt: str) -> str:
    model = genai.GenerativeModel(MODEL_NAME)
    resp = model.generate_content(prompt)
    if not resp or not resp.candidates:
        raise HTTPException(status_code=502, detail="Gemini no devolvió contenido")
    return resp.text.strip()

def safe_json_loads(payload: str):
    # Gemini a veces rodea con backticks o texto extra; intenta limpiar
    txt = payload.strip()
    # recorta bloques de código si vienen
    if txt.startswith("```"):
        txt = txt.strip("`")
        # puede venir como ```json ... ```
        first_brace = txt.find("{")
        last_brace = txt.rfind("}")
        if first_brace != -1 and last_brace != -1:
            txt = txt[first_brace:last_brace+1]
    return json.loads(txt)

# =========================
# Endpoints
# =========================

@app.post("/api/rsvp", response_model=RSVPOut, summary="Generar texto (RSVP) con Gemini")
def generate_rsvp(payload: RSVPIn):
    prompt = PROMPT_TEXTO.format(
        topic=payload.topic,
        palabras_objetivo=payload.palabras_objetivo
    )
    texto = call_gemini(prompt)
    palabras = texto.split()
    return RSVPOut(id=str(uuid.uuid4()), texto=texto, palabras=palabras)

@app.post("/api/quiz", response_model=QuizOut, summary="Generar preguntas con el FORMATO solicitado")
def generate_quiz(payload: QuizIn):
    prompt = PROMPT_QUIZ.format(
        texto_base=payload.texto,
        num_preguntas=payload.num_preguntas
    )
    raw = call_gemini(prompt)
    try:
        data = safe_json_loads(raw)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Respuesta de Gemini no es JSON válido: {e}")

    # Validación con Pydantic (asegura tu formato)
    try:
        quiz = QuizPayload(**data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Estructura de quiz inválida: {e}")

    # Asigna un ID de sesión sintético (útil si luego guardas)
    return QuizOut(rsvp_session_id=str(uuid.uuid4()), preguntas=quiz.preguntas)

from pathlib import Path
from fastapi.responses import HTMLResponse, FileResponse

BASE_DIR = Path(__file__).parent

@app.get("/", response_class=HTMLResponse)
def landing():
    html_path = BASE_DIR / "landing.html"
    if not html_path.exists():
        return {"error": "landing.html no encontrado", "docs": "/docs"}
    # Puedes usar cualquiera de las dos líneas siguientes (las dos funcionan):
    # return html_path.read_text(encoding="utf-8")
    return FileResponse(str(html_path))