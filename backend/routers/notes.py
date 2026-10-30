from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from supabase import create_client
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import base64
from fastapi import UploadFile, File

load_dotenv()

router = APIRouter()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)
gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ── Models ─────────────────────────────────────────────
class NoteCreate(BaseModel):
    book_id:     str
    user_id:     str
    page_number: int
    page_label:  str = ""
    content:     str

class NoteEnhance(BaseModel):
    book_id: str
    content: str


# ── CRUD ───────────────────────────────────────────────
@router.post("/notes")
async def create_note(note: NoteCreate):
    if not note.content.strip():
        raise HTTPException(status_code=400, detail="Note content cannot be empty.")
    res = supabase.table("notes").insert({
        "book_id":     note.book_id,
        "user_id":     note.user_id,
        "page_number": note.page_number,
        "page_label":  note.page_label,
        "content":     note.content.strip(),
    }).execute()
    return res.data[0] if res.data else {}


@router.get("/notes/{book_id}")
async def get_notes(book_id: str, user_id: str):
    res = supabase.table("notes") \
        .select("*") \
        .eq("book_id", book_id) \
        .eq("user_id", user_id) \
        .order("page_number", desc=False) \
        .execute()
    return res.data or []


@router.patch("/notes/{note_id}")
async def update_note(note_id: str, user_id: str, content: str):
    if not content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty.")
    res = supabase.table("notes") \
        .update({"content": content.strip()}) \
        .eq("id", note_id) \
        .eq("user_id", user_id) \
        .execute()
    return res.data[0] if res.data else {}


@router.delete("/notes/{note_id}")
async def delete_note(note_id: str, user_id: str):
    supabase.table("notes") \
        .delete() \
        .eq("id", note_id) \
        .eq("user_id", user_id) \
        .execute()
    return {"deleted": note_id}


# ── AI Enhance ─────────────────────────────────────────
@router.post("/notes/enhance")
async def enhance_note(body: NoteEnhance):
    if not body.content.strip():
        raise HTTPException(status_code=400, detail="Note content is empty.")

    system_prompt = """You are a study note enhancer for students.
Take the user's rough note and:
1. Fix grammar and spelling
2. Expand key concepts with brief context
3. Structure it clearly with natural flow
4. Keep it concise — no longer than 4-5 sentences unless the original is long
5. Preserve the original meaning exactly — do NOT add unrelated content

Return ONLY the enhanced note text. No headings, no bullet points unless the
original used them, no preamble like "Here is the enhanced note:"."""

    try:
        response = gemini.models.generate_content(
            model="gemini-3-flash-preview",
            contents=body.content.strip(),
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
            ),
        )
        return {"enhanced": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI enhance error: {str(e)}")
    

class NoteOCR(BaseModel):
    image_base64: str   # base64 PNG of the selected region
    book_id: str

@router.post("/notes/ocr")
async def ocr_note(body: NoteOCR):
    """Extract text from a cropped image region using Gemini Vision."""
    if not body.image_base64:
        raise HTTPException(status_code=400, detail="No image provided.")

    try:
        # Decode base64 to bytes
        image_bytes = base64.b64decode(body.image_base64)

        response = gemini.models.generate_content(
            model="gemini-2.0-flash",   # vision-capable model
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                "Extract all text from this image exactly as written. "
                "Return only the raw text with no formatting, labels, or commentary. "
                "If there is no readable text, return an empty string."
            ],
            config=types.GenerateContentConfig(temperature=0.0),
        )
        text = (response.text or "").strip()
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR error: {str(e)}")