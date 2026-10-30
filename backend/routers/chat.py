from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI, APIError, APITimeoutError, APIConnectionError
import os, json, base64, httpx
from supabase import create_client
from typing import Optional

load_dotenv()

router = APIRouter()

# ── Two OpenAI-compatible clients ─────────────────────────────────────────────

# Primary: OpenRouter  →  openai/gpt-oss-120b
openrouter_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": os.getenv("APP_URL", "http://localhost:3000"),  # required by OpenRouter
        "X-Title":      os.getenv("APP_NAME", "AI-Professor"),               # optional but recommended
    },
)
PRIMARY_MODEL = "nex-agi/nex-n2.5-pro:free"

# Fallback: Gemini via its OpenAI-compatible endpoint
gemini_client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
FALLBACK_MODEL = "gemini-3-flash-preview"

# ── Supabase ──────────────────────────────────────────────────────────────────
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY"),
)

# ── Language map ──────────────────────────────────────────────────────────────
LANG_MAP = {
    "hindi":   "Always respond in Hindi (हिंदी).",
    "urdu":    "Always respond in Urdu (اردو).",
    "arabic":  "Always respond in Arabic (العربية).",
    "bengali": "Always respond in Bengali (বাংলা).",
    "tamil":   "Always respond in Tamil (தமிழ்).",
    "english": "Always respond in English.",
}

# ── ERRORS that mean "try the fallback" ───────────────────────────────────────
_FALLBACK_ERRORS = (APIError, APITimeoutError, APIConnectionError)


# ════════════════════════════════════════════════════════════════════════════
# Core generation helper — tries primary, falls back to Gemini automatically
# ════════════════════════════════════════════════════════════════════════════

def _chat_complete(messages: list[dict], stream: bool = False):
    """
    Try OpenRouter gpt-oss-120b first.
    On any API-level failure fall back to Gemini flash.
    Returns a completion object or a stream iterator.
    """
    try:
        if stream:
            return openrouter_client.chat.completions.create(
                model=PRIMARY_MODEL,
                messages=messages,
                stream=True,
                temperature=0.7,
            ), "openrouter"
        else:
            return openrouter_client.chat.completions.create(
                model=PRIMARY_MODEL,
                messages=messages,
                temperature=0.7,
            ), "openrouter"
    except _FALLBACK_ERRORS as e:
        print(f"[AI] OpenRouter failed ({e}), falling back to Gemini …")
        if stream:
            return gemini_client.chat.completions.create(
                model=FALLBACK_MODEL,
                messages=messages,
                stream=True,
                temperature=0.7,
            ), "gemini"
        else:
            return gemini_client.chat.completions.create(
                model=FALLBACK_MODEL,
                messages=messages,
                temperature=0.7,
            ), "gemini"


# ════════════════════════════════════════════════════════════════════════════
# Supabase helpers
# ════════════════════════════════════════════════════════════════════════════

def get_book(book_id: str) -> dict:
    res = supabase.table("books").select("*").eq("id", book_id).single().execute()
    return res.data or {}


def get_page_data(book_id: str, page_number: int) -> dict:
    res = (
        supabase.table("pages")
        .select("text_content, image_url")
        .eq("book_id", book_id)
        .eq("page_number", page_number)
        .single()
        .execute()
    )
    return res.data or {}


def get_chat_history(book_id: str, user_id: str, limit: int = 20) -> list[dict]:
    """Fetch the last `limit` chat turns from Supabase and format as OpenAI messages."""
    res = (
        supabase.table("chats")
        .select("role, content")
        .eq("book_id", book_id)
        .eq("user_id", user_id)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return res.data or []


def save_chat_turn(book_id: str, user_id: str, user_msg: str, assistant_msg: str):
    """Persist a user+assistant turn to Supabase."""
    supabase.table("chats").insert([
        {"book_id": book_id, "user_id": user_id, "role": "user",      "content": user_msg},
        {"book_id": book_id, "user_id": user_id, "role": "assistant", "content": assistant_msg},
    ]).execute()


async def get_page_image_base64(user_id: str, book_id: str, page_number: int) -> Optional[str]:
    """Download page image from Supabase Storage and return as base64."""
    try:
        path = f"{user_id}/{book_id}/page_{page_number}.png"
        res  = supabase.storage.from_("book-pages").create_signed_url(path, 60)
        url  = res.get("signedURL") or res.get("signedUrl")
        if not url:
            return None
        async with httpx.AsyncClient() as http:
            r = await http.get(url, timeout=10)
            if r.status_code == 200:
                return base64.b64encode(r.content).decode("utf-8")
    except Exception:
        return None
    return None


def build_spread_text(book_id: str, left_page: int, right_page: int) -> tuple[str, list]:
    parts, image_pages = [], []
    for pnum in [left_page, right_page]:
        if pnum <= 0:
            continue
        data = get_page_data(book_id, pnum)
        text = (data.get("text_content") or "").strip()
        if text:
            parts.append(f"[Page {pnum}]\n{text}")
        else:
            image_pages.append(pnum)
    return "\n\n".join(parts), image_pages


def get_visible_pages(current_page: int, total_pages: int, is_rtl: bool) -> tuple[int, int]:
    if is_rtl:
        right = current_page
        left  = current_page + 1 if current_page + 1 <= total_pages else 0
        return left, right
    left  = current_page
    right = current_page + 1 if current_page + 1 <= total_pages else 0
    return left, right


def clean_json_response(raw: str) -> str:
    raw = raw.strip()
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                raw = part
                break
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end != -1:
        return raw[start : end + 1]
    return raw


# ════════════════════════════════════════════════════════════════════════════
# 1. CHAT  (streaming + memory)
# ════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    question:     str
    book_id:      str
    current_page: int
    language:     str  = "english"
    user_id:      str
    total_pages:  int  = 0
    is_rtl:       bool = False


@router.post("/chat")
async def chat(req: ChatRequest):
    book       = get_book(req.book_id)
    book_title = book.get("title", "this book")
    lang_instr = LANG_MAP.get(req.language, LANG_MAP["english"])

    left_page, right_page = get_visible_pages(req.current_page, req.total_pages, req.is_rtl)
    spread_text, image_pages = build_spread_text(req.book_id, left_page, right_page)
    pages_label = (
        f"pages {left_page} and {right_page}" if right_page else f"page {left_page}"
    )

    # Handle image-based pages — embed as base64 data-URIs in user content
    image_contents: list[dict] = []
    if image_pages and req.user_id:
        for pnum in image_pages:
            b64 = await get_page_image_base64(req.user_id, req.book_id, pnum)
            if b64:
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
                spread_text += f"\n\n[Page {pnum}: image-based]"

    # System prompt
    system_prompt = (
        f'You are an expert AI professor helping a student understand "{book_title}".\n'
        f"The student is currently viewing {pages_label}.\n\n"
        f"Page content:\n---\n"
        f"{spread_text or 'Image-based pages — describe what you see in the images.'}\n"
        f"---\n\n"
        f"Rules:\n"
        f"- Answer about EITHER visible page\n"
        f"- Be clear and concise like a patient professor\n"
        f"- Do NOT ask the student to flip pages\n"
        f"{lang_instr}"
    )

    # Build message list: system + history + current user question
    history = get_chat_history(req.book_id, req.user_id)

    user_content: list | str
    if image_contents:
        user_content = [{"type": "text", "text": req.question}] + image_contents
    else:
        user_content = req.question

    messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": user_content}]
    )

    # Stream response and collect full text to save in history
    try:
        stream, provider = _chat_complete(messages, stream=True)
        full_reply = []

        def generate():
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_reply.append(delta)
                    yield delta
            # After streaming ends, persist the turn (best-effort)
            try:
                save_chat_turn(req.book_id, req.user_id, req.question, "".join(full_reply))
            except Exception as save_err:
                print(f"[AI] Could not save chat turn: {save_err}")

        return StreamingResponse(generate(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {e}")


# ════════════════════════════════════════════════════════════════════════════
# 2. QUIZ
# ════════════════════════════════════════════════════════════════════════════

class QuizRequest(BaseModel):
    book_id:       str
    current_page:  int
    language:      str  = "english"
    num_questions: int  = 5
    total_pages:   int  = 0
    user_id:       str  = ""
    is_rtl:        bool = False


@router.post("/quiz")
async def generate_quiz(req: QuizRequest):
    book       = get_book(req.book_id)
    book_title = book.get("title", "this book")
    lang_instr = LANG_MAP.get(req.language, LANG_MAP["english"])

    left_page, right_page = get_visible_pages(req.current_page, req.total_pages, req.is_rtl)
    spread_text, image_pages = build_spread_text(req.book_id, left_page, right_page)

    image_contents: list[dict] = []
    has_content = bool(spread_text.strip())

    if image_pages and req.user_id:
        for pnum in image_pages:
            b64 = await get_page_image_base64(req.user_id, req.book_id, pnum)
            if b64:
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
                has_content = True

    if not has_content:
        raise HTTPException(status_code=400, detail="No content found on these pages.")

    system_prompt = (
        f'You are a quiz generator for "{book_title}". {lang_instr}\n\n'
        f"Generate exactly {req.num_questions} MCQs.\n"
        f"CRITICAL: Return ONLY a raw JSON array. Start with [ end with ].\n"
        f'[\n  {{"question": "...", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], '
        f'"answer": "A", "explanation": "..."}}\n]'
    )

    user_parts: list[dict] = []
    if spread_text.strip():
        user_parts.append({"type": "text", "text": f"Generate exactly {req.num_questions} MCQs from this content:\n\n{spread_text}"})
    else:
        user_parts.append({"type": "text", "text": f"Generate exactly {req.num_questions} MCQs from the images provided."})
    user_parts.extend(image_contents)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_parts if image_contents else user_parts[0]["text"]},
    ]

    try:
        response, _ = _chat_complete(messages, stream=False)
        raw       = clean_json_response(response.choices[0].message.content)
        questions = json.loads(raw)
        return {
            "questions":  questions,
            "page":       req.current_page,
            "left_page":  left_page,
            "right_page": right_page,
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse quiz. Try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz error: {e}")


# ════════════════════════════════════════════════════════════════════════════
# 3. FLASHCARDS
# ════════════════════════════════════════════════════════════════════════════

class FlashcardRequest(BaseModel):
    book_id:      str
    current_page: int
    language:     str  = "english"
    num_cards:    int  = 6
    total_pages:  int  = 0
    user_id:      str  = ""
    is_rtl:       bool = False


@router.post("/flashcards")
async def generate_flashcards(req: FlashcardRequest):
    book       = get_book(req.book_id)
    book_title = book.get("title", "this book")
    lang_instr = LANG_MAP.get(req.language, LANG_MAP["english"])

    left_page, right_page = get_visible_pages(req.current_page, req.total_pages, req.is_rtl)
    spread_text, image_pages = build_spread_text(req.book_id, left_page, right_page)

    image_contents: list[dict] = []
    has_content = bool(spread_text.strip())

    if image_pages and req.user_id:
        for pnum in image_pages:
            b64 = await get_page_image_base64(req.user_id, req.book_id, pnum)
            if b64:
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
                has_content = True

    if not has_content:
        raise HTTPException(status_code=400, detail="No content found on these pages.")

    system_prompt = (
        f'You are a flashcard generator for "{book_title}". {lang_instr}\n\n'
        f"Extract {req.num_cards} key terms/concepts.\n"
        f"CRITICAL: Return ONLY a raw JSON array. Start with [ end with ].\n"
        f'[\n  {{"term": "...", "definition": "... (1-2 sentences)"}}\n]'
    )

    user_parts: list[dict] = []
    if spread_text.strip():
        user_parts.append({"type": "text", "text": f"Generate exactly {req.num_cards} flashcards from:\n\n{spread_text}"})
    else:
        user_parts.append({"type": "text", "text": f"Generate exactly {req.num_cards} flashcards from the images provided."})
    user_parts.extend(image_contents)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_parts if image_contents else user_parts[0]["text"]},
    ]

    try:
        response, _ = _chat_complete(messages, stream=False)
        raw   = clean_json_response(response.choices[0].message.content)
        cards = json.loads(raw)
        return {
            "cards":      cards,
            "page":       req.current_page,
            "left_page":  left_page,
            "right_page": right_page,
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse flashcards. Try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Flashcard error: {e}")


# ════════════════════════════════════════════════════════════════════════════
# 4. CHAT HISTORY
# ════════════════════════════════════════════════════════════════════════════

@router.get("/chat/history/{book_id}")
async def get_history(book_id: str, user_id: str):
    res = (
        supabase.table("chats")
        .select("*")
        .eq("book_id", book_id)
        .eq("user_id", user_id)
        .order("created_at", desc=False)
        .execute()
    )
    return res.data or []