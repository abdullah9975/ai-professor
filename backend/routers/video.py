import asyncio, base64, io, wave, time as tm, json, os
import httpx
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from supabase import create_client

load_dotenv()
router   = APIRouter()
client   = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

video_jobs: dict = {}

# ── Models ────────────────────────────────────────────
GEMINI_MODEL  = "gemini-3-flash-preview"            # 1500 req/day free tier — keep this model
TTS_MODEL     = "gemini-2.5-flash-preview-tts"
# INFIP_API_KEY = os.getenv("INFIP_API_KEY", "")
# INFIP_BASE    = "https://api.infip.pro"
# INFIP_MODEL   = "img4"
# INFIP_SIZE    = "1792x1024"
# ADD:
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE    =  "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-dev"
NVIDIA_MODEL   = "black-forest-labs/flux-dev"   # swap to flux-schnell for faster gen
NVIDIA_SIZE    = "1344x768"                      # 16:9 landscape

# ── Label overlay via Pillow (rendered server-side, never by AI) ──────────
# We add a clean text label to each image so the user sees the key term.
# This is always 100% accurate because we do it in Python, not via prompt.
try:
    from PIL import Image as _PILImage, ImageDraw as _PILDraw, ImageFont as _PILFont
    import io as _pil_io
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[label] Pillow not installed — labels disabled. pip install Pillow")

def extract_label(prompt: str, narration_sentence: str) -> str:
    """
    Extract the most important 1-4 word key term from a prompt/sentence
    to use as the image label. Uses simple heuristics — no AI call needed.
    """
    import re
    # Patterns: "X fibre", "X cable", "X atom", proper nouns in title case etc.
    # Try the image prompt first — it usually has the exact subject
    combined = narration_sentence + " " + prompt

    # Common educational key-term patterns
    patterns = [
        r'([A-Z][a-z]+ (?:fibre|fiber|cable|wave|ray|cell|atom|molecule|bond|circuit|network|protocol|layer|model|theory|law|force|energy|field|current|voltage|resistance|lens|mirror|beam|particle|electron|proton|neutron|nucleus|orbit|planet|star|galaxy|chromosome|DNA|RNA|virus|bacteria|organ|tissue|muscle|bone|artery|vein|nerve)s?)',
        r'((?:optical|optic) (?:fibre|fiber|cable)s?)',
        r'((?:TCP|IP|HTTP|DNS|FTP|SMTP|HTML|CSS|SQL|RAM|ROM|CPU|GPU|LED|LCD|USB|LAN|WAN|VPN|OSI|WiFi|Bluetooth|4G|5G))',
        r'([A-Z][A-Za-z]+ (?:Model|Protocol|Algorithm|Function|Theorem|Law|Effect|Force|Wave|Ray|Cell|Layer))',
    ]
    for pat in patterns:
        m = re.search(pat, combined, re.IGNORECASE)
        if m:
            label = m.group(1).strip()
            if 2 <= len(label) <= 40:
                return label

    # Fallback: first noun phrase from the narration sentence (up to 4 words)
    words = narration_sentence.split()
    # Skip common stop words at the start
    stops = {"the","a","an","in","on","at","to","for","of","and","or","but",
             "is","are","was","were","has","have","this","that","these","those"}
    key_words = [w for w in words if w.lower().rstrip(".,;:") not in stops]
    if key_words:
        label = " ".join(key_words[:3]).rstrip(".,;:")
        if 2 <= len(label) <= 35:
            return label

    return ""

def add_label_to_image(img_b64: str, label: str) -> str:
    """
    Add a clean bottom-left label badge to a base64 image.
    Returns the modified image as base64. If anything fails, returns original.
    """
    if not HAS_PIL or not label:
        return img_b64
    try:
        raw = base64.b64decode(img_b64)
        img = _PILImage.open(_pil_io.BytesIO(raw)).convert("RGB")
        w, h = img.size
        draw = _PILDraw.Draw(img)

        # Font size proportional to image width
        font_size = max(24, w // 50)
        font = None
        # Try system fonts, fall back to default
        for font_path in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]:
            try:
                font = _PILFont.truetype(font_path, font_size)
                break
            except: pass
        if font is None:
            font = _PILFont.load_default()

        # Measure text
        bbox   = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # Badge position: bottom-left with padding
        pad_x, pad_y = 20, 16
        bx = 24
        by = h - th - pad_y * 2 - 10

        # Semi-transparent dark badge background
        overlay = _PILImage.new("RGBA", img.size, (0, 0, 0, 0))
        ov_draw = _PILDraw.Draw(overlay)
        ov_draw.rounded_rectangle(
            [bx - pad_x, by - pad_y, bx + tw + pad_x, by + th + pad_y],
            radius=8, fill=(0, 0, 0, 165)
        )
        img = _PILImage.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        draw = _PILDraw.Draw(img)

        # White text with subtle shadow
        draw.text((bx + 1, by + 1), label, font=font, fill=(0, 0, 0, 180))
        draw.text((bx, by), label, font=font, fill=(255, 255, 255))

        # Save back to base64
        buf = _pil_io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        print(f"[label] overlay failed: {e}")
        return img_b64

LANG_MAP = {
    "hindi":   "Respond ONLY in Hindi (हिंदी). Never use English.",
    "urdu":    "Respond ONLY in Urdu (اردو). Never use English.",
    "arabic":  "Respond ONLY in Arabic (العربية). Never use English.",
    "bengali": "Respond ONLY in Bengali (বাংলা). Never use English.",
    "tamil":   "Respond ONLY in Tamil (தமிழ்). Never use English.",
    "english": "Respond ONLY in English.",
}
VOICE_MAP = {
    "hindi":   "Sadaltager",
    "urdu":    "Sulafat",
    "arabic":  "Rasalgethi",
    "bengali": "Charon",
    "tamil":   "Iapetus",
    "english": "Sadaltager",
}

# ════════════════════════════════════════════════════
# ISLAMIC SACRED FIGURE PROTECTION
# Never depict Allah, Prophet Muhammad ﷺ, any Sahaba,
# or any Islamic prophet — replace with calligraphy art.
# ════════════════════════════════════════════════════
SACRED_NEGATIVE = (
    "no depiction of Allah, no face or figure of Prophet Muhammad ﷺ, "
    "no face or figure of any Islamic prophet or Sahaba, "
    "no religious figure portrait of any sacred person in Islam"
)

SACRED_KEYWORDS = [
    # English
    "allah", "muhammad", "prophet muhammad", "prophet", "rasool", "rasul", "nabi",
    "abu bakr", "umar ibn", "uthman", "ali ibn", "sahaba", "sahabah", "companion",
    "aisha", "fatima", "khadija", "khadijah", "bilal", "khalid ibn", "salman",
    "abu hurairah", "ibn masud", "anas ibn", "companion of prophet", "holy prophet",
    "messenger of allah", "messenger of god", "prophet of islam",
    "hazrat", "sayyiduna", "imam ali", "imam hussain", "imam hasan",
    "ibrahim", "musa", "isa", "prophet ibrahim", "prophet musa", "prophet isa",
    # Urdu / Arabic transliterations
    "صحابی", "صحابہ", "صحابه", "نبی", "رسول", "محمد", "اللہ",
    "ابوبکر", "عمر", "عثمان", "علی", "عائشہ", "فاطمہ", "خدیجہ",
    "بلال", "خالد", "حضرت", "سیدنا", "امام",
    "النبي", "الرسول", "أبو بكر", "عمر بن", "عثمان", "علي بن",
    "عائشة", "فاطمة", "خديجة",
]

def contains_sacred_figure(text: str) -> bool:
    low = text.lower()
    return any(kw.lower() in low for kw in SACRED_KEYWORDS)




# ════════════════════════════════════════════════════
# SACRED CONTENT DETECTION
#
# When a sentence references Allah, Prophet Muhammad ﷺ,
# Sahaba, or any Islamic prophet:
#   → Generate the LOCATION or SETTING described in the
#     sentence (street, mosque, desert, battlefield, etc.)
#   → STRICTLY exclude ALL human figures, faces, silhouettes
#     or any physical form of any person whatsoever.
#
# NO calligraphy. NO Arabic text rendering.
# NO generic "Islamic art" placeholder.
# Just the place/atmosphere, empty of people.
# ════════════════════════════════════════════════════
import re as _re

_ARABIC_WORD_RE = _re.compile(
    r'[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]'
    r'[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿ً-ٟ ]*'
)

SACRED_FIGURE_KEYWORDS = [
    # Allah
    "allah", "اللہ", "الله", "azzawajal", "subhanahu", "عزوجل", "سبحانه",
    # Prophet Muhammad ﷺ
    "prophet muhammad", "muhammad", "محمد", "مُحَمَّد",
    "rasulullah", "rasool", "rasul", "sallallahu", "ﷺ",
    "holy prophet", "messenger of allah", "messenger of god",
    "نبی", "رسول", "النبي", "الرسول",
    # Sahaba / companions
    "sahaba", "sahabah", "companion of prophet", "companion of the prophet",
    "abu bakr", "umar ibn", "uthman", "ali ibn",
    "aisha", "fatima", "khadija", "khadijah",
    "bilal", "khalid ibn", "salman", "abu hurairah",
    "ibn masud", "anas ibn", "sayyiduna", "hazrat",
    "imam ali", "imam hussain", "imam hasan",
    "صحابی", "صحابہ", "صحابه", "ابوبکر", "عمر", "عثمان", "علی",
    "عائشہ", "فاطمہ", "خدیجہ", "بلال", "خالد", "حضرت", "سیدنا", "امام",
    "أبو بكر", "عمر بن", "علي بن", "عائشة", "فاطمة", "خديجة",
    # Quranic prophets
    "prophet ibrahim", "prophet musa", "prophet isa", "prophet nuh",
    "prophet yusuf", "prophet dawud", "prophet sulaiman",
    "ibrahim", "musa", "isa", "nuh", "yusuf",
    "إبراهيم", "موسى", "عيسى", "نوح", "يوسف",
]

def contains_sacred_figure(text: str) -> bool:
    """True if the text references any sacred Islamic figure."""
    low = text.lower()
    return any(kw.lower() in low for kw in SACRED_FIGURE_KEYWORDS)

def extract_all_arabic(text: str) -> str:
    """Extract Arabic text from a string (used only for detection, not for rendering)."""
    runs = _ARABIC_WORD_RE.findall(text)
    cleaned = [r.strip() for r in runs if r.strip()]
    if not cleaned:
        return ""
    joined = " ".join(cleaned).strip()
    count = sum(1 for c in joined if "؀" <= c <= "ۿ"
                or "ﭐ" <= c <= "﷿" or "ﹰ" <= c <= "﻿")
    return joined if count >= 3 else ""

def contains_arabic(text: str) -> bool:
    return bool(_ARABIC_WORD_RE.search(text))

def strip_person_from_prompt(raw_prompt: str, context: str) -> str:
    """
    Take the Gemini-generated image prompt and rewrite it to:
    1. Remove any reference to a person/figure/face
    2. Keep ONLY the setting, location, atmosphere, objects
    3. Add strict no-figures instruction

    This way we always get the contextually correct scene
    (street of Madinah, battlefield at Badr, cave of Hira, etc.)
    without generating any sacred figure.
    """
    # Common patterns that describe a person doing something —
    # strip the person and keep the location/action context
    # e.g. "Prophet Muhammad walking in the streets of Madinah"
    #   →  "The ancient streets of Madinah"

    person_patterns = [
        # "Prophet X [verb]ing ..." → strip up to first location/preposition
        _re.compile(r"(?:prophet\s+\w+|muhammad|rasulullah|abu\s+bakr|umar|uthman|ali|aisha|fatima|khadija|bilal|imam\s+\w+|hazrat\s+\w+)\s+(?:is\s+|was\s+|were\s+)?(?:\w+ing\s+)?", _re.IGNORECASE),
        # "He/She/They [verb]ing ..." at start — only strip personal pronoun + verb
        _re.compile(r"^(?:he|she|they|him|his|her)\s+(?:is\s+|was\s+|were\s+)?(?:\w+ing\s+)?", _re.IGNORECASE),
    ]
    cleaned = raw_prompt
    for pat in person_patterns:
        cleaned = pat.sub("", cleaned).strip()

    # Capitalise first letter after stripping
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]

    # If stripping left almost nothing, fall back to extracting location from context
    if len(cleaned) < 20:
        cleaned = raw_prompt  # use original — the negative prompt will handle figures

    # Build final prompt: keep the scene/action, only remove the sacred person.
    # Other people (armies, soldiers, crowds) are fine — only sacred figures removed.
    return (
        f"{cleaned.rstrip('.')}. "
        f"No face or figure of Prophet Muhammad ﷺ, no face or figure of any "
        f"Islamic prophet or Sahaba — but other people, soldiers, crowds, "
        f"and non-sacred figures are allowed in the scene. "
        f"Cinematic 16:9, 4K, photorealistic, historically accurate Islamic setting."
    )


# ════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════
def estimate_total_seconds(n_scenes: int, avg_images: int = 2) -> int:
    return 6 + n_scenes * (9 * avg_images + 13)

def retry_text(prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            r = client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt,
                config=types.GenerateContentConfig(temperature=0.7)
            )
            return r.text.strip()
        except Exception as e:
            msg = str(e)
            if ("503" in msg or "429" in msg or "UNAVAILABLE" in msg) and attempt < retries-1:
                wait = (attempt+1)*8; print(f"[retry] {wait}s..."); tm.sleep(wait); continue
            raise
    raise Exception("Model unavailable.")

def _get_image_bytes_from_storage(image_url_or_path: str, book_id: str, page: int) -> bytes | None:
    """
    Fetch the page image bytes from Supabase storage.
    Handles three cases:
      1. image_url is a full https:// URL — download directly
      2. image_url is a storage path like "user/book/page_1.png" — get signed URL
      3. Neither — try common path patterns across buckets
    """
    import httpx as _httpx

    image_bytes = None

    # Case 1: full public URL
    if image_url_or_path.startswith("http"):
        try:
            resp = _httpx.get(image_url_or_path, timeout=30)
            if resp.status_code == 200:
                image_bytes = resp.content
                print(f"[page_text] ✓ downloaded from public URL ({len(image_bytes)} bytes)")
                return image_bytes
        except Exception as e:
            print(f"[page_text] public URL download failed: {e}")

    # Case 2: image_url is a storage path (most common — upload.py stores path like
    # "user_id/book_id/page_1.png" in image_url column)
    if image_url_or_path and not image_url_or_path.startswith("http"):
        storage_path = image_url_or_path
        try:
            signed = supabase.storage.from_("book-pages").create_signed_url(storage_path, 300)
            url = signed.get("signedURL") or signed.get("signedUrl") or ""
            if url:
                resp = _httpx.get(url, timeout=30)
                if resp.status_code == 200:
                    image_bytes = resp.content
                    print(f"[page_text] ✓ signed URL from book-pages/{storage_path} ({len(image_bytes)} bytes)")
                    return image_bytes
        except Exception as e:
            print(f"[page_text] signed URL failed for {storage_path}: {e}")

    # Case 3: no image_url at all — try known path patterns across all buckets
    # upload.py stores pages as: {user_id}/{book_id}/page_{N}.png
    # We don't have user_id here so also try {book_id}/page_{N}.png
    candidate_paths = [
        f"{book_id}/page_{page}.png",
        f"{book_id}/page_{page}.jpg",
        f"{book_id}/{page}.png",
        f"{book_id}/{page}.jpg",
    ]
    for bucket in ["book-pages", "books", "pages"]:
        for path in candidate_paths:
            try:
                signed = supabase.storage.from_(bucket).create_signed_url(path, 120)
                url = signed.get("signedURL") or signed.get("signedUrl") or ""
                if url:
                    resp = _httpx.get(url, timeout=30)
                    if resp.status_code == 200:
                        image_bytes = resp.content
                        print(f"[page_text] ✓ found at {bucket}/{path} ({len(image_bytes)} bytes)")
                        return image_bytes
            except:
                pass

    return None


def _gemini_vision_describe(image_bytes: bytes, page: int) -> str:
    """Send page image to Gemini Vision and return a full content description."""
    mime = "image/jpeg"
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        mime = "image/png"
    elif image_bytes[:4] == b'%PDF':
        mime = "application/pdf"

    print(f"[page_text] Sending page {page} image to Gemini Vision ({mime}, {len(image_bytes)} bytes)...")
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            types.Part.from_text(
                "You are reading a page from an educational book. "
                "Transcribe and describe EVERYTHING on this page in full detail: "
                "all text (copy it exactly), headings, diagrams, tables, figures, "
                "captions, labels, formulas, equations, and any other content. "
                "If there is Arabic or Urdu text, include it verbatim. "
                "Be thorough and complete — this description will be used to "
                "generate an educational video about this exact page."
            ),
        ],
        config=types.GenerateContentConfig(temperature=0.1),
    )
    description = resp.text.strip()
    print(f"[page_text] ✓ Vision described {len(description)} chars")
    return description


def get_page_text(book_id: str, page: int) -> str:
    """
    Get page content for video generation.

    For text PDFs: returns text_content directly from the DB.
    For image PDFs: fetches the page image from Supabase storage and
    uses Gemini Vision to extract a full description — same pipeline
    as flashcards, quiz, and chat for image-based books.

    NOTE: storage_path column does NOT exist in this project's pages table.
    We use only text_content and image_url columns.
    """
    try:
        # ── Step 1: fetch from DB (no storage_path — that column doesn't exist) ──
        r = supabase.table("pages") \
            .select("text_content,image_url") \
            .eq("book_id", book_id) \
            .eq("page_number", page) \
            .single() \
            .execute()

        if not r.data:
            print(f"[page_text] no DB row for book={book_id} page={page}")
            return ""

        # ── Step 2: text PDF — return text directly ───────────────────
        text = (r.data.get("text_content") or "").strip()
        if text:
            print(f"[page_text] ✓ text content ({len(text)} chars)")
            return text

        # ── Step 3: image PDF — fetch image and use Gemini Vision ─────
        # image_url stores the Supabase storage path (e.g. "uid/bid/page_1.png")
        # It may or may not be a full URL depending on how it was uploaded.
        image_url_val = (r.data.get("image_url") or "").strip()
        print(f"[page_text] No text found, trying image fallback. image_url='{image_url_val[:60]}'")

        image_bytes = _get_image_bytes_from_storage(image_url_val, book_id, page)

        if not image_bytes:
            print(f"[page_text] ✗ could not retrieve image for book={book_id} page={page}")
            return ""

        return _gemini_vision_describe(image_bytes, page)

    except Exception as e:
        print(f"[page_text] error: {e}")
        return ""

def get_book_title(book_id: str) -> str:
    try:
        r = supabase.table("books").select("title") \
            .eq("id", book_id).single().execute()
        return r.data.get("title", "this book") if r.data else "this book"
    except: return "this book"

def clean_json(raw: str, b: str = "[") -> str:
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip()
            if part.startswith("json"): part = part[4:].strip()
            if part.startswith(b): return part
    eb = "]" if b == "[" else "}"
    s = raw.find(b); e = raw.rfind(eb)
    return raw[s:e+1] if s != -1 and e != -1 else raw

def pcm_to_wav_b64(pcm_bytes: bytes) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
        wf.writeframes(pcm_bytes)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def wav_duration_seconds(b64wav: str) -> float:
    try:
        raw = base64.b64decode(b64wav)
        buf = io.BytesIO(raw)
        with wave.open(buf, "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception as e:
        print(f"[wav_duration] fallback: {e}")
        return max(10.0, (len(base64.b64decode(b64wav)) - 44) / 48000.0)


# ════════════════════════════════════════════════════
# SUPABASE
# ════════════════════════════════════════════════════
def save_video_to_db(job_id, user_id, book_id, book_title, page, language, frames):
    try:
        supabase.table("video_history").upsert({
            "id": job_id, "user_id": user_id, "book_id": book_id,
            "book_title": book_title, "page": page,
            "language": language, "frames_json": frames,
        }).execute()
        print(f"[db] ✓ saved {job_id}")
    except Exception as e:
        print(f"[db] save failed: {e}")

def load_video_history(user_id: str, book_id: str) -> list:
    """
    Load metadata + first thumbnail only. Fast — avoids fetching full frames_json.
    """
    try:
        import httpx as _httpx
        url = f"{os.getenv('SUPABASE_URL')}/rest/v1/video_history"
        headers = {
            "apikey":        os.getenv("SUPABASE_SERVICE_KEY"),
            "Authorization": f"Bearer {os.getenv('SUPABASE_SERVICE_KEY')}",
            "Accept":        "application/json",
        }
        params = {
            "select":   "id,book_title,page,language,created_at,frames_json->0->images->0",
            "user_id":  f"eq.{user_id}",
            "book_id":  f"eq.{book_id}",
            "order":    "created_at.desc",
            "limit":    "10",
        }
        resp = _httpx.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"[db] history HTTP {resp.status_code}")
            return []
        history = []
        for row in resp.json():
            history.append({
                "id":         row["id"],
                "book_title": row["book_title"],
                "page":       row["page"],
                "language":   row["language"],
                "date":       row["created_at"][:16].replace("T", " ") if row.get("created_at") else "",
                "thumbnail":  row.get("frames_json->0->images->0") or "",
            })
        return history
    except Exception as e:
        print(f"[db] load history failed: {e}")
        return []
    

def load_video_frames(video_id: str, user_id: str) -> list:
    """
    Load frames_json for a single video.
    Uses postgrest's prefer header to avoid statement timeout.
    """
    try:
        # Fetch with explicit timeout via httpx directly — supabase-py
        # doesn't expose per-query timeouts, so we call the REST API ourselves.
        import httpx as _httpx
        url = f"{os.getenv('SUPABASE_URL')}/rest/v1/video_history"
        headers = {
            "apikey":        os.getenv("SUPABASE_SERVICE_KEY"),
            "Authorization": f"Bearer {os.getenv('SUPABASE_SERVICE_KEY')}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }
        params = {
            "select": "frames_json",
            "id":     f"eq.{video_id}",
            "user_id": f"eq.{user_id}",
        }
        resp = _httpx.get(url, headers=headers, params=params, timeout=120)
        if resp.status_code != 200:
            print(f"[db] load_frames HTTP {resp.status_code}: {resp.text[:200]}")
            return []
        data = resp.json()
        if not data:
            return []
        return data[0].get("frames_json") or []
    except Exception as e:
        print(f"[db] load_frames failed: {e}")
        return []
    
    
def delete_video_from_db(video_id: str, user_id: str) -> bool:
    try:
        supabase.table("video_history") \
            .delete().eq("id", video_id).eq("user_id", user_id).execute()
        print(f"[db] ✓ deleted {video_id}")
        return True
    except Exception as e:
        print(f"[db] delete failed: {e}")
        return False


class VideoRequest(BaseModel):
    book_id:      str
    current_page: int
    language:     str = "english"
    user_id:      str
    total_pages:  int = 0


# ── Detect if the book content is Islamic ────────────
ISLAMIC_BOOK_KEYWORDS = [
    # Arabic/Urdu terms
    "allah", "quran", "quraan", "hadith", "sunnah", "prophet", "rasool", "nabi",
    "islamic", "islam", "muslim", "hijra", "ghazwa", "ghazwah", "sariyya",
    "madinah", "makkah", "mecca", "medina", "sahaba", "sahabah", "khilafat",
    "caliphate", "zakat", "salah", "namaz", "jihad", "fiqh", "aqeedah",
    "tafseer", "tafsir", "seerah", "sirat", "sirah", "hijab", "halal", "haram",
    # Urdu
    "اللہ", "قرآن", "حدیث", "نبی", "رسول", "اسلام", "مسلم", "صحابہ",
    "غزوہ", "سریہ", "ہجرت", "مدینہ", "مکہ", "خلافت", "جہاد",
    # Arabic
    "الله", "القرآن", "الحديث", "الإسلام", "المسلم", "الصحابة",
    "الغزوة", "الهجرة", "المدينة", "مكة", "الخلافة",
]

def is_islamic_content(text: str) -> bool:
    """Return True if the page text is from an Islamic book."""
    low = text.lower()
    hits = sum(1 for kw in ISLAMIC_BOOK_KEYWORDS if kw.lower() in low)
    return hits >= 2  # at least 2 matches = likely Islamic content


# ════════════════════════════════════════════════════
# GENERATE
# ════════════════════════════════════════════════════
@router.post("/video/generate")
async def generate_video(req: VideoRequest, bg: BackgroundTasks):
    if not NVIDIA_API_KEY:
        raise HTTPException(500, "NVIDIA_API_KEY not set in backend/.env")

    book_title = get_book_title(req.book_id)
    page_text  = get_page_text(req.book_id, req.current_page)
    lang_instr = LANG_MAP.get(req.language.lower(), LANG_MAP["english"])

    if not page_text.strip():
        raise HTTPException(400, "No content found on this page. If this is an image-based PDF, ensure the page images are stored in Supabase.")

    # Detect if this is an Islamic book — affects image style rules
    is_islamic = is_islamic_content(page_text)

    # Build the content-type-aware image rules
    if is_islamic:
        content_image_rules = """
CONTENT TYPE: This is an ISLAMIC book. ALL images must be:
  ✓ Consistent with Islamic art — mosques, calligraphy, deserts, maps, artifacts, historical settings
  ✓ Contextually relevant to the exact sentence being illustrated
  ✗ NEVER show scenes, objects, or imagery from other religions or unrelated cultures
  ✗ NEVER mix in non-Islamic imagery (churches, temples, secular entertainment, etc.)

════════════ SACRED FIGURE RULE — ABSOLUTE ════════════
NEVER show the face or physical figure of:
  Allah, Prophet Muhammad ﷺ, any Sahaba, any Quranic prophet.

For sentences about THESE SPECIFIC sacred figures only:
  → Remove the sacred person from the prompt
  → Keep the setting, location, atmosphere, objects
  → Other non-sacred people in the same scene ARE allowed

OTHER PEOPLE (soldiers, armies, crowds, civilians) ARE ALLOWED in images.
The restriction is ONLY on sacred figures listed above.

EXAMPLES:

WRONG: "Prophet Muhammad ﷺ leading an army into battle"
RIGHT: "Two large armies clashing on a desert battlefield, swords and banners,
        dust and sunlight, dramatic action, no identifiable sacred figure,
        cinematic 16:9, 4K"

WRONG: "Prophet Ibrahim building the Kaaba with his own hands"
RIGHT: "Workers constructing the ancient Kaaba in Makkah, stone blocks,
        desert landscape, golden light, historically accurate,
        cinematic 16:9, 4K"

✓ ALLOWED: "Two cavalry armies charging on horses at the Battle of Badr,
            dramatic dust clouds, swords raised, banners flying, no faces
            visible, cinematic 16:9, 4K"

✓ ALLOWED: "Muslim soldiers praying together in rows before battle, desert
            setting, morning light, dramatic composition, no identifiable
            sacred figures, cinematic 16:9, 4K"

The rule is: SHOW THE SCENE AND THE ACTION, just never show the face or
figure of the specific sacred persons listed above.
══════════════════════════════════════════════════════════════════"""
    else:
        content_image_rules = """
CONTENT TYPE: This is a NON-ISLAMIC book.

CRITICAL: Every image MUST visually match the subject being described.
Generate images that are 100% relevant to the exact sentence content.

Science book examples:
  "An atom has a nucleus with electrons orbiting it" →
    "Detailed 3D render of an atom showing nucleus with protons and neutrons,
     electron cloud orbiting, glowing blue and gold, dark background, cinematic 16:9"
  
  "Matter exists in three states: solid, liquid, and gas" →
    "Three side-by-side scientific diagrams of solid crystalline structure, 
     flowing liquid molecules, and gas particles dispersing, educational style, cinematic 16:9"

  "A student holds a wooden block to measure its mass" →
    "Young student at a lab bench holding a rectangular wooden block on a scale,
     school science lab setting, bright lighting, cinematic 16:9"

  "Force equals mass multiplied by acceleration (F=ma)" →
    "Physics formula F=ma written on a chalkboard with diagram of a block being
     pushed, arrows showing force and motion, classroom setting, cinematic 16:9"

Geography/History book examples:
  "The Amazon river flows through Brazil" →
    "Aerial view of the Amazon river winding through dense green rainforest,
     misty morning, lush tropical vegetation, cinematic 16:9"

Rules:
  ✓ Science: show atoms, molecules, experiments, lab equipment, diagrams, formulas
  ✓ History: show places, artifacts, maps, buildings, armies, events
  ✓ Geography: show landscapes, maps, cities, natural features
  ✓ People in the scene are ALLOWED and encouraged when the sentence describes them
  ✗ NEVER use Islamic imagery, mosques, or Arabic text for non-Islamic content
  ✗ NEVER use generic or unrelated imagery — always match the exact sentence"""

    plan_prompt = f"""You are creating a documentary-style educational video for page {req.current_page} of "{book_title}".

Page content:
{page_text[:4000]}

LANGUAGE INSTRUCTION: {lang_instr}

════════════════════════════════════════════════════
SCENE STRUCTURE RULES — READ CAREFULLY
════════════════════════════════════════════════════
Create between 3 and 6 scenes. Each scene covers one key concept from the page.

CRITICAL RULE: Each narration sentence gets EXACTLY ONE image_prompt.
— Count the sentences in narration by their ending punctuation (. ! ? ۔ ؟)
— image_prompts array length MUST equal sentence count. No more, no less.
— image_prompt[0] illustrates sentence 1 ONLY
— image_prompt[1] illustrates sentence 2 ONLY
— image_prompt[N] illustrates sentence N+1 ONLY

WHAT MAKES A GOOD image_prompt:
  ✓ Describes ONLY what the matching sentence talks about
  ✓ Uses vivid, specific visual details — location, objects, lighting, atmosphere
  ✓ Written in English regardless of narration language
  ✓ Ends with: "cinematic 16:9, 4K, photorealistic, no text overlay"
  ✗ NEVER describe something from a different sentence
  ✗ NEVER use vague prompts like "Islamic scene" or "historical moment"

EXAMPLE — narration has exactly 4 sentences:
  S1: "The Battle of Badr took place in 624 CE near the wells of Badr."
  S2: "The Muslim army numbered only 313 fighters."
  S3: "The Quraysh force had over 1000 well-armed soldiers."
  S4: "Allah granted the Muslims a decisive victory against all odds."

Correct image_prompts (exactly 4):
  [
    "Ancient desert wells at Badr surrounded by sandy terrain under blazing sun, cinematic 16:9, 4K",
    "313 determined soldiers standing in disciplined formation in desert heat, silhouettes, no faces, cinematic 16:9",
    "Large armored army marching through desert with banners and shields, dust clouds, dusk light, cinematic 16:9",
    "Desert battlefield at dawn with divine golden light breaking through clouds, no human figures, cinematic 16:9"
  ]

If a sentence quotes a Quranic ayah, the image_prompt MUST embed the exact Arabic text.
See QURANIC AYAH RULE below for that case.

{content_image_rules}

════════════════════════════════════════════════════
OUTPUT FORMAT — Return ONLY valid JSON, no markdown:
════════════════════════════════════════════════════
[
  {{
    "scene_number": 1,
    "title": "Scene title in the specified language",
    "narration": "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5.",
    "image_prompts": [
      "Photorealistic English prompt for sentence 1. Cinematic 16:9, 4K. No text overlay.",
      "Prompt for sentence 2.",
      "Prompt for sentence 3.",
      "Prompt for sentence 4.",
      "Prompt for sentence 5."
    ],
    "kenburns": "zoom-in or zoom-out or pan-left or pan-right"
  }}
]

Final checklist before responding:
- narration has 4-6 sentences ✓
- image_prompts has the SAME count as sentences ✓
- every image_prompt is in English ✓
- all narration is in the specified language ✓
- images match the content type of this book ✓"""

    try:
        print(f"[scene planning] Using Gemini model: {GEMINI_MODEL}")
        raw_scenes = json.loads(clean_json(retry_text(plan_prompt)))
    except Exception as e:
        raise HTTPException(500, f"Scene planning failed: {str(e)}")

    n_scenes   = len(raw_scenes)
    total_imgs = sum(len(s.get("image_prompts", [""])) for s in raw_scenes)
    eta_secs   = 6 + total_imgs * 9 + n_scenes * 13
    job_id     = f"vid_{req.user_id[:8]}_{req.current_page}_{int(tm.time())}"

    video_jobs[job_id] = {
        "status": "generating", "progress": 5,
        "progress_msg": f"Planned {n_scenes} scenes, {total_imgs} images. Generating...",
        "eta_seconds": eta_secs, "started_at": tm.time(),
        "scenes": raw_scenes, "frames": [],
        "book_title": book_title, "page": req.current_page,
        "language": req.language, "user_id": req.user_id,
        "book_id": req.book_id, "error": None,
        "is_islamic": is_islamic,   # passed to build_frames for label logic
    }

    bg.add_task(build_frames, job_id, raw_scenes, req, is_islamic)
    return {
        "job_id": job_id, "status": "generating",
        "eta_seconds": eta_secs, "n_scenes": n_scenes, "total_images": total_imgs,
    }


# ════════════════════════════════════════════════════
# BACKGROUND — Images (NVIDIA) + TTS (Gemini)
# ════════════════════════════════════════════════════
async def build_frames(job_id: str, scenes: list, req: VideoRequest, is_islamic: bool = False):
    voice  = VOICE_MAP.get(req.language.lower(), "Sadaltager")
    frames = []
    n      = len(scenes)
    total_imgs = sum(len(s.get("image_prompts", [""])) for s in scenes)
    imgs_done  = 0

    def update(pct: int, msg: str):
        job = video_jobs.get(job_id)
        if job:
            job["progress"]     = max(job.get("progress", 0), min(pct, 99))
            job["progress_msg"] = msg

    for i, scene in enumerate(scenes):
        scene_pct = 5 + int((i / n) * 88)
        update(scene_pct, f"Scene {i+1}/{n}: generating images...")
        print(f"\n[scene {i+1}/{n}] '{scene.get('title','')}'")

        try:
            # Keep original narration for ayah extraction BEFORE translation.
            # Arabic ayah text is lost after translation, so original_narration
            # preserves the Arabic Unicode for exact calligraphy prompts.
            original_narration = scene.get("narration", "")
            narration = original_narration
            if req.language.lower() != "english":
                try:
                    narration = retry_text(
                        f"Translate ONLY this text to {req.language}. Return ONLY the translation:\n{narration}"
                    )
                except: pass

            # ── Generate images ──────────────────────────
            image_prompts = scene.get("image_prompts", [scene.get("image_prompt", "")])
            if not image_prompts:
                image_prompts = [f"Documentary scene: {scene.get('title','')}"]

            images_b64 = []
            # Split narration into sentences so we can match each to its image prompt
            # Split original narration for ayah extraction (Arabic text preserved before translation)
            original_sentences = [s.strip() for s in _re.split(r'(?<=[.!?۔؟])\s+', original_narration) if s.strip()]
            narration_sentences = [s.strip() for s in _re.split(r'(?<=[.!?۔؟])\s+', narration) if s.strip()]

            for j, raw_prompt in enumerate(image_prompts):
                imgs_done += 1
                img_pct = 5 + int((imgs_done / max(total_imgs, 1)) * 60)
                update(img_pct, f"Scene {i+1}/{n} image {j+1}/{len(image_prompts)}...")
                print(f"  [img {j+1}/{len(image_prompts)}] checking...")

                # Use ORIGINAL (pre-translation) sentence for Arabic extraction.
                # After translation, Arabic script is lost — original preserves it.
                original_sentence   = original_sentences[j] if j < len(original_sentences) else ""
                translated_sentence = narration_sentences[j] if j < len(narration_sentences) else ""

                # ── Combine all text sources for detection ────────────
                combined = raw_prompt + " " + original_sentence + " " + translated_sentence

                # ── Sacred figure detected? ───────────────────────────
                # Never render any person — strip the figure from the
                # prompt and render the location/setting only.
                if contains_sacred_figure(combined):
                    print(f"  [img {j+1}] ⚠ Sacred figure → scene-only (no figures)")
                    final_prompt = strip_person_from_prompt(raw_prompt, combined)

                else:
                    # ── Normal image — append sacred safety negative ──
                    final_prompt = f"{raw_prompt}. STRICT: {SACRED_NEGATIVE}."

                # Log exactly what is being sent to NVIDIA for image generation (first 140 chars)
                print(f"  [img {j+1}] PROMPT → {final_prompt[:140]}...")
                print(f"  [img {j+1}] generating...")
                # ADD this in its place:
                try:
                    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=10)) as http:
                        resp = await http.post(
                            NVIDIA_BASE,
                            headers={
                                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                                "Content-Type":  "application/json",
                                "Accept":        "application/json",
                            },
                            json={
                                "mode":           "base",
                                "prompt":          final_prompt,
                                "cfg_scale":      3.5,
                                "width":          1344,
                                "height":         768,
                                "seed":          0,
                                "steps":         50,
                            }
                        )
                    if resp.status_code == 200:
                        d     = resp.json()
                        items = d.get("artifacts", [])
                        if not items:
                            print(f"  [img {j+1}] ✗ empty artifacts array")
                        else:
                            item   = items[0]
                            finish = item.get("finishReason", "")
                            raw    = item.get("base64", "")

                            if finish != "SUCCESS":
                                print(f"  [img {j+1}] ✗ finishReason={finish} — prompt likely filtered")
                            elif not raw:
                                print(f"  [img {j+1}] ✗ base64 is empty")
                            else:
                                if not is_islamic and HAS_PIL:
                                    label = extract_label(final_prompt, original_sentence or translated_sentence)
                                    if label:
                                        raw = add_label_to_image(raw, label)
                                        print(f"  [img {j+1}] label: '{label}'")
                                images_b64.append(raw)
                                print(f"  [img {j+1}] ✓ nvidia ({len(raw)} chars)")
                    else:
                        print(f"  [img {j+1}] ✗ HTTP {resp.status_code}: {resp.text[:200]}")
                except Exception as img_err:
                    print(f"  [img {j+1}] ✗ {img_err}")

            if not images_b64:
                print(f"  [scene {i+1}] all images failed")
                frames.append({
                    "scene_number": scene["scene_number"], "title": scene["title"],
                    "narration": narration, "images": [], "audio_b64": None,
                    "kenburns": scene.get("kenburns","zoom-in"), "duration": 30,
                    "error": "All images failed",
                })
                video_jobs[job_id]["frames"] = list(frames)
                continue
            elif len(images_b64) < len(image_prompts):
                print(f"  [scene {i+1}] partial: {len(images_b64)}/{len(image_prompts)} images succeeded")

            # ── Generate TTS ─────────────────────────────
            tts_pct = 65 + int(((i+1)/n)*30)
            update(tts_pct, f"Scene {i+1}/{n}: generating voice...")
            print(f"  [tts] generating ({voice})...")
            audio_b64 = None
            duration  = 30
            try:
                tts_prompt = (
                    f"Calm, authoritative BBC documentary narrator. "
                    f"Tone: warm, measured, knowledgeable. "
                    f"Language: {req.language.capitalize()}. "
                    f"Speak this text exactly:\n\n{narration}"
                )
                tts_resp = client.models.generate_content(
                    model=TTS_MODEL,
                    contents=tts_prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                            )
                        ),
                        # Disable safety blocking — historical/religious educational content
                        # often triggers false positives on battle/war narration
                        safety_settings=[
                            types.SafetySetting(
                                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                                threshold="BLOCK_NONE"
                            ),
                            types.SafetySetting(
                                category="HARM_CATEGORY_HARASSMENT",
                                threshold="BLOCK_NONE"
                            ),
                            types.SafetySetting(
                                category="HARM_CATEGORY_HATE_SPEECH",
                                threshold="BLOCK_NONE"
                            ),
                            types.SafetySetting(
                                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                threshold="BLOCK_NONE"
                            ),
                        ],
                    )
                )

                # Guard against None candidates (safety block or empty response)
                if (not tts_resp.candidates or
                    tts_resp.candidates[0] is None or
                    not tts_resp.candidates[0].content or
                    not tts_resp.candidates[0].content.parts):
                    # Log finish reason if available
                    reason = ""
                    if tts_resp.candidates and tts_resp.candidates[0]:
                        reason = str(getattr(tts_resp.candidates[0], "finish_reason", ""))
                    print(f"  [tts] ✗ blocked or empty response (reason: {reason})")
                    # Retry once with just the narration text (no style prefix)
                    print(f"  [tts] retrying with plain text...")
                    tts_resp = client.models.generate_content(
                        model=TTS_MODEL,
                        contents=narration,
                        config=types.GenerateContentConfig(
                            response_modalities=["AUDIO"],
                            speech_config=types.SpeechConfig(
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                                )
                            ),
                            safety_settings=[
                                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",         threshold="BLOCK_NONE"),
                                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",        threshold="BLOCK_NONE"),
                                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT",  threshold="BLOCK_NONE"),
                            ],
                        )
                    )

                # Extract PCM from whichever attempt succeeded
                if (tts_resp.candidates and
                    tts_resp.candidates[0] and
                    tts_resp.candidates[0].content and
                    tts_resp.candidates[0].content.parts):
                    pcm = tts_resp.candidates[0].content.parts[0].inline_data.data
                    if pcm and len(pcm) > 100:
                        audio_b64 = pcm_to_wav_b64(pcm)
                        duration  = int(wav_duration_seconds(audio_b64))
                        print(f"  [tts] ✓ {duration}s")
                    else:
                        print(f"  [tts] ✗ empty PCM ({len(pcm) if pcm else 0} bytes)")
                else:
                    print(f"  [tts] ✗ no audio after retry")

            except Exception as tts_err:
                print(f"  [tts] ✗ {tts_err}")

            secs_per_img = max(4, duration // max(len(images_b64), 1))
            frames.append({
                "scene_number": scene["scene_number"],
                "title":        scene["title"],
                "narration":    narration,
                "images":       images_b64,
                "audio_b64":    audio_b64,
                "kenburns":     scene.get("kenburns", "zoom-in"),
                "duration":     duration,
                "secs_per_img": secs_per_img,
                "error":        None,
            })
            print(f"  [scene {i+1}] ✓ {len(images_b64)} images, {duration}s audio")

        except Exception as e:
            print(f"  [scene {i+1}] FATAL: {e}")
            frames.append({
                "scene_number": scene.get("scene_number", i+1),
                "title":        scene.get("title", f"Scene {i+1}"),
                "narration":    scene.get("narration", ""),
                "images": [], "audio_b64": None,
                "kenburns": scene.get("kenburns","zoom-in"),
                "duration": 30, "error": str(e),
            })

        video_jobs[job_id]["frames"] = list(frames)

    update(98, "Saving to your account...")
    job = video_jobs.get(job_id, {})
    save_video_to_db(
        job_id=job_id, user_id=job.get("user_id", req.user_id),
        book_id=job.get("book_id", req.book_id),
        book_title=job.get("book_title",""), page=job.get("page", req.current_page),
        language=job.get("language", req.language), frames=frames,
    )
    update(100, "Video ready!")
    video_jobs[job_id]["status"] = "ready"
    print(f"\n[video] ✓ {job_id} — {len(frames)} scenes")


# ════════════════════════════════════════════════════
# POLL / HISTORY / DELETE
# ════════════════════════════════════════════════════
@router.get("/video/status/{job_id}")
async def video_status(job_id: str):
    job = video_jobs.get(job_id)
    if not job: raise HTTPException(404, "Job not found")
    return {
        "status":       job["status"],
        "progress":     job["progress"],
        "progress_msg": job["progress_msg"],
        "eta_seconds":  job["eta_seconds"],
        "elapsed":      int(tm.time() - job["started_at"]),
        "frames":       job["frames"] if job["status"] == "ready" else [],
        "error":        job["error"],
    }

@router.get("/video/history/{user_id}/{book_id}")
async def get_video_history(user_id: str, book_id: str):
    """Fast — returns metadata only, no frames_json. Instant response."""
    return {"history": load_video_history(user_id, book_id)}


@router.get("/video/frames/{video_id}/{user_id}")
async def get_video_frames(video_id: str, user_id: str):
    """
    Load full frames for one specific video — called on demand when
    the user clicks to replay a saved video. Separate from history list
    so history loads instantly even if frames are large.
    """
    frames = load_video_frames(video_id, user_id)
    if frames is None:
        raise HTTPException(404, "Video not found")
    return {"frames": frames, "video_id": video_id}

@router.delete("/video/{video_id}/{user_id}")
async def delete_video(video_id: str, user_id: str):
    ok = delete_video_from_db(video_id, user_id)
    if not ok:
        raise HTTPException(500, "Failed to delete video")
    return {"deleted": True, "video_id": video_id}