from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from dotenv import load_dotenv
from google import genai
from google.genai import types
import os, json, asyncio, httpx, base64
from supabase import create_client
import httpx

load_dotenv()

router   = APIRouter()
client   = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

MODEL          = "gemini-2.5-flash-native-audio-preview-12-2025"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_URL     = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-dev"

LANG_MAP = {
    "hindi":   "Always respond in Hindi.",
    "urdu":    "Always respond in Urdu.",
    "arabic":  "Always respond in Arabic.",
    "bengali": "Always respond in Bengali.",
    "tamil":   "Always respond in Tamil.",
    "english": "Always respond in English.",
}

# ── Keywords that mean a visual would genuinely help ──
VISUAL_KEYWORDS = [
    # maths / graphs
    "graph", "plot", "curve", "y =", "y=", "f(x)", "x-axis", "y-axis",
    "coordinate", "slope", "intercept", "parabola", "sine", "cosine",
    "tangent", "hyperbola", "ellipse", "derivative", "integral", "gradient",
    "linear equation", "quadratic", "trigonometric",
    # science diagrams
    "diagram", "circuit", "molecule", "atom", "cell", "structure",
    "orbit", "wave", "frequency", "amplitude", "spectrum", "dna",
    # user asking for visual
    "imagine", "visualize", "picture this", "looks like", "look like",
    "hard to imagine", "cannot imagine", "can't imagine", "show me",
    "draw", "sketch", "what does", "i don't understand how",
]

def should_generate_visual(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in VISUAL_KEYWORDS)


# ── Step 1: Ask Gemini to write the best image prompt ──
async def build_image_prompt(ai_response: str, user_question: str) -> str:
    meta = f"""The AI professor just said:
"{ai_response[:500]}"

The student asked:
"{user_question[:200]}"

Write a SHORT image-generation prompt (max 60 words) for a clean EDUCATIONAL DIAGRAM
that visually shows the key concept above.

Rules:
- For math graphs: mention the exact shape (straight line / parabola / sine wave etc.),
  labelled axes (x-axis, y-axis), key points and numbers.
- For science: labelled parts, arrows showing relationships.
- Always add: "clean educational diagram, white background, textbook style, no people"
- Return ONLY the image prompt — no explanation, no preamble."""

    try:
        r = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=meta,
            config=types.GenerateContentConfig(temperature=0.2),
        )
        return r.text.strip()
    except Exception as e:
        print(f"[visual] prompt build failed: {e}")
        return (
            f"Clean educational diagram illustrating: {ai_response[:100]}. "
            "White background, labelled axes, textbook style."
        )


# ── Step 2: Generate image with NVIDIA NIM ────────────
async def nvidia_generate(prompt: str) -> str | None:
    if not NVIDIA_API_KEY:
        print("[visual] NVIDIA_API_KEY not set")
        return None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=10)) as http:
            resp = await http.post(
                NVIDIA_URL,
                headers={
                    "Authorization": f"Bearer {NVIDIA_API_KEY}",
                    "Content-Type":  "application/json",
                    "Accept":        "application/json",
                },
                json={
                    "prompt":    prompt,
                    "mode":      "base",
                    "cfg_scale": 3.5,
                    "width":     1024,
                    "height":    1024,
                    "seed":      0,
                    "steps":     30,
                }
            )
        if resp.status_code == 200:
            d         = resp.json()
            items     = d.get("artifacts", [])
            if not items:
                print("[visual] empty response from NVIDIA")
                return None
            item   = items[0]
            finish = item.get("finishReason", "SUCCESS")
            raw    = item.get("base64", "")
            if finish != "SUCCESS" or not raw:
                print(f"[visual] NVIDIA finishReason={finish}")
                return None
            print(f"[visual] ✓ NVIDIA ({len(raw)} chars)")
            return raw
        else:
            print(f"[visual] NVIDIA HTTP {resp.status_code}: {resp.text[:150]}")
            return None
    except Exception as e:
        print(f"[visual] NVIDIA error: {e}")
        return None
    
    
# ── Step 3: Orchestrate — prompt → image → send ───────
async def generate_and_send_visual(
    ai_text: str, user_question: str, ws: WebSocket
) -> None:
    try:
        # Tell the frontend to show a loading spinner immediately
        await ws.send_json({"type": "visual_loading"})

        img_prompt = await build_image_prompt(ai_text, user_question)
        print(f"[visual] prompt → {img_prompt[:90]}…")

        b64 = await nvidia_generate(img_prompt)

        if b64:
            await ws.send_json({
                "type":      "visual_image",
                "image_b64": b64,
                "caption":   img_prompt[:130],
            })
        else:
            await ws.send_json({"type": "visual_failed"})

    except Exception as e:
        print(f"[visual] orchestrate error: {e}")
        try:
            await ws.send_json({"type": "visual_failed"})
        except Exception:
            pass


# ── DB helpers ────────────────────────────────────────
def get_two_pages_text(book_id: str, page: int) -> tuple[str, str]:
    try:
        res = supabase.table("pages") \
            .select("page_number,text_content") \
            .eq("book_id", book_id) \
            .in_("page_number", [page, page + 1]) \
            .execute()
        m = {r["page_number"]: (r.get("text_content") or "").strip() for r in (res.data or [])}
        return m.get(page, ""), m.get(page + 1, "")
    except Exception:
        return "", ""


def get_book_summary(book_id: str, current_page: int) -> str:
    try:
        res = supabase.table("pages") \
            .select("page_number,text_content") \
            .eq("book_id", book_id) \
            .order("page_number") \
            .execute()
        rows = res.data or []
        if not rows:
            return ""
        pm       = {r["page_number"]: (r.get("text_content") or "").strip() for r in rows}
        all_nums = sorted(pm.keys())
        selected = set(all_nums[:10]) | set(all_nums[-10:])
        selected |= {n for n in all_nums if abs(n - current_page) <= 8}
        parts    = [f"[p{n}] {pm[n][:120].replace(chr(10),' ')}" for n in sorted(selected) if pm.get(n)]
        return "\n".join(parts)[:5000]
    except Exception as e:
        print(f"[summary] {e}")
        return ""


def get_book_title(book_id: str) -> str:
    try:
        r = supabase.table("books").select("title").eq("id", book_id).single().execute()
        return r.data.get("title", "this book") if r.data else "this book"
    except Exception:
        return "this book"


def build_system_instruction(
    book_title, page, p1, p2, summary, lang_instr
) -> str:
    pages_section = f"=== PAGE {page} ===\n{p1 or '[image page]'}"
    if p2:
        pages_section += f"\n\n=== PAGE {page+1} ===\n{p2}"
    book_section = f"\n\n=== BOOK OVERVIEW ===\n{summary}" if summary else ""

    return (
        f'You are a brilliant AI professor teaching "{book_title}". '
        f"The student has pages {page} and {page+1} open.\n\n"

        f"VISUAL RULE (CRITICAL):\n"
        f"Whenever the student cannot imagine something — a graph, diagram, structure, "
        f"wave, molecule, circuit, or anything visual — you MUST explain it AND naturally "
        f"say something like 'let me show you that' or 'here is what that looks like'. "
        f"Do NOT ask for permission. Do NOT say 'should I generate'. "
        f"Just say it as a confident professor and the system will auto-generate the diagram. "
        f"This also triggers automatically whenever you mention a graph or equation.\n\n"

        f"ANSWER RULES:\n"
        f"- Short, conversational answers — 2 to 4 sentences.\n"
        f"- Prioritise the current pages; use book overview for wider questions.\n"
        f"- Speak like a knowledgeable teacher, not a textbook.\n"
        f"- {lang_instr}\n"
        f"\n{pages_section}{book_section}"
    )


# ── WebSocket endpoint ────────────────────────────────
@router.websocket("/voice")
async def voice_endpoint(browser_ws: WebSocket):
    await browser_ws.accept()
    visual_task = None  # track it at endpoint scope so we can cancel it

    try:
        cfg          = json.loads(await browser_ws.receive_text())
        book_id      = cfg.get("book_id", "")
        current_page = cfg.get("current_page", 1)
        language     = cfg.get("language", "english")
        lang_instr   = LANG_MAP.get(language, LANG_MAP["english"])

        book_title = get_book_title(book_id)
        p1, p2     = get_two_pages_text(book_id, current_page)
        summary    = get_book_summary(book_id, current_page)
        sys_instr  = build_system_instruction(
            book_title, current_page, p1, p2, summary, lang_instr
        )

        live_cfg = {
            "response_modalities":        ["AUDIO"],
            "system_instruction":         sys_instr,
            "input_audio_transcription":  {},
            "output_audio_transcription": {},
        }

        async with client.aio.live.connect(model=MODEL, config=live_cfg) as session:
            await browser_ws.send_json({"type": "ready"})

            last_user_q: list[str] = [""]

            async def browser_to_gemini():
                try:
                    while True:
                        msg = await browser_ws.receive()
                        if msg["type"] == "websocket.disconnect":
                            break
                        if "bytes" in msg and msg["bytes"]:
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=msg["bytes"],
                                    mime_type="audio/pcm;rate=16000"
                                )
                            )
                        elif "text" in msg and msg["text"]:
                            d = json.loads(msg["text"])
                            if d.get("type") == "text":
                                await session.send_realtime_input(text=d.get("content", ""))
                except WebSocketDisconnect:
                    pass
                except Exception as e:
                    print(f"[b→g] {e}")

            async def gemini_to_browser():
                nonlocal visual_task  # so we can cancel from finally
                ai_parts: list[str] = []

                try:
                    while True:
                        turn = session.receive()
                        async for resp in turn:
                            sc = resp.server_content if resp.server_content else None

                            if sc and sc.model_turn:
                                for part in sc.model_turn.parts:
                                    if part.inline_data and isinstance(part.inline_data.data, bytes):
                                        await browser_ws.send_bytes(part.inline_data.data)

                            if sc and sc.input_transcription:
                                txt = sc.input_transcription.text or ""
                                if txt.strip():
                                    last_user_q[0] = txt
                                await browser_ws.send_json({"type": "user_transcript", "text": txt})

                            if sc and sc.output_transcription:
                                txt = sc.output_transcription.text or ""
                                ai_parts.append(txt)
                                await browser_ws.send_json({"type": "ai_transcript", "text": txt})

                            if sc and sc.turn_complete:
                                await browser_ws.send_json({"type": "turn_complete"})

                                full = " ".join(ai_parts).strip()
                                ai_parts.clear()

                                if visual_task and not visual_task.done():
                                    visual_task.cancel()

                                if full and should_generate_visual(full):
                                    visual_task = asyncio.create_task(
                                        generate_and_send_visual(full, last_user_q[0], browser_ws)
                                    )

                except WebSocketDisconnect:
                    pass
                except Exception as e:
                    print(f"[g→b] {e}")
                    try:
                        await browser_ws.send_json({"type": "error", "message": str(e)})
                    except Exception:
                        pass

            await asyncio.gather(
                browser_to_gemini(),
                gemini_to_browser(),
                return_exceptions=True,
            )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[voice_endpoint] {e}")
        try:
            await browser_ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        # Cancel any in-flight image generation BEFORE closing
        if visual_task and not visual_task.done():
            visual_task.cancel()
            try:
                await visual_task  # let cancellation propagate cleanly
            except asyncio.CancelledError:
                pass
        try:
            await browser_ws.close()
        except Exception:
            pass