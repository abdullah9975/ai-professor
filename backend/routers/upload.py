from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import fitz
import uuid, os
from dotenv import load_dotenv
from supabase import create_client
import io
from PIL import Image

load_dotenv()
router = APIRouter()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# ── Try to import OCR library if available ──────────────
try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

def extract_text_from_page(page):
    try:
        text = page.get_text()
        return text.strip() if text else ""
    except Exception:
        return ""

@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    title: str = Form(...),
    is_rtl: str = Form("false"),
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    pdf_bytes = await file.read()
    if len(pdf_bytes) > 70 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size must be under 70MB")

    book_id  = str(uuid.uuid4())
    rtl_flag = is_rtl.lower() == "true"

    try:
        pdf_doc     = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(pdf_doc)

        # ── Insert book record FIRST ─────────────────────
        supabase.table("books").insert({
            "id":             book_id,
            "user_id":        user_id,
            "title":          title,
            "total_pages":    total_pages,
            "last_page_read": 1,
            "is_rtl":         rtl_flag,
        }).execute()

        # ── Process each page ────────────────────────────
        for page_num in range(total_pages):
            page      = pdf_doc[page_num]
            mat       = fitz.Matrix(2, 2)
            pix       = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")

            storage_path = f"{user_id}/{book_id}/page_{page_num + 1}.png"
            supabase.storage.from_("book-pages").upload(
                storage_path, img_bytes, {"content-type": "image/png"}
            )

            page_text = extract_text_from_page(page)
            supabase.table("pages").insert({
                "book_id":      book_id,
                "page_number":  page_num + 1,
                "text_content": page_text,
                "image_url":    storage_path,
            }).execute()

        pdf_doc.close()

        return JSONResponse({
            "book_id":     book_id,
            "total_pages": total_pages,
            "title":       title,
            "is_rtl":      rtl_flag,
            "status":      "success",
        })

    except Exception as e:
        try:
            supabase.table("books").delete().eq("id", book_id).execute()
        except:
            pass
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")


# ════════════════════════════════════════════════════
# DELETE BOOK — removes book, all pages rows, and
# all page images from storage in one call.
# Only the book owner can delete their own book.
# ════════════════════════════════════════════════════
@router.delete("/books/{book_id}")
async def delete_book(book_id: str, user_id: str):
    """
    Delete a book and all associated data:
    1. Verify ownership
    2. Delete page images from storage
    3. Delete pages rows from DB
    4. Delete book row from DB
    """
    # 1. Verify the book belongs to this user
    book_resp = supabase.table("books") \
        .select("id, user_id, total_pages") \
        .eq("id", book_id) \
        .single() \
        .execute()

    if not book_resp.data:
        raise HTTPException(status_code=404, detail="Book not found")

    if book_resp.data["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Not authorised to delete this book")

    total_pages = book_resp.data.get("total_pages", 0)
    errors = []

    # 2. Delete all page images from storage
    # Build list of all expected storage paths
    image_paths = [
        f"{user_id}/{book_id}/page_{i + 1}.png"
        for i in range(total_pages)
    ]
    if image_paths:
        try:
            supabase.storage.from_("book-pages").remove(image_paths)
            print(f"[delete] ✓ removed {len(image_paths)} images from storage")
        except Exception as e:
            # Non-fatal — continue with DB deletion
            errors.append(f"Storage cleanup partial: {str(e)}")
            print(f"[delete] storage removal error (non-fatal): {e}")

    # 3. Delete all page rows
    try:
        supabase.table("pages").delete().eq("book_id", book_id).execute()
        print(f"[delete] ✓ deleted pages rows for book {book_id}")
    except Exception as e:
        errors.append(f"Pages deletion failed: {str(e)}")
        print(f"[delete] pages deletion error: {e}")

    # 4. Delete video history if any
    try:
        supabase.table("video_history").delete().eq("book_id", book_id).execute()
        print(f"[delete] ✓ deleted video history for book {book_id}")
    except Exception as e:
        print(f"[delete] video history cleanup (non-fatal): {e}")

    # 5. Delete the book itself
    try:
        supabase.table("books").delete().eq("id", book_id).execute()
        print(f"[delete] ✓ deleted book {book_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete book: {str(e)}")

    return {
        "deleted": True,
        "book_id": book_id,
        "warnings": errors,  # Surface any non-fatal cleanup issues
    }