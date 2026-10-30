from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import upload, chat, notes, voice, video
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="AI Professor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-app.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api")
app.include_router(chat.router,   prefix="/api")
app.include_router(voice.router,  prefix="/api")
app.include_router(video.router,  prefix="/api")
app.include_router(notes.router,  prefix="/api")

@app.get("/")
def health():
    return {"status": "AI Professor API is running"}