"""
Shared NVIDIA NIM image generation helper.
Used by both video.py and voice.py.
"""
import httpx, base64, os

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE    = "https://integrate.api.nvidia.com/v1"

# Model priority list — flux-schnell is fast (good for voice),
# stable-diffusion-xl is higher quality (good for video).
NVIDIA_MODELS_VIDEO = [
    "black-forest-labs/flux-dev",         # best quality, ~8s
    "stabilityai/stable-diffusion-xl",    # reliable fallback
    "black-forest-labs/flux-schnell",     # fast fallback
]
NVIDIA_MODELS_VOICE = [
    "black-forest-labs/flux-schnell",     # ~3-4s — fast for real-time feel
    "stabilityai/stable-diffusion-xl",    # fallback
]

async def generate_image_nvidia(
    prompt: str,
    size: str = "1344x768",
    mode: str = "video"          # "video" | "voice"
) -> str | None:
    """
    Generate image with NVIDIA NIM. Returns base64 string or None.
    Tries models in priority order and returns first success.
    """
    if not NVIDIA_API_KEY:
        print("[nvidia] NVIDIA_API_KEY not set — skipping")
        return None

    models = NVIDIA_MODELS_VIDEO if mode == "video" else NVIDIA_MODELS_VOICE

    for model in models:
        try:
            print(f"[nvidia] trying {model} …")
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(90, connect=10)
            ) as client:
                resp = await client.post(
                    f"{NVIDIA_BASE}/images/generations",
                    headers={
                        "Authorization": f"Bearer {NVIDIA_API_KEY}",
                        "Content-Type":  "application/json",
                        "Accept":        "application/json",
                    },
                    json={
                        "model":           model,
                        "prompt":          prompt,
                        "n":               1,
                        "size":            size,
                        "response_format": "b64_json",
                        "quality":         "standard",
                    }
                )

            if resp.status_code == 200:
                data = resp.json()
                item = data["data"][0]
                b64  = item.get("b64_json") or item.get("url", "")

                if b64 and not b64.startswith("http"):
                    print(f"[nvidia] ✓ {model} — b64 ({len(b64)} chars)")
                    return b64

                if b64.startswith("http"):
                    async with httpx.AsyncClient(timeout=30) as dl:
                        img_r = await dl.get(b64)
                    result = base64.b64encode(img_r.content).decode("utf-8")
                    print(f"[nvidia] ✓ {model} — url downloaded")
                    return result

            else:
                print(f"[nvidia] {model} → HTTP {resp.status_code}: {resp.text[:120]}")

        except Exception as e:
            print(f"[nvidia] {model} failed: {e}")
            continue

    print("[nvidia] ✗ all models failed")
    return None