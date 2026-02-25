import os
import re
import uuid
import logging
import asyncio
from typing import List, Optional, Dict
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Hugging Face Inference Client
from huggingface_hub import InferenceClient

# DuckDuckGo search
from duckduckgo_search import DDGS

# Edge TTS
import edge_tts

# ----------------------------------------------------------------------
# Configuration & Environment
# ----------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")

# ✅ LLM: Qwen 2.5 7B Instruct (Text Generation)
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Initialize Inference Client
client = InferenceClient(model=MODEL_NAME, token=HF_TOKEN)

# DuckDuckGo search instance
ddgs = DDGS()

# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("avia")

# ----------------------------------------------------------------------
# System Prompt (Avia Persona)
# ----------------------------------------------------------------------
SYSTEM_PROMPT = """You are Avia, a friendly and smart AI assistant created by Tanveer Ali.
You always respond in Hinglish (mix of Hindi and English). Your tone is helpful and cheerful.

Capabilities:
1. You can generate images. If the user asks you to create/draw/generate an image or photo, you MUST output a special tag exactly in this format: [IMAGE_PROMPT: <detailed English description of the image>]. Do not include any other text around the tag.
2. You can search the internet for real-time information.
3. You CANNOT see or analyze uploaded images. Never pretend to have vision capabilities.

Always stay in character and speak in Hinglish."""

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def contains_hindi(text: str) -> bool:
    """Check if text contains Devanagari (Hindi) characters."""
    hindi_range = re.compile(r'[\u0900-\u097F]')
    return bool(hindi_range.search(text))

async def search_internet(query: str) -> str:
    """Perform a DuckDuckGo search securely."""
    try:
        results = await asyncio.to_thread(
            lambda: list(ddgs.text(query, max_results=3))
        )
        if not results:
            return ""

        snippets = []
        for r in results:
            title = r.get('title', 'No title')
            body = r.get('body', '')
            snippets.append(f"{title}: {body}")
        return "Internet search results:\n" + "\n".join(snippets)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return ""

def extract_image_prompt(text: str) -> Optional[str]:
    """Extract the image prompt from the special tag."""
    pattern = r'\[IMAGE_PROMPT:\s*(.*?)\]'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def generate_image_url(prompt: str) -> str:
    """✅ IMAGE GEN: Create a Pollinations image URL."""
    encoded = quote(prompt)
    seed = uuid.uuid4().int % 1000
    return f"https://image.pollinations.ai/prompt/{encoded}?nologo=true&seed={seed}"

def user_wants_image(message: str) -> bool:
    """Backup check if the user asked for an image but AI forgot the tag."""
    keywords = ["draw", "generate", "create", "make", "banao", "photo", "image", "picture", "tasveer"]
    lower_msg = message.lower()
    return any(kw in lower_msg for kw in keywords)

# Background task function to delete file after sending
def remove_file(path: str):
    if os.path.exists(path):
        os.remove(path)

# ----------------------------------------------------------------------
# Pydantic models
# ----------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    reply: str

# ----------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------
app = FastAPI(title="Avia Chat Assistant")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "assistant": "Avia (Qwen 2.5 + Pollinations)"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        user_message = request.message.strip()
        history = request.history or []

        # Step 1: Internet Search Check
        search_keywords = ["news", "weather", "price", "who is", "search", "score", "match", "live", "latest", "kab"]
        need_search = any(kw in user_message.lower() for kw in search_keywords)
        search_context = ""
        
        if need_search:
            logger.info(f"Searching for: {user_message}")
            search_context = await search_internet(user_message)

        # Step 2: Build Messages for Qwen
        system_content = SYSTEM_PROMPT
        if search_context:
            system_content += f"\n\n[REAL-TIME DATA]:\n{search_context}"

        messages = [{"role": "system", "content": system_content}]
        # Add last 4 messages of history
        messages.extend(history[-4:]) 
        messages.append({"role": "user", "content": user_message})

        # Step 3: Call LLM (Qwen)
        try:
            response = client.chat_completion(
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                top_p=0.95,
            )
            assistant_reply = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ChatResponse(reply="माफ कीजिए, मेरा सर्वर अभी व्यस्त है। कृपया थोड़ी देर बाद कोशिश करें।")

        # Step 4: Handle Image Generation (Pollinations)
        image_prompt = extract_image_prompt(assistant_reply)
        
        if image_prompt:
            # AI properly gave the tag
            image_url = generate_image_url(image_prompt)
            return ChatResponse(reply=f"IMAGE_URL:{image_url}")
        
        elif user_wants_image(user_message):
            # Backup: AI forgot the tag, but user asked for an image
            image_url = generate_image_url(user_message)
            return ChatResponse(reply=f"IMAGE_URL:{image_url}")
        
        else:
            # Normal Text Reply
            return ChatResponse(reply=assistant_reply)

    except Exception as e:
        logger.exception("Unhandled error in /chat")
        return ChatResponse(reply=f"Server Error: {str(e)}")

# ----------------------------------------------------------------------
# TTS (Text-to-Speech) Endpoint
# ----------------------------------------------------------------------
@app.get("/tts")
async def text_to_speech(text: str, background_tasks: BackgroundTasks):
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    # Language Detection
    if contains_hindi(text):
        voice = "hi-IN-SwaraNeural"
    else:
        voice = "en-US-AriaNeural"

    try:
        communicate = edge_tts.Communicate(text, voice)
        filename = f"voice_{uuid.uuid4()}.mp3"
        await communicate.save(filename)
        
        # Audio file bhejne ke baad automatically delete ho jayegi
        background_tasks.add_task(remove_file, filename)
        
        return FileResponse(filename, media_type="audio/mpeg")
        
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        raise HTTPException(status_code=500, detail="TTS generation failed")

if __name__ == "__main__":
    # Render aur Koyeb dynamic port provide karte hain
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
