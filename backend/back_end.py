# -*- coding: utf-8 -*-
import os
import sys
import json
import base64
import shutil
import tempfile
from datetime import datetime, timezone, timedelta
from typing import Dict
from dateutil import parser
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from PIL import Image

# ===== UTF-8 출력 =====
sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')

# ===== 환경 변수 로드 =====
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_KEY:
    raise ValueError("❌ OPENAI_API_KEY 환경 변수가 없습니다.")

openai_client = AsyncOpenAI(api_key=OPENAI_KEY)
sync_client = OpenAI(api_key=OPENAI_KEY)

# ===== Google Calendar API =====
SCOPES = ['https://www.googleapis.com/auth/calendar']
TOKEN_FILE = 'token.json'
CREDS_FILE = 'credentials.json'

def get_calendar_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    return build('calendar', 'v3', credentials=creds)

def add_event(title, start_time_str, duration_hours=1):
    service = get_calendar_service()
    start = datetime.fromisoformat(start_time_str)
    end = start + timedelta(hours=duration_hours)
    event = {
        'summary': title,
        'start': {'dateTime': start.isoformat(), 'timeZone': 'Asia/Seoul'},
        'end': {'dateTime': end.isoformat(), 'timeZone': 'Asia/Seoul'}
    }
    service.events().insert(calendarId='primary', body=event).execute()
    return f"[✓] '{title}' 일정이 등록되었습니다."

def find_event(service, title, start_time_str):
    start = datetime.fromisoformat(start_time_str).astimezone(timezone.utc)
    end = start + timedelta(hours=1)
    events_result = service.events().list(
        calendarId='primary',
        timeMin=start.isoformat(),
        timeMax=end.isoformat(),
        singleEvents=True,
        orderBy='startTime'
    ).execute()
    for event in events_result.get('items', []):
        if event.get('summary') == title:
            return event
    return None

def delete_event(title, start_time_str):
    service = get_calendar_service()
    event = find_event(service, title, start_time_str)
    if not event:
        return "[X] 삭제할 일정을 찾을 수 없습니다."
    service.events().delete(calendarId='primary', eventId=event['id']).execute()
    return f"[✓] '{title}' 일정이 삭제되었습니다."

def update_event(title, old_time_str, new_time_str):
    service = get_calendar_service()
    event = find_event(service, title, old_time_str)
    if not event:
        return "[X] 수정할 일정을 찾을 수 없습니다."
    new_start = datetime.fromisoformat(new_time_str)
    new_end = new_start + timedelta(hours=1)
    event['start']['dateTime'] = new_start.isoformat()
    event['end']['dateTime'] = new_end.isoformat()
    service.events().update(calendarId='primary', eventId=event['id'], body=event).execute()
    return f"[✓] '{title}' 일정이 수정되었습니다."

# ===== FastAPI =====
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== MongoDB =====
mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client.chat_db
messages_collection = db.messages

# ===== WebSocket 매니저 =====
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, nickname: str):
        await websocket.accept()
        self.active_connections[nickname] = websocket

    def disconnect(self, nickname: str):
        self.active_connections.pop(nickname, None)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections.values()):
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"⚠️ 브로드캐스트 실패: {e}")

manager = ConnectionManager()

# ===== GPT =====
async def ask_chatgpt(prompt: str) -> str:
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 친절한 상담봇입니다."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[GPT 오류] {str(e)}"

# ===== KST 변환 (오늘/내일 + 시간까지 인식) =====
def to_kst(iso_str):
    try:
        text = iso_str.strip()
        now = datetime.now(timezone(timedelta(hours=9)))

        if text.startswith("오늘"):
            base_date = now.date()
            time_part = text.replace("오늘", "").strip()
        elif text.startswith("내일"):
            base_date = (now + timedelta(days=1)).date()
            time_part = text.replace("내일", "").strip()
        else:
            dt = parser.parse(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone(timedelta(hours=9)))
            return dt.astimezone(timezone(timedelta(hours=9)))

        if not time_part:
            dt = datetime.combine(base_date, datetime.min.time()).replace(hour=9, tzinfo=timezone(timedelta(hours=9)))
        else:
            parsed_time = parser.parse(time_part).time()
            dt = datetime.combine(base_date, parsed_time).replace(tzinfo=timezone(timedelta(hours=9)))

        return dt
    except Exception as e:
        print(f"⚠️ 시간 변환 실패: {iso_str}, 오류: {e}")
        return None

# ===== 도구 목록 =====
tools = [
    {
        "type": "function",
        "function": {
            "name": "chatbot_reply",
            "description": "사용자의 질문에 답변합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"}
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "프롬프트를 기반으로 이미지를 생성합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"}
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tts_generate",
            "description": "텍스트를 음성으로 변환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_event",
            "description": "구글 캘린더 일정 추가/삭제/수정. 자연어 날짜·시간 인식 가능.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["create", "delete", "update"]},
                    "title": {"type": "string"},
                    "start_time": {"type": "string"},
                    "new_time": {"type": "string"}
                },
                "required": ["action", "title", "start_time"]
            }
        }
    }
]

# ===== 이미지 생성 =====
def generate_image(prompt: str):
    try:
        res = sync_client.images.generate(
            model="dall-e-2",
            prompt=prompt,
            size="1024x1024"
        )
        return res.data[0].url
    except Exception as e:
        print(f"❌ 이미지 생성 오류: {e}")
        return None

async def tts_generate(text):
    try:
        speech = await openai_client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )
        return base64.b64encode(speech.content).decode("utf-8")
    except Exception as e:
        print(f"❌ TTS 오류: {e}")
        return None

async def stt_transcribe(audio_b64, format="mp3"):
    try:
        audio_bytes = base64.b64decode(audio_b64)
        temp_path = f"temp_input.{format}"
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
        with open(temp_path, "rb") as audio_file:
            transcript = await openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        os.remove(temp_path)
        return transcript.text.strip()
    except Exception as e:
        print(f"❌ STT 오류: {e}")
        return ""

# ===== 이미지 마스킹 편집 =====
@app.post("/image-edit")
async def image_edit(image: UploadFile = File(...), mask: UploadFile = File(None), prompt: str = Form(...)):
    temp_dir = tempfile.mkdtemp()
    try:
        image_path = os.path.join(temp_dir, "image.png")
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        mask_path = None
        if mask:
            mask_path = os.path.join(temp_dir, "mask.png")
            with open(mask_path, "wb") as f:
                shutil.copyfileobj(mask.file, f)

        params = {
            "model": "dall-e-2",
            "image": open(image_path, "rb"),
            "prompt": prompt,
            "size": "1024x1024"
        }
        if mask_path:
            params["mask"] = open(mask_path, "rb")

        res = sync_client.images.edit(**params)
        return {"url": res.data[0].url}
    except Exception as e:
        return {"error": str(e)}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# ===== WebSocket =====
@app.websocket("/ws/{nickname}")
async def websocket_endpoint(websocket: WebSocket, nickname: str):
    await manager.connect(websocket, nickname)
    try:
        while True:
            raw_data = await websocket.receive_text()
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type", "text")
            message = data.get("message", "")

            if msg_type == "text":
                await manager.broadcast({
                    "nickname": nickname,
                    "text": data.get("text", message),
                    "type": "text"
                })

            if msg_type == "text" and message.startswith("@gpt"):
                user_text = message.replace("@gpt", "").strip()
                response = await openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": user_text}],
                    tools=tools,
                    tool_choice="auto"
                )

                if not response.choices[0].message.tool_calls:
                    await manager.broadcast({"nickname": "chatbot", "text": "[GPT 응답 없음]"})
                    continue

                tool_call = response.choices[0].message.tool_calls[0]
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if func_name == "chatbot_reply":
                    result = await ask_chatgpt(args["question"])
                    await manager.broadcast({"nickname": "chatbot", "text": result})

                elif func_name == "generate_image":
                    url = generate_image(args["prompt"])
                    await manager.broadcast({"nickname": "chatbot", "text": "[이미지 생성]", "image_url": url})

                elif func_name == "tts_generate":
                    audio_b64 = await tts_generate(args["text"])
                    await manager.broadcast({"nickname": "chatbot", "text": args["text"], "audio": audio_b64})

                elif func_name == "calendar_event":
                    start_time_kst = to_kst(args.get("start_time"))
                    new_time_kst = to_kst(args.get("new_time")) if args.get("new_time") else None

                    if not start_time_kst:
                        await manager.broadcast({"nickname": "system", "text": "❌ 시작 시간 변환 실패"})
                        continue

                    start_time_str = start_time_kst.strftime("%Y-%m-%dT%H:%M:%S")
                    new_time_str = new_time_kst.strftime("%Y-%m-%dT%H:%M:%S") if new_time_kst else None

                    if args["action"] == "create":
                        result = add_event(args["title"], start_time_str)
                    elif args["action"] == "delete":
                        result = delete_event(args["title"], start_time_str)
                    elif args["action"] == "update" and new_time_str:
                        result = update_event(args["title"], start_time_str, new_time_str)
                    else:
                        result = "[X] 잘못된 action 또는 시간 데이터"
                    await manager.broadcast({"nickname": "system", "text": result})

            elif msg_type == "audio":
                if message.startswith("@stt"):
                    transcript = await stt_transcribe(data["audioData"])
                    await manager.broadcast({"nickname": nickname, "text": transcript})
                elif message.startswith("@talk"):
                    transcript = await stt_transcribe(data["audioData"])
                    gpt_reply = await ask_chatgpt(transcript)
                    audio_b64 = await tts_generate(gpt_reply)
                    await manager.broadcast({"nickname": "chatbot", "text": gpt_reply, "audio": audio_b64})

    except WebSocketDisconnect:
        manager.disconnect(nickname)
    except Exception as e:
        print("웹소켓 예외 발생:", e)
        manager.disconnect(nickname)

