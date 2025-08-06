# -*- coding: utf-8 -*-
import os
import uuid
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
from PIL import Image
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# ===== UTF-8 콘솔 출력 지원 =====
sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')

# ===== 환경 변수 로드 =====
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_KEY:
    raise ValueError("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

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


# ===== FastAPI 앱 =====
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== MongoDB 연결 =====
mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client.chat_db
messages_collection = db.messages


# ===== WebSocket 연결 매니저 =====
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


async def ask_gpt_calendar(user_text: str) -> dict:
    prompt = f"""
    사용자가 한글로 일정 관련 요청을 하면, 아래 JSON 형식만 반환해.
    다른 말이나 설명은 절대 하지 말고, 반드시 JSON만 출력해.
    {{
        "action": "create" | "update" | "delete",
        "title": "일정 제목",
        "start_time": "YYYY-MM-DDTHH:MM:SS",
        "end_time": "YYYY-MM-DDTHH:MM:SS",
        "new_time": "YYYY-MM-DDTHH:MM:SS"
    }}
    사용자의 요청: "{user_text}"
    """
    try:
        res = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = res.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            lines = content.split("\n")
            if lines[0].strip().lower().startswith("json"):
                content = "\n".join(lines[1:])
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 실패: {e}\n응답 내용: {content}")
        return {}
    except Exception as e:
        print(f"❌ 캘린더 변환 오류: {e}")
        return {}


# ===== TTS / STT =====
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


# ===== 이미지 생성 & 변형 =====
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


def variation_image(image_b64: str):
    try:
        img_bytes = base64.b64decode(image_b64)
        temp_path = f"temp_variation.png"
        with open(temp_path, "wb") as f:
            f.write(img_bytes)
        res = sync_client.images.variations(
            model="dall-e-2",
            image=open(temp_path, "rb"),
            size="1024x1024"
        )
        os.remove(temp_path)
        return res.data[0].url
    except Exception as e:
        print(f"❌ 이미지 변형 오류: {e}")
        return None


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
                print("⚠️ 잘못된 JSON 수신:", raw_data)
                continue

            timestamp = datetime.utcnow()
            data["timestamp"] = timestamp.isoformat()
            messages_collection.insert_one({
                "nickname": data.get("nickname", nickname),
                "message": data.get("message", ""),
                "type": data.get("type", "text"),
                "timestamp": timestamp
            })

            msg_type = data.get("type", "text")
            message = data.get("message", "")

            if msg_type == "text":
                await manager.broadcast(data)

                if message.startswith("@chatbot"):
                    gpt_reply = await ask_chatgpt(message.replace("@chatbot", "").strip())
                    await manager.broadcast({
                        "nickname": "chatbot",
                        "text": gpt_reply,
                        "timestamp": datetime.utcnow().isoformat()
                    })

                elif message.startswith("@tts"):
                    text_to_convert = message.replace("@tts", "").strip()
                    audio_b64 = await tts_generate(text_to_convert)
                    if audio_b64:
                        await manager.broadcast({
                            "nickname": "chatbot",
                            "text": text_to_convert,
                            "audio": audio_b64,
                            "timestamp": datetime.utcnow().isoformat()
                        })

                elif message.startswith("@calendar"):
                    cal_data = await ask_gpt_calendar(message.replace("@calendar", "").strip())
                    if not cal_data:
                        await manager.broadcast({
                            "nickname": "system",
                            "text": "❌ 캘린더 요청 처리 실패",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        continue

                    def to_kst(iso_str):
                        try:
                            dt = parser.isoparse(iso_str)
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone(timedelta(hours=9)))
                            return dt.astimezone(timezone(timedelta(hours=9)))
                        except Exception as e:
                            print(f"⚠️ 시간 변환 실패: {iso_str}, 오류: {e}")
                            return None

                    start_time_kst = to_kst(cal_data.get("start_time"))
                    new_time_kst = to_kst(cal_data.get("new_time")) if cal_data.get("new_time") else None

                    if not start_time_kst:
                        await manager.broadcast({
                            "nickname": "system",
                            "text": "❌ 시작 시간 변환 실패",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        continue

                    start_time_str = start_time_kst.strftime("%Y-%m-%dT%H:%M:%S")
                    new_time_str = new_time_kst.strftime("%Y-%m-%dT%H:%M:%S") if new_time_kst else None

                    if cal_data.get("action") == "create":
                        result = add_event(cal_data.get("title"), start_time_str)
                    elif cal_data.get("action") == "delete":
                        result = delete_event(cal_data.get("title"), start_time_str)
                    elif cal_data.get("action") == "update" and new_time_str:
                        result = update_event(cal_data.get("title"), start_time_str, new_time_str)
                    else:
                        result = "[X] 알 수 없는 action 또는 잘못된 데이터"

                    await manager.broadcast({
                        "nickname": "system",
                        "text": result,
                        "timestamp": datetime.utcnow().isoformat()
                    })

                elif message.startswith("@image"):
                    prompt = message.replace("@image", "").strip()
                    url = generate_image(prompt)
                    if url:
                        await manager.broadcast({
                            "nickname": "chatbot",
                            "text": "[이미지 생성 결과]",
                            "image_url": url,
                            "timestamp": datetime.utcnow().isoformat()
                        })

                elif message.startswith("@variation"):
                    img_b64 = data.get("imageData", "")
                    url = variation_image(img_b64)
                    if url:
                        await manager.broadcast({
                            "nickname": "chatbot",
                            "text": "[이미지 변형 결과]",
                            "image_url": url,
                            "timestamp": datetime.utcnow().isoformat()
                        })

            elif msg_type == "audio":
                if message.startswith("@stt"):
                    transcript = await stt_transcribe(data["audioData"])
                    await manager.broadcast({
                        "nickname": nickname,
                        "text": transcript,
                        "timestamp": datetime.utcnow().isoformat()
                    })

                elif message.startswith("@talk"):
                    transcript = await stt_transcribe(data["audioData"])
                    gpt_reply = await ask_chatgpt(transcript)
                    audio_b64 = await tts_generate(gpt_reply)
                    if audio_b64:
                        await manager.broadcast({
                            "nickname": "chatbot",
                            "text": gpt_reply,
                            "audio": audio_b64,
                            "timestamp": datetime.utcnow().isoformat()
                        })

    except WebSocketDisconnect:
            await manager.disconnect(websocket)
    except Exception as e:
            print("웹소켓 예외 발생:", e)
            await manager.disconnect(websocket)




# ===== 이미지 편집 =====
@app.post("/image-edit")
async def image_edit(
    image: UploadFile = File(...),
    mask: UploadFile = File(None),
    prompt: str = Form(...)
):
    temp_dir = tempfile.mkdtemp()
    try:
        def safe_filename(extension):
            return f"{uuid.uuid4().hex}{extension}"

        image_ext = os.path.splitext(image.filename)[1] or ".png"
        image_path = os.path.join(temp_dir, safe_filename(image_ext))
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        mask_path = None
        if mask:
            mask_ext = os.path.splitext(mask.filename)[1] or ".png"
            mask_path = os.path.join(temp_dir, safe_filename(mask_ext))
            with open(mask_path, "wb") as f:
                shutil.copyfileobj(mask.file, f)
            with Image.open(mask_path).convert("RGBA") as m, Image.open(image_path) as img:
                if m.size != img.size:
                    m = m.resize(img.size)
                datas = m.getdata()
                new_data = []
                for item in datas:
                    if item[0] == 0 and item[1] == 0 and item[2] == 0:
                        new_data.append((0, 0, 0, 0))
                    else:
                        new_data.append(item)
                m.putdata(new_data)
                m.save(mask_path, "PNG")

        prompt_utf8 = str(prompt)
        params = {
            "model": "dall-e-2",
            "image": open(image_path, "rb"),
            "prompt": prompt_utf8,
            "size": "1024x1024"
        }
        if mask_path:
            params["mask"] = open(mask_path, "rb")

        response = sync_client.images.edit(**params)
        return {"url": response.data[0].url}
    except Exception as e:
        return {"error": str(e)}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
