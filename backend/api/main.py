from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.agent import WarAndPeaceAgent

load_dotenv()
app = FastAPI()
agent = WarAndPeaceAgent()


class MessageRequest(BaseModel):
    message: str


@app.post('/api/generate', summary='Генерация ответа нейросетью')
async def generate(request: MessageRequest):
    return StreamingResponse(
        agent.astream_answer(query=request.message),
        media_type='text/event-stream'
    )
