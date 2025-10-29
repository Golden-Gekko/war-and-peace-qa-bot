from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent import WarAndPeaceAgent

load_dotenv()
app = FastAPI(prefix='/api', tags=['Bot-API'])
agent = WarAndPeaceAgent()


class MessageRequest(BaseModel):
    message: str


@app.post('/generate', summary='Генерация ответа нейросетью')
async def generate(request: MessageRequest):
    # async def event_stream():
    #     try:
    #         async for token in agent.astream_answer(
    #             query=request.message
    #         ):
    #             yield token
    #         yield 'data: [DONE]\n\n'
    #     except Exception as e:
    #         yield f'data: [ERROR] {str(e)}\n\n'

    # return StreamingResponse(event_stream(), media_type='text/event-stream')
    return StreamingResponse(
        agent.astream_answer(query=request.message),
        media_type='text/event-stream'
    )
