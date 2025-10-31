import os
from pathlib import Path

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), 'static')
templates = Jinja2Templates(directory='static')

load_dotenv()
FRONTEND_PORT = os.getenv('FRONTEND_PORT', '8001')
BACKEND_HOST = os.getenv('BACKEND_HOST', 'localhost')
BACKEND_PORT = os.getenv('BACKEND_PORT', '8000')
BACKEND_URL = f'http://{BACKEND_HOST}:{BACKEND_PORT}'


@app.get('/favicon.ico')
async def favicon():
    return FileResponse(Path('static/image/favicon.png'))


@app.post('/api/generate')
async def proxy_generate(request: Request):
    body = await request.body()
    headers = {'Content-Type': 'application/json'}

    async def stream_from_backend():
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                'POST',
                f'{BACKEND_URL}/api/generate',
                content=body,
                headers=headers,
            ) as backend_resp:
                async for chunk in backend_resp.aiter_bytes():
                    yield chunk

    return StreamingResponse(
        stream_from_backend(),
        media_type='text/event-stream',
    )


@app.get('/')
async def main_page(request: Request):
    return templates.TemplateResponse(
        name='index.html',
        context={'request': request}
    )


if __name__ == '__main__':
    uvicorn.run(
        'main:app',
        port=int(FRONTEND_PORT) if FRONTEND_PORT.isdigit() else 8001)
