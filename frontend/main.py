import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import httpx

app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), 'static')
templates = Jinja2Templates(directory='static')

load_dotenv()
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
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            backend_resp = await client.post(
                f'{BACKEND_URL}/api/generate',
                content=body,
                headers=headers,
            )
            return StreamingResponse(
                backend_resp.aiter_bytes(),
                status_code=backend_resp.status_code,
                media_type='text/event-stream',
                headers=dict(backend_resp.headers),
            )
        except httpx.RequestError as e:
            raise RuntimeError(f'Backend unreachable: {e}')


@app.get('/')
async def main_page(request: Request):
    return templates.TemplateResponse(
        name='index.html',
        context={'request': request}
    )
