from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), 'static')
templates = Jinja2Templates(directory='static')


@app.get('/favicon.ico')
async def favicon():
    return FileResponse(Path('static/image/favicon.png'))


@app.get('/')
async def main_page(request: Request):
    return templates.TemplateResponse(
        name='index.html',
        context={'request': request}
    )
