
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

router = APIRouter()

TEMPLATES_DIR = os.getenv("GUI_TEMPLATES_DIR", "gui/templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@router.get("/ui", response_class=HTMLResponse)
@router.get("/ui/", response_class=HTMLResponse)
def ui_home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@router.get("/ui/generate", response_class=HTMLResponse)
def ui_generate(request: Request):
    return templates.TemplateResponse("generate.html", {"request": request})

@router.get("/ui/batch", response_class=HTMLResponse)
def ui_batch(request: Request):
    return templates.TemplateResponse("batch.html", {"request": request})

@router.get("/ui/verify", response_class=HTMLResponse)
def ui_verify(request: Request):
    return templates.TemplateResponse("verify.html", {"request": request})

@router.get("/ui/datasets", response_class=HTMLResponse)
def ui_datasets(request: Request):
    return templates.TemplateResponse("datasets.html", {"request": request})

@router.get("/ui/jobs", response_class=HTMLResponse)
def ui_jobs(request: Request):
    return templates.TemplateResponse("jobs.html", {"request": request})

@router.get("/ui/ml", response_class=HTMLResponse)
def ui_ml(request: Request):
    return templates.TemplateResponse("ml.html", {"request": request})

@router.get("/ui/monitor", response_class=HTMLResponse)
def ui_monitor(request: Request):
    return templates.TemplateResponse("monitor.html", {"request": request})
