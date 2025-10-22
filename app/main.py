# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .config import ALLOW_CORS_ALL
from .db import init_db, close_db
from . import routes

app = FastAPI(title="Personal Knowledge Base")

if ALLOW_CORS_ALL:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(routes.router)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
def index():
    return FileResponse("static/index.html")

@app.on_event("startup")
def _startup():
    init_db()

@app.on_event("shutdown")
def _shutdown():
    close_db()