from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.routes.v1.chat import router as chat_router


app = FastAPI(
    title="Story Teller"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static file
app.mount("/logs", StaticFiles(directory="./logs"), name="log_dir")

# Include the router with /api/v1 as the prefix
app.include_router(chat_router, prefix="/api/v1")




