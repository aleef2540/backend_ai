from fastapi.middleware.cors import CORSMiddleware

origins = [
    "https://www.enmarks.com",
    "https://www.entraining.net",
    "https://entraining.net",
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

def setup_cors(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )