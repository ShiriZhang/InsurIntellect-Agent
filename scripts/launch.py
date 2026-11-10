"""Shared launcher; no shell evaluation of .env or configuration overrides."""
from app.core.config import settings


def main():
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT,
                reload=settings.RELOAD and settings.DEBUG)


if __name__ == "__main__":
    main()
