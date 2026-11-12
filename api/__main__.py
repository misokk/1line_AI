import os

import uvicorn


def main():
    # The platform-provided PORT wins; local development uses loopback by default.
    uvicorn.run(
        "api.app:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000"))
    )


if __name__ == "__main__":
    main()
