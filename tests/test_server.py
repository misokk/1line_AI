"""Real loopback HTTP test; child process explicitly disables all provider calls."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from src.models import MultipleChoiceAnswer


def test_running_uvicorn_server(tmp_path, context, pdf_data, stored):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = dict(
        os.environ,
        HOST="127.0.0.1",
        PORT=str(port),
        AI_LIVE_ENABLED="false",
        AI_ALLOW_UNLISTED_MODELS="true",
        AI_SERVER_API_KEY="loopback-test-key",
        UPSTAGE_API_KEY="",
        AI_ENV="development",
        AI_RECORD_IO="true",
        AI_RECORDING_ROOT=str(tmp_path),
    )
    log_path = tmp_path / "server.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "api"],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=2
            ) as client:
                for _ in range(100):
                    if process.poll() is not None:
                        raise AssertionError(
                            "Server exited during startup: " + log_path.read_text()
                        )
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.RequestError:
                        time.sleep(0.1)
                else:
                    raise AssertionError("Server startup timeout")
                assert client.get("/docs").status_code == 200
                assert client.post("/internal/materials/analyze", json={}).status_code == 401
                headers = {"X-AI-API-Key": "loopback-test-key"}
                pdf = client.post(
                    "/internal/materials/extract",
                    content=pdf_data,
                    headers={**headers, "Content-Type": "application/pdf"},
                )
                assert pdf.status_code == 503
                txt = client.post(
                    "/internal/materials/extract",
                    content=b"English source text",
                    headers={**headers, "Content-Type": "text/plain"},
                )
                assert txt.status_code == 200
                assert client.post("/study-plan", json={}).status_code == 404
                result = client.post(
                    "/internal/questions/grade",
                    headers=headers,
                    json={
                        "context": context.model_dump(mode="json"),
                        "stored_question": stored.model_dump(mode="json"),
                        "answer": MultipleChoiceAnswer(
                            type="multiple_choice", selected_option_ids=["A"]
                        ).model_dump(),
                    },
                )
                assert result.status_code == 200 and result.json()["is_correct"]
                blocked = client.post(
                    "/internal/evaluations/generate",
                    headers={**headers, "X-AI-Key": "fake-request-key"},
                    json={
                        "context": context.model_dump(mode="json"),
                        "concept_version_id": str(context.concepts[0].version.id),
                        "model_id": "gemini:fixture-model",
                        "explanation_language": "en",
                        "history": [
                            {"role": "user", "content": "The simple present describes habits."}
                        ],
                    },
                )
                assert blocked.status_code == 503
                assert blocked.json()["error"]["code"] == "CONFIGURATION_ERROR"
                for _ in range(50):
                    outputs = list((tmp_path / "output").glob("questions_grade_output.json"))
                    if outputs:
                        break
                    time.sleep(0.02)
                assert len(outputs) == 1
                assert json.loads(outputs[0].read_text(encoding="utf-8"))["is_correct"]
                assert (tmp_path / "input/questions_grade_input.json").is_file()
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
