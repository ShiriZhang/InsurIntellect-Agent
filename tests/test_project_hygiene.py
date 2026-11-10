"""Portfolio hygiene checks for the demo product."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_docker_compose_has_readable_container_name_and_port():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    service = compose["services"]["app"]
    env = service["environment"]

    assert compose["name"] == "insurintellect-agent"
    assert service["container_name"] == "insurintellect-agent"
    assert "8766:8766" in service["ports"]
    assert service.get("healthcheck")
    assert service.get("restart") == "unless-stopped"
    assert "OPENAI_EMBEDDING_MODEL" not in env
    assert "local:hash" not in str(env)
    assert env["LOCAL_EMBEDDING_CACHE_DIR"] == "/app/data/models"


def test_dockerfile_uses_demo_port_and_healthcheck():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "requirements-docker.txt" in dockerfile
    assert "PORT=8766" in dockerfile
    assert "EXPOSE 8766" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "/api/v1/health/live" in dockerfile


def test_main_path_requirements_stay_light():
    """Main-path requirements should not pull in heavy OCR/exotic DB deps.

    The `requirements-advanced.txt` split was retired with the deletion of
    document_parser_service.py and friends; assert the heavy deps ended up
    *nowhere* in the active install graph.
    """
    main = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    dev = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    docker = (ROOT / "requirements-docker.txt").read_text(encoding="utf-8")

    assert "PyMuPDF" in main
    # No advanced-OCR-only file anymore
    assert not (ROOT / "requirements-advanced.txt").exists()
    for heavy in ("unstructured", "pytesseract", "pdfplumber", "PyPDF2", "pandas", "asyncpg"):
        assert heavy not in main
        assert heavy not in dev
        assert heavy not in docker


def test_docker_requirements_avoid_torch_download_path():
    docker = (ROOT / "requirements-docker.txt").read_text(encoding="utf-8")
    package_lines = [
        line.strip().lower()
        for line in docker.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert "chromadb" in docker
    assert not any(line.startswith("sentence-transformers") for line in package_lines)
    assert not any(line.startswith("langchain-huggingface") for line in package_lines)
    assert not any(line.startswith("torch") for line in package_lines)


def test_pyproject_has_project_metadata_and_no_setup_py():
    """setup.py was removed; pyproject.toml now carries full [project] metadata."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "[tool.ruff]" in text
    assert "[tool.pytest.ini_options]" in text
    assert "[project]" in text
    assert 'name = "insurintellect-agent"' in text
    assert 'requires-python = ">=3.11"' in text
    # setup.py must not exist
    assert not (ROOT / "setup.py").exists()


def test_async_database_and_full_model_dependencies_are_declared():
    for filename in ("requirements.txt", "requirements-docker.txt"):
        assert "sqlalchemy[asyncio]" in (ROOT / filename).read_text(encoding="utf-8")
    models = (ROOT / "requirements-models.txt").read_text(encoding="utf-8")
    for package in ("torch", "sentence-transformers", "langchain-huggingface", "huggingface-hub"):
        assert package in models
    for filename in ("run_demo.sh", "run_demo.bat"):
        launcher = (ROOT / "scripts" / filename).read_text(encoding="utf-8")
        assert "-m scripts.launch" in launcher
        assert "OPENAI_EMBEDDING_MODEL=" not in launcher
        assert "SIMILARITY_THRESHOLD=" not in launcher
