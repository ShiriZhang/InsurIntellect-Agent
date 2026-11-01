"""Explicit model preparation/preflight; no documents or questions are accepted."""

import argparse
import asyncio
import json

from app.services.embedding_service import EmbeddingService, LocalEmbeddingUnavailable
from app.services.llm_service import LLMService
from app.services.embedding_profile import EmbeddingInputTooLong, EmbeddingCompatibilityError


async def check(*, download=False, generation=False):
    def local_check():
        try:
            return EmbeddingService(allow_download=download).preflight()
        except LocalEmbeddingUnavailable as exc:
            return {"status": "unavailable", "code": "local_model_unavailable", "message": str(exc)}
        except (EmbeddingInputTooLong, EmbeddingCompatibilityError) as exc:
            return {"status": "unavailable", "code": "embedding_profile_or_budget_invalid", "message": str(exc)}

    local = await asyncio.to_thread(local_check)
    remote = {"status": "not_checked"}
    if generation:
        service = LLMService()
        try:
            remote = await service.preflight()
        finally:
            if service.client is not None:
                await service.client.close()
    return {"local_embedding": local, "generation": remote,
            "model_services_ready": local["status"] == "ready" and remote["status"] == "available"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Allow model/tokenizer resource downloads")
    parser.add_argument("--check-generation", action="store_true", help="Make a minimal generation API request (may cost tokens)")
    args = parser.parse_args()
    result = asyncio.run(check(download=args.download, generation=args.check_generation))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    success = result["model_services_ready"] if args.check_generation else result["local_embedding"]["status"] == "ready"
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
