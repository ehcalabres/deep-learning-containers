"""SageMaker launcher for hf-serve.

Adds the SageMaker HTTP contract (/ping and /invocations) while preserving
hf-serve's native routes such as /health, /predict, and OpenAI-compatible APIs.
"""

import os
import re

from starlette.types import ASGIApp, Receive, Scope, Send

from hf_serve.cli import parser as hf_serve_parser
from hf_serve.server import app, launch


def _parse_route(headers: list[tuple[bytes, bytes]]) -> str | None:
    for key, value in headers:
        if key.lower() == b"x-amzn-sagemaker-custom-attributes":
            match = re.search(r"route=(/[^\s,]+)", value.decode())
            return match.group(1) if match else None
    return None


class SageMakerRouteMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            path = scope["path"]
            if path == "/ping":
                scope = dict(scope)
                scope["path"] = "/health"
                scope["raw_path"] = b"/health"
            elif path == "/invocations":
                route = _parse_route(scope.get("headers", [])) or "/predict"
                scope = dict(scope)
                scope["path"] = route
                scope["raw_path"] = route.encode()

        await self.app(scope, receive, send)


def _default_model_args() -> None:
    if os.getenv("MODEL_DIR") or os.getenv("MODEL_ID"):
        return

    if os.path.isdir("/opt/ml/model") and os.listdir("/opt/ml/model"):
        os.environ["MODEL_DIR"] = "/opt/ml/model"
    elif os.getenv("HF_MODEL_ID"):
        os.environ["MODEL_ID"] = os.environ["HF_MODEL_ID"]


def main() -> None:
    _default_model_args()
    app.add_middleware(SageMakerRouteMiddleware)
    args = hf_serve_parser.parse_args()

    launch(
        host=args.host,
        port=args.port,
        model_id=args.model_id,
        model_dir=args.model_dir,
        revision=args.revision,
        task=args.task,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        accepted_mimetypes=args.accepted_mimetypes.split(",") if args.accepted_mimetypes else None,
        max_file_size=args.max_file_size,
        cloud=args.cloud,
    )


if __name__ == "__main__":
    main()
