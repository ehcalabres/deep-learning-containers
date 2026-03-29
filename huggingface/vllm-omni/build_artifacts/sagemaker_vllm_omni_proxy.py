import json
import requests

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

app = FastAPI()
VLLM_BASE = "http://127.0.0.1:8091"

ALLOWED_ROUTES = {
    "/v1/audio/speech",
    "/v1/audio/voices",
    "/v1/images/generations",
    "/v1/images/edits",
    "/v1/videos",
    "/v1/chat/completions",
    "/v1/completions",
}

TASK_TO_ROUTE = {
    "text-to-speech": "/v1/audio/speech",
    "text-to-image": "/v1/images/generations",
    "image-to-image": "/v1/images/edits",
    "text-to-video": "/v1/videos",
    "text-generation": "/v1/chat/completions",
    "completion": "/v1/completions",
}

# Route-specific method overrides controlled by CustomAttributes method=...
# Only routes listed here are allowed to change method behavior.
ROUTE_METHOD_OVERRIDES = {
    "/v1/audio/voices": {"GET", "POST"},
}


def parse_custom_attributes(value: str | None) -> dict[str, str]:
    if not value:
        return {}

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}

    if not isinstance(parsed, dict):
        return {}

    return {str(k).strip(): str(v).strip() for k, v in parsed.items()}


def resolve_route(custom_attrs: dict[str, str]) -> str | None:
    route = custom_attrs.get("route")
    if route:
        return route

    task = custom_attrs.get("task")
    if task:
        return TASK_TO_ROUTE.get(task)

    return None


def resolve_upstream_method(route: str, custom_attrs: dict[str, str]) -> str | None:
    allowed_methods = ROUTE_METHOD_OVERRIDES.get(route)
    requested_method = custom_attrs.get("method", "").upper()

    if allowed_methods is None:
        # Default behavior for all non-overridden routes.
        return "POST"

    if requested_method:
        if requested_method in allowed_methods:
            return requested_method
        return None

    return "POST"  # Default method if not specified


def build_routing_error(
    custom_attrs: dict[str, str],
    allowed_routes: set[str],
    task_to_route: dict[str, str],
) -> str:
    route = custom_attrs.get("route")
    task = custom_attrs.get("task")

    base_msg = "Invalid or missing routing metadata."

    if route:
        return (
            f"{base_msg} You provided route='{route}', which is not allowed.\n"
            f"Allowed routes: {sorted(allowed_routes)}"
        )

    if task:
        return (
            f"{base_msg} You provided task='{task}', which is not supported.\n"
            f"Supported tasks: {sorted(task_to_route.keys())}"
        )

    return (
        f"{base_msg} You must provide routing via CustomAttributes.\n"
        f"Examples:\n"
        f'  {{"route": "/v1/audio/speech"}}\n'
        f'  {{"task": "text-to-speech"}}\n\n'
        f"Allowed routes: {sorted(allowed_routes)}\n"
        f"Supported tasks: {sorted(task_to_route.keys())}"
    )


@app.get("/ping")
def ping():
    try:
        r = requests.get(f"{VLLM_BASE}/ping", timeout=2)
        if r.status_code == 200:
            return {"status": "ok"}
    except requests.RequestException:
        pass

    return JSONResponse(
        content={"status": "vllm not ready"},
        status_code=503,
    )


@app.post("/invocations")
async def invocations(request: Request):
    accept = request.headers.get("accept", "application/json")

    custom_attrs = parse_custom_attributes(
        request.headers.get("x-amzn-sagemaker-custom-attributes")
    )

    route = resolve_route(custom_attrs)

    if not route or route not in ALLOWED_ROUTES:
        return JSONResponse(
            content={
                "error": build_routing_error(
                    custom_attrs, ALLOWED_ROUTES, TASK_TO_ROUTE
                )
            },
            status_code=400,
        )

    method = resolve_upstream_method(route, custom_attrs)
    if not method:
        allowed_methods = ",".join(sorted(ROUTE_METHOD_OVERRIDES.get(route, ["POST"])))
        return JSONResponse(
            content={
                "error": f"Method not allowed for route {route}. "
                f'Use CustomAttributes JSON like {{"route":"{route}","method":"{allowed_methods}"}}. '
                f"Default is POST."
            },
            status_code=400,
        )

    upstream_url = f"{VLLM_BASE}{route}"

    try:
        if method == "GET":
            upstream_resp = requests.get(
                upstream_url,
                headers={"Accept": accept},
                timeout=300,
            )
        elif method == "POST":
            try:
                payload = await request.json()
            except Exception:
                return JSONResponse(
                    content={"error": "Request body must be valid JSON"},
                    status_code=400,
                )

            upstream_resp = requests.post(
                upstream_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": accept,
                },
                timeout=300,
            )
        else:
            return JSONResponse(
                content={"error": f"Unsupported method: {method}"},
                status_code=400,
            )
    except requests.RequestException as e:
        return JSONResponse(
            content={"error": f"Upstream request failed: {e}"},
            status_code=502,
        )

    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        media_type=upstream_resp.headers.get("content-type", "application/json"),
    )
