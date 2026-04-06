# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def proxy_module():
    repo_root = Path(__file__).resolve().parents[7]
    proxy_path = (
        repo_root
        / "huggingface"
        / "vllm-omni"
        / "build_artifacts"
        / "sagemaker_vllm_omni_proxy.py"
    )

    spec = importlib.util.spec_from_file_location(
        "sagemaker_vllm_omni_proxy", proxy_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def test_client(proxy_module):
    return TestClient(proxy_module.app)


@pytest.fixture
def stub_upstream(monkeypatch, proxy_module):
    calls = []

    def _fake_response(payload):
        return SimpleNamespace(
            content=payload,
            status_code=200,
            headers={"content-type": "application/json"},
        )

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(
            {
                "method": "POST",
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return _fake_response(b'{"ok": true}')

    def fake_get(url, headers=None, timeout=None):
        calls.append(
            {
                "method": "GET",
                "url": url,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return _fake_response(b'{"data": []}')

    monkeypatch.setattr(proxy_module.requests, "post", fake_post)
    monkeypatch.setattr(proxy_module.requests, "get", fake_get)
    return calls


@pytest.mark.parametrize(
    "task,payload,expected_route",
    [
        (
            "text-to-speech",
            {
                "input": "Hello, how are you?",
                "voice": "vivian",
                "language": "English",
            },
            "/v1/audio/speech",
        ),
        (
            "text-to-image",
            {"prompt": "A cat sitting on a mat."},
            "/v1/images/generations",
        ),
        (
            "image-to-image",
            {
                "prompt": "Turn this into a watercolor painting.",
                "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=",
            },
            "/v1/images/edits",
        ),
        (
            "text-generation",
            {"messages": [{"role": "user", "content": "What is deep learning?"}]},
            "/v1/chat/completions",
        ),
    ],
)
def test_invocations_routes_omni_tasks_via_custom_attributes(
    test_client,
    stub_upstream,
    proxy_module,
    task,
    payload,
    expected_route,
):
    response = test_client.post(
        "/invocations",
        json=payload,
        headers={
            "x-amzn-sagemaker-custom-attributes": json.dumps({"task": task}),
        },
    )

    assert response.status_code == 200
    assert len(stub_upstream) == 1
    assert stub_upstream[0]["method"] == "POST"
    assert stub_upstream[0]["url"] == f"{proxy_module.VLLM_BASE}{expected_route}"


def test_voices_route_accepts_get_and_post_via_method_custom_attribute(
    test_client,
    stub_upstream,
    proxy_module,
):
    response_get = test_client.post(
        "/invocations",
        json={},
        headers={
            "x-amzn-sagemaker-custom-attributes": json.dumps(
                {"route": "/v1/audio/voices", "method": "GET"}
            )
        },
    )

    assert response_get.status_code == 200
    assert stub_upstream[0]["method"] == "GET"
    assert stub_upstream[0]["url"] == f"{proxy_module.VLLM_BASE}/v1/audio/voices"

    response_post = test_client.post(
        "/invocations",
        json={"voice": "vivian"},
        headers={
            "x-amzn-sagemaker-custom-attributes": json.dumps(
                {"route": "/v1/audio/voices", "method": "POST"}
            )
        },
    )

    assert response_post.status_code == 200
    assert stub_upstream[1]["method"] == "POST"
    assert stub_upstream[1]["url"] == f"{proxy_module.VLLM_BASE}/v1/audio/voices"
