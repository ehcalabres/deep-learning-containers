"""SageMaker endpoint integration tests for Hugging Face PyTorch inference DLC."""

import json
import logging
import os
from pprint import pformat

import boto3
import pytest
from test_utils import clean_string
from test_utils import random_suffix_name
from test_utils.constants import INFERENCE_AMI_VERSION
from test_utils.constants import SAGEMAKER_ROLE
from test_utils.huggingface_helper import get_hf_token

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

ASR_SAMPLE_URL = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"
STARTUP_HEALTH_CHECK_TIMEOUT = 3600
ENDPOINT_WAIT_TIMEOUT = 4200
INSTANCE_TYPE = "ml.g6.xlarge"


@pytest.fixture(scope="function")
def model_config(request):
    return request.param


def _get_hf_token(aws_session):
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    return get_hf_token(aws_session)


def _build_production_variant(model_name):
    return {
        "VariantName": "AllTraffic",
        "ModelName": model_name,
        "InitialInstanceCount": 1,
        "InstanceType": INSTANCE_TYPE,
        "InferenceAmiVersion": INFERENCE_AMI_VERSION,
        "ContainerStartupHealthCheckTimeoutInSeconds": STARTUP_HEALTH_CHECK_TIMEOUT,
    }


def _delete_endpoint(sagemaker_client, endpoint_name):
    if not endpoint_name:
        return
    try:
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        sagemaker_client.get_waiter("endpoint_deleted").wait(
            EndpointName=endpoint_name,
            WaiterConfig={"Delay": 30, "MaxAttempts": 20},
        )
    except Exception as e:
        LOGGER.warning(f"Cleanup endpoint failed: {e}")


def _delete_endpoint_config(sagemaker_client, endpoint_config_name):
    if not endpoint_config_name:
        return
    try:
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    except Exception as e:
        LOGGER.warning(f"Cleanup endpoint config failed: {e}")


def _delete_model(sagemaker_client, model_name):
    if not model_name:
        return
    try:
        sagemaker_client.delete_model(ModelName=model_name)
    except Exception as e:
        LOGGER.warning(f"Cleanup model failed: {e}")


@pytest.fixture(scope="function")
def model_endpoint(aws_session, image_uri, model_config):
    model_id = model_config["model_id"]
    cleaned_id = clean_string(model_id.split("/")[-1], "_./")
    model_name = random_suffix_name(f"hfpt-{cleaned_id}", 50)
    endpoint_name = random_suffix_name(f"hfpt-{cleaned_id}", 50)

    LOGGER.info(f"Using image: {image_uri}")
    LOGGER.info(f"Model ID: {model_id}")

    sagemaker_client = aws_session.sagemaker

    hf_token = _get_hf_token(aws_session)
    role_arn = aws_session.resolve_role_arn(SAGEMAKER_ROLE)
    env = {
        "MODEL_ID": model_id,
        "TASK": model_config["task"],
        "HF_TOKEN": hf_token,
    }
    env.update(model_config.get("env", {}))

    model_created = endpoint_config_created = endpoint_created = False
    try:
        LOGGER.info(f"Creating model: {model_name}")
        sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": image_uri,
                "Environment": env,
            },
            ExecutionRoleArn=role_arn,
        )
        model_created = True

        LOGGER.info(f"Creating endpoint config: {endpoint_name}")
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_name,
            ProductionVariants=[_build_production_variant(model_name)],
        )
        endpoint_config_created = True

        LOGGER.info(f"Deploying endpoint: {endpoint_name}")
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_name,
        )
        endpoint_created = True
        sagemaker_client.get_waiter("endpoint_in_service").wait(
            EndpointName=endpoint_name,
            WaiterConfig={"Delay": 30, "MaxAttempts": ENDPOINT_WAIT_TIMEOUT // 30},
        )
        LOGGER.info("Endpoint deployment completed successfully")

        yield {"name": endpoint_name, "config": model_config}
    finally:
        if endpoint_created:
            _delete_endpoint(sagemaker_client, endpoint_name)
        if endpoint_config_created:
            _delete_endpoint_config(sagemaker_client, endpoint_name)
        if model_created:
            _delete_model(sagemaker_client, model_name)


@pytest.mark.parametrize(
    "model_config",
    [
        {
            "model_id": "LiquidAI/LFM2.5-230M",
            "task": "text-generation",
            "env": {"DTYPE": "bfloat16"},
        },
    ],
    indirect=True,
)
def test_text_generation_endpoint(model_endpoint):
    endpoint_name = model_endpoint["name"]
    runtime = boto3.client("sagemaker-runtime")

    payload = {
        "model": "LiquidAI/LFM2.5-230M",
        "messages": [
            {"role": "system", "content": "You answer with a short factual sentence."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "max_tokens": 32,
        "temperature": 0,
    }
    LOGGER.info(f"Sending chat-completion payload: {payload}")

    result = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps(payload),
        ContentType="application/json",
        Accept="application/json",
        CustomAttributes="route=/v1/chat/completions",
    )
    body = json.loads(result["Body"].read())
    LOGGER.info(f"Model response: {pformat(body)}")

    content = body["choices"][0]["message"]["content"]
    assert content.strip(), "Generated chat message is empty"


@pytest.mark.parametrize(
    "model_config",
    [
        {
            "model_id": "nvidia/parakeet-tdt-0.6b-v3",
            "task": "automatic-speech-recognition",
            "env": {"DTYPE": "bfloat16"},
        },
    ],
    indirect=True,
)
def test_asr_endpoint(model_endpoint):
    endpoint_name = model_endpoint["name"]
    runtime = boto3.client("sagemaker-runtime")

    payload = {"inputs": ASR_SAMPLE_URL}
    LOGGER.info(f"Sending ASR payload: {payload}")

    result = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps(payload),
        ContentType="application/json",
        Accept="application/json",
        CustomAttributes="route=/predict-json",
    )
    body = json.loads(result["Body"].read())
    LOGGER.info(f"Model response: {pformat(body)}")

    assert body.get("text", "").strip(), "Transcription text is empty"
