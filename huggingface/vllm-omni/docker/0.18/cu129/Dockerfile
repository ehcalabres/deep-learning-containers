ARG FINAL_BASE_IMAGE=763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:omni-sagemaker-cuda-v1
FROM ${FINAL_BASE_IMAGE} AS vllm-base

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"

ARG TRANSFORMERS_VERSION=4.57.6
ARG HUGGINGFACE_HUB_VERSION=0.36.2
ARG HF_XET_VERSION=1.4.2

RUN uv pip install --no-cache-dir \
    transformers==${TRANSFORMERS_VERSION} \
    huggingface-hub==${HUGGINGFACE_HUB_VERSION} \
    hf-xet==${HF_XET_VERSION} \
    grpcio

FROM vllm-base AS sagemaker

ENV HF_HUB_USER_AGENT_ORIGIN="aws:sagemaker:gpu-cuda:inference:hf-vllm-omni"

RUN HOME_DIR=/root \
    && uv pip install --system --upgrade pip requests PTable \
    && curl -o ${HOME_DIR}/oss_compliance.zip https://aws-dlinfra-utilities.s3.amazonaws.com/oss_compliance.zip \
    && unzip ${HOME_DIR}/oss_compliance.zip -d ${HOME_DIR}/ \
    && cp ${HOME_DIR}/oss_compliance/test/testOSSCompliance /usr/local/bin/testOSSCompliance \
    && chmod +x /usr/local/bin/testOSSCompliance \
    && chmod +x ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh \
    && ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh ${HOME_DIR} python3 \
    && rm -rf ${HOME_DIR}/oss_compliance*

RUN uv pip install --no-cache-dir prettytable

ENTRYPOINT ["/usr/local/bin/sagemaker_entrypoint.sh"]
