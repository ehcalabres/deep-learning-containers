ARG FINAL_BASE_IMAGE=763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:omni-sagemaker-cuda-v1
FROM ${FINAL_BASE_IMAGE} AS vllm-omni-base

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"

ARG HUGGINGFACE_HUB_VERSION=1.12.0

RUN apt-get update -y \
&& apt-get install -y --no-install-recommends curl unzip \
&& rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        huggingface-hub==${HUGGINGFACE_HUB_VERSION}

FROM vllm-omni-base AS sagemaker
ENV HF_HUB_ENABLE_HF_TRANSFER="1" \
    HF_HUB_USER_AGENT_ORIGIN="aws:sagemaker:gpu-cuda:inference:hf-vllm-omni"

# Copy CUDA compat and entrypoint scripts 
COPY start_cuda_compat.sh /usr/local/bin/start_cuda_compat.sh
COPY sagemaker_entrypoint.sh /usr/local/bin/sagemaker_entrypoint.sh

RUN chmod +x /usr/local/bin/start_cuda_compat.sh \
 && chmod +x /usr/local/bin/sagemaker_entrypoint.sh

RUN HOME_DIR=/root \
 && uv pip install --system --upgrade pip requests PTable \
 && curl -o ${HOME_DIR}/oss_compliance.zip https://aws-dlinfra-utilities.s3.amazonaws.com/oss_compliance.zip \
 && unzip ${HOME_DIR}/oss_compliance.zip -d ${HOME_DIR}/ \
 && cp ${HOME_DIR}/oss_compliance/test/testOSSCompliance /usr/local/bin/testOSSCompliance \
 && chmod +x /usr/local/bin/testOSSCompliance \
 && chmod +x ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh \
 && ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh ${HOME_DIR} python3 \
 && rm -rf ${HOME_DIR}/oss_compliance*

ENTRYPOINT ["/usr/local/bin/sagemaker_entrypoint.sh"]
