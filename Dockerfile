# SPDX-License-Identifier: MIT
ARG SOURCE_DATE_EPOCH=1790035200
FROM ghcr.io/fujitsupolycom/sparkring@sha256:2375f876bc9ea065e85ae10cebad7a8db8a2ec0e6862b4441c269c5bf56365c6
ARG SOURCE_DATE_EPOCH
USER root
RUN --mount=type=bind,source=.,target=/recipe,readonly \
    SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} /opt/venv/bin/python /recipe/apply_patches.py && \
    SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} /opt/venv/bin/python /recipe/image/prefill-lifetime/apply_patches.py && \
    SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} /opt/venv/bin/python /recipe/image/dsa-ckv/apply_patches.py && \
    SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} /opt/venv/bin/python /recipe/image/dsa-ckv-mixed/apply_patches.py && \
    /opt/venv/bin/python /recipe/image/dsa-ckv-mixed/test_opt_in.py --source-path /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/b12x_mla_sparse.py
LABEL org.opencontainers.image.title="Altar-1 on four DGX Sparks" \
      io.altar.experimental.dsa-ckv="1" \
      io.altar.experimental.dsa-ckv-mixed="1" \
      org.opencontainers.image.description="Pinned SparkRing image with Altar-1 serving patches"
