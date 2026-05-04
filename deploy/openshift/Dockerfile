# UBI9 Python 3.12 — matches the convention used by uhura and tiberius-openshift.
# Restricted-SCC compatible (non-root UID 1001, no privileged caps).
FROM registry.access.redhat.com/ubi9/python-312:latest

USER 0
RUN dnf install -y --setopt=install_weak_deps=False \
        gcc gcc-c++ make \
    && dnf clean all && rm -rf /var/cache/dnf

USER 1001
WORKDIR /opt/app-root/src

# Install the package + its production extras
COPY --chown=1001:0 pyproject.toml README.md ./
COPY --chown=1001:0 src/ src/
COPY --chown=1001:0 eval/ eval/
COPY --chown=1001:0 configs/ configs/
COPY --chown=1001:0 scripts/ scripts/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[llm,scotty,kafka]"

# KServe expects the model server to listen on 8080 by default
ENV PORT=8080
EXPOSE 8080

# Default entrypoint: KServe-compatible model server (see deploy/kserve_predictor.py)
CMD ["python", "-m", "scripts.serve_kserve"]
