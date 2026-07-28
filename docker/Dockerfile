# =====================================================================
# docker/Dockerfile — Week 11 (D9): Offline Docker build
#
# Document basis:
#   Section 9 repository tree: "docker/ └── Dockerfile  # Offline
#     Docker build".
#   Section 4 Week 11: "Offline deployment packaging: Docker image,
#     setup scripts, no-internet validation test".
#   Section 8 D9: "Docker image OR install archive".
#
# IMPORTANT — platform prerequisite (engineering note, stated
# honestly): this project's offline_packages/ currently contains
# Windows (cp311-win_amd64) wheels, because the development target
# machine is Windows. A Linux container cannot install Windows
# wheels. To build this image, first prepare Linux wheels ONCE while
# online:
#
#   pip download -r requirements.txt -d offline_packages_linux \
#       --platform manylinux2014_x86_64 --python-version 311 \
#       --only-binary=:all:
#
# then build with NO network (the build itself is fully offline):
#
#   docker build --network=none -t log-analysis-slm -f docker/Dockerfile .
#
# The document's D9 acceptance criterion is "Docker image OR install
# archive" — the tested deliverable on this project's Windows target
# is the install archive (package_offline.sh); this Dockerfile is the
# Section 9-mandated offline build definition for Linux hosts.
# =====================================================================

FROM python:3.11-slim

# Offline enforcement — identical env contract to api.py/dashboard.py.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PIP_NO_INDEX=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Dependencies from LOCAL wheels only — no index, no network.
COPY requirements.txt ./
COPY offline_packages_linux/ ./offline_packages_linux/
RUN pip install --no-index \
        --find-links ./offline_packages_linux \
        -r requirements.txt \
    && rm -rf ./offline_packages_linux

# 2. Frozen Weeks 1-10 application code and offline assets.
COPY src/ ./src/
COPY .streamlit/ ./.streamlit/
COPY data/faiss.index data/faiss_incidents.json ./data/
COPY data/dataset/ ./data/dataset/
COPY models/ ./models/

# 3. Default service: the Week 9 REST API (Section 5.6).
#    The Week 10 dashboard can be started instead with:
#      docker run ... log-analysis-slm \
#        streamlit run src/dashboard/dashboard.py --server.port 8501
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.api.api:app", \
     "--host", "0.0.0.0", "--port", "8000"]
