FROM python:3.9-bookworm@sha256:ff12c273af1e1814efcd8dfdefe16a70a4d901afe94dce721ffed8a34176f285

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/alfworld

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake curl git && \
    rm -rf /var/lib/apt/lists/*

COPY third_party/TextWorld /opt/textworld
COPY third_party/ALFWorld /opt/alfworld

RUN python -m pip install --upgrade pip && \
    sed -i 's/from textworld.envs.zmachine.jericho import JerichoEnv/JerichoEnv = None/' \
      /opt/textworld/textworld/envs/__init__.py && \
    sed -i 's/\r$//' /opt/textworld/setup.sh && \
    python -m pip install \
      "numpy<2" \
      "gym==0.15.4" \
      "tqdm" \
      "cffi>=1" \
      "networkx>=2" \
      "PyYAML>=6" \
      "more-itertools" \
      "tatsu>=4.3,<5" \
      "hashids>=1.2" \
      "mementos>=1.3" \
      "prompt-toolkit" \
      "fast-downward @ https://github.com/MarcCote/downward/archive/faster_replan.zip" \
      "fastapi<1" \
      "uvicorn<1" && \
    python -m pip install --no-deps /opt/textworld

RUN sed -i '/alfred_thor_env/d; /alfred_hybrid/d' \
      /opt/alfworld/alfworld/agents/environment/__init__.py

COPY services/alfworld_server.py /opt/trace2tower/alfworld_server.py

WORKDIR /opt/trace2tower

CMD ["uvicorn", "alfworld_server:app", "--host", "0.0.0.0", "--port", "8000"]
