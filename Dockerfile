FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    pkg-config \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk2.0-dev \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

RUN echo '#!/bin/bash\npython -c "import asyncio, platform; asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy()) if platform.system() == \"Linux\" else None; exec(open(\"start_app.py\").read())"' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh


RUN echo 'import streamlit.web.cli as stcli\nimport sys\nsys.argv = ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]\nstcli.main()' > /app/start_app.py

CMD ["/app/entrypoint.sh"]
