# ใช้ Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# ติดตั้ง system dependencies (สำหรับ OpenCV)
RUN apt-get update && apt-get install -y \
        libgl1 \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*

# คัดลอก requirements.txt
COPY requirements.txt .

# ติดตั้ง torch/torchvision
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# ติดตั้ง dependencies อื่น ๆ
# before installing requirements
RUN python -m pip install -U pip setuptools wheel

# cache pip downloads between builds + longer timeout/retries
RUN --mount=type=cache,target=/root/.cache/pip \
        pip install --prefer-binary --timeout 600 --retries 10 -r requirements.txt


# คัดลอกโค้ดที่เหลือ
COPY . .

# เปิด port FastAPI
EXPOSE 8000

# Run FastAPI (production mode)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
