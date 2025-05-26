FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime


WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get clean && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
COPY model_weights.pt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    bash install_torch.sh

COPY . .

CMD ["python", "autoscaler.py"]