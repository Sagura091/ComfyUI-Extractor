FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /srv
RUN pip install --no-cache-dir transformers==4.40.2 accelerate==0.27.1 \
    pillow fastapi uvicorn python-multipart

# Server code
COPY server.py .

ENV MODEL=blip2-flan-t5-xl
EXPOSE 5000
CMD ["python", "server.py"]
