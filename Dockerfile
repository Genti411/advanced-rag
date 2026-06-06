FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 HF_HOME=/opt/hf SENTENCE_TRANSFORMERS_HOME=/opt/hf
WORKDIR /srv
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Bake the bi-encoder (retrieval) and cross-encoder (rerank) so it runs offline.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" \
 && chmod -R a+rX /opt/hf
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COPY . .
CMD ["python", "run_eval.py"]
