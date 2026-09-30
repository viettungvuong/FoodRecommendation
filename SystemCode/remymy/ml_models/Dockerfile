FROM python:3.12-slim

WORKDIR /opt/remymy-ml

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    INFERENCE_DIR=/opt/remymy-ml/inference \
    ARTIFACT_DIR=/opt/remymy-ml/weights/gru_xattn_reranker \
    NLTK_DATA=/usr/share/nltk_data \
    HOST=0.0.0.0 \
    PORT=8003 \
    CATALOG_PATH=/catalog/price_mapped_nutrients.csv \
    INDEX_DIR=/var/cache/remymy/retrieval-index

COPY SystemCode/remymy/ml_models/requirements-serve.txt .
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && pip install --no-cache-dir -r requirements-serve.txt \
 && apt-get purge -y build-essential \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/* \
 && python -m nltk.downloader -d /usr/share/nltk_data wordnet omw-1.4 stopwords

COPY DataMappingRecommendation/model_inference_reranker_avgemb.py DataMappingRecommendation/model_inference_score_recommend.py ./inference/
COPY DataMappingRecommendation/model_artifacts/gru_xattn_reranker/model.pt DataMappingRecommendation/model_artifacts/gru_xattn_reranker/config.json ./weights/gru_xattn_reranker/
COPY SystemCode/remymy/ml_models/serve.py .

EXPOSE 8003

HEALTHCHECK --interval=10s --timeout=3s --start-period=120s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT', '8003') + '/health', timeout=2)" || exit 1

CMD ["python", "serve.py"]
