.PHONY: install data train eda test serve docker
install:  ; pip install -r requirements.txt
data:     ; PYTHONPATH=src python -m credit_risk.pipeline generate
ingest:   ; PYTHONPATH=src python -m credit_risk.pipeline generate --refresh-macro
train:    ; PYTHONPATH=src python -m credit_risk.pipeline train
all:      ; PYTHONPATH=src python -m credit_risk.pipeline all
eda:      ; python scripts/eda.py
test:     ; PYTHONPATH=src:. pytest -q
serve:    ; PYTHONPATH=src uvicorn app.main:app --reload
docker:   ; docker build -t nigeria-credit-risk . && docker run -p 8000:8000 nigeria-credit-risk
