# Transfer checklist: copy to server, run v2 in parallel, configure nginx

Ниже — безопасный сценарий, при котором **v1 остаётся как есть**, а v2 поднимается отдельно.

## 0) Verify the branch with v2 changes on your PC

```bash
git fetch --all --prune
git checkout codex-update
git pull --ff-only
git status -sb
git log --oneline -n 5
```

Ожидаемо: ветка `codex-update` должна смотреть на `origin/codex/update-project-based-on-tester-feedback`.

---

## 1) Create archive from the exact branch (Windows PowerShell)

Run from repo root on your PC:

```powershell
# optional cleanup of Python cache
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# create zip one level above the repo
Compress-Archive -Path .\* -DestinationPath ..\rag_ord_2_v2.zip -Force
```

---

## 2) Copy archive to server

```powershell
scp ..\rag_ord_2_v2.zip ubuntu@<SERVER_IP>:/home/ubuntu/
```

---

## 3) Deploy into a separate folder on server

```bash
ssh ubuntu@<SERVER_IP>
cd /home/ubuntu
mkdir -p rag_ord_2_v2
unzip -o rag_ord_2_v2.zip -d rag_ord_2_v2
cd rag_ord_2_v2
```

> Важно: не перезаписывайте старую папку v1 проекта.

---

## 4) Create separate env file for v2

Create `/home/ubuntu/rag_ord_2_v2/faq_rag_v2.env`:

```dotenv
# Server / model
API_KEY=...
BASE_URL=...
LLM_MODEL=...
EMBED_MODEL=...

# Qdrant
QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION=faq_collection
QDRANT_EXACT_SEARCH=true

# Legacy paths (optional fallback)
QA_DB_PATH=/home/ubuntu/rag_ord_2_v2/qa_db.json
TFIDF_VOCAB_PATH=/home/ubuntu/rag_ord_2_v2/tfidf_vocab.json

# V2 dedicated paths (important)
QA_DB_PATH_V2=/home/ubuntu/rag_ord_2_v2/qa_db_merged.json
TFIDF_VOCAB_PATH_V2=/home/ubuntu/rag_ord_2_v2/tfidf_vocab.json
```

---

## 5) Run v2 service on another port (example: 8010)

```bash
cd /home/ubuntu/rag_ord_2_v2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
set -a
source ./faq_rag_v2.env
set +a

# run in parallel to old service
nohup uvicorn app:app --host 127.0.0.1 --port 8010 > /home/ubuntu/rag_ord_2_v2/uvicorn_v2.log 2>&1 &
```

Quick check:

```bash
curl -sS http://127.0.0.1:8010/docs >/dev/null && echo "v2 up"
tail -n 50 /home/ubuntu/rag_ord_2_v2/uvicorn_v2.log
```

---

## 6) nginx: route only v2 endpoints to v2 backend

Open your nginx site config (example: `/etc/nginx/sites-available/default`) and add locations:

```nginx
# New v2 endpoints -> v2 backend (port 8010)
location /answer_v2 {
    proxy_pass http://127.0.0.1:8010;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}

location /answer_full_v2 {
    proxy_pass http://127.0.0.1:8010;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}

# Keep existing routes for v1 as they are now:
# /answer and /answer_full -> old backend
```

Validate + reload:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

---

## 7) Smoke test through nginx

```bash
curl -sS -X POST http://<DOMAIN_OR_IP>/answer_v2 \
  -H 'Content-Type: application/json' \
  -d '{"question":"Как вернуть товар?"}'

curl -sS -X POST http://<DOMAIN_OR_IP>/answer_full_v2 \
  -H 'Content-Type: application/json' \
  -d '{"question":"Как вернуть товар?"}'
```

---

## 8) Rollback (instant)

If anything goes wrong with v2:

1. Comment out/remove nginx locations `/answer_v2` and `/answer_full_v2`.
2. `sudo nginx -t && sudo systemctl reload nginx`
3. Stop v2 process:

```bash
pkill -f "uvicorn app:app --host 127.0.0.1 --port 8010"
```

v1 continues working unchanged.
