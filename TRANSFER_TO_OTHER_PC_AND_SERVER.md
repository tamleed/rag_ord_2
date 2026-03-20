# Transfer checklist: deploy v2 in parallel to v1 (one server)

Ниже — сценарий для **одного сервера**:
- v1 в `/home/ubuntu/faq_rag_v1`
- v2 в `/home/ubuntu/faq_rag_v2`
- Qdrant в Docker, как на первом проекте (`qdrant/qdrant:latest`, порт `127.0.0.1:6333`)

---

## 0) Verify branch on PC

> Подтвержденная default branch на сервере: `My_ish_files_from_server`.
> Перед деплоем всё равно проверьте `git branch -a` и `git remote show origin`.

```bash
git fetch --all --prune
git branch -a
git remote show origin
git status -sb
git log --oneline -n 5
```

Если `origin/HEAD` указывает на `My_ish_files_from_server`, используйте:

```bash
git checkout -B My_ish_files_from_server origin/My_ish_files_from_server
git pull --ff-only
```

---

## 0.1) Recommended: clone latest version into a new directory on server

Если вы хотите поднять **новую директорию** рядом с существующей установкой, удобнее забирать код напрямую из git:

```bash
ssh ubuntu@<SERVER_IP>
cd /home/ubuntu
git clone --branch My_ish_files_from_server <GIT_REMOTE_URL> faq_rag_v2_new
cd faq_rag_v2_new
git branch -a
git remote show origin
git status -sb
git log --oneline -n 5
```

Если `faq_rag_v2_new` уже существует:

```bash
cd /home/ubuntu/faq_rag_v2_new
git fetch --all --prune
git branch -a
git remote show origin
git status -sb
git log --oneline -n 5
git checkout -B My_ish_files_from_server origin/My_ish_files_from_server
git pull --ff-only
```

> Если git-remote недоступен, используйте сценарий ниже через zip-архив.

---

## 0.2) Local files that must not go to git

На сервере не коммитьте и не переносите обратно в git:

- `.venv/`
- `faq_rag_v2.env`
- любые локальные `*.env` с ключами
- runtime-логи, например `uvicorn_v2.log`

---

## 1) Archive project on PC (Windows PowerShell)

```powershell
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Compress-Archive -Path .\* -DestinationPath ..\faq_rag_v2.zip -Force
```

---

## 2) Copy archive to server

```powershell
scp ..\faq_rag_v2.zip ubuntu@<SERVER_IP>:/home/ubuntu/
```

---

## 3) Deploy to `/home/ubuntu/faq_rag_v2`

```bash
ssh ubuntu@<SERVER_IP>
cd /home/ubuntu
mkdir -p faq_rag_v2
unzip -o faq_rag_v2.zip -d faq_rag_v2
cd faq_rag_v2
```

> v1 каталог `/home/ubuntu/faq_rag_v1` не трогаем.

---

## 4) Start Qdrant in Docker (same style as v1 server)

Проверка docker:

```bash
sudo docker --version
```

Запуск Qdrant (имя контейнера и проброс порта как в рабочем примере):

```bash
sudo docker rm -f qdrant || true
sudo docker run -d \
  --name qdrant \
  -p 127.0.0.1:6333:6333 \
  qdrant/qdrant:latest
```

Проверка:

```bash
sudo docker ps --format 'table {{.ID}}\t{{.Image}}\t{{.Names}}\t{{.Ports}}'
curl -sS http://127.0.0.1:6333/
curl -sS http://127.0.0.1:6333/collections
```

Ожидаемо в `docker ps`:

```text
qdrant/qdrant:latest   qdrant   127.0.0.1:6333->6333/tcp, 6334/tcp
```

---

## 5) Create env file for v2

Create `/home/ubuntu/faq_rag_v2/faq_rag_v2.env`:

```dotenv
QDRANT_URL=http://localhost:6333
QDRANT_EXACT_SEARCH=1

# legacy resources in v2 directory
QA_DB_PATH=/home/ubuntu/faq_rag_v2/qa_db_merged.json
TFIDF_VOCAB_PATH=/home/ubuntu/faq_rag_v2/tfidf_vocab.json

# dedicated v2 resources
QA_DB_PATH_V2=/home/ubuntu/faq_rag_v2/qa_db_merged.json
TFIDF_VOCAB_PATH_V2=/home/ubuntu/faq_rag_v2/tfidf_vocab.json

LOCAL_LLM_URL=https://localgpu2.myapidev.ru/v1/chat/qwen
LOCAL_LLM_API_KEY=PUT_REAL_KEY_HERE
LLM_MAX_TOKENS_V2=1200
RAG_API_KEY=PUT_REAL_KEY_HERE
```

---

## 6) Rebuild search index after updating JSON / code

После обновления проекта обязательно пересоберите TF-IDF и переиндексируйте Qdrant,
иначе приложение перечитает новый `qa_db_merged.json`, но поиск может остаться на старых данных.

```bash
cd /home/ubuntu/faq_rag_v2
source .venv/bin/activate
set -a
source ./faq_rag_v2.env
set +a
python index_faq.py
```

---

## 7) Run v2 API on port 8010

```bash
cd /home/ubuntu/faq_rag_v2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
set -a
source ./faq_rag_v2.env
set +a
nohup uvicorn app:app --host 127.0.0.1 --port 8010 > /home/ubuntu/faq_rag_v2/uvicorn_v2.log 2>&1 &
```

Check:

```bash
curl -sS http://127.0.0.1:8010/health
tail -n 80 /home/ubuntu/faq_rag_v2/uvicorn_v2.log
```

---

## 8) nginx routing for v2

Используйте **активный** site-файл nginx.

Если v1 уже в `/etc/nginx/sites-available/response-v1.myapidev.ru`,
создайте отдельный файл для v2: `/etc/nginx/sites-available/response-v2.myapidev.ru`.

Добавьте v2 locations:

```nginx
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
```

Enable and reload:

```bash
sudo ln -sf /etc/nginx/sites-available/response-v2.myapidev.ru /etc/nginx/sites-enabled/response-v2.myapidev.ru
sudo nginx -t
sudo systemctl reload nginx
```

---

## 9) Smoke test

```bash
curl -sS -X POST http://127.0.0.1:8010/answer_full_v2 \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: PUT_REAL_KEY_HERE' \
  -d '{"question":"смайлики?","debug":true}'
```

and through domain:

```bash
curl -sS -X POST https://response-v2.myapidev.ru/answer_full_v2 \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: PUT_REAL_KEY_HERE' \
  -d '{"question":"смайлики?","debug":true}'
```

---

## 10) Rollback v2 only

```bash
pkill -f "uvicorn app:app --host 127.0.0.1 --port 8010" || true
sudo docker stop qdrant || true
sudo nginx -t && sudo systemctl reload nginx
```

v1 остаётся без изменений.
