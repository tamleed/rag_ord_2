# Где лежат последние изменения

Если нужно быстро понять, **что именно поменялось в последнем апдейте**, начните с последнего коммита:

```bash
git log --oneline -n 3
git show --stat --summary HEAD
git show --name-only --format=fuller HEAD
```

На момент подготовки этого файла последний проблемный апдейт был таким:

- commit: `8713c54`
- title: `Improve v2 retrieval/answering logic, add raw scoring & multi-answer handling; update deployment checklist and tests`

## Где обновленный код и JSON

- Обновлённый **код приложения** лежит в `app.py`.
- Обновлённые **тесты под эту логику** лежат в `tests/test_app_logic.py`.
- Обновлённый **JSON с FAQ-данными** лежит в `qa_db_merged.json`.
- Инструкции по деплою лежат в `TRANSFER_TO_OTHER_PC_AND_SERVER.md`.

Быстрые команды, чтобы открыть именно их:

```bash
sed -n '1,260p' app.py
sed -n '1,220p' tests/test_app_logic.py
sed -n '1,120p' qa_db_merged.json
sed -n '1,220p' TRANSFER_TO_OTHER_PC_AND_SERVER.md
```

## Как собрать проект из нового кода

Собирать лучше из корня репозитория. Минимальный локальный сценарий:

```bash
cd /workspace/rag_ord_2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export QDRANT_URL=http://localhost:6333
export QA_DB_PATH=./qa_db_merged.json
export TFIDF_VOCAB_PATH=./tfidf_vocab.json
python index_faq.py
uvicorn app:app --host 127.0.0.1 --port 8010
```

Если нужен smoke-check после запуска:

```bash
curl -sS http://127.0.0.1:8010/health
```

Если у вас уже есть `.venv` и env-файл, тогда короткая версия такая:

```bash
cd /workspace/rag_ord_2
source .venv/bin/activate
export QDRANT_URL=http://localhost:6333
export QA_DB_PATH=./qa_db_merged.json
export TFIDF_VOCAB_PATH=./tfidf_vocab.json
python index_faq.py
uvicorn app:app --host 127.0.0.1 --port 8010
```

## Команда, чтобы забрать код с git на сервер и перезапустить v2

Если вы уже на сервере, используйте такой блок:

```bash
cd /home/ubuntu/faq_rag_v2
git fetch --all --prune
git checkout codex/update-project-based-on-tester-feedback
git pull --ff-only origin codex/update-project-based-on-tester-feedback
source .venv/bin/activate
set -a
source ./faq_rag_v2.env
set +a
python index_faq.py
pkill -f "uvicorn app:app --host 127.0.0.1 --port 8010" || true
nohup uvicorn app:app --host 127.0.0.1 --port 8010 > /home/ubuntu/faq_rag_v2/uvicorn_v2.log 2>&1 &
sleep 3
curl -sS http://127.0.0.1:8010/health
```

Если вы запускаете это со своей локальной машины одной командой через SSH, используйте такой вариант:

```bash
ssh ubuntu@<SERVER_IP> 'cd /home/ubuntu/faq_rag_v2 && git fetch --all --prune && git checkout codex/update-project-based-on-tester-feedback && git pull --ff-only origin codex/update-project-based-on-tester-feedback && source .venv/bin/activate && set -a && source ./faq_rag_v2.env && set +a && python index_faq.py && pkill -f "uvicorn app:app --host 127.0.0.1 --port 8010" || true && nohup uvicorn app:app --host 127.0.0.1 --port 8010 > /home/ubuntu/faq_rag_v2/uvicorn_v2.log 2>&1 & sleep 3 && curl -sS http://127.0.0.1:8010/health'
```

## Что делать сейчас

### Если вам нужно просто понять, где были последние правки

```bash
./show_latest_changes.sh
git diff HEAD~1 HEAD -- app.py tests/test_app_logic.py TRANSFER_TO_OTHER_PC_AND_SERVER.md qa_db_merged.json
```

### Если вы хотите проверить логику локально

```bash
pytest -q tests/test_app_logic.py
python -m py_compile app.py rag_core.py tests/test_app_logic.py
```

### Если ваша задача — выкатить обновлённую v2 на сервер

```bash
cd /home/ubuntu/faq_rag_v2
git checkout codex/update-project-based-on-tester-feedback
git pull --ff-only origin codex/update-project-based-on-tester-feedback
source .venv/bin/activate
set -a
source ./faq_rag_v2.env
set +a
python index_faq.py
nohup uvicorn app:app --host 127.0.0.1 --port 8010 > /home/ubuntu/faq_rag_v2/uvicorn_v2.log 2>&1 &
curl -sS http://127.0.0.1:8010/health
```

### Если нужно быстро понять, что именно изменилось по смыслу

- `app.py` — вся новая логика retrieval / no-answer / multi-answer.
- `tests/test_app_logic.py` — что именно ожидалось от этой логики.
- `qa_db_merged.json` — изменения в самих FAQ-данных.
- `TRANSFER_TO_OTHER_PC_AND_SERVER.md` — что делать при деплое.

## Какие файлы были изменены

### 1. `app.py`

Здесь лежит основная логика изменений по v2:

- новые пороги и константы для retriever / no-answer / multi-answer;
- сохранение `score_dense_raw` и `score_sparse_raw`;
- новая логика `_select_close_candidates`;
- подтверждение no-answer через `_confirm_no_answer_with_llm`;
- расширенные debug-поля в `answer_question_logic_v2`.

Чтобы посмотреть только изменения в логике приложения:

```bash
git diff HEAD~1 HEAD -- app.py
```

Или открыть только нужные места:

```bash
rg -n "LOW_RELEVANCE|MULTI_ANSWER|score_dense_raw|_select_close_candidates|_confirm_no_answer_with_llm|answer_question_logic_v2" app.py
```

### 2. `tests/test_app_logic.py`

Здесь лежат новые unit-тесты именно под последний апдейт:

- exact raw match;
- поведение no-answer;
- fallback при недоступной модели;
- multiple answers;
- debug candidate fields.

Быстрый просмотр:

```bash
git diff HEAD~1 HEAD -- tests/test_app_logic.py
pytest -q tests/test_app_logic.py
```

### 3. `TRANSFER_TO_OTHER_PC_AND_SERVER.md`

Здесь лежат изменения по деплою:

- какая ветка считалась актуальной;
- как клонировать новую копию на сервер;
- какие локальные файлы нельзя тащить в git;
- обязательный шаг `python index_faq.py` после обновления данных/кода.

Быстрый просмотр:

```bash
git diff HEAD~1 HEAD -- TRANSFER_TO_OTHER_PC_AND_SERVER.md
```

### 4. `qa_db_merged.json`

Здесь менялись данные базы знаний: часть коротких question variants была деактивирована через `"is_active": false`.

Быстрый просмотр:

```bash
git diff HEAD~1 HEAD -- qa_db_merged.json
```

## Как быстро найти всё сразу

Самый короткий сценарий:

```bash
git show --stat --summary HEAD
git diff HEAD~1 HEAD -- app.py tests/test_app_logic.py TRANSFER_TO_OTHER_PC_AND_SERVER.md qa_db_merged.json
```

## Если нужно посмотреть именно предыдущий спорный апдейт

```bash
git show --stat --summary 8713c54
git show 8713c54 -- app.py
git show 8713c54 -- tests/test_app_logic.py
git show 8713c54 -- TRANSFER_TO_OTHER_PC_AND_SERVER.md
git show 8713c54 -- qa_db_merged.json
```

## Локальный helper-скрипт

В репозитории есть `show_latest_changes.sh` — он печатает последний коммит, список изменённых файлов и краткую статистику diff:

```bash
./show_latest_changes.sh
```
