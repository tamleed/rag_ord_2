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
