#!/usr/bin/env bash
set -euo pipefail

commit_ref="${1:-HEAD}"

printf '=== Commit ===\n'
git log --oneline -n 1 "$commit_ref"

printf '\n=== Summary ===\n'
git show --stat --summary --format=fuller "$commit_ref"

printf '\n=== Changed files ===\n'
git show --name-only --format='' "$commit_ref" | sed '/^$/d'

printf '\n=== What to do now ===\n'
printf '%s\n' \
  '1) Run ./show_latest_changes.sh to see the latest commit and changed files' \
  '2) Run git diff HEAD~1 HEAD -- app.py tests/test_app_logic.py TRANSFER_TO_OTHER_PC_AND_SERVER.md qa_db_merged.json' \
  '3) Run pytest -q tests/test_app_logic.py if you need to verify the v2 logic locally' \
  '4) Open TRANSFER_TO_OTHER_PC_AND_SERVER.md if your next step is deployment'

printf '\n=== Where is the updated code? ===\n'
printf '%s\n' \
  'app code: app.py' \
  'updated JSON: qa_db_merged.json' \
  'tests: tests/test_app_logic.py' \
  'deploy doc: TRANSFER_TO_OTHER_PC_AND_SERVER.md'

printf '\n=== How to build from the updated code ===\n'
printf '%s\n' \
  'python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt' \
  'export QDRANT_URL=http://localhost:6333 && export QA_DB_PATH=./qa_db_merged.json && export TFIDF_VOCAB_PATH=./tfidf_vocab.json' \
  'python index_faq.py && uvicorn app:app --host 127.0.0.1 --port 8010'

printf '\n=== Search hints ===\n'
printf '%s\n' \
  "app.py: rg -n 'LOW_RELEVANCE|MULTI_ANSWER|score_dense_raw|_select_close_candidates|_confirm_no_answer_with_llm|answer_question_logic_v2' app.py" \
  "tests/test_app_logic.py: pytest -q tests/test_app_logic.py" \
  "deploy doc: git diff HEAD~1 HEAD -- TRANSFER_TO_OTHER_PC_AND_SERVER.md" \
  "FAQ data: git diff HEAD~1 HEAD -- qa_db_merged.json"
