#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_root"

archive_name="${1:-rag_ord_2_release_bundle.tar.gz}"
archive_path="$repo_root/artifacts/$archive_name"

mkdir -p "$repo_root/artifacts"
rm -f "$archive_path"

tar \
  --exclude='./.git' \
  --exclude='./artifacts' \
  --exclude='./.venv' \
  --exclude='./.pytest_cache' \
  --exclude='./__pycache__' \
  --exclude='./tests/__pycache__' \
  --exclude='./*.pyc' \
  --exclude='./faq_rag.env' \
  --exclude='./env.sh' \
  --exclude='./*.env' \
  -czf "$archive_path" \
  .

printf 'Created archive: %s\n' "$archive_path"
printf '\nArchive contents preview:\n'
tar -tzf "$archive_path" | sed -n '1,80p'
