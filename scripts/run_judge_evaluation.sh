#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if [[ ! -f .env ]]; then
  echo "Error: $project_root/.env is missing." >&2
  exit 1
fi

set -a
source .env
set +a

exec uv run llm-judge-evaluation "$@"
