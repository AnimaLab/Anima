#!/bin/bash
set -e

# All args are passed to both runners (--anima-url, --llm-base-url, etc.)

echo "=== Running LoCoMo benchmark ==="
cd /bench/locomo
python run.py "$@" --context-window 2

echo ""
echo "=== Running LongMemEval benchmark ==="
cd /bench/longmemeval
python run.py "$@" --reflect

echo ""
echo "=== All benchmarks complete ==="
