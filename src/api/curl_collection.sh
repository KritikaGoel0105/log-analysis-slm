#!/usr/bin/env bash
# ==================================================================
# Week 9 — D7 curl collection (document Section 8: "Postman/curl
# collection"; a Postman-importable JSON is provided alongside at
# src/api/postman_collection.json).
#
# Prerequisite: the API is running offline from the repository root:
#     uvicorn src.api.api:app --host 127.0.0.1 --port 8000
#
# Usage:
#     bash src/api/curl_collection.sh
# ==================================================================
set -u
BASE="${API_BASE:-http://127.0.0.1:8000}"

echo "== 1. GET /health (Section 5.6 health check) =="
curl -s "$BASE/health"
echo; echo

echo "== 2. POST /analyze — default parameters (top_k_context=3, max_tokens=512) =="
echo "   (generation is slow: ~1-2 min on GPU, several minutes on CPU)"
curl -s -X POST "$BASE/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "logs": "2026-07-18 09:14:02 kernel: EXT4-fs error (device sda1): ext4_find_entry:1455: inode #131077: comm smartd: reading directory lblock 0\n2026-07-18 09:14:02 smartd[1024]: Device: /dev/sda [SAT], 8 Currently unreadable (pending) sectors\n2026-07-18 09:14:05 kernel: blk_update_request: critical medium error, dev sda, sector 52428800"
  }'
echo; echo

echo "== 3. POST /analyze — custom top_k_context and max_tokens =="
curl -s -X POST "$BASE/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "logs": "2026-07-18 10:00:01 systemd: Started Session 42 of user admin.\n2026-07-18 10:00:03 sshd[2211]: Accepted publickey for admin from 10.0.0.5 port 51122",
    "top_k_context": 5,
    "max_tokens": 256
  }'
echo; echo

echo "== 4. POST /analyze — request validation: missing required field 'logs' (expect 422) =="
curl -s -o /dev/null -w "HTTP %{http_code}\n" -X POST "$BASE/analyze" \
  -H "Content-Type: application/json" \
  -d '{"top_k_context": 3}'
echo

echo "== 5. POST /analyze — error handling: empty logs (expect 400) =="
curl -s -w "\nHTTP %{http_code}\n" -X POST "$BASE/analyze" \
  -H "Content-Type: application/json" \
  -d '{"logs": "   \n  \n"}'
echo

echo "== 6. GET /docs — auto-generated endpoint documentation (expect 200) =="
curl -s -o /dev/null -w "HTTP %{http_code}\n" "$BASE/docs"
echo
echo "Collection complete."
