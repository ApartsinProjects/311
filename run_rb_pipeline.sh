#!/bin/bash
PY=/c/Python314/python
log(){ echo "[$(date +%H:%M:%S)] $*" >> results/rb_pipeline.log; }
rm -f results/rb_pipeline.log
for i in $(seq 1 160); do
  # --- 311 ---
  if [ ! -f results/pilot_rulebook.json ]; then
    if [ ! -f results/rb_rules.json ]; then
      out=$($PY pilot_rulebook.py induce 2>&1 | grep -v FutureWarning | grep -v pynvml)
      echo "$out" | grep -q "SOURCE RULEBOOK" && log "311 induced + round2 submitted"
    else
      out=$($PY pilot_rulebook.py report 2>&1 | grep -v FutureWarning | grep -v pynvml)
      echo "$out" | grep -q "invariants" && log "311 REPORTED"
    fi
  fi
  # --- cfpb ---
  if [ ! -f results/cfpb_rulebook.json ]; then
    if [ ! -f results/cfpb_rb_rules.json ]; then
      out=$($PY cfpb_rulebook.py induce 2>&1 | grep -v FutureWarning | grep -v pynvml)
      echo "$out" | grep -q "SOURCE RULEBOOK" && log "cfpb induced + round2 submitted"
    else
      out=$($PY cfpb_rulebook.py report 2>&1 | grep -v FutureWarning | grep -v pynvml)
      echo "$out" | grep -q "invariants" && log "cfpb REPORTED"
    fi
  fi
  if [ -f results/pilot_rulebook.json ] && [ -f results/cfpb_rulebook.json ]; then log "BOTH DONE"; break; fi
  sleep 120
done
