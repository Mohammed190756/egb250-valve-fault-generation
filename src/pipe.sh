while pgrep -f revs2.py >/dev/null; do sleep 10; done
python3 feat.py > feat.log 2>&1
python3 events.py > events.log 2>&1
python3 stats_all.py > stats_all.log 2>&1
echo DONE >> stats_all.log
