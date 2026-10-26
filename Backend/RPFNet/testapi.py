# test_api.py
from api import analyze, clean
print("Start")
report = analyze('uci', 73)

print(report.summary())

report = clean('uci', 73)

print(report)
