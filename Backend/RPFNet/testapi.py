# test_api.py
from api import analyze, clean
print("Start")
report = analyze('uci', 73)

print(report.summary())

report = clean('uci', 73)

print(report)
# print("Rows:", report["n_rows"])
# print("Flagged:", report["n_flagged"])
# print("Percent:", report["pct_flagged"])
# print("Mode:", report["mode"])
