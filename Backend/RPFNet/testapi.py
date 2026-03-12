# test_api.py
from api import analyze
print("Start")
report = analyze('uci', 73)

print("Rows:", report["n_rows"])
print("Flagged:", report["n_flagged"])
print("Percent:", report["pct_flagged"])
print("Mode:", report["mode"])