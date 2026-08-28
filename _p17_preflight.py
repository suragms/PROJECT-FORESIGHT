"""Phase 17 pre-flight: verify frozen models and create directory structure."""
import hashlib, os, json

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Verify frozen models
with open("docs/final_model_registry.json") as f:
    registry = json.load(f)
all_pass = True
for e in registry:
    mf = e["model_file"].replace("\\", os.sep)
    if os.path.exists(mf):
        h = hashlib.sha256(open(mf, "rb").read()).hexdigest()
        ok = h == e["hash"]
        if not ok:
            all_pass = False
        s = "PASS" if ok else "FAIL"
        print(f"{s}: {e['model_id']}")
    else:
        all_pass = False
        print(f"FAIL: {e['model_id']} NOT FOUND")
print(f"\nAll frozen models intact: {all_pass}")

# Create Phase 17 directories
dirs = [
    "data/phase17/external", "data/phase17/raw", "data/phase17/processed",
    "data/phase17/features", "data/phase17/forecasts", "data/phase17/backtests",
    "data/phase17/risk", "models/phase17/uci", "models/phase17/synthetic",
]
for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"Created: {d}")
