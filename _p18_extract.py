import json

with open("docs/phase18_gate_results.json") as f:
    r = json.load(f)

for src, fs in r["fold_stability"].items():
    print(f"\n{src} FOLDS:")
    for fd in fs["fold_details"]:
        print(f"  fold={fd['fold']} origin={fd['origin']} baseline={fd['baseline_wape_pct']}% "
              f"cand={fd['candidate_wape_pct']}% bias={fd['candidate_bias']} impr={fd['improvement_pp']}pp "
              f"beats={fd['candidate_beats_baseline']}")

for src, sa in r["sku_analysis"].items():
    print(f"\n{src} SKU: total={sa['total_skus']} median_wape={sa['median_sku_wape']}% "
          f"p75={sa['p75_sku_wape']}% p90={sa['p90_sku_wape']}% high_err={sa['high_error_skus']}")

for src, h in r["horizon"].items():
    if isinstance(h, dict) and h.get("by_horizon"):
        print(f"\n{src} HORIZON ({h['horizon_status']}, degradation={h['degradation_pp']} pp):")
        for hr in h["by_horizon"]:
            print(f"  h={hr['horizon_step']} base={hr['baseline_wape']}% "
                  f"cand={hr['candidate_wape']}% bias={hr['candidate_bias']}")

for src, fi in r["feat_importance"].items():
    print(f"\n{src} TOP FEATURES:")
    for feat in fi.get("top_features", [])[:8]:
        print(f"  {feat['feature']}: {feat['importance']:.4f} leakage={feat['leakage_status']}")

print("\nSAMPLE CRITICAL SKUs:")
for s in r["risk"].get("sample_critical_skus", []):
    print(f"  {s}")

print("\nBIAS:")
for src, b in r["bias"].items():
    print(f"  {src}: bias={b.get('overall_bias')} direction={b.get('direction')} "
          f"relative={b.get('relative_bias')} severity={b.get('severity')}")

print("\nDECISIONS:")
for src, d in r["decisions"].items():
    print(f"  {src}: {d['decision']} | issues={d['issues']} | warnings={d['warnings']}")
