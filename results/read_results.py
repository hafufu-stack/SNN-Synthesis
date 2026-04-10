import json, os
files = ['phase33_gsm8k_exit.json','phase34_sigma_prediction.json',
         'phase37_cross_task_transfer.json','phase38_multi_model_nbs.json']
for f in files:
    path = os.path.join('results', f)
    d = json.load(open(path))
    exp = d.get('experiment', f)
    print(f"=== {exp} ===")
    for k, v in d.items():
        if k not in ['experiment', 'timestamp']:
            s = json.dumps(v, ensure_ascii=False)
            print(f"  {k}: {s[:250]}")
    print()
