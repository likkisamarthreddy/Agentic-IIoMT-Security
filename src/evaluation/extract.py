import json
with open('checkpoints/training_results.json') as f:
    report = json.load(f)['test_results']['per_class_report']
for k,v in report.items():
    if isinstance(v, dict) and 'precision' in v:
        print(f"{k:<15} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1-score']:.4f} | {int(v['support'])}")
