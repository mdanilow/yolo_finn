import os
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str)
args = parser.parse_args()

results = []
models = os.listdir(args.model_dir)
models.sort()
for model in models:
    cocoeval_txt = Path(args.model_dir) / model / "cocoeval.txt"
    if os.path.exists(cocoeval_txt):
        print(model)
        with open(cocoeval_txt, "r") as f:
            results.append((model, float(f.read())))

with open("collected_eval_results.csv", "w") as f:
    out = ""
    for res in results:
        out += "{},{}\n".format(res[0], res[1])
    f.write(out)

