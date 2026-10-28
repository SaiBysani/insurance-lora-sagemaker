"""Run insurance_lora_sagemaker.ipynb unchanged as a SageMaker AI training job.

Builds train.py from the notebook's own code cells (skipping only the %pip and
%%writefile magics, which become requirements.txt and insurance_data.py), uploads
it to S3 and starts one ml.g5.2xlarge training job.

    python build_and_launch.py build            # build the source dir only
    python build_and_launch.py launch           # build, upload, start the job
"""
import io
import json
import os
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = next(d for d in [os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]   # folder holding the notebook
            if os.path.exists(os.path.join(d, "insurance_lora_sagemaker.ipynb")))
SRC = os.path.join(HERE, "src")

REGION = os.environ.get("AWS_REGION", "us-east-2")
ROLE_NAME = os.environ.get("SAGEMAKER_ROLE_NAME", "SageMakerInsuranceLoRARole")   # needs AmazonSageMakerFullAccess
BUCKET_PREFIX = "sagemaker-insurance-lora"   # bucket = <prefix>-<account id>, created beforehand in REGION
IMAGE = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker"
INSTANCE = sys.argv[2] if len(sys.argv) > 2 else "ml.g5.2xlarge"
OUT = "/opt/ml/checkpoints/outputs"       # synced to S3 while the job runs, so partial results survive a failure

PRELUDE = f'''# Generated from insurance_lora_sagemaker.ipynb by build_and_launch.py. Do not edit by hand.
import matplotlib
matplotlib.use("Agg")
import pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
def display(x):
    print(x.to_string() if hasattr(x, "to_string") else x, flush=True)
import time as _time
_JOB_T0 = _time.time()
'''

EPILOGUE = f'''
# ---- job metadata (not in the notebook) ----
import importlib.metadata as _md, json as _json
with open(f"{{OUT_DIR}}/results.json") as _f:
    _r = _json.load(_f)
_r["run_environment"] = {{
    "where": "Amazon SageMaker AI training job", "instance": "{INSTANCE}", "region": "{REGION}",
    "container": "{IMAGE.split('/')[-1]}",
    "versions": {{p: _md.version(p) for p in ["torch", "transformers", "peft", "accelerate"]}},
    "job_wall_minutes": round((_time.time() - _JOB_T0) / 60, 1),
}}
_r["base_eval_seconds"] = base_summary.get("seconds"); _r["finetuned_eval_seconds"] = ft_summary.get("seconds")
with open(f"{{OUT_DIR}}/results.json", "w") as _f:
    _json.dump(_r, _f, indent=2)
print("JOB DONE", _json.dumps(_r["run_environment"]), flush=True)
'''


def build():
    nb = json.load(open(os.path.join(ROOT, "insurance_lora_sagemaker.ipynb"), encoding="utf-8"))
    parts = [PRELUDE]
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if s.startswith("%pip") or s.startswith("%%writefile"):
            continue
        s = s.replace('OUT_DIR    = "outputs"', f'OUT_DIR    = "{OUT}"')
        parts.append(f"\n# ======== notebook cell {i} ========\nprint('>>> cell {i}', flush=True)\n{s}\n")
    parts.append(EPILOGUE)
    os.makedirs(SRC, exist_ok=True)
    code = "".join(parts)
    compile(code, "train.py", "exec")
    open(os.path.join(SRC, "train.py"), "w", encoding="utf-8").write(code)
    open(os.path.join(SRC, "insurance_data.py"), "w", encoding="utf-8").write(
        open(os.path.join(ROOT, "insurance_data.py"), encoding="utf-8").read())
    # pinned so the run is reproducible; satisfies the notebook's transformers>=4.56 / peft>=0.15 / accelerate>=1.5
    open(os.path.join(SRC, "requirements.txt"), "w").write("transformers==4.57.1\npeft==0.17.1\naccelerate==1.10.1\n")
    print("built", SRC)


def launch():
    import boto3
    account = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
    role_arn, bucket = f"arn:aws:iam::{account}:role/{ROLE_NAME}", f"{BUCKET_PREFIX}-{account}"
    build()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for fn in ["train.py", "insurance_data.py", "requirements.txt"]:
            t.add(os.path.join(SRC, fn), arcname=fn)
    name = f"insurance-lora-{INSTANCE.split('.')[1]}-" + time.strftime("%Y%m%d-%H%M%S")
    key = f"{name}/source/sourcedir.tar.gz"
    boto3.client("s3", region_name=REGION).put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    sm = boto3.client("sagemaker", region_name=REGION)
    sm.create_training_job(
        TrainingJobName=name,
        AlgorithmSpecification={"TrainingImage": IMAGE, "TrainingInputMode": "File"},
        RoleArn=role_arn,
        HyperParameters={"sagemaker_program": json.dumps("train.py"),
                         "sagemaker_submit_directory": json.dumps(f"s3://{bucket}/{key}"),
                         "sagemaker_region": json.dumps(REGION)},
        OutputDataConfig={"S3OutputPath": f"s3://{bucket}/{name}/output"},
        CheckpointConfig={"S3Uri": f"s3://{bucket}/{name}/checkpoints", "LocalPath": "/opt/ml/checkpoints"},
        ResourceConfig={"InstanceType": INSTANCE, "InstanceCount": 1, "VolumeSizeInGB": 60},
        StoppingCondition={"MaxRuntimeInSeconds": 3 * 3600},
        Environment={"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"},
        Tags=[{"Key": "project", "Value": "insurance-lora-newsletter"}],
    )
    print("started", name)


if __name__ == "__main__":
    {"build": build, "launch": launch}[sys.argv[1] if len(sys.argv) > 1 else "build"]()
