# Run Guide: Insurance Policy LoRA on SageMaker AI

About 1 hour end to end, on one `ml.g5.2xlarge`.

## 1. Before you start

1. **Region:** use one region throughout, for example **us-east-1** or **us-west-2**.
2. **GPU quota:** in **Service Quotas** → **AWS services** → **Amazon SageMaker**, find **Studio JupyterLab Apps running on ml.g5.2xlarge instance**. If the applied value is 0, choose **Request increase at account level** and request **1**.
3. **Budget alert (optional):** **Billing and Cost Management** → **Budgets** → **Create budget** → **Monthly cost budget**.

Cost: one `ml.g5.2xlarge` for about an hour. See [SageMaker AI pricing](https://aws.amazon.com/sagemaker/ai/pricing/) for the current rate. The space bills only while it is running.

## 2. Set up SageMaker Studio

1. **Amazon SageMaker AI** console → **Admin configurations** → **Domains** → **Create domain** → **Set up for single user (Quick setup)**. Wait for **InService**.
2. **Studio** → **Open Studio**.
3. **JupyterLab** → **Create JupyterLab space**, named `insurance-lora`.
4. On the space page set:
   - **Instance:** `ml.g5.2xlarge`
   - **Storage (GB):** `50` (the 5 GB default is too small for the model download)
   - **Image:** the default SageMaker Distribution
5. **Run space** → **Open JupyterLab**.

## 3. Run the notebook

1. Upload `insurance_lora_sagemaker.ipynb` with the **Upload files** button in the JupyterLab file browser.
2. Open it, choose the **Python 3 (ipykernel)** kernel, and run all cells in order (**Run** → **Run All Cells**).
3. If the install cell asks for a restart, choose **Kernel** → **Restart Kernel** and continue from the next cell.

| Notebook section | What happens | Approx. time |
| --- | --- | --- |
| 1. Install + GPU check | Installs libraries, prints the GPU | 2 min |
| 3. Dataset | Generates 800 training and 150 test policies | Seconds |
| 4. Load model | Downloads `Qwen/Qwen3-4B-Instruct-2507` (about 8 GB) | 3–5 min |
| 6. Base model eval | Base model answers the 150 test policies | 5–10 min |
| 7. LoRA training | 2 epochs of SFT with LoRA | 15–25 min |
| 8. Fine-tuned eval | The same 150 policies again | 5–10 min |
| 9. Consistency | 30 policies × 3 sampled runs, both models | 10 min |
| 10–11. Results + save | Table, charts, `results.json`, adapter | Seconds |

## 4. Collect results and stop billing

1. Download `outputs/results.json`, `outputs/base_vs_finetuned.png` and `outputs/loss_curve.png`.
2. **Stop** the `insurance-lora` space under **JupyterLab** in Studio.
3. To remove everything afterwards, delete the space and then the domain.

## 5. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `NO GPU FOUND` in section 1 | The space isn't on a GPU instance. Stop it, set `ml.g5.2xlarge`, run again. |
| `ResourceLimitExceeded` when starting the space | GPU quota not yet approved (step 1.2). |
| `No space left on device` while loading the model | Stop the space, set **Storage (GB)** to 50, run again. |
| Model download fails | Set `MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"` in section 2 and rerun from section 4. |
| `CUDA out of memory` during training | Set `per_device_train_batch_size=2` and `gradient_accumulation_steps=8`, then rerun from section 4. |

The notebook uses only the space's own GPU. It creates no training jobs or endpoints.
