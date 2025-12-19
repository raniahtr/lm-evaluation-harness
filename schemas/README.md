# Structured-Output evaluation for medical benchmarks 

## How to Reproduce Experiments

### Step 0: Enable Schema Mode in Task Configuration

**Important:** Before running evaluations with schemas, you must ensure that `is_schema_mode_active` is set to `True` in the `doc_to_text` function of the task you're evaluating.

For example, in `lm_eval/tasks/medqa/generation/preprocess_medqa.py`, the function signature should be:
```python
def doc_to_text_medqa_neutral(doc, is_schema_mode_active: bool = True) -> str:
```

Or when calling the function, make sure to pass `is_schema_mode_active=True`. This ensures that the prompt formatting is appropriate for schema-constrained generation (without explicit format instructions that might conflict with the schema). 

### Step 1: Submit a Job to the RCP Cluster

```bash

# Submit an interactive job with GPU
runai submit \
  --name meditron-basic \
  --image registry.rcp.epfl.ch/multimeditron/basic:latest-GASPAR_USERNAME\
  --pvc light-scratch:/mloscratch \
  --large-shm \
  -e NAS_HOME=/mloscratch/users/GASPAR_USERNAME \
  -e HF_API_KEY_FILE_AT=/mloscratch/users/GASPAR_USERNAME/keys/hf_key.txt \
  -e WANDB_API_KEY_FILE_AT=/mloscratch/users/GASPAR_USERNAME/keys/wandb_key.txt \
  -e GITCONFIG_AT=/mloscratch/users/GASPAR_USERNAME/.gitconfig \
  -e GIT_CREDENTIALS_AT=/mloscratch/users/GASPAR_USERNAME/.git-credentials \
  -e VSCODE_CONFIG_AT=/mloscratch/users/GASPAR_USERNAME/.vscode-server \
  --backoff-limit 0 \
  --run-as-gid 84257 \
  --node-pool h100 \
  --gpu 1 \
  -- sleep infinity

# Attach to the running pod
runai bash <job-name>
```

### Step 2: Inside the Kubernetes Pod

#### 2.1 Launch the SGLang Server

```bash
python3 -m sglang.launch_server \
  --model-path OpenMeditron/Meditron3-8B \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --port 31000 \
  --mem-fraction-static 0.6
```

Wait for the server to be ready (you'll see "Server is ready" in the logs).

#### 2.2 Run Evaluations

In a new terminal (or background the server), run the evaluation with your desired schema:

```bash
python3 -m lm_eval \
  --model sglang-schema \
  --model_args pretrained=OpenMeditron/Meditron3-8B,base_url=http://localhost:31000,schema_model=schemas.medical_qa_schemas.<SCHEMA_CLASS> \
  --tasks <TASK_NAME> \
  --output_path ./results/<output_file>.json \
  --log_samples
```

**Example - MedQA with Level 1 (Strict Answer):**
```bash
python3 -m lm_eval \
  --model sglang-schema \
  --model_args pretrained=OpenMeditron/Meditron3-8B,base_url=http://localhost:31000,schema_model=schemas.medical_qa_schemas.StrictMultipleChoice \
  --tasks medqa_4options_generation \
  --output_path ./results/medqa/medqa_level1.json \
  --log_samples
```
