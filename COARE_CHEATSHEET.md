# 📋 COARE Supercomputer — Quick Command Cheat Sheet

> **How to use this sheet:** When you need to do something quickly and don't want to re-read the full tutorial, look it up here. Commands labeled 🪟 are run in **PowerShell on your laptop**. Commands labeled 🖥️ are run in the **SSH Terminal (inside COARE)**.

---

## 🚨 READ THIS FIRST — For Teammates Receiving This Codebase

> **This section is specifically for Neil and Kent (or anyone who received the codebase from Nathaniel via Google Drive).** Please read this entire section before touching any file.

### ⚠️ Warning 1: Hardcoded Paths in the Code

Several scripts in this project contain **hardcoded paths that are specific to Nathaniel's setup**. These will NOT work on your machine or COARE account out of the box. You need to find and update them before running anything.

**Files with hardcoded paths you need to fix:**

| File | What's Hardcoded | What to Change It To |
|---|---|---|
| `classification/train.py` | `E:\\Thesis_Datasets\\images\\` in error messages | Your own dataset path |
| `classification/evaluate.py` | `E:\\Thesis_Datasets\\images\\` in error messages | Your own dataset path |
| `classification/dataset/image_dataset.py` | `E:\\Thesis_Datasets\\images\\` in docstring | Your own dataset path |
| `classification/model_soup.py` | `E:\\Thesis_Datasets\\images\\updated_data_4` in print statements | Your own dataset path |
| `classification/slurm_scripts/*.slurm` | `/scratch1/nathaniel.merka/...` in ALL SLURM scripts | Your own COARE username (e.g., `/scratch1/neil.mallari/...`) |

**How to fix them (the easy way):** Open each file in VS Code, press `Ctrl + H` (Find & Replace), and:
- Replace `nathaniel.merka` → your COARE username (e.g., `neil.mallari`)
- Replace `E:\\Thesis_Datasets` → wherever your datasets live on your machine

> 💡 **Tip:** If you are unsure about a specific line, just paste the relevant file into ChatGPT or Gemini and ask: *"What paths in this script need to be changed for a different user's machine?"* It will point them out for you immediately.

---

### ⚠️ Warning 2: The SLURM Scripts Are in a Separate Folder

The SLURM job scripts (`.slurm` files) are located inside `classification/slurm_scripts/`. This means when you submit a job, you need to write the path:

```bash
# ✅ Correct — include the subfolder name
sbatch slurm_scripts/train_exp2_enhanced_arch_baseline_data.slurm

# ❌ Wrong — this won't find the file
sbatch train_exp2_enhanced_arch_baseline_data.slurm
```

Similarly, the dataset download/extraction Python scripts are inside `classification/dataset_scripts/`.

---

### ⚠️ Warning 3: Submitting a Job Does NOT Mean It Worked

This is a **very common mistake**. When you run `sbatch` and see `Submitted batch job 548XXX`, the job has been **queued** — it has NOT started running yet, and it has NOT finished successfully.

**You MUST always verify a job actually ran correctly by following these steps:**

**Step 1 — Check if it is even running:**
```bash
squeue -u your.name
```
Look at the `ST` column:
- `PD` = Still waiting in queue (normal, could take minutes to hours)
- `R` = Running now ✅
- (Absent from list) = Finished (could be success OR crash)

**Step 2 — Check the output log once it finishes:**
```bash
# Replace 548XXX with your actual job number
cat train_baseline_548XXX.out
```
A **successful run** will end with `Training Complete!` and show epoch-by-epoch AUC scores.
A **crashed run** will end immediately with `Training Complete!` but show no epoch data — this is the bug.

**Step 3 — Always check the error log too:**
```bash
cat train_baseline_548XXX.err
```
If you see a Python `Traceback` or `Error` here, the job crashed. Copy the error and fix the script, then re-submit.

> ⚠️ **A job that ends in 10 seconds is almost certainly a crash.** A real training job should take hours. If your job disappeared from `squeue` within 1-2 minutes of submitting, check the `.err` file immediately.

---

### ⚠️ Warning 4: The Model Checkpoints Are NOT in GitHub

The trained `.pth` model files are too large for GitHub and are gitignored. They were shared with you separately via Google Drive. After downloading them, place them in:

```
classification/output/efficientnet_b4_20260903_144444/best_model.pth
```

This path is important because the SLURM training scripts point to it as the `--pretrained_image_checkpoint`. If this file is missing, every training job will crash immediately.

---


## 🔌 Connecting to COARE

> **When to use:** Every time you want to work on COARE.  
> **Run in:** 🪟 PowerShell (on your laptop)

Replace `your.name` with your actual COARE username (e.g. `neil.mallari`):
```powershell
ssh -i C:\Users\<YourName>\.ssh\id_rsa_coare your.name@saliksik.asti.dost.gov.ph
```

**What you should see:** The COARE welcome banner with your storage quota. You are now inside the supercomputer.

> 💡 If you see `Connection reset`, just run the command again. This happens often and is normal.

---

## 📤 Uploading Files From Your Laptop to COARE

> **When to use:** When you edit a Python or SLURM script locally and need to update it on COARE.  
> **Run in:** 🪟 PowerShell (on your laptop)

**Upload a single Python script:**
```powershell
scp -i C:\Users\<YourName>\.ssh\id_rsa_coare C:\path\to\script.py your.name@saliksik.asti.dost.gov.ph:/scratch1/your.name/EfficientNet-PyTorch/classification/
```

**Upload an entire folder at once (use `-r` for "recursive"):**
```powershell
scp -i C:\Users\<YourName>\.ssh\id_rsa_coare -r C:\path\to\classification your.name@saliksik.asti.dost.gov.ph:/scratch1/your.name/EfficientNet-PyTorch/
```

**Upload a dataset zip file:**
```powershell
scp -i C:\Users\<YourName>\.ssh\id_rsa_coare C:\path\to\dataset.zip your.name@saliksik.asti.dost.gov.ph:/scratch1/your.name/datasets/thesis/
```

---

## 📥 Downloading Files FROM COARE to Your Laptop

> **When to use:** After training finishes and you want the results on your laptop.  
> **Run in:** 🪟 PowerShell (on your laptop)

**Download a single file (e.g. trained model checkpoint):**
```powershell
scp -i C:\Users\<YourName>\.ssh\id_rsa_coare your.name@saliksik.asti.dost.gov.ph:/scratch1/your.name/datasets/output_video/video_tsm_mhsa_XXXXXXXX/best_video_model.pth C:\path\to\save\here\
```

**Download an entire folder:**
```powershell
scp -i C:\Users\<YourName>\.ssh\id_rsa_coare -r your.name@saliksik.asti.dost.gov.ph:/scratch1/your.name/datasets/output_video/ C:\path\to\save\here\
```

---

## ⚙️ Submitting and Managing Jobs (SLURM)

> **When to use:** To start training or download jobs.  
> **Run in:** 🖥️ SSH Terminal (inside COARE)

**First, always go to the classification folder:**
```bash
cd /scratch1/your.name/EfficientNet-PyTorch/classification
```

**Submit a job:**
```bash
sbatch train_video_prototype.slurm
# You will see: Submitted batch job 548XXX
# Write down that job number!
```

**Check all your active jobs:**
```bash
squeue -u your.name
```
The **ST** column tells you the status:
- `R` = Running ✅
- `PD` = Pending (waiting in queue, will start soon)

**Cancel a job you want to stop:**
```bash
scancel 548XXX   # Replace with your actual job ID
```

---

## 📄 Checking Job Logs (Progress & Errors)

> **When to use:** To see if a job is working, what progress it made, or why it crashed.  
> **Run in:** 🖥️ SSH Terminal (inside COARE)

**Print the full output log:**
```bash
cat train_video_proto_548XXX.out
```

**Watch the output live (auto-updates, press `Ctrl+C` to stop):**
```bash
tail -f train_video_proto_548XXX.out
```

**Check for errors:**
```bash
cat train_video_proto_548XXX.err
```

> 💡 HuggingFace download progress bars appear in the `.err` file (not the `.out` file). This is normal — they're not actually errors!

**View the detailed training log (per epoch, with timestamps):**
```bash
# First find your run ID:
ls /scratch1/your.name/datasets/output_video/

# Then read the log:
cat /scratch1/your.name/datasets/output_video/video_tsm_mhsa_XXXXXXXX_XXXXXX/console.log
```

---

## 🗂️ Navigating Files on COARE

> **Run in:** 🖥️ SSH Terminal (inside COARE)

**Go to the project scripts folder:**
```bash
cd /scratch1/your.name/EfficientNet-PyTorch/classification
```

**Go to the datasets folder:**
```bash
cd /scratch1/your.name/datasets/extracted/videos-zip
```

**List all files and folders in the current location:**
```bash
ls
```

**List with more details (sizes, dates):**
```bash
ls -lh
```

**Check if your datasets finished extracting:**
```bash
ls /scratch1/your.name/datasets/extracted/videos-zip/
# You should see folders like: wilddeepfake/  synth_vid_detect/  dfdc_frames/  etc.
```

**Check how much disk space you've used:**
```bash
du -sh /scratch1/your.name/datasets/
```

---

## ✏️ Editing a Script Directly on COARE

> **When to use:** When you need to quickly change a SLURM script (e.g. add `--resume`) without re-uploading from your laptop.  
> **Run in:** 🖥️ SSH Terminal (inside COARE)

Use `nano`, a simple text editor:
```bash
nano /scratch1/your.name/EfficientNet-PyTorch/classification/train_video_prototype.slurm
```

- Use **arrow keys** to move around
- Make your edit
- Press **`Ctrl + O`** to save
- Press **`Ctrl + X`** to exit

---

## ⚠️ Common Errors At a Glance

| Error | Meaning | Fix |
|---|---|---|
| `Connection reset` | SSH timed out | Just reconnect — your jobs are still running |
| `Permission denied (publickey)` | SSH key missing | Check `id_rsa_coare` is in `C:\Users\<YourName>\.ssh\` |
| `Invalid qos specification` | Wrong QoS name in SLURM file | Use `gpu-p40_default` for `gpu` partition, `gpu-a100_default` for `gpu_a100` |
| `QOSMaxCpuPerUserLimit` | Already have a GPU job running | Wait for current GPU job to finish before submitting another |
| `ImportError: numpy.core.multiarray` | NumPy upgraded to 2.x | Add `"numpy<2.0.0"` to the pip install in the SLURM script |
| Job stuck as `PD` for hours | GPU partition is congested | Switch to the P40 partition (see tutorial) |
