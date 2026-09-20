# 📋 COARE Supercomputer — Quick Command Cheat Sheet

> **How to use this sheet:** When you need to do something quickly and don't want to re-read the full tutorial, look it up here. Commands labeled 🪟 are run in **PowerShell on your laptop**. Commands labeled 🖥️ are run in the **SSH Terminal (inside COARE)**.

---

## 🚨 READ THIS FIRST — For Teammates Receiving This Codebase

> **This section is specifically for Neil and Kent.** Please read this entire section before touching any file.

### ⚠️ Warning 1: Shared Datasets (DO NOT DOWNLOAD)
Nathaniel has already downloaded, extracted, and preprocessed all 150GB+ of datasets into his COARE account. **You do not need to download or extract any datasets.** 
Nathaniel has granted your COARE accounts "Read & Execute" access to his `datasets/` directory.

In the SLURM training scripts (`.slurm` files), the `--video_dirs` are already pointing to `/scratch1/nathaniel.merka/datasets/...`. 
✅ **Leave these dataset paths exactly as they are.** Your jobs will read the data directly from Nathaniel's folder at lightning speed.

### ⚠️ Warning 2: Hardcoded Output Paths (MUST BE CHANGED)
While you will read datasets from Nathaniel's folder, you CANNOT save your trained models to his folder (you don't have write access, and you would overwrite his work).

You must update the **output paths** in the SLURM scripts to point to your own scratch folder.
Open every `.slurm` file in `classification/slurm_scripts/` and change:
`--output_dir /scratch1/nathaniel.merka/...`
to:
`--output_dir /scratch1/your.username/...`

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
cat exp1_548XXX.out
```
A **successful run** will end with `Training Complete!` and show epoch-by-epoch AUC scores.
A **crashed run** will end immediately with `Training Complete!` but show no epoch data — this is the bug.

**Step 3 — Always check the error log too:**
```bash
cat exp1_548XXX.err
```
If you see a Python `Traceback` or `Error` here, the job crashed. Copy the error and fix the script, then re-submit.

> ⚠️ **A job that ends in 10 seconds is almost certainly a crash.** A real training job should take hours. If your job disappeared from `squeue` within 1-2 minutes of submitting, check the `.err` file immediately.

### ⚠️ Warning 4: Initial Pretrained Model Weights

The Enhanced architecture requires initializing the backbone with weights from Nathaniel's Phase 2 Image model (`best_model.pth`). 
Just like the datasets, Nathaniel has shared this file with you on COARE. 

In your SLURM scripts, ensure the `--pretrained_image_checkpoint` points directly to his folder:
`/scratch1/nathaniel.merka/EfficientNet-PyTorch/classification/output/efficientnet_b4_20260903_144444/best_model.pth`

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

---

## 📥 Downloading Files FROM COARE to Your Laptop

> **When to use:** After training finishes and you want the results on your laptop.  
> **Run in:** 🪟 PowerShell (on your laptop)

**Download a single file (e.g. trained model checkpoint):**
```powershell
scp -i C:\Users\<YourName>\.ssh\id_rsa_coare your.name@saliksik.asti.dost.gov.ph:/scratch1/your.name/output_exp4_enh_new/best_video_model.pth C:\path\to\save\here\
```

**Download an entire folder:**
```powershell
scp -i C:\Users\<YourName>\.ssh\id_rsa_coare -r your.name@saliksik.asti.dost.gov.ph:/scratch1/your.name/output_exp4_enh_new/ C:\path\to\save\here\
```

---

## ⚙️ Submitting and Managing Jobs (SLURM)

> **When to use:** To start training or download jobs.  
> **Run in:** 🖥️ SSH Terminal (inside COARE)

**First, always go to the classification folder:**
```bash
cd /scratch1/your.name/EfficientNet-PyTorch/classification
```

**Fix Windows Line Endings (CRITICAL):**
If you uploaded the script from your Windows laptop, it has invisible DOS line breaks (`\r\n`). You MUST strip them before submitting, or SLURM will crash.
```bash
sed -i 's/\r$//' slurm_scripts/train_exp1_baseline_arch_baseline_data.slurm
```

**Submit a job:**
```bash
sbatch slurm_scripts/train_exp1_baseline_arch_baseline_data.slurm
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
cat exp1_548XXX.out
```

**Watch the output live (auto-updates, press `Ctrl+C` to stop):**
```bash
tail -f exp1_548XXX.out
```

**Check for errors:**
```bash
cat exp1_548XXX.err
```

**View the detailed training log (per epoch, with timestamps):**
Every run automatically creates a `console.log` file in the `--output_dir` you specified.
```bash
cat /scratch1/your.name/output_exp1_base_base/console.log
```

---

## ⚠️ Common Errors At a Glance

| Error | Meaning | Fix |
|---|---|---|
| `Batch script contains DOS line breaks (\r\n)` | You edited the SLURM file on Windows | Run `sed -i 's/\r$//' your_script.slurm` before submitting |
| `Connection reset` | SSH timed out | Just reconnect — your jobs are still running |
| `Permission denied (publickey)` | SSH key missing | Check `id_rsa_coare` is in `C:\Users\<YourName>\.ssh\` |
| `Invalid qos specification` | Wrong QoS name in SLURM file | Use `gpu-p40_default` for `gpu` partition |
| `QOSMaxCpuPerUserLimit` | Already have a GPU job running | Wait for current GPU job to finish before submitting another |
| Job stuck as `PD` for hours | GPU partition is congested | Normal. COARE is busy. Wait for it to start. |
