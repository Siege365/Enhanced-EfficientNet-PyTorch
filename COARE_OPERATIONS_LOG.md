# 🖥️ COARE Supercomputer Training Guide (Beginner Edition)
**Project:** General Synthetic Media Detection | University of Mindanao  
**For:** Neil Ian R. Mallari, Kent Lenoel C. Sevellino  
**Written by:** Nathaniel Keene M. Merka  
**Last Updated:** September 19, 2026

---

> ## 📌 Read This First!
> This guide is designed to be **completely foolproof**. You do not need to be an expert programmer — just follow every step **in order** and copy the commands exactly as written. 
> 
> **Whenever a command says `your.name`, replace it with your actual COARE username (e.g. `neil.mallari` or `kent.sevellino`).**
>
> **Two types of terminals are used in this guide:**
> - 🪟 **PowerShell** — This is the Windows terminal on your own laptop.
> - 🖥️ **SSH Terminal** — This is the COARE supercomputer's terminal (which you access via PowerShell).

---

## ✅ Before You Start — The "Must-Haves"

Make sure you have ALL of these before doing anything:
1. **A COARE account** (request one at https://coarehub.asti.dost.gov.ph — Nathaniel will endorse you).
2. **Your SSH private key** (Nathaniel will give this to you or help you generate it). Save it on your laptop exactly at this location:
   `C:\Users\YOUR_WINDOWS_USERNAME\.ssh\id_rsa_coare`
3. **The Project Code:** Download this GitHub repository to your laptop. (You can just click "Download ZIP" on GitHub and extract it to your Documents folder).

---

## PART 1 — Connecting to the Supercomputer

### Step 1.1 — Open PowerShell
Press the **Windows Key**, type **PowerShell**, and press Enter. A blue or black text window will open. 

### Step 1.2 — SSH into COARE
Copy this command, paste it into PowerShell, change `your.name` to your actual COARE username, and press Enter:

```powershell
ssh -i ~/.ssh/id_rsa_coare your.name@saliksik.asti.dost.gov.ph
```

*(Note: The `~` automatically finds your Windows user folder, so you don't need to type `C:\Users\Kent...`)*

**What you should see:** A big ASCII art logo of COARE will appear, followed by a table showing your storage quota. This means you are successfully logged into the supercomputer! 
**Leave this window open. It is now your 🖥️ SSH Terminal.**

> **⚠️ Error: `Connection reset`?** Don't panic. Just press the UP arrow on your keyboard and hit Enter to run the command again.
> **⚠️ Error: `Permission denied (publickey)`?** Your SSH key is missing or named wrong. Make sure it is exactly in the `.ssh` folder and named `id_rsa_coare`.

---

## PART 2 — Uploading the Code to COARE

> 🚨 **IMPORTANT:** Open a **NEW, SECOND PowerShell window** for this step. Do not type this in the COARE SSH terminal!

### Step 2.1 — Navigate to the project folder
In your new 🪟 PowerShell window, use the `cd` command to go to the folder where you extracted the GitHub repository. For example:
```powershell
cd C:\Users\Kent\Documents\EfficientNet-PyTorch
```

### Step 2.2 — Upload the code
Now run this exact command to upload the `classification` folder to your COARE account (remember to change `your.name`!):

```powershell
scp -i ~/.ssh/id_rsa_coare -r classification/ your.name@saliksik.asti.dost.gov.ph:/scratch1/your.name/
```

**What you should see:** A long list of files scrolling rapidly down your screen as they upload. Wait until it completely finishes and your PowerShell cursor returns.

---

## PART 3 — The Magic Step (Shared Datasets)

Here is the best part: **Nathaniel has already downloaded, extracted, and preprocessed all 150GB of datasets for you.** He has granted your COARE accounts direct permission to read his files. 

You **DO NOT** need to download any datasets. You **DO NOT** need to download any starting model weights. 

### Step 3.1 — Update your script output paths
Because you are reading Nathaniel's datasets, the scripts are currently set up to save the trained models in *Nathaniel's* folder. You don't have permission to write to his folder. 

Go back to your **🖥️ SSH Terminal** (the one with the COARE logo) and run these exact commands one by one. It will automatically update all the SLURM scripts to save the trained models in YOUR folder instead of his:

```bash
# 1. Go to your scripts folder
cd /scratch1/your.name/classification/slurm_scripts/

# 2. Automatically replace Nathaniel's name with yours in the output paths
# (Make sure to replace your.name with your actual username before pressing Enter!)
sed -i 's/nathaniel.merka\/datasets/your.name/g' *.slurm
```

---

## PART 4 — Running the Training Experiments

> 🚨 **CRITICAL: Fix Windows Line Endings First**
> Because you uploaded these files from a Windows laptop, they contain invisible Windows line breaks that will immediately crash the Linux supercomputer. Always run this `sed` command before submitting a script!

Go to your classification folder in your **🖥️ SSH Terminal**:
```bash
cd /scratch1/your.name/classification
```

### 🧪 Experiment 1 — Baseline Architecture on Baseline Datasets
Vanilla EfficientNet-B4 trained on face-cropped DFDC and FaceForensics++ frames. 
```bash
# 1. Fix line endings
sed -i 's/\r$//' slurm_scripts/train_exp1_baseline_arch_baseline_data.slurm

# 2. Submit the job
sbatch slurm_scripts/train_exp1_baseline_arch_baseline_data.slurm
```

### 🧪 Experiment 2 — Enhanced Architecture on Baseline Datasets
EfficientNet-B4 + TSM + MHSA trained on face-cropped DFDC and FaceForensics++ frames.
```bash
sed -i 's/\r$//' slurm_scripts/train_exp2_enhanced_arch_baseline_data.slurm
sbatch slurm_scripts/train_exp2_enhanced_arch_baseline_data.slurm
```

### 🧪 Experiment 3 — Baseline Architecture on New Datasets
Vanilla EfficientNet-B4 trained on modern AI video datasets (Kling, Sora, CivitAI, etc.).
```bash
sed -i 's/\r$//' slurm_scripts/train_exp3_baseline_arch_new_data.slurm
sbatch slurm_scripts/train_exp3_baseline_arch_new_data.slurm
```

### 🧪 Experiment 4 — Enhanced Architecture on New Datasets
The full proposed model — EfficientNet-B4 + TSM + MHSA — trained on all modern AI video datasets. This is the flagship experiment.
```bash
sed -i 's/\r$//' slurm_scripts/train_exp4_enhanced_arch_new_data.slurm
sbatch slurm_scripts/train_exp4_enhanced_arch_new_data.slurm
```

---

## PART 5 — Monitoring Your Jobs

> ## 🚨 CRITICAL: Submitting a Job Does NOT Mean It Worked
> When you see `Submitted batch job 548XXX`, the job is **queued**, not done. You MUST verify it **actually ran** by following the steps below. A job that finishes in under 2 minutes almost certainly crashed.

### Step 1 — Check all your active jobs at any time:
```bash
squeue -u your.name
```
You will see a table. The **ST** column shows the status:
- `R` — Running ✅ (the job is actively computing on a node)
- `PD` — Pending (waiting in queue, COARE is busy, it will start automatically later)
- *(Absent from list)* — Finished — could be **success or crash**, check the log immediately!

### Step 2 — Check the output log to confirm it actually trained:
When you submitted the job, it gave you a number (e.g., 548123). Run this command in your **🖥️ SSH Terminal**:
```bash
cat exp1_548123.out
```
A **successful run** will show epoch-by-epoch progress bars. 
A **crashed run** will end immediately with `Training Complete!` but show **no epoch lines**. If you see this, check the `.err` file next.

### Step 3 — Always check the error log:
```bash
cat exp1_548123.err
```
If you see a Python `Traceback` or any line starting with `Error:`, the job crashed. Copy the error message and send it to Nathaniel.

### Watch live training progress:
Instead of opening the file, you can watch it live as it trains:
```bash
tail -f exp1_548123.out
```
Press `Ctrl + C` to stop watching. The job keeps running in the background even if you disconnect!

---

## PART 6 — Downloading Results Back to Your Laptop

After the 15-hour training finishes, you will want the trained model checkpoint on your laptop. 
Open a **🪟 PowerShell** window (NOT the SSH terminal) and run this to download your output folder:

```powershell
scp -i ~/.ssh/id_rsa_coare -r your.name@saliksik.asti.dost.gov.ph:/scratch1/your.name/output_exp4_enh_new/ C:\Users\Kent\Documents\EfficientNet-PyTorch\output_exp4\
```
*(Remember to change the paths to wherever you want to save it!)*
