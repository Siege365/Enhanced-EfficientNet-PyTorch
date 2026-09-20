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

## PART 2 — The Magic Step (Shared Code & Datasets)

Here is the best part: **Nathaniel has already uploaded the code and all 150GB of datasets to COARE.** He has granted your COARE accounts direct permission to read and use his files! 

You **DO NOT** need to upload any code. You **DO NOT** need to download any datasets. You just need to create "Symbolic Links" (shortcuts) that connect your account directly to Nathaniel's files.

Go back to your **🖥️ SSH Terminal** (the one with the COARE logo) and run these exact commands:

```bash
# 1. Go to your scratch folder
cd /scratch1/your.name/

# 2. Create the Symbolic Links pointing to Nathaniel's folders
ln -s /scratch1/nathaniel.merka/EfficientNet-PyTorch EfficientNet-PyTorch
ln -s /scratch1/nathaniel.merka/datasets datasets

# 3. Verify they were created (you should see little arrows -> pointing to his folders)
ls -la
```

---

## PART 3 — Running the Training Experiments

> 🚨 **CRITICAL: You Must Copy the Script to Your Own Folder First**
> SLURM is very strict about security. If you try to submit a job from inside Nathaniel's folder (or from the symlink), SLURM will instantly crash the job because it refuses to write `.out` log files into a folder you don't officially own. 

You must create your own `my_scripts` folder, copy the script you want to run into it, and submit it from there.

**Run these commands in your 🖥️ SSH Terminal:**
```bash
# 1. Create a folder you officially own
mkdir -p /scratch1/your.name/my_scripts

# 2. Copy the script you want to run (e.g., Experiment 3) into your folder
cp /scratch1/your.name/EfficientNet-PyTorch/classification/slurm_scripts/train_exp3_baseline_arch_new_data.slurm /scratch1/your.name/my_scripts/

# 3. Go to your folder
cd /scratch1/your.name/my_scripts/

# 4. Submit the job
sbatch train_exp3_baseline_arch_new_data.slurm
```

### 🎯 Who is Running What?
To bypass the single-user GPU limit and speed up the thesis, we are dividing the workload across all 3 accounts simultaneously:
- **Nathaniel (Experiment 2):** Enhanced Arch on Baseline Data
- **Ken (Experiment 3):** Vanilla Baseline on Modern Data
- **Neil (Experiment 4):** Enhanced Arch on Modern Data

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
Instead of opening the file, you can watch it live as it trains. We recommend opening two terminals to watch both logs at the same time:
```bash
cd /scratch1/your.name/my_scripts/
tail -f exp1_548123.out   # Watch system logs (PyTorch installing, etc.)
tail -f exp1_548123.err   # Watch Python console & training progress bar
tail -f output_exp1_base_base/*/console.log  # Watch the permanent Python log file
```
Press `Ctrl + C` to stop watching. The job keeps running in the background even if you disconnect!

## PART 6 — Downloading Results Back to Your Laptop

After the 15-hour training finishes, you will want the trained model checkpoint on your laptop. 
Open a **🪟 PowerShell** window (NOT the SSH terminal) and run this to download your output folder:

```powershell
scp -i ~/.ssh/id_rsa_coare -r your.name@saliksik.asti.dost.gov.ph:/scratch1/your.name/output_exp4_enh_new/ C:\Users\Kent\Documents\EfficientNet-PyTorch\output_exp4\
```
*(Remember to change the paths to wherever you want to save it!)*
