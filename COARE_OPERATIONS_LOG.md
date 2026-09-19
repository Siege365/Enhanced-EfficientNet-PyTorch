# 🖥️ COARE Supercomputer Training Guide
**Project:** General Synthetic Media Detection | University of Mindanao  
**For:** Neil Ian R. Mallari, Kent Lenoel C. Sevellino  
**Written by:** Nathaniel Keene M. Merka  
**Last Updated:** September 19, 2026

---

> ## 📌 Read This First!
> This guide teaches you step by step how to run our AI training experiments on the DOST-ASTI **COARE Supercomputer** (Saliksik). You don't need to be an expert programmer — just follow every step **in order** and copy the commands exactly as written. When a command says `your.name`, replace it with your actual COARE username (e.g. `neil.mallari` or `kent.sevellino`).
>
> **Two types of terminals are used in this guide:**
> - 🪟 **PowerShell** — This is the Windows terminal on your own laptop. Open it by pressing `Win + X` → "Windows Terminal" or "PowerShell".
> - 🖥️ **SSH Terminal** — This is the COARE supercomputer's terminal. You get to it by running the SSH command in PowerShell.

---

## ✅ Before You Start — Checklist

Make sure you have ALL of these before continuing:
- [ ] A COARE account (request one at https://coarehub.asti.dost.gov.ph — ask Nathaniel to endorse you)
- [ ] Your SSH key file saved at `C:\Users\<YourName>\.ssh\id_rsa_coare` on your laptop
- [ ] Access to the shared **Google Drive folder** with the dataset zip files (ask Nathaniel for the link)
- [ ] A **Kaggle account** (free, sign up at https://kaggle.com — needed for DFDC and image datasets)
- [ ] A **HuggingFace account** (free, sign up at https://huggingface.co — needed for some datasets)
- [ ] This GitHub repository cloned to your laptop

---

## PART 1 — Connecting to the Supercomputer

### Step 1.1 — Open PowerShell on your laptop
Press `Win + X` on your keyboard and click **"Windows Terminal"** or **"PowerShell"**. A black or blue window will open. This is your 🪟 PowerShell.

### Step 1.2 — SSH into COARE
Type this command and press Enter. Replace `your.name` with your actual COARE username:

```powershell
ssh -i C:\Users\<YourName>\.ssh\id_rsa_coare your.name@saliksik.asti.dost.gov.ph
```

**What you should see:** A big ASCII art logo of COARE will appear, followed by your storage quota. This means you are now INSIDE the supercomputer! This window is now your 🖥️ SSH Terminal.

> **⚠️ If you see `Connection reset`:** Don't panic. The login node sometimes drops idle connections. Just run the SSH command again.

> **⚠️ If you see `Permission denied (publickey)`:** Your SSH key is not in the right place. Make sure the file exists at `C:\Users\<YourName>\.ssh\id_rsa_coare`. If not, ask Nathaniel for the key file.

---

## PART 2 — Setting Up the Project Code on COARE

> **Do Steps 2.1 and 2.2 in your 🪟 PowerShell (on your laptop).**
> **Do Step 2.3 in your 🖥️ SSH Terminal (inside COARE).**

### Step 2.1 — Create the Project Folders on COARE
Run these in your **🖥️ SSH Terminal**. These create the directory structure where everything will live:

```bash
mkdir -p /scratch1/your.name/EfficientNet-PyTorch
mkdir -p /scratch1/your.name/datasets/thesis
mkdir -p /scratch1/your.name/datasets/extracted/videos-zip
mkdir -p /scratch1/your.name/datasets/extracted/images
mkdir -p /scratch1/your.name/datasets/output_video
mkdir -p /scratch1/your.name/EfficientNet-PyTorch/classification/output/efficientnet_b4_20260903_144444
```

**What this does:** Creates all the folders we will use to store code, datasets, and trained model checkpoints. The `/scratch1/` folder is the fast, large-storage drive (2.2 TB) where we do all our heavy work.

### Step 2.2 — Upload the Project Code
This is a single command that uploads the entire `classification/` folder (all Python scripts, all SLURM job files, and all preprocessing tools) to COARE. Run this in your **🪟 PowerShell**:

```powershell
scp -i C:\Users\<YourName>\.ssh\id_rsa_coare -r C:\path\to\EfficientNet-PyTorch\classification your.name@saliksik.asti.dost.gov.ph:/scratch1/your.name/EfficientNet-PyTorch/
```

> Replace `C:\path\to\EfficientNet-PyTorch` with the actual location where you cloned the GitHub repository on your laptop.

**What you should see:** A list of files being uploaded, showing their transfer speeds. Wait until it finishes completely (the command prompt `PS >` returns).

### Step 2.3 — Replace `nathaniel.merka` With YOUR Username
All the scripts are currently configured with Nathaniel's COARE username. You must replace every instance of `nathaniel.merka` in the scripts with your own username. Run this single command in your **🖥️ SSH Terminal**:

```bash
grep -rl "nathaniel.merka" /scratch1/your.name/EfficientNet-PyTorch/ | xargs sed -i 's/nathaniel.merka/your.name/g'
```

Then verify it worked (the output should be completely empty — no results means success):
```bash
grep -r "nathaniel.merka" /scratch1/your.name/EfficientNet-PyTorch/classification/
```

### Step 2.4 — Upload the Pretrained Image Checkpoint
The Phase 3 video model does NOT start from zero. It starts with the weights of the best Phase 2 image model (the one trained by Nathaniel). Get `best_model.pth` from the Google Drive folder and upload it:

```powershell
scp -i C:\Users\<YourName>\.ssh\id_rsa_coare C:\path\to\downloaded\best_model.pth your.name@saliksik.asti.dost.gov.ph:/scratch1/your.name/EfficientNet-PyTorch/classification/output/efficientnet_b4_20260903_144444/best_model.pth
```

---

## PART 3 — Image Datasets (For Phase 1 & 2 Image Training)

> These datasets are used to train and evaluate the **image-based deepfake detector** (EfficientNet-B4). They consist of single images — not videos.

### 3.1 — D1: AI vs Human Generated Dataset (Kaggle)

**What it is:** Real photos from Shutterstock + fake AI-generated images from various generators.

1. Go to https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset
2. Click **Download** (you need a Kaggle account)
3. Upload to COARE from **🪟 PowerShell**:
```powershell
scp -i C:\Users\<YourName>\.ssh\id_rsa_coare C:\path\to\ai-vs-human-generated-dataset.zip your.name@saliksik.asti.dost.gov.ph:/scratch1/your.name/datasets/thesis/
```

### 3.2 — D2: DeepDetect-2025 (Kaggle)

**What it is:** Portrait photos (real) + StyleGAN3, DALL-E 3, Midjourney v5, SD3 (fake).

1. Go to https://www.kaggle.com/datasets/ayushmandatta1/deepdetect-2025
2. Click **Download**
3. Upload to COARE:
```powershell
scp -i C:\Users\<YourName>\.ssh\id_rsa_coare C:\path\to\deepdetect-2025.zip your.name@saliksik.asti.dost.gov.ph:/scratch1/your.name/datasets/thesis/
```

### 3.3 — D3: OpenFake (HuggingFace)

**What it is:** Diverse real photos + newest 2025-26 AI generators (Flux, GPT-Image 1, Imagen 4, Midjourney v7, Grok-2).

Download directly to COARE using a SLURM job. Run these in your **🖥️ SSH Terminal**:

> There is currently no pre-written download script for this. You can download it manually:
```bash
module load anaconda/3-2024.10-1
/opt/modules/library/cpu/anaconda/3-2024.10-1/bin/pip install --user huggingface_hub "numpy<2.0.0"
/opt/modules/library/cpu/anaconda/3-2024.10-1/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='ComplexDataLab/OpenFake', repo_type='dataset', local_dir='/scratch1/your.name/datasets/extracted/images/openfake')
"
```

### 3.4 — D5: SuSy-Dataset (HuggingFace)

**What it is:** MS COCO natural scenes (real) + DiffusionDB, RealisticSDXL, DALL-E 3, Midjourney (fake).

```bash
/opt/modules/library/cpu/anaconda/3-2024.10-1/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='HPAI-BSC/SuSy-Dataset', repo_type='dataset', local_dir='/scratch1/your.name/datasets/extracted/images/susy')
"
```

### 3.5 — D6: Defactify Image Dataset (HuggingFace)

**What it is:** MS COCO (real) + SD 2.1, SD 3, SDXL, DALL-E 3, Midjourney v6 (fake). AAAI 2025 competition dataset.

```bash
/opt/modules/library/cpu/anaconda/3-2024.10-1/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Rajarshi-Roy-research/Defactify_Image_Dataset', repo_type='dataset', local_dir='/scratch1/your.name/datasets/extracted/images/defactify')
"
```

### 3.6 — Running Image Training (Phase 1/2)
Once the image datasets are ready, run the image training script:
```bash
cd /scratch1/your.name/EfficientNet-PyTorch/classification
/opt/modules/library/cpu/anaconda/3-2024.10-1/bin/python train.py \
    --data_dir /scratch1/your.name/datasets/extracted/images \
    --epochs 20 --batch_size 16 --lr 0.0001
```

---

## PART 4A — Video Datasets from Google Drive

> These datasets come as `.zip` files from the shared Google Drive folder. Download each one to your laptop first, then upload to COARE.

**Complete list of Google Drive datasets to upload:**

| What it contains | Zip filename | Used for |
|---|---|---|
| WildDeepfake (face-swap deepfakes from the internet) | `wilddeepfake.zip` | Video training |
| SynthVidDetect (Sora, Pika, SVD AI videos) | `synth_vid_detect.zip` | Video training |
| CivitAI scraped AI videos | `civitai_scraped.zip` | Video training |
| Hailuo AI videos | `hailuo_scraped.zip` | Video training |
| Kling AI videos | `kling_scraped.zip` | Video training |
| FaceForensics++ C23 frames | `FaceForensics++_C23.zip` | Video training (needs preprocessing) |
| Celeb-DF face-swap deepfakes | `celebdf_frames.zip` | External benchmark only |
| Google DFD deepfakes | `dfd_frames.zip` | External benchmark only |

### Step 4A.1 — How to Upload Each Zip File

**Option A (For files under ~10 GB):** Download to your laptop, then upload via PowerShell:
```powershell
scp -i C:\Users\<YourName>\.ssh\id_rsa_coare C:\path\to\wilddeepfake.zip your.name@saliksik.asti.dost.gov.ph:/scratch1/your.name/datasets/thesis/
```
Repeat this for every zip file in the table above.

**Option B (Faster, for large files):** Download directly from Google Drive INTO COARE without going through your laptop. This skips your home internet connection entirely:

1. Get the shareable link for each file from Google Drive
2. Run in your **🖥️ SSH Terminal**:
```bash
/opt/modules/library/cpu/anaconda/3-2024.10-1/bin/pip install --user gdown
/home/your.name/.local/bin/gdown "https://drive.google.com/uc?id=<FILE_ID>" -O /scratch1/your.name/datasets/thesis/wilddeepfake.zip
```
> To find the `<FILE_ID>`: Right-click the file in Google Drive → "Get link" → the ID is the long string between `/d/` and `/view` in the URL. For example: `https://drive.google.com/file/d/`**`1aBcDeFgH...`**`/view`

### Step 4A.2 — Extract All Google Drive Datasets
Once all zip files are uploaded to `/scratch1/your.name/datasets/thesis/`, extract them all at once:
```bash
cd /scratch1/your.name/EfficientNet-PyTorch/classification
/opt/modules/library/cpu/anaconda/3-2024.10-1/bin/python dataset_scripts/extract_datasets_coare.py
```

**What this does:** Goes through every `.zip` file, extracts it to the `extracted/videos-zip/` folder, and **immediately deletes the zip file** to save disk space. Do not worry — your original Google Drive files are untouched.

**What you should see:** It prints the name of each zip file as it processes it. When done it says `ALL DATASETS SUCCESSFULLY EXTRACTED!`

---

## PART 4B — Video Datasets from HuggingFace

> These three datasets are downloaded directly from HuggingFace. Pre-written SLURM scripts handle the entire download + frame extraction automatically. You just need to submit the jobs and wait.

**What is a SLURM job?** COARE uses a job queue system called SLURM. Instead of running a heavy Python script directly on the login node (which would get you kicked off), you submit a "job script" that COARE automatically runs on a dedicated compute node. Jobs are submitted with `sbatch` and you can monitor them with `squeue`.

### Step 4B.1 — Submit All Three Download Jobs
Run ALL three commands back to back in your **🖥️ SSH Terminal**:
```bash
cd /scratch1/your.name/EfficientNet-PyTorch/classification
sbatch slurm_scripts/download_deepaction.slurm   # AI-generated human motion videos
sbatch slurm_scripts/download_sdfvd.slurm        # Multiple AI generator videos
sbatch slurm_scripts/download_genvidbench.slurm  # GenVidBench benchmark (evaluation only, NOT training)
```

**What you should see** after each `sbatch` command:
```
Submitted batch job 548XXX
```
That number is your Job ID. Write it down!

### Step 4B.2 — Check That They Are Running
```bash
squeue -u your.name
```
You should see a table showing your three jobs with status **`R`** (Running). If you see **`PD`** (Pending), they are just waiting in the queue — this is normal. They will start automatically when a compute node is available.

### Step 4B.3 — Monitor Download Progress
The download progress bars from HuggingFace appear in the `.err` file (confusingly, they're not errors — COARE just routes all non-print output there):
```bash
cat dl_deepaction_<JOBID>.err
```
The frame extraction progress appears in the `.out` file:
```bash
cat dl_deepaction_<JOBID>.out
```
To watch it live (refreshes automatically):
```bash
tail -f dl_deepaction_<JOBID>.out
```
Press `Ctrl + C` to stop watching without canceling the job.

> **These jobs can run for several hours, especially GenVidBench (large RAR archives). You can close your terminal and the jobs will keep running on COARE.**

---

## PART 4C — DFDC Sample Dataset (Kaggle)

**What it is:** The Deepfake Detection Challenge sample — a subset of face-swap deepfake videos from Facebook's competition.

> **Faster option:** Ask Nathaniel to copy the already-extracted `dfdc_frames` and `dfdc_faces` folders directly to your scratch directory using `cp -r`.

If you need to download it yourself:

### Step 4C.1 — Create a Kaggle API Key
1. Go to https://kaggle.com and log in
2. Click your profile picture (top right) → **Settings**
3. Scroll down to **API** section → Click **"Create New API Token"**
4. A file called `kaggle.json` will download to your laptop

### Step 4C.2 — Upload the API Key to COARE
From **🪟 PowerShell**:
```powershell
scp -i C:\Users\<YourName>\.ssh\id_rsa_coare C:\Users\<YourName>\.kaggle\kaggle.json your.name@saliksik.asti.dost.gov.ph:/home/your.name/.kaggle/kaggle.json
```

Then in your **🖥️ SSH Terminal**, set the correct security permissions:
```bash
chmod 600 /home/your.name/.kaggle/kaggle.json
```

### Step 4C.3 — Download DFDC Sample
```bash
module load anaconda/3-2024.10-1
/opt/modules/library/cpu/anaconda/3-2024.10-1/bin/pip install --user kaggle "numpy<2.0.0"

/home/your.name/.local/bin/kaggle datasets download \
    -d deepfake-detection-challenge/dfdc-deepfake-detection-challenge-sample \
    -p /scratch1/your.name/datasets/thesis/
```

### Step 4C.4 — Extract DFDC
```bash
cd /scratch1/your.name/EfficientNet-PyTorch/classification
/opt/modules/library/cpu/anaconda/3-2024.10-1/bin/python extract_datasets_coare.py
```

---

## PART 5 — Preprocessing DFDC and FaceForensics++ (Face Cropping)

> **Skip this entire Part 5 if you are only running the prototype or Phase 3 experiments.** This is only needed for the Baseline Replication experiment.

The DFDC and FaceForensics++ datasets contain full video frames. For the **baseline replication** experiment, we need to crop only the face regions from each frame (just like the baseline study did). This uses a face detector called **MTCNN**.

**This is a two-step process:**

### Step 5.1 — Extract Raw Frames from DFDC Videos
The DFDC sample comes with raw `.mp4` video files. This script reads each video, saves every frame as a `.jpg` image, and then deletes the video file to save space:

```bash
cd /scratch1/your.name/EfficientNet-PyTorch/classification
/opt/modules/library/cpu/anaconda/3-2024.10-1/bin/python preprocess_dfdc_coare.py
```

**What you should see:** Progress counter like `Processed 10/400 videos...` finishing with `DFDC PREPROCESSING COMPLETE!`

### Step 5.2 — Crop Face Regions Using MTCNN (GPU Job)
This step uses the GPU to run face detection on every frame. Submit it as a SLURM job:

```bash
cd /scratch1/your.name/EfficientNet-PyTorch/classification
sbatch preprocess_faces.slurm
```

**What you should see:**
```
Submitted batch job 5XXXXX
```
Check it is running:
```bash
squeue -u your.name
# Look for a job named "extract_faces" with status R
```

> ⚠️ **This takes 6–12 hours.** You CANNOT run any other GPU training jobs while this is running. Wait for it to finish first (it will disappear from `squeue` when done).

---

## PART 6 — Running the Training Experiments

> **Before submitting any training job:** Make sure the datasets needed for that experiment are fully extracted on COARE (check using `ls /scratch1/your.name/datasets/extracted/videos-zip/`).

### 🧪 Experiment A — Baseline Replication
Trains ONLY on face-cropped DFDC + FaceForensics++ frames. This replicates the methodology of the baseline study so we have a controlled comparison point.

**Requires:** Part 5 to be fully complete (both `dfdc_faces` and `ff_faces` folders must exist).\

> ⚠️ **This script is now located in the `slurm_scripts/` subfolder. You MUST include the subfolder name when submitting:**

```bash
cd /scratch1/your.name/EfficientNet-PyTorch/classification
sbatch slurm_scripts/train_exp2_enhanced_arch_baseline_data.slurm
```

> **⚠️ GPU Queue Note:** If you see it stuck as **`PD`** for more than 1 hour, the GPU nodes are all occupied. You can check congestion using `squeue -p gpu`. Try again later or ask Nathaniel.

---

### 🧪 Experiment 1 — Baseline Architecture on Baseline Datasets (DFDC + FF++)
The "old way" approach — Vanilla EfficientNet-B4 with Mean Pooling only, trained on face-cropped DFDC and FaceForensics++ frames. This is the control experiment to show what happens when you use a simple architecture on traditional deepfake datasets.

**Requires:** Part 5 to be fully complete (both `dfdc_faces` and `ff_faces` folders must exist).

```bash
cd /scratch1/your.name/EfficientNet-PyTorch/classification
sbatch slurm_scripts/train_exp1_baseline_arch_baseline_data.slurm
```

---

### 🧪 Experiment 3 — Baseline Architecture on New Datasets
Vanilla EfficientNet-B4 trained on modern AI video datasets (Kling, Sora, CivitAI, etc.) without TSM or MHSA. Shows what the old architecture achieves on modern generators.

```bash
cd /scratch1/your.name/EfficientNet-PyTorch/classification
sbatch slurm_scripts/train_exp3_baseline_arch_new_data.slurm
```

---

### 🧪 Experiment 4 — Enhanced Architecture (TSM + MHSA) on New Datasets
The full proposed model — EfficientNet-B4 + TSM + MHSA — trained on all modern AI video datasets. This is the flagship experiment for the thesis.

**Requires:** ALL new datasets to be extracted (`wilddeepfake`, `synth_vid_detect`, `civitai_scraped`, `hailuo_scraped`, `kling_scraped`, `deepaction`, `sdfvd`).

```bash
cd /scratch1/your.name/EfficientNet-PyTorch/classification
sbatch slurm_scripts/train_exp4_enhanced_arch_new_data.slurm
```

---

## PART 7 — Monitoring Your Jobs

> ## 🚨 CRITICAL: Submitting a Job Does NOT Mean It Worked
> This is the most common mistake beginners make with SLURM. When you see `Submitted batch job 548XXX`, the job is **queued**, not done. You MUST always verify it **actually ran** by following the steps below. A job that finishes in under 2 minutes almost certainly crashed — real training runs take hours.

### Step 1 — Check all your active jobs at any time:
```bash
squeue -u your.name
```
You will see a table. The **ST** column shows the status:
- `R` — Running ✅ (the job is actively computing on a node)
- `PD` — Pending (waiting in queue, will start automatically)
- `CG` — Completing (almost finished)
- *(Absent from list)* — Finished — could be **success or crash**, check the log immediately!

### Step 2 — Check the output log to confirm it actually trained:
```bash
cat exp2_<JOBID>.out
```
A **successful run** will show epoch-by-epoch lines like this somewhere in the middle of the file:
```
[2026-09-19 12:15:18] Epoch [01/15] (412.3s) | Train Loss: 0.4821 Acc: 74.1% | Val Loss: 0.3901 Acc: 82.3% F1: 0.7900 AUC: 0.8743
```

A **crashed run** will end immediately after dataset loading with `Training Complete!` but show **no epoch lines**. If you see this, check the `.err` file next.

### Step 3 — Always check the error log:
```bash
cat exp2_<JOBID>.err
```
If you see a Python `Traceback` or any line starting with `Error:`, the job crashed. Copy the error message and fix the script, then re-submit.

> 💡 **HuggingFace download progress bars** (e.g., `100%|████████| 74.4M/74.4M`) appear in the `.err` file — that is normal and not a real error!

### Watch live training output (per-epoch progress):
```bash
tail -f exp2_<JOBID>.out
```
Press `Ctrl + C` to stop watching. The job keeps running in the background even if you disconnect.

> 💡 **If your SSH session closes mid-training:** Don't panic! The SLURM job runs on the compute node completely independently of your SSH session. Reconnect to COARE and run `squeue -u your.name` to confirm it is still running.

### View the detailed training log with timestamps:
Every training run automatically saves a `console.log` inside its output folder.
```bash
# First, find the run ID:
ls /scratch1/your.name/datasets/output_video/
# It looks like: video_tsm_mhsa_20260919_021500

# Then view the log:
cat /scratch1/your.name/datasets/output_video/video_tsm_mhsa_20260919_021500/console.log
```

### Cancel a job you don't want anymore:
```bash
scancel <JOBID>
```

---

## PART 8 — Resuming a Crashed or Interrupted Job

Training jobs may be interrupted due to time limits or node failures. Every epoch, the training script automatically saves a `resume.pth` checkpoint, so you can pick up exactly where you left off.

### Step 8.1 — Find the run ID
```bash
ls /scratch1/your.name/datasets/output_video/
# Example: video_tsm_mhsa_20260919_021500
```

### Step 8.2 — Edit the SLURM script to add the `--resume` flag
Open the training SLURM script (e.g. `train_video_prototype.slurm`) in `nano`, which is a simple text editor:
```bash
nano /scratch1/your.name/EfficientNet-PyTorch/classification/train_video_prototype.slurm
```
Find the Python command inside the file and add one more line at the end before the `\` closing:
```bash
    --resume /scratch1/your.name/datasets/output_video/video_tsm_mhsa_20260919_021500 \
```
Press `Ctrl + O` to save, then `Ctrl + X` to exit nano.

### Step 8.3 — Resubmit
```bash
cd /scratch1/your.name/EfficientNet-PyTorch/classification
sbatch train_video_prototype.slurm
```
The training will continue from where it stopped, and all logs will be appended to the SAME `console.log` file.

---

## PART 9 — Downloading Results Back to Your Laptop

After training finishes, download the output folder (containing the model checkpoint and training logs) to your laptop:

```powershell
# Run this in your 🪟 PowerShell
scp -i C:\Users\<YourName>\.ssh\id_rsa_coare -r your.name@saliksik.asti.dost.gov.ph:/scratch1/your.name/datasets/output_video/ C:\path\to\EfficientNet-PyTorch\classification\output_video\
```

---

## ❌ Common Errors and How to Fix Them

| Error message | What went wrong | How to fix it |
|---|---|---|
| `Invalid qos specification` | Wrong QoS name in the SLURM file | For `gpu` partition use `gpu-p40_default`. For `gpu_a100` use `gpu-a100_default`. |
| `QOSMaxCpuPerUserLimit` | You already have a GPU job running | You can only run one GPU job at a time. Wait for the current one to finish. |
| `ImportError: numpy.core.multiarray` | NumPy was accidentally upgraded to version 2.x | All pip install commands must include `"numpy<2.0.0"` at the end. |
| `RuntimeError: Dataset scripts are no longer supported` | Outdated HuggingFace loading code | Use `snapshot_download()` instead of `load_dataset(trust_remote_code=True)`. Already fixed in our scripts. |
| `Connection reset` | SSH timed out | Just reconnect. Your Slurm jobs keep running even when you are disconnected. |
| `Permission denied (publickey)` | SSH key missing or wrong path | Make sure `id_rsa_coare` is at `C:\Users\<YourName>\.ssh\`. Ask Nathaniel for the key file. |

---

## 📋 Quick Command Reference

| Task | Command (run in 🖥️ SSH Terminal) |
|---|---|
| Check all active jobs | `squeue -u your.name` |
| Submit a job | `sbatch <script>.slurm` |
| Cancel a job | `scancel <JOBID>` |
| Watch live output | `tail -f <logfile>_<JOBID>.out` |
| Check errors | `cat <logfile>_<JOBID>.err` |
| List dataset folders | `ls /scratch1/your.name/datasets/extracted/videos-zip/` |
| Check disk usage | `du -sh /scratch1/your.name/datasets/` |
