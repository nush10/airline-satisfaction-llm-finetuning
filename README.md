✈️ Airline Passenger Satisfaction — LLM Fine-Tuning with DistilBERT

📘 Project Overview

This project fine-tunes DistilBERT, a transformer-based encoder model, to classify passenger feedback as Satisfied or Neutral/Dissatisfied.
The model learns from structured airline survey data converted into natural-language sentences describing each passenger’s experience (Wi-Fi, seat comfort, delays, etc.).

It demonstrates a full LLM fine-tuning pipeline — from dataset curation and preprocessing to model training, evaluation, error analysis, and real-time inference.

🧠 Dataset

Source: Kaggle – Airline Passenger Satisfaction

Records: ~130 k rows (103 k train + 26 k test)

Task: Binary text classification (satisfied vs neutral or dissatisfied)

Format: Converted structured fields into descriptive sentences (e.g.,
“Passenger is a 45-year-old loyal customer traveling in Business class with Wi-Fi rating 5 and zero delays.”)

🛠️ System Setup
Option 1: Run on Google Colab (Recommended)

Open the main notebook
→ FineTuning_DistilBERT_AirlineSatisfaction.ipynb

Mount Google Drive

from google.colab import drive
drive.mount('/content/drive')


Create the project directory structure

/content/drive/MyDrive/llm-finetune/
├── data/
│   ├── raw/              ← Kaggle dataset (train.csv, test.csv)
│   └── processed/        ← Cleaned & split CSVs
├── outputs/
│   ├── checkpoints/      ← Saved model checkpoints
│   ├── metrics/          ← Accuracy, F1, confusion matrix JSONs
│   └── logs/
├── report/figs/          ← Charts & visualizations
└── FineTuning_DistilBERT_AirlineSatisfaction.ipynb


Download Kaggle Dataset

Upload your kaggle.json API key to Colab.

Run:

!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d teejmahal20/airline-passenger-satisfaction -p /content/drive/MyDrive/llm-finetune/data/raw
!unzip /content/drive/MyDrive/llm-finetune/data/raw/airline-passenger-satisfaction.zip -d /content/drive/MyDrive/llm-finetune/data/raw


Install dependencies

!pip install transformers==4.57.1 datasets==4.0.0 evaluate==0.4.6 accelerate==1.11.0 torch==2.8.0


Run the notebook cells sequentially

Part 1: Data preparation → creates processed train/val/test CSVs

Part 2: Model loading & justification

Part 3: Fine-tuning with Hugging Face Trainer

Part 4: Hyperparameter optimization (learning-rate + batch size)

Part 5: Model evaluation & baseline comparison

Part 6: Error analysis + pattern mining

Part 7: Inference pipeline demonstration

Outputs & artifacts

📂 All metrics, checkpoints, and plots are stored under
/content/drive/MyDrive/llm-finetune/outputs/

Option 2: Run Locally (on VS Code / Jupyter)

Clone the repo

git clone https://github.com/nush10/airline-satisfaction-llm-finetuning.git
cd airline-satisfaction-llm-finetuning


Create a virtual environment

python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt


(Optional) Download the dataset manually from Kaggle link

and place it under:

data/raw/train.csv
data/raw/test.csv


Run the notebook locally

jupyter notebook FineTuning_DistilBERT_AirlineSatisfaction.ipynb


or open it directly in VS Code’s Jupyter extension.

📊 Results Summary
Metric	Baseline (TF-IDF + LR)	Fine-Tuned DistilBERT	Δ Improvement
Accuracy	0.7889	0.9596	+ 0.1707
F1 (Macro)	0.7860	0.9588	+ 0.1728

The model converged within ~2 epochs using early stopping.

Mixed-precision (fp16) training reduced GPU memory usage by 35 %.

Inference latency ≈ 5 ms per sample (Tesla T4).

Visualizations (confusion matrix, ROC curve, class distribution) are saved under
/report/figs/.

💻 Reproducibility Notes

Environment: Python 3.10 | PyTorch 2.8 (CUDA 12.6) | Transformers 4.57

Hardware: Tesla T4 GPU (16 GB VRAM)

Random Seeds: 42 for NumPy, PyTorch, and Hugging Face datasets

Checkpoints: outputs/checkpoints/run_lr2e5_bs16_len192/checkpoint-23380

To reproduce, ensure the same environment or run directly in Colab with the provided notebook.

🧩 Lessons Learned

Data quality matters: transforming structured data into natural text greatly improved context understanding.

Efficiency > size: DistilBERT offered near-BERT accuracy at half the computation cost.

Explainability through error analysis: misclassified samples revealed sensitivity to mid-range ratings (3 stars).

MLOps mindset: systematic logging, checkpointing, and Drive-based storage made experiments fully reproducible.
