✈️ Airline Passenger Satisfaction — LLM Fine-Tuning with DistilBERT

📘 Project Overview

This project fine-tunes DistilBERT, a transformer-based encoder model, to classify airline passenger feedback as Satisfied or Neutral/Dissatisfied.
It converts structured airline survey data (seat comfort, Wi-Fi, delays, etc.) into natural-language sentences and trains a contextual language model to understand satisfaction sentiment.

The project demonstrates an end-to-end fine-tuning pipeline — covering data preparation, model training, evaluation, error analysis, and deployment-ready inference.

🧠 Dataset

Source: Kaggle – [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

Records: ~130k samples

Task: Binary text classification (satisfied vs neutral or dissatisfied)

Transformation: Structured features were rewritten into human-readable text such as:

“Passenger is a 45-year-old loyal customer traveling in Business class with Wi-Fi rating 5 and zero delays.”

⚙️ System Setup
🧩 Option 1: Run on Google Colab (Recommended)

Open the notebook:
FineTuning_DistilBERT_AirlineSatisfaction.ipynb

Mount your Google Drive:

from google.colab import drive
drive.mount('/content/drive')


All outputs, processed data, and checkpoints are automatically saved under your Drive folder:
📂 [Google Drive Folder (Project Files)](https://drive.google.com/drive/folders/1hoGPACebhM7Dh82RgzdyrQf-y1diLVcr?usp=sharing)

This folder includes:

data/ → raw + processed datasets

outputs/ → checkpoints, metrics, logs

report/figs/ → generated charts and visualizations

final trained model under outputs/checkpoints/run_lr2e5_bs16_len192/checkpoint-23380

Install dependencies (if needed):

!pip install transformers==4.57.1 datasets==4.0.0 evaluate==0.4.6 accelerate==1.11.0 torch==2.8.0


Run all notebook cells sequentially — each section handles a specific assignment part (data prep → training → evaluation → inference).

🖥️ Option 2: Run Locally on VS Code / Jupyter

Clone the repository:

git clone https://github.com/nush10/airline-satisfaction-llm-finetuning.git
cd airline-satisfaction-llm-finetuning


Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt


Open the notebook directly in VS Code or Jupyter Notebook.

(Optional) Mount Google Drive or download the dataset manually from Kaggle.

📊 Results Summary
Metric	Baseline (TF-IDF + LR)	Fine-Tuned DistilBERT	Δ Improvement
Accuracy	0.7889	0.9596	+ 0.1707
F1 (Macro)	0.7860	0.9588	+ 0.1728

Converged within ~2 epochs (early stopping).

Mixed-precision (fp16) training reduced GPU memory use by 35%.

Inference latency ≈ 5 ms per sample on Tesla T4 GPU.

Visualizations and charts (confusion matrix, class distribution, comparison plots) are saved under report/figs/.

💻 Reproducibility

Environment: Python 3.10, PyTorch 2.8 (CUDA 12.6), Transformers 4.57

Hardware: Tesla T4 GPU (16 GB)

Random Seed: 42 (NumPy, PyTorch, Hugging Face)

All artifacts stored on Drive: [LLM-Finetune Outputs Folder](https://drive.google.com/drive/folders/1hoGPACebhM7Dh82RgzdyrQf-y1diLVcr?usp=sharing)

🧩 Lessons Learned

Data transformation is key: Converting structured survey data into natural text significantly improved model contextual understanding.

Efficiency matters: DistilBERT achieved near-BERT accuracy with 40% fewer parameters.

Explainability: Error analysis exposed cases where borderline reviews confused the classifier, inspiring ideas for interpretability improvements.

MLOps mindset: Using Google Drive for checkpointing, logging, and reproducible storage ensured smooth experimentation and collaboration.
