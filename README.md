# 22i2065_AminaAnjum_A2-CS452
🧠 Legal Clause Similarity — Deep Learning Assignment 02

FAST NUCES, Islamabad - Department of Computer Science
Course: CS452 – Deep Learning
Instructors: Mahnoor Tariq & Dr. Qurat-ul-Ain
Due Date: 10 November 2025

📘 Overview

This project implements two deep learning baselines for identifying semantic similarity between legal clauses.
Legal clauses often express the same meaning in different words — this project builds neural models to detect when two clauses are semantically equivalent.

🎯 Objectives

Detect semantic similarity between two legal clauses.

Build two different architectures from scratch (no pre-trained transformers).

Evaluate and compare both models using standard NLP metrics.

🧩 Dataset

Source: Kaggle — Legal Clause Dataset

395 CSV files, each representing one clause category

Total clauses after cleaning: ≈150,881

Balanced positive (similar) and negative (dissimilar) pairs

Splits:

Train → 80% (100,000 pairs)

Validation → 10% (12,000 pairs)

Test → 10% (12,000 pairs)

🧠 Models Implemented
Baseline 1 — Siamese BiLSTM

Embedding: 40k vocab × 128 dim

Encoder: BiLSTM (128 units each direction) + GlobalMaxPooling

Head: |u−v| and u∘v concatenation → Dense(128→64→Sigmoid)

Parameters: 5.46M

Optimizer: Adam (lr=1e-3), Batch=256

Callbacks: EarlyStopping, ReduceLROnPlateau

Performance (Test Set)

Metric	Score
Accuracy	0.9988
Precision	0.9995
Recall	0.9980
F1-Score	0.9988
ROC-AUC	0.9994
PR-AUC	0.9996

Confusion Matrix: TP=6009, TN=5976, FP=3, FN=12
Training Time: ≈6 minutes (Tesla T4)

Baseline 2 — Siamese Attention Encoder

Embedding: 40k vocab × 128 dim

Encoder: BiGRU (128 units each direction) + Multi-Head Self-Attention (4 heads) + GlobalMaxPooling

Head: |u−v| and u∘v → Dense(128→64→Sigmoid)

Parameters: 5.47M

Optimizer: Adam (lr=1e-3), Batch=256

Callbacks: EarlyStopping, ReduceLROnPlateau

Performance (Test Set)

Metric	Score
Accuracy	0.9994
Precision	0.9998
Recall	0.9990
F1-Score	0.9994
ROC-AUC	0.9999
PR-AUC	0.9998

Confusion Matrix: TP=6015, TN=5978, FP=1, FN=6
Training Time: ≈8.3 minutes (Tesla T4)

⚖️ Comparative Summary
Model	Accuracy	F1-Score	ROC-AUC	PR-AUC	Train Time
BiLSTM	0.9988	0.9988	0.9994	0.9996	6.2 min
Attention Encoder	0.9994	0.9994	0.9999	0.9998	8.3 min

✅ Attention Encoder slightly outperforms BiLSTM due to its ability to focus on key legal terms while preserving contextual relationships.

📊 Visualizations

Figures are saved in /figures/:

bilstm_test_roc.png

bilstm_test_pr.png

attn_test_roc.png

attn_test_pr.png

Each shows the ROC and PR curves for the respective baseline.

🧪 How to Run
# Clone repository
git clone https://github.com/<your-username>/DL_Assignment2_LegalClauseSimilarity.git
cd DL_Assignment2_LegalClauseSimilarity

# Install requirements
pip install -r requirements.txt

# Run notebook
jupyter notebook main_notebook.ipynb


Requirements:

tensorflow>=2.15
keras>=3.0
numpy
pandas
matplotlib
scikit-learn


🧠 Key Takeaways

Siamese networks are highly effective for textual similarity tasks.

Adding attention improves contextual comprehension and performance.

Both baselines achieve >99% accuracy and AUC, proving strong generalization.

👩‍🏫 Acknowledgement

This repository is part of Assignment 02 - Deep Learning (CS452) under the supervision of Mahnoor Tariq and Dr. Qurat-ul-Ain, Department of Computer Science, FAST-NUCES Islamabad.
