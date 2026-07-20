# 💬 Hard & Soft-Label for Hate Speech Classification (BERT)

This project investigates the automatic classification of hate speech on social media by addressing the inherent subjectivity of toxic language. It challenges the standard supervised learning approach of using discrete "hard labels," proposing a probabilistic "soft-label" training instead, using BERT to capture linguistic ambiguity and human uncertainty.

### Author
* Marco Rosato

### Core Challenge: The Polarization Problem
A primary challenge in hate speech classification is the severe lexical overlap between genuine hate speech and colloquial offensive slang (e.g., profanity used casually). Furthermore, exploratory analysis reveals a profound lack of consensus among human annotators on these ambiguous tweets. 

Training a deep learning model exclusively on deterministic "hard labels" (majority vote) forces the network to become artificially polarized and overconfident, generating highly confident false positives on disputed text.

To solve this, we implemented a **Soft-Label Training Paradigm**:
* Instead of predicting a single discrete class, the target variable was redefined as a probability distribution derived from the raw human annotator votes (e.g., a tweet with 1 Hate vote, 2 Offensive votes, and 0 Neither votes becomes `[0.33, 0.67, 0.00]`).
* The network was trained via a custom cross-entropy loss function to perfectly replicate the exact proportion of human disagreement, effectively rewarding the model for expressing uncertainty on ambiguous text.

### Dataset
The project utilizes the **Hate Speech and Offensive Language dataset** (Davidson et al., 2017), consisting of 24,783 tweets.
* **Classes:** Hate Speech (0), Offensive Language (1), Neither (2).
* **Imbalance:** The dataset is heavily skewed toward the "Offensive" class (19,190 tweets), with "Hate" being a severe minority class (only 1,430 tweets).
* **Preprocessing:** Custom cleaning preserved contextual metadata by mapping specific elements to standardized tokens (`user`, `url`, `retweet`).

### Project Workflow
The project is structured into several key analytical stages:

1. **Linguistic Profiling & Baselines**
   * **EDA:** Part-of-Speech (POS) tagging revealed syntactic differences (Hate speech uses more adjectives and direct `user` mentions, while Neither uses more proper nouns and `url` links).
   * **Baselines:** Evaluated a traditional TF-IDF + Logistic Regression model and a Frozen BERT Feature Extraction approach. Both struggled with contextual disambiguation.

2. **Hard-Label Fine-Tuning (LoRA vs. Full FT)**
   * **Parameter-Efficient Fine-Tuning (LoRA):** Optimized only the internal transformer projection matrices (query, key, value), outperforming frozen embeddings.
   * **Full Fine-Tuning:** Updated all 109M parameters of `bert-base-cased`. While achieving high F1-scores, probability distribution analyses revealed that the model became "confidently wrong," artificially polarizing predictions to 0.0 or 1.0.

3. **Soft-Label Training & Threshold Optimization**
   * Trained the full BERT architecture on soft labels without class weighting to prevent distribution distortion.
   * Addressed the severe class imbalance post-training by optimizing the classification decision boundary (threshold = 0.34 for the Hate class) on validation set.

4. **Explainability & Uncertainty Evaluation**
   * **Metrics:** Evaluated model calibration using Kullback-Leibler (KL) Divergence and Mean Absolute Error (MAE) to measure the informational distance between human and model distributions.
   * **SHAP Analysis:** Utilized SHapley Additive exPlanations to visualize how the soft-label model distributes probability mass across the entire syntax rather than allowing a single toxic subword to monopolize the decision.

### Key Results
* **Performance:** The soft-label BERT model with an optimized threshold performed better than the traditional hard-label fine-tuning on the minority Hate class (F1-score: 0.50 vs. 0.48) while maintaining excellent performance on the majority classes.
* **Capturing Human Doubt:** The soft-label model successfully mirrored human uncertainty. On disputed tweets, it achieved near-zero KL Divergence, proving the network learned the exact boundaries of human disagreement.
* **Conclusion:** Modeling subjective human language requires probabilistic representations of uncertainty. Incorporating human doubt directly into the loss function improves both model calibration and discrete classification boundaries.
