# 🧠 Mapping and Predicting Depression: Bridging Psychometric Networks with Implicit Behaviors

This project investigates the structural and behavioral dynamics of psychological distress, focusing specifically on depression. By framing the analysis within **Dual-Process Theory**, the study integrates explicit symptom evaluations (System 2 - controlled cognitive reflection) with implicit reaction times (System 1 - automatic reactions) to map and predict depression severity.

The project combines Explanatory Structural Modeling (Psychometric Network Analysis) with Predictive Machine Learning and Explainable AI (XAI).

### Authors
* Marco Rosato

### Core Challenge: Integrating Structure and Behavior
A primary challenge in psychological modeling is that traditional explicit questionnaires rely heavily on self-awareness, potentially missing automatic, implicit cognitive processes. 

To solve this, we implemented a dual-stage analytical strategy:
* **Structural Mapping:** We used the Psychometric Network Approach (PNA) to understand how psychological symptoms directly interact and trigger one another, treating mental disorders as complex systems rather than latent variables.
* **Behavioral Prediction:** We proved that *how fast* a user answers a question is highly predictive of their distress. By conducting an XGBoost ablation study, we quantified the exact predictive value of behavioral latencies over standard demographic baselines.

### Dataset
The project utilizes a public dataset from Open Psychometrics containing responses to the **Depression Anxiety Stress Scales (DASS-42)**.
* **Size:** 22,848 valid observations (after rigorous behavioral data cleaning).
* **Features:** Partitioned into three subsets: Explicit answers (0-3 Likert scale), Implicit reaction times (milliseconds, log-transformed), and Demographics/Personality (including TIPI Big Five scores).

### Project Workflow
The project is structured into three stages:

1. **Explanatory Modeling (Psychometric Network Approach)**
   * **Algorithm:** Graphical Lasso (L1 regularization) with 5-fold cross-validation.
   * **Execution:** Edges with partial correlations below 0.05 were thresholded to generate an interpretable network of explicit symptoms.
   * **Findings:** Identified "hopelessness" as the central hub of the depressive network, and demonstrated that stress symptoms act as a structural bridge connecting depression and anxiety clusters.

2. **Predictive Modeling (Ablation Study)**
   * **Algorithm:** XGBoost Regressor tuned via GridSearch, targeting the continuous Depression subscale score (0-42).
   * **Strategy:** An ablation study was conducted to test the Dual-Process Theory, comparing three models:
      * *Model A (Baseline):* Demographics + Personality traits.
      * *Model B (System 1):* Implicit reaction times only.
      * *Model C (Hybrid):* Full model combining all features.

3. **Explainable AI (XAI) & Interpretation**
   * **Techniques:** SHAP (SHapley Additive exPlanations) and Partial Dependence Plots (PDPs) were implemented to interpret the hybrid model.
   * **Insights:** SHAP values uncovered non-linear relationships and socio-demographic gradients.

### Key Results
* **Predictive Boost:** Integrating reaction times improved depression severity predictions by over **22%** compared to standard demographic and personality baselines.
* **Behavioral Latencies:** PDPs and SHAP revealed that *faster* response latencies to core affective symptoms (e.g., hopelessness, sadness) were strongly associated with higher predicted distress.
