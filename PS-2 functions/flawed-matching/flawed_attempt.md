# A Methodological Review of an Evolving Resume Matching System

## Abstract

This report documents the iterative development of a resume-to-job-description matching system. It presents a chronological analysis of four distinct Jupyter Notebooks, beginning with an initial flawed implementation and culminating in a methodologically sound, predictive model. The primary focus of this document is to dissect the critical error of data leakage, demonstrate its impact on model evaluation, and present a robust workflow that corrects this flaw. This journey serves as a case study in the importance of rigorous validation and the potential for misleading results in machine learning applications.

---

## 1. Initial Prototype: `flawed_matching.ipynb`

### 1.1. Methodology

The initial investigation aimed to create a complete, end-to-end classification system. The methodology was as follows:

1.  **Dataset:** The analysis utilized a dataset of 145 resumes (`Final.csv`) and a single job description for a 'React Native Developer'.
2.  **Feature Engineering:** Textual data from both resumes and the job description were preprocessed. Feature vectors were then generated using a **TF-IDF (Term Frequency-Inverse Document Frequency)** model. The similarity between each resume and the job description was calculated using **Cosine Similarity** on these TF-IDF vectors.
3.  **Target Labeling:** A `target` variable was created to enable supervised learning. This was done by calculating a `combined_similarity` score and applying a percentile-based threshold (the 70th percentile). Resumes above this threshold were labeled as 'Fit' (1), and those below were labeled as 'Not Fit' (0).
4.  **Model Training:** Several classification models, including **SVM, Logistic Regression, Random Forest, and XGBoost**, were trained on the generated features to predict the `target` label.

### 1.2. Results & Analysis

The models produced exceptionally high, and ultimately misleading, performance metrics. The **Random Forest and XGBoost models reported 100% accuracy and F1-scores**, while SVM and Logistic Regression also achieved near-perfect scores (96.6% accuracy).

### 1.3. Critique of Methodology

The perfect scores were a direct result of a critical methodological flaw: **data leakage**. The `target` variable was created *after* calculating similarity scores for the entire dataset. This means that the ground truth for the test set was determined by its own features, creating a circular dependency. The model was not learning to generalize; it was merely learning to reverse-engineer the thresholding rule that had been applied to the data it was being tested on.

### 1.4. Justification for Next Iteration

The results from this notebook were deemed invalid due to the data leakage. The 100% accuracy was an artifact of the flawed process, not a reflection of a successful model. The next iteration needed to employ more sophisticated text representation methods to see if the problem was related to the simplicity of TF-IDF, and to further refine the analysis.

---

## 2. Second Iteration: `146_flawed_new.ipynb`

### 2.1. Methodology

This notebook improved upon the first by incorporating more advanced NLP techniques:

1.  **Enhanced Feature Engineering:** The primary similarity metric was upgraded from TF-IDF to **S-BERT embeddings (`all-MiniLM-L6-v2`)**, which capture semantic context. **Jaccard Similarity** was added as a secondary, keyword-based metric.
2.  **Hybrid Scoring:** A `Combined_Score` was created by normalizing the S-BERT and Jaccard scores (to a 0-1 scale) and taking their average. This provided a more balanced measure of both semantic and lexical similarity.
3.  **Additional Features:** The model was enriched with keyword-based features, such as `keyword_count` and `skills_keyword_count`.

### 2.2. Results & Analysis

The analysis showed a moderate correlation of **0.6241** between S-BERT and Jaccard similarities, confirming they captured different aspects of the text. The ranking produced by the `Combined_Score` was more robust. However, when this score was used to create a `target` variable for the machine learning section, the **Random Forest model once again achieved 100% accuracy**.

### 2.3. Critique of Methodology

This iteration, while more sophisticated in its feature engineering, **repeated the exact same data leakage flaw**. The `target` variable was still created based on a global calculation of the `Combined_Score` across the entire dataset before the train-test split. The perfect accuracy score confirmed that the issue was not the choice of features (TF-IDF vs. S-BERT) but the fundamental workflow of the supervised learning task.

### 2.4. Justification for Next Iteration

It became clear that the classification approach itself was the problem. The immediate next step was to salvage the valid part of the analysis (the unsupervised ranking) and formally separate it from the flawed supervised learning experiment. This would create a clean, methodologically sound baseline.

---

## 3. First Corrective Action: `146_ranking_analysis_fixed.ipynb`

### 3.1. Methodology

This notebook represented a critical pivot. The corrective action was to:

1.  **Isolate the Valid Analysis:** Retain only the unsupervised parts of the previous notebook: data preprocessing, S-BERT and Jaccard similarity calculations, and the comparative analysis of the resulting rankings.
2.  **Remove the Flawed Section:** The entire machine learning section, which attempted to classify resumes based on the leaky target variable, was completely removed.
3.  **Document the Flaw:** A markdown cell was explicitly added to the end of the notebook, explaining the concept of data leakage and why the original ML analysis was invalid.

### 3.2. Results & Analysis

This notebook does not produce any machine learning metrics. Its output is a valid, unsupervised ranking of candidates based on similarity scores. It successfully identifies top candidates (e.g., Sarthak Thakral, Gonuguntla Udaya Kiran) based on a hybrid semantic and lexical score, which is a valuable outcome in itself.

### 3.3. Critique of Methodology

This approach is sound but limited. It provides a ranked list but does not yield a predictive model. It cannot, for instance, be deployed as an automated screening tool that has *learned* the characteristics of a good resume from the data. It is a static ranking system, not a predictive one.

### 3.4. Justification for Next Iteration

With a clean and valid ranking analysis established, the final step was to demonstrate how to build a machine learning model *correctly*. The goal was to create a model that could learn to approximate the results of our ranking system without succumbing to data leakage, thereby providing a truly predictive and automated solution.

---

## 4. Final Corrected Implementation: `Corrected_ML_Analysis.ipynb`

### 4.1. Methodology

This notebook presents the definitive, methodologically sound workflow:

1.  **Initial Feature Calculation:** S-BERT and Jaccard similarities are calculated for the entire dataset as a preliminary feature engineering step.
2.  **The Critical Split:** The dataset is immediately split into a training set (70%) and a test set (30%). **This is the most important step.** The test set is now isolated and held out.
3.  **Target Creation on Training Data ONLY:** The `Combined_Score` is calculated, and a threshold (the 70th percentile, which was **0.5316**) is determined using **only the training data**.
4.  **Application of the Rule:** This single, fixed threshold is then applied to both the training and test sets to create the `is_top_candidate` target variable. This correctly simulates a real-world scenario where a rule learned from past data is applied to new, unseen data.
5.  **Model Training and Evaluation:** The machine learning models are trained on the training set and evaluated on the truly unseen test set.

### 4.2. Results & Analysis

The results from this corrected workflow are realistic and trustworthy:

-   **No Perfect Scores:** The models no longer achieve 100% accuracy. The best-performing models (**Logistic Regression, Naive Bayes, and SVM**) achieved a high but believable **F1-score of 0.9655** and an accuracy of **97.7%**.
-   **Valid Feature Importance:** The feature importance analysis from the Random Forest model is now credible. It correctly identifies `sbert_similarity` (Importance: 0.497) and `jaccard_similarity` (Importance: 0.402) as the most influential features, which aligns with our analytical goals.

### 4.3. Conclusion

This iterative process highlights a critical lesson in applied machine learning. The initial allure of perfect accuracy scores was correctly identified as a symptom of data leakage. By systematically diagnosing the problem, isolating the valid components of the analysis, and redesigning the machine learning workflow around a strict train-test separation, we arrived at a final model that is both powerful and reliable. The `Corrected_ML_Analysis.ipynb` notebook now stands as a robust and valid blueprint for building a predictive resume matching system. The journey itself underscores that the integrity of the process is as important as the final results.
