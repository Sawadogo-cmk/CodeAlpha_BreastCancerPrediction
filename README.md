\# 🎗️ Breast Cancer Prediction



> An end-to-end Machine Learning project that predicts breast tumor malignancy from cell nuclei measurements, with a live interactive demo.



\[!\[Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://www.python.org/)

\[!\[scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?style=for-the-badge\&logo=scikitlearn\&logoColor=white)](https://scikit-learn.org/)

\[!\[XGBoost](https://img.shields.io/badge/XGBoost-3.4-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io/)

\[!\[Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)](https://streamlit.io/)

\[!\[License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)



\---



\## 📸 App Preview



!\[Breast Cancer Risk Predictor - Screenshot](images/app\_preview.png)



\---



\## 🚀 Live Demo



👉 \*\*\[Try the interactive app here](https://sawadogo-breast-cancer.streamlit.app)\*\*



Enter 10 key cell nuclei measurements and get an instant malignancy risk assessment powered by a Logistic Regression model.



\---



\## 📖 Overview



This project was developed as part of the \*\*Machine Learning Internship at \[CodeAlpha](https://www.codealpha.tech)\*\*.



The goal is to predict whether a breast tumor is \*\*malignant\*\* or \*\*benign\*\* from features computed on digitized images of a fine needle aspirate (FNA) of a breast mass, a diagnostic task where early detection greatly improves outcomes.



Four classification models were trained and compared:



\- \*\*Logistic Regression\*\*

\- \*\*Support Vector Machine (RBF kernel)\*\*

\- \*\*Random Forest\*\*

\- \*\*XGBoost\*\*



\---



\## 📊 Results



| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |

|---|---|---|---|---|---|

| \*\*Logistic Regression\*\* ⭐ | \*\*0.9825\*\* | \*\*0.9861\*\* | \*\*0.9861\*\* | \*\*0.9861\*\* | \*\*0.9954\*\* |

| SVM (RBF) | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9950 |

| Random Forest | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9932 |

| XGBoost | 0.9474 | 0.9459 | 0.9722 | 0.9589 | 0.9927 |



> ⭐ \*\*Logistic Regression\*\* was selected as the final model. With 98.25% accuracy and 0.9954 ROC-AUC, it matches or beats the more complex models, which suggests the classes are close to linearly separable in this feature space.



> ℹ️ \*\*Limitations:\*\*

> - Metrics come from a single stratified 80/20 split, so the test set contains only 114 patients. One patient is about 0.9 points of accuracy, so the gap between Logistic Regression and SVM (identical except for a 0.0004 difference in ROC-AUC) is not meaningful.

> - In this dataset `1 = benign` and `0 = malignant`, so precision and recall above are computed with \*\*benign as the positive class\*\*. A false positive here is a \*malignant tumor predicted as benign\*, the costlier error in screening. Reporting recall on the malignant class would be more informative for a clinical use case.



\### Key Insights: Feature Importance



| Feature | Meaning |

|---|---|

| \*\*Worst concave points\*\* | Mean of the 3 largest values of the number of concave portions of the contour |

| \*\*Worst perimeter\*\* | Mean of the 3 largest nucleus perimeters |

| \*\*Worst radius\*\* | Mean of the 3 largest nucleus radii |

| \*\*Mean concave points\*\* | Average number of concave portions of the contour |

| \*\*Worst area\*\* | Mean of the 3 largest nucleus areas |



Larger nuclei with irregular, concave contours are associated with malignancy, which is consistent with these results. Note that radius, perimeter and area are highly correlated, so the importance assigned to each one individually should be read with caution.



\---



\## 🖼️ Visualizations



\### Model comparison



!\[Model Performance Comparison](images/model\_comparison.png)



\### ROC curves



!\[ROC Curves](images/roc\_curves.png)



\### Feature importance



!\[Feature Importance](images/feature\_importance.png)



\---



\## 📁 Dataset



\*\*Breast Cancer Wisconsin (Diagnostic)\*\* (UCI ML Repository, loaded through scikit-learn)



\- \*\*569 patients\*\* with FNA samples

\- \*\*30 numerical features\*\* derived from cell nuclei: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension (each as \*mean\*, \*standard error\* and \*worst\* variants, where \*worst\* is the mean of the three largest values)

\- \*\*Target\*\*: binary (0 = malignant, 1 = benign)

\- \*\*Class distribution\*\*: 62.7% benign (357) / 37.3% malignant (212)

\- \*\*Zero missing values\*\*



\### Challenges addressed



\- \*\*Feature scaling\*\*: feature values range from about 0.001 to over 4000 (area), so `StandardScaler` was essential for Logistic Regression and SVM.

\- \*\*High dimensionality and correlation\*\*: 30 features, many strongly correlated, which makes it a good test for both linear and non-linear models.

\- \*\*Mild class imbalance\*\*: 63/37, handled with a stratified split. No resampling was needed.



\---



\## 🛠️ Tech Stack



| Layer | Tools |

|---|---|

| \*\*Language\*\* | Python 3.13 |

| \*\*ML\*\* | scikit-learn, XGBoost |

| \*\*Data\*\* | pandas, NumPy |

| \*\*Visualization\*\* | Matplotlib, Seaborn |

| \*\*Web app\*\* | Streamlit |

| \*\*Deployment\*\* | Streamlit Community Cloud |



\---



\## 📂 Project Structure



```text

CodeAlpha\_BreastCancerPrediction/

├── .streamlit/

│   └── config.toml                    # Light theme with pink accent

├── images/

│   ├── app\_preview.png

│   ├── model\_comparison.png

│   ├── roc\_curves.png

│   └── feature\_importance.png

├── app.py                             # Streamlit web application

├── breast\_cancer\_prediction.ipynb     # Full ML analysis notebook

├── model.pkl                          # Trained Logistic Regression model

├── scaler.pkl                         # StandardScaler

├── features.pkl                       # Feature names for alignment

├── medians.pkl                        # Feature medians for defaults

├── results.csv                        # Model comparison table

├── requirements.txt

└── README.md

```



The dataset is loaded with `sklearn.datasets.load\_breast\_cancer()`, so no `data/` folder is needed.



\---



\## 🚀 Run Locally



\### 1. Clone the repository



```bash

git clone https://github.com/Sawadogo-cmk/CodeAlpha\_BreastCancerPrediction.git

cd CodeAlpha\_BreastCancerPrediction

```



\### 2. Create and activate a virtual environment



```bash

python -m venv .venv



\# Windows

.venv\\Scripts\\activate



\# macOS / Linux

source .venv/bin/activate

```



\### 3. Install dependencies



```bash

pip install -r requirements.txt

```



\### 4. Launch the Streamlit app



```bash

streamlit run app.py

```



The app opens at <http://localhost:8501>.



\---



\## 🧠 Model Pipeline



1\. \*\*Data loading\*\*: `sklearn.datasets.load\_breast\_cancer()`, no download required

2\. \*\*Split\*\*: 80% train / 20% test (stratified)

3\. \*\*Scaling\*\*: `StandardScaler` fit on the training data only (no leakage)

4\. \*\*Training\*\*: 4 models compared on 5 metrics

5\. \*\*Selection\*\*: Logistic Regression (best ROC-AUC, tied best F1)

6\. \*\*Deployment\*\*: serialized with `joblib`, served via Streamlit



\---



\## 🎯 Future Improvements



\- 🔁 \*\*K-fold cross-validation\*\* for robust estimates and a fairer model comparison

\- 🩺 \*\*Report recall and a confusion matrix on the malignant class\*\*, and tune the decision threshold to minimize missed malignancies

\- ⚙️ \*\*Hyperparameter tuning\*\* with Optuna

\- 📈 \*\*Probability calibration\*\* so risk scores can be read as probabilities

\- 🌐 \*\*REST API with FastAPI\*\* for integration into clinical systems

\- 🧪 \*\*Additional datasets\*\* (e.g. CBIS-DDSM) for external validation



\---



\## ⚠️ Disclaimer



This is a machine learning demonstration project, \*\*not a medical device\*\*. Predictions should never replace professional medical advice. Always consult a qualified oncologist for clinical decisions.



\---



\## 👨‍💻 Author



\*\*Sawadogo Noé\*\*, Machine Learning Intern at CodeAlpha



\- 🐙 GitHub: \[@Sawadogo-cmk](https://github.com/Sawadogo-cmk)

\- 💼 LinkedIn: Sawadogo Noé

\- 🌐 Live demo: \[sawadogo-breast-cancer.streamlit.app](https://sawadogo-breast-cancer.streamlit.app)



\---



\## 🙏 Acknowledgements



\- \*\*CodeAlpha\*\* for the internship program

\- \*\*UCI Machine Learning Repository\*\* for the Breast Cancer Wisconsin (Diagnostic) dataset

\- Original creators: Dr. William H. Wolberg, W. Nick Street and Olvi L. Mangasarian (University of Wisconsin)



\---



\## 📜 License



This project is licensed under the MIT License. See the \[LICENSE](LICENSE) file for details.



<p align="center"><i>Built with 🎗️ and 🐍 during the CodeAlpha Machine Learning Internship</i></p>

