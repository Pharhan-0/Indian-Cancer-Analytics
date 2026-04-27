# 🔬 Advanced Cancer Mortality Analytics: Indian PBCR Data

> **Published IEEE Paper + Full Python Pipeline**  
> Demographic Disparities and Predictive Modeling using Indian Population-Based Cancer Registry (PBCR) Data

---

## 📄 Overview

This repository contains the research paper and complete Python source code for a five-objective computational analytics pipeline applied to the **ICMR PBCR Mortality Dataset**. The study analyzes cancer mortality patterns across India, examining demographic disparities, geographic hotspots, healthcare access, and machine learning-based cancer type prediction.

**Author:** Pharhan Anzum Haque  
**Institution:** Lovely Professional University, Phagwara, India  
**Degree:** B.Tech Computer Science and Engineering  

---

## 📁 Repository Structure

```
├── Project375.py                  # Full Python analytics pipeline
├── IEEECancerPaperPharhan.pdf     # IEEE paper
└── README.md
```

---

## 🎯 Research Objectives

| # | Objective | Method |
|---|-----------|--------|
| 1 | **Mortality Burden Estimation** | Horizontal bar chart of top 5 cancer sites |
| 2 | **Demographic Age Disparities** | Grouped box plots by gender & cancer type |
| 3 | **Healthcare Access Analysis** | 100% stacked bar chart of place-of-death |
| 4 | **Geographic Mortality Hotspot Mapping** | State-level stacked bar chart |
| 5 | **Predictive Modeling (ML)** | Random Forest classifier + confusion matrix |

---

## 🧬 Key Findings

- **Lung cancer** is the leading cause of cancer mortality in the registry, consistent with NCRP reports
- **Breast cancer** patients die at significantly younger ages compared to other cancer sites — especially relevant for South Asian populations
- **Over 30%** of Gallbladder and Stomach cancer deaths occur at home, indicating a critical gap in palliative care infrastructure
- **Northeastern states** (e.g., Mizoram) show disproportionate lung cancer burden; cervical cancer correlates with low female literacy regions
- **Age** is the strongest predictor in the Random Forest model, followed by State (geography); Gender contributes the least
- **Breast–Cervix** and **Gallbladder–Stomach** are the hardest-to-distinguish cancer pairs using demographic features alone

---

## 🛠️ Tech Stack

- **Language:** Python 3.10
- **Data Handling:** `pandas`, `NumPy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn` (Random Forest Classifier)
- **Data Format:** Stata `.dta` (ICMR PBCR Mortality Dataset)

---

## ⚙️ Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Add the dataset
Download the ICMR PBCR Mortality dataset (`SAMPLE_PBCR_Mortality.dta`) from the [ICMR data portal](https://main.icmr.nic.in/) and update the file path in `Project375.py`:
```python
file_path = "path/to/your/SAMPLE_PBCR_Mortality.dta"
```

### 4. Run the pipeline
```bash
python Project375.py
```
This will sequentially generate all 5 objective figures and print the ML classification report.

---

## 📊 Dataset

**Source:** Indian Council of Medical Research (ICMR) — National Cancer Registry Programme (NCRP)  
**Format:** Stata `.dta`  
**Variables:** Patient ID, Registry ID, Date of Death, Location (District–State), Age, Gender, Place of Death, Cause of Death, ICD-10 Code, Histology Code  

> ⚠️ The raw dataset is **not included** in this repository due to data sharing restrictions. Access must be requested directly from ICMR.

---

## 🤖 ML Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Random Forest Classifier |
| Estimators | 300 |
| Max Depth | 10 |
| Class Weights | Balanced |
| Train/Test Split | 80% / 20% (stratified) |
| Features | Age, Gender (encoded), State (encoded) |
| Target | Cancer Type (5 classes) |

---

## 📚 References

1. Ferlay et al., *Global Cancer Observatory: Cancer Today*, IARC, 2024
2. ICMR, *Three-Year Report of Population Based Cancer Registries 2019–2021*, NCDIR, 2023
3. Dhillon et al., *Lancet Oncology*, vol. 19, no. 10, pp. 1289–1306, 2018
4. Noronha et al., *J. Gastrointestinal Oncology*, vol. 12, no. 4, 2021
5. Kumar et al., *IEEE BIBM 2022*, pp. 1204–1211
6. Deo, *Circulation*, vol. 132, no. 20, pp. 1920–1930, 2015
7. Srinivasan & Agarwal, *Expert Systems with Applications*, vol. 180, 2021

---

## 📜 License

This project is intended for academic and research purposes. Please cite the paper if you use this code or findings in your work.

---

## 🙋 Contact

**Pharhan Anzum Haque**  
B.Tech CSE, Lovely Professional University  
pharhanhaque@gmail.com

 
