# =============================================================================
# INDIAN PBCR MORTALITY DATASET - ADVANCED CANCER ANALYTICS
# =============================================================================

#  IMPORTS 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')
sns.set_context("paper", font_scale=1.2)
sns.set_style("white") 

#================================
# DATA EXTRACTION & PREPROCESSING
#================================

print("--- Loading & Preprocessing Data ---")

file_path = "C:/Users/Pharhan/Downloads/1751886102_SAMPLE_PBCR_Mortality.dta"
df = pd.read_stata(file_path)

# Rename columns
df = df.rename(columns={
    'v1': 'Patient_ID', 'v2': 'Registry_ID', 'v3': 'Date_of_Death',
    'v4': 'Location', 'v5': 'Age', 'v6': 'Gender',
    'v7': 'Place_of_Death', 'v8': 'Cause_of_Death',
    'v9': 'ICD10_Code', 'v10': 'Histology_Code'
})

#  Cleaning
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df.loc[df['Age'] == 99, 'Age'] = np.nan
df['Age'] = df['Age'].fillna(df['Age'].median())

# Location
df[['District', 'State']] = df['Location'].str.split('-', n=1, expand=True)
df['State'] = df['State'].str.strip()
df['District'] = df['District'].str.strip()

# Place of Death Mapping 
place_map = {1: 'Hospital', 2: 'Nursing Home', 3: 'Residence', '1': 'Hospital', '2': 'Nursing Home', '3': 'Residence'}
df['Place_Mapped'] = df['Place_of_Death'].map(place_map).fillna('Unknown')

# ICD-10 to Clinical Name Mapping
df['ICD10_Group'] = df['ICD10_Code'].str[:3]
icd10_map = {
    'C34': 'Lung', 'C50': 'Breast', 'C16': 'Stomach', 'C53': 'Cervix', 'C61': 'Prostate', 
    'C22': 'Liver', 'C18': 'Colon', 'C20': 'Rectum', 'C25': 'Pancreas', 'C56': 'Ovary', 
    'C73': 'Thyroid', 'C02': 'Tongue', 'C06': 'Mouth', 'C15': 'Esophagus', 'C23': 'Gallbladder', 
    'C80': 'Unknown Primary', 'C92': 'Leukemia', 'C90': 'Multiple Myeloma'
}
df['Cancer_Type'] = df['ICD10_Group'].map(icd10_map).fillna('Other')

# Filter for the Top 5 specific cancers
top_cancers = df[(df['Cancer_Type'] != 'Other') & (df['Cancer_Type'] != 'Unknown Primary')]['Cancer_Type'].value_counts().head(5).index.tolist()
df_top5 = df[df['Cancer_Type'].isin(top_cancers)].copy()

print("Preprocessing Complete.\n")

# =================================
# OBJECTIVE 1: THE MORTALITY BURDEN
# =================================
print("--- Objective 1: Visualizing Mortality Burden ---")

plt.figure(figsize=(12, 7))
# bar chart 
ax = sns.barplot(
    x=df_top5['Cancer_Type'].value_counts().values, 
    y=df_top5['Cancer_Type'].value_counts().index, 
    palette="flare_r", edgecolor='black', linewidth=1
)

# data labels  
for p in ax.patches:
    ax.annotate(f'{int(p.get_width())}', 
                (p.get_width() + 5, p.get_y() + p.get_height()/2), 
                ha='left', va='center', fontsize=12, fontweight='bold', color='#333')

plt.title("Objective 1: Top 5 Deadliest Cancers (Empirical Distribution)", fontsize=16, pad=20, fontweight='bold')
plt.xlabel("Total Recorded Deaths", fontsize=12)
plt.ylabel("", fontsize=12)
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()


# ========================================
# OBJECTIVE 2: DEMOGRAPHIC AGE DISPARITIES
# ========================================

print("--- Objective 2: Analyzing Age & Gender Disparities ---")

plt.figure(figsize=(12, 7))
# Boxplot with mean markers 
sns.boxplot(
    data=df_top5, x='Cancer_Type', y='Age', hue='Gender', 
    palette='vlag', fliersize=3, linewidth=1.2,
    showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"5"}
)

plt.title("Objective 2: Age Profile at Mortality by Cancer Site", fontsize=15, fontweight='bold', pad=15)
plt.xlabel("Cancer Type", fontsize=12)
plt.ylabel("Age at Death", fontsize=12)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.legend(title='Gender', frameon=True, shadow=False, loc='upper right')
sns.despine()
plt.tight_layout()
plt.show()

# ================================================
# OBJECTIVE 3: ENHANCED HEALTHCARE ACCESS ANALYSIS
# ================================================
print("--- Objective 3: Place of Death Proportions (Enhanced) ---")

place_data = pd.crosstab(df_top5['Cancer_Type'], df_top5['Place_Mapped'], normalize='index') * 100

ax = place_data.plot(
    kind='bar', 
    stacked=True, 
    figsize=(12, 7), 
    color=['#0077B6', '#00B4D8', '#90E0EF', '#343A40'], 
    edgecolor='white', 
    width=0.7
)

for p in ax.patches:
    h = p.get_height()
    if h > 5:
        ax.text(
            p.get_x() + p.get_width()/2., 
            p.get_y() + h/2., 
            f'{int(h)}%', 
            ha='center', 
            va='center', 
            color='white' if p.get_facecolor()[0] < 0.4 else 'black', 
            fontsize=10, 
            fontweight='bold'
        )

plt.title("Objective 3: Healthcare Access: Terminal Care Setting by Cancer Site", fontsize=15, fontweight='bold', pad=20)
plt.xlabel("Primary Cancer Site", fontsize=12)
plt.ylabel("Percentage of Deaths (%)", fontsize=12)
plt.xticks(rotation=0)
plt.legend(title="Location", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
sns.despine(left=True)
plt.tight_layout()
plt.show()

# ==========================================
# OBJECTIVE 4: GEOGRAPHIC MORTALITY HOTSPOTS
# ==========================================
print("--- Objective 4: Geographic Distribution ---")

plt.figure(figsize=(14, 7))
state_counts = pd.crosstab(df_top5['State'], df_top5['Cancer_Type'])
state_counts.plot(kind='bar', stacked=True, colormap='Spectral', edgecolor='white', ax=plt.gca())
plt.title("Objective 4: Regional Cancer Burden by Indian State", fontsize=15, fontweight='bold')
plt.ylabel("Number of Cases")
plt.xticks(rotation=45, ha='right')
plt.legend(title="Cancer Type", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()


# =================================================
# OBJECTIVE 5: CLINICAL INTELLIGENCE DASHBOARD (ML)
# =================================================
print("--- Objective 5: Machine Learning Insights ---")

# Data Encoding
ml_df = df_top5[['Age', 'Gender', 'State', 'Cancer_Type']].dropna()
le_gender, le_state = LabelEncoder(), LabelEncoder()
ml_df['Gender_Enc'] = le_gender.fit_transform(ml_df['Gender'])
ml_df['State_Enc'] = le_state.fit_transform(ml_df['State'])

# Model Training
X = ml_df[['Age', 'Gender_Enc', 'State_Enc']]
y = ml_df['Cancer_Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Create a Combined Model Diagnostic Figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 1. Styled Confusion Matrix
cm = confusion_matrix(y_test, model.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='magma', ax=ax1, 
            xticklabels=model.classes_, yticklabels=model.classes_, cbar=False)
ax1.set_title("ML Accuracy: Prediction vs Reality", fontweight='bold', fontsize=14)
ax1.set_xlabel("Predicted Type")
ax1.set_ylabel("Actual Type")

# 2. Stylized Feature Importance
importances = pd.Series(model.feature_importances_, index=['Age', 'Gender', 'State']).sort_values()
importances.plot(kind='barh', color=['#2a9d8f', '#e9c46a', '#f4a261'], ax=ax2, edgecolor='black')
ax2.set_title("Key Drivers of Cancer Mortality", fontweight='bold', fontsize=14)
ax2.set_xlabel("Impact on Model Prediction")

plt.suptitle("Objective 5: Machine Learning Diagnostic Dashboard", fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Final Report Summary
print("\n--- Project Analytics Summary ---")
print(classification_report(y_test, model.predict(X_test)))
print("Pipeline Complete. Figures ready for report insertion.")