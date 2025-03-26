"""
Salary Classification from LinkedIn Job Postings
------------------------------------------------
This script performs salary classification using a variety of machine learning models
based on job posting data collected from LinkedIn.

"""

import pandas as pd
import numpy as np

# DATA CLEANING

## Loading and Merging Data


# Load the datasets
companies_df = pd.read_csv('companies.csv')
company_industries_df = pd.read_csv('company_industries.csv')
company_specialities_df = pd.read_csv('company_specialities.csv')
employee_counts_df = pd.read_csv('employee_counts.csv')
job_industries_df = pd.read_csv('job_industries.csv')
job_skills_df = pd.read_csv('job_skills.csv')
postings_df = pd.read_csv('postings.csv')
skills_df = pd.read_csv('skills.csv')

postings_df.nunique()

postings_df.columns

postings_df = postings_df.drop(columns = ['description','med_salary','applies','original_listed_time',
                                          'views','job_posting_url','application_url','application_type',
                                          'expiry','closed_time','skills_desc','listed_time','posting_domain',
                                          'sponsored', 'work_type','currency', 'compensation_type', 'fips', 'normalized_salary','location'])

postings_df.head()

companies_df = companies_df.drop(columns = ['name', 'description','zip_code','address','url'])

companies_df.head()

merged_df = postings_df.merge(companies_df, on = 'company_id', how = 'left')

merged_df.head()

merged_df = merged_df.merge(company_industries_df, on = 'company_id', how = 'left')

merged_df.head()

merged_df = merged_df.merge(company_specialities_df, on = 'company_id', how = 'left')

merged_df.head()

employee_counts_df = employee_counts_df.drop(columns = ['follower_count', 'time_recorded'])

merged_df = merged_df.merge(employee_counts_df, on = 'company_id', how = 'left')

merged_df.head()

merged_df = merged_df.merge(job_industries_df, on = 'job_id', how = 'left')
merged_df = merged_df.merge(job_skills_df, on = 'job_id', how = 'left')
merged_df = merged_df.merge(skills_df, on = 'skill_abr', how = 'left')

merged_df.head()

## Filtering Data

merged_df['avg_salary'] = (merged_df['max_salary'] + merged_df['min_salary']) / 2

merged_df.head()

merged_df = merged_df.dropna(subset =['avg_salary'])

merged_df.shape

merged_df.isna().sum()

# Replace NaN value with 0: Remote Allowed Yes = 1, Remote Allowed No = 0

merged_df['remote_allowed'] = merged_df['remote_allowed'].fillna(0)

merged_df = merged_df.dropna(subset =['formatted_experience_level','zip_code',
                                     'company_size', 'company_name', 'speciality','state'])

merged_df['formatted_work_type'].value_counts()

# Keep only full time jobs
merged_df = merged_df[merged_df['formatted_work_type'].isin(['Full-time', 'Contract', 'Part-time'])]

# Keep only US countries since it represents 93% of the dataset

merged_df = merged_df[merged_df['country'] == 'US']
merged_df = merged_df.drop(columns = ['country'])

merged_df.shape

## Normalization

# Convert med_salary from hourly to yearly

merged_df.loc[(merged_df['pay_period'] == 'HOURLY') & (merged_df['formatted_work_type'].isin(['Full-time', 'Contract'])),'avg_salary'] *= 2080
merged_df.loc[(merged_df['pay_period'] == 'HOURLY') & (merged_df['formatted_work_type'] == 'Part-time'),'avg_salary'] *= 1040

merged_df.head(100)

# drop pay_period column

merged_df = merged_df.drop(columns = ['pay_period','max_salary','min_salary','speciality'])

merged_df.head(50)

merged_df['avg_salary'].describe()

# Set median of avg_salary as the cut off value
med_avg_salary = 88400.00

# Categorize into 'Low Salary' and 'High Salary'
merged_df['salary'] = merged_df['avg_salary'].apply(
    lambda x: 'Low Salary' if x <= med_avg_salary else 'High Salary'
)

merged_df

## Duplicates

duplicate_count = merged_df['job_id'].duplicated().sum()
print(f"Number of duplicate job_id values: {duplicate_count}")

# Create a job_id count column, grouby job_id and remove duplicates

merged_df['job_id_count'] = merged_df.groupby('job_id').cumcount() + 1
merged_df = merged_df[merged_df['job_id_count'] <= 1].drop(columns=['job_id_count'])

merged_df

merged_df['salary'].value_counts()

# Define state mapping

state_mapping = {
    # Standard abbreviations
    'CA': 'CA', 'FL': 'FL', 'VA': 'VA', 'IL': 'IL', 'NJ': 'NJ', 'GA': 'GA', 'MI': 'MI',
    'NY': 'NY', 'MA': 'MA', 'IN': 'IN', 'WA': 'WA', 'AZ': 'AZ', 'PA': 'PA', 'MN': 'MN',
    'MD': 'MD', 'TN': 'TN', 'CO': 'CO', 'CT': 'CT', 'UT': 'UT', 'TX': 'TX', 'HI': 'HI',
    'NC': 'NC', 'WI': 'WI', 'OR': 'OR', 'OH': 'OH', 'MO': 'MO', 'IA': 'IA', 'NE': 'NE',
    'DC': 'DC', 'KY': 'KY', 'NH': 'NH', 'LA': 'LA', 'NV': 'NV', 'AR': 'AR', 'SD': 'SD',
    'OK': 'OK', 'SC': 'SC', 'KS': 'KS', 'AK': 'AK', 'AL': 'AL', 'NM': 'NM', 'ID': 'ID',
    'RI': 'RI', 'MT': 'MT', 'ND': 'ND', 'DE': 'DE', 'WY': 'WY', 'ME': 'ME',

    # Variants and alternate formats
    'New York': 'NY', 'New York, NY': 'NY', 'New York (NY)': 'NY', 'New york': 'NY',
    'Florida': 'FL', 'Virginia': 'VA', 'Illinois': 'IL', 'Georgia': 'GA',
    'Michigan': 'MI', 'Massachusetts': 'MA', 'Indiana': 'IN', 'Washington': 'WA',
    'California': 'CA', 'Minnesota': 'MN', 'Connecticut': 'CT', 'Utah': 'UT',
    'New Jersey': 'NJ', 'Texas': 'TX', 'North Carolina': 'NC', 'Wisconsin': 'WI',
    'Colorado': 'CO', 'Ohio': 'OH', 'Missouri': 'MO', 'Arizona': 'AZ', 'Iowa': 'IA',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maryland': 'MD', 'Oregon': 'OR', 'Nevada': 'NV',
    'Arkansas': 'AR', 'Hawaii': 'HI', 'Kansas': 'KS', 'Nebraska': 'NE', 'Oklahoma': 'OK',
    'Alabama': 'AL', 'South Dakota': 'SD', 'South Carolina': 'SC', 'Idaho': 'ID',
    'Montana': 'MT', 'New Mexico': 'NM', 'Rhode Island': 'RI', 'Maine': 'ME',
    'Delaware': 'DE', 'Alaska': 'AK',

    # Variants with different cases and extra text
    'ky': 'KY', 'Ky': 'KY', 'md': 'MD', 'oh': 'OH', 'ga': 'GA', 'Hi': 'HI', 'mo': 'MO',
    'ny': 'NY', 'ca': 'CA', 'Fl': 'FL', 'Il': 'IL', 'Ca': 'CA', 'Ga.': 'GA',
    'TEXAS': 'TX', 'CALIFORNIA': 'CA', 'NEW YORK': 'NY', 'NEW JERSEY': 'NJ',
    'DISTRICT OF COLUMBIA': 'DC', 'D.C.': 'DC', 'CO - Colorado': 'CO',
    'CA - California': 'CA', 'WISCONSIN': 'WI', 'MASSACHUSETTS': 'MA',
    'Louisianna': 'LA', 'Pa': 'PA', '01824': 'REMOVE', '55077': 'REMOVE',
    'Plymouth Meeting, PA,': 'PA', 'Colorado/Utah': 'CO', 'Greater London': 'REMOVE',
    'Pennsylvania': 'PA', 'Tennessee': 'TN', 'District of Columbia': 'DC',
    'New Hampshire': 'NH', 'tx': 'TX', 'Va': 'VA',

    # Handle invalid or ambiguous entries
    'globally': 'REMOVE', 'Global': 'REMOVE', '0': 'REMOVE', '94086': 'REMOVE'
}

# Apply mapping
merged_df['state'] = merged_df['state'].str.strip()  # Remove extra spaces
merged_df['state'] = merged_df['state'].replace(state_mapping)

# Remove invalid rows
merged_df = merged_df[merged_df['state'] != 'REMOVE']

# Verify results
print(merged_df['state'].unique())
print(f"Number of unique states after preprocessing: {merged_df['state'].nunique()}")

# Decision Tree

import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import tree
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report,f1_score,roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

#X = merged_df.drop(columns=['salary','job_id','pay_period','company_id','med_salary','formatted_work_type'])

tree_df = merged_df.copy()
tree_df.reset_index(drop=True, inplace=True)

tree_df = tree_df.drop(columns=['job_id','company_name',
                          'zip_code','skill_abr',
                          'avg_salary', 'company_id','industry_id',
                            'city','title'])

tree_df.head(100)

unique_levels = tree_df['formatted_experience_level'].unique()

# Print the unique titles
print(f"Number of unique : {len(unique_levels)}")
print(unique_levels)

# Change 'formatted_experience_level' type to represent categories as numbers,

experience_mapping = {
    'Internship': 0,
    'Entry level': 1,
    'Associate': 2,
    'Mid-Senior level': 3,
    'Executive': 4,
    'Director': 5
}

tree_df['formatted_experience_level'] = tree_df['formatted_experience_level'].map(experience_mapping)

tree_df

# Create dummy variables for 'formatted_work_type'

tree_df = pd.get_dummies(tree_df, columns=['formatted_work_type','state','industry','skill_name'], drop_first=True)

# Convert all boolean columns to integers (0 and 1)
tree_df = tree_df.astype({col: 'int' for col in tree_df.select_dtypes('bool').columns})

tree_df

X = tree_df.drop(columns=['salary'])
y = tree_df['salary']
print(X.shape)
print(y.shape)

tree_df['salary'].value_counts()

# Split 80-20: 80% training and 20% testing

x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)
model = tree.DecisionTreeClassifier(criterion='entropy')
model.fit(x_train,y_train)

y_train_pred = model.predict(x_train)
y_test_pred  = model.predict(x_test)

classes = ['High Salary','Low Salary']

def plot_confusionmatrix(y_train_pred,y_train,dom):
    print(f'{dom} Confusion matrix')
    cf = confusion_matrix(y_train_pred,y_train)
    # When annot is set to True, it adds text annotations to each cell of the heatmap.
    sns.heatmap(cf,annot=True,yticklabels=classes,xticklabels=classes,cmap='Blues', fmt='g')
    plt.tight_layout()
    plt.show()

print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')
#plot_confusionmatrix(y_train_pred,y_train,dom='Train') # apply the function plot_confusionmatrix
plot_confusionmatrix(y_test_pred,y_test,dom='Test')

print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# Transform y_test to binary labels
y_test_numeric = y_test.map({'High Salary': 1, 'Low Salary': 0})

# Extract probabilities for class 1 (High Salary)
y_test_prob = model.predict_proba(x_test)[:, 0]

# Calculate AUC
fpr, tpr, thresholds = roc_curve(y_test_numeric, y_test_prob, pos_label=1)
roc_auc = auc(fpr, tpr)
print(f"AUC Score: {roc_auc:.2f}")

plt.figure(figsize=(7,7))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC Decision Tree un-processed = %0.2f'  % roc_auc)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

### Pre & Post Pruning Decision Tree

params = {'max_depth': [2,4,6,8,10,12],
         'min_samples_split': [2,3,4],
         'min_samples_leaf': [1,2]}

clf = tree.DecisionTreeClassifier()
gcv = GridSearchCV(estimator=clf,param_grid=params)
gcv.fit(x_train,y_train)

# Pre Pruning
model_pre_pruning = gcv.best_estimator_
print("Best Fit Model")
print(model_pre_pruning)

model_pre_pruning.fit(x_train,y_train)
y_train_pred = model_pre_pruning.predict(x_train)
y_test_pred = model_pre_pruning.predict(x_test)

print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')
#plot_confusionmatrix(y_train_pred,y_train,dom='Train')
plot_confusionmatrix(y_test_pred,y_test,dom='Test')

print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# Transform y_test to binary labels
y_test_numeric = y_test.map({'High Salary': 1, 'Low Salary': 0})

# Extract probabilities for class 1 (High Salary)
y_test_prob = model_pre_pruning.predict_proba(x_test)[:, 0]

# Calculate AUC
fpr, tpr, thresholds = roc_curve(y_test_numeric, y_test_prob, pos_label=1)
roc_auc = auc(fpr, tpr)
print(f"AUC Score: {roc_auc:.2f}")

plt.figure(figsize=(7,7))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC Decision Tree pre-pruning = %0.2f'  % roc_auc)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Post Pruning
path = model.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
print(ccp_alphas)

# For each alpha we will append our model to a list
models = []
for ccp_alpha in ccp_alphas:
    model = tree.DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    model.fit(x_train, y_train)
    models.append(model)

train_acc = []
test_acc  = []

for m in models:
    y_train_pred = m.predict(x_train)
    y_test_pred = m.predict(x_test)
    train_acc.append(accuracy_score(y_train_pred,y_train))
    test_acc.append(accuracy_score(y_test_pred,y_test))

plt.scatter(ccp_alphas,train_acc)
plt.scatter(ccp_alphas,test_acc)
plt.plot(ccp_alphas,train_acc,label='train_accuracy',drawstyle="steps-post")
plt.plot(ccp_alphas,test_acc,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Accuracy vs Alpha')
plt.show()

model_post_pruning = tree.DecisionTreeClassifier(random_state=42,ccp_alpha=0.020)
model_post_pruning.fit(x_train,y_train)
y_train_pred = model_post_pruning .predict(x_train)
y_test_pred = model_post_pruning .predict(x_test)

print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')
#plot_confusionmatrix(y_train_pred,y_train,dom='Train')
plot_confusionmatrix(y_test_pred,y_test,dom='Test')

print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# Transform y_test to binary labels
y_test_numeric = y_test.map({'High Salary': 1, 'Low Salary': 0})

# Extract probabilities for class 1 (High Salary)
y_test_prob = model_post_pruning.predict_proba(x_test)[:, 0]

# Calculate AUC
fpr, tpr, thresholds = roc_curve(y_test_numeric, y_test_prob, pos_label=1)
roc_auc = auc(fpr, tpr)
print(f"AUC Score: {roc_auc:.2f}")

plt.figure(figsize=(7,7))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC Decision Tree post pruning = %0.2f'  % roc_auc)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

### Smote Decision Tree

!pip install imbalanced-learn
from imblearn.over_sampling import SMOTE

print(y.value_counts())

smote = SMOTE(sampling_strategy=0.90, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(X_resampled, y_resampled, test_size=0.20, random_state=42)

smote_model = tree.DecisionTreeClassifier(criterion="entropy")
smote_model.fit(X_train_resampled, y_train_resampled)

y_train_pred_resampled = smote_model.predict(X_train_resampled)
y_test_pred_resampled = smote_model.predict(X_test_resampled)

def plot_confusionmatrix(y_train_pred_resampled, y_train_resampled, dom):
    print(f'{dom} Confusion Matrix')
    cf = confusion_matrix(y_train_resampled, y_train_pred_resampled)
    sns.heatmap(cf, annot=True, yticklabels = classes, xticklabels = classes, cmap = 'Blues', fmt = 'g')
    plt.tight_layout()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

print(f'Train score {accuracy_score(y_train_pred_resampled, y_train_resampled)}')
print(f'Test score {accuracy_score(y_test_pred_resampled, y_test_resampled)}')
plot_confusionmatrix(y_test_pred_resampled, y_test_resampled, dom = 'Test')

print("\nClassification Report:\n", classification_report(y_test_resampled, y_test_pred_resampled))

# Transform y_test to binary labels
y_test_numeric = y_test.map({'High Salary': 1, 'Low Salary': 0})

# Extract probabilities for class 1 (High Salary)
y_test_prob = smote_model.predict_proba(x_test)[:, 0]

# Calculate AUC
fpr, tpr, thresholds = roc_curve(y_test_numeric, y_test_prob, pos_label=1)
roc_auc = auc(fpr, tpr)
print(f"AUC Score: {roc_auc:.2f}")

plt.figure(figsize=(7,7))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC Decision Tree SMOTE = %0.2f'  % roc_auc)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Random Forest

forest_df = tree_df.copy()

forest_df

### Random Forest: without pre-processing

from sklearn.ensemble import RandomForestClassifier

# Define features (X) and target variable (y)
X = forest_df.drop(columns=['salary'])  # Exclude target column
y = forest_df['salary']

# Split data into training and testing sets (80-20 split)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(x_train, y_train)

# Predict on training data
y_train_pred = rf_model.predict(x_train)

# Predict on testing data
y_test_pred = rf_model.predict(x_test)

print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')
plot_confusionmatrix(y_test_pred,y_test,dom='Test')

print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# Transform y_test to binary labels
y_test_numeric = y_test.map({'High Salary': 1, 'Low Salary': 0})

# Extract probabilities for class 1 (High Salary)
y_test_prob = rf_model.predict_proba(x_test)[:, 0]

# Calculate AUC
fpr, tpr, thresholds = roc_curve(y_test_numeric, y_test_prob, pos_label=1)
roc_auc = auc(fpr, tpr)
print(f"AUC Score: {roc_auc:.2f}")

plt.figure(figsize=(7,7))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC Random Forest without pre-processing = %0.2f'  % roc_auc)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

### Random Forest: Feature Reduction

# Train the Random Forest
gini_model = RandomForestClassifier(n_estimators=100,criterion='gini', random_state=42)
gini_model.fit(X, y)

# Extract feature importances
importances = gini_model.feature_importances_
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

top_features = feature_importances.head(20)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_features)
plt.title('Top Features by Gini Importance')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()

# Select important features
selected_features = feature_importances[feature_importances['Importance'] > 0.025]['Feature']
X_selected = X[selected_features]

# Retrain the model with selected features
x_train, x_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
gini_model = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42)
gini_model.fit(x_train, y_train)

y_train_pred = gini_model.predict(x_train)
y_test_pred  = gini_model.predict(x_test)

selected_features

print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')
plot_confusionmatrix(y_test_pred,y_test,dom='Test')

print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# Transform y_test to binary labels
y_test_numeric = y_test.map({'High Salary': 1, 'Low Salary': 0})

# Extract probabilities for class 1 (High Salary)
y_test_prob = gini_model.predict_proba(x_test)[:, 0]

# Calculate AUC
fpr, tpr, thresholds = roc_curve(y_test_numeric, y_test_prob, pos_label=1)
roc_auc = auc(fpr, tpr)
print(f"AUC Score: {roc_auc:.2f}")

plt.figure(figsize=(7,7))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC Random Forest Feature Selection = %0.2f'  % roc_auc)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

### Random Forest Boosting

from sklearn.ensemble import AdaBoostClassifier

# AdaBoost

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Split the data into training and testing sets

# Create an AdaBoost classifier
boost_model = AdaBoostClassifier(n_estimators=100, random_state=0)

boost_model.fit(X_train, y_train) # Train the model on the training data

y_train_pred = boost_model.predict(X_train)
y_test_pred  = boost_model.predict(X_test)

print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')
plot_confusionmatrix(y_test_pred,y_test,dom='Test')

print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# Transform y_test to binary labels
y_test_numeric = y_test.map({'High Salary': 1, 'Low Salary': 0})

# Extract probabilities for class 1 (High Salary)
y_test_prob = boost_model.predict_proba(X_test)[:, 0]

# Calculate AUC
fpr, tpr, thresholds = roc_curve(y_test_numeric, y_test_prob, pos_label=1)
roc_auc = auc(fpr, tpr)
print(f"AUC Score: {roc_auc:.2f}")

plt.figure(figsize=(7,7))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC Random Forest Ada Boosting = %0.2f'  % roc_auc)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()