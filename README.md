# Home Equity Decision Making Analytics

**Credit risk modeling and borrower analysis using HMEQ loan data**

## Project overview

This project analyzes home equity loan data for a regional bank seeking more consistent, scalable lending decisions. I used regression to predict property value, classification to assess default risk, and clustering to explore borrower segments.

## Business question

How can a lender identify borrowers with higher default risk while supporting consistent decisions and balancing risk with loan growth?

## Data

The dataset contains 5,960 approved and funded loans. The target for the classification models is `BAD`:

- `BAD = 1`: default or serious delinquency — 1,189 loans, about 20%
- `BAD = 0`: no default

Predictors describe loan and property values, debt-to-income, credit history, employment, and loan purpose.

## Approach

- Explored missing values, distributions, outliers, and relationships among variables
- Used MICE with predictive mean matching to impute missing values
- Created a missing-value indicator and dummy variables for categorical predictors
- Used a stratified 70/30 train-test split; applied SMOTE to the training data for classification
- Compared Logistic Regression, Decision Tree, Random Forest, and Naive Bayes for default classification
- Used Linear Regression to predict property value (`VALUE`)
- Used k-means clustering to explore borrower segments

## Results

### Default classification

| Model | Accuracy | AUC |
|---|---:|---:|
| Random Forest | 88.53% | 0.946 |
| Logistic Regression | 85.57% | 0.888 |
| Decision Tree | 86.69% | 0.848 |
| Naive Bayes | 82.66% | 0.868 |

Random Forest had the strongest overall results. On the test set, it identified 93.82% of non-defaulters and 67.86% of defaulters. That means it still missed some default cases, so the decision threshold would need to reflect the bank’s tolerance for missed risk versus false alarms.

### Property value regression

Linear Regression achieved an R² of 0.8006, with an MAE of 13,416.09 and an RMSE of 26,446.76, in the dataset’s value units.

### Borrower clustering

K-means produced three borrower groups. The average silhouette width was 0.138, indicating substantial overlap between groups; I treated the clusters as exploratory profiles rather than reliable risk categories.

## Business takeaways

- Defaulters tended to have higher debt-to-income ratios, more delinquent credit lines, and more derogatory credit reports.
- Random Forest offered the best classification performance, while Logistic Regression provided a more interpretable comparison.
- The sensitivity result highlights a business tradeoff: strong overall accuracy does not mean every high-risk borrower is detected.

## Limitations

The data contains only loans that were approved and funded, so results may not generalize to applicants who were denied. Model performance alone does not establish that a model is suitable for automated lending or meets fair-lending and adverse-action requirements.

## Project files

- [Project report](./Home%20Equity%20Decision%20Making%20Analytics.docx)
- [R program](./Home%20Equity%20Decision%20Making%20Analytics.R)
