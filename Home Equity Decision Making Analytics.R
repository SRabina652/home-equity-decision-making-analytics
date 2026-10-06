# -------------------------------
# Install Packages
# -------------------------------

install.packages("e1071")
install.packages("smotefamily")
install.packages("dplyr")
install.packages("caret")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("randomForest")
install.packages("corrplot")
install.packages("pROC")
install.packages("fastDummies")
install.packages("mice")
install.packages("cluster")
install.packages("ggplot2")

# -------------------------------
# Load Packages
# -------------------------------

library(e1071)
library(smotefamily)
library(dplyr)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)
library(pROC)
library(fastDummies)
library(mice)
library(cluster)
library(ggplot2)
# -------------------------------
# Read Data
# -------------------------------

# Set working directory to the folder where the dataset is stored
setwd("C:/Users/shahi/OneDrive/Desktop/webster/Analytics Practicum Analytics Pract/project2")

# Read the Home Equity dataset
HomeEquityData <- read.csv("hmeq.csv")

# Preview the first few rows of the dataset
head(HomeEquityData)

# Check the structure of the dataset
str(HomeEquityData)

# Check the number of rows and columns
dim(HomeEquityData)

# Display variable names
names(HomeEquityData)

# View summary statistics
summary(HomeEquityData)

# -------------------------------
# Missing Values Check
# -------------------------------

# Count missing values in each column
colSums(is.na(HomeEquityData))

# Calculate percentage of missing values in each column
round(colMeans(is.na(HomeEquityData)) * 100, 2)

# Count missing values in each row
rowSums(is.na(HomeEquityData))

# Summarize how many rows have 0, 1, 2, etc. missing values
table(rowSums(is.na(HomeEquityData)))

# Check unique values in JOB and REASON to identify blank strings
unique(HomeEquityData$JOB)
unique(HomeEquityData$REASON)

# Convert blank strings in REASON and JOB into proper NA values
HomeEquityData$REASON[HomeEquityData$REASON == ""] <- NA
HomeEquityData$JOB[HomeEquityData$JOB == ""] <- NA

# Check again after cleaning blank values
unique(HomeEquityData$JOB)
unique(HomeEquityData$REASON)

# -------------------------------
# Missingness Indicator
# -------------------------------

# Create an indicator variable showing whether DEBTINC is missing
HomeEquityData$DEBTINC_MISS <- ifelse(is.na(HomeEquityData$DEBTINC), 1, 0)

# Compare missingness of DEBTINC with BAD to see if missingness is informative
prop.table(table(HomeEquityData$DEBTINC_MISS, HomeEquityData$BAD), 1)

# -------------------------------
# Zero Handling in Numeric Columns
# -------------------------------

# Identify numeric columns
num_cols <- sapply(HomeEquityData, is.numeric)

# Count how many zero values appear in each numeric variable
sapply(HomeEquityData[, num_cols], function(x) sum(x == 0, na.rm = TRUE))

# -------------------------------
# Missing Pattern Visualization
# -------------------------------

# Adjust plotting size for the missing data pattern plot
par(cex = 0.5)

# Display the pattern of missing values across variables
md.pattern(HomeEquityData, rotate.names = TRUE)

# -------------------------------
# Numeric Summary
# -------------------------------

# Identify numeric variables
comp_num_vars <- sapply(HomeEquityData, is.numeric)

# Show summary statistics for numeric variables only
summary(HomeEquityData[, comp_num_vars])

# Compute average values of numeric variables grouped by BAD
aggregate(HomeEquityData[, setdiff(names(HomeEquityData)[comp_num_vars], "BAD")],
          by = list(BAD = HomeEquityData$BAD),
          FUN = function(x) mean(x, na.rm = TRUE))

# -------------------------------
# Data Exploration
# -------------------------------

# Plot the distribution of the target variable BAD to check class imbalance
barplot(table(HomeEquityData$BAD),
        main = "Distribution of Default (BAD)",
        col = c("lightblue","salmon"),
        names.arg = c("No Default","Default"))

# Scatter plot to explore the relationship between loan amount and property value
plot(HomeEquityData$LOAN, HomeEquityData$VALUE,
     main = "Loan vs Property Value",
     xlab = "Loan",
     ylab = "Value")

# Density plot of DEBTINC grouped by BAD
ggplot(HomeEquityData[!is.na(HomeEquityData$DEBTINC), ],
       aes(x = DEBTINC, fill = as.factor(BAD))) +
  geom_density(alpha = 0.5) +
  labs(title = "Debt-to-Income by Default Status",
       fill = "BAD")

# Histogram to inspect the distribution of DEBTINC
hist(HomeEquityData$DEBTINC,
     main = "Distribution of Debt-to-Income Ratio",
     xlab = "DEBTINC",
     col = "lightblue",
     cex.main = 1.8,
     cex.axis = 1.8,
     cex.lab = 1.5)

# Boxplot to compare DEBTINC between default and non-default borrowers
boxplot(DEBTINC ~ BAD, data = HomeEquityData,
        main = "DEBTINC by Default Status",
        col = c("lightblue", "salmon"),
        cex.main = 1.8,
        cex.axis = 1.8,
        cex.lab = 1.5)

# Boxplot to compare DELINQ between default and non-default borrowers
boxplot(DELINQ ~ BAD, data = HomeEquityData,
        main = "Delinquency by Default",
        col = c("lightblue", "salmon"),
        cex.main = 1.8,
        cex.axis = 1.8,
        cex.lab = 1.5)

# Outlier detection for LOAN
boxplot(HomeEquityData$LOAN,
        main = "Loan Outliers",
        cex.main = 1.8,
        cex.axis = 1.8)

# Outlier detection for VALUE
boxplot(HomeEquityData$VALUE,
        main = "Value Outliers",
        cex.main = 1.8,
        cex.axis = 1.8)

# Outlier detection for DEBTINC
boxplot(HomeEquityData$DEBTINC,
        main = "DEBTINC Outliers",
        cex.main = 1.8,
        cex.axis = 1.8)

# -------------------------------
# Convert Categorical Variables to Factor
# -------------------------------

# Convert JOB to factor for analysis and modeling
HomeEquityData$JOB <- as.factor(HomeEquityData$JOB)

# Convert REASON to factor for analysis and modeling
HomeEquityData$REASON <- as.factor(HomeEquityData$REASON)

# -------------------------------
# Split Data
# -------------------------------

# Set seed for reproducibility
set.seed(2026)

# Split data into 70% training and 30% testing while preserving BAD distribution
train_idx <- createDataPartition(HomeEquityData$BAD, p = 0.70, list = FALSE)

# Create training dataset
train_df <- HomeEquityData[train_idx, ]

# Create testing dataset
test_df  <- HomeEquityData[-train_idx, ]


# -------------------------------
# Impute Missing Values
# -------------------------------
train_for_impute <- train_df[, !(names(train_df) %in% c("BAD", "DEBTINC_MISS"))]
test_for_impute  <- test_df[, !(names(test_df) %in% c("BAD", "DEBTINC_MISS"))]

imp_train <- mice(train_for_impute, m = 5, method = "pmm", seed = 2026, printFlag = FALSE)
train_imputed_part <- complete(imp_train, 1)

imp_test <- mice.mids(imp_train, newdata = test_for_impute, maxit = 5, printFlag = FALSE)
test_imputed_part <- complete(imp_test, 1)

train_imputed <- cbind(
  BAD = train_df$BAD,
  train_imputed_part,
  DEBTINC_MISS = train_df$DEBTINC_MISS
)

test_imputed <- cbind(
  BAD = test_df$BAD,
  test_imputed_part,
  DEBTINC_MISS = test_df$DEBTINC_MISS
)
# -------------------------------
# Correlation Analysis
# -------------------------------

# Identify numeric variables in imputed training data
comp_num_vars <- sapply(train_imputed, is.numeric)

# Compute correlation matrix for numeric variables
cor_matrix <- cor(train_imputed[, comp_num_vars])

# Display rounded correlation matrix
round(cor_matrix, 2)

# Visualize correlation matrix
corrplot(cor_matrix, method = "color", tl.cex = 1.2)

# -------------------------------
# Categorical Predictors vs BAD
# -------------------------------

# Show row proportions of REASON by BAD
prop.table(table(train_imputed$REASON, train_imputed$BAD), 1)

# Show row proportions of JOB by BAD
prop.table(table(train_imputed$JOB, train_imputed$BAD), 1)

# Count BAD values
table(train_imputed$BAD)

# Show proportion of BAD values
prop.table(table(train_imputed$BAD))

# Convert BAD to factor in both train and test datasets
train_imputed$BAD <- as.factor(train_imputed$BAD)
test_imputed$BAD <- as.factor(test_imputed$BAD)

# =========================================================
# CLASSIFICATION SECTION
# =========================================================

# -------------------------------
# Create Dummy Variables
# -------------------------------

# Convert categorical predictors in training data into dummy variables
train_dummy <- dummy_cols(
  train_imputed,
  select_columns = c("REASON", "JOB"),
  remove_first_dummy = TRUE,
  remove_selected_columns = TRUE
)

# Convert categorical predictors in testing data into dummy variables
test_dummy <- dummy_cols(
  test_imputed,
  select_columns = c("REASON", "JOB"),
  remove_first_dummy = TRUE,
  remove_selected_columns = TRUE
)

# Identify columns present in train but missing in test
missing_cols <- setdiff(names(train_dummy), names(test_dummy))

# Add missing columns to test data and fill them with 0
for (col in missing_cols) {
  test_dummy[[col]] <- 0
}

# Identify any extra columns in test not present in train
extra_cols <- setdiff(names(test_dummy), names(train_dummy))

# Remove extra columns from test data
test_dummy <- test_dummy[, !(names(test_dummy) %in% extra_cols)]

# Reorder test columns to match training data exactly
test_dummy <- test_dummy[, names(train_dummy)]

# -------------------------------
# Separate Predictors and Target
# -------------------------------

# Separate predictor variables from target variable in training data
predictors_train <- train_dummy[, names(train_dummy) != "BAD"]

# Store target variable separately
target_train <- train_dummy$BAD

# -------------------------------
# Apply SMOTE
# -------------------------------

# Set seed for reproducibility
set.seed(2026)

# Apply SMOTE to oversample the minority class in the training set
smote_result <- SMOTE(
  X = predictors_train,
  target = target_train,
  K = 5,
  dup_size = 3
)

# Extract balanced training dataset
train_smote <- smote_result$data

# Rename last column as BAD
names(train_smote)[ncol(train_smote)] <- "BAD"

# Convert BAD back to factor
train_smote$BAD <- as.factor(train_smote$BAD)

# Check class counts after SMOTE
table(train_smote$BAD)

# Check class proportions after SMOTE
prop.table(table(train_smote$BAD))

# -------------------------------
# Logistic Regression
# -------------------------------

# Fit logistic regression model on SMOTE-balanced training data
log_model <- glm(BAD ~ ., data = train_smote, family = binomial)

# Display model summary
summary(log_model)

# Predict probabilities on test data
log_prob <- predict(log_model, test_dummy, type = "response")

# Convert predicted probabilities into class labels using 0.5 threshold
log_pred <- ifelse(log_prob > 0.5, 1, 0)
log_pred <- as.factor(log_pred)

# Evaluate logistic regression performance
confusionMatrix(log_pred, test_dummy$BAD, positive = "1")

# Compute ROC curve for logistic regression
roc_log <- roc(test_dummy$BAD, log_prob)

# Compute AUC for logistic regression
auc(roc_log)

# Plot ROC curve
plot(roc_log, col = "red", main = "ROC Curve - Logistic Regression")

# -------------------------------
# Decision Tree
# -------------------------------

# Fit classification tree
tree_model <- rpart(BAD ~ ., data = train_smote, method = "class",
                    control = rpart.control(cp = 0.01))

# Plot decision tree
rpart.plot(tree_model)

# Predict class labels on test data
tree_pred <- predict(tree_model, test_dummy, type = "class")

# Evaluate decision tree performance
confusionMatrix(tree_pred, test_dummy$BAD, positive = "1")

# Predict class probabilities for ROC/AUC
tree_prob <- predict(tree_model, test_dummy, type = "prob")[, 2]

# Compute ROC curve
roc_tree <- roc(test_dummy$BAD, tree_prob)

# Compute AUC
auc(roc_tree)

# Plot ROC curve
plot(roc_tree, col = "red", main = "ROC Curve - Decision Tree")

# -------------------------------
# Random Forest
# -------------------------------

# Fit random forest model
rf_model <- randomForest(BAD ~ ., data = train_smote, ntree = 500)

# Predict class labels on test data
rf_pred <- predict(rf_model, test_dummy)

# Evaluate random forest performance
confusionMatrix(rf_pred, test_dummy$BAD, positive = "1")

# Predict class probabilities
rf_prob <- predict(rf_model, test_dummy, type = "prob")[, 2]

# Compute ROC curve
roc_rf <- roc(test_dummy$BAD, rf_prob)

# Compute AUC
auc(roc_rf)

# Plot ROC curve
plot(roc_rf, col = "red", main = "ROC Curve - Random Forest")

# -------------------------------
# Naive Bayes
# -------------------------------

# Fit Naive Bayes model
nb_model <- naiveBayes(BAD ~ ., data = train_smote)

# Predict class labels on test data
nb_pred <- predict(nb_model, test_dummy)

# Evaluate Naive Bayes performance
confusionMatrix(nb_pred, test_dummy$BAD, positive = "1")

# Predict class probabilities
nb_prob <- predict(nb_model, test_dummy, type = "raw")[, 2]

# Compute ROC curve
roc_nb <- roc(test_dummy$BAD, nb_prob)

# Compute AUC
auc(roc_nb)

# Plot ROC curve
plot(roc_nb, col = "red", main = "ROC Curve - Naive Bayes")

# -------------------------------
# Compare AUC of Models
# -------------------------------

# Create a summary table to compare AUC values across models
model_auc <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Random Forest", "Naive Bayes"),
  AUC = c(
    auc(roc_log),
    auc(roc_tree),
    auc(roc_rf),
    auc(roc_nb)
  )
)

# Display model comparison table
model_auc

# =========================================================
# LINEAR REGRESSION SECTION
# Predicting Property Value
# =========================================================

# Convert BAD back to numeric/factor-safe dataset is already imputed
# Use training and testing imputed data

# Fit linear regression model to predict VALUE
linear_model <- lm(VALUE ~ LOAN + MORTDUE + YOJ + DEROG + DELINQ + CLAGE +
                     NINQ + CLNO + DEBTINC + DEBTINC_MISS,
                   data = train_imputed)

# View model summary
summary(linear_model)

# Predict VALUE on test data
linear_pred <- predict(linear_model, newdata = test_imputed)

# Calculate regression errors
actual_value <- test_imputed$VALUE

MAE <- mean(abs(actual_value - linear_pred))
MSE <- mean((actual_value - linear_pred)^2)
RMSE <- sqrt(MSE)

# Show results
MAE
MSE
RMSE

# =========================================================
# CLUSTERING SECTION
# =========================================================

# -------------------------------
# Prepare Data for Clustering
# -------------------------------

# Remove target variable because clustering is unsupervised
cluster_data <- train_imputed[, names(train_imputed) != "BAD"]

# Convert categorical variables into dummy variables for clustering
cluster_dummy <- dummy_cols(
  cluster_data,
  select_columns = c("REASON", "JOB"),
  remove_first_dummy = TRUE,
  remove_selected_columns = TRUE
)

# Standardize variables so distance-based clustering is not dominated by scale
cluster_scaled <- scale(cluster_dummy)

# -------------------------------
# Elbow Method
# -------------------------------

# Set seed for reproducibility
set.seed(2026)

# Create vector to store within-cluster sum of squares for different k values
wss <- numeric(10)

# Compute WSS for k = 1 to 10
for (k in 1:10) {
  wss[k] <- kmeans(cluster_scaled, centers = k, nstart = 25)$tot.withinss
}

# Plot elbow method to help choose the number of clusters
plot(1:10, wss, type = "b",
     xlab = "Number of Clusters",
     ylab = "Within Cluster Sum of Squares",
     main = "Elbow Method for K-Means")

# -------------------------------
# K-Means Clustering
# -------------------------------

# Fit K-means clustering with 3 clusters
set.seed(2026)
kmeans_model <- kmeans(cluster_scaled, centers = 3, nstart = 25)

# Assign cluster labels to the training data
train_imputed$cluster <- as.factor(kmeans_model$cluster)

# Check cluster sizes
table(train_imputed$cluster)

# Compare clusters with BAD to understand default patterns within clusters
prop.table(table(train_imputed$cluster, train_imputed$BAD), 1)

# -------------------------------
# Silhouette Analysis
# -------------------------------

# Compute silhouette values to evaluate clustering quality
sil <- silhouette(kmeans_model$cluster, dist(cluster_scaled))

# Plot silhouette results
plot(sil, main = "Silhouette Plot for K-Means Clustering")

# Calculate average silhouette width
mean_silhouette <- mean(sil[, 3])

# Display average silhouette value
mean_silhouette
