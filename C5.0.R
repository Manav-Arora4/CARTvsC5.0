# ----- STEP 1: Load Required Packages -----

# Load the libraries
library(C50)         # For implementing C5.0
library(tidyverse)   # For data manipulation and visualization
library(caret)       # For model evaluation and cross-validation
library(rpart.plot)  # For visualizing decision trees
library(readr)       # For reading CSV files
library(pROC)        # For ROC curve analysis

# ----- STEP 2: Import and Explore the Dataset -----
# Import the dataset
file_path <- "C:/Users/MANAV/PycharmProjects/PythonProject/jupyter/college/CARTvsC50/UCI_Credit_Card.csv"
credit_data <- read_csv(file_path)

# Examine the structure of the dataset
str(credit_data)
summary(credit_data)

# Check for missing values
sum(is.na(credit_data))

# ----- STEP 3: Data Preprocessing -----
# Convert the target variable to a factor (categorical)
credit_data$`default.payment.next.month` <- as.factor(credit_data$`default.payment.next.month`)

# Convert payment history variables to factors (if needed)
payment_cols <- c("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")
credit_data[payment_cols] <- lapply(credit_data[payment_cols], factor)

# ----- STEP 4: Exploratory Data Analysis and Visualization -----
# Set up a plotting area with 2x2 layout
par(mfrow = c(2, 2))

# Visualize the distribution of the target variable
ggplot(credit_data, aes(x = `default.payment.next.month`, fill = `default.payment.next.month`)) +
  geom_bar() +
  scale_fill_manual(values = c("0" = "blue", "1" = "red")) +
  labs(title = "Distribution of Default Payments",
       x = "Default Payment (0 = No, 1 = Yes)",
       y = "Count")

# Visualize age distribution by default status
ggplot(credit_data, aes(x = AGE, fill = `default.payment.next.month`)) +
  geom_histogram(binwidth = 5, position = "dodge") +
  labs(title = "Age Distribution by Default Status",
       x = "Age",
       y = "Count")

# Visualize credit limit by default status
ggplot(credit_data, aes(x = `default.payment.next.month`, y = LIMIT_BAL)) +
  geom_boxplot() +
  labs(title = "Credit Limit by Default Status",
       x = "Default Payment (0 = No, 1 = Yes)",
       y = "Credit Limit")

# Visualize payment history and default status
ggplot(credit_data, aes(x = PAY_0, fill = `default.payment.next.month`)) +
  geom_bar(position = "fill") +
  labs(title = "Default Rate by Recent Payment Status",
       x = "Payment Status",
       y = "Proportion")

# Reset plotting parameters
par(mfrow = c(1, 1))

# ----- STEP 5: Split the Data into Training and Testing Sets -----
# Set a seed for reproducibility
set.seed(123)

# Create a partition index (70% training, 30% testing)
train_index <- createDataPartition(credit_data$`default.payment.next.month`, p = 0.7, list = FALSE)

# Create training and testing datasets
train_data <- credit_data[train_index, ]
test_data <- credit_data[-train_index, ]

# Check the dimensions of the datasets
dim(train_data)
dim(test_data)

# ----- STEP 6: Implement C5.0 Model -----
# Remove the ID column as it's not a predictor
train_data_no_id <- train_data %>% select(-ID)
test_data_no_id <- test_data %>% select(-ID)

# Train the C5.0 model
c5_model <- C5.0(x = train_data_no_id %>% select(-`default.payment.next.month`),
                y = train_data_no_id$`default.payment.next.month`)

# Print the model summary
print(c5_model)

# Get the rules from the model
summary(c5_model)

# ----- STEP 7: Visualize the Decision Tree -----
# Plot the decision tree
plot(c5_model)

# Variable importance
importance <- C5imp(c5_model)
print(importance)

# ----- STEP 8: Evaluate the Model Performance -----
# Make predictions on the test set
predictions <- predict(c5_model, test_data_no_id %>% select(-`default.payment.next.month`))

# Create a confusion matrix
conf_matrix <- confusionMatrix(predictions, test_data_no_id$`default.payment.next.month`)
print(conf_matrix)

# Plot ROC curve
prob_predictions <- predict(c5_model, test_data_no_id %>% select(-`default.payment.next.month`), type = "prob")
roc_obj <- roc(test_data_no_id$`default.payment.next.month`, prob_predictions[,"1"])
plot(roc_obj, main = "ROC Curve for C5.0 Model")
auc_value <- auc(roc_obj)  # Area under the curve
print(paste("AUC:", auc_value))

# ----- STEP 9: Fine-tune the Model with Cross-Validation -----
# Set up 10-fold cross-validation
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)

# Define the model grid for tuning
grid <- expand.grid(trials = c(1, 5, 10, 15, 20),
                    model = c("tree", "rules"),
                    winnow = c(TRUE, FALSE))

# Train the model with cross-validation
set.seed(123)
c5_tuned <- train(x = train_data_no_id %>% select(-`default.payment.next.month`),
                 y = train_data_no_id$`default.payment.next.month`,
                 method = "C5.0",
                 metric = "ROC",
                 trControl = ctrl,
                 tuneGrid = grid)

# Print the results of the tuning
print(c5_tuned)
plot(c5_tuned)

# Get the best model
best_model <- c5_tuned$finalModel

# Make predictions with the best model
best_predictions <- predict(best_model, test_data_no_id %>% select(-`default.payment.next.month`))
best_conf_matrix <- confusionMatrix(best_predictions, test_data_no_id$`default.payment.next.month`)
print(best_conf_matrix)

# ----- STEP 10: Feature Importance and Interpretation -----
# Get variable importance for the best model
best_importance <- C5imp(best_model)
print(best_importance)

# Plot variable importance
ggplot(best_importance, aes(x = reorder(attribute, importance), y = importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Variable Importance in C5.0 Model",
       x = "Attributes",
       y = "Importance")