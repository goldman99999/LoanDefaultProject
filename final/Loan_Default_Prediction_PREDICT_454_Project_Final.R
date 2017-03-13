# Northwestern University
# PREDICT 454
# Loan Default Prediction

# Install packages
# Can comment out (#) if installed already
install.packages("stats", dependencies = TRUE)
install.packages("pastecs", dependencies = TRUE)
install.packages("psych", dependencies = TRUE)
install.packages("lattice", dependencies = TRUE)
install.packages("plyr", dependencies = TRUE)
install.packages("corrplot", dependencies = TRUE)
install.packages("RColorBrewer", dependencies = TRUE)
install.packages("caTools", dependencies = TRUE)
install.packages("class", dependencies = TRUE)
install.packages("gmodels", dependencies = TRUE)
install.packages("C50", dependencies = TRUE)
install.packages("rpart", dependencies = TRUE)
install.packages("rpart.plot", dependencies = TRUE)
install.packages("modeest", dependencies = TRUE)
install.packages("randomForest", dependencies = TRUE)
install.packages("caret", dependencies = TRUE)
install.packages("MASS", dependencies = TRUE)
install.packages("e1071", dependencies = TRUE)
install.packages("ggplo2", dependencies = TRUE)
install.packages("readr", dependencies = TRUE)
install.packages("AUC", dependencies = TRUE)

# Include the following libraries
library(stats)
library(pastecs)
library(psych)
library(lattice)
library(plyr)
library(corrplot)
library(RColorBrewer)
library(caTools)
library(class)
library(gmodels)
library(C50)
library(rpart)
library(rpart.plot)
library(modeest)
library(randomForest)
library(caret)
library(MASS)
library(e1071)
library(ggplot2)
library(readr)
library(AUC)

# Get list of installed packages as check
search()

# Variable names start with Capital letters and for two words X_Y
# Examples: Path, Train_Data, Train_Data_Split1, etc.

# Set working directory for your computer
# Set path to file location for your computer
# Read in data to data file
setwd("C:/Users/Stephen D Young/Documents")
Path <- "C:/Users/Stephen D Young/Documents/Stephen D. Young/Northwestern/Predict 454/Project/Train Data/train_v2.csv"
# setwd("C:/Users/IBM_ADMIN/Northwestern/454/Project/train_v2.csv/")
# Path = ("C:/Users/IBM_ADMIN/Northwestern/454/Project/train_v2.csv/")

Train_Data <- read.csv(file.path(Path,"train_v2.csv"), stringsAsFactors=FALSE)

# Get structure and dimensions of data file
str(Train_Data)
dim(Train_Data)

# Characteristics of data frame from above
# 105471 rows(records), 771 columns, 770 not including loss which is last column 
# there is no indicator variable for default but can be 0 or 1 for when there is
# an observed loss.  So we can create binary default flag.
# Summary statistics for loss variable
# options used to define number and decimal places
options(scipen = 100)
options(digits = 4)
summary(Train_Data$loss)
# Get mode using mlv from modeest package
mlv(Train_Data$loss, method = "mfv")
# Results of loss are min = 0, max = 100, mean = .8, mode = 0
# Loss is not in dollars but as percent of loan amount (e.g. 50 is loss of 
# 50 out of 100 loan amount)

# Check for missing values in loss column
sum(is.na(Train_Data$loss))
# There are no missing values in loss column

# Add in default indicator and determine number and proportion of defaults
# Use ifelse to set default to 1 if loss > 0 otherwise default is 0
# Get table of 0 and 1 values (i.e. no default, default)
Train_Data$default <- ifelse(Train_Data$loss>0,1,0)
table(Train_Data$default)
dim(Train_Data)

# Box plot and density plot of loss amount
# Lattice package for bwplot and densityplot
# Boxplot is for no default (0) and default (1)
# Boxplot should have zero loss for no default and range of loss upon default
bwplot(loss~factor(default), data=Train_Data, 
       main = "Box Plot for No Default (0) and Default (1)",
       xlab = "No Default (0), Default (1)",
       ylab = "Loss Amount (% of Loan)")

# Density plot of loss should skew right as 100% loss is max but uncommon
densityplot(~(loss[loss>0]), data = Train_Data, plot.points=FALSE, ref=TRUE,
            main = "Kernel Density of Loss Amount (% of Loan)",
            xlab = "Loss for Values Greater than Zero",
            ylab = "Density")

# Density plot of loss should skew right even with 40% as max value for plot
densityplot(~loss[loss>0 & loss<40], data = Train_Data, plot.points=FALSE, ref=TRUE,
            main = "Kernel Density of Loss Amount (% of Loan) up to 40%",
            xlab = "Loss for Values Greater than Zero and up to 40%",
            ylab = "Density")

# Basic summary statistics for loss values when greater than zero
# Get mode using mlv from modeest package
summary(Train_Data$loss[Train_Data$loss>0])
mlv(Train_Data$loss[Train_Data$loss>0], method = "mfv")
# Results of loss are min = 1, max = 100, mean = 8.62, mode = 2

# Function to compute summary statistics
myStatsCol <- function(x,i){
  
  # Nine statistics from min to na
  mi <- round(min(x[,i], na.rm = TRUE),4)
  q25 <- round(quantile(x[,i],probs = 0.25,  na.rm=TRUE),4)
  md <- round(median(x[,i], na.rm = TRUE),4)
  mn <- round(mean(x[,i], na.rm = TRUE),4)
  st <- round(sd(x[,i], na.rm = TRUE),4)
  q75 <- round(quantile(x[,i], probs = 0.75, na.rm = TRUE),4)
  mx <- round(max(x[,i], na.rm = TRUE))
  ul <- length(unique(x[complete.cases(x[,i]),i]))
  na <- sum(is.na(x[,i]))
  
  # Get results and name columns
  results <- c(mi, q25, md, mn, st, q75, mx, ul, na)
  names(results) <- c("Min.", "Q.25", "Median", "Mean",
                      "Std.Dev.", "Q.75", "Max.",
                      "Unique", "NA's")
  results
}

# Check dimensions and names for calculation of summary statistics
dim(Train_Data)
names(Train_Data)

# Call to summary statistics function
# There are 9 summary statistics and 769 predictors excluding id, loss, 
# and default which are columns 1, 771, and 772
Summary_Statistics <- matrix(ncol = 9, nrow = 770)
colnames(Summary_Statistics) <- c("Min.", "Q.25", "Median", "Mean",
                                  "Std.Dev.", "Q.75", "Max.",
                                  "Unique", "NA's")
row.names(Summary_Statistics) <- names(Train_Data)[2:770]

# Loop for each variable included in summary statistics calculation
for(i in 1:770){
  
  Summary_Statistics[i,] <- myStatsCol(Train_Data,i+1)
  
}

# Create data frame of results
Summary_Statistics <- data.frame(Summary_Statistics)
# View select records for reasonableness
head(Summary_Statistics,20)

# Output file of predictor variable summaries
write.csv(Summary_Statistics, file = file.path(Path,"myStats_Data1.csv"))

# Creates summary table from which we get n which is number of missing
# records (i.e. Summary$n)
Summary <- describe(Train_Data, IQR = TRUE, quant=TRUE)

# Calculate number of missing records for each variable
Missing_Var <- nrow(Train_Data) - Summary$n
plot(Missing_Var, main = "Plot of Missing Values", xlab = "Variable Index",
     ylab = "Number of Rows - Missing Records")

# Get variable names missing more than 5000 records as we may want to remove
# those with n missing > 5000 as imputation could be problematic
Lot_Missing <- colnames(Train_Data)[Missing_Var >= 5000]
print(Lot_Missing)
# Variables for which n missing > 5,000
# [1] "f72"  "f159" "f160" "f169" "f170" "f179" "f180" "f189" "f190" "f199"
# [11] "f200" "f209" "f210" "f330" "f331" "f340" "f341" "f422" "f586" "f587"
# [21] "f588" "f618" "f619" "f620" "f621" "f640" "f648" "f649" "f650" "f651"
# [31] "f653" "f662" "f663" "f664" "f665" "f666" "f667" "f668" "f669" "f672"
# [41] "f673" "f679" "f726"

# Impute missing values
# Need to check size of data here to get for loop correct
dim(Train_Data)
# Loss and default have no missing values and will not be imputed
Imputed_Data <- Train_Data
for(i in 1:772){
  
  Imputed_Data[is.na(Train_Data[,i]), i] = median(Train_Data[,i], na.rm = TRUE)
  
}

Train_Data <- Imputed_Data
dim(Train_Data)

# Remove duplicate columns which we know exist from summary statistics
# and inspection
Train_Data <- Train_Data[!duplicated(as.list(Train_Data))]
dim(Train_Data)

# Remove constant columns which we know exist from summary statistics 
# and inspection
Train_Data <- Train_Data[,apply(Train_Data, 2, var, na.rm=TRUE) != 0]
dim(Train_Data)

# Principal Components Analysis (PCA) for Dimensionality Reduction
# Run PCA removing observations with missing data as first pass and scaling
# which is to unit variance
PCs <- prcomp(Train_Data[complete.cases(Train_Data),], scale=TRUE)

#Variance explained by each principal component
PCs_Var <- PCs$sdev^2

#Proportion of variance explained 
Prop_Var_Expl <- PCs_Var / sum(PCs_Var)

# Plot of proportion of variance explained and cumulative proportion of variance explained
plot(Prop_Var_Expl, main = "Proportion of Variance Explained per PC", xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1),type='b')
plot(cumsum(Prop_Var_Expl), main = "Cumulative Proportion of Variance Explained",xlab=" Principal Component", ylab ="Cumulative Proportion of Variance Explained", ylim=c(0,1),type='b')
# Based on PCA may be able to reduce features to ~ 200

# Find Correlations
High_Correlation <- cor(Train_Data)
High_Correlation1 <- findCorrelation(High_Correlation,cutoff=0.99)
High_Correlation1 <- sort(High_Correlation1)
High_Correlation <- High_Correlation[-1,-1]
Corr_Vars <- colnames(High_Correlation)[c(High_Correlation1)]
dim(Train_Data)
names(Train_Data)

Imputed_Predictors <- Train_Data[,-1] # id
dim(Imputed_Predictors)
names(Imputed_Predictors)
# 679 values at this point

# Calculate differences in all correlated variables
Imputed_Data <- Imputed_Predictors
diff <- NA
diff_mat <- rep(NA,nrow(Imputed_Predictors))
for (i in 1:ncol(High_Correlation)) {
  for (j in 1:nrow(High_Correlation)) {
    diff <- NA
    if (High_Correlation[j,i] >= 0.99 & High_Correlation[j,i] < 1.00){
      diff <- Imputed_Data[,i] - Imputed_Data[,j]
    } #ends if
    if (sum(as.numeric(diff),na.rm = TRUE)==0){
      temp_delete <- rep(NA,nrow(Imputed_Data))
    } else {
      diff_mat <- data.frame(cbind(diff_mat,diff))
      colnames(diff_mat)[ncol(diff_mat)] <- paste(colnames(High_Correlation)[i],"-",colnames(High_Correlation)[j],sep="")
    }
  } #ends j
} #ends i
head(diff_mat) 
names(diff_mat)
dim(diff_mat)
dim(Imputed_Data)
dim(Train_Data)
names(Imputed_Data)
names(Train_Data)

# Bring data back together to run analysis and determine important variables
# 1587 differences from diff_mat
# 679 predictors from Imputed_Data
# default
# New_Train_Data should be 105471 x 2266
New_Train_Data <- cbind(Train_Data,diff_mat) 
dim(New_Train_Data)
#names(New_Train_Data)

#-----------------------------------------------------------------------------------------
#  Rerun all previous lines and jump to 'loss' model after this
#-----------------------------------------------------------------------------------------

# Decision trees using rpart, rpart.plot and variable importance
Decision_Tree_Train <- rpart(as.factor(default)~. -loss, data=New_Train_Data)
rpart.plot(Decision_Tree_Train)
summary(Decision_Tree_Train)

# Splits and Variable importance
Decision_Tree_Train$splits
Decision_Tree_Train$variable.importance

barplot(Decision_Tree_Train$variable.importance, main="Variable Importance Plot from Decision Tree",
        xlab="Variables", col= "beige")

# Random Forest Model for Variable Importance EDA
set.seed(1030)
Random_Forest_Var_Imp <- randomForest(as.factor(default) ~ f274.f528 + f527.f274 + f527.f528 + f414.f415 + f264.f578 + f254.f569 +
                                        f2 + f334 + f339 + f378 + f338 + f221 + f222 + f272 + f653 + f663 + f662 +
                                        f664 + f73 + f776 + f332, data=New_Train_Data, importance=TRUE, ntree=200) 
VI_Fit1 <- importance(Random_Forest_Var_Imp)
varImpPlot(Random_Forest_Var_Imp, main = "Random Forest Variable Importance")
VI_Fit1

# Using 21 variables from decision tree importance and random forests importance
# we create a new data frame and output to csv so it can easily be read in at
# this point and used for model fitting.  Reading in entire dataset is too cumbersome
# and not necessary
New_Datafile <- as.data.frame(cbind(New_Train_Data$f274.f528, New_Train_Data$f527.f274, 
  New_Train_Data$f527.f528, New_Train_Data$f414.f415, New_Train_Data$f264.f578, 
  New_Train_Data$f254.f569,New_Train_Data$f2, New_Train_Data$f334, New_Train_Data$f339, 
  New_Train_Data$f378, New_Train_Data$f338, New_Train_Data$f221, New_Train_Data$f222, 
  New_Train_Data$f272, New_Train_Data$f653, New_Train_Data$f663, New_Train_Data$f662, 
  New_Train_Data$f664, New_Train_Data$f73, New_Train_Data$f776, New_Train_Data$f332, 
  New_Train_Data$default, New_Train_Data$loss))
dim(New_Datafile)
names(New_Datafile)

write.csv(New_Datafile, file = file.path(Path,"New_Datafile.csv"))
Reduced_Dataset <- read.csv(file.path(Path,"New_Datafile.csv"), stringsAsFactors=FALSE)
dim(Reduced_Dataset)
names(Reduced_Dataset)

New_Dataset <- Reduced_Dataset[,-1]
dim(New_Dataset)
names(New_Dataset)
# We now have V1, V2,..., V22, V23 where V22 is default and V23 loss

# Stepwise logistic regression
Logistic_Regression_EDA <- glm(as.factor(V22)~. -V23, data = New_Dataset, family=binomial)
summary(Logistic_Regression_EDA)
backwards <- step(Logistic_Regression_EDA, trace = 0)
formula(backwards)
summary(backwards)

# Stepwise logistic results in following formula:
#  as.factor(V22) ~ V1 + V2 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + 
#  V11 + V13 + V14 + V15 + V16 + V17 + V19 + V20 + V21

# Partitioning training data for training and validation
# Set seed for reproducibility
set.seed(999)
sample <- sample.split(New_Dataset$V22, SplitRatio = .70)
Training_Default <- subset(New_Dataset, sample == TRUE)
Validation_Default  <- subset(New_Dataset, sample == FALSE)
dim(Training_Default)
dim(Validation_Default)
table(Training_Default$V22)
table(Validation_Default$V22)

# Logistic Regression fitting in and out of sample results
Logistic_Regression <- glm(V22 ~ V1 + V2 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + 
                             V11 + V13 + V14 + V15 + V16 + V17 + V19 + V20 + V21, data=Training_Default, family = binomial)
# In sample
Logistic_Regression_Pred <- predict(Logistic_Regression,Training_Default, type="response")
Logistic_Default_Pred <- ifelse(Logistic_Regression_Pred >= 0.5,1,0)
table(Logistic_Default_Pred,Training_Default$V22)
# Out of sample
Logistic_Regression_Pred <- predict(Logistic_Regression,Validation_Default, type="response")
Logistic_Default_Pred <- ifelse(Logistic_Regression_Pred >= 0.5,1,0)
table(Logistic_Default_Pred,Validation_Default$V22)

# Plot ROC curve and calculate Area under curve for Logistic Regression
# Out of sample
Log_ROC_Data <- cbind(Logistic_Default_Pred,Validation_Default$V22)
Log_ROC <- roc(predictions = Log_ROC_Data[,1] , labels = factor(Validation_Default$V22))
Log_AUC <- auc(Log_ROC)
plot(Log_ROC, col = "red", lty = 1, lwd = 2, main = "ROC Curve for Logistic Regression")
Log_AUC

# Decision Tree for Prediction on Validation
Decision_Tree <- rpart(as.factor(V22) ~ V1 + V2 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + 
                         V11 + V13 + V14 + V15 + V16 + V17 + V19 + V20 + V21, data=Training_Default, method = "class")

# In sample
Decision_Tree_Pred <- predict(Decision_Tree,Training_Default, type="class")
table(Decision_Tree_Pred,Training_Default$V22)
# Out of sample
Decision_Tree_Pred <- predict(Decision_Tree,Validation_Default, type="class")
table(Decision_Tree_Pred,Validation_Default$V22)

# Plot ROC curve and calculate Area under curve for Decision Tree
Tree_ROC_Data <- cbind(Decision_Tree_Pred,Validation_Default$V22)
Tree_ROC <- roc(predictions = Tree_ROC_Data[,1] , labels = factor(Validation_Default$V22))
Tree_AUC <- auc(Tree_ROC)
plot(Tree_ROC, col = "red", lty = 1, lwd = 2, main = "ROC Curve for Decision Tree")
Tree_AUC

# Random Forest Model for Prediction on Validation
set.seed(1010)
Random_Forest <- randomForest(as.factor(V22) ~ V1 + V2 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + 
                                V11 + V13 + V14 + V15 + V16 + V17 + V19 + V20 + V21, data=Training_Default, importance = FALSE, ntree=1000) 

# In sample
Random_Forest_Pred <- predict(Random_Forest, Training_Default, type = "class")
table(Random_Forest_Pred,Training_Default$V22)

# Out of sample
Random_Forest_Pred <- predict(Random_Forest, Validation_Default, type = "class")
table(Random_Forest_Pred,Validation_Default$V22)

# Plot ROC curve and calculate Area under curve for Random Forest
RF_Roc_Data <- cbind(Random_Forest_Pred,Validation_Default$V22)
RF_ROC <- roc(predictions = RF_Roc_Data[,1] , labels = factor(Validation_Default$V22))
RF_AUC <- auc(RF_ROC)
plot(RF_ROC, col = "red", lty = 1, lwd = 2, main = "ROC Curve for Random Forest Model")
RF_AUC

#-----------------------------------------------------------------------------------------
#                                Analysis of 'Loss' 
#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
#                                Transformation
#-----------------------------------------------------------------------------------------
# boxcox power transformation using forecast library
library(forecast)
#  find optimal lambda
lambda = BoxCox.lambda( Train_Data$loss )
lambda
#0.0005656
# Lambda = 0.00: natural log transformation

# natural log transformation is used for 'loss'

# new dataframe to analyse 'loss'
Train_Data_loss = Train_Data

# Transformed target = log of (loss + 1). 1 is added because loss has data points with '0' values.
Train_Data_loss$logloss = log(Train_Data_loss$loss + 1)

# Get rid of loss
Train_Data_loss$loss = NULL

# Get rid of ID
Train_Data_loss$ID = NULL

# Get rid of default
Train_Data_loss$default = NULL

dim(Train_Data_loss)
#105471    679
#-----------------------------------------------------------------------------------------
#                                Variable Selection Process
#-----------------------------------------------------------------------------------------
#sample before modeling
set.seed(1234)
smpl1 <- Train_Data_loss[sample(nrow(Train_Data_loss), 20000),]

t.test(Train_Data_loss$logloss,smpl1$logloss)
# p-value = 0.3
# fail to reject the null hypothesis, which means there is no difference in the mean.
#-----------------------------------------------------------------------------------------
#                             Random Forest as variable selection
#-----------------------------------------------------------------------------------------
set.seed(100)
Random_Forest <- randomForest(logloss ~ ., data=smpl1, importance=TRUE, ntree=50) 
Random_Forest

varImpPlot(Random_Forest)
VF = varImp(Random_Forest)

# top 20 importance variables
importanceOrder=order(VF, decreasing = TRUE)
names=rownames(VF)[importanceOrder][1:20]
names

#[1] "f674" "f289" "f514" "f640" "f71"  "f442" "f624" "f384" "f20"  "f340" "f436" "f23"  "f433" 
# "f631" "f278" "f424" "f468" "f663" "f368" "f282"

VF = varImp(Random_Forest)
# Output file of predictor variable summaries
write.csv(VF, file = file.path(Path,"vfrf.csv"))
#-----------------------------------------------------------------------------------------
#                 Lasso as varriable selection
#-----------------------------------------------------------------------------------------
set.seed(100)
library(glmnet)
grid=10^seq(10,-2,length=100)
x=model.matrix(logloss~.,smpl1)[,-1]
y=smpl1$logloss

lasso.mod=glmnet(x,y,alpha=1,lambda=grid)
lasso.mod

set.seed(1)
cv.out=cv.glmnet(x,y,alpha=1)
bestlam=cv.out$lambda.min

lasso.pred=predict(lasso.mod,s=bestlam,newx=x[])
mean((lasso.pred-y)^2)

out=glmnet(x,y,alpha=1,lambda=grid)

#lasso.coef=predict(out,type="coefficients",s=bestlam)[1:20,]
lasso.coef=predict(out,type="coefficients",s=bestlam)[1:679,]
lasso.coef = lasso.coef[lasso.coef!=0]
lasso.coef
# Output file of predictor variable summaries
write.csv(lasso.coef, file = file.path(Path,"vflass.csv"))

# following variables have non-zero coefficients in Lasso model.

#  f2           f13           f23           f55          f211          f221 
#f228          f251          f261          f263          f270          f277          f308 
#f343          f360          f376          f404          f406          f411          f416 
#f459          f471          f482          f514          f536          f621          f629 
#f648          f657          f674          f699          f768          f776 
#-----------------------------------------------------------------------------------------
#                             Gradient Boosting as variable selection
#-----------------------------------------------------------------------------------------
set.seed(123)
gbmVarFit <- train(logloss ~ ., data = smpl1, method = "gbm",metric='RMSE')

gbmVarFit
# View variable importance
VFGBM = varImp(gbmVarFit)
VFGBM
#20 most important variables
# f536 f468  f471  f674  f211  f221  f766  f363  f376  f67 f404  f271 f629  f368  f28   
# f250 f213 f568 f273 f387

# Output to a file
sink(file.path(Path,"vfgbm.txt"), append=FALSE, split=FALSE)
VFGBM
sink()
#-----------------------------------------------------------------------------------------
#                       Varriable selection
#-----------------------------------------------------------------------------------------
# Following variables are selected for modeling:
# (1) All variables from Lasso + 
# (2) Top 20 variables from Random forest + 
# (3) Top 20 variables from GBM +
# (4) Variables from the classification model above.
# Some variables overlap.

# Transformed target = log of (loss + 1). 1 is added because loss has data points with '0' values.
New_Train_Data$logloss = log(New_Train_Data$loss + 1)

subset_data = subset(New_Train_Data, select=c(logloss,f2,f221,f222,f254.f569,f264.f578,f272,
                                            f332,f334,f338,f339,f378,f414.f415,f527.f274,
                                            f527.f528,f653,f662,f663,f664,f73,f776,f13,f2,
                                            f20,f211,f211,f221,f221,f228,f23,f23,f251,f261,
                                            f263,f270,f271,f274.f528,f277,f278,f28,f282,
                                            f289,f308,f340,f360,f363,f368,f368,f376,f376,
                                            f384,f404,f404,f406,f411,f416,f424,f43,f433,
                                            f436,f442,f459,f468,f468,f471,f471,f482,f514,
                                            f514,f536,f536,f55,f621,f624,f629,f629,f631,
                                            f640,f648,f657,f663,f67,f674,f674,f674,f699,
                                            f71,f766,f768,f776))
dim(subset_data)
#105471     90
# Save the subset
write.csv(subset_data, file = file.path(Path,"subset_train_v2.csv"))

#subset_data <- read.csv(file.path(Path,"subset_train_v2.csv"), stringsAsFactors=FALSE)
#subset_data$X = NULL

#-----------------------------------------------------------------------------------------
#                                Modeling Process
#-----------------------------------------------------------------------------------------
#sink(file='console.txt')
library(caret)
#-----------------------------------------------------------------------------------------
#                                Preparation for Modeling
#-----------------------------------------------------------------------------------------
# Partition the data into training and test sets
set.seed(998)
inTraining = createDataPartition(subset_data$logloss, p = 0.70, list = F)
training <- subset_data[ inTraining,]
testing  <- subset_data[-inTraining,]

#-----------------------------------------------------------------------------------------
#                        Linear Regression with stepwise Selection
#-----------------------------------------------------------------------------------------
set.seed(1234)
lm.stepwise = train(logloss~., data = training, method = "leapSeq")
# Model Summary
lm.stepwise
# In-Sample 
#nvmax  RMSE    Rsquared
#4      0.5869  0.03843
#4      0.5868  0.03573 

# predict using test data
lm.stepwise.pred = predict(lm.stepwise, testing)

#Out of sample
lm.stepwise.results = postResample(pred = lm.stepwise.pred, obs = testing$logloss)
lm.stepwise.results
# Run 1
#RMSE   Rsquared 
#0.59544  0.0361 
#0.59585  0.03483 
#-----------------------------------------------------------------------------------------
#                                Random Forest
#-----------------------------------------------------------------------------------------
set.seed(1234)
rffit <- randomForest(logloss ~ ., data=training, importance=TRUE, metric='RMSE') 
#rffit = train(logloss~ .,data = training, method = "rf", metric='RMSE')
# View summary information
rffit
plot(rffit, main="Random Forest")
# In-Sample 
#Type of random forest: regression
#Number of trees: 500
#No. of variables tried at each split: 29
#Mean of squared residuals: 0.05637945
#% Var explained: 84.11

#Mean of squared residuals: 0.05623
#% Var explained: 84.15
sqrt(0.05637945)
plot(rffit)

#For Regression, %IncMSE is the mean decrease in accuracy,
#and IncNodePurity is the mean decrease in MSE.
#IncNodePurity (increase node impurity)
varImpPlot(rffit, main="Random Forest Variable Importance")
plot(rffit, log="y")
VF = varImp(rffit)

# top 20 importance variables
importanceOrder=order(VF, decreasing = TRUE)
names=rownames(VF)[importanceOrder][1:20]
names
#[1] "f274.f528" "f67"       "f2"        "f2.1"      "f13"       "f264.f578" "f527.f274" "f332"     
#[9] "f254.f569" "f228"      "f263"      "f527.f528" "f776.1"    "f282"      "f776"      "f270"     
#[17] "f334"      "f424"      "f277"      "f55" 

# predict using test data
rffit.pred <- predict(rffit, testing)

#Out of sample
rffit.results = postResample(pred = rffit.pred, obs = testing$logloss)
rffit.results
#     RMSE  Rsquared 
#0.2400085 0.8456089 
#0.2397   0.8460 

#RMSE
RMSE.rffit <- sqrt(mean((rffit.pred-testing$logloss)^2))
RMSE.rffit
#0.2397

#MAE
MAE.rffit <- mean(abs(rffit.pred-testing$logloss))
MAE.rffit
#0.06621

# add the predicted value in testing data
testing[ ,(ncol(testing)+1)] <- rffit.pred

dim(testing)
write.csv(testing, file = file.path(Path,"testing_pred.csv"))
#-----------------------------------------------------------------------------------------
#                                Random Forest with ntree = 1500
#-----------------------------------------------------------------------------------------
set.seed(1234)
#rffit = train(logloss~., data = training, method = "rf",metric='RMSE')

#set.seed(100)
rffit2 <- randomForest(logloss ~ ., data=training, importance=TRUE, metric='RMSE', ntree=1500) 
# View summary information
rffit2
# In-Sample 
#Type of random forest: regression
#Number of trees: 500
#No. of variables tried at each split: 29
#Mean of squared residuals: 0.05637945
#% Var explained: 84.11
sqrt(0.05637945)
#0.2374
plot(rffit2)

VF = varImp(rffit2)

# top 20 importance variables
importanceOrder=order(VF, decreasing = TRUE)
names=rownames(VF)[importanceOrder][1:20]
names

# predict using test data
rffit2.pred <- predict(rffit2, testing)

#Out of sample
rffit2.results = postResample(pred = rffit2.pred, obs = testing$logloss)
rffit2.results
#     RMSE  Rsquared 
#0.2397   0.8461 
# Conclusion: ntree=1500 did not improve the RMSE much.
#-----------------------------------------------------------------------------------------
#                               Gradient Boosting
#-----------------------------------------------------------------------------------------
# GBM
set.seed(825)
gbmFit <- train(logloss ~ ., data = training, method = "gbm",metric='RMSE',verbose = TRUE)
# View summary information
gbmFit
# In-Sample
#interaction.depth  n.trees  RMSE    Rsquared
#3                  150      0.2798  0.7818
gbmFit$finalModel

# View variable importance
varImp(gbmFit)
plot(varImp(gbmFit))

# predict using test data
gbmFit.pred <- predict(gbmFit, testing)

#Out of sample
gbmFit.results = postResample(pred = gbmFit.pred, obs = testing$logloss)
gbmFit.results
#RMSE  Rsquared 
#0.2860   0.7806

#RMSE   
gbmFit.results[1]
#
#Rsquared 
gbmFit.results[2]
#
#-----------------------------------------------------------------------------------------
#                                Ridge Regression
#-----------------------------------------------------------------------------------------
set.seed(825) 
ridge <- train(logloss ~., data = training, method='ridge',lambda = 4,
               preProcess=c('scale', 'center'))
ridge
# In-Sample
#lambda  RMSE    Rsquared
#0.0001  0.5722  0.08314

ridge$finalModel

# View variable importance
#varImp(ridge)
#plot(varImp(ridge))

# predict using test data
ridge.pred <- predict(ridge, testing)

#Out of sample
ridge.results = postResample(pred = ridge.pred, obs = testing$logloss)
ridge.results
#RMSE   Rsquared 
#   RMSE Rsquared 
#0.58149  0.08011 
#-----------------------------------------------------------------------------------------
#                                     Lasso
#-----------------------------------------------------------------------------------------
set.seed(825) 
lasso <- train(logloss ~., training, method='lasso',
               preProc=c('scale','center'), metric='RMSE')
lasso
# In-Sample 
#fraction  RMSE    Rsquared
#0.1       0.5722  0.08327
lasso$finalModel

# View variable importance
varImp(lasso)
plot(varImp(lasso))

# predict using test data
lasso.pred <- predict(lasso, testing)

#Out of sample
lasso.results = postResample(pred = lasso.pred, obs = testing$logloss)
lasso.results
#RMSE   Rsquared 
# RMSE Rsquared 
#0.58136  0.08051 
sink()

#-----------------------------------------------------------------------------------------
#                      Model comparison: Training sample
#-----------------------------------------------------------------------------------------

resamps1 = resamples(list(RandomForest = rffit,
                          RidgeRegression = ridge,
                          Lasso = lasso,
                          StepwiseLinear  = lm.stepwise,
                          GradientBoosting = gbmfit ))

resamps1

resamps1$values

# summary of the models
summary(resamps1)

#RMSE plots
bwplot(resamps1,metric="RMSE",main="Model Comparison")	# boxplot
dotplot(resamps1,metric="RMSE",main="Model Comparison")	# dotplot


#Rsquared plots
bwplot(resamps1,metric="Rsquared",main="Model Comparison")	# boxplot
dotplot(resamps1,metric="Rsquared",main="Model Comparison")	# dotplot

#-----------------------------------------------------------------------------------------
#                     Model comparison table: Training and Test samples
#-----------------------------------------------------------------------------------------
# rffit2 values are used.
library(formattable)
DF <- data.frame(Model=c("Random Forest", "Gradient Boosting", "Lasso",
                         "Ridge Regression", "Linear Regression Stepwise"),
                 Training.RSquared=(c(0.8411, 0.7818, 0.08327, 0.08314, 0.03843)),
                 Training.RMSE=(c(0.2374, 0.2798, 0.5722, 0.5722, 0.5869)),
                 Test.RSquared=(c(0.8461, 0.7806, 0.08051,0.08011,0.0361  )),
                 Test.RMSE=(c(0.2397,0.286, 0.58136, 0.58149, 0.59544)))


#Sort based on AUC
DF <- DF[order(DF$Test.RMSE, decreasing=FALSE),]
formattable(DF)

formattable(DF, list(
  Test.RMSE = color_tile("lightgreen", "lightpink"),
  Test.RSquared = color_tile("lightpink", "lightgreen")))

DF
#-----------------------------------------------------------------------------------------
# Final Model: Random Forest. Target: logloss, Test.RMSE = 0.23970, TEST.MAE = 0.06621
#-----------------------------------------------------------------------------------------
#             Back transform the MAE to compare with Kaggle contestants.
#           https://www.kaggle.com/c/loan-default-prediction/leaderboard
#-----------------------------------------------------------------------------------------
testing_pred <- read.csv(file.path(Path,"testing_pred.csv"), stringsAsFactors=FALSE)

dim(testing_pred)

# (1) back transform loss
testing_pred$loss_back_tr = exp(testing_pred$logloss) - 1

# (2) back transform predicted loss
testing_pred$loss_back_tr_pred = exp(testing_pred$V91) - 1

# abs(1 - 2)
testing_pred$abs = abs( testing_pred$loss_back_tr - testing_pred$loss_back_tr_pred)

# sum of (1 - 2)^2
sum = sum(testing_pred$abs) 
sum

n = nrow(testing_pred)
n
#MAE
sqrt(sum/n)
#0.7149

#-----------------------------------------------------------------------------------------
#           After transforming back logloss to loss, the Test.MAE = 0.7149
#               Our rank would be 193 out of 675 in the leaderboard.
#-----------------------------------------------------------------------------------------
