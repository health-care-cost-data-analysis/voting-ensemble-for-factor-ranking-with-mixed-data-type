# Enter input.
# Where to load data.
dir_in = ".../dataset.csv";



# Where to save results.
dir_out = ".../output";


# Print process start time.
now = Sys.time();
# Define results folder name.
folder_name = paste0(format(now, "%Y%m%d_%H%M%S_"),"output");
# Define log file name.
result_name_txt = paste0(format(now, "%Y%m%d_%H%M%S_"),"output.txt");
# Define table file name.
result_name_csv = paste0(format(now, "%Y%m%d_%H%M%S_"),"output.csv");
# Define Diagnostics Report file name.
report_name = paste0(format(now, "%Y%m%d_%H%M%S_"),"Diagnostics_Report.doc");
# Define Short Diagnostics Report file name.
report_short_name = paste0(format(now, "%Y%m%d_%H%M%S_"),"High_level_Diagnostics_Report.doc");
# Test on partial data. Enter a fraction.
percentage = 1;
# Diagnostics on 3 subset values of response variable.
subset_1 = c("Private to Private", "Private to Uninsured");
subset_2 = c("Public to Private", "Public to Uninsured");
subset_3 = c("Uninsured to Uninsured", "Uninsured to Private");
subset_select = rbind(subset_1,subset_2,subset_3);

# Test on subset variables. The names should correspond to the column names of dataset.

# j=1
#subset = c("INSURC_CHANGE", "CHGJ34", "MARITALY2", "OBTOTV_DIFF", #"NUMBER_OF_COMORBIDITY", "FAMINCY2");
# j=2
#subset = c("INSURC_CHANGE", "REGIONY2", "RACEV1X", "FAMINC_DIFF", #"NUMBER_OF_COMORBIDITY", "FAMINCY2");
# j=3
#subset = c("INSURC_CHANGE", "CHGJ34", "MARITALY2", "RACEV1X", #"NUMBER_OF_COMORBIDITY", "FAMINCY2");


# Install packages.
# Random forests for regression and classification.
install.packages("randomForest");
library(randomForest);
# Convert dates to integers.
install.packages("lubridate");
library("lubridate");
# Remove constant variables.
install.packages ("mlr");
library (mlr);
# Perform subset selection.
install.packages ("leaps");
library (leaps);
# Ridge and lasso regressions.
install.packages("glmnet", dependencies=TRUE);
library(glmnet);
# Variable selection using random forests.
install.packages("VSURF");
library(VSURF);

install.packages("dataPreparation");
library(dataPreparation);


# Create a new folder for saving output.

if (! file.exists(dir_out)){
dir.create(dir_out, showWarnings = FALSE);
}
dir_out = file.path(dir_out,folder_name);
dir.create(dir_out);
setwd (dir_out);
sink(file.path(dir_out, result_name_txt));

cat ("\n\n\n\nDiagnostics on Policy Data\n\n\n\n");

# Test 0: Load and Prepare Data.
# Read sample policies.
raw_data= read.csv(dir_in, header = TRUE);

# Use a subset of sample policies for testing purpose.
set.seed(1);
train = sample(1:nrow(raw_data),nrow(raw_data) * percentage, replace=FALSE);
pre_raw_data = raw_data[train,];

# Define response variable.
names(pre_raw_data)[1]="INSURC_CHANGE";

# Replace NA with 0.
pre_raw_data = as.data.frame(pre_raw_data, stringsAsFactors = FALSE);
pre_raw_data[is.na(pre_raw_data)] = numeric(1);
hist_raw_data = pre_raw_data;
# Convert characters to factors.
# character_vars <- lapply(pre_raw_data, class) == "character";
# pre_raw_data[, character_vars] <- lapply(pre_raw_data[, character_vars], as.factor);
#for (j in 1:3) {
j=2;
# Print process start time.
now = Sys.time();
# Test on subset variables.
pre_raw_data = hist_raw_data[hist_raw_data[,1]%in%subset_select[j,], ];
pre_raw_data[,1]  = droplevels(pre_raw_data[,1]);

# Remove constant variables.
#index=whichAreConstant(pre_raw_data[,-1])+1;
#if(length(index) > 0){
#pre_raw_data= pre_raw_data[,-index]};

pre_raw_data$INSURC_CHANGE = factor(pre_raw_data$INSURC_CHANGE, levels=subset_select[j,])
# Garbage collection to spare more memory.
invisible(gc());
# Run logistic regression over full dataset, to convert indicators to log-odds.
glm.fit = glm(factor(INSURC_CHANGE)~., data = pre_raw_data, family = "binomial", control = glm.control(epsilon=1e-12, maxit = 10000));
lapse_rate = predict(glm.fit, type = "response");
logit_lapse_rate = log(lapse_rate/(1-lapse_rate));
logit_lapse_rate [is.na(logit_lapse_rate)] = 0;
mrm_raw_data = pre_raw_data[, -1];
mrm_raw_data$logit_lapse_rate = logit_lapse_rate;

# Test 1: Principal Component Analysis (PCA) for Detecting Linearity among Variables.
# Prepare pool of variables with interaction variables, for PCA.
pca_data = data.matrix (mrm_raw_data);

# Run PCA over the data.
pr.out = prcomp ( pca_data, scale = FALSE );

# Print principal component loadings of each principal component.
cat ("Test 1: PCA Analysis on All Variables");
cat ("\n\n\nResult 1.1: Principal Component Loadings of Each Principal Component:\n\n");
pr.out$rotation;

# Print the variance explained by each principal component.
cat ("\n\n\nResult 1.2: Variance Explained by Each Principal Component:\n\n");
pr.var = pr.out$sdev^2;
pr.var;

# Print the proportional of variance explained by each principal component.
cat ("\n\n\nResult 1.3: Proportional of Variance Explained by Each Principal Component:\n\n");
pve = pr.var/sum ( pr.var );
pve;

# Plot the cumulative proportion explained by the principal components.
png(file.path(dir_out, "Result_1_3_PCA_Cumulative_Proportion_of_Variance.png"));
plot ( cumsum ( pve ), xlab = "Number of Principal Components", ylab = "Cumulative Proportion Variance Explained", ylim = c ( 0.45, 1 ), type = 'b' );
dev.off();

# Garbage collection to spare more memory.
invisible(gc());

# Test 2: Variable Selection: Linear Regression on Full Data.
# Run linear regression.
mrm_data = mrm_raw_data;
lm.fit=lm(logit_lapse_rate~., mrm_data);
cat ("\n\n\nResult 2.1: Linear Regression on Full Data:\n\n");
summary(lm.fit);
# Garbage collection to spare more memory.
invisible(gc());

# Test 3: Variable Selection: Forward Stepwise Subset Selection.
# Run forward stepwise subset variables selection, using linear regression model:
regfit.fwd = regsubsets (logit_lapse_rate~., mrm_data, nvmax = 200, method = "forward", really.big=TRUE);
reg.summary = summary (regfit.fwd);
cat ("\n\n\nResult 3.1: Forward Subset Variable Selection:\n\n");
reg.summary;
np = length(reg.summary$bic);
for (i in 1:np){
list_data =list(coef(regfit.fwd,i));
cat("\n", file = file.path(dir_out, result_name_csv), append = TRUE);
list_name = paste("Result 3.1: Forward Subset Variable Selection: Model with",i, "Variables");
write.table(list_name, file = file.path(dir_out,result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = FALSE);
write.table(list_data, file = file.path(dir_out,result_name_csv), append = TRUE, sep = ',', row.names = TRUE, col.names = FALSE)};

# Plot the RSS, adjusted r-square, Cp and BIC for all the models.
png(file.path(dir_out, "Result_3_2_Forward_Subset_Measurements_Goodness_of_Fit.png"));

# Split screen into 2 by 2 windows.
par(mfrow = c(2,2));

# Plot the RSS of all the models, and highlight the one minimizing the RSS.
plot ( reg.summary$rss, xlab = "Number of Variables", ylab = "Residual Sum of Squares", type = "b" );
points ( which.min ( reg.summary$rss ), reg.summary$rss[which.min ( reg.summary$rss ) ], col = "red", cex = 2, pch = 20 );

# Plot the adjusted r-square, and highlight the one maximizing the adjusted r-square.
plot ( reg.summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", type="b");
points ( which.max ( reg.summary$adjr2 ), reg.summary$adjr2 [ which.max ( reg.summary$adjr2 ) ], col = "red", cex = 2, pch = 20 );

# Plot the Cp of all the models, and highlight the one minimizing the Cp.
plot ( reg.summary$cp, xlab = "Number of Variables", ylab = "Cp", type = "b" );
points ( which.min ( reg.summary$cp ), reg.summary$cp [ which.min ( reg.summary$cp ) ], col = "red", cex = 2, pch = 20 );

# Plot the BIC of all the models, and highlight the one minimizing the BIC.
plot ( reg.summary$bic,xlab = "Number of Variables", ylab = "BIC", type = "b" );
points ( which.min ( reg.summary$bic ), reg.summary$bic [ which.min ( reg.summary$bic ) ], col = "red", cex = 2, pch = 20 );
dev.off();

# Display the selected variables for the best model, by the particular statistical measure.
png(file.path(dir_out, "Result_3_2_Forward_Subset_QR_Code_Goodness_of_Fit.png"));

# Black squares at the top row denote the variable is selected by the best model.
par ( mfrow = c ( 2, 2 ) );
plot ( regfit.fwd, scale = "r2" );
plot ( regfit.fwd, scale = "adjr2" );
plot ( regfit.fwd, scale = "Cp" );
plot ( regfit.fwd, scale = "bic" );
dev.off();

# Display the coefficients estimates associated to the best fitting model.
cat ("\n\n\nResult 3.3: Forward Best Fitting Subset Variables and their Coefficients:\n\n");
coef ( regfit.fwd, which.min ( reg.summary$bic ) );
cat("\n", file = file.path(dir_out, result_name_csv), append = TRUE);
list_name = paste("Result 3.3: Forward Best Fitting Subset Variables and their Coefficients: with", which.min ( reg.summary$bic ), "Variables");
write.table(list_name, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = FALSE);
write.table(coef ( regfit.fwd, which.min ( reg.summary$bic ) ), file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = TRUE, col.names = FALSE);

# k-fold cross-validation for variable selection.
# Define logistic function.
logistic = function (x) {1 / (1+exp(-x))>=1/2};

# Define Prediction function for the method regsubsets().
predict.regsubsets = function (object, newdata, id, ...)
{ form = as.formula ( object$call[[2]] );
mat = model.matrix ( form, newdata );
coefi = coef ( object, id = id );
xvars = names ( coefi );
mat [ , xvars ]%*%coefi
}

# k-fold cross-validation for all the models with different subsets of variables.
k = 5;
set.seed ( 1 );
n_row = nrow (mrm_data);
n_col = ncol(mrm_data)-1;
folds = sample ( 1:k, n_row, replace = TRUE );
cv.errors = matrix ( NA, k, n_col, dimnames = list ( NULL, paste (1:n_col ) ) );
for (j in 1:k){
best.fit = regsubsets ( logit_lapse_rate~., data = mrm_data [folds!=j, ], nvmax = 200, method = "forward", really.big = TRUE);
response = as.numeric(pre_raw_data [folds == j,1])-1;
for ( i in 1:n_col ){
pred = predict(best.fit, mrm_data [folds == j, ], id = i);
cv.errors [j, i] = mean((response - logistic(pred))^2, na.rm=TRUE)}};
cv.fwd = apply ( cv.errors, 2, mean );
cat ("\n\n\nResult 3.4: Forward Subset: Misclassification Rates:\n\n");
cv.fwd[is.na(cv.fwd)]=0;
cv.fwd;
cat("\n", file = file.path(dir_out, result_name_csv), append = TRUE);
list_name = paste("Result 3.4: Forward Subset: Number of Variables VS Misclassification Rates");
write.table(list_name, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = FALSE);
write.table(cv.fwd, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = TRUE, col.names = FALSE);

# Plot the best model via k-fold cross-validation.
png(file.path(dir_out, "Result_3_5_Forward_Subset_CV_Errors.png"));
plot (cv.fwd, xlab = "Number of Variables", ylab = "Misclassification Rates", type = "b" );
points ( which.min(cv.fwd), cv.fwd [ which.min (cv.fwd)], col = "red", cex = 2, pch = 20 );
dev.off();

# Display the coefficients estimates associated to the best forecasting model.
cat ("\n\n\nResult 3.6: Forward Forecasting Subset Variables and their Coefficients:\n\n");
coef ( regfit.fwd, which.min (cv.fwd) );
cat("\n", file = file.path(dir_out, result_name_csv), append = TRUE);
list_name = paste("Result 3.6: Forward Best Forecasting Subset Variables and their Coefficients: with", which.min (cv.fwd) , "Variables");
write.table(list_name, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = FALSE);
write.table(coef ( regfit.fwd,which.min (cv.fwd)), file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = TRUE, col.names = FALSE);


# Garbage collection to spare more memory.
invisible(gc());

# Test 4: Variable Selection: Backward Stepwise Subset Selection.
# Run backward stepwise subset variables selection, using linear regression model:
regfit.bwd = regsubsets ( logit_lapse_rate~., mrm_data, nvmax = 200, method = "backward", really.big=TRUE);
reg.summary = summary ( regfit.bwd );
cat ("\n\n\nResult 4.1: Backward Subset Variable Selection:\n\n");
reg.summary;
np = length(reg.summary$bic);
for (i in 1:np){
list_data =list(coef(regfit.bwd,i));
cat("\n", file = file.path(dir_out, result_name_csv), append = TRUE);
list_name = paste("Result 4.1: Backward Subset Variable Selection: Model with",i, "Variables");
write.table(list_name, file = file.path(dir_out,result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = FALSE);
write.table(list_data, file = file.path(dir_out,result_name_csv), append = TRUE, sep = ',', row.names = TRUE, col.names = FALSE)};

# Plot the RSS, adjusted r-square, Cp and BIC for all the models.
png(file.path(dir_out, "Result_4_2_Backward_Subset_Measurements_Goodness_of_Fit.png"));

# Split screen into 2 by 2 windows.
par(mfrow = c(2,2));
# Plot the RSS of all the models, and highlight the one minimizing the RSS.
plot ( reg.summary$rss, xlab = "Number of Variables", ylab = "Residual Sum of Squares", type = "b" );
points ( which.min ( reg.summary$rss ), reg.summary$rss[which.min ( reg.summary$rss ) ], col = "red", cex = 2, pch = 20 );

# Plot the adjusted r-square, and highlight the one maximizing the adjusted r-square.
plot ( reg.summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", type="b");
points ( which.max ( reg.summary$adjr2 ), reg.summary$adjr2 [ which.max ( reg.summary$adjr2 ) ], col = "red", cex = 2, pch = 20 );

# Plot the Cp of all the models, and highlight the one minimizing the Cp.
plot ( reg.summary$cp, xlab = "Number of Variables", ylab = "Cp", type = "b" );
points ( which.min ( reg.summary$cp ), reg.summary$cp [ which.min ( reg.summary$cp ) ], col = "red", cex = 2, pch = 20 );

# Plot the BIC of all the models, and highlight the one minimizing the BIC.
plot ( reg.summary$bic,xlab = "Number of Variables", ylab = "BIC", type = "b" );
points ( which.min ( reg.summary$bic ), reg.summary$bic [ which.min ( reg.summary$bic ) ], col = "red", cex = 2, pch = 20 );
dev.off();

# Display the selected variables for the best model, by the particular statistical measure.
png(file.path(dir_out, "Result_4_2_Backward_Subset_QR_Code_Goodness_of_Fit.png"));

# Black squares at the top row denote the variable is selected by the best model.
par ( mfrow = c ( 2, 2 ) );
plot ( regfit.bwd, scale = "r2" );
plot ( regfit.bwd, scale = "adjr2" );
plot ( regfit.bwd, scale = "Cp" );
plot ( regfit.bwd, scale = "bic" );
dev.off();

# Display the coefficients estimates associated to the best fitting model.
cat ("\n\n\nResult 4.3: Backward Best Fitting Subset Variables and their Coefficients:\n\n");
coef ( regfit.bwd, which.min ( reg.summary$bic ) );
cat("\n", file = file.path(dir_out, result_name_csv), append = TRUE);
list_name = paste("Result 4.3: Backward Best Fitting Subset Variables and their Coefficients: with", which.min ( reg.summary$bic ), "Variables");
write.table(list_name, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = FALSE);
write.table(coef ( regfit.bwd, which.min ( reg.summary$bic ) ), file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = TRUE, col.names = FALSE);

# k-fold cross-validation for variable selection.
# k-fold cross-validation for all the models with different subsets of variables.
k = 5;
set.seed ( 1 );
n_row = nrow (mrm_data);
n_col =  ncol (mrm_data)-1;
folds = sample ( 1:k, n_row, replace = TRUE );
cv.errors = matrix ( NA, k, n_col, dimnames = list ( NULL, paste (1:n_col ) ) );
for (j in 1:k){
best.fit = regsubsets ( logit_lapse_rate~., data = mrm_data [folds!=j, ], nvmax = 200, method = "backward", really.big = TRUE);
response = as.numeric(pre_raw_data[folds == j,1])-1;
for ( i in 1:n_col ){
pred = predict(best.fit, mrm_data [folds == j, ], id = i);
cv.errors [j, i] = mean((response - logistic(pred))^2, na.rm=TRUE)}};
cv.bwd = apply ( cv.errors, 2, mean );
cat ("\n\n\nResult 4.4: Backward Subset: Misclassification Rates:\n\n");
cv.bwd[is.na(cv.bwd)]=0;
cv.bwd;
cat("\n", file = file.path(dir_out, result_name_csv), append = TRUE);
list_name = paste("Result 4.4: Backward Subset: Number of Variables VS Misclassification Rates");
write.table(list_name, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = FALSE);
write.table(cv.bwd, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = TRUE, col.names = FALSE);

# Plot the best model via k-fold cross-validation.
png(file.path(dir_out, "Result_4_5_Backward_Subset_CV_Errors.png"));
plot (cv.bwd,xlab = "Number of Variables", ylab = "Misclassification Rates", type = "b" );
points ( which.min(cv.bwd), cv.bwd [ which.min (cv.bwd) ], col = "red", cex = 2, pch = 20 );
dev.off();

# Display the coefficients estimates associated to the best forecasting model.
cat ("\n\n\nResult 4.6: Backward Forecasting Subset Variables and their Coefficients:\n\n");
coef ( regfit.bwd, which.min (cv.bwd) );
cat("\n", file = file.path(dir_out, result_name_csv), append = TRUE);
list_name = paste("Result 4.6: Backward Best Forecasting Subset Variables and their Coefficients: with", which.min (cv.bwd) , "Variables");
write.table(list_name, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = FALSE);
write.table(coef ( regfit.bwd,which.min (cv.bwd)), file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = TRUE, col.names = FALSE);


# Garbage collection to spare more memory.
invisible(gc());

# Test 5: Variable Selection: Forward and Backward Stepwise Logistic Regression with
# Categorical Responses.
mrm_data = pre_raw_data;
model = glm(INSURC_CHANGE~1, mrm_data, family = "binomial", control = glm.control(epsilon=1e-12, maxit = 10000));
biggest = formula(glm(INSURC_CHANGE~., mrm_data, family = "binomial", control = glm.control(epsilon=1e-12, maxit = 10000)));
fwd.model = step(model, direction = "forward", scope=biggest);
invisible(gc());
model = glm(INSURC_CHANGE~., mrm_data, family = "binomial", control = glm.control(epsilon=1e-12, maxit = 10000));
invisible(gc());
bwd.model = step(model, direction = "backward", scope=biggest);

# Best forward stepwise subset selection with categorical responses.
# Variables and coefficients.
cat ("\n\n\nResult 5.1: the Best Forward Stepwise Logistic Regression with Categorical Responses and their Coefficients:\n\n");
fwd_model.coef = coef(fwd.model);
cat("\n", file = file.path(dir_out, result_name_csv), append = TRUE);
list_name = paste("Result 5.1: the Best Forward Stepwise Logistic Regression with Categorical Responses and their Coefficients:");
write.table(list_name, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = FALSE);
write.table(fwd_model.coef, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = TRUE, col.names = FALSE);

# Garbage collection to spare more memory.
invisible(gc());
# Test misclassification rates.
k = 5;
set.seed ( 1 );
n_row = nrow ( mrm_data );
folds = sample ( 1:k, n_row, replace = TRUE );
cv.errors =numeric(k);
for (j in 1:k){
# Remove constant variables.
mrm_subset_data_1 = mrm_data[folds!=j,];
#index=whichAreConstant(mrm_subset_data_1[,-1])+1;
#if(length(index) > 0){
#mrm_subset_data_1= mrm_subset_data_1[,-index]};
mrm_subset_data_2 = mrm_data[folds==j,names(mrm_subset_data_1)];
model = glm(INSURC_CHANGE~1, mrm_subset_data_1, family = "binomial", control = glm.control(epsilon=1e-12, maxit = 10000));
biggest = formula(glm(INSURC_CHANGE~., mrm_subset_data_1, family = "binomial", control = glm.control(epsilon=1e-12, maxit = 10000)));
fwd.model = step(model, direction = "forward", scope=biggest);
pred = predict(fwd.model, mrm_subset_data_2, type = "response");
cv.errors [j] = mean((as.numeric(mrm_subset_data_2[,1]) -1- pred)^2)};
cv.fwd_ind = mean ( cv.errors, na.rm = TRUE);
cv.fwd_ind[is.na(cv.fwd_ind)]=0;
cat ("\n\n\nResult 5.1: CV Forward Indicator:\n\n");
cv.fwd_ind;

# Variables and coefficients.
cat ("\n\n\nResult 5.2: the Best Backward Stepwise Logistic Regression with Categorical Responses and their Coefficients:\n\n");
bwd_model.coef = coef(bwd.model);
cat("\n", file = file.path(dir_out, result_name_csv), append = TRUE);
list_name = paste("Result 5.2: the Best Backward Stepwise Logistic Regression with Categorical Responses and their Coefficients:");
write.table(list_name, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = FALSE);
write.table(bwd_model.coef, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = TRUE, col.names = FALSE);

# Garbage collection to spare more memory.
invisible(gc());

# Test misclassification rates.
k = 5;
set.seed ( 1 );
n_row = nrow ( mrm_data );
folds = sample ( 1:k, n_row, replace = TRUE );
cv.errors =numeric(k);
for (j in 1:k){
# Remove constant variables.
mrm_subset_data_1 = mrm_data[folds!=j,];
#index=whichAreConstant(mrm_subset_data_1[,-1])+1;
#if(length(index) > 0){
#mrm_subset_data_1= mrm_subset_data_1[,-index]};
mrm_subset_data_2 = mrm_data[folds==j,names(mrm_subset_data_1)];
model = glm(INSURC_CHANGE~., mrm_subset_data_1, family = "binomial", control = glm.control(epsilon=1e-12, maxit = 10000));
biggest = formula(glm(INSURC_CHANGE~., mrm_subset_data_1, family = "binomial", control = glm.control(epsilon=1e-12, maxit = 10000)));
bwd.model = step(model, direction = "backward", scope=biggest);
pred = predict(bwd.model, mrm_subset_data_2, type = "response");
cv.errors [j] = mean((as.numeric(mrm_subset_data_2[,1]) -1- pred)^2)};
cv.bwd_ind = mean ( cv.errors, na.rm=TRUE);
cv.bwd_ind[is.na(cv.bwd_ind)]=0;
cat ("\n\n\nResult 5.2: CV Backward Indicator:\n\n");
cv.bwd_ind;

# Garbage collection to spare more memory.
invisible(gc());

# Test 6: Variable Selection: the lasso.
# Remove constant variables.
#index=whichAreConstant(pre_raw_data[,-1])+1;
#if(length(index) > 0){
#pre_raw_data= pre_raw_data[,-index]};

mrm_data = pre_raw_data;
mrm_data[,1] = as.numeric(mrm_data[,1])-1;

# x is the set of factors. y is the response variable. grid is candidate values of lambda.

x = model.matrix (INSURC_CHANGE~.,mrm_data) [, -1];
y = as.factor(mrm_data[,1]);
y = droplevels(y);
grid = 10^seq (1,-4, length = 10^3);

# Compute the best lasso model coefficients, by selecting the lambda which minimizes the fitting error.
set.seed(1);
lasso.mod = glmnet (x, y, family = "binomial", alpha = 1, lambda = grid, thresh = 1e-12);
cv.out = cv.glmnet (x, y, family = "binomial", alpha = 1, lambda = grid, thresh = 1e-12);
best_lambda = cv.out$lambda.min;
lasso_model.coef = predict (lasso.mod, type = "coefficients", s = best_lambda);
lasso_model.coef = lasso_model.coef[-length(lasso_model.coef),];
cat ("\n\n\nResult 6.1: the lasso and their Coefficients:\n\n");
lasso_model.coef;
cat("\n", file = file.path(dir_out, result_name_csv), append = TRUE);
list_name = paste("Result 6.1: the lasso and their Coefficients:");
write.table(list_name, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = FALSE);
write.table(lasso_model.coef, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = TRUE, col.names = FALSE);
# Garbage collection to spare more memory.
invisible(gc());

# Test 7: Variable Selection: Random Forests.
# Create candidate variables and response.
mrm_data = pre_raw_data;
x = model.matrix (INSURC_CHANGE~., mrm_data) [, -1];
y = as.factor(mrm_data[,1]);
# Perform variable selection.
set.seed(1);
train = sample (1:nrow(x), nrow(x) ,replace=FALSE);
data.vsurf = VSURF (x = x[train,], y = y[train], ntree=100,mtry = floor(ncol(mrm_data)/3));

cat ("\n\n\nResult 7.1: Variable Selection using Random Forests:\n\n");
summary (data.vsurf);
png(file.path(dir_out, "Result_6_2_ Variable_Selection_using_Random_Forests.png"));
plot (data.vsurf);
dev.off();
# List the best subset variables for fitting performance.
cat ("\n\n\nResult 7.2: Thresholding Variables Selected by Random Forests:\n\n");
colnames(x)[data.vsurf$varselect.thres];
cat("\n", file = file.path(dir_out, result_name_csv), append = TRUE);
list_name = paste("Result 7.2: Thresholding Variables Selected by Random Forests:");
write.table(list_name, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = FALSE);
write.table(colnames(x)[data.vsurf$varselect.thres], file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = TRUE, col.names = FALSE);
# List the best subset variables for fitting performance.
cat ("\n\n\nResult 7.3: Interpretation Variables Selected by Random Forests:\n\n");
importance_rf  = colnames(x)[data.vsurf$varselect.interp];
importance_rf;
cat("\n", file = file.path(dir_out, result_name_csv), append = TRUE);
list_name = paste("Result 6.3: Interpretation Variables Selected by Random Forests:");
write.table(list_name, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = FALSE);
write.table(colnames(x)[data.vsurf$varselect.interp], file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = TRUE, col.names = FALSE);
# List the best subset variables for prediction performance.
cat ("\n\n\nResult 7.4: Prediction Variables Selected by Random Forests:\n\n");
colnames(x)[data.vsurf$varselect.pred];
cat("\n", file = file.path(dir_out, result_name_csv), append = TRUE);
list_name = paste("Result 7.4: Prediction Variables Selected by Random Forests:");
write.table(list_name, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = FALSE);
write.table(colnames(x)[data.vsurf$varselect.pred], file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = TRUE, col.names = FALSE);
# Garbage collection to spare more memory.
gc();

# Test 8: Alternative Methods.
# Method 1: Regression Tree.
# Fit regression tree.
set.seed(1);
mrm_data = pre_raw_data;
train = sample(1:nrow(mrm_data),nrow(mrm_data)/2,replace=FALSE);
tree.fit=randomForest(factor(INSURC_CHANGE)~., data=mrm_data,subset=train, mtry= floor(ncol(mrm_data)/3),ntree=100,importance=TRUE);

# Use the regression tree to predict the other half data and compute its test misclassification rates.
yhat=as.numeric(predict(tree.fit,newdata=mrm_data[-train,]))-1;
tree.test=as.numeric(mrm_data[-train,1])-1;
cat ("\n\n\nResult 8.4: CV Tree:\n\n");
cv.tree = mean((yhat-tree.test)^2);
cv.tree[is.na(cv.tree)]=0;
cv.tree;

# Method 2: Ridge regression.
mrm_data = pre_raw_data;

mrm_data[,1] = as.numeric(mrm_data[,1])-1;

# x is the set of factors. y is the response variable. grid is candidate values of lambda.
x = model.matrix (INSURC_CHANGE~., mrm_data) [, -1];
y = as.factor(mrm_data[,1]);
grid = 10^seq (1,-4, length = 10^3);
# Use half-set cross validation to compute the misclassification rates.
set.seed(100);
train = sample (1:nrow(x), nrow(x) / 2);
y.test = y[-train];
ridge.mod = glmnet (x[train,], y[train], family = "binomial", alpha = 0, lambda = grid, thresh = 1e-12);
cv.out = cv.glmnet (x[train,], y[train], family = "binomial", alpha = 0, lambda = grid, thresh = 1e-12);
best_lambda = cv.out$lambda.min;
ridge.pred = predict (ridge.mod, s = best_lambda, newx = x [-train,], type="class");
ridge.coef = predict (ridge.mod, type = "coefficients", s = best_lambda);
ridge.coef = ridge.coef[-length(ridge.coef),];
cv.ridge = mean ( (as.numeric(ridge.pred) - as.numeric(y.test)+1)^2, na.rm=TRUE );
cv.ridge[is.na(cv.ridge)]=0;
cat ("\n\n\nResult 8.5: CV Ridge:\n\n");
cv.ridge;

# Method 3: Lasso regression.
set.seed(50);
train = sample (1:nrow(x), nrow(x) / 2);
y.test = y[-train];

lasso.mod = glmnet (x[train,], y[train], family = "binomial", alpha = 1, lambda = grid, thresh = 1e-12);
cv.out = cv.glmnet (x[train,], y[train], family = "binomial", alpha = 1, lambda = grid, thresh = 1e-12);
best_lambda = cv.out$lambda.min;
lasso.pred = predict (lasso.mod, s = best_lambda, newx = x [-train,], type="class");
lasso.coef = predict (lasso.mod, type = "coefficients", s = best_lambda);
lasso.coef = lasso.coef[-length(lasso.coef),];
cv.lasso = mean ( (as.numeric(lasso.pred) - as.numeric(y.test)+1)^2, na.rm = TRUE );
cv.lasso[is.na(cv.lasso)]=0;
cat ("\n\n\nResult 8.6: CV Lasso:\n\n");
cv.lasso;

# Recommended model.


# Use k-fold cross-validation to estimate the misclassification rates of the current model.
subset = c("INSURC_CHANGE","obtotvy1","obtotv_diff","famszey1","chronic_y1","agey1x","famincy1","move_y1","year","racev1x","regiony1","currently_smoke","faminc_diff");
mrm_data = pre_raw_data[,subset];
current.fit = glm (INSURC_CHANGE~., data = mrm_data, family=binomial, control = glm.control(epsilon=1e-12, maxit = 10000));
k = 5;
set.seed ( 1 );
n_row = nrow ( mrm_data );
folds = sample ( 1:k, n_row, replace = TRUE );
cv.errors =numeric(k);
for (j in 1:k){
current = glm (factor(INSURC_CHANGE)~., data = mrm_data [folds!=j, ], family=binomial, control = glm.control(epsilon=1e-12, maxit = 10000));
pred = predict(current, mrm_data[folds == j, ], type = "response");
cv.errors [j] = mean((as.numeric(mrm_data [folds == j,1]) -1- pred)^2)};
cv.current = mean ( cv.errors, na.rm = TRUE);
cv.current[is.na(cv.current)]=numeric(1);
cat ("\n\n\nResult 8.7: CV Current:\n\n");
cv.current;

# Compare the test misclassification rates of the logit function of the lapse rates of the 5 models: best,
# logistic regression, tree, ridge and lasso, current.
misclassification_rates=c(min(cv.fwd), min(cv.bwd), cv.fwd_ind, cv.bwd_ind, cv.tree, cv.ridge, cv.lasso, cv.current);
cat ("\n\n\nResult 8.8: Comparison of all Models – Misclassification Rate:\n\n");
list_data = list(c("reg fwd rate","reg bwd rate", "reg fwd ind","reg bwd ind","random forest","ridge","lasso","current"), misclassification_rates);
list_data;
cat("\n", file = file.path(dir_out, result_name_csv), append = TRUE);
list_name = paste("Result 8.8: Comparison of all Models – Misclassification Rate:");
write.table(list_name, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = FALSE);
write.table(list_data, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = TRUE, col.names = FALSE);
png(file.path(dir_out, "Result_7_5_ Comparison_of_Predictive_Models.png"));
n=length(misclassification_rates);
plot (1:n, misclassification_rates,xlab = "Models", ylab = "Prediction Misclassification Rates", type = "b", xaxt = "n");
points ( which.min(misclassification_rates), misclassification_rates [ which.min (misclassification_rates) ], col = "red", cex = 2, pch = 20 );
points ( n, misclassification_rates [ n ], col = "blue", cex = 2, pch = 20 );
axis (1, at = 1:n, labels = c("reg fwd rate","reg bwd rate", "reg fwd ind", "reg bwd ind", "rf","ridge","lasso","current"), las = 2);
dev.off();

# List the variables selected by the 8 models.
# Generate grade() function. It is used to compute the standardized coefficient.
pre_raw_data_table = pre_raw_data;
for(i in 1:ncol(pre_raw_data))
{pre_raw_data_table[,i] = as.character(pre_raw_data[,i]);
for(j in 1:nrow(pre_raw_data))
{pre_raw_data_table[j,i]=paste(names(pre_raw_data)[i], pre_raw_data[j,i], sep = "");
}
};

variable_table = pre_raw_data_table[,-1];
for(i in which(sapply(pre_raw_data[,-1], is.numeric)))
{variable_table[,i] = names(variable_table)[i];
};

# Obtain all predictor names and levels.
variables_full = unique(unlist(variable_table));
# Add intercept.
variables_full = c(names(coef(lm.fit))[1],variables_full);
variables_full = unique(c(colnames(summary(regfit.fwd)$which), colnames(summary(regfit.bwd)$which),variables_full));

log_odds = function(x, p_low, p_up){x = min(max(x, p_low), p_up); log(x/(1-x))};

grade = function(data, l, variables_full, variables_sub, coefficient, table)
{
output = variables_sub;
N=nrow(data);
for( i in 1:l){if (variables_sub[i] ==1){if (variables_full[i] %in%names (data)){output[i] = sd(as.numeric(data[,variables_full[i]]))*as.numeric(coefficient[variables_full[i]])} else {
n= sum(table==as.character(variables_full[i]));
output[i]= sd(c(rep(1,n),rep(0,N-n)))*as.numeric(coefficient[variables_full[i]])}}};
output
};

grade_numeric_rf = function(data, l, variables_full, variables_sub)
{
N=nrow(data);
p_low = 1/N;
p_up = (N-1)/N;
set.seed(1);
tree.fit=randomForest(factor(INSURC_CHANGE)~., data=data , mtry= floor(ncol(data)/3),ntree=100);
output = variables_sub;
for( i in 1:l){
if ((variables_sub[i] ==1) & (variables_full[i] %in%names (data))){
U = sort(unique(data[, names(data)%in%variables_full[i]]));
L = length(U);
v  = numeric(L);
x=U;
for (j in 1:L)
{new_data = data;
new_data[, names(new_data)%in%variables_full[i]] = U[j];
set.seed(1);
v[j]= log_odds(mean(as.numeric(predict(tree.fit,newdata=new_data))-1), p_low, p_up);
}
r = sd(as.numeric(data[,variables_full[i]]))* mean((v[2:L]-v[1:(L-1)])/(x[2:L]-x[1:(L-1)]), na.rm=TRUE);
r[is.na(r)] = 0;
output[i] = r[1];
}}
output
};

grade_categorical_rf = function(data, l, variables_full, variables_sub, table)
{
N=nrow(data);
p_low = 1/N;
p_up = (N-1)/N;
set.seed(1);
tree.fit=randomForest(factor(INSURC_CHANGE)~., data=data , mtry= floor(ncol(data)/3),ntree=100);
output = variables_sub;
for( i in 1:l){
if ((variables_sub[i] ==1) & ! (variables_full[i] %in%names (data))){
variable_index = (table==variables_full[i]);
n= sum(variable_index);
variable_index_col = apply (variable_index, 2, function (r) {any(r^2%in%c(1))});
variable_index_raw = apply (variable_index, 1, function (r) {any(r^2%in%c(1))});
set.seed(1);
tree.fit_orig=randomForest(factor(INSURC_CHANGE)~., data=data[,!variable_index_col] , mtry= floor(ncol(data[,!variable_index_col])/3),ntree=100);
new_data = data;
new_data[,variable_index_col] = unique(data[variable_index]);
new_data = rbind (data[1,], new_data);
new_data = new_data[-1,];
set.seed(1);
v= log_odds(mean(as.numeric(predict(tree.fit, newdata=new_data))-1), p_low, p_up) - log_odds(mean(as.numeric(predict(tree.fit_orig, newdata= data[,!variable_index_col]))-1), p_low, p_up);
r = sd(c(rep(1,n),rep(0,N-n)))*v;
r[is.na(r)] = 0;
output[i] = r[1];
}}
output
};


# Generate impact() function. It is used to compute the standardized coefficient.
impact = function(x)
{
y = x/abs(x);
y[is.na(y)]=0;
y[y==1] = "positive";
y[y==-1] = "negative";
y[y==0] = "not applicable";
y
};
l = length(variables_full);
coefficient_fwd = coef (regfit.fwd, which.min ( cv.fwd ) );
variables_fwd = (variables_full%in%names(coefficient_fwd))^2;
grade_fwd_raw = grade(pre_raw_data, l, variables_full, variables_fwd, coefficient_fwd, pre_raw_data_table);
impact_fwd = impact(grade_fwd_raw);
grade_fwd = abs(grade_fwd_raw);
coefficient_bwd = coef (regfit.bwd, which.min ( cv.bwd ) );
variables_bwd = (variables_full%in%names(coefficient_bwd))^2;
grade_bwd_raw = grade(pre_raw_data, l, variables_full, variables_bwd, coefficient_bwd, pre_raw_data_table);
impact_bwd = impact(grade_bwd_raw);
grade_bwd = abs(grade_bwd_raw);
variables_fwd_ind = (variables_full%in%names(fwd_model.coef))^2;
grade_fwd_ind_raw = grade(pre_raw_data, l, variables_full, variables_fwd_ind, fwd_model.coef, pre_raw_data_table);
impact_fwd_ind = impact(grade_fwd_ind_raw);
grade_fwd_ind = abs(grade_fwd_ind_raw);
variables_bwd_ind = (variables_full%in%names(bwd_model.coef))^2;
grade_bwd_ind_raw = grade(pre_raw_data, l, variables_full, variables_bwd_ind, bwd_model.coef, pre_raw_data_table);
impact_bwd_ind = impact(grade_bwd_ind_raw);
grade_bwd_ind = abs(grade_bwd_ind_raw);
coefficient_lasso = lasso_model.coef[lasso_model.coef!=0];
variables_lasso = (variables_full%in%names(coefficient_lasso))^2;
grade_lasso_raw = grade(pre_raw_data, l, variables_full, variables_lasso, coefficient_lasso, pre_raw_data_table);
grade_lasso_raw[is.na(grade_lasso_raw)] = 0;
impact_lasso = impact(grade_lasso_raw);
grade_lasso = abs(grade_lasso_raw);
variables_rf = (variables_full%in%importance_rf)^2;
grade_rf_raw = grade_numeric_rf(pre_raw_data, l, variables_full, variables_rf) + grade_categorical_rf(pre_raw_data, l, variables_full, variables_rf, pre_raw_data_table);
impact_rf = impact(grade_rf_raw);
grade_rf = abs(grade_rf_raw);

cat ("\n\n\nResult 8.9: Comparison of all Models – Best Subset Variables:\n\n");
list_data = cbind((1:l), variables_full, variables_fwd_ind,variables_bwd_ind, variables_lasso, variables_rf);
colnames(list_data) = c("Number","Full Variable", "reg fwd ind", "reg bwd ind", "lasso", "rf");
list_data;
cat("\n", file = file.path(dir_out, result_name_csv), append = TRUE);
list_name = paste("Result 8.9: Comparison of all Models – Best Subset Variables:");
write.table(list_name, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = FALSE);
write.table(list_data, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = TRUE);
ranking = variables_fwd_ind + variables_bwd_ind + variables_lasso + variables_rf;
variables_ranking_6 = (ranking==6)^2;
variables_ranking_5 = (ranking==5)^2;
variables_ranking_4 = (ranking==4)^2;
variables_ranking_3 = (ranking==3)^2;
variables_ranking_2 = (ranking==2)^2;
variables_ranking_1 = (ranking==1)^2;
variables_ranking_0 = (ranking==0)^2;
list_data = cbind((1:l), variables_full, variables_ranking_4, variables_ranking_3, variables_ranking_2, variables_ranking_1, variables_ranking_0);
colnames(list_data) = c("Number","Variable","4", "3", "2", "1", "0");
list_data;
cat("\n", file = file.path(dir_out, result_name_csv), append = TRUE);
list_name = paste("Result 8.10: Comparison of all Models – Variables Rankings:");
write.table(list_name, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = FALSE);
write.table(list_data, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = TRUE);
variables_ranking_6 = (ranking>=6)^2;
variables_ranking_5 = (ranking>=5)^2;
variables_ranking_4 = (ranking>=4)^2;
variables_ranking_3 = (ranking>=3)^2;
variables_ranking_2 = (ranking>=2)^2;
variables_ranking_1 = (ranking>=1)^2;
list_data = cbind((1:l), variables_full, variables_ranking_4, variables_ranking_3, variables_ranking_2, variables_ranking_1);
colnames(list_data) = c("Number","Variable",">=4", ">=3", ">=2", ">=1");
list_data;
cat("\n", file = file.path(dir_out, result_name_csv), append = TRUE);
list_name = paste("Result 8.11: Comparison of all Models – Cumulative Variables Rankings:");
write.table(list_name, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = FALSE);
write.table(list_data, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = TRUE);
grades = round(grade_fwd_ind+grade_bwd_ind+grade_lasso+grade_rf,8)/4;
grades_impact = impact(grade_fwd_ind_raw+grade_bwd_ind_raw+grade_lasso_raw+grade_rf_raw);
ranking = cbind(variables_full,ranking,grades, grades_impact);
ranking_sort = ranking[order(ranking[,2], decreasing = TRUE),];
for (i in 1:5) { if (sum(as.numeric(ranking_sort[,2])==(5-i))>1){r=ranking_sort[ranking_sort[,2]==(5-i),]; r=as.matrix(r); ranking_sort[ranking_sort[,2]==(5-i),]  = r[order(as.numeric(r[,3]), decreasing = TRUE),]}};
list_data = cbind((1:l), ranking_sort);
colnames(list_data) = c("Number","Variable","Ranking", "Grades", "Impact");
list_data;
cat("\n", file = file.path(dir_out, result_name_csv), append = TRUE);
list_name = paste("Result 8.14: Comparison of all Models – Variables Standings:");
write.table(list_name, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = FALSE);
write.table(list_data, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = TRUE);
ranking_sort = ranking[order(as.numeric(ranking[,3]), decreasing = TRUE),];
list_data = cbind((1:l), ranking_sort);
colnames(list_data) = c("Number","Variable","Ranking", "Grades", "Impact");
list_data;
cat("\n", file = file.path(dir_out, result_name_csv), append = TRUE);
list_name = paste("Result 8.13: Comparison of all Models – Variables Grades:");
write.table(list_name, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = FALSE);
write.table(list_data, file = file.path(dir_out, result_name_csv), append = TRUE, sep = ',', row.names = FALSE, col.names = TRUE);
end = Sys.time();
sink (result_name_txt, append = TRUE, split = TRUE);

# Generate diagnostics report.
# Use the package “rtf”.
install.packages("rtf");
library(rtf);
# Create report.
report = RTF(report_name);
# Add Title, date and table of content.
addHeader(report, "              Diagnostics Result Report (Full Version)", subtitle = "                   ------ Experience Study, AIG ", font.size = 20);
addParagraph (report, "Data analytics process started at ",now," (",Sys.timezone(),")");
addParagraph (report, "Data analytics process ended at ",end," (",Sys.timezone(),")");
start = as.POSIXct(now);
ends = as.POSIXct(end);
addParagraph (report, "The time duration is ",difftime(ends, start, units="hours"), " hours");
addParagraph (report, "The diagnostics are performed on ",percentage*100, "% of records in the dataset, i.e., ", nrow(pre_raw_data), " out of a total number of ",nrow(raw_data), " records");
addParagraph (report, "Report generated at ",Sys.time()," (",Sys.timezone(),")");
addNewLine(report, n = 1);
addParagraph (report, "Response factors include: ", levels(pre_raw_data$INSURC_CHANGE)[1], ", ",levels(pre_raw_data$INSURC_CHANGE)[2]);
addNewLine(report, n = 1);
addText (report, "Table of Content", bold = TRUE);
addNewLine(report, n = 1);
addParagraph (report, "    Test 0: Dataset Information");
addParagraph (report, "    Test 1: Principal Component Analysis (PCA) for Detecting Linearity among Variables");
addParagraph (report, "        Result 1.1: Principal Component Loadings of Each Principal Component");
addParagraph (report, "        Result 1.2: Variance Explained by Each Principal Component");
addParagraph (report, "        Result 1.3: Proportion of Variance Explained by Each Principal Component");
addNewLine(report, n = 1);
addParagraph (report, "    Test 2: Variable Selection: Subset, the Lasso, Random Forests");
addParagraph (report, "        Test 2.1: Variable Selection: Forward Stepwise Subset Selection");
addParagraph (report, "            Result 2.1.1: Linear Regression on Full Dataset");
addParagraph (report, "            Result 2.1.2: Forward Stepwise Subset Variable Selection");
addParagraph (report, "            Result 2.1.3: Forward Stepwise Subset Selection – Measurements of Goodness of Fit");
addParagraph (report, "            Result 2.1.4: Forward Stepwise Best Fitting Subset Variables and their Coefficients");
addParagraph (report, "            Result 2.1.5: Forward Stepwise Subset: Number of Variables VS Misclassification Rates");
addParagraph (report, "            Result 2.1.6: Forward Stepwise Forecasting Subset Variables and their Coefficients");
addParagraph (report, "        Test 2.2: Variable Selection: Backward Stepwise Subset Selection");
addParagraph (report, "            Result 2.2.1: Backward Stepwise Subset variable Selection");
addParagraph (report, "            Result 2.2.2: Backward Stepwise Subset Selection – Measurements of Goodness of Fit");
addParagraph (report, "            Result 2.2.3: Backward Stepwise Best Fitting Subset Variables and their Coefficients");
addParagraph (report, "            Result 2.2.4: Backward Stepwise Subset: Number of Variables VS Misclassification Rates");
addParagraph (report, "        Test 2.3: Variable Selection: Best Subset Selection with Indicator");
addParagraph (report, "            Result 2.3.1: Forward Stepwise Best Subset Variables with Indicator Responses");
addParagraph (report, "            Result 2.3.2: Backward Stepwise Best Subset Variables with Indicator Responses");
addParagraph (report, "        Test 2.4: Variable Selection: the Lasso");
addParagraph (report, "            Result 2.4.1: the Lasso and its Coefficients");
addParagraph (report, "        Test 2.5: Variable Selection: Random Forests");
addParagraph (report, "            Result 2.5.1: Summary: Variable Selection using Random Forests");
addParagraph (report, "            Result 2.5.2: Figure: Variable Selection using Random Forests");
addParagraph (report, "            Result 2.5.3: Thresholding Variables Selected by Random Forests");
addParagraph (report, "            Result 2.5.4: Interpretation Variables Selected by Random Forests");
addParagraph (report, "            Result 2.5.5: Prediction Variables Selected by Random Forests");
addParagraph (report, "            Result 2.5.6: Figure: Importance of Variables using Random Forests");
addNewLine(report, n = 1);
addParagraph (report, "    Test 3: Model Selection: Logistic Regression, Random Forests, Ridge, the Lasso");
addParagraph (report, "        Result 3.1: Ridge Regression and its Coefficients");
addParagraph (report, "        Result 3.2: The Lasso and its Coefficients");
addParagraph (report, "        Result 3.3: Comparison of the Models: Full Subset, Forward Stepwise Subset, Backward Stepwise Subset, Random Forests, Ridge, Lasso");
addNewLine(report, n = 1);
addParagraph (report, "    Conclusion 4: Best Subset Variables");
addParagraph (report, "        Result 4.1: Best Subset Variables for Each Model");
addParagraph (report, "        Result 4.2: Best Subset Variables Selected by All Models");
addParagraph (report, "        Result 4.3: Comparison of All Models");
addPageBreak(report);
# Test 0: Dataset Information.
addText (report, " Test 0: Dataset Information.", bold = TRUE);
addNewLine(report, n = 1);
addParagraph (report, "The dataset consists of ", nrow(pre_raw_data), " monthly valuation records and each record contains information on 1 response variable and ", ncol(pre_raw_data)-1, " predictor variables.");
addNewLine(report, n = 1);
addParagraph (report, "The list of response variables is given below:");
addNewLine(report, n = 1);
list_data = sapply(pre_raw_data["INSURC_CHANGE"], class);
list_level = sapply(pre_raw_data["INSURC_CHANGE"], nlevels);
table = as.data.frame(cbind(1:length(list_data), names(list_data), list_data, list_level));
colnames(table)[1:4] = c("Number", "Response Variable", "Type", "Number of Categories");
addTable(report,table);
addNewLine(report, n=1);
addParagraph (report, "The list of predictor variables is given below:");
addNewLine(report, n = 1);
list_data = sapply(pre_raw_data[, !(names(pre_raw_data)%in% "INSURC_CHANGE")], class);
list_level = sapply(pre_raw_data[, !(names(pre_raw_data)%in% "INSURC_CHANGE")], nlevels);
list_level[list_level==0]= "-";
table = as.data.frame(cbind(1:length(list_data), names(list_data), list_data, list_level));
colnames(table)[1:4] = c("Number", "Predictor Variable", "Type", "Number of Categories");
addTable(report,table);
addNewLine(report, n=1);
addParagraph (report, "We consider the following 4 approaches for subset variables selection:");
addNewLine(report, n = 1);
method = c("fwd logistic regression", "bwd logistic regression", "lasso", "random forests");
measure = c("AIC", "AIC", "Misclassification Error", "Out of Bagging Error");
table = as.data.frame(cbind(1:length(method),method, measure));
colnames(table)[1:3] = c("Number", "Subset Selection Method", "Criterion");
addTable(report,table);
addNewLine(report, n=1);
addParagraph (report, "We consider the following 4 models for fitting the response variable:");
addNewLine(report, n = 1);
model = c("logistic regression", "ridge", "lasso", "random forests");
feature = c("low bias, easy to implement", "high bias, low variance", "high bias, remove variables", "low bias, hard to implement");
table = as.data.frame(cbind(1:length(model),model, feature));
colnames(table)[1:3] = c("Number", "Predictive Model", "Feature");
addTable(report,table);
addNewLine(report, n=1);
# Test 1: Principal Component Analysis (PCA) for Detecting Linearity among Variables.
addText (report, " Test 1: Principal Component Analysis (PCA) for Detecting Linearity among Variables.", bold = TRUE);
addNewLine(report, n = 1);
addParagraph (report, "Goal: detecting linearity among the variables.");
addNewLine(report, n = 1);
# Test 1.1: Print principal component loadings of each principal component.
addText (report, "Result 1.1: Principal Component Loadings of Each Principal Component:", bold = TRUE);
addNewLine(report, n = 1);
addParagraph (report, "The table below presents the coefficients of all variables in each principal component.");
addNewLine(report, n = 1);
table = cbind(rownames(pr.out$rotation), round(pr.out$rotation,10));
colnames(table)[1] = "Variable";
addTable(report,table);
addNewLine(report, n=1);
# Test 1.2: Print variance explained by each principal component.
addText (report, "Result 1.2: Variance Explained by Each Principal Component:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The table below presents the variance of the entire data explained by each principal component.");
addNewLine(report, n=1);
table = cbind(colnames(pr.out$rotation), round(pr.var,10));
colnames(table)[1:2] = c("PC","Variance Explained");
addTable(report,table);
addNewLine(report, n=1);
# Test 1.3: Print the proportion of variance explained by each principal component.
addText (report, "Result 1.3: Proportion of Variance Explained by Each Principal Component:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The table below presents the proportion of variance explained by each principal component.");
addNewLine(report, n=1);
table = cbind(colnames(pr.out$rotation), round(pve,10));
colnames(table)[1:2] = c("PC","Proportional of Variance Explained");
addTable(report,table);
addNewLine(report, n=1);
# Plot the cumulative proportion explained by the principal components.
addText (report, "Result 1.3: Figure 1.1: Proportional of Variance Explained by Each Principal Component:", bold = TRUE);
addNewLine(report, n = 1);
addParagraph (report, "The figure below illustrates the cumulative proportions of variance explained by the principal components. Higher proportion of the first principal component indicates more likely the linear model is a good fit.");
addPng(report, "Result_1_3_PCA_Cumulative_Proportion_of_Variance.png", width = 6, height = 6);
# Test 2: Variable Selection.
# Test 2.1: Variable Selection: Forward Stepwise Subset Selection.
addText (report, "Test 2.1: Variable Selection: Forward Stepwise Subset Selection.", bold = TRUE);
addNewLine(report, n = 1);
# Test 2.1.1: Run linear regression on full dataset.
addParagraph (report, "Goal: Select the subset of variables which best explains the response variable, using forward stepwise subset selection approach.");
addNewLine(report, n = 1);
addText (report, "Result 2.1.1: Linear Regression on Full Data:", bold = TRUE);
addNewLine(report, n = 1);
addParagraph (report, "The table below presents the statistics from the ordinary linear regression over all variables. It roughly illustrates the importance of each variable on explaining the values of the corresponding variable.");
addNewLine(report, n = 1);
out = function (s1)
{
# coefficients
r1 <- round(coef(s1), 9);
# summary statistics
sigma <- round(s1$sigma, 9);
rsq <- round(s1$adj.r.squared, 9);
# sample sizes
sample_size <- length(s1$residuals);
outtab <- rbind(r1, sigma, rsq, sample_size);
rownames(outtab) <- c(rownames(coef(s1)), "sigma", "Adj. R-Squared", "sample size");
outtab
}
table = cbind(rownames(out(summary(lm.fit))), out(summary(lm.fit)));
colnames(table)[1] = "Variable";
addTable(report,table);
addNewLine(report, n = 1);
# Test 2.1.2: Run forward stepwise subset variables selection, using linear regression model.
addText (report, "Result 2.1.2: Forward Stepwise Subset Variable Selection:", bold = TRUE);
addNewLine(report, n = 1);
addParagraph (report, "The table below presents the forward stepwise best subset of i variables,  for i going from 1 to", np, "through comparing the RSS. * denotes that the corresponding variable is selected into the subset.");
addNewLine(report, n=1);
table = summary (regfit.fwd)$outmat;
rownames(table) = 1:nrow(table);
table = cbind(rownames(table),table)
colnames(table)[1] = "Number of Variables";
addTable(report,table,header.col.justify = "L", font.size = 5);
addNewLine(report, n=1);
addParagraph (report, "The tables below present the variables and coefficients of the best models with i variables, for i going from 1 to", np, ".");
addNewLine(report, n=1);
for (i in 1:np){
addText (report, "Result 2.1.2: Forward Stepwise Subset Variable Selection: Model with ",i, " Variables:", bold = TRUE);
addNewLine(report, n=1);
table = cbind((1:length(coef(regfit.fwd,i)))-1, names(coef(regfit.fwd,i)), round(coef(regfit.fwd,i),10));
colnames(table)[1:3] = c("Number", "Variable", "Coefficient");
addTable(report,table);
addNewLine(report, n=1);
}
# Test 2.1.3: Plot the RSS, adjusted r-square, Cp and BIC for all the models.
addText (report, "Result 2.1.3: Figure 2.1.1: Forward Stepwise Subset Selection – Measurements of Goodness of Fit:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The figure below presents the RSS, adjusted r-square, Cp and BIC for the forward stepwise best models with i variables, for i going from 1 to", np, ". The red point indicates the number of variables which optimizes the corresponding measurement.");
addPng(report, "Result_3_2_Forward_Subset_Measurements_Goodness_of_Fit.png", width = 6, height = 6);
addText (report, "Result 2.1.3: Figure 2.1.2: Forward Stepwise Subset Selection – QR Code Plot:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The figure below presents the forward stepwise best subset chosen via the thresholding values of  r-square, adjusted r-square, Cp and BIC.");
addPng(report, "Result_3_2_Forward_Subset_QR_Code_Goodness_of_Fit.png", width = 6, height = 6);
addNewLine(report, n = 1);
# Test 2.1.4: Print Best Fitting Subset of Variables and their Coefficients.
addText (report, "Result 2.1.4: Forward Stepwise Best Fitting Subset Variables and their Coefficients:", bold = TRUE);
addNewLine(report, n = 1);
addParagraph (report, "The table below presents the forward stepwise best subset for fitting with its coefficients, selected by the BIC.");
addNewLine(report, n = 1);
coefficient_fit_fwd = coef (regfit.fwd, which.min ( summary(regfit.fwd)$bic ) );
table = cbind((1:length(coefficient_fit_fwd)), names(coefficient_fit_fwd), round(coefficient_fit_fwd,10));
colnames(table)[1:3] = c("Number", "Variable", "Coefficient");
addTable(report,table);
addNewLine(report, n=1);
# Test 2.1.5: Print the Misclassification Rates of the Models via k-fold Cross Validation.
addText (report, "Result 2.1.5: Forward Stepwise Subset: Number of Variables VS Misclassification Rates:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The table below presents the misclassification rates of each subset variables, selected by using the fitting error.");
addNewLine(report, n=1);
table = cbind(1:length(cv.fwd), round(cv.fwd,10));
colnames(table)[1:2] = c("Number of Variables", "Misclassification Rates");
addTable(report,table);
addNewLine(report, n=1);
# Plot the best model via k-fold cross-validation.
addText (report, "Result 2.1.5: Figure 2.1.3: Forward Stepwise Subset: Number of Variables VS Misclassification Rates:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The figure below illustrates the misclassification rates of each model containing i variables, for i going from 1 to", np, ".");
addPng(report, "Result_3_5_Forward_Subset_CV_Errors.png", width = 6, height = 6);
# Test 2.1.6: Print the coefficients estimates associated to the best forecasting model.
addText (report, "Result 2.1.6: Forward Stepwise: Best Forecasting Subset Variables and their Coefficients:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The table below presents the forward stepwise best subset and its variables' coefficients, which minimizes the misclassification rates.");
addNewLine(report, n=1);
coefficient = coef (regfit.fwd, which.min ( cv.fwd ) );
table = cbind((1:length(coefficient))-1, names(coefficient), round(coefficient,10));
colnames(table)[1:3] = c("Number", "Variable", "Coefficient");
addTable(report,table);
addNewLine(report, n=1);
# Test 2.2: Variable Selection: Backward Stepwise Subset Selection.
addText (report, "Test 2.2: Variable Selection: Backward Stepwise Subset Selection.", bold = TRUE);
addNewLine(report, n = 1);
# Test 2.2.1: Run backward stepwise subset variables selection, using linear regression model.
addText (report, "Result 2.2.1: Backward Stepwise Subset Variable Selection:", bold = TRUE);
addNewLine(report, n = 1);
addParagraph (report, "Goal: Select the subset of variables which best explain the response variable, using backward stepwise subset selection approach.");
addNewLine(report, n = 1);
addParagraph (report, "The table below presents the backward stepwise best subset of i variables,  for i going from 1 to", np, "through comparing the RSS. * denotes that the corresponding variable is selected into the subset.");
addNewLine(report, n=1);
table = summary ( regfit.bwd)$outmat;
rownames(table) = 1:nrow(table);
table = cbind(rownames(table),table)
colnames(table)[1] = "Number of Variables";
addTable(report,table,header.col.justify = "L", font.size = 5);
addNewLine(report, n=1);
addParagraph (report, "The tables below present the variables and coefficients of the best models with i variables, for i going from 1 to", np, ".");
addNewLine(report, n=1);
for (i in 1:np){
addText (report, "Result 2.2.1: Backward Stepwise Subset Variable Selection: Model with ",i, " Variables:", bold = TRUE);
addNewLine(report, n=1);
table = cbind((1:length(coef(regfit.bwd,i))), names(coef(regfit.bwd,i)), round(coef(regfit.bwd,i),10));
colnames(table)[1:3] = c("Number", "Variable", "Coefficient");
addTable(report,table);
addNewLine(report, n=1);
}
# Test 2.2.2: Plot the RSS, adjusted r-square, Cp and BIC for all the models.
addText (report, "Result 2.2.2: Figure 2.2.1: Backward Stepwise Subset Selection – Measurements of Goodness of Fit:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The figure below presents the RSS, adjusted r-square, Cp and BIC for the backward stepwise best models with i variables, for i going from 1 to", np, ". The red point indicates the number of variables which optimizes the measurement.");
addPng(report, "Result_4_2_Backward_Subset_Measurements_Goodness_of_Fit.png", width = 6, height = 6);
addText (report, "Result 2.2.2: Figure 2.2.2: Backward Stepwise Subset Selection – QR Code Plot:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The figure below presents the backward stepwise best subset chosen via the thresholding values of  r-square, adjusted r-square, Cp and BIC.");
addPng(report, "Result_4_2_Backward_Subset_QR_Code_Goodness_of_Fit.png", width = 6, height = 6);
addNewLine(report, n = 1);
# Test 2.2.3: Print Best Fitting Subset of Variables and their Coefficients.
addText (report, "Result 2.2.3: Backward Stepwise Best Fitting Subset Variables and their Coefficients:", bold = TRUE);
addNewLine(report, n = 1);
addParagraph (report, "The table below presents the backward stepwise best subset for fitting with its coefficients, selected by the BIC.");
addNewLine(report, n = 1);
coefficient_fit_bwd = coef (regfit.bwd, which.min ( summary(regfit.bwd)$bic ) );
table = cbind((1:length(coefficient_fit_bwd)), names(coefficient_fit_bwd), round(as.numeric(coefficient_fit_bwd),10));
colnames(table)[1:3] = c("Number", "Variable", "Coefficient");
addTable(report,table);
addNewLine(report, n=1);
# Test 2.2.4: Print the Test Misclassification Rates of the Models via k-fold Cross Validation.
addText (report, "Result 2.2.4: Backward Stepwise Subset: Number of Variables VS Misclassification Rates:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The table below presents the misclassification rates of each subset variables, selected by using the fitting error.");
addNewLine(report, n=1);
table = cbind(1:length(cv.bwd), round(cv.bwd,10));
colnames(table)[1:2] = c("Number of Variables", "Misclassification Rates");
addTable(report,table);
addNewLine(report, n=1);
# Plot the best model via k-fold cross-validation.
addText (report, "Result 2.2.4: Figure 2.2.3: Backward Stepwise Subset: Number of Variables VS Misclassification Rates:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The figure below illustrates the misclassification rates of each model containing i variables, for i going from 1 to", np, ".");
addPng(report, "Result_4_5_Backward_Subset_CV_Errors.png", width = 6, height = 6);
# Test 2.2.5: Print the coefficients estimates associated to the best forecasting model.
addText (report, "Result 2.2.5: Backward Stepwise: Best Forecasting Subset Variables and their Coefficients:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The table below presents the backward stepwise best subset and its variables' coefficients, which minimizes the misclassification rates.");
addNewLine(report, n=1);
coefficient = coef (regfit.bwd, which.min ( cv.bwd ) );
table = cbind((1:length(coefficient)), names(coefficient), round(coefficient,10));
colnames(table)[1:3] = c("Number", "Variable", "Coefficient");
addTable(report,table);
addNewLine(report, n=1);
# Test 2.3: Variable Selection: best subset with indicator responses.
addText (report, "Test 2.3: Variable Selection: Best Subset Variables Selection with Indicator Responses.", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "Goal: use logistic regression to select the best subset variables with considering the responses are indicators.");
addNewLine(report, n=1);
# Print the fwd best subset variables and coefficients.
addText (report, "Result 2.3.1: Indicator Responses Forward Stepwise Subset Variables and their Coefficients:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The table below presents the variables selected by the forward stepwise logistic regression and their coefficients.");
addNewLine(report, n=1);
coefficient = fwd_model.coef;
table = cbind((1:length(coefficient)), names(coefficient), round(coefficient,10));
colnames(table)[1:3] = c("Number", "Variable", "Coefficient");
addTable(report,table);
addNewLine(report, n=1);
addText (report, "Result 2.3.2: Indicator Responses Backward Stepwise Subset Variables and their Coefficients:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The table below presents the variables selected by the backward stepwise logistic regression and their coefficients.");
addNewLine(report, n=1);
coefficient = bwd_model.coef;
table = cbind((1:length(coefficient)), names(coefficient), round(coefficient,10));
colnames(table)[1:3] = c("Number", "Variable", "Coefficient");
addTable(report,table);
addNewLine(report, n=1);
# Test 2.4: Variable Selection: the lasso.
addText (report, "Test 2.4: Variable Selection: the lasso.", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "Goal: Remove the variables if their coefficients are estimated as 0 by the lasso approach.");
addNewLine(report, n=1);
# Print the coefficients estimates of the lasso model.
addText (report, "Result 2.4.1: the lasso and their Coefficients:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The table below presents the variables selected by the lasso and their coefficients.");
addNewLine(report, n=1);
coefficient = lasso.coef;
table = cbind((1:length(coefficient)), names(coefficient), round(coefficient,10));
colnames(table)[1:3] = c("Number", "Variable", "Coefficient");
addTable(report,table);
addNewLine(report, n=1);
# Test 2.4: Variable Selection: Random Forests.
addText (report, "Test 2.5: Variable Selection: Random Forests.", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "Goal: Use Random Forests to determine the thresholding variables.");
addNewLine(report, n=1);
# Test 2.5.1: Summarize the number of variables selected by the random forest.
addText (report, "Result 2.5.1: Summary: Variable Selection using Random Forests:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The table below summarizes the number of variables selected by the random forests.");
addNewLine(report, n=1);
value = data.vsurf$nums.varselect;
name = c("Thresholding Variables", "Interpretation Variables", "Prediction Variables");
table = cbind(name, value);
colnames(table)[1:2] = c("Subset Variables", "Number of Variables");
addTable(report,table);
addNewLine(report, n=1);
# Test 2.5.2: Plot the subset selection results by using the random forest.
addText (report, "Result 2.5.2: Figure 2.4.1: Variable Selection using Random Forests:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The figure below illustrates the variable selection results by the random forests. The 3 set of variables are respectively thresholding, interpretation and prediction variables. They are determined via different measurements.");
addPng(report, "Result_6_2_ Variable_Selection_using_Random_Forests.png", width = 6, height = 6);
# Test 2.5.3: List the thresholding subset variables for fitting performance.
addText (report, "Result 2.5.3: Thresholding Variables Selected by Random Forests:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The table below lists the thresholding variables selected by the random forests.");
addNewLine(report, n=1);
coefficient = colnames(x)[data.vsurf$varselect.thres];
table = as.data.frame(cbind(1:length(coefficient),coefficient));
colnames(table)[1:2] =c("Number", "Thresholding Variables");
addTable(report, table);
addNewLine(report, n=1);
# Test 2.5.4: List the Interpretation subset variables for fitting performance.
addText (report, "Result 2.5.4: Interpretation Variables Selected by Random Forests:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The table below lists the interpretation variables selected by the random forests.");
addNewLine(report, n=1);
coefficient = colnames(x)[data.vsurf$varselect.interp];
table = as.data.frame(cbind(1:length(coefficient),coefficient));
colnames(table)[1:2] =c("Number", "Interpretation Variables");
addTable(report,table);
addNewLine(report, n=1);
# Test 2.5.5: List the prediction subset variables for fitting performance.
addText (report, "Result 2.5.5: Prediction Variables Selected by Random Forests:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The table below lists the prediction variables selected by the random forests.");
addNewLine(report, n=1);
coefficient = colnames(x)[data.vsurf$varselect.pred];
table = as.data.frame(cbind(1:length(coefficient),coefficient));
colnames(table)[1:2] =c("Number", "Prediction Variables");
addTable(report,table);
addNewLine(report, n=1);
# Test 2.5.6: Illustrate the importance of variables.
#addText (report, "Result 2.5.6: Figure 2.4.2: Importance of Variables using Random Forests:", bold = TRUE);
#addNewLine(report, n=1);
#addParagraph (report, "The figure below ranks the variables through their level of importance.");
#addPng(report, "Result_7_3_ Importance_of_Variables.png", width = 6, height = 6);
# Test 3: Model Selection.
addText (report, "Test 3: Model Selection: Logistic Regression, Random Forests, Ridge, the Lasso.", bold = TRUE);
addNewLine(report, n=1);
# Test 3.1: Ridge regression and its coefficients.
addText (report, "Result 3.1: Ridge Regression and its Coefficients:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The table below presents the ridge regression coefficients.");
addNewLine(report, n=1);
coefficient = ridge.coef;
table = cbind((1:length(coefficient)), names(coefficient), round(coefficient,10));
colnames(table)[1:3] = c("Number", "Variable", "Coefficient");
addTable(report,table);
addNewLine(report, n=1);
# Test 3.2: the lasso and its coefficients.
addText (report, "Result 3.2: the lasso and its Coefficients:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The table below presents the lasso coefficients.");
addNewLine(report, n=1);
coefficient = lasso.coef;
table = cbind((1:length(coefficient)), names(coefficient), round(coefficient,10));
colnames(table)[1:3] = c("Number", "Variable", "Coefficient");
addTable(report,table);
addNewLine(report, n=1);
# Test 3.3: Compare the Misclassification Rates of the 8 models: logistic regression full subset, fwd,
# bwd , tree, ridge and lasso, current.
addText (report, "Result 3.3: Comparison of the Models: Forward Stepwise Subset, Backward Stepwise Subset, Random Forests, Lasso, Current:", bold = TRUE);
addNewLine(report, n = 1);
# Print the coefficients estimates of the lasso model.
addParagraph (report, "The table below presents our current variables selected and their coefficients.");
addNewLine(report, n=1);
coefficient = coef(current.fit);
table = cbind((1:length(coefficient)), names(coefficient), round(coefficient,10));
colnames(table)[1:3] = c("Number", "Variable", "Coefficient");
addTable(report,table);
addNewLine(report, n=1);
addParagraph (report, "The table below presents the misclassification rates of the 8 models, estimated by the cross validation method.");
addNewLine(report, n=1);
table = as.data.frame(misclassification_rates);
Model = c("reg fwd","reg bwd", "reg fwd ind","reg bwd ind","random forest","ridge","lasso","current");
addTable(report,cbind(Model,table));
addNewLine(report, n=1);
addText (report, "Result 3.3: Figure 3.1: Comparison of Predictive Models - Misclassification Rates:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The figure below illustrates the misclassification rates of the 8 models, estimated by the cross validation method.");
addPng(report, "Result_7_5_ Comparison_of_Predictive_Models.png", width = 6, height = 6);
addPageBreak(report);
# Conclusion 4: Best Subset Variables.
addText (report, "Conclusion 4: Best Subset Variables.", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "In this section we summarize the best forecasting subset variables selected by each model and provide the best subset variables included by all models.");
addNewLine(report, n=1);
# Result 4.1: Best Subset Variables for Each Model.
addText (report, "Result 4.1: Best Subset Variables for Each Model:", bold = TRUE);
addNewLine(report, n=1);
# Best subset variables selected by the forward stepwise subset variables selection method.
addParagraph (report, "Variables selected by the forward stepwise subset selection method.");
addNewLine(report, n = 1);
ranking = cbind(as.character(variables_full[as.numeric(variables_fwd)==1]), round(as.numeric(coefficient_fwd[variables_full[as.numeric(variables_fwd)==1]]),10), round(as.numeric(grade_fwd[as.numeric(variables_fwd)==1]),10));
ranking_sort = ranking[order(as.numeric(ranking[,3]), decreasing = TRUE),];
table = cbind((1:length(coefficient_fwd)), ranking_sort);
colnames(table)[1:4] = c("Number", "Variable", "Coefficient", "Grades");
addTable(report,table);
addNewLine(report, n=1);
# Best subset variables selected by the backward stepwise subset variables selection method.
addParagraph (report, "Variables selected by the backward stepwise subset selection method.");
addNewLine(report, n = 1);
ranking = cbind(as.character(variables_full[as.numeric(variables_bwd)==1]), round(as.numeric(coefficient_bwd[variables_full[as.numeric(variables_bwd)==1]]),10), round(grade_bwd[as.numeric(variables_bwd)==1],10));
ranking_sort = ranking[order(as.numeric(ranking[,3]), decreasing = TRUE),];
table = cbind((1:length(coefficient_bwd)), ranking_sort);
colnames(table)[1:4] = c("Number", "Variable", "Coefficient", "Grades");
addTable(report,table);
addNewLine(report, n=1);
# Best subset variables selected by the forward stepwise subset variables selection method with
# indicator.
addParagraph (report, "Variables selected by the forward stepwise subset selection method with indicator responses.");
addNewLine(report, n = 1);
ranking = cbind(as.character(variables_full[as.numeric(variables_fwd_ind)==1]), round(as.numeric(fwd_model.coef[variables_full[as.numeric(variables_fwd_ind)==1]]),10), round(grade_fwd_ind[as.numeric(variables_fwd_ind)==1],10));
ranking_sort = ranking[order(as.numeric(ranking[,3]), decreasing = TRUE),];
table = cbind((1:length(fwd_model.coef)), ranking_sort);
colnames(table)[1:4] = c("Number", "Variable", "Coefficient", "Grades");
addTable(report,table);
addNewLine(report, n=1);
# Best subset variables selected by the backward stepwise subset variables selection method with
# indicator.
addParagraph (report, "Variables selected by the backward stepwise subset selection method with indicator responses.");
addNewLine(report, n = 1);
ranking = cbind(as.character(variables_full[as.numeric(variables_bwd_ind)==1]), round(as.numeric(bwd_model.coef [variables_full[as.numeric(variables_bwd_ind)==1]]),10), round(grade_bwd_ind[as.numeric(variables_bwd_ind)==1],10));
ranking_sort = ranking[order(as.numeric(ranking[,3]), decreasing = TRUE),];
table = cbind((1:length(bwd_model.coef)), ranking_sort);
colnames(table)[1:4] = c("Number", "Variable", "Coefficient", "Grades");
addTable(report,table);
addNewLine(report, n=1);
# Best subset variables selected by the lasso.
addParagraph (report, "Variables selected by the lasso.");
addNewLine(report, n = 1);
ranking = cbind(as.character(variables_full[as.numeric(variables_lasso)==1]), round(as.numeric(lasso_model.coef[variables_full[as.numeric(variables_lasso)==1]]),10), round(grade_lasso[as.numeric(variables_lasso)==1],10));
ranking_sort = ranking[order(as.numeric(ranking[,3]), decreasing = TRUE),];
table = cbind((1:length(coefficient_lasso)), ranking_sort);
colnames(table)[1:4] = c("Number", "Variable", "Coefficient", "Grades");
addTable(report,table);
addNewLine(report, n=1);
# Best subset variables selected by the random forests.
addParagraph (report, "Variables selected by the random forests.");
addNewLine(report, n=1);
ranking = cbind(variables_full, round(grade_rf,10), impact_rf);
ranking = ranking[ranking[,2]>0,];
ranking_sort = ranking[order(as.numeric(ranking[,2]), decreasing = TRUE),];
table = cbind((1:nrow(ranking_sort)), ranking_sort);
colnames(table)[1:4] = c("Number", "Variable", "Grades", "Impact");
addTable(report,table);
addNewLine(report, n=1);
# Result 4.2: Best Subset Variables Selected by All Models.
addText (report, "Result 4.2: Best Subset Variables Selected by All Models:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The table below presents the best forecasting subset variables selected by all models. It is obtained by collecting all the common variables selected by all the models.");
addNewLine(report, n=1);
variables_best =Reduce(intersect, list(names(fwd_model.coef), names(bwd_model.coef), names(coefficient_lasso), c("(Intercept)",importance_rf)));
table = as.data.frame(cbind(1:length(variables_best), variables_best));
colnames(table)[1:2] =c("Number", "Variable");
addTable(report,table);
addNewLine(report, n=1);
# Result 4.3: Comparison of All Models.
addText (report, "Result 4.3: Comparison of All Models:", bold = TRUE);
addNewLine(report, n=1);
# List the variables selected by the 4 models.
addParagraph (report, "The table below presents the best subset variables selected by each of the 4 models, through comparing the misclassification rates.");
addNewLine(report, n=1);
table = cbind(1:l, variables_full, variables_fwd_ind,variables_bwd_ind, variables_lasso, variables_rf);
colnames(table) = c("Number","Variable", "reg fwd ind", "reg bwd ind", "lasso", "rf");
addTable(report,table);
addNewLine(report, n=1);
# Rank the variables by the 8 models.
addParagraph (report, "The table below presents the ranking of each variable made by the 4 models, high record means high importance.");
addNewLine(report, n=1);
ranking = variables_fwd_ind + variables_bwd_ind + variables_lasso + variables_rf;
variables_ranking_6 = (ranking==6)^2;
variables_ranking_5 = (ranking==5)^2;
variables_ranking_4 = (ranking==4)^2;
variables_ranking_3 = (ranking==3)^2;
variables_ranking_2 = (ranking==2)^2;
variables_ranking_1 = (ranking==1)^2;
variables_ranking_0 = (ranking==0)^2;
table = cbind((1:l), variables_full, variables_ranking_4, variables_ranking_3, variables_ranking_2, variables_ranking_1, variables_ranking_0);
colnames(table) = c("Number","Variable","4", "3", "2", "1", "0");
addTable(report,table);
addNewLine(report, n=1);
addParagraph (report, "The table below presents the cumulative ranking of each variable made by the 4 models, high record means high importance.");
addNewLine(report, n=1);
variables_ranking_6 = (ranking>=6)^2;
variables_ranking_5 = (ranking>=5)^2;
variables_ranking_4 = (ranking>=4)^2;
variables_ranking_3 = (ranking>=3)^2;
variables_ranking_2 = (ranking>=2)^2;
variables_ranking_1 = (ranking>=1)^2;
table = cbind((1:l), variables_full, variables_ranking_4, variables_ranking_3, variables_ranking_2, variables_ranking_1);
colnames(table) = c("Number","Variable",">=4", ">=3", ">=2", ">=1");
addTable(report,table);
addNewLine(report, n=1);
addParagraph (report, "The table below presents the ranking of each variable made by the 4 models, high record means high impact on explaining the values of the response variable. In each ranking, the variables are ordered with deceasing grades. A feature’s grade denotes its contribution to the logit lapse rates values.");
addNewLine(report, n=1);
grades = round(grade_fwd_ind+grade_bwd_ind+grade_lasso+grade_rf, 8)/4;
grades_impact = impact(grade_fwd_ind_raw+grade_bwd_ind_raw+grade_lasso_raw+grade_rf_raw);
ranking = cbind(variables_full,ranking,grades,grades_impact);
ranking_sort = ranking[order(ranking[,2], decreasing = TRUE),];
for (i in 1:5) { if (sum(as.numeric(ranking_sort[,2])==(5-i))>1){r=ranking_sort[ranking_sort[,2]==(5-i),]; r=as.matrix(r); ranking_sort[ranking_sort[,2]==(5-i),]  = r[order(as.numeric(r[,3]), decreasing = TRUE),]}};
table = cbind((1:l), ranking_sort);
colnames(table) = c("Number","Variable","Ranking", "Grades", "Impact");
addTable(report,table);
addNewLine(report, n=1);
addParagraph (report, "The table below presents the features with decreasing order of grades for the 3 logistic regression based approaches. A feature’s grade denotes its contribution to the logit lapse rates values.");
addNewLine(report, n=1);
ranking_sort = ranking[order(as.numeric(ranking[,3]), decreasing = TRUE),];
table = cbind((1:l), ranking_sort);
colnames(table) = c("Number","Variable","Ranking", "Grades", "Impact");
addTable(report,table);
addNewLine(report, n=1);
# Compare the misclassification rates of the 8 models.
addParagraph (report, "The table below presents the misclassification rates of the 8 models, estimated by using the cross validation method.");
addNewLine(report, n=1);
table = as.data.frame(misclassification_rates);
Model = c("reg fwd rate","reg bwd rate", "reg fwd ind","reg bwd ind","random forest","ridge","lasso","current");
addTable(report,cbind(Model,table));
addNewLine(report, n=1);
addParagraph (report, "The figure below compares the misclassification rates of all the 8 models, each with the best subset variables.");
addPng(report, "Result_7_5_ Comparison_of_Predictive_Models.png", width = 6, height = 6);
done(report);
# Create high level report.
report = RTF(report_short_name);
# Add Title, date and table of content.
addHeader(report, "        Diagnostics Result Report (Short Version)", subtitle = "                   ------ Experience Study, AIG ", font.size = 20);
addParagraph (report, "Data analytics process started at ",now," (",Sys.timezone(),")");
addParagraph (report, "Data analytics process ended at ",end," (",Sys.timezone(),")");
start = as.POSIXct(now);
ends = as.POSIXct(end);
addParagraph (report, "The time duration is ",difftime(ends, start, units="hours"), " hours");
addParagraph (report, "The diagnostics are performed on ",percentage*100, "% of records in the dataset, i.e., ", nrow(pre_raw_data), " out of a total number of ",nrow(raw_data), " records");
addParagraph (report, "Report generated at ",Sys.time()," (",Sys.timezone(),")");
addNewLine(report, n = 1);
addParagraph (report, "Response factors include: ", levels(pre_raw_data$INSURC_CHANGE)[1], ", ",levels(pre_raw_data$INSURC_CHANGE)[2]);
addNewLine(report, n = 1);
addText (report, "Table of Content", bold = TRUE);
addNewLine(report, n = 1);
addParagraph (report, "    Section 1: Dataset Information and Candidate Predictive Models");
addParagraph (report, "        Section 1.1: Dataset Information");
addParagraph (report, "        Section 1.2: Candidate Predictive Models");
addParagraph (report, "    Section 2: Best Subset Variables");
addParagraph (report, "        Section 2.1: Best Subset Variables for Each Model");
addParagraph (report, "        Section 2.2: Best Subset Variables Selected by All Models");
addParagraph (report, "    Section 3: Model Selection");
addPageBreak(report);
# Section 1: Dataset Information and Candidate Predictive Models.
addText (report, "Section 1: Dataset Information and Candidate Predictive Models.", bold = TRUE);
addNewLine(report, n = 1);
addText (report, "Section 1.1: Dataset Information.", bold = TRUE);
addNewLine(report, n = 1);
addParagraph (report, "The dataset consists of ", nrow(pre_raw_data), " monthly valuation records and each record contains information on 1 response variable and ", ncol(pre_raw_data), " predictor variables.");
addNewLine(report, n = 1);
addParagraph (report, "The list of response variables is given below:");
addNewLine(report, n = 1);
list_data = sapply(pre_raw_data["INSURC_CHANGE"], class);
list_level = sapply(pre_raw_data["INSURC_CHANGE"], nlevels);
table = as.data.frame(cbind(1:length(list_data), names(list_data), list_data, list_level));
colnames(table)[1:4] = c("Number", "Response Variable", "Type", "Number of Categories");
addTable(report,table);
addNewLine(report, n=1);
addParagraph (report, "The list of predictor variables is given below:");
addNewLine(report, n = 1);
list_data = sapply(pre_raw_data[, !(names(pre_raw_data)%in% "INSURC_CHANGE")], class);
list_level = sapply(pre_raw_data[, !(names(pre_raw_data)%in% "INSURC_CHANGE")], nlevels);
list_level[list_level==0]= "-";
table = as.data.frame(cbind(1:length(list_data), names(list_data), list_data, list_level));
colnames(table)[1:4] = c("Number", "Predictor Variable", "Type", "Number of Categories");
addTable(report,table);
addNewLine(report, n=1);
addText (report, "Section 1.2: Candidate Predictive Models.", bold = TRUE);
addNewLine(report, n = 1);
addParagraph (report, "We consider the following 4 approaches for subset variables selection:");
addNewLine(report, n = 1);
method = c("fwd logistic regression", "bwd logistic regression", "lasso", "random forests");
measure = c("AIC", "AIC", "Misclassification Error", "Out of Bagging Error");
table = as.data.frame(cbind(1:length(method),method, measure));
colnames(table)[1:3] = c("Number", "Subset Selection Method", "Criterion");
addTable(report,table);
addNewLine(report, n=1);
addParagraph (report, "We consider the following 4 models for fitting the response variable:");
addNewLine(report, n = 1);
model = c("logistic regression", "ridge", "lasso", "random forests");
feature = c("low bias, easy to implement", "high bias, low variance", "high bias, low variance, remove variables", "low bias, hard to implement");
table = as.data.frame(cbind(1:length(feature),model, feature));
colnames(table)[1:3] = c("Number", "Predictive Model", "Feature");
addTable(report,table);
addNewLine(report, n=1);
# Conclusion: Best Subset Variables.
addText (report, "Section 2: Best Subset Variables.", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "In this section we summarize the best forecasting subset variables selected by each model and provide the best subset variables included by all models.");
addNewLine(report, n=1);
# Result 1.1: Best Subset Variables for Each Model.
addText (report, "Section 2.1: Best Subset Variables for Each Model:", bold = TRUE);
addNewLine(report, n=1);
# Best subset variables selected by the forward stepwise subset variables selection method.
addParagraph (report, "Variables selected by the forward stepwise subset selection method.");
addNewLine(report, n = 1);
ranking = cbind(as.character(variables_full[as.numeric(variables_fwd)==1]), round(as.numeric(coefficient_fwd[variables_full[as.numeric(variables_fwd)==1]]),10), round(as.numeric(grade_fwd[as.numeric(variables_fwd)==1]),10));
ranking_sort = ranking[order(as.numeric(ranking[,3]), decreasing = TRUE),];
table = cbind((1:length(coefficient_fwd)), ranking_sort);
colnames(table)[1:4] = c("Number", "Variable", "Coefficient", "Grades");
addTable(report,table);
addNewLine(report, n=1);
# Best subset variables selected by the backward stepwise subset variables selection method.
addParagraph (report, "Variables selected by the backward stepwise subset selection method.");
addNewLine(report, n = 1);
ranking = cbind(as.character(variables_full[as.numeric(variables_bwd)==1]), round(as.numeric(coefficient_bwd[variables_full[as.numeric(variables_bwd)==1]]),10), round(grade_bwd[as.numeric(variables_bwd)==1],10));
ranking_sort = ranking[order(as.numeric(ranking[,3]), decreasing = TRUE),];
table = cbind((1:length(coefficient_bwd)), ranking_sort);
colnames(table)[1:4] = c("Number", "Variable", "Coefficient", "Grades");
addTable(report,table);
addNewLine(report, n=1);
# Best subset variables selected by the forward stepwise subset variables selection
# method with
# indicator.
addParagraph (report, "Variables selected by the forward stepwise subset selection method with indicator responses.");
addNewLine(report, n = 1);
ranking = cbind(as.character(variables_full[as.numeric(variables_fwd_ind)==1]), round(as.numeric(fwd_model.coef[variables_full[as.numeric(variables_fwd_ind)==1]]),10), round(grade_fwd_ind[as.numeric(variables_fwd_ind)==1],10));
ranking_sort = ranking[order(as.numeric(ranking[,3]), decreasing = TRUE),];
table = cbind((1:length(fwd_model.coef)), ranking_sort);
colnames(table)[1:4] = c("Number", "Variable", "Coefficient", "Grades");
addTable(report,table);
addNewLine(report, n=1);
# Best subset variables selected by the backward stepwise subset variables selection method with
# indicator.
addParagraph (report, "Variables selected by the backward stepwise subset selection method with indicator responses.");
addNewLine(report, n = 1);
ranking = cbind(as.character(variables_full[as.numeric(variables_bwd_ind)==1]), round(as.numeric(bwd_model.coef [variables_full[as.numeric(variables_bwd_ind)==1]]),10), round(grade_bwd_ind[as.numeric(variables_bwd_ind)==1],10));
ranking_sort = ranking[order(as.numeric(ranking[,3]), decreasing = TRUE),];
table = cbind((1:length(bwd_model.coef)), ranking_sort);
colnames(table)[1:4] = c("Number", "Variable", "Coefficient", "Grades");
addTable(report,table);
addNewLine(report, n=1);
# Best subset variables selected by the lasso.
addParagraph (report, "Variables selected by the lasso.");
addNewLine(report, n = 1);
ranking = cbind(as.character(variables_full[as.numeric(variables_lasso)==1]), round(as.numeric(lasso_model.coef[variables_full[as.numeric(variables_lasso)==1]]),10), round(grade_lasso[as.numeric(variables_lasso)==1],10));
ranking_sort = ranking[order(as.numeric(ranking[,3]), decreasing = TRUE),];
table = cbind((1:length(coefficient_lasso)), ranking_sort);
colnames(table)[1:4] = c("Number", "Variable", "Coefficient", "Grades");
addTable(report,table);
addNewLine(report, n=1);
# Best subset variables selected by the random forests.
addParagraph (report, "Variables selected by the random forests.");
addNewLine(report, n=1);
ranking = cbind(variables_full, round(grade_rf,10), impact_rf);
ranking = ranking[ranking[,2]>0,];
ranking_sort = ranking[order(as.numeric(ranking[,2]), decreasing = TRUE),];
table = cbind(1:nrow(ranking_sort), ranking_sort);
colnames(table)[1:4] = c("Number", "Variable", "Grades", "Impact");
addTable(report,table);
addNewLine(report, n=1);
# Result 2.2: Best Subset Variables Selected by All Models.
addText (report, "Section 2.2: Best Subset Variables Selected by All Models:", bold = TRUE);
addNewLine(report, n=1);
addParagraph (report, "The table below presents the best forecasting subset variables selected by all models. It is obtained by collecting all the common variables selected by all the models.");
addNewLine(report, n=1);
variables_best =Reduce(intersect, list(names(fwd_model.coef), names(bwd_model.coef), names(coefficient_lasso), c("(Intercept)",importance_rf)));
table = as.data.frame(cbind(1:length(variables_best), variables_best));
colnames(table)[1:2] =c("Number", "Variable");
addTable(report,table);
addNewLine(report, n=1);
# Result 2.3: Comparison of All Models.
addText (report, "Section 2.3: Comparison of All Models:", bold = TRUE);
addNewLine(report, n=1);
# List the variables selected by the 4 models.
addParagraph (report, "The table below presents the best subset variables selected by each of the 4 models, through comparing the misclassification rates.");
addNewLine(report, n=1);
table = cbind(1:l, variables_full, variables_fwd_ind,variables_bwd_ind, variables_lasso, variables_rf);
colnames(table) = c("Number","Variable", "reg fwd ind", "reg bwd ind", "lasso", "rf");
addTable(report,table);
addNewLine(report, n=1);
# Rank the variables by the 8 models.
addParagraph (report, "The table below presents the ranking of each variable evaluated by the 4 models, high record means high importance.");
addNewLine(report, n=1);
ranking = variables_fwd_ind + variables_bwd_ind + variables_lasso + variables_rf;
variables_ranking_6 = (ranking==6)^2;
variables_ranking_5 = (ranking==5)^2;
variables_ranking_4 = (ranking==4)^2;
variables_ranking_3 = (ranking==3)^2;
variables_ranking_2 = (ranking==2)^2;
variables_ranking_1 = (ranking==1)^2;
variables_ranking_0 = (ranking==0)^2;
table = cbind(1:l, variables_full, variables_ranking_4, variables_ranking_3, variables_ranking_2, variables_ranking_1, variables_ranking_0);
colnames(table) = c("Number","Variable", "4", "3", "2", "1", "0");
addTable(report,table);
addNewLine(report, n=1);
addParagraph (report, "The table below presents the cumulative ranking of each variable made by the 4 models, high record means high importance.");
addNewLine(report, n=1);
variables_ranking_6 = (ranking>=6)^2;
variables_ranking_5 = (ranking>=5)^2;
variables_ranking_4 = (ranking>=4)^2;
variables_ranking_3 = (ranking>=3)^2;
variables_ranking_2 = (ranking>=2)^2;
variables_ranking_1 = (ranking>=1)^2;
table = cbind(1:l, variables_full, variables_ranking_4, variables_ranking_3, variables_ranking_2, variables_ranking_1);
colnames(table) = c("Number","Variable", ">=4", ">=3", ">=2", ">=1");
addTable(report,table);
addNewLine(report, n=1);
addParagraph (report, "The table below presents the ranking of each variable made by the 4 models, high record means high impact on explaining the values of the response variable. In each ranking, the variables are ordered with deceasing grades. A feature’s grade denotes its contribution to the logit lapse rates values.");
addNewLine(report, n=1);
grades = round(grade_fwd_ind+grade_bwd_ind+grade_lasso+grade_rf, 8)/4;
grades_impact = impact(grade_fwd_ind_raw+grade_bwd_ind_raw+grade_lasso_raw+grade_rf);
ranking = cbind(variables_full,ranking,grades,grades_impact);
ranking_sort = ranking[order(ranking[,2], decreasing = TRUE),];
for (i in 1:5) { if (sum(as.numeric(ranking_sort[,2])==(5-i))>1){r=ranking_sort[ranking_sort[,2]==(5-i),]; r=as.matrix(r); ranking_sort[ranking_sort[,2]==(5-i),]  = r[order(as.numeric(r[,3]), decreasing = TRUE),]}};
table = cbind(1:l, ranking_sort);
colnames(table) = c("Number","Variable","Ranking", "Grades", "Impact");
addTable(report,table);
addNewLine(report, n=1);
addParagraph (report, "The table below presents the features with decreasing order of grades for the 3 logistic regression based approaches. A feature’s grade denotes its contribution to the logit lapse rates values.");
addNewLine(report, n=1);
ranking_sort = ranking[order(as.numeric(ranking[,3]), decreasing = TRUE),];
table = cbind(1:l, ranking_sort);
colnames(table) = c("Number","Variable","Ranking", "Grades", "Impact");
addTable(report,table);
addNewLine(report, n=1);
# Compare the misclassification rates of the 8 models.
addParagraph (report, "Section 3: Model Selection.");
addNewLine(report, n=1);
# Print the coefficients estimates of the current model.
addParagraph (report, "The table below presents our current variables selected and their coefficients.");
addNewLine(report, n=1);
coefficient = coef(current.fit);
table = cbind(1:length(coefficient), names(coefficient), round(coefficient,10));
colnames(table)[1:3] = c("Number", "Variable", "Coefficient");
addTable(report,table);
addNewLine(report, n=1);
addParagraph (report, "The table below presents the misclassification rates of the 8 models, estimated by using the cross validation method.");
addNewLine(report, n=1);
table = as.data.frame(misclassification_rates);
Model = c("reg fwd rate","reg bwd rate", "reg fwd ind","reg bwd ind","random forest","ridge","lasso", "current");
addTable(report,cbind(Model,table));
addNewLine(report, n=1);
addParagraph (report, "The figure below compares the misclassification rates of all the 8 models, each with the best subset variables.");
addPng(report, "Result_7_5_ Comparison_of_Predictive_Models.png", width = 6, height = 6);
done(report)
#};
# Save workspace.
# save.image();

