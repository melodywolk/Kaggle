# Assign the training set
train <- read.csv(url("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"))
  
# Assign the testing set
test <- read.cvs(url("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"))

# absolute numbers
table(train$Survived)
# percentages
prop.table(table(train$Survived))

# Passengers that survived vs passengers that passed aways
table(train$Survived)
prop.table(table(train$Survived))
  
# Males & females that survived vs males & females that passed away
table(train$Sex, train$Survived)
# row-wise proportions
prop.table(table(train$Sex, train$Survived),1)
# column-wise proportions
prop.table(table(train$Sex, train$Survived),2)


# Create the column child, and indicate whether child or no child
train$Child <- 0
train$Child[train$Age < 18] <- 1
# Two-way comparison
table(train$Child, train$Survived)
# row-wise proportions
prop.table(table(train$Child, train$Survived),1)
# column-wise proportions
prop.table(table(train$Child, train$Survived),2)

# prediction based on gender 
test_one <- test
test_one$Survived <- 0
test_one$Survived[test$Sex == 'female'] <- 1

install.packages("rpart")
library("rpart")


# Build the decision tree
my_tree_two <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train, method="class")

# Visualize the decision tree using plot() and text()
plot(my_tree_two)
text(my_tree_two)

# Load in the packages to create a fancified version of your tree
library(rattle)
library(rpart.plot)
library(RColorBrewer)

# Time to plot your fancified tree
fancyRpartPlot(my_tree_two)

# Make your prediction using the test set
my_prediction <- predict(my_tree_two, test, type = "class")

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
my_solution <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)

# Check that your data frame has 418 entries
nrow(my_solution)

# Write your solution away to a csv file with the name my_solution.csv
write.csv(my_solution, file = "my_solution.csv", row.names = FALSE)

#A valid assumption is that larger families need more time to get together on a sinking ship, and hence have less chance of surviving
# create a new train set with the new variable
train_two <- train
train_two$family_size <- train_two$SibSp + train_two$Parch + 1

# Create a new decision tree my_tree_three
my_tree_four <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + family_size, data=train_two, method="class")
  
# Visualize your new decision tree
fancyRpartPlot(my_tree_four)

#You have access to a new train and test set named train_new and test_new. These data sets contain a new column with the name Title #(referring to Miss, Mr, etc.)
# Create a new model `my_tree_five`
my_tree_five <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title, data=train_new, method="class")

# Visualize your new decision tree
fancyRpartPlot(my_tree_five)

# Make your prediction using `my_tree_five` and `test_new`
my_prediction <- predict(my_tree_five, test_new, type = "class")

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
my_solution <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)

# Write your solution away to a csv file with the name my_solution.csv
write.csv(my_solution, file = "my_solution.csv", row.names = FALSE)

# All data, both training and test set
all_data

# Passenger on row 62 and 830 do not have a value for embarkment. 
# Since many passengers embarked at Southampton, we give them the value S.
# We code all embarkment codes as factors.
all_data$Embarked[c(62,830)] = "S"
all_data$Embarked <- factor(combi$Embarked)

# Passenger on row 1044 has an NA Fare value. Let's replace it with the median fare value.
all_data$Fare[1044] <- median(combi$Fare, na.rm=TRUE)

# How to fill in missing Age values?
# We make a prediction of a passengers Age using the other variables and a decision tree model. 
# This time you give method="anova" since you are predicting a continuous variable.
predicted_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + family_size,
                       data=all_data[!is.na(all_data$Age),], method="anova")
all_data$Age[is.na(all_data$Age)] <- predict(predicted_age, all_data[is.na(all_data$Age),])

# Split the data back into a train set and a test set
train <- all_data[1:891,]
test <- all_data[892:1309,]

#One more important element in Random Forest is randomization to avoid the creation of the same tree from the training set. You #randomize in two ways (i) by taking a randomized sample of the rows in your training set, and (ii) by only working with a limited and changing number of the available variables for every node of the tree


# Set seed for reproducibility
set.seed(111)

# Apply the Random Forest Algorithm
my_forest <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title, data=train, importance = TRUE, ntree=1000)

# Make your prediction using the test set
my_prediction <- predict(my_forest, test)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
my_solution <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)

# Write your solution away to a csv file with the name my_solution.csv
write.csv(my_solution, file = "my_solution.csv", row.names = FALSE)
