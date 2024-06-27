from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
import pandas as pd



def create_spark_session(app_name="Mobile Price Classification"):
    """
    Create and return a Spark session.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    return spark

def load_data(spark, file_path):
    """
    Load and return the dataset from the given file path.
    """
    return spark.read.csv(file_path, header=True, inferSchema=True)

def preprocess_data(data, feature_columns):
    """
    Preprocess the data by assembling feature columns into a feature vector.
    """
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    return assembler.transform(data)

def train_random_forest(train_data, label_col="price_range", features_col="features"):
    """
    Train a RandomForestClassifier using CrossValidator and return the model.
    """
    rf = RandomForestClassifier(labelCol=label_col, featuresCol=features_col)
    
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 20, 30]) \
        .addGrid(rf.maxDepth, [5, 10, 15]) \
        .build()
    
    evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy")
    
    crossval = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
    
    cv_model = crossval.fit(train_data)
    
    return cv_model

def train_logistic_regression(train_data, label_col="price_range", features_col="features"):
    """
    Train a LogisticRegression model using CrossValidator and return the model.
    """
    lr = LogisticRegression(labelCol=label_col, featuresCol=features_col)
    
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
        .build()
    
    evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy")
    
    crossval = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
    
    cv_model = crossval.fit(train_data)
    
    return cv_model

def evaluate_model(cv_model, test_data, label_col="price_range"):
    """
    Evaluate the model's accuracy on the test data and return the accuracy and predictions.
    """
    predictions = cv_model.transform(test_data)
    
    evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    
    return accuracy, predictions

def plot_confusion_matrix(predictions, label_col="price_range", prediction_col="prediction", title="Confusion Matrix"):
    """
    Generate and plot a confusion matrix heatmap.
    """
    preds_and_labels = predictions.select(prediction_col, label_col).collect()
    
    predictions_array = [row[prediction_col] for row in preds_and_labels]
    labels_array = [row[label_col] for row in preds_and_labels]
    
    conf_matrix = confusion_matrix(labels_array, predictions_array)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

def check_likely_distribution():
    df_train = pd.read_csv('train.csv')
    data_norm=df_train[['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 
                'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']]
   
    for c in data_norm.columns[:]:
        plt.figure(figsize=(16,16))
        fig=qqplot(data_norm[c],line='45',fit='True')
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlabel("Theoretical quantiles",fontsize=15)
        plt.ylabel("Sample quantiles",fontsize=15)
        plt.title("Q-Q plot of {}".format(c),fontsize=16)
        plt.grid(True)
        plt.show()

def check_variable_correlation():
    corr_matrix = df_train.corr()
    plt.figure(figsize =(20, 15))
    # Plot the correlation matrix using Seaborn's heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='magma')
    plt.show()

def main(file_path):
    """
    Main function to run the entire pipeline.
    """
    # Create a Spark session
    spark = create_spark_session()
    
    # Load the data
    data = load_data(spark, file_path)
    print(data)
    exit()
    # List of feature columns
    feature_columns = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
                       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'px_height', 
                       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 
                       'touch_screen', 'wifi']
    
    # Preprocess the data
    data = preprocess_data(data, feature_columns)

    # Split the data into training and test sets
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)
    
    # Train the RandomForest model
    rf_model = train_random_forest(train_data)
    
    # Evaluate the RandomForest model
    rf_accuracy, rf_predictions = evaluate_model(rf_model, test_data)
    print(f"RandomForest Test Accuracy = {rf_accuracy}")
    
    # Plot the confusion matrix for RandomForest
    plot_confusion_matrix(rf_predictions, title="RandomForest Confusion Matrix")
    
    # Train the Logistic Regression model
    lr_model = train_logistic_regression(train_data)
    
    # Evaluate the Logistic Regression model
    lr_accuracy, lr_predictions = evaluate_model(lr_model, test_data)
    print(f"Logistic Regression Test Accuracy = {lr_accuracy}")
    
    # Plot the confusion matrix for Logistic Regression
    plot_confusion_matrix(lr_predictions, title="Logistic Regression Confusion Matrix")

# Run the main function with the dataset path
if __name__ == "__main__":
    file_path = "train.csv"  # Replace with your dataset path
    main(file_path)
