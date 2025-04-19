import os
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassificationModel
import findspark

# Set Java and Spark environment variables
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"  # Adjust with your correct Java path
os.environ["SPARK_HOME"] = "/content/spark-3.1.2-bin-hadoop3.2"  # Adjust with your correct Spark path

# Initialize Spark
findspark.init()

# Start Spark session
spark = SparkSession.builder \
    .appName("ModelDeployment") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Model path (ensure correct path for your model)
model_path = '/content/gbt_model'  # Adjust path if necessary

# Check if model exists
if os.path.exists(model_path):
    st.write("Model path exists.")
else:
    st.write("Model path does not exist.")

# Try loading the model
try:
    model = GBTClassificationModel.load(model_path)
    st.write("Model loaded successfully.")
except Exception as e:
    st.write(f"Error loading model: {e}")

# Add UI components for input
st.title("PySpark GBT Model Deployment")

# Input features for prediction
age = st.number_input('Age', min_value=0)
job = st.selectbox('Job', ['admin.', 'technician', 'services', 'management', 'retired', 'blue-collar', 'student', 'unemployed', 'housemaid', 'entrepreneur', 'self-employed'])
marital = st.selectbox('Marital', ['single', 'married', 'divorced'])
education = st.selectbox('Education', ['primary', 'secondary', 'tertiary'])
credit_default = st.selectbox('Credit Default', ['no', 'yes'])
house_loan = st.selectbox('House Loan', ['no', 'yes'])
loan = st.selectbox('Loan', ['no', 'yes'])
contact = st.selectbox('Contact', ['cellular', 'telephone', 'unknown'])
month = st.selectbox('Month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
day_of_week = st.selectbox('Day of Week', ['mon', 'tue', 'wed', 'thu', 'fri'])
duration = st.number_input('Duration', min_value=0)
campaign = st.number_input('Campaign', min_value=0)
pdays = st.number_input('Pdays', min_value=-1)  # Typically -1 for not contacted
previous = st.number_input('Previous', min_value=0)
poutcome = st.selectbox('Poutcome', ['failure', 'nonexistent', 'success'])

# Encoding categorical features using StringIndexer (or pre-defined mappings)
def encode_features():
    # Example of encoding categorical columns
    job_index = {'admin.': 0, 'technician': 1, 'services': 2, 'management': 3, 'retired': 4, 'blue-collar': 5, 'student': 6, 'unemployed': 7, 'housemaid': 8, 'entrepreneur': 9, 'self-employed': 10}
    marital_index = {'single': 0, 'married': 1, 'divorced': 2}
    education_index = {'primary': 0, 'secondary': 1, 'tertiary': 2}
    credit_default_int = {'no': 0, 'yes': 1}
    house_loan_int = {'no': 0, 'yes': 1}
    loan_int = {'no': 0, 'yes': 1}
    contact_index = {'cellular': 0, 'telephone': 1, 'unknown': 2}
    month_index = {'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5, 'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11}
    day_of_week_index = {'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4}
    poutcome_index = {'failure': 0, 'nonexistent': 1, 'success': 2}

    # Encode the input features into corresponding integers
    return [
        age, 
        job_index.get(job, -1), 
        marital_index.get(marital, -1), 
        education_index.get(education, -1), 
        credit_default_int.get(credit_default, -1), 
        house_loan_int.get(house_loan, -1), 
        loan_int.get(loan, -1), 
        contact_index.get(contact, -1), 
        month_index.get(month, -1), 
        day_of_week_index.get(day_of_week, -1), 
        duration, 
        campaign, 
        pdays, 
        previous, 
        poutcome_index.get(poutcome, -1)
    ]

# Get encoded features
encoded_features = encode_features()

# Create DataFrame for prediction
def make_prediction(features):
    columns = ['age', 'job_index', 'marital_index', 'education_index', 'credit_default_int', 'house_loan_int', 'loan_int', 'contact_index', 'month_index', 'day_of_week_index', 'duration', 'campaign', 'pdays', 'previous', 'poutcome_index']
    df = spark.createDataFrame([features], columns)
    
    # Make prediction
    predictions = model.transform(df)
    return predictions

# When user clicks 'Predict' button
if st.button("Predict"):
    result = make_prediction(encoded_features)
    
    # Display prediction result
    st.write("Prediction Results:")
    prediction = result.select("prediction").head()[0]
    st.write(f"Prediction: {prediction}")
