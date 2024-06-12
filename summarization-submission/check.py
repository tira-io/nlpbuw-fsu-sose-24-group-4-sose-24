from tira.rest_api_client import Client

# Initialize the Client
tira = Client()

# Load the data
df = tira.pd.inputs("nlpbuw-fsu-sose-24", "summarization-validation-20240530-training")

# Display the first few rows and data types of the dataframe
print(df.head())
print(df.info())
