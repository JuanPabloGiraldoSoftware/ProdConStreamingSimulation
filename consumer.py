from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from pyspark.sql.functions import dayofweek
import streaming as st

spark = SparkSession.builder.appName("StructuredNetworkWordCount").getOrCreate()

# Define schema of the csv
userSchema = StructType().add("ID", "integer").add("Case Number", "string").add("Block","string").add("IUCR","string").add("Primary Type","string").add("Description","string").add("Location Description","string").add("Arrest","boolean").add("Domestic","boolean").add("Beat","integer").add("District","float").add("Ward","float").add("Community Area","float").add("FBI Code","string").add("X Coordinate","float").add("Y Coordinate","float").add("Year","integer").add("Updated On","string").add("Latitude","float").add("Longitude","float").add("Location","string").add("Month","integer").add("Day","integer").add("Hour","integer").add("Minute","integer")

# Read CSV files from set path
df = spark.readStream.option("sep", ",").option("header", "true").schema(userSchema).csv("batches/")

def foreach_batch_function(df, epoch_id):
    # Transform and write batchDF
    print("------BATCH",epoch_id,"------")
    print((df.count(), len(df.columns)))
    df = df.drop('Unnamed: 0')
    st.init_training(df)
    print("-----------------------------")

query = df.writeStream.foreachBatch(foreach_batch_function).start()
query.awaitTermination()
