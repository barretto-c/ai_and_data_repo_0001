# Step 1: Define schema for the Adult dataset
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

schema = StructType([
    StructField("age", IntegerType(), True),
    StructField("workclass", StringType(), True),
    StructField("fnlwgt", IntegerType(), True),
    StructField("education", StringType(), True),
    StructField("education_num", IntegerType(), True),
    StructField("marital_status", StringType(), True),
    StructField("occupation", StringType(), True),
    StructField("relationship", StringType(), True),
    StructField("race", StringType(), True),
    StructField("sex", StringType(), True),
    StructField("capital_gain", IntegerType(), True),
    StructField("capital_loss", IntegerType(), True),
    StructField("hours_per_week", IntegerType(), True),
    StructField("native_country", StringType(), True),
    StructField("income", StringType(), True)
])

# Step 2: Load dataset from Databricks sample path
df = spark.read.schema(schema).csv("dbfs:/databricks-datasets/adult/adult.data")

# Step 3: Inspect first few rows
df.show(5)

# Step 4: Register DataFrame as a SQL table
df.createOrReplaceTempView("adult")

# Step 5: Run SQL queries

# Count total rows
spark.sql("SELECT COUNT(*) AS total_rows FROM adult").show()

# Distinct occupations
spark.sql("SELECT DISTINCT occupation FROM adult").show()

# Average age by education
spark.sql("""
SELECT education, AVG(age) AS avg_age
FROM adult
GROUP BY education
ORDER BY avg_age DESC
""").show()

# Income distribution
spark.sql("""
SELECT income, COUNT(*) AS count
FROM adult
GROUP BY income
ORDER BY count DESC
""").show()

# Top 5 occupations by count
spark.sql("""
SELECT occupation, COUNT(*) AS count
FROM adult
GROUP BY occupation
ORDER BY count DESC
LIMIT 5
""").show()


# Step 5: Transform - ETL pipeline
# Clean: filter out null ages, aggregate counts by age
cleaned = df.filter(df.age.isNotNull()).groupBy("age").count()

# Load: write results to Delta Lake
# NOTE: replace '/mnt/datalake/cleaned_people' with your mounted storage path
cleaned.write.format("delta").mode("overwrite").save("/mnt/datalake/cleaned_people")

# Verify: read back the Delta table
result = spark.read.format("delta").load("/mnt/datalake/cleaned_people")
result.show(10)
