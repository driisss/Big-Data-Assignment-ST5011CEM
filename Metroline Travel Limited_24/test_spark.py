# Quick Test for PySpark Configuration (Fixed Version)
# Tests the configuration WITHOUT StandardScaler

import os
import sys
from pyspark.sql import SparkSession, Row
from pyspark.ml.feature import VectorAssembler

# Set Python executable (critical for Python 3.13)
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

print("Testing PySpark configuration (WITHOUT StandardScaler)...")

# Initialize Spark with optimized settings (matching app_v2.py)
spark = SparkSession.builder \
    .appName("Test") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "2") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .config("spark.python.worker.reuse", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("✓ Spark initialized")

# Test data
data = [Row(a=1.0, b=2.0, c=3.0) for _ in range(10)]
df = spark.createDataFrame(data)

print("✓ DataFrame created")

# Test VectorAssembler
assembler = VectorAssembler(inputCols=['a', 'b', 'c'], outputCol='features')
assembled = assembler.transform(df)

print("✓ VectorAssembler works")

# Test collect (this is where it would crash before)
result = assembled.select('features').collect()

print(f"✓ Collect works - got {len(result)} rows")

# Test with more data
large_data = [Row(a=float(i), b=float(i*2), c=float(i*3)) for i in range(100)]
large_df = spark.createDataFrame(large_data)
large_assembled = assembler.transform(large_df)
large_result = large_assembled.collect()

print(f"✓ Large collect works - got {len(large_result)} rows")

spark.stop()

print("\n✅ All tests passed! The app should work now.")
print("Next step: Run the Streamlit app with:")
print("  python3.13 -m streamlit run app_v2.py")
