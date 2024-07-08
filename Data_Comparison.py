from pyspark.sql import SparkSession,Row
from pyspark.sql.functions import udf, concat_ws, col, count
from pyspark.sql.types import StringType, BooleanType
import hashlib
import pandas as pd
import xlsxwriter
import time
# Initialize Spark session
def init_spark():
    spark = SparkSession.builder \
        .appName("exampleApp") \
        .getOrCreate()
    return spark

# Convert Pandas DataFrame to Spark DataFrame
def pandas_to_spark(df, spark):
    spark_df = spark.createDataFrame(df)
    return spark_df

# Match schemas between two DataFrames and convert data types
def match_schemas(df1, df2):
    # Compare schemas and find mismatched data types
    schema1 = df1.dtypes
    schema2 = df2.dtypes

    mismatched_columns = []

    for col1, col2 in zip(schema1, schema2):
        if col1[0] == col2[0] and col1[1] != col2[1]:
            mismatched_columns.append((col1[0], col1[1], col2[1]))

    if mismatched_columns:
        print("\nColumns with mismatched data types:")
        for col in mismatched_columns:
            print(f"Column '{col[0]}' has mismatched data types: {col[1]} (DataFrame 1) vs {col[2]} (DataFrame 2)")
    else:
        print("\nNo mismatched data types found.")

    # Function to convert column data type in df2 to match df1
    def convert_column(df, col_name, target_type):
        return df.withColumn(col_name, df[col_name].cast(target_type))

    # Convert mismatched data types in df2 to match df1
    for col in mismatched_columns:
        df2 = convert_column(df2, col[0], col[1])

    return df1, df2

# Compare two DataFrames and identify unique, matched, and duplicate records
# Define UDF for hashing
def hash_row(concatenated_string):
    return hashlib.md5(concatenated_string.encode('utf-8')).hexdigest()

# Function to compare dataframes
def compare_dataframes(df1, df2,spark):
    try:
        # Register UDF for hashing
        hash_udf = udf(hash_row, StringType())
        # Apply the UDF to each row in df1 and df2
        df1_hashes = df1.withColumn("hash", hash_udf(concat_ws(",", *df1.columns))).select("hash", "*")
        df2_hashes = df2.withColumn("hash", hash_udf(concat_ws(",", *df2.columns))).select("hash", "*")
        # Collect sets of hashed rows for comparison
        df1_hash_set = set(df1_hashes.select("hash").rdd.map(lambda x: x[0]).collect())
        df2_hash_set = set(df2_hashes.select("hash").rdd.map(lambda x: x[0]).collect())
        # Broadcast the hash sets
        df1_hash_broadcast = spark.sparkContext.broadcast(df1_hash_set)
        df2_hash_broadcast = spark.sparkContext.broadcast(df2_hash_set)

        # Define UDFs for broadcasted set lookup returning boolean
        def is_in_df2_hashes(hash_value):
            return hash_value in df2_hash_broadcast.value

        def is_in_df1_hashes(hash_value):
            return hash_value in df1_hash_broadcast.value

        is_in_df2_hashes_udf = udf(is_in_df2_hashes, BooleanType())
        is_in_df1_hashes_udf = udf(is_in_df1_hashes, BooleanType())
        # Filter rows based on hash comparison using broadcasted sets
        df1_unique_hash = df1_hashes.filter(~is_in_df2_hashes_udf(col("hash")))
        df2_unique_hash = df2_hashes.filter(~is_in_df1_hashes_udf(col("hash")))
        matched_hash = df1_hashes.filter(is_in_df2_hashes_udf(col("hash"))).limit(1000)
        # Find duplicate records within each DataFrame
        df1_duplicates = df1_hashes.groupBy("hash").agg(count("*").alias("count")).filter(col("count") > 1).select("hash")
        df2_duplicates = df2_hashes.groupBy("hash").agg(count("*").alias("count")).filter(col("count") > 1).select("hash")

        # Join back to get the original duplicate rows
        df1_duplicate_rows = df1_hashes.join(df1_duplicates, "hash").drop("count")
        df2_duplicate_rows = df2_hashes.join(df2_duplicates, "hash").drop("count")
        # Return the original rows corresponding to unique hashes
        return df1_unique_hash.drop("hash"), df2_unique_hash.drop("hash"), matched_hash.drop("hash"),df1_duplicate_rows.drop("hash"),df2_duplicate_rows.drop("hash")

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None, None
        

# Identify rows that are only in A or B using a similarity metric
def identify_unique_rows(df1_unique, df2_unique):
    # Add a unique identifier column to each DataFrame
    df1_identifiers = df1_unique.withColumn("identifier", concat_ws("_", *df1_unique.columns))
    df2_identifiers = df2_unique.withColumn("identifier", concat_ws("_", *df2_unique.columns))

    # Collect all identifiers from both DataFrames
    identifiers_df1 = set(df1_identifiers.select("identifier").rdd.map(lambda row: row[0]).collect())
    identifiers_df2 = set(df2_identifiers.select("identifier").rdd.map(lambda row: row[0]).collect())

    # Define a function to calculate similarity between two identifiers
    def jaccard_similarity(id1, id2):
        set1 = set(id1.split("_"))
        set2 = set(id2.split("_"))
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if len(union) != 0 else 0.0

    # Filter identifiers that are too similar
    filtered_identifiers = set()
    for id1 in identifiers_df1:
        similar_found = False
        for id2 in identifiers_df2:
            if jaccard_similarity(id1, id2) >= 0.5:  # Adjust threshold as needed
                similar_found = True
                break
        if not similar_found:
            filtered_identifiers.add(id1)

    for id2 in identifiers_df2:
        similar_found = False
        for id1 in identifiers_df1:
            if jaccard_similarity(id1, id2) >= 0.5:  # Adjust threshold as needed
                similar_found = True
                break
        if not similar_found:
            filtered_identifiers.add(id2)

    # Filter and show only rows with these unique identifiers
    mismatched_df = df1_identifiers.filter(col("identifier").isin(filtered_identifiers)).union(
        df2_identifiers.filter(col("identifier").isin(filtered_identifiers))
    )

    # Separate records exclusive to df1 and df2
    only_df1 = mismatched_df.filter(~col("identifier").isin(identifiers_df2)).drop("identifier")
    only_df2 = mismatched_df.filter(~col("identifier").isin(identifiers_df1)).drop("identifier")
    df1_unique_filter = df1_unique.exceptAll(only_df1)
    # Remove rows from df1_unique that are in only_df2
    df2_unique_filter = df2_unique.exceptAll(only_df2)

    # Convert DataFrames to RDDs and sort by the entire row
    df_actual_rdd = df1_unique_filter.rdd.map(lambda row: tuple(row)).sortBy(lambda row: row)
    df_expected_rdd = df2_unique_filter.rdd.map(lambda row: tuple(row)).sortBy(lambda row: row)
     # Convert RDDs back to Spark DataFrames
    df1_unique_filtered = df_actual_rdd.toDF(df1_unique_filter.schema)
    df2_unique_filtered = df_expected_rdd.toDF(df2_unique_filter.schema)

    return only_df1, only_df2,df1_unique_filtered,df2_unique_filtered


# Function to find column changes and create a new row with actual and expected values
def find_column_changes_and_create_new_row(actual_row, expected_row):
    changes = actual_row.asDict()
    for key in actual_row.asDict().keys():
        actual_value = actual_row[key]
        expected_value = expected_row[key]
        if actual_value != expected_value:
            changes[key] = f"actual: {actual_value}\n mismatched: {expected_value}"
    return Row(**changes)

# Function to compare DataFrames and create a new DataFrame with changes
def compare_dataframes_and_create_diff_dataframe(df_actual, df_expected,spark):
    # Collect rows from DataFrames
    actual_rows = df_actual.collect()
    expected_rows = df_expected.collect()

    # Ensure the number of rows is the same for proper comparison
    if len(actual_rows) != len(expected_rows):
        raise ValueError(f"Row count mismatch: {len(actual_rows)} (actual) vs {len(expected_rows)} (mismatched)")

    # List to hold new rows with differences
    new_rows = []

    # Compare each row and create a new row with differences
    for actual_row, expected_row in zip(actual_rows, expected_rows):
        new_row = find_column_changes_and_create_new_row(actual_row, expected_row)
        new_rows.append(new_row)

    # Create a new DataFrame from the new rows
    df_diff = spark.createDataFrame(new_rows)
    return df_diff


# Function to generate Excel report with various result sets
def generate_excel_report(df_diff, only_df1, only_df2, matched_records, duplicates_in_source, duplicates_in_target, excel_file_path):
    try:
        # Convert collected data to Pandas DataFrames
        data_list1 = df_diff.collect()
        data_list2 = only_df1.collect()
        data_list3 = only_df2.collect()
        data_list4 = matched_records.collect()
        data_list5 = duplicates_in_source.collect()
        data_list6 = duplicates_in_target.collect()

        # Convert collected data to Pandas DataFrames
        pandas_df1 = pd.DataFrame(data_list1, columns=df_diff.schema.names)
        pandas_df2 = pd.DataFrame(data_list2, columns=only_df1.schema.names)
        pandas_df3 = pd.DataFrame(data_list3, columns=only_df2.schema.names)
        pandas_df4 = pd.DataFrame(data_list4, columns=matched_records.schema.names)
        pandas_df5 = pd.DataFrame(data_list5, columns=duplicates_in_source.schema.names)
        pandas_df6 = pd.DataFrame(data_list6, columns=duplicates_in_target.schema.names)

        # Write Pandas DataFrames to Excel file
        with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
            # Function to write DataFrame or "No records" message to Excel
            def write_to_excel(df, sheet_name):
                if df.empty:
                    df_no_records = pd.DataFrame(["No records"])
                    df_no_records.to_excel(writer, sheet_name=sheet_name, header=False, index=False)
                else:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Write each DataFrame to a separate sheet
            write_to_excel(pandas_df1, 'Miss_Matched_Records')
            write_to_excel(pandas_df2, 'Only in Source')
            write_to_excel(pandas_df3, 'Only In Target')
            write_to_excel(pandas_df4, 'Top 1000 Matched_records')
            write_to_excel(pandas_df5, 'Duplicates_in_source')
            write_to_excel(pandas_df6, 'Duplicates_in_Traget')

        print(f"DataFrames saved successfully in {excel_file_path}")

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        print("Completed")

print("All functions Excuted Sucessfully")

# --------------------------Calling the main function-------------------------
# Main function to orchestrate the entire process
def run_comparison_project(Dataframe1, Dataframe2):
    try:
        start_time = time.time()

        spark = init_spark()

        # Convert DataFrames to Spark DataFrames if they are Pandas DataFrames
        if isinstance(Dataframe1, pd.DataFrame):
            df1 = pandas_to_spark(Dataframe1, spark)
            print("Conversion happening")
        else:
            df1 = Dataframe1
        
        if isinstance(Dataframe2, pd.DataFrame):
            df2 = pandas_to_spark(Dataframe2, spark)
            print("Conversion happening")
        else:
            df2 = Dataframe2

        # Match schemas and convert data types
        df1_con, df2_con = match_schemas(df1, df2)

        # Compare DataFrames and get result sets
        df1_unique, df2_unique, matched_records, duplicates_in_source, duplicates_in_target = compare_dataframes(df1_con, df2_con, spark)
        
        # Identify rows only in A or B
        only_df1, only_df2, df1_unique_filtered, df2_unique_filtered = identify_unique_rows(df1_unique, df2_unique)

        # Find column changes and create diff DataFrame
        df_diff = compare_dataframes_and_create_diff_dataframe(df1_unique_filtered, df2_unique_filtered, spark)

        # Generate Excel report
        excel_file_path = 'Spark_Comparison2.xlsx'
        generate_excel_report(df_diff, only_df1, only_df2, matched_records, duplicates_in_source, duplicates_in_target, excel_file_path)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Comparison and report generation completed successfully. Excel file saved at: {excel_file_path}")
        print(f"Total Execution Time: {execution_time} seconds")

    except Exception as e:
        print(f"Error occurred during comparison and report generation: {e}")
if __name__ == "__main__":
    file1_path = 'C:\\Users\\rakesh.chappa\\Downloads\\medical_records3.csv'
    file2_path = 'C:\\Users\\rakesh.chappa\\Downloads\\medical_records4.csv'
    df1_pd = pd.read_csv(file1_path)
    df2_pd = pd.read_csv(file2_path)
    print("Reading over")
    run_comparison_project(df1_pd,df2_pd)
