from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import udf, concat_ws, col, count
from pyspark.sql.types import StringType, BooleanType
from pyspark.testing import assertSchemaEqual
import hashlib
import pandas as pd
import xlsxwriter
import time

def init_spark():
    return SparkSession.builder.appName("exampleApp").getOrCreate()

def pandas_to_spark(df, spark):
    return spark.createDataFrame(df)

def match_schemas(df1, df2):
    try:
        columns1 = set(df1.columns)
        columns2 = set(df2.columns)
        
        if columns1 == columns2:
            print("Column names are equal.")
            
            schema1 = df1.dtypes
            schema2 = df2.dtypes

            mismatched_columns = [(col1[0], col1[1], col2[1]) for col1, col2 in zip(schema1, schema2) if col1[0] == col2[0] and col1[1] != col2[1]]

            if mismatched_columns:
                print("\nColumns with mismatched data types:")
                for col in mismatched_columns:
                    print(f"Column '{col[0]}' has mismatched data types: {col[1]} (DataFrame 1) vs {col[2]} (DataFrame 2)")
                
                # Adjust df2 columns to match df1 data types
                for col in mismatched_columns:
                    df2 = df2.withColumn(col[0], df2[col[0]].cast(col[1]))
            else:
                print("\nNo mismatched data types found.")

        else:
            print("Error: Column names do not match.")
            columns_only_in_df1 = columns1 - columns2
            columns_only_in_df2 = columns2 - columns1
            
            if columns_only_in_df1:
                print(f"Columns in DataFrame 1 only: {columns_only_in_df1}")
            
            if columns_only_in_df2:
                print(f"Columns in DataFrame 2 only: {columns_only_in_df2}")
            
            raise AssertionError("Column names do not match.")

    except AssertionError as e:
        return None, None

    return df1, df2


def hash_row(concatenated_string):
    return hashlib.md5(concatenated_string.encode('utf-8')).hexdigest()

def compare_dataframes(df1, df2, spark):
    hash_udf = udf(hash_row, StringType())
    df1_hashes = df1.withColumn("hash", hash_udf(concat_ws(",", *df1.columns)))
    df2_hashes = df2.withColumn("hash", hash_udf(concat_ws(",", *df2.columns)))

    df1_hash_set = set(df1_hashes.select("hash").rdd.map(lambda x: x[0]).collect())
    df2_hash_set = set(df2_hashes.select("hash").rdd.map(lambda x: x[0]).collect())

    df1_hash_broadcast = spark.sparkContext.broadcast(df1_hash_set)
    df2_hash_broadcast = spark.sparkContext.broadcast(df2_hash_set)

    is_in_df2_hashes_udf = udf(lambda hash_value: hash_value in df2_hash_broadcast.value, BooleanType())
    is_in_df1_hashes_udf = udf(lambda hash_value: hash_value in df1_hash_broadcast.value, BooleanType())

    df1_unique_hash = df1_hashes.filter(~is_in_df2_hashes_udf(col("hash")))
    df2_unique_hash = df2_hashes.filter(~is_in_df1_hashes_udf(col("hash")))
    matched_hash = df1_hashes.filter(is_in_df2_hashes_udf(col("hash"))).limit(1000)

    df1_duplicates = df1_hashes.groupBy("hash").agg(count("*").alias("count")).filter(col("count") > 1).select("hash")
    df2_duplicates = df2_hashes.groupBy("hash").agg(count("*").alias("count")).filter(col("count") > 1).select("hash")

    df1_duplicate_rows = df1_hashes.join(df1_duplicates, "hash")
    df2_duplicate_rows = df2_hashes.join(df2_duplicates, "hash")

    return df1_unique_hash.drop("hash"), df2_unique_hash.drop("hash"), matched_hash.drop("hash"), df1_duplicate_rows.drop("hash"), df2_duplicate_rows.drop("hash")

def identify_unique_rows(df1_unique, df2_unique):
    df1_identifiers = df1_unique.withColumn("identifier", concat_ws("_", *df1_unique.columns))
    df2_identifiers = df2_unique.withColumn("identifier", concat_ws("_", *df2_unique.columns))

    identifiers_df1 = set(df1_identifiers.select("identifier").rdd.map(lambda row: row[0]).collect())
    identifiers_df2 = set(df2_identifiers.select("identifier").rdd.map(lambda row: row[0]).collect())

    def jaccard_similarity(id1, id2):
        set1 = set(id1.split("_"))
        set2 = set(id2.split("_"))
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if len(union) != 0 else 0.0

    filtered_identifiers = {id1 for id1 in identifiers_df1 if not any(jaccard_similarity(id1, id2) >= 0.5 for id2 in identifiers_df2)} | \
                           {id2 for id2 in identifiers_df2 if not any(jaccard_similarity(id2, id1) >= 0.5 for id1 in identifiers_df1)}

    mismatched_df = df1_identifiers.filter(col("identifier").isin(filtered_identifiers)).union(
        df2_identifiers.filter(col("identifier").isin(filtered_identifiers))
    )

    only_df1 = mismatched_df.filter(~col("identifier").isin(identifiers_df2)).drop("identifier")
    only_df2 = mismatched_df.filter(~col("identifier").isin(identifiers_df1)).drop("identifier")

    df1_unique_filter = df1_unique.subtract(only_df1)
    df2_unique_filter = df2_unique.subtract(only_df2)
     # Convert DataFrames to RDDs and sort by the entire row
    df_actual_rdd = df1_unique_filter.rdd.map(lambda row: tuple(row)).sortBy(lambda row: row)
    df_expected_rdd = df2_unique_filter.rdd.map(lambda row: tuple(row)).sortBy(lambda row: row)
     # Convert RDDs back to Spark DataFrames
    df1_unique_filtered = df_actual_rdd.toDF(df1_unique_filter.schema)
    df2_unique_filtered = df_expected_rdd.toDF(df2_unique_filter.schema)

    return only_df1, only_df2, df1_unique_filtered, df2_unique_filtered

def find_column_changes_and_create_new_row(actual_row, expected_row):
    changes = {key: (f"actual: {actual_row[key]}\n mismatched: {expected_row[key]}" if actual_row[key] != expected_row[key] else actual_row[key]) for key in actual_row.asDict().keys()}
    return Row(**changes)

def compare_dataframes_and_create_diff_dataframe(df_actual, df_expected, spark):
    actual_rows = df_actual.collect()
    expected_rows = df_expected.collect()

    if len(actual_rows) != len(expected_rows):
        raise ValueError(f"Row count mismatch: {len(actual_rows)} (actual) vs {len(expected_rows)} (mismatched)")

    new_rows = [find_column_changes_and_create_new_row(actual_row, expected_row) for actual_row, expected_row in zip(actual_rows, expected_rows)]

    return spark.createDataFrame(new_rows)
 
def generate_excel_report(df_diff, only_df1, only_df2, matched_records, duplicates_in_source, duplicates_in_target, excel_file_path):
    try:
        data_frames = {
            'Miss_Matched_Records': df_diff,
            'Only in Source': only_df1,
            'Only In Target': only_df2,
            'Top 1000 Matched_records': matched_records,
            'Duplicates_in_source': duplicates_in_source,
            'Duplicates_in_Target': duplicates_in_target
        }

        with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
            for sheet_name, df in data_frames.items():
                pandas_df = pd.DataFrame(df.collect(), columns=df.schema.names)
                if pandas_df.empty:
                    pandas_df = pd.DataFrame(["No records"])
                    pandas_df.to_excel(writer, sheet_name=sheet_name, header=False, index=False)
                else:
                    pandas_df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"DataFrames saved successfully in {excel_file_path}")

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        print("Completed")

def run_comparison_project(Dataframe1, Dataframe2):
    try:
        start_time = time.time()

        spark = init_spark()

        # Convert DataFrames to Spark DataFrames if they are Pandas DataFrames
        if isinstance(Dataframe1, pd.DataFrame):
            df1 = pandas_to_spark(Dataframe1, spark)
        else:
            df1 = Dataframe1
        
        if isinstance(Dataframe2, pd.DataFrame):
            df2 = pandas_to_spark(Dataframe2, spark)
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
        excel_file_path = 'Spark_Comparison.xlsx'
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
    run_comparison_project(df1_pd, df2_pd)
