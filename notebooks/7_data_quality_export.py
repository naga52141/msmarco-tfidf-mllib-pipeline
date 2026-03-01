import argparse
from pyspark.sql import SparkSession, functions as F


def build_spark():
    return (
        SparkSession.builder.appName("MSMARCO_DataQuality")
        .master("local[*]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_parquet", default="data/processed/msmarco_parquet")
    p.add_argument("--out_dir", default="data/processed/tableau_data_quality")
    args = p.parse_args()

    spark = build_spark()
    df = spark.read.parquet(args.in_parquet).select("qid", "docid", "query", "passage", "label")


    missing = df.select(
        F.sum(F.col("query").isNull().cast("int")).alias("missing_query"),
        F.sum(F.col("passage").isNull().cast("int")).alias("missing_passage"),
        F.sum(F.col("label").isNull().cast("int")).alias("missing_label"),
    )


    label_dist = df.groupBy("label").count()


    text_stats = df.select(
        F.length("query").alias("query_len"),
        F.length("passage").alias("passage_len"),
    ).select(
        F.expr("percentile_approx(query_len, 0.5)").alias("query_len_median"),
        F.expr("percentile_approx(passage_len, 0.5)").alias("passage_len_median"),
        F.avg("query_len").alias("query_len_mean"),
        F.avg("passage_len").alias("passage_len_mean"),
        F.max("passage_len").alias("passage_len_max"),
    )

    missing.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{args.out_dir}/missingness")
    label_dist.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{args.out_dir}/label_distribution")
    text_stats.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{args.out_dir}/text_length_stats")

    print(f" Saved Data Quality CSVs to: {args.out_dir}")
    spark.stop()


if __name__ == "__main__":
    main()
