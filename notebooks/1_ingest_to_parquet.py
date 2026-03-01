import argparse
import os
from pyspark.sql import SparkSession, functions as F, types as T


def build_spark(app_name: str, driver_mem: str, exec_mem: str, shuffle_parts: int) -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", driver_mem)
        .config("spark.executor.memory", exec_mem)
        .config("spark.sql.shuffle.partitions", str(shuffle_parts))
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="data/raw")
    p.add_argument("--out_dir", default="data/processed/msmarco_parquet")
    p.add_argument("--driver_mem", default="2g")
    p.add_argument("--exec_mem", default="4g")
    p.add_argument("--shuffle_parts", type=int, default=8)


    p.add_argument("--max_queries", type=int, default=20000)     
    p.add_argument("--neg_per_query", type=int, default=1)     
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    spark = build_spark(
        "MSMARCO_Ingestion",
        driver_mem=args.driver_mem,
        exec_mem=args.exec_mem,
        shuffle_parts=args.shuffle_parts,
    )

    raw = args.raw_dir
    collection_path = os.path.join(raw, "collection.tsv")
    queries_path = os.path.join(raw, "queries.train.tsv")
    qrels_path = os.path.join(raw, "qrels.train.tsv")

    queries_schema = T.StructType([
        T.StructField("qid", T.StringType(), False),
        T.StructField("query", T.StringType(), True),
    ])
    queries = (
        spark.read.option("sep", "\t").schema(queries_schema).csv(queries_path)
        .dropna(subset=["qid"])
        .dropDuplicates(["qid"])
    )


    queries = queries.orderBy("qid").limit(args.max_queries)


    qrels_schema = T.StructType([
        T.StructField("qid", T.StringType(), False),
        T.StructField("unused", T.StringType(), True),
        T.StructField("docid", T.StringType(), False),
        T.StructField("relevance", T.IntegerType(), True),
    ])
    qrels = spark.read.option("sep", "\t").schema(qrels_schema).csv(qrels_path)


    qrels = qrels.join(F.broadcast(queries.select("qid")), on="qid", how="inner")


    pos = (
        qrels.filter(F.col("relevance") > 0)
        .select("qid", "docid")
        .withColumn("label", F.lit(1))
        .dropDuplicates(["qid", "docid"])
    )

    all_docids = qrels.select("docid").dropDuplicates()

    qid_list = pos.select("qid").dropDuplicates()
    candidates = (
        qid_list.crossJoin(all_docids.sample(withReplacement=False, fraction=0.001, seed=args.seed))
        .withColumn("rnd", F.rand(args.seed))
        .withColumn("rn", F.row_number().over(
            __import__("pyspark").sql.Window.partitionBy("qid").orderBy(F.col("rnd"))
        ))
        .filter(F.col("rn") <= F.lit(args.neg_per_query))
        .select("qid", "docid")
    )

    neg = (
        candidates.join(pos.select("qid", "docid"), on=["qid", "docid"], how="left_anti")
        .withColumn("label", F.lit(0))
        .dropDuplicates(["qid", "docid"])
    )

    pairs = pos.unionByName(neg).repartition("label").persist()

    collection_schema = T.StructType([
        T.StructField("docid", T.StringType(), False),
        T.StructField("passage", T.StringType(), True),
    ])

    collection = spark.read.option("sep", "\t").schema(collection_schema).csv(collection_path)

    needed_docids = pairs.select("docid").dropDuplicates()
    collection_small = collection.join(F.broadcast(needed_docids), on="docid", how="inner")

    dataset = (
        pairs.join(F.broadcast(queries), on="qid", how="inner")
        .join(collection_small, on="docid", how="inner")
        .withColumn("text", F.concat_ws(" [SEP] ", F.col("query"), F.col("passage")))
        .select("qid", "docid", "query", "passage", "text", "label")
    )

    dataset = dataset.dropna(subset=["text", "label"]).dropDuplicates(["qid", "docid"])

    out = args.out_dir
    (dataset
        .write
        .mode("overwrite")
        .partitionBy("label")
        .parquet(out)
    )

    sample_out = os.path.join("data/samples", "msmarco_sample.parquet")
    dataset.limit(50000).write.mode("overwrite").parquet(sample_out)

    print(f"\n Wrote Parquet dataset to: {out}")
    print(f" Wrote sample to: {sample_out}")
    print("Rows:", dataset.count())
    print("Label counts:")
    dataset.groupBy("label").count().show()

    pairs.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
