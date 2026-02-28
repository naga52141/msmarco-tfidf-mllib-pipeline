import argparse
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF


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
    p.add_argument("--in_dir", default="data/processed/msmarco_parquet")
    p.add_argument("--out_dir", default="data/processed/features_tfidf")
    p.add_argument("--driver_mem", default="2g")
    p.add_argument("--exec_mem", default="4g")
    p.add_argument("--shuffle_parts", type=int, default=8)

    p.add_argument("--num_features", type=int, default=1 << 18)  # 262,144
    p.add_argument("--min_doc_freq", type=int, default=2)

    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    spark = build_spark("MSMARCO_FeatureEngineering", args.driver_mem, args.exec_mem, args.shuffle_parts)

    df = spark.read.parquet(args.in_dir).select("text", "label")


    df = df.withColumn("text", F.trim(F.col("text"))).dropna(subset=["text", "label"])

    tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\\W+", toLowercase=True)
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
    tf = HashingTF(inputCol="filtered", outputCol="tf", numFeatures=args.num_features)
    idf = IDF(inputCol="tf", outputCol="features", minDocFreq=args.min_doc_freq)

    pipeline = Pipeline(stages=[tokenizer, remover, tf, idf])
    model = pipeline.fit(df)
    featurized = model.transform(df).select("features", "label")

    train_df, test_df = featurized.randomSplit([args.train_ratio, 1 - args.train_ratio], seed=args.seed)
    train_df = train_df.persist()
    test_df = test_df.persist()

    (train_df.write.mode("overwrite").parquet(f"{args.out_dir}/train"))
    (test_df.write.mode("overwrite").parquet(f"{args.out_dir}/test"))

    model.write().overwrite().save(f"{args.out_dir}/tfidf_pipeline_model")

    print(f"\n Saved train set: {args.out_dir}/train")
    print(f" Saved test set : {args.out_dir}/test")
    print(f" Saved TF-IDF pipeline model: {args.out_dir}/tfidf_pipeline_model")
    print("Train rows:", train_df.count())
    print("Test rows :", test_df.count())

    train_df.unpersist()
    test_df.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
