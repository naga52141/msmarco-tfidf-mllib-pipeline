import argparse
import time
import os

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def build_spark(app_name: str, driver_mem: str, exec_mem: str, shuffle_parts: int) -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.memory", driver_mem)
        .config("spark.executor.memory", exec_mem)
        .config("spark.sql.shuffle.partitions", str(shuffle_parts))
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_parquet", default="data/processed/msmarco_parquet")
    p.add_argument("--out_csv", default="data/processed/scalability_results.csv")
    p.add_argument("--driver_mem", default="4g")
    p.add_argument("--exec_mem", default="4g")

    p.add_argument("--partitions_list", default="4,8,16")


    p.add_argument("--sizes_list", default="5000,10000,20000") 

    p.add_argument("--num_features", type=int, default=1 << 18)
    p.add_argument("--min_doc_freq", type=int, default=2)
    p.add_argument("--lr_regparam", type=float, default=0.1)
    p.add_argument("--max_iter", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    partitions_list = [int(x.strip()) for x in args.partitions_list.split(",")]
    sizes_list = [int(x.strip()) for x in args.sizes_list.split(",")]


    results = []
    header = "mode,shuffle_partitions,max_queries,train_rows,train_time_s,auc\n"


    for shuffle_parts in partitions_list:
        spark = build_spark("MSMARCO_Scalability", args.driver_mem, args.exec_mem, shuffle_parts)

        base = spark.read.parquet(args.in_parquet).select("qid", "text", "label")
        base = base.withColumn("label", F.col("label").cast("double"))

        evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

        for max_q in sizes_list:

            qids = base.select("qid").dropDuplicates().orderBy("qid").limit(max_q)
            df = base.join(F.broadcast(qids), on="qid", how="inner").select("text", "label")


            tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\\W+", toLowercase=True)
            remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
            tf = HashingTF(inputCol="filtered", outputCol="tf", numFeatures=args.num_features)
            idf = IDF(inputCol="tf", outputCol="features", minDocFreq=args.min_doc_freq)

            lr = LogisticRegression(featuresCol="features", labelCol="label", regParam=args.lr_regparam, maxIter=args.max_iter)

            pipe = Pipeline(stages=[tokenizer, remover, tf, idf, lr])


            train_df, test_df = df.randomSplit([0.8, 0.2], seed=args.seed)
            train_df = train_df.persist()
            test_df = test_df.persist()


            t0 = time.time()
            model = pipe.fit(train_df)
            train_time = time.time() - t0

            pred = model.transform(test_df)
            auc = evaluator.evaluate(pred)

            train_rows = train_df.count()


            results.append(("scalability", shuffle_parts, max_q, train_rows, train_time, auc))
            print(f"shuffle={shuffle_parts} max_queries={max_q} train_rows={train_rows} time={train_time:.2f}s auc={auc:.4f}")

            train_df.unpersist()
            test_df.unpersist()

        spark.stop()


    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8") as f:
        f.write(header)
        for mode, shuffle_parts, max_q, train_rows, train_time, auc in results:
            f.write(f"{mode},{shuffle_parts},{max_q},{train_rows},{train_time:.4f},{auc:.6f}\n")

    print(f"\n Saved scalability results to: {args.out_csv}")


if __name__ == "__main__":
    main()
