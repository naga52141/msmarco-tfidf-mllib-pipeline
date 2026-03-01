

import argparse
import time
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.classification import LogisticRegression, LinearSVC
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
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


def tune_lr(train_df, evaluator, parallelism: int):
    lr = LogisticRegression(featuresCol="features", labelCol="label")


    grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.01, 0.1])
        .addGrid(lr.elasticNetParam, [0.0, 0.5])
        .addGrid(lr.maxIter, [20])
        .build()
    )

    cv = CrossValidator(
        estimator=lr,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=parallelism,
        seed=42,
    )
    return cv.fit(train_df)


def tune_svm(train_df, evaluator, parallelism: int):
    svm = LinearSVC(featuresCol="features", labelCol="label")


    grid = (
        ParamGridBuilder()
        .addGrid(svm.regParam, [0.01, 0.1])
        .addGrid(svm.maxIter, [20])
        .build()
    )

    cv = CrossValidator(
        estimator=svm,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=parallelism,
        seed=42,
    )
    return cv.fit(train_df)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", default="data/processed/features_tfidf")
    p.add_argument("--out_dir", default="data/processed/cv_models")
    p.add_argument("--driver_mem", default="4g")
    p.add_argument("--exec_mem", default="4g")
    p.add_argument("--shuffle_parts", type=int, default=8)
    p.add_argument("--parallelism", type=int, default=2) 
    p.add_argument("--tune_fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    spark = build_spark("MSMARCO_CrossValidator_Lite", args.driver_mem, args.exec_mem, args.shuffle_parts)

    train_df = (
        spark.read.parquet(f"{args.features_dir}/train")
        .withColumn("label", F.col("label").cast("double"))
        .select("features", "label")
    )

    if args.tune_fraction < 1.0:
        train_df = train_df.sample(withReplacement=False, fraction=args.tune_fraction, seed=args.seed)

    train_df = train_df.persist()

    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )

    print("\n=== CrossValidator: Logistic Regression (Laptop Grid) ===")
    t0 = time.time()
    lr_cv_model = tune_lr(train_df, evaluator, args.parallelism)
    lr_time = time.time() - t0
    best_lr = lr_cv_model.bestModel
    print(f"Done in {lr_time:.2f}s")
    print(
        "Best LR params:",
        "regParam=", best_lr.getRegParam(),
        "elasticNetParam=", best_lr.getElasticNetParam(),
        "maxIter=", best_lr.getMaxIter(),
    )

    lr_path = f"{args.out_dir}/logreg_cv"
    lr_cv_model.write().overwrite().save(lr_path)
    print("Saved:", lr_path)

    print("\n=== CrossValidator: LinearSVC (Laptop Grid) ===")
    t1 = time.time()
    svm_cv_model = tune_svm(train_df, evaluator, args.parallelism)
    svm_time = time.time() - t1
    best_svm = svm_cv_model.bestModel
    print(f"Done in {svm_time:.2f}s")
    print(
        "Best SVM params:",
        "regParam=", best_svm.getRegParam(),
        "maxIter=", best_svm.getMaxIter(),
    )

    svm_path = f"{args.out_dir}/svm_cv"
    svm_cv_model.write().overwrite().save(svm_path)
    print("Saved:", svm_path)

    train_df.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()