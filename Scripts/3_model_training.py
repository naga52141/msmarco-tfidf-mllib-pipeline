import argparse
import time

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.classification import LogisticRegression, LinearSVC, NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics


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


def evaluate_predictions(pred_df):

    acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    prec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    rec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

    accuracy = acc_eval.evaluate(pred_df)
    f1 = f1_eval.evaluate(pred_df)
    precision = prec_eval.evaluate(pred_df)
    recall = rec_eval.evaluate(pred_df)
    roc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    roc_auc = roc_eval.evaluate(pred_df)

    preds_and_labels = pred_df.select("prediction", "label").rdd.map(lambda r: (float(r[0]), float(r[1])))
    metrics = MulticlassMetrics(preds_and_labels)
    cm = metrics.confusionMatrix().toArray()

    return accuracy, f1, precision, recall, roc_auc, cm


def train_one(name, estimator, train_df, test_df, model_out_dir):
    t0 = time.time()
    model = estimator.fit(train_df)
    train_time = time.time() - t0

    t1 = time.time()
    pred = model.transform(test_df)
    infer_time = time.time() - t1

    accuracy, f1, precision, recall, roc_auc, cm = evaluate_predictions(pred)

    model_path = f"{model_out_dir}/{name}"
    model.write().overwrite().save(model_path)

    return train_time, infer_time, accuracy, f1, precision, recall, roc_auc, cm, model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="data/processed/features_tfidf")
    parser.add_argument("--model_out_dir", default="data/processed/models")
    parser.add_argument("--driver_mem", default="4g")   
    parser.add_argument("--exec_mem", default="4g")
    parser.add_argument("--shuffle_parts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    spark = build_spark("MSMARCO_ModelTraining", args.driver_mem, args.exec_mem, args.shuffle_parts)


    train_df = (
        spark.read.parquet(f"{args.features_dir}/train")
        .withColumn("label", F.col("label").cast("double"))
        .select("features", "label")
        .persist()
    )
    test_df = (
        spark.read.parquet(f"{args.features_dir}/test")
        .withColumn("label", F.col("label").cast("double"))
        .select("features", "label")
        .persist()
    )


    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=30, regParam=0.1)
    svm = LinearSVC(featuresCol="features", labelCol="label", maxIter=30, regParam=0.1)
    nb = NaiveBayes(featuresCol="features", labelCol="label", modelType="multinomial", smoothing=1.0)

    models = [
        ("logistic_regression", lr),
        ("linear_svc", svm),
        ("naive_bayes", nb),
    ]

    print("\n==================== MODEL RESULTS ====================")

    results_rows = []

    for name, estimator in models:
        train_time, infer_time, accuracy, f1, precision, recall, roc_auc, cm, path = train_one(
            name, estimator, train_df, test_df, args.model_out_dir
        )

        print(f"\nModel: {name}")
        print(f"Saved: {path}")
        print(f"Train time (s): {train_time:.2f}")
        print(f"Infer time (s): {infer_time:.2f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"Precision (weighted): {precision:.4f}")
        print(f"Recall (weighted): {recall:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print("Confusion matrix [[TN, FP],[FN, TP]]:")
        print(cm)

        results_rows.append(
            (name, float(train_time), float(infer_time), float(accuracy), float(f1), float(precision), float(recall), float(roc_auc))
        )

    # Save metrics for Tableau as CSV (single file)
    metrics_df = spark.createDataFrame(
        results_rows,
        ["model", "train_time_s", "infer_time_s", "accuracy", "f1", "precision_w", "recall_w", "roc_auc"],
    )
    out_csv_dir = "data/processed/metrics_tableau"
    metrics_df.coalesce(1).write.mode("overwrite").option("header", True).csv(out_csv_dir)
    print(f"\n Saved metrics CSV for Tableau: {out_csv_dir}")

    train_df.unpersist()
    test_df.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()