import argparse
import time
import numpy as np

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.linalg import SparseVector

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix


def build_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def to_scipy_sparse(rows, num_features: int):

    data = []
    indices = []
    indptr = [0]
    y = np.zeros(len(rows), dtype=np.int32)

    for i, r in enumerate(rows):
        v: SparseVector = r["features"]
        y[i] = int(r["label"])
        data.extend(v.values)
        indices.extend(v.indices)
        indptr.append(len(data))

    return np.array(data, dtype=np.float32), np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int32), y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", default="data/processed/features_tfidf")
    p.add_argument("--sample_n", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    spark = build_spark("MSMARCO_sklearn_baseline")


    train = spark.read.parquet(f"{args.features_dir}/train").withColumn("label", F.col("label").cast("int"))
    test = spark.read.parquet(f"{args.features_dir}/test").withColumn("label", F.col("label").cast("int"))


    train_s = train.orderBy(F.rand(args.seed)).limit(args.sample_n)
    test_s = test.orderBy(F.rand(args.seed + 1)).limit(max(5000, args.sample_n // 4))


    train_rows = train_s.select("features", "label").collect()
    test_rows = test_s.select("features", "label").collect()

    num_features = train_rows[0]["features"].size

    try:
        import scipy.sparse as sp

        tr_data, tr_idx, tr_ptr, y_train = to_scipy_sparse(train_rows, num_features)
        te_data, te_idx, te_ptr, y_test = to_scipy_sparse(test_rows, num_features)

        X_train = sp.csr_matrix((tr_data, tr_idx, tr_ptr), shape=(len(y_train), num_features))
        X_test = sp.csr_matrix((te_data, te_idx, te_ptr), shape=(len(y_test), num_features))

    except Exception:

        if args.sample_n > 8000:
            raise RuntimeError(
           
            )
        y_train = np.array([int(r["label"]) for r in train_rows])
        y_test = np.array([int(r["label"]) for r in test_rows])
        X_train = np.vstack([r["features"].toArray() for r in train_rows])
        X_test = np.vstack([r["features"].toArray() for r in test_rows])


    clf = LogisticRegression(max_iter=200, n_jobs=-1)

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    t1 = time.time()
    y_pred = clf.predict(X_test)
    infer_time = time.time() - t1


    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)


    try:
        y_score = clf.predict_proba(X_test)[:, 1]
    except Exception:
        y_score = clf.decision_function(X_test)

    auc = roc_auc_score(y_test, y_score)
    cm = confusion_matrix(y_test, y_pred)


    print(f"Train sample: {len(y_train)} rows")
    print(f"Test sample : {len(y_test)} rows")
    print(f"Train time (s): {train_time:.2f}")
    print(f"Infer time (s): {infer_time:.2f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print("Confusion matrix [[TN, FP],[FN, TP]]:")
    print(cm)

    spark.stop()


if __name__ == "__main__":
    main()
