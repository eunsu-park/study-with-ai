# Data Lake와 Data Warehouse

## 개요

데이터 저장소 아키텍처는 조직의 데이터 전략에 핵심적입니다. Data Lake, Data Warehouse, 그리고 둘을 결합한 Lakehouse 아키텍처의 특성과 사용 사례를 이해합니다.

---

## 1. Data Warehouse

### 1.1 개념

```
┌────────────────────────────────────────────────────────────────┐
│                    Data Warehouse                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   특징:                                                        │
│   - 구조화된 데이터 (스키마 정의 필수)                           │
│   - Schema-on-Write (쓰기 시 스키마 적용)                       │
│   - 분석 최적화 (OLAP)                                         │
│   - SQL 기반 쿼리                                              │
│                                                                │
│   ┌──────────────────────────────────────────────────────┐    │
│   │                    Data Warehouse                     │    │
│   │   ┌─────────────────────────────────────────────────┐│    │
│   │   │  Dim Tables    │    Fact Tables                 ││    │
│   │   │  ┌──────────┐  │  ┌──────────┐                 ││    │
│   │   │  │dim_date  │  │  │fact_sales│                 ││    │
│   │   │  │dim_product│  │  │fact_orders│                ││    │
│   │   │  │dim_customer│ │                               ││    │
│   │   │  └──────────┘  │  └──────────┘                 ││    │
│   │   └─────────────────────────────────────────────────┘│    │
│   └──────────────────────────────────────────────────────┘    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 주요 솔루션

| 솔루션 | 유형 | 특징 |
|--------|------|------|
| **Snowflake** | 클라우드 | 분리된 스토리지/컴퓨팅, 자동 확장 |
| **BigQuery** | 클라우드 (GCP) | 서버리스, 페타바이트 규모 |
| **Redshift** | 클라우드 (AWS) | Columnar, MPP 아키텍처 |
| **Synapse** | 클라우드 (Azure) | 통합 분석 플랫폼 |
| **PostgreSQL** | 온프레미스 | 소규모, 오픈소스 |

### 1.3 Data Warehouse SQL 예시

```sql
-- Snowflake/BigQuery 스타일 분석 쿼리

-- 월별 매출 트렌드
SELECT
    d.year,
    d.month,
    d.month_name,
    SUM(f.sales_amount) AS total_sales,
    COUNT(DISTINCT f.customer_sk) AS unique_customers,
    AVG(f.sales_amount) AS avg_order_value,
    -- 전월 대비 성장률
    (SUM(f.sales_amount) - LAG(SUM(f.sales_amount)) OVER (ORDER BY d.year, d.month))
        / NULLIF(LAG(SUM(f.sales_amount)) OVER (ORDER BY d.year, d.month), 0) * 100
        AS mom_growth_pct
FROM fact_sales f
JOIN dim_date d ON f.date_sk = d.date_sk
WHERE d.year >= 2023
GROUP BY d.year, d.month, d.month_name
ORDER BY d.year, d.month;


-- 고객 세그먼트별 LTV (Life Time Value)
WITH customer_metrics AS (
    SELECT
        c.customer_sk,
        c.customer_segment,
        MIN(d.full_date) AS first_purchase_date,
        MAX(d.full_date) AS last_purchase_date,
        COUNT(DISTINCT f.order_id) AS total_orders,
        SUM(f.sales_amount) AS total_revenue
    FROM fact_sales f
    JOIN dim_customer c ON f.customer_sk = c.customer_sk
    JOIN dim_date d ON f.date_sk = d.date_sk
    GROUP BY c.customer_sk, c.customer_segment
)
SELECT
    customer_segment,
    COUNT(*) AS customer_count,
    AVG(total_orders) AS avg_orders,
    AVG(total_revenue) AS avg_ltv,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_revenue) AS median_ltv
FROM customer_metrics
GROUP BY customer_segment
ORDER BY avg_ltv DESC;
```

---

## 2. Data Lake

### 2.1 개념

```
┌────────────────────────────────────────────────────────────────┐
│                      Data Lake                                 │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   특징:                                                        │
│   - 모든 형태의 데이터 (구조화, 반구조화, 비구조화)               │
│   - Schema-on-Read (읽기 시 스키마 적용)                        │
│   - 원본 데이터 보존                                           │
│   - 저비용 스토리지                                            │
│                                                                │
│   ┌──────────────────────────────────────────────────────┐    │
│   │                     Data Lake                         │    │
│   │  ┌────────────────────────────────────────────────┐  │    │
│   │  │  Raw Zone (Bronze)                              │  │    │
│   │  │  - 원본 데이터 (JSON, CSV, Logs, Images)        │  │    │
│   │  └────────────────────────────────────────────────┘  │    │
│   │                         ↓                             │    │
│   │  ┌────────────────────────────────────────────────┐  │    │
│   │  │  Processed Zone (Silver)                        │  │    │
│   │  │  - 정제된 데이터 (Parquet, Delta)               │  │    │
│   │  └────────────────────────────────────────────────┘  │    │
│   │                         ↓                             │    │
│   │  ┌────────────────────────────────────────────────┐  │    │
│   │  │  Curated Zone (Gold)                            │  │    │
│   │  │  - 분석/ML 준비 데이터                          │  │    │
│   │  └────────────────────────────────────────────────┘  │    │
│   └──────────────────────────────────────────────────────┘    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 주요 스토리지

| 스토리지 | 클라우드 | 특징 |
|----------|----------|------|
| **S3** | AWS | 객체 스토리지, 높은 내구성 |
| **GCS** | GCP | Google Cloud Storage |
| **ADLS** | Azure | Azure Data Lake Storage |
| **HDFS** | 온프레미스 | Hadoop Distributed File System |

### 2.3 Data Lake 파일 구조

```
s3://my-data-lake/
├── raw/                          # Bronze 레이어
│   ├── orders/
│   │   ├── year=2024/
│   │   │   ├── month=01/
│   │   │   │   ├── day=15/
│   │   │   │   │   ├── orders_20240115_001.json
│   │   │   │   │   └── orders_20240115_002.json
│   ├── customers/
│   │   └── snapshot_20240115.csv
│   └── logs/
│       └── app_logs_20240115.log
│
├── processed/                    # Silver 레이어
│   ├── orders/
│   │   └── year=2024/
│   │       └── month=01/
│   │           └── part-00000.parquet
│   └── customers/
│       └── part-00000.parquet
│
└── curated/                      # Gold 레이어
    ├── fact_sales/
    │   └── year=2024/
    │       └── month=01/
    └── dim_customers/
        └── current/
```

```python
# PySpark로 Data Lake 계층 처리
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder \
    .appName("DataLakeProcessing") \
    .getOrCreate()

# Raw → Processed (Bronze → Silver)
def process_raw_orders():
    # Raw JSON 읽기
    raw_df = spark.read.json("s3://my-data-lake/raw/orders/")

    # 정제
    processed_df = raw_df \
        .filter(col("order_id").isNotNull()) \
        .withColumn("processed_at", current_timestamp()) \
        .dropDuplicates(["order_id"])

    # Parquet으로 저장
    processed_df.write \
        .mode("overwrite") \
        .partitionBy("year", "month") \
        .parquet("s3://my-data-lake/processed/orders/")


# Processed → Curated (Silver → Gold)
def create_fact_sales():
    orders = spark.read.parquet("s3://my-data-lake/processed/orders/")
    customers = spark.read.parquet("s3://my-data-lake/processed/customers/")

    fact_sales = orders \
        .join(customers, "customer_id") \
        .select(
            col("order_id"),
            col("customer_sk"),
            col("order_date"),
            col("amount").alias("sales_amount")
        )

    fact_sales.write \
        .mode("overwrite") \
        .partitionBy("year", "month") \
        .parquet("s3://my-data-lake/curated/fact_sales/")
```

---

## 3. Data Warehouse vs Data Lake

### 3.1 비교

| 특성 | Data Warehouse | Data Lake |
|------|----------------|-----------|
| **데이터 유형** | 구조화 | 모든 유형 |
| **스키마** | Schema-on-Write | Schema-on-Read |
| **사용자** | 비즈니스 분석가 | 데이터 과학자, 엔지니어 |
| **처리** | OLAP | 배치, 스트리밍, ML |
| **비용** | 높음 | 낮음 |
| **쿼리 성능** | 최적화됨 | 가변적 |
| **데이터 품질** | 높음 (정제됨) | 가변적 |

### 3.2 선택 기준

```python
def choose_architecture(requirements: dict) -> str:
    """아키텍처 선택 가이드"""

    warehouse_factors = [
        requirements.get('structured_data_only', False),
        requirements.get('sql_analytics_primary', False),
        requirements.get('strict_governance', False),
        requirements.get('fast_query_response', False),
    ]

    lake_factors = [
        requirements.get('unstructured_data', False),
        requirements.get('ml_workloads', False),
        requirements.get('raw_data_preservation', False),
        requirements.get('cost_sensitive', False),
        requirements.get('schema_flexibility', False),
    ]

    if sum(warehouse_factors) > sum(lake_factors):
        return "Data Warehouse 권장"
    elif sum(lake_factors) > sum(warehouse_factors):
        return "Data Lake 권장"
    else:
        return "Lakehouse 고려"
```

---

## 4. Lakehouse

### 4.1 개념

Lakehouse는 Data Lake의 유연성과 Data Warehouse의 성능/관리 기능을 결합한 아키텍처입니다.

```
┌────────────────────────────────────────────────────────────────┐
│                      Lakehouse Architecture                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   ┌────────────────────────────────────────────────────────┐  │
│   │                   Applications                          │  │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │  │
│   │  │    BI    │ │    ML    │ │  SQL     │ │ Streaming│  │  │
│   │  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │  │
│   └────────────────────────────────────────────────────────┘  │
│                              ↓                                 │
│   ┌────────────────────────────────────────────────────────┐  │
│   │                  Query Engine                           │  │
│   │        (Spark, Presto, Trino, Dremio)                  │  │
│   └────────────────────────────────────────────────────────┘  │
│                              ↓                                 │
│   ┌────────────────────────────────────────────────────────┐  │
│   │              Lakehouse Format Layer                     │  │
│   │     ┌──────────────────────────────────────────────┐   │  │
│   │     │  ACID Transactions │ Schema Enforcement      │   │  │
│   │     │  Time Travel       │ Unified Batch/Streaming │   │  │
│   │     └──────────────────────────────────────────────┘   │  │
│   │           Delta Lake / Apache Iceberg / Apache Hudi    │  │
│   └────────────────────────────────────────────────────────┘  │
│                              ↓                                 │
│   ┌────────────────────────────────────────────────────────┐  │
│   │              Object Storage (Data Lake)                 │  │
│   │                  S3 / GCS / ADLS / HDFS                 │  │
│   └────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 4.2 핵심 기능

| 기능 | 설명 |
|------|------|
| **ACID 트랜잭션** | 데이터 무결성 보장 |
| **스키마 진화** | 스키마 변경 지원 |
| **타임 트래블** | 과거 데이터 버전 조회 |
| **Upsert/Merge** | 효율적인 데이터 갱신 |
| **통합 처리** | 배치 + 스트리밍 단일 테이블 |

---

## 5. Delta Lake

### 5.1 Delta Lake 기본

```python
from pyspark.sql import SparkSession
from delta import *

# Delta Lake 설정
spark = SparkSession.builder \
    .appName("DeltaLake") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# Delta 테이블 생성
df = spark.createDataFrame([
    (1, "Alice", 100),
    (2, "Bob", 200),
], ["id", "name", "amount"])

df.write.format("delta").save("/data/delta/users")

# 읽기
delta_df = spark.read.format("delta").load("/data/delta/users")

# SQL로 접근
spark.sql("CREATE TABLE users USING DELTA LOCATION '/data/delta/users'")
spark.sql("SELECT * FROM users").show()
```

### 5.2 Delta Lake 고급 기능

```python
from delta.tables import DeltaTable

# MERGE (Upsert)
delta_table = DeltaTable.forPath(spark, "/data/delta/users")

new_data = spark.createDataFrame([
    (1, "Alice Updated", 150),  # 업데이트
    (3, "Charlie", 300),        # 삽입
], ["id", "name", "amount"])

delta_table.alias("target").merge(
    new_data.alias("source"),
    "target.id = source.id"
).whenMatchedUpdate(set={
    "name": "source.name",
    "amount": "source.amount"
}).whenNotMatchedInsert(values={
    "id": "source.id",
    "name": "source.name",
    "amount": "source.amount"
}).execute()


# Time Travel (과거 버전 조회)
# 버전 번호로
df_v0 = spark.read.format("delta") \
    .option("versionAsOf", 0) \
    .load("/data/delta/users")

# 타임스탬프로
df_yesterday = spark.read.format("delta") \
    .option("timestampAsOf", "2024-01-14") \
    .load("/data/delta/users")


# 히스토리 확인
delta_table.history().show()


# Vacuum (오래된 파일 정리)
delta_table.vacuum(retentionHours=168)  # 7일 보존


# 스키마 진화
spark.read.format("delta") \
    .option("mergeSchema", "true") \
    .load("/data/delta/users")


# Z-Order 최적화 (쿼리 성능)
delta_table.optimize().executeZOrderBy("date", "customer_id")
```

---

## 6. Apache Iceberg

### 6.1 Iceberg 기본

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Iceberg") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.iceberg.type", "hive") \
    .config("spark.sql.catalog.iceberg.uri", "thrift://localhost:9083") \
    .getOrCreate()

# Iceberg 테이블 생성
spark.sql("""
    CREATE TABLE iceberg.db.users (
        id INT,
        name STRING,
        amount DECIMAL(10, 2)
    ) USING ICEBERG
    PARTITIONED BY (bucket(16, id))
""")

# 데이터 삽입
spark.sql("""
    INSERT INTO iceberg.db.users VALUES
    (1, 'Alice', 100.00),
    (2, 'Bob', 200.00)
""")

# Time Travel
spark.sql("SELECT * FROM iceberg.db.users VERSION AS OF 1").show()
spark.sql("SELECT * FROM iceberg.db.users TIMESTAMP AS OF '2024-01-15'").show()

# 스냅샷 확인
spark.sql("SELECT * FROM iceberg.db.users.snapshots").show()
```

### 6.2 Delta Lake vs Iceberg 비교

| 특성 | Delta Lake | Iceberg |
|------|------------|---------|
| **개발사** | Databricks | Netflix → Apache |
| **호환성** | Spark 중심 | 엔진 독립적 |
| **메타데이터** | 트랜잭션 로그 | 스냅샷 기반 |
| **파티션 진화** | 제한적 | 강력한 지원 |
| **숨겨진 파티션** | 미지원 | 지원 |
| **커뮤니티** | Databricks 생태계 | 다양한 벤더 |

---

## 7. 모던 데이터 스택

### 7.1 아키텍처 패턴

```
┌─────────────────────────────────────────────────────────────────┐
│                   Modern Data Stack                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Data Sources                                                  │
│   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                  │
│   │ SaaS   │ │Database│ │  API   │ │  IoT   │                  │
│   └────┬───┘ └───┬────┘ └───┬────┘ └───┬────┘                  │
│        └─────────┴──────────┴──────────┘                        │
│                         ↓                                       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              Ingestion (EL)                              │  │
│   │        Fivetran / Airbyte / Stitch                       │  │
│   └─────────────────────────────────────────────────────────┘  │
│                         ↓                                       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │           Cloud Data Warehouse / Lakehouse              │  │
│   │        Snowflake / BigQuery / Databricks                │  │
│   └─────────────────────────────────────────────────────────┘  │
│                         ↓                                       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              Transformation (T)                          │  │
│   │                      dbt                                 │  │
│   └─────────────────────────────────────────────────────────┘  │
│                         ↓                                       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                 BI / Analytics                           │  │
│   │        Looker / Tableau / Metabase / Mode               │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 연습 문제

### 문제 1: 아키텍처 선택
다음 요구사항에 맞는 아키텍처를 선택하고 이유를 설명하세요:
- 일일 10TB의 로그 데이터
- ML 모델 학습에 사용
- 원본 데이터 5년 보존 필요

### 문제 2: Delta Lake 구현
고객 데이터에 대한 SCD Type 2를 Delta Lake MERGE로 구현하세요.

---

## 요약

| 아키텍처 | 특징 | 사용 사례 |
|----------|------|----------|
| **Data Warehouse** | 구조화, SQL 최적화 | BI, 리포팅 |
| **Data Lake** | 모든 데이터, 저비용 | ML, 원본 보존 |
| **Lakehouse** | Lake + Warehouse 장점 | 통합 분석 |

---

## 참고 자료

- [Delta Lake Documentation](https://docs.delta.io/)
- [Apache Iceberg Documentation](https://iceberg.apache.org/)
- [Databricks Lakehouse](https://www.databricks.com/product/data-lakehouse)
