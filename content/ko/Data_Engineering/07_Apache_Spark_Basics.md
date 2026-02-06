# Apache Spark 기초

## 개요

Apache Spark는 대규모 데이터 처리를 위한 통합 분석 엔진입니다. 인메모리 처리로 Hadoop MapReduce보다 빠른 성능을 제공하며, 배치 처리와 스트리밍을 모두 지원합니다.

---

## 1. Spark 개요

### 1.1 Spark의 특징

```
┌────────────────────────────────────────────────────────────────┐
│                    Apache Spark 특징                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   1. 속도 (Speed)                                              │
│      - 인메모리 처리로 Hadoop보다 100배 빠름                     │
│      - 디스크 기반보다 10배 빠름                                 │
│                                                                │
│   2. 사용 편의성 (Ease of Use)                                  │
│      - Python, Scala, Java, R 지원                             │
│      - SQL 인터페이스 제공                                      │
│                                                                │
│   3. 범용성 (Generality)                                       │
│      - SQL, 스트리밍, ML, 그래프 처리                           │
│      - 하나의 엔진으로 다양한 워크로드                           │
│                                                                │
│   4. 호환성 (Compatibility)                                     │
│      - HDFS, S3, Cassandra 등 다양한 데이터 소스                │
│      - YARN, Kubernetes, Standalone 클러스터                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 Spark 생태계

```
┌─────────────────────────────────────────────────────────────────┐
│                     Spark Ecosystem                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐  │
│   │  Spark SQL │ │ Streaming  │ │   MLlib    │ │  GraphX    │  │
│   │    + DF    │ │ (Structured)│ │(Machine   │ │  (Graph)   │  │
│   │            │ │             │ │ Learning) │ │            │  │
│   └────────────┘ └────────────┘ └────────────┘ └────────────┘  │
│   ─────────────────────────────────────────────────────────────│
│   │                     Spark Core                           │  │
│   │                 (RDD, Task Scheduling)                   │  │
│   ─────────────────────────────────────────────────────────────│
│   ─────────────────────────────────────────────────────────────│
│   │    Standalone    │    YARN    │    Kubernetes    │ Mesos │  │
│   ─────────────────────────────────────────────────────────────│
│   ─────────────────────────────────────────────────────────────│
│   │  HDFS  │   S3   │   GCS   │  Cassandra  │  JDBC  │ etc │  │
│   ─────────────────────────────────────────────────────────────│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Spark 아키텍처

### 2.1 클러스터 구성

```
┌─────────────────────────────────────────────────────────────────┐
│                    Spark Cluster Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌───────────────────────────────────────────────────────┐    │
│   │                    Driver Program                      │    │
│   │   ┌─────────────────────────────────────────────────┐ │    │
│   │   │              SparkContext                        │ │    │
│   │   │   - 애플리케이션 진입점                          │ │    │
│   │   │   - 클러스터와 연결                              │ │    │
│   │   │   - Job 생성 및 스케줄링                         │ │    │
│   │   └─────────────────────────────────────────────────┘ │    │
│   └───────────────────────────────────────────────────────┘    │
│                              ↓                                  │
│   ┌───────────────────────────────────────────────────────┐    │
│   │                  Cluster Manager                       │    │
│   │       (Standalone, YARN, Kubernetes, Mesos)            │    │
│   └───────────────────────────────────────────────────────┘    │
│                              ↓                                  │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│   │   Worker    │  │   Worker    │  │   Worker    │           │
│   │  ┌───────┐  │  │  ┌───────┐  │  │  ┌───────┐  │           │
│   │  │Executor│ │  │  │Executor│ │  │  │Executor│ │           │
│   │  │ Task  │  │  │  │ Task  │  │  │  │ Task  │  │           │
│   │  │ Task  │  │  │  │ Task  │  │  │  │ Task  │  │           │
│   │  │ Cache │  │  │  │ Cache │  │  │  │ Cache │  │           │
│   │  └───────┘  │  │  └───────┘  │  │  └───────┘  │           │
│   └─────────────┘  └─────────────┘  └─────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 핵심 개념

| 개념 | 설명 |
|------|------|
| **Driver** | 메인 프로그램 실행, SparkContext 생성 |
| **Executor** | Worker 노드에서 Task 실행 |
| **Task** | 실행의 기본 단위 |
| **Job** | Action에 의해 생성되는 병렬 계산 |
| **Stage** | Job 내의 Task 그룹 (Shuffle 경계) |
| **Partition** | 데이터의 논리적 분할 단위 |

### 2.3 실행 흐름

```python
"""
Spark 실행 흐름:
1. Driver에서 SparkContext 생성
2. 애플리케이션 코드 해석
3. Transformation → DAG (Directed Acyclic Graph) 생성
4. Action 호출 시 Job 생성
5. Job → Stages → Tasks로 분해
6. Cluster Manager가 Executor에 Task 할당
7. Executor에서 Task 실행
8. 결과를 Driver로 반환
"""

# 예시 코드 흐름
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Example").getOrCreate()

# Transformations (Lazy - 실행 안 됨)
df = spark.read.csv("data.csv", header=True)  # 읽기 계획
df2 = df.filter(df.age > 20)                  # 필터 계획
df3 = df2.groupBy("city").count()             # 집계 계획

# Action (실제 실행 트리거)
result = df3.collect()  # Job 생성 → Stages → Tasks → 실행
```

---

## 3. RDD (Resilient Distributed Dataset)

### 3.1 RDD 개념

RDD는 Spark의 기본 데이터 구조로, 분산된 불변 데이터 컬렉션입니다.

```python
from pyspark import SparkContext

sc = SparkContext("local[*]", "RDD Example")

# RDD 생성 방법
# 1. 컬렉션에서 생성
rdd1 = sc.parallelize([1, 2, 3, 4, 5])

# 2. 외부 데이터에서 생성
rdd2 = sc.textFile("data.txt")

# 3. 기존 RDD 변환
rdd3 = rdd1.map(lambda x: x * 2)

# RDD 특성
"""
R - Resilient: 장애 복구 가능 (Lineage로 재계산)
D - Distributed: 클러스터에 분산 저장
D - Dataset: 데이터 컬렉션
"""
```

### 3.2 RDD 연산

```python
# Transformations (Lazy)
# - 새로운 RDD 반환
# - 실행 계획만 생성

rdd = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# map: 각 요소에 함수 적용
mapped = rdd.map(lambda x: x * 2)  # [2, 4, 6, ...]

# filter: 조건에 맞는 요소만 선택
filtered = rdd.filter(lambda x: x % 2 == 0)  # [2, 4, 6, 8, 10]

# flatMap: map 후 flatten
flat = rdd.flatMap(lambda x: [x, x*2])  # [1, 2, 2, 4, 3, 6, ...]

# distinct: 중복 제거
distinct = rdd.distinct()

# union: 두 RDD 합치기
union = rdd.union(sc.parallelize([11, 12]))

# groupByKey: 키별 그룹화
pairs = sc.parallelize([("a", 1), ("b", 2), ("a", 3)])
grouped = pairs.groupByKey()  # [("a", [1, 3]), ("b", [2])]

# reduceByKey: 키별 리듀스
reduced = pairs.reduceByKey(lambda a, b: a + b)  # [("a", 4), ("b", 2)]


# Actions (Eager)
# - 결과 반환 또는 저장
# - 실제 실행 트리거

# collect: 모든 요소를 Driver로 반환
result = rdd.collect()  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# count: 요소 개수
count = rdd.count()  # 10

# first / take: 첫 번째 / n개 요소
first = rdd.first()  # 1
take3 = rdd.take(3)  # [1, 2, 3]

# reduce: 전체 리듀스
total = rdd.reduce(lambda a, b: a + b)  # 55

# foreach: 각 요소에 함수 적용 (부수 효과)
rdd.foreach(lambda x: print(x))

# saveAsTextFile: 파일로 저장
rdd.saveAsTextFile("output/")
```

### 3.3 Pair RDD 연산

```python
# Key-Value 쌍 RDD 연산
sales = sc.parallelize([
    ("Electronics", 100),
    ("Clothing", 50),
    ("Electronics", 200),
    ("Clothing", 75),
    ("Food", 30),
])

# 키별 합계
total_by_category = sales.reduceByKey(lambda a, b: a + b)
# [("Electronics", 300), ("Clothing", 125), ("Food", 30)]

# 키별 평균
count_sum = sales.combineByKey(
    lambda v: (v, 1),                      # createCombiner
    lambda acc, v: (acc[0] + v, acc[1] + 1),  # mergeValue
    lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])  # mergeCombiner
)
avg_by_category = count_sum.mapValues(lambda x: x[0] / x[1])

# 정렬
sorted_rdd = sales.sortByKey()

# Join
inventory = sc.parallelize([
    ("Electronics", 50),
    ("Clothing", 100),
])

joined = sales.join(inventory)
# [("Electronics", (100, 50)), ("Electronics", (200, 50)), ...]
```

---

## 4. 설치 및 실행

### 4.1 로컬 설치 (PySpark)

```bash
# pip 설치
pip install pyspark

# 버전 확인
pyspark --version

# PySpark 셸 시작
pyspark

# spark-submit으로 스크립트 실행
spark-submit my_script.py
```

### 4.2 Docker 설치

```yaml
# docker-compose.yaml
version: '3'

services:
  spark-master:
    image: bitnami/spark:3.4
    environment:
      - SPARK_MODE=master
    ports:
      - "8080:8080"
      - "7077:7077"

  spark-worker:
    image: bitnami/spark:3.4
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER_URL=spark://spark-master:7077
    depends_on:
      - spark-master
```

```bash
# 실행
docker-compose up -d

# 클러스터에 작업 제출
spark-submit --master spark://localhost:7077 my_script.py
```

### 4.3 클러스터 모드

```bash
# Standalone 클러스터
spark-submit \
    --master spark://master:7077 \
    --deploy-mode cluster \
    --executor-memory 4G \
    --executor-cores 2 \
    --num-executors 10 \
    my_script.py

# YARN 클러스터
spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --executor-memory 4G \
    my_script.py

# Kubernetes 클러스터
spark-submit \
    --master k8s://https://k8s-master:6443 \
    --deploy-mode cluster \
    --conf spark.kubernetes.container.image=my-spark-image \
    my_script.py
```

---

## 5. SparkSession

### 5.1 SparkSession 생성

```python
from pyspark.sql import SparkSession

# 기본 SparkSession
spark = SparkSession.builder \
    .appName("My Application") \
    .getOrCreate()

# 설정 포함
spark = SparkSession.builder \
    .appName("My Application") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", 200) \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.adaptive.enabled", "true") \
    .enableHiveSupport() \
    .getOrCreate()

# SparkContext 접근
sc = spark.sparkContext

# 설정 확인
print(spark.conf.get("spark.sql.shuffle.partitions"))

# 세션 종료
spark.stop()
```

### 5.2 주요 설정

```python
# 자주 사용하는 설정
common_configs = {
    # 메모리 설정
    "spark.executor.memory": "4g",
    "spark.driver.memory": "2g",
    "spark.executor.memoryOverhead": "512m",

    # 병렬성 설정
    "spark.executor.cores": "4",
    "spark.default.parallelism": "100",
    "spark.sql.shuffle.partitions": "200",

    # 직렬화 설정
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",

    # Adaptive Query Execution (Spark 3.0+)
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",

    # 캐시 설정
    "spark.storage.memoryFraction": "0.6",

    # 셔플 설정
    "spark.shuffle.compress": "true",
}

# 설정 적용 예시
spark = SparkSession.builder \
    .config("spark.sql.shuffle.partitions", 100) \
    .config("spark.sql.adaptive.enabled", True) \
    .getOrCreate()
```

---

## 6. 기본 예제

### 6.1 Word Count

```python
from pyspark.sql import SparkSession

# SparkSession 생성
spark = SparkSession.builder \
    .appName("Word Count") \
    .getOrCreate()

sc = spark.sparkContext

# 텍스트 파일 읽기
text_rdd = sc.textFile("input.txt")

# Word Count 로직
word_counts = text_rdd \
    .flatMap(lambda line: line.split()) \
    .map(lambda word: (word.lower(), 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .sortBy(lambda x: x[1], ascending=False)

# 결과 출력
for word, count in word_counts.take(10):
    print(f"{word}: {count}")

# 파일로 저장
word_counts.saveAsTextFile("output/word_counts")

spark.stop()
```

### 6.2 DataFrame 기본

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, avg

spark = SparkSession.builder.appName("DataFrame Example").getOrCreate()

# DataFrame 생성
data = [
    ("Alice", "Engineering", 50000),
    ("Bob", "Engineering", 60000),
    ("Charlie", "Marketing", 45000),
    ("Diana", "Marketing", 55000),
]

df = spark.createDataFrame(data, ["name", "department", "salary"])

# 기본 연산
df.show()
df.printSchema()

# 필터링
df.filter(col("salary") > 50000).show()

# 집계
df.groupBy("department") \
    .agg(
        _sum("salary").alias("total_salary"),
        avg("salary").alias("avg_salary")
    ) \
    .show()

# SQL 사용
df.createOrReplaceTempView("employees")
spark.sql("""
    SELECT department, AVG(salary) as avg_salary
    FROM employees
    GROUP BY department
""").show()

spark.stop()
```

---

## 연습 문제

### 문제 1: RDD 기본 연산
1부터 100까지의 숫자 중 짝수만 선택하여 제곱의 합을 구하세요.

```python
# 풀이
sc = spark.sparkContext
result = sc.parallelize(range(1, 101)) \
    .filter(lambda x: x % 2 == 0) \
    .map(lambda x: x ** 2) \
    .reduce(lambda a, b: a + b)
print(result)  # 171700
```

### 문제 2: Pair RDD
로그 파일에서 에러 수준별 로그 수를 집계하세요.

```python
# 입력: "2024-01-01 ERROR: Connection failed"
logs = sc.textFile("logs.txt")
error_counts = logs \
    .map(lambda line: line.split()[1].replace(":", "")) \
    .map(lambda level: (level, 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .collect()
```

---

## 요약

| 개념 | 설명 |
|------|------|
| **Spark** | 대규모 데이터 처리 통합 엔진 |
| **RDD** | 기본 분산 데이터 구조 |
| **Transformation** | 새 RDD 생성 (Lazy) |
| **Action** | 결과 반환 (Eager) |
| **Driver** | 메인 프로그램 실행 노드 |
| **Executor** | Task 실행 워커 |

---

## 참고 자료

- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [PySpark API Reference](https://spark.apache.org/docs/latest/api/python/)
- [Learning Spark (O'Reilly)](https://www.oreilly.com/library/view/learning-spark-2nd/9781492050032/)
