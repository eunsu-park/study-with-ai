# PySpark DataFrame

## 개요

Spark DataFrame은 분산된 데이터를 테이블 형태로 표현하는 고수준 API입니다. SQL과 유사한 연산을 제공하며, Catalyst 옵티마이저를 통해 자동으로 최적화됩니다.

---

## 1. SparkSession과 DataFrame 생성

### 1.1 SparkSession 초기화

```python
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType

# SparkSession 생성
spark = SparkSession.builder \
    .appName("PySpark DataFrame Tutorial") \
    .config("spark.sql.shuffle.partitions", 100) \
    .config("spark.sql.adaptive.enabled", True) \
    .getOrCreate()

# Spark 버전 확인
print(f"Spark Version: {spark.version}")
```

### 1.2 DataFrame 생성 방법

```python
# 방법 1: Python 리스트에서 생성
data = [
    ("Alice", 30, "Engineering"),
    ("Bob", 25, "Marketing"),
    ("Charlie", 35, "Engineering"),
]
df1 = spark.createDataFrame(data, ["name", "age", "department"])

# 방법 2: 스키마 명시
schema = StructType([
    StructField("name", StringType(), nullable=False),
    StructField("age", IntegerType(), nullable=True),
    StructField("department", StringType(), nullable=True),
])
df2 = spark.createDataFrame(data, schema)

# 방법 3: 딕셔너리 리스트에서 생성
dict_data = [
    {"name": "Alice", "age": 30, "department": "Engineering"},
    {"name": "Bob", "age": 25, "department": "Marketing"},
]
df3 = spark.createDataFrame(dict_data)

# 방법 4: Pandas DataFrame에서 생성
import pandas as pd
pdf = pd.DataFrame(data, columns=["name", "age", "department"])
df4 = spark.createDataFrame(pdf)

# 방법 5: RDD에서 생성
rdd = spark.sparkContext.parallelize(data)
df5 = rdd.toDF(["name", "age", "department"])
```

### 1.3 파일에서 DataFrame 읽기

```python
# CSV 파일
df_csv = spark.read.csv(
    "data.csv",
    header=True,           # 첫 행을 헤더로
    inferSchema=True,      # 스키마 자동 추론
    sep=",",               # 구분자
    nullValue="NA",        # NULL 표현
    dateFormat="yyyy-MM-dd"
)

# 스키마 명시 (권장 - 성능 향상)
schema = StructType([
    StructField("id", IntegerType()),
    StructField("name", StringType()),
    StructField("amount", DoubleType()),
    StructField("date", DateType()),
])
df_csv = spark.read.csv("data.csv", header=True, schema=schema)

# Parquet 파일 (권장 - 컬럼 형식)
df_parquet = spark.read.parquet("data.parquet")

# JSON 파일
df_json = spark.read.json("data.json")

# ORC 파일
df_orc = spark.read.orc("data.orc")

# JDBC (데이터베이스)
df_jdbc = spark.read.format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/mydb") \
    .option("dbtable", "public.users") \
    .option("user", "user") \
    .option("password", "password") \
    .option("driver", "org.postgresql.Driver") \
    .load()

# Delta Lake
df_delta = spark.read.format("delta").load("path/to/delta")
```

---

## 2. DataFrame 기본 연산

### 2.1 데이터 확인

```python
# 데이터 미리보기
df.show()           # 상위 20행
df.show(5)          # 상위 5행
df.show(truncate=False)  # 컬럼 잘림 없이

# 스키마 확인
df.printSchema()
df.dtypes           # [(컬럼명, 타입), ...]
df.columns          # 컬럼 목록

# 통계 정보
df.describe().show()        # 기술 통계
df.summary().show()         # 확장 통계

# 레코드 수
df.count()

# 유니크 값 수
df.select("department").distinct().count()

# 첫 번째 행
df.first()
df.head(5)

# Pandas로 변환 (작은 데이터셋만)
pdf = df.toPandas()
```

### 2.2 컬럼 선택

```python
from pyspark.sql.functions import col, lit

# 단일 컬럼
df.select("name")
df.select(col("name"))
df.select(df.name)
df.select(df["name"])

# 여러 컬럼
df.select("name", "age")
df.select(["name", "age"])
df.select(col("name"), col("age"))

# 모든 컬럼 + 추가 컬럼
df.select("*", lit(1).alias("constant"))

# 컬럼 제외
df.drop("department")

# 컬럼 이름 변경
df.withColumnRenamed("name", "full_name")

# 여러 컬럼 이름 변경
df.toDF("name_new", "age_new", "dept_new")

# alias 사용
df.select(col("name").alias("employee_name"))
```

### 2.3 필터링

```python
from pyspark.sql.functions import col

# 기본 필터
df.filter(col("age") > 30)
df.filter(df.age > 30)
df.filter("age > 30")           # SQL 표현식
df.where(col("age") > 30)       # filter와 동일

# 복합 조건
df.filter((col("age") > 25) & (col("department") == "Engineering"))
df.filter((col("age") < 25) | (col("department") == "Marketing"))
df.filter(~(col("age") > 30))   # NOT

# 문자열 필터
df.filter(col("name").startswith("A"))
df.filter(col("name").endswith("e"))
df.filter(col("name").contains("li"))
df.filter(col("name").like("%li%"))
df.filter(col("name").rlike("^[A-C].*"))  # 정규식

# IN 조건
df.filter(col("department").isin(["Engineering", "Marketing"]))

# NULL 처리
df.filter(col("age").isNull())
df.filter(col("age").isNotNull())

# BETWEEN
df.filter(col("age").between(25, 35))
```

---

## 3. 변환 (Transformations)

### 3.1 컬럼 추가/수정

```python
from pyspark.sql.functions import col, lit, when, concat, upper, lower, length

# 새 컬럼 추가
df.withColumn("bonus", col("salary") * 0.1)

# 상수 컬럼
df.withColumn("country", lit("USA"))

# 기존 컬럼 수정
df.withColumn("name", upper(col("name")))

# 조건부 컬럼 (CASE WHEN)
df.withColumn("age_group",
    when(col("age") < 30, "Young")
    .when(col("age") < 50, "Middle")
    .otherwise("Senior")
)

# 여러 컬럼 동시에
df.withColumns({
    "name_upper": upper(col("name")),
    "age_plus_10": col("age") + 10,
})

# 문자열 결합
df.withColumn("full_info", concat(col("name"), lit(" - "), col("department")))

# 타입 캐스팅
df.withColumn("age_double", col("age").cast("double"))
df.withColumn("age_string", col("age").cast(StringType()))
```

### 3.2 집계 연산

```python
from pyspark.sql.functions import (
    count, sum as _sum, avg, min as _min, max as _max,
    countDistinct, collect_list, collect_set,
    first, last, stddev, variance
)

# 전체 집계
df.agg(
    count("*").alias("total_count"),
    _sum("salary").alias("total_salary"),
    avg("salary").alias("avg_salary"),
    _min("salary").alias("min_salary"),
    _max("salary").alias("max_salary"),
).show()

# 그룹별 집계
df.groupBy("department").agg(
    count("*").alias("employee_count"),
    avg("salary").alias("avg_salary"),
    _sum("salary").alias("total_salary"),
    countDistinct("name").alias("unique_names"),
)

# 여러 컬럼 그룹화
df.groupBy("department", "age_group").count()

# 리스트/집합 집계
df.groupBy("department").agg(
    collect_list("name").alias("employee_names"),
    collect_set("age").alias("unique_ages"),
)

# 피벗 테이블
df.groupBy("department") \
    .pivot("age_group", ["Young", "Middle", "Senior"]) \
    .agg(count("*"))
```

### 3.3 정렬

```python
from pyspark.sql.functions import col, asc, desc

# 단일 컬럼 정렬
df.orderBy("age")                    # 오름차순 (기본)
df.orderBy(col("age").desc())        # 내림차순
df.orderBy(desc("age"))

# 여러 컬럼 정렬
df.orderBy(["department", "age"])
df.orderBy(col("department").asc(), col("age").desc())

# NULL 처리
df.orderBy(col("age").asc_nulls_first())
df.orderBy(col("age").desc_nulls_last())

# sort는 orderBy와 동일
df.sort("age")
```

### 3.4 조인

```python
# 테스트 데이터
employees = spark.createDataFrame([
    (1, "Alice", 101),
    (2, "Bob", 102),
    (3, "Charlie", 101),
], ["id", "name", "dept_id"])

departments = spark.createDataFrame([
    (101, "Engineering"),
    (102, "Marketing"),
    (103, "Finance"),
], ["dept_id", "dept_name"])

# Inner Join (기본)
employees.join(departments, employees.dept_id == departments.dept_id)
employees.join(departments, "dept_id")  # 동일 컬럼명

# Left Join
employees.join(departments, "dept_id", "left")

# Right Join
employees.join(departments, "dept_id", "right")

# Full Outer Join
employees.join(departments, "dept_id", "full")

# Cross Join (Cartesian)
employees.crossJoin(departments)

# Semi Join (왼쪽 테이블만, 조건 충족)
employees.join(departments, "dept_id", "left_semi")

# Anti Join (왼쪽 테이블만, 조건 미충족)
employees.join(departments, "dept_id", "left_anti")

# 복합 조건 조인
employees.join(
    departments,
    (employees.dept_id == departments.dept_id) & (employees.id > 1),
    "inner"
)
```

---

## 4. 액션 (Actions)

### 4.1 데이터 수집

```python
# Driver로 데이터 수집
result = df.collect()           # 전체 데이터 (주의: 메모리)
result = df.take(10)            # 상위 10개
result = df.first()             # 첫 번째 행
result = df.head(5)             # 상위 5개

# 리스트로 변환
ages = df.select("age").rdd.flatMap(lambda x: x).collect()

# Pandas DataFrame으로
pdf = df.toPandas()             # 작은 데이터만

# Iterator로 (대용량)
for row in df.toLocalIterator():
    print(row)
```

### 4.2 파일 저장

```python
# Parquet (권장)
df.write.parquet("output/data.parquet")

# 모드 지정
df.write.mode("overwrite").parquet("output/data.parquet")
# overwrite: 덮어쓰기
# append: 추가
# ignore: 존재하면 무시
# error: 존재하면 에러 (기본)

# 파티션 저장
df.write.partitionBy("date", "department").parquet("output/partitioned")

# CSV
df.write.csv("output/data.csv", header=True)

# JSON
df.write.json("output/data.json")

# 단일 파일로 저장
df.coalesce(1).write.csv("output/single_file.csv", header=True)

# JDBC (데이터베이스)
df.write.format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/mydb") \
    .option("dbtable", "public.output_table") \
    .option("user", "user") \
    .option("password", "password") \
    .mode("overwrite") \
    .save()
```

---

## 5. UDF (User Defined Functions)

### 5.1 기본 UDF

```python
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, IntegerType

# Python 함수 정의
def categorize_age(age):
    if age is None:
        return "Unknown"
    elif age < 30:
        return "Young"
    elif age < 50:
        return "Middle"
    else:
        return "Senior"

# UDF 등록 (데코레이터 방식)
@udf(returnType=StringType())
def categorize_age_udf(age):
    if age is None:
        return "Unknown"
    elif age < 30:
        return "Young"
    elif age < 50:
        return "Middle"
    else:
        return "Senior"

# UDF 등록 (함수 방식)
categorize_udf = udf(categorize_age, StringType())

# 사용
df.withColumn("age_category", categorize_udf(col("age")))
df.withColumn("age_category", categorize_age_udf(col("age")))
```

### 5.2 Pandas UDF (성능 향상)

```python
from pyspark.sql.functions import pandas_udf
import pandas as pd

# Scalar Pandas UDF (1:1 매핑)
@pandas_udf(StringType())
def categorize_pandas_udf(age_series: pd.Series) -> pd.Series:
    return age_series.apply(
        lambda x: "Unknown" if x is None
        else "Young" if x < 30
        else "Middle" if x < 50
        else "Senior"
    )

# 사용
df.withColumn("age_category", categorize_pandas_udf(col("age")))

# Grouped Pandas UDF (그룹별 처리)
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

result_schema = StructType([
    StructField("department", StringType()),
    StructField("avg_salary", DoubleType()),
    StructField("employee_count", IntegerType()),
])

@pandas_udf(result_schema, PandasUDFType.GROUPED_MAP)
def analyze_department(pdf: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "department": [pdf["department"].iloc[0]],
        "avg_salary": [pdf["salary"].mean()],
        "employee_count": [len(pdf)],
    })

# 사용
df.groupby("department").apply(analyze_department)
```

### 5.3 SQL에서 UDF 사용

```python
# SQL용 UDF 등록
spark.udf.register("categorize_age", categorize_age, StringType())

# SQL에서 사용
df.createOrReplaceTempView("employees")
spark.sql("""
    SELECT name, age, categorize_age(age) as age_category
    FROM employees
""").show()
```

---

## 6. 윈도우 함수

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    row_number, rank, dense_rank,
    lead, lag, sum as _sum, avg,
    first, last, ntile
)

# 윈도우 정의
window_dept = Window.partitionBy("department").orderBy("salary")
window_all = Window.orderBy("salary")

# 순위 함수
df.withColumn("row_num", row_number().over(window_dept))
df.withColumn("rank", rank().over(window_dept))
df.withColumn("dense_rank", dense_rank().over(window_dept))
df.withColumn("ntile_4", ntile(4).over(window_dept))

# 이전/다음 값
df.withColumn("prev_salary", lag("salary", 1).over(window_dept))
df.withColumn("next_salary", lead("salary", 1).over(window_dept))

# 누적 합계
window_cumsum = Window.partitionBy("department") \
    .orderBy("date") \
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)

df.withColumn("cumsum_salary", _sum("salary").over(window_cumsum))

# 이동 평균
window_moving = Window.partitionBy("department") \
    .orderBy("date") \
    .rowsBetween(-2, 0)  # 현재 + 이전 2개

df.withColumn("moving_avg", avg("salary").over(window_moving))

# 그룹 내 첫 번째/마지막 값
df.withColumn("first_name", first("name").over(window_dept))
df.withColumn("last_name", last("name").over(window_dept))
```

---

## 연습 문제

### 문제 1: 데이터 변환
판매 데이터에서 월별, 카테고리별 총 매출과 평균 매출을 계산하세요.

### 문제 2: 윈도우 함수
각 부서별로 급여 순위를 매기고, 부서 내 급여 상위 3명을 추출하세요.

### 문제 3: UDF 작성
이메일 주소에서 도메인을 추출하는 UDF를 작성하고 적용하세요.

---

## 요약

| 연산 | 설명 | 예시 |
|------|------|------|
| **select** | 컬럼 선택 | `df.select("name", "age")` |
| **filter** | 행 필터링 | `df.filter(col("age") > 30)` |
| **groupBy** | 그룹화 | `df.groupBy("dept").agg(...)` |
| **join** | 테이블 조인 | `df1.join(df2, "key")` |
| **orderBy** | 정렬 | `df.orderBy(desc("salary"))` |
| **withColumn** | 컬럼 추가/수정 | `df.withColumn("new", ...)` |

---

## 참고 자료

- [PySpark DataFrame Guide](https://spark.apache.org/docs/latest/sql-getting-started.html)
- [PySpark Functions](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html)
