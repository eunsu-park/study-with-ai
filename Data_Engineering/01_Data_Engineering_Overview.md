# 데이터 엔지니어링 개요

## 개요

데이터 엔지니어링은 조직의 데이터를 수집, 저장, 처리, 전달하는 시스템을 설계하고 구축하는 분야입니다. 데이터 엔지니어는 데이터 파이프라인을 구축하여 원시 데이터를 분석 가능한 형태로 변환합니다.

---

## 1. 데이터 엔지니어의 역할

### 1.1 핵심 책임

```
┌─────────────────────────────────────────────────────────────┐
│                    데이터 엔지니어 역할                        │
├─────────────────────────────────────────────────────────────┤
│  1. 데이터 수집 (Ingestion)                                   │
│     - 다양한 소스에서 데이터 추출                               │
│     - API, 데이터베이스, 파일, 스트리밍                         │
│                                                             │
│  2. 데이터 저장 (Storage)                                     │
│     - Data Lake, Data Warehouse 설계                         │
│     - 스키마 설계 및 최적화                                    │
│                                                             │
│  3. 데이터 변환 (Transformation)                              │
│     - ETL/ELT 파이프라인 구축                                 │
│     - 데이터 품질 보장                                        │
│                                                             │
│  4. 데이터 전달 (Serving)                                     │
│     - 분석가/과학자에게 데이터 제공                            │
│     - BI 도구, API, 대시보드 연동                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 데이터 엔지니어 vs 데이터 과학자 vs 데이터 분석가

| 역할 | 주요 업무 | 필요 기술 |
|------|----------|----------|
| **데이터 엔지니어** | 파이프라인 구축, 인프라 관리 | Python, SQL, Spark, Airflow, Kafka |
| **데이터 과학자** | 모델 개발, 예측 분석 | Python, ML/DL, 통계, 수학 |
| **데이터 분석가** | 비즈니스 인사이트 도출 | SQL, BI 도구, 시각화, 통계 |

### 1.3 데이터 엔지니어 필수 기술

```python
# 데이터 엔지니어 기술 스택 예시
tech_stack = {
    "programming": ["Python", "SQL", "Scala", "Java"],
    "databases": ["PostgreSQL", "MySQL", "MongoDB", "Redis"],
    "big_data": ["Spark", "Hadoop", "Flink", "Hive"],
    "orchestration": ["Airflow", "Prefect", "Dagster"],
    "streaming": ["Kafka", "Kinesis", "Pub/Sub"],
    "cloud": ["AWS", "GCP", "Azure"],
    "infrastructure": ["Docker", "Kubernetes", "Terraform"],
    "storage": ["S3", "GCS", "HDFS", "Delta Lake"]
}
```

---

## 2. 데이터 파이프라인 개념

### 2.1 파이프라인이란?

데이터 파이프라인은 데이터를 소스에서 목적지까지 이동시키는 일련의 처리 단계입니다.

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Source  │ → │  Extract │ → │Transform │ → │   Load   │
│ (소스)   │    │  (추출)  │    │ (변환)   │    │ (적재)   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
     ↓               ↓               ↓               ↓
  Database        Raw Data      Cleaned Data    Warehouse
  API, Files      Staging       Processed       Analytics
```

### 2.2 파이프라인 구성 요소

```python
# 간단한 파이프라인 예시
from datetime import datetime
import pandas as pd

class DataPipeline:
    """기본 데이터 파이프라인 클래스"""

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None

    def extract(self, source: str) -> pd.DataFrame:
        """데이터 추출 단계"""
        print(f"[{datetime.now()}] Extracting from {source}")
        # 실제로는 DB, API, 파일 등에서 데이터 추출
        data = pd.read_csv(source)
        return data

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 변환 단계"""
        print(f"[{datetime.now()}] Transforming data")
        # 데이터 정제, 변환, 집계 등
        df = df.dropna()  # 결측치 제거
        df['processed_at'] = datetime.now()
        return df

    def load(self, df: pd.DataFrame, destination: str):
        """데이터 적재 단계"""
        print(f"[{datetime.now()}] Loading to {destination}")
        # 실제로는 DB, 파일, 클라우드 스토리지 등에 저장
        df.to_parquet(destination, index=False)

    def run(self, source: str, destination: str):
        """전체 파이프라인 실행"""
        self.start_time = datetime.now()
        print(f"Pipeline '{self.name}' started")

        # ETL 프로세스
        raw_data = self.extract(source)
        transformed_data = self.transform(raw_data)
        self.load(transformed_data, destination)

        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).seconds
        print(f"Pipeline completed in {duration} seconds")


# 파이프라인 실행
if __name__ == "__main__":
    pipeline = DataPipeline("daily_sales")
    pipeline.run("sales_raw.csv", "sales_processed.parquet")
```

### 2.3 파이프라인 유형

| 유형 | 설명 | 사용 사례 |
|------|------|----------|
| **배치 (Batch)** | 정해진 시간에 대량 데이터 처리 | 일일 보고서, 월간 집계 |
| **스트리밍 (Streaming)** | 실시간 데이터 처리 | 실시간 대시보드, 이상 탐지 |
| **마이크로배치 (Micro-batch)** | 짧은 간격의 작은 배치 | 준실시간 분석 (5-15분) |
| **이벤트 기반 (Event-driven)** | 특정 이벤트 발생 시 처리 | 트리거 기반 처리 |

---

## 3. 배치 처리 vs 스트리밍 처리

### 3.1 배치 처리 (Batch Processing)

```python
# 배치 처리 예시: 일일 매출 집계
from datetime import datetime, timedelta
import pandas as pd

def daily_sales_batch():
    """일일 매출 배치 처리"""

    # 1. 어제 날짜의 데이터 추출
    yesterday = datetime.now() - timedelta(days=1)
    date_str = yesterday.strftime('%Y-%m-%d')

    # 2. 데이터 추출 (시뮬레이션)
    query = f"""
    SELECT
        product_id,
        SUM(quantity) as total_quantity,
        SUM(amount) as total_amount
    FROM sales
    WHERE DATE(created_at) = '{date_str}'
    GROUP BY product_id
    """

    # 3. 집계 결과 저장
    print(f"Processing batch for {date_str}")
    # df = execute_query(query)
    # df.to_parquet(f"sales_summary_{date_str}.parquet")

    return {"status": "success", "date": date_str}

# 배치 처리 특징
batch_characteristics = {
    "latency": "높음 (분~시간)",
    "throughput": "높음 (대량 처리에 효율적)",
    "use_cases": ["일일 보고서", "주간 집계", "데이터 마이그레이션"],
    "tools": ["Spark", "Airflow", "dbt", "AWS Glue"]
}
```

### 3.2 스트리밍 처리 (Stream Processing)

```python
# 스트리밍 처리 예시: 실시간 이벤트 처리
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Any
import json

@dataclass
class Event:
    """스트리밍 이벤트"""
    event_type: str
    data: dict
    timestamp: datetime

class StreamProcessor:
    """간단한 스트림 프로세서"""

    def __init__(self):
        self.handlers: dict[str, list[Callable]] = {}

    def register_handler(self, event_type: str, handler: Callable):
        """이벤트 핸들러 등록"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    def process(self, event: Event):
        """이벤트 처리"""
        handlers = self.handlers.get(event.event_type, [])
        for handler in handlers:
            handler(event)

    def consume(self, stream):
        """스트림에서 이벤트 소비 (시뮬레이션)"""
        for message in stream:
            event = Event(
                event_type=message['type'],
                data=message['data'],
                timestamp=datetime.now()
            )
            self.process(event)


# 핸들러 예시
def log_handler(event: Event):
    """이벤트 로깅"""
    print(f"[{event.timestamp}] {event.event_type}: {event.data}")

def alert_handler(event: Event):
    """이상 탐지 알림"""
    if event.data.get('amount', 0) > 10000:
        print(f"ALERT: High value transaction detected!")

# 스트리밍 특징
streaming_characteristics = {
    "latency": "낮음 (밀리초~초)",
    "throughput": "중간 (레코드 단위)",
    "use_cases": ["실시간 대시보드", "이상 탐지", "알림"],
    "tools": ["Kafka", "Flink", "Spark Streaming", "Kinesis"]
}
```

### 3.3 배치 vs 스트리밍 비교

| 특성 | 배치 처리 | 스트리밍 처리 |
|------|----------|--------------|
| **지연 시간** | 분~시간 | 밀리초~초 |
| **데이터 처리량** | 대량 | 소량/연속 |
| **복잡성** | 상대적 단순 | 상대적 복잡 |
| **재처리** | 용이 | 어려움 |
| **비용** | 저렴 | 고가 |
| **사용 사례** | 보고서, 집계 | 실시간 분석, 알림 |

---

## 4. 데이터 아키텍처 패턴

### 4.1 전통적인 데이터 웨어하우스 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│                  전통적 Data Warehouse 아키텍처                 │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐   ┌─────────┐   ┌─────────────────────────┐    │
│  │ Source 1│   │ Source 2│   │       Source N          │    │
│  │  (ERP)  │   │  (CRM)  │   │      (Other)            │    │
│  └────┬────┘   └────┬────┘   └───────────┬─────────────┘    │
│       │             │                     │                  │
│       └─────────────┼─────────────────────┘                  │
│                     ↓                                        │
│           ┌─────────────────┐                                │
│           │   ETL Process   │                                │
│           │ (Extract-Transform-Load)                         │
│           └────────┬────────┘                                │
│                    ↓                                         │
│           ┌─────────────────┐                                │
│           │  Data Warehouse │                                │
│           │   (Star Schema) │                                │
│           └────────┬────────┘                                │
│                    ↓                                         │
│           ┌─────────────────┐                                │
│           │    BI Tools     │                                │
│           │ (Tableau, Power BI)                              │
│           └─────────────────┘                                │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 모던 데이터 레이크 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│                  Modern Data Lake 아키텍처                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Sources                                                     │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                    │
│  │ API │ │ DB  │ │ IoT │ │ Log │ │Files│                    │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘                    │
│     └───────┴───────┴───────┴───────┘                        │
│                     ↓                                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    Data Lake                         │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐             │    │
│  │  │  Bronze │→│  Silver │→│  Gold   │              │    │
│  │  │   Raw   │  │ Cleaned │  │Curated │              │    │
│  │  └─────────┘  └─────────┘  └─────────┘             │    │
│  └─────────────────────────────────────────────────────┘    │
│                     ↓                                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │BI/Reports│ │ ML/AI    │ │ Data Apps│ │ API      │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└──────────────────────────────────────────────────────────────┘
```

### 4.3 람다 아키텍처 (Lambda Architecture)

배치와 스트리밍을 결합한 하이브리드 아키텍처입니다.

```python
# 람다 아키텍처 개념 구현
class LambdaArchitecture:
    """람다 아키텍처: 배치 + 스트리밍 레이어"""

    def __init__(self):
        self.batch_layer = BatchLayer()
        self.speed_layer = SpeedLayer()
        self.serving_layer = ServingLayer()

    def ingest(self, data):
        """데이터 수집: 두 레이어에 동시 전달"""
        # 배치 레이어 (마스터 데이터셋)
        self.batch_layer.append(data)

        # 스피드 레이어 (실시간 처리)
        self.speed_layer.process(data)

    def query(self, params):
        """쿼리: 배치 뷰 + 실시간 뷰 병합"""
        batch_result = self.serving_layer.get_batch_view(params)
        realtime_result = self.speed_layer.get_realtime_view(params)

        return self.merge_views(batch_result, realtime_result)


class BatchLayer:
    """배치 레이어: 전체 데이터셋 처리"""

    def append(self, data):
        """마스터 데이터셋에 추가"""
        # 불변 데이터 저장 (append-only)
        pass

    def compute_batch_views(self):
        """배치 뷰 계산 (주기적 실행)"""
        # MapReduce, Spark 등으로 전체 데이터 처리
        pass


class SpeedLayer:
    """스피드 레이어: 실시간 데이터 처리"""

    def process(self, data):
        """실시간 처리"""
        # 스트리밍 처리 (Kafka, Flink 등)
        pass

    def get_realtime_view(self, params):
        """실시간 뷰 반환"""
        pass


class ServingLayer:
    """서빙 레이어: 쿼리 처리"""

    def get_batch_view(self, params):
        """배치 뷰 반환"""
        pass
```

### 4.4 카파 아키텍처 (Kappa Architecture)

스트리밍만 사용하는 단순화된 아키텍처입니다.

```
┌──────────────────────────────────────────────────────────────┐
│                    Kappa Architecture                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Sources                                                     │
│  ┌─────┐ ┌─────┐ ┌─────┐                                    │
│  │Event│ │Event│ │Event│                                    │
│  └──┬──┘ └──┬──┘ └──┬──┘                                    │
│     └───────┴───────┘                                        │
│             ↓                                                │
│  ┌─────────────────────────────────────┐                    │
│  │         Message Queue (Kafka)       │                    │
│  │         - Event Log                 │                    │
│  │         - Replayable                │                    │
│  └─────────────────┬───────────────────┘                    │
│                    ↓                                         │
│  ┌─────────────────────────────────────┐                    │
│  │      Stream Processing Layer        │                    │
│  │      (Flink, Spark Streaming)       │                    │
│  └─────────────────┬───────────────────┘                    │
│                    ↓                                         │
│  ┌─────────────────────────────────────┐                    │
│  │          Serving Layer              │                    │
│  │    (Database, Cache, API)           │                    │
│  └─────────────────────────────────────┘                    │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. 데이터 엔지니어링 도구 생태계

### 5.1 주요 도구 분류

```python
data_engineering_tools = {
    "orchestration": {
        "batch": ["Apache Airflow", "Prefect", "Dagster", "Luigi"],
        "streaming": ["Apache Kafka", "Apache Flink", "Spark Streaming"]
    },
    "processing": {
        "batch": ["Apache Spark", "Apache Hive", "Presto/Trino"],
        "streaming": ["Apache Kafka Streams", "Apache Flink", "Apache Storm"]
    },
    "storage": {
        "data_lake": ["S3", "GCS", "HDFS", "Azure Blob"],
        "data_warehouse": ["Snowflake", "BigQuery", "Redshift", "Databricks"],
        "databases": ["PostgreSQL", "MySQL", "MongoDB", "Cassandra"]
    },
    "transformation": {
        "sql_based": ["dbt", "SQLMesh"],
        "code_based": ["PySpark", "Pandas", "Polars"]
    },
    "quality": {
        "testing": ["Great Expectations", "dbt tests", "Soda"],
        "monitoring": ["Monte Carlo", "Datadog", "Grafana"]
    },
    "catalog": ["Apache Atlas", "DataHub", "Amundsen", "OpenMetadata"]
}
```

### 5.2 클라우드 서비스 매핑

| 기능 | AWS | GCP | Azure |
|------|-----|-----|-------|
| **오케스트레이션** | Step Functions, MWAA | Cloud Composer | Data Factory |
| **스트리밍** | Kinesis | Pub/Sub, Dataflow | Event Hubs |
| **배치 처리** | EMR, Glue | Dataproc, Dataflow | HDInsight |
| **Data Lake** | S3 + Lake Formation | GCS + BigLake | ADLS + Synapse |
| **Data Warehouse** | Redshift | BigQuery | Synapse Analytics |

---

## 6. 데이터 엔지니어링 모범 사례

### 6.1 파이프라인 설계 원칙

```python
# 좋은 파이프라인 설계 원칙
pipeline_best_practices = {
    "idempotency": "같은 입력에 같은 결과 보장",
    "atomicity": "전체 성공 또는 전체 실패",
    "incremental": "증분 처리로 효율성 확보",
    "monitoring": "모든 단계에서 모니터링",
    "error_handling": "실패 시 재시도 및 알림",
    "documentation": "코드와 문서화 함께 관리"
}

# 멱등성(Idempotency) 예시
def idempotent_upsert(df, table_name, key_columns):
    """멱등성을 보장하는 upsert 함수"""
    # 기존 데이터 삭제 후 삽입 (MERGE 또는 DELETE + INSERT)
    delete_query = f"""
    DELETE FROM {table_name}
    WHERE (key1, key2) IN (
        SELECT DISTINCT key1, key2 FROM staging_table
    )
    """
    # execute(delete_query)
    # insert_dataframe(df, table_name)
    pass
```

### 6.2 에러 처리와 재시도

```python
import time
from functools import wraps
from typing import Callable, Type

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,)
):
    """재시도 데코레이터"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        print(f"Attempt {attempt} failed: {e}")
                        time.sleep(delay * attempt)  # 지수 백오프
            raise last_exception
        return wrapper
    return decorator


@retry(max_attempts=3, delay=2.0)
def fetch_data_from_api(url: str):
    """API에서 데이터 가져오기 (재시도 포함)"""
    import requests
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()
```

---

## 연습 문제

### 문제 1: 파이프라인 설계
온라인 쇼핑몰의 일일 매출 리포트를 생성하는 파이프라인을 설계하세요.

```python
# 풀이 예시
class DailySalesReportPipeline:
    def extract(self):
        """주문, 상품, 고객 데이터 추출"""
        pass

    def transform(self):
        """매출 집계, 카테고리별 분석"""
        pass

    def load(self):
        """리포트 테이블 적재"""
        pass
```

### 문제 2: 배치 vs 스트리밍 선택
다음 사례에서 배치와 스트리밍 중 적합한 방식을 선택하고 이유를 설명하세요:
- 일일 판매 보고서 생성
- 실시간 재고 부족 알림
- 월간 고객 세그먼테이션

---

## 요약

| 개념 | 설명 |
|------|------|
| **데이터 파이프라인** | 소스에서 목적지까지 데이터 이동 및 변환 |
| **배치 처리** | 대량 데이터를 주기적으로 처리 |
| **스트리밍 처리** | 실시간으로 데이터 처리 |
| **Data Lake** | 원시 데이터를 저장하는 저장소 |
| **Data Warehouse** | 정제된 데이터를 저장하는 분석용 저장소 |
| **ETL/ELT** | 데이터 추출, 변환, 적재 프로세스 |

---

## 참고 자료

- [Fundamentals of Data Engineering (O'Reilly)](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108298/)
- [The Data Engineering Cookbook](https://github.com/andkret/Cookbook)
- [Data Engineering Weekly Newsletter](https://dataengineeringweekly.com/)
