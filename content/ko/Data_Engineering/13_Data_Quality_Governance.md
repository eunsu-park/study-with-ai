# 데이터 품질과 거버넌스

## 개요

데이터 품질은 데이터의 정확성, 완전성, 일관성을 보장하는 것이고, 데이터 거버넌스는 데이터 자산을 체계적으로 관리하는 프레임워크입니다. 신뢰할 수 있는 데이터 파이프라인을 위해 필수적입니다.

---

## 1. 데이터 품질 차원

### 1.1 품질 차원 정의

```
┌────────────────────────────────────────────────────────────────┐
│                   데이터 품질 6대 차원                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   1. 정확성 (Accuracy)                                         │
│      - 데이터가 실제 값을 올바르게 반영하는가?                    │
│      - 예: 고객 이메일이 유효한 형식인가?                        │
│                                                                │
│   2. 완전성 (Completeness)                                     │
│      - 필요한 모든 데이터가 존재하는가?                          │
│      - 예: 필수 필드에 NULL이 없는가?                           │
│                                                                │
│   3. 일관성 (Consistency)                                      │
│      - 데이터가 여러 시스템 간 일치하는가?                       │
│      - 예: 주문 수가 주문 테이블과 집계 테이블에서 동일한가?      │
│                                                                │
│   4. 적시성 (Timeliness)                                       │
│      - 데이터가 적절한 시간 내에 제공되는가?                     │
│      - 예: 실시간 대시보드가 5분 내 갱신되는가?                  │
│                                                                │
│   5. 유일성 (Uniqueness)                                       │
│      - 중복 데이터가 없는가?                                    │
│      - 예: 동일한 주문이 중복 기록되지 않았는가?                 │
│                                                                │
│   6. 유효성 (Validity)                                         │
│      - 데이터가 정의된 규칙을 준수하는가?                        │
│      - 예: 날짜가 올바른 형식인가?                              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 품질 메트릭 예시

```python
from dataclasses import dataclass
from typing import Optional
import pandas as pd

@dataclass
class DataQualityMetrics:
    """데이터 품질 메트릭"""
    table_name: str
    row_count: int
    null_count: dict[str, int]
    duplicate_count: int
    freshness_hours: float
    schema_valid: bool

def calculate_quality_metrics(df: pd.DataFrame, table_name: str) -> DataQualityMetrics:
    """품질 메트릭 계산"""

    # 완전성: NULL 수
    null_count = {col: df[col].isna().sum() for col in df.columns}

    # 유일성: 중복 수
    duplicate_count = df.duplicated().sum()

    return DataQualityMetrics(
        table_name=table_name,
        row_count=len(df),
        null_count=null_count,
        duplicate_count=duplicate_count,
        freshness_hours=0,  # 별도 계산 필요
        schema_valid=True    # 별도 검증 필요
    )


def quality_score(metrics: DataQualityMetrics) -> float:
    """0-100 품질 점수 계산"""
    scores = []

    # 완전성 점수 (NULL 비율)
    total_cells = metrics.row_count * len(metrics.null_count)
    total_nulls = sum(metrics.null_count.values())
    completeness = (1 - total_nulls / total_cells) * 100 if total_cells > 0 else 100
    scores.append(completeness)

    # 유일성 점수 (중복 비율)
    uniqueness = (1 - metrics.duplicate_count / metrics.row_count) * 100 if metrics.row_count > 0 else 100
    scores.append(uniqueness)

    return sum(scores) / len(scores)
```

---

## 2. Great Expectations

### 2.1 설치 및 초기화

```bash
# 설치
pip install great_expectations

# 프로젝트 초기화
great_expectations init
```

### 2.2 기본 사용법

```python
import great_expectations as gx
import pandas as pd

# Context 생성
context = gx.get_context()

# 데이터 소스 추가
datasource = context.sources.add_pandas("my_datasource")

# 데이터 에셋 정의
data_asset = datasource.add_dataframe_asset(name="orders")

# DataFrame 로드
df = pd.read_csv("orders.csv")

# Batch Request
batch_request = data_asset.build_batch_request(dataframe=df)

# Expectation Suite 생성
suite = context.add_expectation_suite("orders_suite")

# Validator 생성
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name="orders_suite"
)
```

### 2.3 Expectations 정의

```python
# 기본 Expectations

# NULL 없음
validator.expect_column_values_to_not_be_null("order_id")

# 유니크
validator.expect_column_values_to_be_unique("order_id")

# 값 범위
validator.expect_column_values_to_be_between(
    "amount",
    min_value=0,
    max_value=1000000
)

# 허용 값 목록
validator.expect_column_values_to_be_in_set(
    "status",
    ["pending", "completed", "cancelled", "refunded"]
)

# 정규식 매칭
validator.expect_column_values_to_match_regex(
    "email",
    r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
)

# 테이블 행 수
validator.expect_table_row_count_to_be_between(
    min_value=1000,
    max_value=1000000
)

# 컬럼 존재
validator.expect_table_columns_to_match_set(
    ["order_id", "customer_id", "amount", "status", "order_date"]
)

# 날짜 형식
validator.expect_column_values_to_match_strftime_format(
    "order_date",
    "%Y-%m-%d"
)

# 참조 무결성 (다른 테이블)
validator.expect_column_values_to_be_in_set(
    "customer_id",
    customer_ids_list  # 고객 테이블의 ID 목록
)

# Suite 저장
validator.save_expectation_suite(discard_failed_expectations=False)
```

### 2.4 검증 실행

```python
# Checkpoint 생성 및 실행
checkpoint = context.add_or_update_checkpoint(
    name="orders_checkpoint",
    validations=[
        {
            "batch_request": batch_request,
            "expectation_suite_name": "orders_suite"
        }
    ]
)

# 검증 실행
result = checkpoint.run()

# 결과 확인
print(f"Success: {result.success}")
print(f"Statistics: {result.statistics}")

# 실패한 Expectations 확인
for validation_result in result.list_validation_results():
    for exp_result in validation_result.results:
        if not exp_result.success:
            print(f"Failed: {exp_result.expectation_config.expectation_type}")
            print(f"  Column: {exp_result.expectation_config.kwargs.get('column')}")
            print(f"  Result: {exp_result.result}")
```

### 2.5 데이터 문서 생성

```python
# Data Docs 빌드 및 열기
context.build_data_docs()
context.open_data_docs()
```

---

## 3. Airflow 통합

### 3.1 Great Expectations Operator

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import great_expectations as gx

def validate_data(**kwargs):
    """Great Expectations 검증 Task"""
    context = gx.get_context()

    # Checkpoint 실행
    result = context.run_checkpoint(
        checkpoint_name="orders_checkpoint"
    )

    if not result.success:
        raise ValueError("Data quality check failed!")

    return result.statistics


with DAG(
    'data_quality_dag',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily',
) as dag:

    validate = PythonOperator(
        task_id='validate_orders',
        python_callable=validate_data,
    )
```

### 3.2 커스텀 품질 검사

```python
from airflow.operators.python import PythonOperator, BranchPythonOperator

def check_row_count(**kwargs):
    """행 수 검증"""
    import pandas as pd

    df = pd.read_parquet(f"/data/{kwargs['ds']}/orders.parquet")
    row_count = len(df)

    # XCom으로 메트릭 저장
    kwargs['ti'].xcom_push(key='row_count', value=row_count)

    if row_count < 1000:
        raise ValueError(f"Row count too low: {row_count}")

    return row_count


def check_freshness(**kwargs):
    """데이터 신선도 검증"""
    from datetime import datetime, timedelta

    # 파일 수정 시간 확인
    import os
    file_path = f"/data/{kwargs['ds']}/orders.parquet"
    mtime = datetime.fromtimestamp(os.path.getmtime(file_path))

    age_hours = (datetime.now() - mtime).total_seconds() / 3600

    if age_hours > 24:
        raise ValueError(f"Data too old: {age_hours:.1f} hours")

    return age_hours


def decide_next_step(**kwargs):
    """품질 결과에 따른 분기"""
    ti = kwargs['ti']
    row_count = ti.xcom_pull(task_ids='check_row_count', key='row_count')

    if row_count > 10000:
        return 'process_large_batch'
    else:
        return 'process_small_batch'


with DAG('quality_checks_dag', ...) as dag:

    check_rows = PythonOperator(
        task_id='check_row_count',
        python_callable=check_row_count,
    )

    check_fresh = PythonOperator(
        task_id='check_freshness',
        python_callable=check_freshness,
    )

    branch = BranchPythonOperator(
        task_id='decide_processing',
        python_callable=decide_next_step,
    )

    [check_rows, check_fresh] >> branch
```

---

## 4. 데이터 카탈로그

### 4.1 카탈로그 개념

```
┌────────────────────────────────────────────────────────────────┐
│                    데이터 카탈로그                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   메타데이터 관리 시스템:                                       │
│                                                                │
│   ┌────────────────────────────────────────────────────────┐  │
│   │  기술 메타데이터                                        │  │
│   │  - 스키마, 데이터 타입, 파티션                          │  │
│   │  - 위치, 형식, 크기                                     │  │
│   │  - 생성일, 수정일                                       │  │
│   └────────────────────────────────────────────────────────┘  │
│                                                                │
│   ┌────────────────────────────────────────────────────────┐  │
│   │  비즈니스 메타데이터                                    │  │
│   │  - 설명, 정의, 용어                                     │  │
│   │  - 소유자, 관리자                                       │  │
│   │  - 태그, 분류                                           │  │
│   └────────────────────────────────────────────────────────┘  │
│                                                                │
│   ┌────────────────────────────────────────────────────────┐  │
│   │  운영 메타데이터                                        │  │
│   │  - 사용 빈도, 쿼리 패턴                                 │  │
│   │  - 품질 점수, 이슈                                      │  │
│   │  - 접근 권한                                            │  │
│   └────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 4.2 카탈로그 도구

| 도구 | 유형 | 특징 |
|------|------|------|
| **DataHub** | 오픈소스 | LinkedIn 개발, 범용 |
| **Apache Atlas** | 오픈소스 | Hadoop 생태계 |
| **Amundsen** | 오픈소스 | Lyft 개발, 검색 중심 |
| **OpenMetadata** | 오픈소스 | 올인원 플랫폼 |
| **Atlan** | 상용 | 협업 중심 |
| **Alation** | 상용 | 엔터프라이즈 |

### 4.3 DataHub 예시

```python
# DataHub 메타데이터 수집 예시
from datahub.emitter.mce_builder import make_dataset_urn
from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.metadata.schema_classes import (
    DatasetPropertiesClass,
    SchemaMetadataClass,
    SchemaFieldClass,
    StringTypeClass,
    NumberTypeClass,
)

# Emitter 생성
emitter = DatahubRestEmitter(gms_server="http://localhost:8080")

# 데이터셋 URN
dataset_urn = make_dataset_urn(
    platform="postgres",
    name="analytics.public.fact_orders",
    env="PROD"
)

# 데이터셋 속성
properties = DatasetPropertiesClass(
    description="주문 팩트 테이블",
    customProperties={
        "owner": "data-team@company.com",
        "sla": "daily",
        "pii": "false"
    }
)

# 스키마 정의
schema = SchemaMetadataClass(
    schemaName="fact_orders",
    platform=f"urn:li:dataPlatform:postgres",
    fields=[
        SchemaFieldClass(
            fieldPath="order_id",
            type=StringTypeClass(),
            description="주문 고유 ID"
        ),
        SchemaFieldClass(
            fieldPath="amount",
            type=NumberTypeClass(),
            description="주문 금액"
        ),
    ]
)

# 메타데이터 emit
emitter.emit_mce(properties)
emitter.emit_mce(schema)
```

---

## 5. 데이터 리니지

### 5.1 리니지 개념

```
┌────────────────────────────────────────────────────────────────┐
│                     데이터 리니지 (Lineage)                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   데이터의 출처와 변환 과정을 추적:                              │
│                                                                │
│   Raw Sources          Staging           Marts                 │
│   ┌──────────┐        ┌──────────┐      ┌──────────┐          │
│   │ orders   │───────→│stg_orders│─────→│fct_orders│          │
│   │ (raw)    │        │          │      │          │          │
│   └──────────┘        └──────────┘      └────┬─────┘          │
│                                               │                │
│   ┌──────────┐        ┌──────────┐           │                │
│   │customers │───────→│stg_customers│────────→│                │
│   │ (raw)    │        │          │           │                │
│   └──────────┘        └──────────┘           │                │
│                                               ↓                │
│                                         ┌──────────┐          │
│                                         │ dashboard│          │
│                                         │ (BI)     │          │
│                                         └──────────┘          │
│                                                                │
│   활용:                                                        │
│   - 영향 분석: 소스 변경 시 영향받는 대상 파악                   │
│   - 근본 원인 분석: 데이터 이슈의 원인 추적                     │
│   - 규정 준수: 데이터 흐름 감사                                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 dbt 리니지

```bash
# dbt 리니지 생성
dbt docs generate

# 리니지 확인 (docs 서버)
dbt docs serve
```

```yaml
# dbt 모델 메타데이터
version: 2

models:
  - name: fct_orders
    description: "주문 팩트 테이블"
    meta:
      owner: "data-team"
      upstream:
        - stg_orders
        - stg_customers
      downstream:
        - sales_dashboard
        - ml_model_features
```

### 5.3 OpenLineage

```python
# OpenLineage를 사용한 리니지 추적
from openlineage.client import OpenLineageClient
from openlineage.client.run import Run, Job, RunEvent, RunState
from openlineage.client.facet import (
    SqlJobFacet,
    SchemaDatasetFacet,
    SchemaField,
)
from datetime import datetime
import uuid

client = OpenLineageClient(url="http://localhost:5000")

# Job 정의
job = Job(
    namespace="my_pipeline",
    name="transform_orders"
)

# Run 시작
run_id = str(uuid.uuid4())
run = Run(runId=run_id)

# 입력 데이터셋
input_datasets = [
    {
        "namespace": "postgres",
        "name": "raw.orders",
        "facets": {
            "schema": SchemaDatasetFacet(
                fields=[
                    SchemaField(name="order_id", type="string"),
                    SchemaField(name="amount", type="decimal"),
                ]
            )
        }
    }
]

# 출력 데이터셋
output_datasets = [
    {
        "namespace": "postgres",
        "name": "analytics.fct_orders",
    }
]

# Start 이벤트
client.emit(
    RunEvent(
        eventType=RunState.START,
        eventTime=datetime.now().isoformat(),
        run=run,
        job=job,
        inputs=input_datasets,
        outputs=output_datasets,
    )
)

# ... 실제 변환 작업 ...

# Complete 이벤트
client.emit(
    RunEvent(
        eventType=RunState.COMPLETE,
        eventTime=datetime.now().isoformat(),
        run=run,
        job=job,
    )
)
```

---

## 6. 거버넌스 프레임워크

### 6.1 데이터 거버넌스 구성 요소

```
┌────────────────────────────────────────────────────────────────┐
│                 데이터 거버넌스 프레임워크                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   1. 조직 (Organization)                                       │
│      - 데이터 스튜어드 지정                                     │
│      - 역할과 책임 정의                                         │
│      - 거버넌스 위원회                                          │
│                                                                │
│   2. 정책 (Policies)                                           │
│      - 데이터 분류 정책                                         │
│      - 접근 제어 정책                                           │
│      - 보존/삭제 정책                                           │
│      - 품질 기준                                                │
│                                                                │
│   3. 프로세스 (Processes)                                      │
│      - 데이터 요청/승인 프로세스                                │
│      - 이슈 관리 프로세스                                       │
│      - 변경 관리 프로세스                                       │
│                                                                │
│   4. 기술 (Technology)                                         │
│      - 데이터 카탈로그                                          │
│      - 품질 모니터링                                            │
│      - 접근 제어 시스템                                         │
│      - 감사 로그                                                │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 6.2 데이터 분류

```python
from enum import Enum

class DataClassification(Enum):
    """데이터 민감도 분류"""
    PUBLIC = "public"           # 공개 가능
    INTERNAL = "internal"       # 내부 사용
    CONFIDENTIAL = "confidential"  # 기밀
    RESTRICTED = "restricted"   # 제한적 (PII, 금융)

class DataClassifier:
    """자동 데이터 분류"""

    PII_PATTERNS = {
        'email': r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
        'phone': r'\d{3}-\d{3,4}-\d{4}',
        'ssn': r'\d{3}-\d{2}-\d{4}',
        'credit_card': r'\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}',
    }

    PII_COLUMN_NAMES = [
        'email', 'phone', 'ssn', 'social_security',
        'credit_card', 'password', 'address'
    ]

    @classmethod
    def classify_column(cls, column_name: str, sample_values: list) -> DataClassification:
        """컬럼 분류"""
        column_lower = column_name.lower()

        # 컬럼명 기반 분류
        if any(pii in column_lower for pii in cls.PII_COLUMN_NAMES):
            return DataClassification.RESTRICTED

        # 값 패턴 기반 분류
        import re
        for value in sample_values[:100]:  # 샘플링
            if value is None:
                continue
            for pii_type, pattern in cls.PII_PATTERNS.items():
                if re.match(pattern, str(value)):
                    return DataClassification.RESTRICTED

        return DataClassification.INTERNAL
```

---

## 연습 문제

### 문제 1: Great Expectations
주문 데이터에 대한 Expectation Suite를 작성하세요 (NULL 체크, 유니크, 값 범위, 참조 무결성).

### 문제 2: 품질 대시보드
일별 데이터 품질 점수를 계산하고 시각화하는 파이프라인을 설계하세요.

### 문제 3: 리니지 추적
ETL 파이프라인의 리니지를 자동으로 추적하는 시스템을 설계하세요.

---

## 요약

| 개념 | 설명 |
|------|------|
| **데이터 품질** | 정확성, 완전성, 일관성, 적시성 보장 |
| **Great Expectations** | Python 기반 데이터 품질 프레임워크 |
| **데이터 카탈로그** | 메타데이터 관리 시스템 |
| **데이터 리니지** | 데이터 출처와 변환 추적 |
| **데이터 거버넌스** | 데이터 자산의 체계적 관리 |

---

## 참고 자료

- [Great Expectations Documentation](https://docs.greatexpectations.io/)
- [DataHub Documentation](https://datahubproject.io/docs/)
- [OpenLineage](https://openlineage.io/)
- [DMBOK (Data Management Body of Knowledge)](https://www.dama.org/cpages/body-of-knowledge)
