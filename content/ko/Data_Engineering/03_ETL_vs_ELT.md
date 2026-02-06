# ETL vs ELT

## 개요

ETL(Extract, Transform, Load)과 ELT(Extract, Load, Transform)는 데이터 파이프라인의 두 가지 주요 패턴입니다. 전통적인 ETL은 변환 후 적재하고, 모던 ELT는 적재 후 변환합니다.

---

## 1. ETL (Extract, Transform, Load)

### 1.1 ETL 프로세스

```
┌─────────────────────────────────────────────────────────────┐
│                      ETL Process                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐    ┌──────────────┐    ┌──────────┐         │
│   │ Sources  │ → │ ETL Server   │ → │ Target   │          │
│   │          │    │              │    │ (DW)     │          │
│   │ - DB     │    │ 1. Extract   │    │          │          │
│   │ - Files  │    │ 2. Transform │    │ Clean    │          │
│   │ - APIs   │    │ 3. Load      │    │ Data     │          │
│   └──────────┘    └──────────────┘    └──────────┘         │
│                                                             │
│   변환이 중간 서버에서 수행됨                                  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 ETL 예시 코드

```python
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

class ETLPipeline:
    """전통적인 ETL 파이프라인"""

    def __init__(self, source_conn: str, target_conn: str):
        self.source_engine = create_engine(source_conn)
        self.target_engine = create_engine(target_conn)

    def extract(self, query: str) -> pd.DataFrame:
        """
        Extract: 소스에서 데이터 추출
        """
        print(f"[Extract] Starting at {datetime.now()}")
        df = pd.read_sql(query, self.source_engine)
        print(f"[Extract] Extracted {len(df)} rows")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform: 데이터 정제 및 변환
        - 이 단계가 ETL 서버에서 수행됨 (리소스 소모)
        """
        print(f"[Transform] Starting at {datetime.now()}")

        # 1. 결측치 처리
        df = df.dropna(subset=['customer_id', 'amount'])
        df['email'] = df['email'].fillna('unknown@example.com')

        # 2. 데이터 타입 변환
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['amount'] = df['amount'].astype(float)

        # 3. 파생 컬럼 생성
        df['year'] = df['order_date'].dt.year
        df['month'] = df['order_date'].dt.month
        df['day_of_week'] = df['order_date'].dt.dayofweek

        # 4. 비즈니스 로직 적용
        df['customer_segment'] = df['total_purchases'].apply(
            lambda x: 'Gold' if x > 10000 else ('Silver' if x > 5000 else 'Bronze')
        )

        # 5. 데이터 품질 검증
        assert df['amount'].min() >= 0, "Negative amounts found"

        print(f"[Transform] Transformed {len(df)} rows")
        return df

    def load(self, df: pd.DataFrame, table_name: str):
        """
        Load: 타겟 데이터 웨어하우스에 적재
        """
        print(f"[Load] Starting at {datetime.now()}")

        # Full refresh (테이블 교체)
        df.to_sql(
            table_name,
            self.target_engine,
            if_exists='replace',
            index=False,
            chunksize=10000
        )

        print(f"[Load] Loaded {len(df)} rows to {table_name}")

    def run(self, source_query: str, target_table: str):
        """ETL 파이프라인 실행"""
        start_time = datetime.now()
        print(f"ETL Pipeline started at {start_time}")

        # E-T-L 순서로 실행
        raw_data = self.extract(source_query)
        transformed_data = self.transform(raw_data)
        self.load(transformed_data, target_table)

        end_time = datetime.now()
        print(f"ETL Pipeline completed in {(end_time - start_time).seconds} seconds")


# 사용 예시
if __name__ == "__main__":
    pipeline = ETLPipeline(
        source_conn="postgresql://user:pass@source-db:5432/sales",
        target_conn="postgresql://user:pass@warehouse:5432/analytics"
    )

    pipeline.run(
        source_query="""
            SELECT
                o.order_id,
                o.customer_id,
                c.email,
                o.order_date,
                o.amount,
                c.total_purchases
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            WHERE o.order_date >= CURRENT_DATE - INTERVAL '1 day'
        """,
        target_table="fact_daily_orders"
    )
```

### 1.3 ETL 도구

| 도구 | 유형 | 특징 |
|------|------|------|
| **Informatica** | 상용 | 엔터프라이즈급, GUI 기반 |
| **Talend** | 오픈소스/상용 | Java 기반, 다양한 커넥터 |
| **SSIS** | 상용 (MS) | SQL Server 통합 |
| **Pentaho** | 오픈소스 | 경량, 사용 편의 |
| **Apache NiFi** | 오픈소스 | 데이터 플로우, 실시간 |

---

## 2. ELT (Extract, Load, Transform)

### 2.1 ELT 프로세스

```
┌─────────────────────────────────────────────────────────────┐
│                      ELT Process                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │ Sources  │ → │ Load Raw     │ → │ Transform    │      │
│   │          │    │ (Data Lake)  │    │ (in DW)      │      │
│   │ - DB     │    │              │    │              │      │
│   │ - Files  │    │ Raw Zone     │    │ SQL/Spark    │      │
│   │ - APIs   │    │ (as-is)      │    │ (DW 리소스)  │      │
│   └──────────┘    └──────────────┘    └──────────────┘     │
│                                                             │
│   변환이 타겟 시스템(DW/Lake)에서 수행됨                       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 ELT 예시 코드

```python
import pandas as pd
from datetime import datetime

class ELTPipeline:
    """모던 ELT 파이프라인"""

    def __init__(self, source_conn: str, warehouse_conn: str):
        self.source_conn = source_conn
        self.warehouse_conn = warehouse_conn

    def extract_and_load(self, source_query: str, raw_table: str):
        """
        Extract & Load: 원본 데이터를 그대로 적재
        - 변환 없이 raw 데이터를 빠르게 적재
        """
        print(f"[Extract & Load] Starting at {datetime.now()}")

        # 소스에서 데이터 추출
        df = pd.read_sql(source_query, self.source_conn)

        # Raw 테이블에 그대로 적재 (변환 없음)
        df.to_sql(
            raw_table,
            self.warehouse_conn,
            if_exists='replace',
            index=False
        )

        print(f"[Extract & Load] Loaded {len(df)} rows to {raw_table}")

    def transform_in_warehouse(self, transform_sql: str):
        """
        Transform: 웨어하우스 내에서 SQL로 변환
        - DW의 컴퓨팅 파워 활용
        - SQL 기반 변환 (dbt 등 사용)
        """
        print(f"[Transform] Starting at {datetime.now()}")

        # 웨어하우스에서 SQL 실행
        with self.warehouse_conn.connect() as conn:
            conn.execute(transform_sql)

        print(f"[Transform] Transformation completed")


# dbt 모델 예시 (SQL 기반 변환)
DBT_MODEL_EXAMPLE = """
-- models/staging/stg_orders.sql
-- dbt를 사용한 ELT 변환

WITH source AS (
    SELECT * FROM {{ source('raw', 'orders_raw') }}
),

cleaned AS (
    SELECT
        order_id,
        customer_id,
        COALESCE(email, 'unknown@example.com') AS email,
        CAST(order_date AS DATE) AS order_date,
        CAST(amount AS DECIMAL(10, 2)) AS amount,
        total_purchases,
        -- 파생 컬럼
        EXTRACT(YEAR FROM order_date) AS order_year,
        EXTRACT(MONTH FROM order_date) AS order_month,
        EXTRACT(DOW FROM order_date) AS day_of_week,
        -- 비즈니스 로직
        CASE
            WHEN total_purchases > 10000 THEN 'Gold'
            WHEN total_purchases > 5000 THEN 'Silver'
            ELSE 'Bronze'
        END AS customer_segment,
        -- 메타데이터
        CURRENT_TIMESTAMP AS loaded_at
    FROM source
    WHERE customer_id IS NOT NULL
      AND amount IS NOT NULL
      AND amount >= 0
)

SELECT * FROM cleaned
"""


# 실제 ELT 파이프라인 (Snowflake/BigQuery 스타일)
class ModernELTWithSQL:
    """SQL 기반 모던 ELT"""

    def __init__(self, warehouse):
        self.warehouse = warehouse

    def extract_load(self, source: str, target_raw: str):
        """원본 → Raw 레이어"""
        copy_sql = f"""
        COPY INTO {target_raw}
        FROM @{source}
        FILE_FORMAT = (TYPE = 'PARQUET')
        """
        self.warehouse.execute(copy_sql)

    def transform_staging(self):
        """Raw → Staging 레이어"""
        staging_sql = """
        CREATE OR REPLACE TABLE staging.orders AS
        SELECT
            order_id,
            customer_id,
            PARSE_JSON(raw_data):email::STRING AS email,
            TO_DATE(raw_data:order_date) AS order_date,
            raw_data:amount::NUMBER(10,2) AS amount
        FROM raw.orders_raw
        """
        self.warehouse.execute(staging_sql)

    def transform_mart(self):
        """Staging → Mart 레이어"""
        mart_sql = """
        CREATE OR REPLACE TABLE mart.fact_orders AS
        SELECT
            o.order_id,
            d.date_sk,
            c.customer_sk,
            o.amount,
            -- 집계
            SUM(o.amount) OVER (
                PARTITION BY o.customer_id
                ORDER BY o.order_date
            ) AS cumulative_amount
        FROM staging.orders o
        JOIN dim_date d ON o.order_date = d.full_date
        JOIN dim_customer c ON o.customer_id = c.customer_id
        """
        self.warehouse.execute(mart_sql)
```

### 2.3 ELT 도구

| 도구 | 유형 | 특징 |
|------|------|------|
| **dbt** | 오픈소스 | SQL 기반 변환, 테스트, 문서화 |
| **Fivetran** | 상용 | 자동 스키마 관리, 150+ 커넥터 |
| **Airbyte** | 오픈소스 | 커스텀 커넥터, EL 특화 |
| **Stitch** | 상용 | 간편한 설정, SaaS 친화적 |
| **AWS Glue** | 클라우드 | 서버리스, Spark 기반 |

---

## 3. ETL vs ELT 비교

### 3.1 상세 비교

| 특성 | ETL | ELT |
|------|-----|-----|
| **변환 위치** | 중간 서버 | 타겟 시스템 (DW/Lake) |
| **데이터 이동** | 변환된 데이터만 | 원본 데이터 전체 |
| **스키마** | 미리 정의 필요 | 유연 (Schema-on-Read) |
| **처리 속도** | 느림 (중간 처리) | 빠름 (병렬 처리) |
| **비용** | 별도 인프라 필요 | DW 리소스 사용 |
| **유연성** | 낮음 | 높음 (원본 보존) |
| **복잡한 변환** | 적합 | 제한적 |
| **실시간 처리** | 어려움 | 비교적 용이 |

### 3.2 선택 기준

```python
def choose_etl_or_elt(requirements: dict) -> str:
    """ETL/ELT 선택 가이드"""

    # ETL 선호 상황
    etl_factors = [
        requirements.get('data_privacy', False),      # 민감 데이터 마스킹 필요
        requirements.get('complex_transforms', False), # 복잡한 비즈니스 로직
        requirements.get('legacy_systems', False),    # 레거시 시스템 연동
        requirements.get('small_data', False),        # 소규모 데이터
    ]

    # ELT 선호 상황
    elt_factors = [
        requirements.get('big_data', False),          # 대용량 데이터
        requirements.get('cloud_dw', False),          # 클라우드 DW 사용
        requirements.get('data_lake', False),         # 데이터 레이크 구축
        requirements.get('flexible_schema', False),   # 스키마 유연성 필요
        requirements.get('raw_data_access', False),   # 원본 데이터 접근 필요
        requirements.get('sql_transforms', False),    # SQL로 변환 가능
    ]

    etl_score = sum(etl_factors)
    elt_score = sum(elt_factors)

    if etl_score > elt_score:
        return "ETL 권장"
    elif elt_score > etl_score:
        return "ELT 권장"
    else:
        return "하이브리드 고려"


# 사용 예시
project_requirements = {
    'big_data': True,
    'cloud_dw': True,  # Snowflake, BigQuery
    'sql_transforms': True,
    'raw_data_access': True
}

recommendation = choose_etl_or_elt(project_requirements)
print(recommendation)  # "ELT 권장"
```

---

## 4. 하이브리드 접근법

### 4.1 ETLT 패턴

```
┌─────────────────────────────────────────────────────────────┐
│                    ETLT (Hybrid) Pattern                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Sources → [E] → [T] → [L] → [T] → Mart                   │
│                    ↑          ↑                             │
│                    │          │                             │
│            Light Transform   Heavy Transform                │
│            (마스킹, 검증)    (집계, 조인)                    │
│            ETL Server        Data Warehouse                 │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 하이브리드 구현 예시

```python
class HybridPipeline:
    """ETL + ELT 하이브리드 파이프라인"""

    def __init__(self, source, staging_area, warehouse):
        self.source = source
        self.staging = staging_area
        self.warehouse = warehouse

    def extract_with_light_transform(self):
        """
        E + Light T: 추출하면서 가벼운 변환 수행
        - PII 마스킹 (개인정보 보호)
        - 기본 데이터 타입 변환
        - 필수 필드 검증
        """
        query = """
        SELECT
            order_id,
            -- PII 마스킹 (소스에서 수행)
            MD5(customer_email) AS customer_email_hash,
            SUBSTRING(phone, 1, 3) || '****' || SUBSTRING(phone, -4) AS phone_masked,
            -- 기본 변환
            CAST(order_date AS DATE) AS order_date,
            CAST(amount AS DECIMAL(10, 2)) AS amount
        FROM orders
        WHERE amount IS NOT NULL
        """
        return self.source.execute(query)

    def load_to_staging(self, data):
        """L: 스테이징 영역에 적재"""
        self.staging.load(data, 'orders_staging')

    def transform_in_warehouse(self):
        """
        Heavy T: 웨어하우스에서 복잡한 변환
        - 조인
        - 집계
        - 윈도우 함수
        """
        heavy_transform_sql = """
        CREATE TABLE mart.order_analysis AS
        SELECT
            o.order_date,
            c.customer_segment,
            p.product_category,
            COUNT(*) AS order_count,
            SUM(o.amount) AS total_amount,
            AVG(o.amount) AS avg_order_value,
            -- 윈도우 함수 (DW에서 효율적)
            SUM(o.amount) OVER (
                PARTITION BY c.customer_segment
                ORDER BY o.order_date
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ) AS rolling_7day_amount
        FROM orders_staging o
        JOIN dim_customer c ON o.customer_id = c.customer_id
        JOIN dim_product p ON o.product_id = p.product_id
        GROUP BY o.order_date, c.customer_segment, p.product_category
        """
        self.warehouse.execute(heavy_transform_sql)

    def run(self):
        """하이브리드 파이프라인 실행"""
        # Phase 1: ETL (Extract + Light Transform)
        data = self.extract_with_light_transform()

        # Phase 2: Load to staging
        self.load_to_staging(data)

        # Phase 3: ELT (Heavy Transform in DW)
        self.transform_in_warehouse()
```

---

## 5. 실무 사례

### 5.1 ETL 사용 사례

```python
# 사례 1: 개인정보 처리 (GDPR 준수)
class GDPRCompliantETL:
    """GDPR 준수 ETL - 개인정보 마스킹 후 적재"""

    def transform(self, df):
        # 민감 정보 마스킹 (적재 전 수행)
        df['email'] = df['email'].apply(self.mask_email)
        df['ssn'] = df['ssn'].apply(lambda x: 'XXX-XX-' + x[-4:])
        df['credit_card'] = df['credit_card'].apply(lambda x: '**** **** **** ' + x[-4:])

        # EU 외 지역으로 데이터 전송 전 변환
        df = df[df['consent_given'] == True]

        return df

    def mask_email(self, email):
        if pd.isna(email):
            return None
        local, domain = email.split('@')
        return local[:2] + '***@' + domain


# 사례 2: 레거시 시스템 통합
class LegacySystemETL:
    """레거시 메인프레임 데이터 통합"""

    def transform(self, raw_data):
        # 고정 길이 레코드 파싱
        records = []
        for line in raw_data.split('\n'):
            record = {
                'account_no': line[0:10].strip(),
                'account_type': line[10:12],
                'balance': int(line[12:24]) / 100,  # 소수점 변환
                'status': 'A' if line[24:25] == '1' else 'I',
                'date': self.parse_legacy_date(line[25:33])
            }
            records.append(record)
        return pd.DataFrame(records)

    def parse_legacy_date(self, date_str):
        # YYYYMMDD → YYYY-MM-DD
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
```

### 5.2 ELT 사용 사례

```sql
-- 사례 1: dbt를 활용한 이커머스 분석

-- models/staging/stg_orders.sql
WITH raw_orders AS (
    SELECT * FROM {{ source('raw', 'orders') }}
)
SELECT
    order_id,
    customer_id,
    order_date,
    status,
    total_amount
FROM raw_orders
WHERE order_date IS NOT NULL

-- models/marts/core/fct_orders.sql
WITH orders AS (
    SELECT * FROM {{ ref('stg_orders') }}
),
customers AS (
    SELECT * FROM {{ ref('dim_customers') }}
),
products AS (
    SELECT * FROM {{ ref('dim_products') }}
)
SELECT
    o.order_id,
    o.order_date,
    c.customer_segment,
    p.product_category,
    o.total_amount,
    -- DW에서 윈도우 함수 활용
    ROW_NUMBER() OVER (PARTITION BY o.customer_id ORDER BY o.order_date) AS order_sequence,
    LAG(o.order_date) OVER (PARTITION BY o.customer_id ORDER BY o.order_date) AS prev_order_date
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id


-- 사례 2: BigQuery ELT
-- 대용량 로그 분석 (서버리스 처리)
CREATE OR REPLACE TABLE analytics.user_behavior AS
SELECT
    user_id,
    DATE(timestamp) AS event_date,
    event_type,
    COUNT(*) AS event_count,
    COUNTIF(event_type = 'purchase') AS purchase_count,
    SUM(CASE WHEN event_type = 'purchase' THEN revenue ELSE 0 END) AS total_revenue,
    -- 세션 분석 (복잡한 윈도우 함수)
    ARRAY_AGG(
        STRUCT(timestamp, event_type, page_url)
        ORDER BY timestamp
    ) AS event_sequence
FROM raw.events
WHERE DATE(timestamp) = CURRENT_DATE() - 1
GROUP BY user_id, DATE(timestamp), event_type;
```

---

## 6. 도구 선택 가이드

### 6.1 데이터 규모별 권장 도구

| 데이터 규모 | ETL 도구 | ELT 도구 |
|-------------|----------|----------|
| **소규모** (< 1GB) | Python + Pandas | dbt + PostgreSQL |
| **중규모** (1GB-100GB) | Airflow + Python | dbt + Snowflake |
| **대규모** (> 100GB) | Spark | dbt + BigQuery/Databricks |

### 6.2 아키텍처별 권장

```python
architecture_recommendations = {
    "traditional_dw": {
        "approach": "ETL",
        "tools": ["Informatica", "Talend", "SSIS"],
        "reason": "스키마 엄격, 변환 후 적재"
    },
    "cloud_dw": {
        "approach": "ELT",
        "tools": ["dbt", "Fivetran + dbt", "Airbyte + dbt"],
        "reason": "DW 컴퓨팅 파워 활용, 원본 보존"
    },
    "data_lake": {
        "approach": "ELT",
        "tools": ["Spark", "AWS Glue", "Databricks"],
        "reason": "스키마 유연, 대용량 처리"
    },
    "hybrid": {
        "approach": "ETLT",
        "tools": ["Airflow + dbt", "Prefect + dbt"],
        "reason": "민감 정보 처리 + DW 변환"
    }
}
```

---

## 연습 문제

### 문제 1: ETL vs ELT 선택
다음 상황에서 ETL과 ELT 중 어떤 방식이 적합한지 선택하고 이유를 설명하세요:
- 일일 100GB의 로그 데이터를 BigQuery에 적재
- 개인정보가 포함된 고객 데이터를 처리

### 문제 2: ELT SQL 작성
Raw 테이블 `raw_orders`에서 일별 매출 집계 테이블을 생성하는 ELT SQL을 작성하세요.

---

## 요약

| 개념 | 설명 |
|------|------|
| **ETL** | 변환 후 적재, 중간 서버에서 처리 |
| **ELT** | 적재 후 변환, 타겟 시스템에서 처리 |
| **ETL 장점** | 데이터 품질 보장, 민감 정보 처리 |
| **ELT 장점** | 빠른 적재, 유연한 스키마, 원본 보존 |
| **하이브리드** | ETL + ELT 조합, 상황에 맞게 선택 |

---

## 참고 자료

- [dbt Documentation](https://docs.getdbt.com/)
- [Modern Data Stack](https://www.moderndatastack.xyz/)
- [ETL vs ELT: The Difference](https://www.fivetran.com/blog/etl-vs-elt)
