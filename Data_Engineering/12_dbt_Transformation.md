# dbt 변환 도구

## 개요

dbt(data build tool)는 SQL 기반의 데이터 변환 도구입니다. ELT 패턴에서 Transform 단계를 담당하며, 소프트웨어 엔지니어링 모범 사례(버전 관리, 테스트, 문서화)를 데이터 변환에 적용합니다.

---

## 1. dbt 개요

### 1.1 dbt란?

```
┌────────────────────────────────────────────────────────────────┐
│                        dbt 역할                                 │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   ELT 파이프라인에서 T(Transform) 담당                          │
│                                                                │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│   │ Extract  │ →  │   Load   │ →  │Transform │               │
│   │ (Fivetran│    │  (to DW) │    │  (dbt)   │               │
│   │  Airbyte)│    │          │    │          │               │
│   └──────────┘    └──────────┘    └──────────┘               │
│                                                                │
│   dbt 핵심 기능:                                               │
│   - SQL 기반 모델 정의                                         │
│   - 의존성 자동 관리                                           │
│   - 테스트 및 문서화                                           │
│   - Jinja 템플릿 지원                                          │
│   - 버전 관리 (Git)                                            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 dbt Core vs dbt Cloud

| 특성 | dbt Core | dbt Cloud |
|------|----------|-----------|
| **비용** | 무료 (오픈소스) | 유료 (SaaS) |
| **실행** | CLI | Web UI + API |
| **스케줄링** | 외부 도구 필요 (Airflow) | 내장 스케줄러 |
| **IDE** | VS Code 등 | 내장 IDE |
| **협업** | Git 사용 | 내장 협업 기능 |

### 1.3 설치

```bash
# dbt Core 설치
pip install dbt-core

# 데이터베이스별 어댑터 설치
pip install dbt-postgres      # PostgreSQL
pip install dbt-snowflake     # Snowflake
pip install dbt-bigquery      # BigQuery
pip install dbt-redshift      # Redshift
pip install dbt-databricks    # Databricks

# 버전 확인
dbt --version
```

---

## 2. 프로젝트 구조

### 2.1 프로젝트 초기화

```bash
# 새 프로젝트 생성
dbt init my_project
cd my_project

# 프로젝트 구조
my_project/
├── dbt_project.yml          # 프로젝트 설정
├── profiles.yml             # 연결 설정 (~/.dbt/profiles.yml)
├── models/                  # SQL 모델
│   ├── staging/            # 스테이징 모델
│   ├── intermediate/       # 중간 모델
│   └── marts/              # 최종 모델
├── tests/                   # 커스텀 테스트
├── macros/                  # 재사용 매크로
├── seeds/                   # 시드 데이터 (CSV)
├── snapshots/               # SCD 스냅샷
├── analyses/                # 분석 쿼리
└── target/                  # 컴파일된 결과
```

### 2.2 설정 파일

```yaml
# dbt_project.yml
name: 'my_project'
version: '1.0.0'
config-version: 2

profile: 'my_project'

model-paths: ["models"]
analysis-paths: ["analyses"]
test-paths: ["tests"]
seed-paths: ["seeds"]
macro-paths: ["macros"]
snapshot-paths: ["snapshots"]

target-path: "target"
clean-targets:
  - "target"
  - "dbt_packages"

# 모델별 설정
models:
  my_project:
    staging:
      +materialized: view
      +schema: staging
    intermediate:
      +materialized: ephemeral
    marts:
      +materialized: table
      +schema: marts
```

```yaml
# profiles.yml (~/.dbt/profiles.yml)
my_project:
  target: dev
  outputs:
    dev:
      type: postgres
      host: localhost
      port: 5432
      user: postgres
      password: "{{ env_var('DB_PASSWORD') }}"
      dbname: analytics
      schema: dbt_dev
      threads: 4

    prod:
      type: postgres
      host: prod-db.example.com
      port: 5432
      user: "{{ env_var('PROD_USER') }}"
      password: "{{ env_var('PROD_PASSWORD') }}"
      dbname: analytics
      schema: dbt_prod
      threads: 8
```

---

## 3. 모델 (Models)

### 3.1 기본 모델

```sql
-- models/staging/stg_orders.sql
-- 스테이징 모델: 소스 데이터 정제

SELECT
    order_id,
    customer_id,
    CAST(order_date AS DATE) AS order_date,
    CAST(amount AS DECIMAL(10, 2)) AS amount,
    status,
    CURRENT_TIMESTAMP AS loaded_at
FROM {{ source('raw', 'orders') }}
WHERE order_id IS NOT NULL
```

```sql
-- models/staging/stg_customers.sql
SELECT
    customer_id,
    TRIM(first_name) AS first_name,
    TRIM(last_name) AS last_name,
    LOWER(email) AS email,
    created_at
FROM {{ source('raw', 'customers') }}
```

```sql
-- models/marts/core/fct_orders.sql
-- 팩트 테이블: 주문

{{
    config(
        materialized='table',
        unique_key='order_id',
        partition_by={
            'field': 'order_date',
            'data_type': 'date',
            'granularity': 'month'
        }
    )
}}

WITH orders AS (
    SELECT * FROM {{ ref('stg_orders') }}
),

customers AS (
    SELECT * FROM {{ ref('stg_customers') }}
)

SELECT
    o.order_id,
    o.customer_id,
    c.first_name,
    c.last_name,
    c.email,
    o.order_date,
    o.amount,
    o.status,
    -- 파생 컬럼
    EXTRACT(YEAR FROM o.order_date) AS order_year,
    EXTRACT(MONTH FROM o.order_date) AS order_month,
    CASE
        WHEN o.amount > 1000 THEN 'high'
        WHEN o.amount > 100 THEN 'medium'
        ELSE 'low'
    END AS order_tier
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
```

### 3.2 소스 정의

```yaml
# models/staging/_sources.yml
version: 2

sources:
  - name: raw
    description: "원본 데이터 소스"
    database: raw_db
    schema: public
    tables:
      - name: orders
        description: "주문 원본 테이블"
        columns:
          - name: order_id
            description: "주문 고유 ID"
            tests:
              - unique
              - not_null
          - name: customer_id
            description: "고객 ID"
          - name: amount
            description: "주문 금액"

      - name: customers
        description: "고객 원본 테이블"
        freshness:
          warn_after: {count: 12, period: hour}
          error_after: {count: 24, period: hour}
        loaded_at_field: updated_at
```

### 3.3 Materialization 유형

```sql
-- View (기본값): 매번 쿼리 실행
{{ config(materialized='view') }}

-- Table: 물리적 테이블 생성
{{ config(materialized='table') }}

-- Incremental: 증분 처리
{{ config(
    materialized='incremental',
    unique_key='order_id',
    on_schema_change='append_new_columns'
) }}

SELECT *
FROM {{ source('raw', 'orders') }}
{% if is_incremental() %}
WHERE order_date > (SELECT MAX(order_date) FROM {{ this }})
{% endif %}

-- Ephemeral: CTE로 인라인 (테이블 미생성)
{{ config(materialized='ephemeral') }}
```

---

## 4. 테스트

### 4.1 스키마 테스트

```yaml
# models/marts/core/_schema.yml
version: 2

models:
  - name: fct_orders
    description: "주문 팩트 테이블"
    columns:
      - name: order_id
        description: "주문 고유 ID"
        tests:
          - unique
          - not_null

      - name: customer_id
        tests:
          - not_null
          - relationships:
              to: ref('dim_customers')
              field: customer_id

      - name: amount
        tests:
          - not_null
          - dbt_utils.expression_is_true:
              expression: ">= 0"

      - name: status
        tests:
          - accepted_values:
              values: ['pending', 'completed', 'cancelled', 'refunded']
```

### 4.2 커스텀 테스트

```sql
-- tests/assert_positive_amounts.sql
-- 모든 주문 금액이 양수인지 확인

SELECT
    order_id,
    amount
FROM {{ ref('fct_orders') }}
WHERE amount < 0
```

```sql
-- macros/test_row_count_equal.sql
{% test row_count_equal(model, compare_model) %}

WITH model_count AS (
    SELECT COUNT(*) AS cnt FROM {{ model }}
),

compare_count AS (
    SELECT COUNT(*) AS cnt FROM {{ compare_model }}
)

SELECT
    m.cnt AS model_count,
    c.cnt AS compare_count
FROM model_count m
CROSS JOIN compare_count c
WHERE m.cnt != c.cnt

{% endtest %}
```

### 4.3 테스트 실행

```bash
# 모든 테스트 실행
dbt test

# 특정 모델 테스트
dbt test --select fct_orders

# 소스 freshness 테스트
dbt source freshness
```

---

## 5. 문서화

### 5.1 문서 정의

```yaml
# models/marts/core/_schema.yml
version: 2

models:
  - name: fct_orders
    description: |
      ## 주문 팩트 테이블

      이 테이블은 모든 주문 트랜잭션을 포함합니다.

      ### 사용 사례
      - 일별/월별 매출 분석
      - 고객 구매 패턴 분석
      - 재구매율 계산

      ### 갱신 주기
      - 매일 06:00 UTC

    meta:
      owner: "data-team@company.com"
      contains_pii: false

    columns:
      - name: order_id
        description: "주문 고유 식별자 (UUID)"
        meta:
          dimension: true

      - name: amount
        description: "주문 총액 (USD)"
        meta:
          measure: true
          aggregation: sum
```

### 5.2 문서 생성 및 서빙

```bash
# 문서 생성
dbt docs generate

# 문서 서버 실행
dbt docs serve --port 8080

# http://localhost:8080 에서 확인
```

---

## 6. Jinja 템플릿

### 6.1 기본 Jinja 문법

```sql
-- 변수
{% set my_var = 'value' %}
SELECT '{{ my_var }}' AS col

-- 조건문
SELECT
    CASE
        {% if target.name == 'prod' %}
        WHEN amount > 1000 THEN 'high'
        {% else %}
        WHEN amount > 100 THEN 'high'
        {% endif %}
        ELSE 'low'
    END AS tier
FROM orders

-- 반복문
SELECT
    order_id,
    {% for col in ['amount', 'quantity', 'discount'] %}
    SUM({{ col }}) AS total_{{ col }}{% if not loop.last %},{% endif %}
    {% endfor %}
FROM order_items
GROUP BY order_id
```

### 6.2 Macros

```sql
-- macros/generate_schema_name.sql
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- set default_schema = target.schema -%}
    {%- if custom_schema_name is none -%}
        {{ default_schema }}
    {%- else -%}
        {{ default_schema }}_{{ custom_schema_name }}
    {%- endif -%}
{%- endmacro %}


-- macros/cents_to_dollars.sql
{% macro cents_to_dollars(column_name, precision=2) %}
    ROUND({{ column_name }} / 100.0, {{ precision }})
{% endmacro %}


-- macros/limit_data_in_dev.sql
{% macro limit_data_in_dev() %}
    {% if target.name == 'dev' %}
        LIMIT 1000
    {% endif %}
{% endmacro %}
```

```sql
-- 매크로 사용
SELECT
    order_id,
    {{ cents_to_dollars('amount_cents') }} AS amount_dollars
FROM orders
{{ limit_data_in_dev() }}
```

### 6.3 dbt 내장 함수

```sql
-- ref(): 다른 모델 참조
SELECT * FROM {{ ref('stg_orders') }}

-- source(): 소스 테이블 참조
SELECT * FROM {{ source('raw', 'orders') }}

-- this: 현재 모델 참조 (incremental에서 유용)
{% if is_incremental() %}
SELECT MAX(updated_at) FROM {{ this }}
{% endif %}

-- config(): 설정 값 접근
{{ config.get('materialized') }}

-- target: 타겟 환경 정보
{{ target.name }}    -- dev, prod
{{ target.schema }}  -- dbt_dev
{{ target.type }}    -- postgres, snowflake
```

---

## 7. 증분 처리 (Incremental)

### 7.1 기본 증분 모델

```sql
-- models/marts/fct_events.sql
{{
    config(
        materialized='incremental',
        unique_key='event_id',
        incremental_strategy='delete+insert'
    )
}}

SELECT
    event_id,
    user_id,
    event_type,
    event_data,
    created_at
FROM {{ source('raw', 'events') }}

{% if is_incremental() %}
-- 마지막 실행 이후 데이터만 처리
WHERE created_at > (SELECT MAX(created_at) FROM {{ this }})
{% endif %}
```

### 7.2 증분 전략

```sql
-- Append (기본): 새 데이터 추가만
{{ config(
    materialized='incremental',
    incremental_strategy='append'
) }}

-- Delete+Insert: 키 기준 삭제 후 삽입
{{ config(
    materialized='incremental',
    unique_key='id',
    incremental_strategy='delete+insert'
) }}

-- Merge (Snowflake, BigQuery): MERGE 문 사용
{{ config(
    materialized='incremental',
    unique_key='id',
    incremental_strategy='merge',
    merge_update_columns=['name', 'amount', 'updated_at']
) }}
```

---

## 8. 실행 명령

### 8.1 기본 명령

```bash
# 연결 테스트
dbt debug

# 모든 모델 실행
dbt run

# 특정 모델만 실행
dbt run --select fct_orders
dbt run --select staging.*
dbt run --select +fct_orders+  # 의존성 포함

# 테스트 실행
dbt test

# 빌드 (run + test)
dbt build

# Seed 데이터 로드
dbt seed

# 컴파일만 (실행 안 함)
dbt compile

# 정리
dbt clean
```

### 8.2 선택자 (Selectors)

```bash
# 모델 이름으로
dbt run --select my_model

# 경로로
dbt run --select models/staging/*

# 태그로
dbt run --select tag:daily

# 상위 의존성 포함
dbt run --select +my_model

# 하위 의존성 포함
dbt run --select my_model+

# 양방향
dbt run --select +my_model+

# 특정 모델 제외
dbt run --exclude my_model
```

---

## 연습 문제

### 문제 1: 스테이징 모델
원본 products 테이블에서 stg_products 모델을 생성하세요. 가격을 달러로 변환하고 NULL 값을 처리하세요.

### 문제 2: 증분 모델
일별 판매 집계 테이블을 증분으로 처리하는 모델을 작성하세요.

### 문제 3: 테스트 작성
fct_sales 모델에 대한 테스트를 작성하세요 (unique, not_null, 금액 양수 확인).

---

## 요약

| 개념 | 설명 |
|------|------|
| **Model** | SQL 기반 데이터 변환 정의 |
| **Source** | 원본 데이터 참조 |
| **ref()** | 모델 간 참조 (의존성 자동 관리) |
| **Test** | 데이터 품질 검증 |
| **Materialization** | view, table, incremental, ephemeral |
| **Macro** | 재사용 가능한 SQL 템플릿 |

---

## 참고 자료

- [dbt Documentation](https://docs.getdbt.com/)
- [dbt Learn](https://courses.getdbt.com/)
- [dbt Best Practices](https://docs.getdbt.com/guides/best-practices)
