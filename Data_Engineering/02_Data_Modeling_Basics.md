# 데이터 모델링 기초

## 개요

데이터 모델링은 데이터의 구조, 관계, 제약 조건을 정의하는 과정입니다. 데이터 웨어하우스와 분석 시스템에서는 차원 모델링(Dimensional Modeling)이 널리 사용됩니다.

---

## 1. 차원 모델링 (Dimensional Modeling)

### 1.1 차원 모델링 개념

차원 모델링은 비즈니스 프로세스를 **팩트(Fact)**와 **디멘전(Dimension)**으로 분리하여 모델링하는 기법입니다.

```
┌──────────────────────────────────────────────────────────────┐
│                    차원 모델링 구조                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐                                           │
│   │  Dimension   │  WHO, WHAT, WHERE, WHEN, HOW              │
│   │   (차원)     │  - Customer (누가)                         │
│   │              │  - Product (무엇을)                        │
│   │              │  - Location (어디서)                       │
│   │              │  - Time (언제)                             │
│   └──────┬───────┘                                           │
│          │                                                   │
│          ↓                                                   │
│   ┌──────────────┐                                           │
│   │    Fact      │  MEASURES (측정값)                         │
│   │   (팩트)     │  - Sales Amount (판매금액)                  │
│   │              │  - Quantity (수량)                         │
│   │              │  - Profit (이익)                           │
│   └──────────────┘                                           │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 팩트 vs 디멘전

| 구분 | 팩트 테이블 | 디멘전 테이블 |
|------|------------|--------------|
| **내용** | 측정 가능한 수치 데이터 | 설명적 속성 데이터 |
| **예시** | 판매금액, 수량, 이익 | 고객명, 상품명, 날짜 |
| **레코드 수** | 매우 많음 (수억 건) | 상대적으로 적음 |
| **변경 빈도** | 계속 추가됨 | 가끔 변경됨 |
| **분석 역할** | 집계 대상 | 필터/그룹 기준 |

---

## 2. 스타 스키마 (Star Schema)

### 2.1 스타 스키마 구조

스타 스키마는 중앙에 팩트 테이블이 있고, 주변에 디멘전 테이블이 연결된 형태입니다.

```
                    ┌─────────────────┐
                    │   dim_customer  │
                    │  - customer_sk  │
                    │  - customer_id  │
                    │  - name         │
                    │  - email        │
                    └────────┬────────┘
                             │
┌─────────────────┐          │          ┌─────────────────┐
│   dim_product   │          │          │    dim_date     │
│  - product_sk   │          │          │  - date_sk      │
│  - product_id   │          │          │  - full_date    │
│  - name         │          ↓          │  - year         │
│  - category     │   ┌─────────────┐   │  - quarter      │
│  - price        │───│ fact_sales  │───│  - month        │
└─────────────────┘   │ - date_sk   │   └─────────────────┘
                      │ - customer_sk│
                      │ - product_sk │
                      │ - store_sk   │
┌─────────────────┐   │ - quantity   │
│   dim_store     │   │ - amount     │
│  - store_sk     │   │ - discount   │
│  - store_id     │───└─────────────┘
│  - store_name   │
│  - city         │
└─────────────────┘
```

### 2.2 스타 스키마 SQL 구현

```sql
-- 1. 디멘전 테이블 생성

-- 날짜 디멘전
CREATE TABLE dim_date (
    date_sk         INT PRIMARY KEY,           -- Surrogate Key
    full_date       DATE NOT NULL,
    year            INT NOT NULL,
    quarter         INT NOT NULL,
    month           INT NOT NULL,
    month_name      VARCHAR(20) NOT NULL,
    week            INT NOT NULL,
    day_of_week     INT NOT NULL,
    day_name        VARCHAR(20) NOT NULL,
    is_weekend      BOOLEAN NOT NULL,
    is_holiday      BOOLEAN DEFAULT FALSE
);

-- 고객 디멘전
CREATE TABLE dim_customer (
    customer_sk     INT PRIMARY KEY,           -- Surrogate Key
    customer_id     VARCHAR(50) NOT NULL,      -- Natural Key
    first_name      VARCHAR(100) NOT NULL,
    last_name       VARCHAR(100) NOT NULL,
    email           VARCHAR(200),
    phone           VARCHAR(50),
    city            VARCHAR(100),
    country         VARCHAR(100),
    customer_segment VARCHAR(50),              -- Gold, Silver, Bronze
    created_at      DATE NOT NULL,
    -- SCD Type 2 지원 컬럼
    effective_date  DATE NOT NULL,
    end_date        DATE,
    is_current      BOOLEAN DEFAULT TRUE
);

-- 상품 디멘전
CREATE TABLE dim_product (
    product_sk      INT PRIMARY KEY,           -- Surrogate Key
    product_id      VARCHAR(50) NOT NULL,      -- Natural Key
    product_name    VARCHAR(200) NOT NULL,
    category        VARCHAR(100),
    subcategory     VARCHAR(100),
    brand           VARCHAR(100),
    unit_price      DECIMAL(10, 2),
    cost_price      DECIMAL(10, 2),
    -- SCD Type 2 지원 컬럼
    effective_date  DATE NOT NULL,
    end_date        DATE,
    is_current      BOOLEAN DEFAULT TRUE
);

-- 매장 디멘전
CREATE TABLE dim_store (
    store_sk        INT PRIMARY KEY,           -- Surrogate Key
    store_id        VARCHAR(50) NOT NULL,      -- Natural Key
    store_name      VARCHAR(200) NOT NULL,
    store_type      VARCHAR(50),               -- Online, Retail
    city            VARCHAR(100),
    state           VARCHAR(100),
    country         VARCHAR(100),
    region          VARCHAR(50),
    opened_date     DATE
);


-- 2. 팩트 테이블 생성

CREATE TABLE fact_sales (
    sales_sk        BIGINT PRIMARY KEY,        -- Surrogate Key
    -- 디멘전 외래 키
    date_sk         INT NOT NULL REFERENCES dim_date(date_sk),
    customer_sk     INT NOT NULL REFERENCES dim_customer(customer_sk),
    product_sk      INT NOT NULL REFERENCES dim_product(product_sk),
    store_sk        INT NOT NULL REFERENCES dim_store(store_sk),
    -- 측정값 (Measures)
    quantity        INT NOT NULL,
    unit_price      DECIMAL(10, 2) NOT NULL,
    discount_amount DECIMAL(10, 2) DEFAULT 0,
    sales_amount    DECIMAL(12, 2) NOT NULL,   -- quantity * unit_price - discount
    cost_amount     DECIMAL(12, 2),
    profit_amount   DECIMAL(12, 2),            -- sales_amount - cost_amount
    -- 메타 데이터
    transaction_id  VARCHAR(50),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스 생성 (쿼리 성능 향상)
CREATE INDEX idx_fact_sales_date ON fact_sales(date_sk);
CREATE INDEX idx_fact_sales_customer ON fact_sales(customer_sk);
CREATE INDEX idx_fact_sales_product ON fact_sales(product_sk);
CREATE INDEX idx_fact_sales_store ON fact_sales(store_sk);
```

### 2.3 스타 스키마 쿼리 예시

```sql
-- 월별, 카테고리별 매출 집계
SELECT
    d.year,
    d.month,
    d.month_name,
    p.category,
    SUM(f.sales_amount) AS total_sales,
    SUM(f.quantity) AS total_quantity,
    SUM(f.profit_amount) AS total_profit,
    COUNT(DISTINCT f.customer_sk) AS unique_customers
FROM fact_sales f
JOIN dim_date d ON f.date_sk = d.date_sk
JOIN dim_product p ON f.product_sk = p.product_sk
WHERE d.year = 2024
GROUP BY d.year, d.month, d.month_name, p.category
ORDER BY d.year, d.month, total_sales DESC;


-- 지역별 상위 10개 상품
SELECT
    s.region,
    p.product_name,
    SUM(f.sales_amount) AS total_sales,
    RANK() OVER (PARTITION BY s.region ORDER BY SUM(f.sales_amount) DESC) AS rank
FROM fact_sales f
JOIN dim_store s ON f.store_sk = s.store_sk
JOIN dim_product p ON f.product_sk = p.product_sk
GROUP BY s.region, p.product_name
QUALIFY rank <= 10;


-- 고객 세그먼트별 구매 패턴
SELECT
    c.customer_segment,
    COUNT(DISTINCT f.customer_sk) AS customer_count,
    AVG(f.sales_amount) AS avg_order_value,
    SUM(f.sales_amount) / COUNT(DISTINCT f.customer_sk) AS revenue_per_customer
FROM fact_sales f
JOIN dim_customer c ON f.customer_sk = c.customer_sk
WHERE c.is_current = TRUE
GROUP BY c.customer_segment
ORDER BY revenue_per_customer DESC;
```

---

## 3. 스노우플레이크 스키마 (Snowflake Schema)

### 3.1 스노우플레이크 스키마 구조

디멘전 테이블을 정규화하여 중복을 제거한 형태입니다.

```
┌──────────────┐
│ dim_category │
│ - category_sk│
│ - category   │
└──────┬───────┘
       │
       ↓
┌──────────────┐     ┌──────────────┐
│dim_subcategory│    │  dim_brand   │
│-subcategory_sk│    │ - brand_sk   │
│- category_sk │     │ - brand_name │
│- subcategory │     └──────┬───────┘
└──────┬───────┘            │
       │                    │
       └──────────┬─────────┘
                  ↓
          ┌─────────────┐
          │ dim_product │
          │- product_sk │
          │-subcategory_sk
          │- brand_sk   │────→ ┌─────────────┐
          │- product_name      │ fact_sales  │
          └─────────────┘      └─────────────┘
```

### 3.2 스노우플레이크 vs 스타 스키마

| 특성 | 스타 스키마 | 스노우플레이크 스키마 |
|------|------------|---------------------|
| **정규화** | 비정규화 | 정규화 |
| **저장 공간** | 더 많음 | 더 적음 |
| **쿼리 성능** | 더 빠름 (조인 적음) | 더 느림 (조인 많음) |
| **유지보수** | 중복 관리 필요 | 관리 용이 |
| **복잡성** | 단순 | 복잡 |
| **권장 사용** | OLAP, 분석 | 저장 공간 제한 시 |

---

## 4. 팩트 테이블 유형

### 4.1 트랜잭션 팩트 (Transaction Fact)

개별 트랜잭션을 기록합니다. 가장 일반적인 형태입니다.

```sql
-- 트랜잭션 팩트 예시: 개별 주문
CREATE TABLE fact_order_line (
    order_line_sk   BIGINT PRIMARY KEY,
    date_sk         INT NOT NULL,
    customer_sk     INT NOT NULL,
    product_sk      INT NOT NULL,
    order_id        VARCHAR(50) NOT NULL,
    line_number     INT NOT NULL,
    quantity        INT NOT NULL,
    unit_price      DECIMAL(10, 2) NOT NULL,
    line_amount     DECIMAL(12, 2) NOT NULL
);
```

### 4.2 주기적 스냅샷 팩트 (Periodic Snapshot Fact)

일정 기간의 집계 데이터를 기록합니다.

```sql
-- 주기적 스냅샷: 일일 재고 현황
CREATE TABLE fact_daily_inventory (
    inventory_sk    BIGINT PRIMARY KEY,
    date_sk         INT NOT NULL,
    product_sk      INT NOT NULL,
    warehouse_sk    INT NOT NULL,
    -- 스냅샷 측정값
    quantity_on_hand INT NOT NULL,
    quantity_reserved INT DEFAULT 0,
    quantity_available INT NOT NULL,
    days_of_supply  INT,
    inventory_value DECIMAL(12, 2)
);


-- 일일 계정 잔액 스냅샷
CREATE TABLE fact_daily_account_balance (
    balance_sk      BIGINT PRIMARY KEY,
    date_sk         INT NOT NULL,
    account_sk      INT NOT NULL,
    customer_sk     INT NOT NULL,
    opening_balance DECIMAL(15, 2) NOT NULL,
    total_credits   DECIMAL(15, 2) DEFAULT 0,
    total_debits    DECIMAL(15, 2) DEFAULT 0,
    closing_balance DECIMAL(15, 2) NOT NULL
);
```

### 4.3 누적 스냅샷 팩트 (Accumulating Snapshot Fact)

프로세스의 시작부터 종료까지 추적합니다.

```sql
-- 누적 스냅샷: 주문 처리 프로세스
CREATE TABLE fact_order_fulfillment (
    order_fulfillment_sk BIGINT PRIMARY KEY,
    order_id        VARCHAR(50) UNIQUE NOT NULL,

    -- 마일스톤 날짜 (각 단계 완료 시점)
    order_date_sk       INT NOT NULL,
    payment_date_sk     INT,
    ship_date_sk        INT,
    delivery_date_sk    INT,

    -- 디멘전 외래 키
    customer_sk     INT NOT NULL,
    product_sk      INT NOT NULL,
    warehouse_sk    INT,
    carrier_sk      INT,

    -- 측정값
    order_amount    DECIMAL(12, 2) NOT NULL,
    shipping_cost   DECIMAL(10, 2),

    -- 계산된 측정값 (리드 타임)
    days_to_payment     INT,  -- order -> payment
    days_to_ship        INT,  -- payment -> ship
    days_to_delivery    INT,  -- ship -> delivery
    total_lead_time     INT   -- order -> delivery
);
```

---

## 5. SCD (Slowly Changing Dimensions)

### 5.1 SCD 유형 개요

| 유형 | 설명 | 히스토리 | 사용 사례 |
|------|------|----------|----------|
| **Type 0** | 변경 안 함 | 없음 | 고정 속성 (생년월일) |
| **Type 1** | 덮어쓰기 | 없음 | 오류 수정, 히스토리 불필요 |
| **Type 2** | 새 행 추가 | 전체 보관 | 가격 변경, 주소 변경 |
| **Type 3** | 컬럼 추가 | 이전 값만 | 제한적 히스토리 필요 |
| **Type 4** | 히스토리 테이블 분리 | 전체 보관 | 자주 변경되는 속성 |

### 5.2 SCD Type 1: 덮어쓰기

```sql
-- SCD Type 1: 기존 값 덮어쓰기 (히스토리 없음)
UPDATE dim_customer
SET
    email = 'new_email@example.com',
    phone = '010-1234-5678'
WHERE customer_id = 'C001';
```

### 5.3 SCD Type 2: 새 행 추가

```python
# SCD Type 2 구현 예시
import pandas as pd
from datetime import date

def scd_type2_update(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    natural_key: str,
    tracked_columns: list[str]
) -> pd.DataFrame:
    """SCD Type 2 업데이트 로직"""

    today = date.today()
    result_rows = []

    for _, source_row in source_df.iterrows():
        # 현재 활성 레코드 찾기
        current_mask = (
            (target_df[natural_key] == source_row[natural_key]) &
            (target_df['is_current'] == True)
        )
        current_record = target_df[current_mask]

        if current_record.empty:
            # 신규 레코드
            new_row = source_row.copy()
            new_row['effective_date'] = today
            new_row['end_date'] = None
            new_row['is_current'] = True
            result_rows.append(new_row)
        else:
            # 기존 레코드 비교
            current_row = current_record.iloc[0]
            has_changes = False

            for col in tracked_columns:
                if current_row[col] != source_row[col]:
                    has_changes = True
                    break

            if has_changes:
                # 기존 레코드 만료
                target_df.loc[current_mask, 'end_date'] = today
                target_df.loc[current_mask, 'is_current'] = False

                # 새 레코드 추가
                new_row = source_row.copy()
                new_row['effective_date'] = today
                new_row['end_date'] = None
                new_row['is_current'] = True
                result_rows.append(new_row)

    # 새 레코드 추가
    if result_rows:
        new_records = pd.DataFrame(result_rows)
        target_df = pd.concat([target_df, new_records], ignore_index=True)

    return target_df


# 사용 예시
"""
-- SQL로 SCD Type 2 구현
-- 1. 변경된 레코드 만료
UPDATE dim_customer
SET
    end_date = CURRENT_DATE,
    is_current = FALSE
WHERE customer_id IN (
    SELECT customer_id FROM staging_customer
    WHERE customer_id IN (SELECT customer_id FROM dim_customer WHERE is_current = TRUE)
    AND (email != (SELECT email FROM dim_customer d WHERE d.customer_id = staging_customer.customer_id AND d.is_current = TRUE)
         OR phone != (SELECT phone FROM dim_customer d WHERE d.customer_id = staging_customer.customer_id AND d.is_current = TRUE))
);

-- 2. 새 레코드 삽입
INSERT INTO dim_customer (customer_id, email, phone, effective_date, end_date, is_current)
SELECT
    customer_id,
    email,
    phone,
    CURRENT_DATE,
    NULL,
    TRUE
FROM staging_customer
WHERE customer_id IN (
    SELECT customer_id FROM dim_customer WHERE is_current = FALSE AND end_date = CURRENT_DATE
);
"""
```

### 5.4 SCD Type 2 SQL 구현

```sql
-- MERGE를 사용한 SCD Type 2 (PostgreSQL 15+)
WITH changes AS (
    -- 변경된 레코드 식별
    SELECT
        s.customer_id,
        s.email,
        s.phone,
        s.city
    FROM staging_customer s
    JOIN dim_customer d ON s.customer_id = d.customer_id AND d.is_current = TRUE
    WHERE s.email != d.email OR s.phone != d.phone OR s.city != d.city
)
-- 1. 기존 레코드 만료
UPDATE dim_customer
SET
    end_date = CURRENT_DATE - INTERVAL '1 day',
    is_current = FALSE
FROM changes
WHERE dim_customer.customer_id = changes.customer_id
  AND dim_customer.is_current = TRUE;

-- 2. 새 레코드 삽입
INSERT INTO dim_customer (
    customer_id, email, phone, city,
    effective_date, end_date, is_current
)
SELECT
    customer_id, email, phone, city,
    CURRENT_DATE, NULL, TRUE
FROM staging_customer
WHERE customer_id IN (
    SELECT customer_id FROM dim_customer
    WHERE end_date = CURRENT_DATE - INTERVAL '1 day'
);
```

---

## 6. 디멘전 테이블 설계 패턴

### 6.1 날짜 디멘전 생성

```python
import pandas as pd
from datetime import date, timedelta

def generate_date_dimension(start_date: str, end_date: str) -> pd.DataFrame:
    """날짜 디멘전 테이블 생성"""

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    records = []
    for i, d in enumerate(date_range):
        record = {
            'date_sk': int(d.strftime('%Y%m%d')),
            'full_date': d.date(),
            'year': d.year,
            'quarter': (d.month - 1) // 3 + 1,
            'month': d.month,
            'month_name': d.strftime('%B'),
            'week': d.isocalendar()[1],
            'day_of_week': d.weekday() + 1,  # 1=Monday
            'day_name': d.strftime('%A'),
            'day_of_month': d.day,
            'day_of_year': d.timetuple().tm_yday,
            'is_weekend': d.weekday() >= 5,
            'is_month_start': d.day == 1,
            'is_month_end': (d + timedelta(days=1)).day == 1,
            'fiscal_year': d.year if d.month >= 4 else d.year - 1,  # 4월 시작 회계연도
            'fiscal_quarter': ((d.month - 4) % 12) // 3 + 1
        }
        records.append(record)

    return pd.DataFrame(records)


# 사용 예시
date_dim = generate_date_dimension('2020-01-01', '2030-12-31')
print(date_dim.head())
```

### 6.2 정크 디멘전 (Junk Dimension)

여러 저-카디널리티 플래그/상태를 하나의 디멘전으로 통합합니다.

```sql
-- 정크 디멘전: 주문 상태 플래그들
CREATE TABLE dim_order_flags (
    order_flags_sk  INT PRIMARY KEY,
    is_gift_wrapped BOOLEAN,
    is_expedited    BOOLEAN,
    is_return       BOOLEAN,
    payment_method  VARCHAR(20),  -- Credit, Debit, Cash, PayPal
    order_channel   VARCHAR(20)   -- Web, Mobile, Store, Phone
);

-- 모든 조합 미리 생성
INSERT INTO dim_order_flags (order_flags_sk, is_gift_wrapped, is_expedited, is_return, payment_method, order_channel)
SELECT
    ROW_NUMBER() OVER () as order_flags_sk,
    gift, expedited, return_flag, payment, channel
FROM
    (VALUES (TRUE), (FALSE)) AS gift(gift),
    (VALUES (TRUE), (FALSE)) AS expedited(expedited),
    (VALUES (TRUE), (FALSE)) AS return_flag(return_flag),
    (VALUES ('Credit'), ('Debit'), ('Cash'), ('PayPal')) AS payment(payment),
    (VALUES ('Web'), ('Mobile'), ('Store'), ('Phone')) AS channel(channel);
```

---

## 연습 문제

### 문제 1: 스타 스키마 설계
온라인 서점의 판매 분석을 위한 스타 스키마를 설계하세요. 필요한 팩트 테이블과 디멘전 테이블을 정의하세요.

### 문제 2: SCD Type 2
고객의 등급(Bronze, Silver, Gold)이 변경될 때 히스토리를 보관하는 SCD Type 2 SQL을 작성하세요.

---

## 요약

| 개념 | 설명 |
|------|------|
| **차원 모델링** | 팩트와 디멘전으로 데이터 구조화 |
| **스타 스키마** | 비정규화된 디멘전, 빠른 쿼리 |
| **스노우플레이크** | 정규화된 디멘전, 저장 공간 절약 |
| **팩트 테이블** | 측정 가능한 수치 데이터 저장 |
| **디멘전 테이블** | 설명적 속성 데이터 저장 |
| **SCD** | 디멘전 변경 이력 관리 전략 |

---

## 참고 자료

- [The Data Warehouse Toolkit (Kimball)](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/books/data-warehouse-dw-toolkit/)
- [Dimensional Modeling Techniques](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/)
