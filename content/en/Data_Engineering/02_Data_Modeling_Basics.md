# Data Modeling Basics

## Introduction

Data modeling is the process of defining the structure, relationships, and constraints of data. In data warehouses and analytics systems, dimensional modeling is widely used.

---

## 1. Dimensional Modeling

### 1.1 Dimensional Modeling Concept

Dimensional modeling is a technique that separates business processes into **Facts** and **Dimensions**.

```
┌──────────────────────────────────────────────────────────────┐
│                  Dimensional Modeling Structure               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐                                           │
│   │  Dimension   │  WHO, WHAT, WHERE, WHEN, HOW              │
│   │              │  - Customer (who)                         │
│   │              │  - Product (what)                         │
│   │              │  - Location (where)                       │
│   │              │  - Time (when)                            │
│   └──────┬───────┘                                           │
│          │                                                   │
│          ↓                                                   │
│   ┌──────────────┐                                           │
│   │    Fact      │  MEASURES                                 │
│   │              │  - Sales Amount                           │
│   │              │  - Quantity                               │
│   │              │  - Profit                                 │
│   └──────────────┘                                           │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 Fact vs Dimension

| Aspect | Fact Table | Dimension Table |
|------|------------|--------------|
| **Content** | Measurable numeric data | Descriptive attribute data |
| **Example** | Sales amount, quantity, profit | Customer name, product name, date |
| **Record Count** | Very high (hundreds of millions) | Relatively low |
| **Change Frequency** | Continuously added | Occasionally changed |
| **Analysis Role** | Aggregation target | Filter/group criteria |

---

## 2. Star Schema

### 2.1 Star Schema Structure

The star schema has a fact table at the center with dimension tables connected around it.

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

### 2.2 Star Schema SQL Implementation

```sql
-- 1. Create dimension tables

-- Date dimension
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

-- Customer dimension
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
    -- SCD Type 2 support columns
    effective_date  DATE NOT NULL,
    end_date        DATE,
    is_current      BOOLEAN DEFAULT TRUE
);

-- Product dimension
CREATE TABLE dim_product (
    product_sk      INT PRIMARY KEY,           -- Surrogate Key
    product_id      VARCHAR(50) NOT NULL,      -- Natural Key
    product_name    VARCHAR(200) NOT NULL,
    category        VARCHAR(100),
    subcategory     VARCHAR(100),
    brand           VARCHAR(100),
    unit_price      DECIMAL(10, 2),
    cost_price      DECIMAL(10, 2),
    -- SCD Type 2 support columns
    effective_date  DATE NOT NULL,
    end_date        DATE,
    is_current      BOOLEAN DEFAULT TRUE
);

-- Store dimension
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


-- 2. Create fact table

CREATE TABLE fact_sales (
    sales_sk        BIGINT PRIMARY KEY,        -- Surrogate Key
    -- Dimension foreign keys
    date_sk         INT NOT NULL REFERENCES dim_date(date_sk),
    customer_sk     INT NOT NULL REFERENCES dim_customer(customer_sk),
    product_sk      INT NOT NULL REFERENCES dim_product(product_sk),
    store_sk        INT NOT NULL REFERENCES dim_store(store_sk),
    -- Measures
    quantity        INT NOT NULL,
    unit_price      DECIMAL(10, 2) NOT NULL,
    discount_amount DECIMAL(10, 2) DEFAULT 0,
    sales_amount    DECIMAL(12, 2) NOT NULL,   -- quantity * unit_price - discount
    cost_amount     DECIMAL(12, 2),
    profit_amount   DECIMAL(12, 2),            -- sales_amount - cost_amount
    -- Metadata
    transaction_id  VARCHAR(50),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes (improve query performance)
CREATE INDEX idx_fact_sales_date ON fact_sales(date_sk);
CREATE INDEX idx_fact_sales_customer ON fact_sales(customer_sk);
CREATE INDEX idx_fact_sales_product ON fact_sales(product_sk);
CREATE INDEX idx_fact_sales_store ON fact_sales(store_sk);
```

### 2.3 Star Schema Query Examples

```sql
-- Monthly sales by category
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


-- Top 10 products by region
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


-- Purchase patterns by customer segment
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

## 3. Snowflake Schema

### 3.1 Snowflake Schema Structure

Normalized dimension tables to eliminate redundancy.

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

### 3.2 Snowflake vs Star Schema

| Characteristic | Star Schema | Snowflake Schema |
|------|------------|---------------------|
| **Normalization** | Denormalized | Normalized |
| **Storage Space** | More | Less |
| **Query Performance** | Faster (fewer joins) | Slower (more joins) |
| **Maintenance** | Redundancy management needed | Easier to maintain |
| **Complexity** | Simple | Complex |
| **Recommended Use** | OLAP, analytics | Storage space constraints |

---

## 4. Fact Table Types

### 4.1 Transaction Fact

Records individual transactions. The most common type.

```sql
-- Transaction fact example: Individual orders
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

### 4.2 Periodic Snapshot Fact

Records aggregated data for a specific period.

```sql
-- Periodic snapshot: Daily inventory status
CREATE TABLE fact_daily_inventory (
    inventory_sk    BIGINT PRIMARY KEY,
    date_sk         INT NOT NULL,
    product_sk      INT NOT NULL,
    warehouse_sk    INT NOT NULL,
    -- Snapshot measures
    quantity_on_hand INT NOT NULL,
    quantity_reserved INT DEFAULT 0,
    quantity_available INT NOT NULL,
    days_of_supply  INT,
    inventory_value DECIMAL(12, 2)
);


-- Daily account balance snapshot
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

### 4.3 Accumulating Snapshot Fact

Tracks a process from start to completion.

```sql
-- Accumulating snapshot: Order fulfillment process
CREATE TABLE fact_order_fulfillment (
    order_fulfillment_sk BIGINT PRIMARY KEY,
    order_id        VARCHAR(50) UNIQUE NOT NULL,

    -- Milestone dates (completion time of each stage)
    order_date_sk       INT NOT NULL,
    payment_date_sk     INT,
    ship_date_sk        INT,
    delivery_date_sk    INT,

    -- Dimension foreign keys
    customer_sk     INT NOT NULL,
    product_sk      INT NOT NULL,
    warehouse_sk    INT,
    carrier_sk      INT,

    -- Measures
    order_amount    DECIMAL(12, 2) NOT NULL,
    shipping_cost   DECIMAL(10, 2),

    -- Calculated measures (lead times)
    days_to_payment     INT,  -- order -> payment
    days_to_ship        INT,  -- payment -> ship
    days_to_delivery    INT,  -- ship -> delivery
    total_lead_time     INT   -- order -> delivery
);
```

---

## 5. SCD (Slowly Changing Dimensions)

### 5.1 SCD Type Overview

| Type | Description | History | Use Cases |
|------|------|----------|----------|
| **Type 0** | No changes | None | Fixed attributes (birth date) |
| **Type 1** | Overwrite | None | Error correction, no history needed |
| **Type 2** | Add new row | Full preservation | Price changes, address changes |
| **Type 3** | Add column | Previous value only | Limited history needed |
| **Type 4** | Separate history table | Full preservation | Frequently changing attributes |

### 5.2 SCD Type 1: Overwrite

```sql
-- SCD Type 1: Overwrite existing value (no history)
UPDATE dim_customer
SET
    email = 'new_email@example.com',
    phone = '010-1234-5678'
WHERE customer_id = 'C001';
```

### 5.3 SCD Type 2: Add New Row

```python
# SCD Type 2 implementation example
import pandas as pd
from datetime import date

def scd_type2_update(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    natural_key: str,
    tracked_columns: list[str]
) -> pd.DataFrame:
    """SCD Type 2 update logic"""

    today = date.today()
    result_rows = []

    for _, source_row in source_df.iterrows():
        # Find current active record
        current_mask = (
            (target_df[natural_key] == source_row[natural_key]) &
            (target_df['is_current'] == True)
        )
        current_record = target_df[current_mask]

        if current_record.empty:
            # New record
            new_row = source_row.copy()
            new_row['effective_date'] = today
            new_row['end_date'] = None
            new_row['is_current'] = True
            result_rows.append(new_row)
        else:
            # Compare with existing record
            current_row = current_record.iloc[0]
            has_changes = False

            for col in tracked_columns:
                if current_row[col] != source_row[col]:
                    has_changes = True
                    break

            if has_changes:
                # Expire existing record
                target_df.loc[current_mask, 'end_date'] = today
                target_df.loc[current_mask, 'is_current'] = False

                # Add new record
                new_row = source_row.copy()
                new_row['effective_date'] = today
                new_row['end_date'] = None
                new_row['is_current'] = True
                result_rows.append(new_row)

    # Add new records
    if result_rows:
        new_records = pd.DataFrame(result_rows)
        target_df = pd.concat([target_df, new_records], ignore_index=True)

    return target_df


# Usage example
"""
-- SQL implementation of SCD Type 2
-- 1. Expire changed records
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

-- 2. Insert new records
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

### 5.4 SCD Type 2 SQL Implementation

```sql
-- SCD Type 2 using MERGE (PostgreSQL 15+)
WITH changes AS (
    -- Identify changed records
    SELECT
        s.customer_id,
        s.email,
        s.phone,
        s.city
    FROM staging_customer s
    JOIN dim_customer d ON s.customer_id = d.customer_id AND d.is_current = TRUE
    WHERE s.email != d.email OR s.phone != d.phone OR s.city != d.city
)
-- 1. Expire existing records
UPDATE dim_customer
SET
    end_date = CURRENT_DATE - INTERVAL '1 day',
    is_current = FALSE
FROM changes
WHERE dim_customer.customer_id = changes.customer_id
  AND dim_customer.is_current = TRUE;

-- 2. Insert new records
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

## 6. Dimension Table Design Patterns

### 6.1 Date Dimension Generation

```python
import pandas as pd
from datetime import date, timedelta

def generate_date_dimension(start_date: str, end_date: str) -> pd.DataFrame:
    """Generate date dimension table"""

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
            'fiscal_year': d.year if d.month >= 4 else d.year - 1,  # Fiscal year starts April
            'fiscal_quarter': ((d.month - 4) % 12) // 3 + 1
        }
        records.append(record)

    return pd.DataFrame(records)


# Usage example
date_dim = generate_date_dimension('2020-01-01', '2030-12-31')
print(date_dim.head())
```

### 6.2 Junk Dimension

Consolidate multiple low-cardinality flags/statuses into one dimension.

```sql
-- Junk dimension: Order status flags
CREATE TABLE dim_order_flags (
    order_flags_sk  INT PRIMARY KEY,
    is_gift_wrapped BOOLEAN,
    is_expedited    BOOLEAN,
    is_return       BOOLEAN,
    payment_method  VARCHAR(20),  -- Credit, Debit, Cash, PayPal
    order_channel   VARCHAR(20)   -- Web, Mobile, Store, Phone
);

-- Pre-generate all combinations
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

## Practice Problems

### Problem 1: Star Schema Design
Design a star schema for sales analysis of an online bookstore. Define the necessary fact and dimension tables.

### Problem 2: SCD Type 2
Write SQL for SCD Type 2 that preserves history when a customer's tier (Bronze, Silver, Gold) changes.

---

## Summary

| Concept | Description |
|------|------|
| **Dimensional Modeling** | Structure data with facts and dimensions |
| **Star Schema** | Denormalized dimensions, fast queries |
| **Snowflake** | Normalized dimensions, storage savings |
| **Fact Table** | Store measurable numeric data |
| **Dimension Table** | Store descriptive attribute data |
| **SCD** | Strategy for managing dimension change history |

---

## References

- [The Data Warehouse Toolkit (Kimball)](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/books/data-warehouse-dw-toolkit/)
- [Dimensional Modeling Techniques](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/)
