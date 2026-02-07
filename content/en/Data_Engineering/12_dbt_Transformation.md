# dbt Transformation Tool

## Overview

dbt (data build tool) is a SQL-based data transformation tool. It handles the Transform step in the ELT pattern and applies software engineering best practices (version control, testing, documentation) to data transformations.

---

## 1. dbt Overview

### 1.1 What is dbt?

```
┌────────────────────────────────────────────────────────────────┐
│                        dbt Role                                 │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Handles T(Transform) in ELT Pipeline                         │
│                                                                │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│   │ Extract  │ →  │   Load   │ →  │Transform │               │
│   │ (Fivetran│    │  (to DW) │    │  (dbt)   │               │
│   │  Airbyte)│    │          │    │          │               │
│   └──────────┘    └──────────┘    └──────────┘               │
│                                                                │
│   Core dbt Features:                                           │
│   - SQL-based model definition                                 │
│   - Automatic dependency management                            │
│   - Testing and documentation                                  │
│   - Jinja template support                                     │
│   - Version control (Git)                                      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 dbt Core vs dbt Cloud

| Feature | dbt Core | dbt Cloud |
|------|----------|-----------|
| **Cost** | Free (Open Source) | Paid (SaaS) |
| **Execution** | CLI | Web UI + API |
| **Scheduling** | External tool required (Airflow) | Built-in scheduler |
| **IDE** | VS Code etc. | Built-in IDE |
| **Collaboration** | Git-based | Built-in collaboration |

### 1.3 Installation

```bash
# Install dbt Core
pip install dbt-core

# Install database-specific adapters
pip install dbt-postgres      # PostgreSQL
pip install dbt-snowflake     # Snowflake
pip install dbt-bigquery      # BigQuery
pip install dbt-redshift      # Redshift
pip install dbt-databricks    # Databricks

# Check version
dbt --version
```

---

## 2. Project Structure

### 2.1 Project Initialization

```bash
# Create new project
dbt init my_project
cd my_project

# Project structure
my_project/
├── dbt_project.yml          # Project configuration
├── profiles.yml             # Connection settings (~/.dbt/profiles.yml)
├── models/                  # SQL models
│   ├── staging/            # Staging models
│   ├── intermediate/       # Intermediate models
│   └── marts/              # Final models
├── tests/                   # Custom tests
├── macros/                  # Reusable macros
├── seeds/                   # Seed data (CSV)
├── snapshots/               # SCD snapshots
├── analyses/                # Analysis queries
└── target/                  # Compiled results
```

### 2.2 Configuration Files

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

# Model-specific configuration
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

## 3. Models

### 3.1 Basic Models

```sql
-- models/staging/stg_orders.sql
-- Staging model: Clean source data

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
-- Fact table: Orders

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
    -- Derived columns
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

### 3.2 Source Definition

```yaml
# models/staging/_sources.yml
version: 2

sources:
  - name: raw
    description: "Raw data sources"
    database: raw_db
    schema: public
    tables:
      - name: orders
        description: "Raw orders table"
        columns:
          - name: order_id
            description: "Unique order ID"
            tests:
              - unique
              - not_null
          - name: customer_id
            description: "Customer ID"
          - name: amount
            description: "Order amount"

      - name: customers
        description: "Raw customers table"
        freshness:
          warn_after: {count: 12, period: hour}
          error_after: {count: 24, period: hour}
        loaded_at_field: updated_at
```

### 3.3 Materialization Types

```sql
-- View (default): Query executed each time
{{ config(materialized='view') }}

-- Table: Create physical table
{{ config(materialized='table') }}

-- Incremental: Incremental processing
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

-- Ephemeral: Inline as CTE (no table created)
{{ config(materialized='ephemeral') }}
```

---

## 4. Tests

### 4.1 Schema Tests

```yaml
# models/marts/core/_schema.yml
version: 2

models:
  - name: fct_orders
    description: "Orders fact table"
    columns:
      - name: order_id
        description: "Unique order ID"
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

### 4.2 Custom Tests

```sql
-- tests/assert_positive_amounts.sql
-- Verify all order amounts are positive

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

### 4.3 Running Tests

```bash
# Run all tests
dbt test

# Test specific model
dbt test --select fct_orders

# Test source freshness
dbt source freshness
```

---

## 5. Documentation

### 5.1 Documentation Definition

```yaml
# models/marts/core/_schema.yml
version: 2

models:
  - name: fct_orders
    description: |
      ## Orders Fact Table

      This table contains all order transactions.

      ### Use Cases
      - Daily/monthly revenue analysis
      - Customer purchase pattern analysis
      - Repeat purchase rate calculation

      ### Refresh Schedule
      - Daily at 06:00 UTC

    meta:
      owner: "data-team@company.com"
      contains_pii: false

    columns:
      - name: order_id
        description: "Unique order identifier (UUID)"
        meta:
          dimension: true

      - name: amount
        description: "Total order amount (USD)"
        meta:
          measure: true
          aggregation: sum
```

### 5.2 Generate and Serve Documentation

```bash
# Generate documentation
dbt docs generate

# Serve documentation
dbt docs serve --port 8080

# View at http://localhost:8080
```

---

## 6. Jinja Templates

### 6.1 Basic Jinja Syntax

```sql
-- Variables
{% set my_var = 'value' %}
SELECT '{{ my_var }}' AS col

-- Conditionals
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

-- Loops
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
-- Using macros
SELECT
    order_id,
    {{ cents_to_dollars('amount_cents') }} AS amount_dollars
FROM orders
{{ limit_data_in_dev() }}
```

### 6.3 Built-in dbt Functions

```sql
-- ref(): Reference another model
SELECT * FROM {{ ref('stg_orders') }}

-- source(): Reference source table
SELECT * FROM {{ source('raw', 'orders') }}

-- this: Reference current model (useful in incremental)
{% if is_incremental() %}
SELECT MAX(updated_at) FROM {{ this }}
{% endif %}

-- config(): Access configuration values
{{ config.get('materialized') }}

-- target: Target environment information
{{ target.name }}    -- dev, prod
{{ target.schema }}  -- dbt_dev
{{ target.type }}    -- postgres, snowflake
```

---

## 7. Incremental Processing

### 7.1 Basic Incremental Model

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
-- Process only data since last run
WHERE created_at > (SELECT MAX(created_at) FROM {{ this }})
{% endif %}
```

### 7.2 Incremental Strategies

```sql
-- Append (default): Only add new data
{{ config(
    materialized='incremental',
    incremental_strategy='append'
) }}

-- Delete+Insert: Delete by key then insert
{{ config(
    materialized='incremental',
    unique_key='id',
    incremental_strategy='delete+insert'
) }}

-- Merge (Snowflake, BigQuery): Use MERGE statement
{{ config(
    materialized='incremental',
    unique_key='id',
    incremental_strategy='merge',
    merge_update_columns=['name', 'amount', 'updated_at']
) }}
```

---

## 8. Execution Commands

### 8.1 Basic Commands

```bash
# Test connection
dbt debug

# Run all models
dbt run

# Run specific model only
dbt run --select fct_orders
dbt run --select staging.*
dbt run --select +fct_orders+  # Include dependencies

# Run tests
dbt test

# Build (run + test)
dbt build

# Load seed data
dbt seed

# Compile only (no execution)
dbt compile

# Clean
dbt clean
```

### 8.2 Selectors

```bash
# By model name
dbt run --select my_model

# By path
dbt run --select models/staging/*

# By tag
dbt run --select tag:daily

# Include upstream dependencies
dbt run --select +my_model

# Include downstream dependencies
dbt run --select my_model+

# Both directions
dbt run --select +my_model+

# Exclude specific model
dbt run --exclude my_model
```

---

## Practice Problems

### Problem 1: Staging Model
Create a stg_products model from the raw products table. Convert prices to dollars and handle NULL values.

### Problem 2: Incremental Model
Write a model that incrementally processes daily sales aggregates.

### Problem 3: Write Tests
Write tests for the fct_sales model (unique, not_null, positive amount check).

---

## Summary

| Concept | Description |
|------|------|
| **Model** | SQL-based data transformation definition |
| **Source** | Raw data reference |
| **ref()** | Reference between models (automatic dependency management) |
| **Test** | Data quality validation |
| **Materialization** | view, table, incremental, ephemeral |
| **Macro** | Reusable SQL templates |

---

## References

- [dbt Documentation](https://docs.getdbt.com/)
- [dbt Learn](https://courses.getdbt.com/)
- [dbt Best Practices](https://docs.getdbt.com/guides/best-practices)
