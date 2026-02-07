# Apache Airflow Basics

## Introduction

Apache Airflow is a platform for programmatically authoring, scheduling, and monitoring workflows. It manages complex data pipelines by defining DAGs (Directed Acyclic Graphs) in Python.

---

## 1. Airflow Architecture

### 1.1 Core Components

```
┌──────────────────────────────────────────────────────────────┐
│                    Airflow Architecture                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────┐         ┌─────────────┐                   │
│   │  Web Server │         │  Scheduler  │                   │
│   │    (UI)     │         │             │                   │
│   └──────┬──────┘         └──────┬──────┘                   │
│          │                       │                          │
│          │    ┌─────────────┐    │                          │
│          └───→│  Metadata   │←───┘                          │
│               │  Database   │                               │
│               │ (PostgreSQL)│                               │
│               └──────┬──────┘                               │
│                      │                                      │
│          ┌───────────┴───────────┐                          │
│          ↓                       ↓                          │
│   ┌─────────────┐         ┌─────────────┐                   │
│   │   Worker    │         │   Worker    │                   │
│   │  (Celery)   │         │  (Celery)   │                   │
│   └─────────────┘         └─────────────┘                   │
│                                                              │
│   DAGs Folder: /opt/airflow/dags/                           │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 Component Roles

| Component | Role |
|-----------|------|
| **Web Server** | Provide UI, visualize DAGs, view logs |
| **Scheduler** | Parse DAGs, schedule tasks, trigger execution |
| **Executor** | Determine task execution method (Local, Celery, K8s) |
| **Worker** | Execute actual tasks (Celery/K8s Executor) |
| **Metadata DB** | Store DAG metadata and execution history |

### 1.3 Executor Types

```python
# airflow.cfg settings
executor_types = {
    "SequentialExecutor": "Single process, for development",
    "LocalExecutor": "Multi-process, single machine",
    "CeleryExecutor": "Distributed processing, production",
    "KubernetesExecutor": "Run as K8s Pods"
}

# Recommended configuration
# Development: LocalExecutor
# Production: CeleryExecutor or KubernetesExecutor
```

---

## 2. Installation and Environment Setup

### 2.1 Docker Compose Installation (Recommended)

```yaml
# docker-compose.yaml
version: '3.8'

x-airflow-common: &airflow-common
  image: apache/airflow:2.7.0
  environment:
    &airflow-common-env
    AIRFLOW__CORE__EXECUTOR: CeleryExecutor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql://airflow:airflow@postgres/airflow
    AIRFLOW__CELERY__BROKER_URL: redis://:@redis:6379/0
    AIRFLOW__CORE__FERNET_KEY: ''
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'true'
    AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
  volumes:
    - ./dags:/opt/airflow/dags
    - ./logs:/opt/airflow/logs
    - ./plugins:/opt/airflow/plugins

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres-db-volume:/var/lib/postgresql/data

  redis:
    image: redis:latest

  airflow-webserver:
    <<: *airflow-common
    command: webserver
    ports:
      - 8080:8080
    depends_on:
      - postgres
      - redis

  airflow-scheduler:
    <<: *airflow-common
    command: scheduler
    depends_on:
      - postgres
      - redis

  airflow-worker:
    <<: *airflow-common
    command: celery worker
    depends_on:
      - airflow-scheduler

  airflow-init:
    <<: *airflow-common
    entrypoint: /bin/bash
    command:
      - -c
      - |
        airflow db init
        airflow users create \
          --username admin \
          --password admin \
          --firstname Admin \
          --lastname User \
          --role Admin \
          --email admin@example.com

volumes:
  postgres-db-volume:
```

### 2.2 pip Installation (Local Development)

```bash
# Create virtual environment
python -m venv airflow-venv
source airflow-venv/bin/activate

# Install Airflow
pip install "apache-airflow[celery,postgres,redis]==2.7.0" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.0/constraints-3.9.txt"

# Initialize
export AIRFLOW_HOME=~/airflow
airflow db init

# Create user
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Start services
airflow webserver --port 8080 &
airflow scheduler &
```

---

## 3. DAG (Directed Acyclic Graph)

### 3.1 Basic DAG Structure

```python
# dags/simple_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# DAG default arguments
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'email': ['data-team@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
with DAG(
    dag_id='simple_example_dag',
    default_args=default_args,
    description='Simple example DAG',
    schedule_interval='0 9 * * *',  # Daily at 9 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,  # Skip past runs
    tags=['example', 'tutorial'],
) as dag:

    # Task 1: Execute Python function
    def print_hello():
        print("Hello, Airflow!")
        return "Hello returned"

    task_hello = PythonOperator(
        task_id='print_hello',
        python_callable=print_hello,
    )

    # Task 2: Execute Bash command
    task_date = BashOperator(
        task_id='print_date',
        bash_command='date',
    )

    # Task 3: Python function (with arguments)
    def greet(name, **kwargs):
        execution_date = kwargs['ds']
        print(f"Hello, {name}! Today is {execution_date}")

    task_greet = PythonOperator(
        task_id='greet_user',
        python_callable=greet,
        op_kwargs={'name': 'Data Engineer'},
    )

    # Define task dependencies
    task_hello >> task_date >> task_greet
    # Or: task_hello.set_downstream(task_date)
```

### 3.2 DAG Parameters

```python
from airflow import DAG

dag = DAG(
    # Required parameters
    dag_id='my_dag',                    # Unique identifier
    start_date=datetime(2024, 1, 1),    # Start date

    # Schedule related
    schedule_interval='@daily',         # Execution frequency
    # schedule_interval='0 0 * * *'     # Cron expression
    # schedule_interval=timedelta(days=1)

    # Execution control
    catchup=False,                      # Whether to run past executions
    max_active_runs=1,                  # Limit concurrent runs
    max_active_tasks=10,                # Limit concurrent tasks

    # Other
    default_args=default_args,          # Default arguments
    description='DAG description',
    tags=['production', 'etl'],
    doc_md="""
    ## DAG Documentation
    This DAG performs daily ETL.
    """
)

# Schedule presets
schedule_presets = {
    '@once': 'Run once',
    '@hourly': 'Every hour (0 * * * *)',
    '@daily': 'Daily at midnight (0 0 * * *)',
    '@weekly': 'Every Sunday (0 0 * * 0)',
    '@monthly': 'First of month (0 0 1 * *)',
    '@yearly': 'January 1st (0 0 1 1 *)',
    None: 'Manual trigger only'
}
```

---

## 4. Operator Types

### 4.1 Main Operators

```python
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.email import EmailOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.http.operators.http import SimpleHttpOperator

# 1. PythonOperator - Execute Python function
def my_function(arg1, arg2):
    return arg1 + arg2

python_task = PythonOperator(
    task_id='python_task',
    python_callable=my_function,
    op_args=[1, 2],              # Positional arguments
    op_kwargs={'arg1': 1},       # Keyword arguments
)


# 2. BashOperator - Execute Bash command
bash_task = BashOperator(
    task_id='bash_task',
    bash_command='echo "Hello" && date',
    env={'MY_VAR': 'value'},     # Environment variables
    cwd='/tmp',                  # Working directory
)


# 3. EmptyOperator - Dummy task (group dependencies)
start = EmptyOperator(task_id='start')
end = EmptyOperator(task_id='end')


# 4. PostgresOperator - Execute SQL
sql_task = PostgresOperator(
    task_id='sql_task',
    postgres_conn_id='my_postgres',
    sql="""
        INSERT INTO logs (message, created_at)
        VALUES ('Task executed', NOW());
    """,
)


# 5. EmailOperator - Send email
email_task = EmailOperator(
    task_id='send_email',
    to='user@example.com',
    subject='Airflow Notification',
    html_content='<h1>Task completed!</h1>',
)


# 6. SimpleHttpOperator - HTTP request
http_task = SimpleHttpOperator(
    task_id='http_task',
    http_conn_id='my_api',
    endpoint='/api/data',
    method='GET',
    response_check=lambda response: response.status_code == 200,
)
```

### 4.2 Branch Operator

```python
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator

def choose_branch(**kwargs):
    """Choose task to execute based on condition"""
    execution_date = kwargs['ds']
    day_of_week = datetime.strptime(execution_date, '%Y-%m-%d').weekday()

    if day_of_week < 5:  # Weekday
        return 'weekday_task'
    else:  # Weekend
        return 'weekend_task'

with DAG('branch_example', ...) as dag:

    branch_task = BranchPythonOperator(
        task_id='branch',
        python_callable=choose_branch,
    )

    weekday_task = EmptyOperator(task_id='weekday_task')
    weekend_task = EmptyOperator(task_id='weekend_task')
    join_task = EmptyOperator(task_id='join', trigger_rule='none_failed_min_one_success')

    branch_task >> [weekday_task, weekend_task] >> join_task
```

### 4.3 Custom Operator

```python
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from typing import Any

class MyCustomOperator(BaseOperator):
    """Custom operator example"""

    template_fields = ['param']  # Fields supporting Jinja templates

    @apply_defaults
    def __init__(
        self,
        param: str,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.param = param

    def execute(self, context: dict) -> Any:
        """Task execution logic"""
        self.log.info(f"Executing with param: {self.param}")

        # Access execution info from context
        execution_date = context['ds']
        task_instance = context['ti']

        # Business logic
        result = f"Processed {self.param} on {execution_date}"

        # Return result via XCom
        return result


# Usage
custom_task = MyCustomOperator(
    task_id='custom_task',
    param='my_value',
)
```

---

## 5. Task Dependencies

### 5.1 Dependency Definition Methods

```python
from airflow import DAG
from airflow.operators.empty import EmptyOperator

with DAG('dependency_example', ...) as dag:

    task_a = EmptyOperator(task_id='task_a')
    task_b = EmptyOperator(task_id='task_b')
    task_c = EmptyOperator(task_id='task_c')
    task_d = EmptyOperator(task_id='task_d')
    task_e = EmptyOperator(task_id='task_e')

    # Method 1: >> operator (recommended)
    task_a >> task_b >> task_c

    # Method 2: << operator (reverse)
    task_c << task_b << task_a  # Same as above

    # Method 3: set_downstream / set_upstream
    task_a.set_downstream(task_b)
    task_b.set_downstream(task_c)

    # Parallel execution
    task_a >> [task_b, task_c] >> task_d

    # Complex dependencies
    #     ┌→ B ─┐
    # A ──┤     ├──→ E
    #     └→ C → D ─┘

    task_a >> task_b >> task_e
    task_a >> task_c >> task_d >> task_e
```

### 5.2 Trigger Rules

```python
from airflow.utils.trigger_rule import TriggerRule

# Trigger rule types
trigger_rules = {
    'all_success': 'All upstream tasks succeeded (default)',
    'all_failed': 'All upstream tasks failed',
    'all_done': 'All upstream tasks completed (success/failure irrelevant)',
    'one_success': 'At least one succeeded',
    'one_failed': 'At least one failed',
    'none_failed': 'No failures (skips allowed)',
    'none_failed_min_one_success': 'No failures and at least one success',
    'none_skipped': 'No skips',
    'always': 'Always run',
}

# Usage example
task_join = EmptyOperator(
    task_id='join',
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
)

# Error handling task
task_error_handler = EmptyOperator(
    task_id='error_handler',
    trigger_rule=TriggerRule.ONE_FAILED,
)
```

---

## 6. Scheduling

### 6.1 Cron Expressions

```python
# Cron format: minute hour day month day_of_week
cron_examples = {
    '0 0 * * *': 'Daily at midnight',
    '0 9 * * 1-5': 'Weekdays at 9 AM',
    '0 */2 * * *': 'Every 2 hours',
    '30 8 1 * *': 'First of month at 8:30 AM',
    '0 0 * * 0': 'Every Sunday at midnight',
}

# Use in DAG
dag = DAG(
    dag_id='scheduled_dag',
    schedule_interval='0 9 * * 1-5',  # Weekdays at 9 AM
    start_date=datetime(2024, 1, 1),
    ...
)
```

### 6.2 Data Interval

```python
# Airflow 2.0+ data interval concept
"""
schedule_interval = @daily, start_date = 2024-01-01

Execution time: 2024-01-02 00:00
data_interval_start: 2024-01-01 00:00
data_interval_end: 2024-01-02 00:00
logical_date (execution_date): 2024-01-01 00:00

→ Runs on 2024-01-02 to process 2024-01-01 data
"""

def process_daily_data(**kwargs):
    # Data period to process
    data_interval_start = kwargs['data_interval_start']
    data_interval_end = kwargs['data_interval_end']

    print(f"Processing data from {data_interval_start} to {data_interval_end}")

# Using Jinja templates
sql_task = PostgresOperator(
    task_id='load_data',
    sql="""
        SELECT * FROM sales
        WHERE sale_date >= '{{ data_interval_start }}'
          AND sale_date < '{{ data_interval_end }}'
    """,
)
```

---

## 7. Basic DAG Writing Example

### 7.1 Daily ETL DAG

```python
# dags/daily_etl_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    'owner': 'data_team',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email': ['data-alerts@company.com'],
}

def extract_data(**kwargs):
    """Extract data"""
    import pandas as pd

    ds = kwargs['ds']  # execution date (YYYY-MM-DD)

    # Extract data from source
    query = f"""
        SELECT * FROM source_table
        WHERE date = '{ds}'
    """

    # df = pd.read_sql(query, source_conn)
    # df.to_parquet(f'/tmp/extract_{ds}.parquet')

    print(f"Extracted data for {ds}")
    return f"/tmp/extract_{ds}.parquet"


def transform_data(**kwargs):
    """Transform data"""
    import pandas as pd

    ti = kwargs['ti']
    extract_path = ti.xcom_pull(task_ids='extract')

    # df = pd.read_parquet(extract_path)
    # Transformation logic
    # df['new_column'] = df['column'].apply(transform_func)
    # df.to_parquet(f'/tmp/transform_{kwargs["ds"]}.parquet')

    print("Data transformed")
    return f"/tmp/transform_{kwargs['ds']}.parquet"


with DAG(
    dag_id='daily_etl_pipeline',
    default_args=default_args,
    description='Daily ETL pipeline',
    schedule_interval='0 6 * * *',  # Daily at 6 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['etl', 'daily', 'production'],
) as dag:

    start = EmptyOperator(task_id='start')

    extract = PythonOperator(
        task_id='extract',
        python_callable=extract_data,
    )

    transform = PythonOperator(
        task_id='transform',
        python_callable=transform_data,
    )

    load = PostgresOperator(
        task_id='load',
        postgres_conn_id='warehouse',
        sql="""
            COPY target_table FROM '/tmp/transform_{{ ds }}.parquet'
            WITH (FORMAT 'parquet');
        """,
    )

    validate = PostgresOperator(
        task_id='validate',
        postgres_conn_id='warehouse',
        sql="""
            SELECT
                CASE WHEN COUNT(*) > 0 THEN 1
                     ELSE 1/0  -- Raise error
                END
            FROM target_table
            WHERE date = '{{ ds }}';
        """,
    )

    end = EmptyOperator(task_id='end')

    # Define dependencies
    start >> extract >> transform >> load >> validate >> end
```

---

## Practice Problems

### Problem 1: Basic DAG Creation
Create a DAG that runs hourly. It should include two tasks: one that logs the current time and another that creates a temporary file.

### Problem 2: Conditional Execution
Create a DAG using BranchPythonOperator that executes different tasks on weekdays versus weekends.

---

## Summary

| Concept | Description |
|------|------|
| **DAG** | Directed Acyclic Graph defining task dependencies |
| **Operator** | Task execution type (Python, Bash, SQL, etc.) |
| **Task** | Individual work unit within a DAG |
| **Scheduler** | DAG parsing and task scheduling |
| **Executor** | Task execution method (Local, Celery, K8s) |

---

## References

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Astronomer Guides](https://www.astronomer.io/guides/)
