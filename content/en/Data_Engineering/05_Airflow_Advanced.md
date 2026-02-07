# Airflow Advanced

## Overview

This document covers advanced Airflow features including XCom for data sharing between tasks, dynamic DAG generation, Sensors, Hooks, TaskGroups, and more. Leveraging these features allows you to build more flexible and powerful pipelines.

---

## 1. XCom (Cross-Communication)

### 1.1 Basic XCom Usage

XCom is a mechanism for sharing small amounts of data between tasks.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def push_data(**kwargs):
    """Push data to XCom"""
    ti = kwargs['ti']

    # Method 1: Using xcom_push
    ti.xcom_push(key='my_key', value={'status': 'success', 'count': 100})

    # Method 2: Return value (automatically saved with key='return_value')
    return {'result': 'completed', 'rows': 500}


def pull_data(**kwargs):
    """Pull data from XCom"""
    ti = kwargs['ti']

    # Method 1: Pull by specific key
    custom_data = ti.xcom_pull(key='my_key', task_ids='push_task')
    print(f"Custom data: {custom_data}")

    # Method 2: Pull return value
    return_value = ti.xcom_pull(task_ids='push_task')  # key='return_value' by default
    print(f"Return value: {return_value}")

    # Method 3: Pull from multiple tasks
    multiple_results = ti.xcom_pull(task_ids=['task1', 'task2'])


with DAG('xcom_example', start_date=datetime(2024, 1, 1), schedule_interval=None) as dag:

    push_task = PythonOperator(
        task_id='push_task',
        python_callable=push_data,
    )

    pull_task = PythonOperator(
        task_id='pull_task',
        python_callable=pull_data,
    )

    push_task >> pull_task
```

### 1.2 Using XCom in Jinja Templates

```python
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator

# Using XCom in Bash
bash_task = BashOperator(
    task_id='bash_with_xcom',
    bash_command='echo "Result: {{ ti.xcom_pull(task_ids="push_task") }}"',
)

# Using XCom in SQL
sql_task = PostgresOperator(
    task_id='sql_with_xcom',
    postgres_conn_id='my_postgres',
    sql="""
        INSERT INTO process_log (task_id, result_count, processed_at)
        VALUES (
            'data_load',
            {{ ti.xcom_pull(task_ids='count_task', key='row_count') }},
            NOW()
        );
    """,
)
```

### 1.3 XCom Limitations and Alternatives

```python
# XCom limitation: default 1GB (stored in DB, recommended for small data only)

# Handling large data
class LargeDataHandler:
    """Pattern for handling large data"""

    @staticmethod
    def save_to_storage(data, path: str):
        """Save data to external storage and pass only the path via XCom"""
        import pandas as pd

        # Save to S3, GCS, etc.
        data.to_parquet(path)
        return path  # Return only the path

    @staticmethod
    def load_from_storage(path: str):
        """Load data from path"""
        import pandas as pd
        return pd.read_parquet(path)


# Usage example
def produce_large_data(**kwargs):
    import pandas as pd

    # Generate large dataset
    df = pd.DataFrame({'col': range(1000000)})

    # Save to S3 and return only the path
    path = f"s3://bucket/data/{kwargs['ds']}/output.parquet"
    df.to_parquet(path)

    return path  # Store only path in XCom


def consume_large_data(**kwargs):
    import pandas as pd

    ti = kwargs['ti']
    path = ti.xcom_pull(task_ids='produce_task')

    # Load data from path
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} rows from {path}")
```

---

## 2. Dynamic DAG Generation

### 2.1 Configuration-Based Dynamic DAGs

```python
# dags/dynamic_dag_factory.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Define configuration
DAG_CONFIGS = [
    {
        'dag_id': 'etl_customers',
        'table': 'customers',
        'schedule': '0 1 * * *',
    },
    {
        'dag_id': 'etl_orders',
        'table': 'orders',
        'schedule': '0 2 * * *',
    },
    {
        'dag_id': 'etl_products',
        'table': 'products',
        'schedule': '0 3 * * *',
    },
]


def create_dag(config: dict) -> DAG:
    """Create DAG based on configuration"""

    def extract_table(table_name: str, **kwargs):
        print(f"Extracting {table_name} for {kwargs['ds']}")

    def load_table(table_name: str, **kwargs):
        print(f"Loading {table_name} for {kwargs['ds']}")

    dag = DAG(
        dag_id=config['dag_id'],
        schedule_interval=config['schedule'],
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['dynamic', 'etl'],
    )

    with dag:
        extract = PythonOperator(
            task_id='extract',
            python_callable=extract_table,
            op_kwargs={'table_name': config['table']},
        )

        load = PythonOperator(
            task_id='load',
            python_callable=load_table,
            op_kwargs={'table_name': config['table']},
        )

        extract >> load

    return dag


# Register DAGs in globals() (so Airflow can discover them)
for config in DAG_CONFIGS:
    dag_id = config['dag_id']
    globals()[dag_id] = create_dag(config)
```

### 2.2 YAML/JSON-Based Dynamic DAGs

```python
# dags/yaml_driven_dag.py
import yaml
from pathlib import Path
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Load YAML configuration
config_path = Path(__file__).parent / 'configs' / 'dag_configs.yaml'

# Example configs/dag_configs.yaml:
"""
dags:
  - id: sales_etl
    schedule: "0 6 * * *"
    tasks:
      - name: extract
        type: python
        function: extract_sales
      - name: transform
        type: python
        function: transform_sales
      - name: load
        type: python
        function: load_sales
"""

def load_config():
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_task_callable(func_name: str):
    """Create callable from function name"""
    def task_func(**kwargs):
        print(f"Executing {func_name} for {kwargs['ds']}")
    return task_func


def create_dag_from_yaml(dag_config: dict) -> DAG:
    """Create DAG from YAML configuration"""

    dag = DAG(
        dag_id=dag_config['id'],
        schedule_interval=dag_config['schedule'],
        start_date=datetime(2024, 1, 1),
        catchup=False,
    )

    with dag:
        tasks = {}
        for task_config in dag_config['tasks']:
            task = PythonOperator(
                task_id=task_config['name'],
                python_callable=create_task_callable(task_config['function']),
            )
            tasks[task_config['name']] = task

        # Set sequential dependencies
        task_list = list(tasks.values())
        for i in range(len(task_list) - 1):
            task_list[i] >> task_list[i + 1]

    return dag


# Create and register DAGs
try:
    config = load_config()
    for dag_config in config.get('dags', []):
        dag_id = dag_config['id']
        globals()[dag_id] = create_dag_from_yaml(dag_config)
except Exception as e:
    print(f"Error loading DAG config: {e}")
```

### 2.3 Dynamic Task Generation

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime

# List of tables to process
TABLES = ['users', 'orders', 'products', 'reviews', 'inventory']

with DAG(
    dag_id='dynamic_tasks_example',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily',
    catchup=False,
) as dag:

    start = EmptyOperator(task_id='start')
    end = EmptyOperator(task_id='end')

    # Dynamically create tasks
    for table in TABLES:
        def process_table(table_name=table, **kwargs):
            print(f"Processing table: {table_name}")

        task = PythonOperator(
            task_id=f'process_{table}',
            python_callable=process_table,
            op_kwargs={'table_name': table},
        )

        start >> task >> end
```

---

## 3. Sensors

### 3.1 Built-in Sensors

```python
from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.sensors.time_delta import TimeDeltaSensor
from airflow.providers.http.sensors.http import HttpSensor
from airflow.providers.postgres.sensors.postgres import SqlSensor
from datetime import datetime, timedelta

with DAG('sensor_examples', start_date=datetime(2024, 1, 1), schedule_interval='@daily') as dag:

    # 1. FileSensor - wait for file existence
    wait_for_file = FileSensor(
        task_id='wait_for_file',
        filepath='/data/input/{{ ds }}/data.csv',
        poke_interval=60,           # Check interval (seconds)
        timeout=3600,               # Timeout (seconds)
        mode='poke',                # poke or reschedule
    )

    # 2. ExternalTaskSensor - wait for another DAG's task completion
    wait_for_upstream = ExternalTaskSensor(
        task_id='wait_for_upstream',
        external_dag_id='upstream_dag',
        external_task_id='final_task',
        execution_delta=timedelta(hours=0),  # Same execution_date
        timeout=7200,
        mode='reschedule',          # Return worker and reschedule
    )

    # 3. HttpSensor - check HTTP endpoint
    wait_for_api = HttpSensor(
        task_id='wait_for_api',
        http_conn_id='my_api',
        endpoint='/health',
        request_params={},
        response_check=lambda response: response.status_code == 200,
        poke_interval=30,
        timeout=600,
    )

    # 4. SqlSensor - check SQL condition
    wait_for_data = SqlSensor(
        task_id='wait_for_data',
        conn_id='my_postgres',
        sql="""
            SELECT COUNT(*) > 0
            FROM staging_table
            WHERE date = '{{ ds }}'
        """,
        poke_interval=300,
        timeout=3600,
    )

    # 5. TimeDeltaSensor - wait for time duration
    wait_30_minutes = TimeDeltaSensor(
        task_id='wait_30_minutes',
        delta=timedelta(minutes=30),
    )
```

### 3.2 Custom Sensor

```python
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults
import boto3

class S3KeySensorCustom(BaseSensorOperator):
    """Custom Sensor to check S3 key existence"""

    template_fields = ['bucket_key']

    @apply_defaults
    def __init__(
        self,
        bucket_name: str,
        bucket_key: str,
        aws_conn_id: str = 'aws_default',
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.bucket_name = bucket_name
        self.bucket_key = bucket_key
        self.aws_conn_id = aws_conn_id

    def poke(self, context) -> bool:
        """Check condition (returns True on success)"""
        self.log.info(f"Checking for s3://{self.bucket_name}/{self.bucket_key}")

        # Create S3 client
        s3 = boto3.client('s3')

        try:
            s3.head_object(Bucket=self.bucket_name, Key=self.bucket_key)
            self.log.info("File found!")
            return True
        except s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                self.log.info("File not found, waiting...")
                return False
            raise


# Usage
wait_for_s3 = S3KeySensorCustom(
    task_id='wait_for_s3_file',
    bucket_name='my-bucket',
    bucket_key='data/{{ ds }}/input.parquet',
    poke_interval=60,
    timeout=3600,
    mode='reschedule',
)
```

### 3.3 Sensor Modes

```python
# poke vs reschedule mode comparison
sensor_modes = {
    'poke': {
        'description': 'Occupies worker slot while waiting',
        'pros': 'Fast response time',
        'cons': 'Wastes worker resources',
        'use_case': 'Short wait time expected'
    },
    'reschedule': {
        'description': 'Returns worker and reschedules',
        'pros': 'Efficient worker resource usage',
        'cons': 'Slightly slower response time',
        'use_case': 'Long wait time expected'
    }
}

# Recommended configuration
wait_for_file = FileSensor(
    task_id='wait_for_file',
    filepath='/data/input.csv',
    poke_interval=300,      # Check every 5 minutes
    timeout=86400,          # 24 hour timeout
    mode='reschedule',      # Use reschedule for long waits
    soft_fail=True,         # Skip on timeout (instead of failing)
)
```

---

## 4. Hooks and Connections

### 4.1 Connection Configuration

```python
# Configure Connection via Airflow UI or CLI
# Admin > Connections > Add

# Add Connection via CLI
"""
airflow connections add 'my_postgres' \
    --conn-type 'postgres' \
    --conn-host 'localhost' \
    --conn-port '5432' \
    --conn-login 'user' \
    --conn-password 'password' \
    --conn-schema 'mydb'

airflow connections add 'my_s3' \
    --conn-type 'aws' \
    --conn-extra '{"aws_access_key_id": "xxx", "aws_secret_access_key": "yyy", "region_name": "us-east-1"}'
"""

# Configure Connection via environment variable
# AIRFLOW_CONN_MY_POSTGRES='postgresql://user:password@localhost:5432/mydb'
```

### 4.2 Using Hooks

```python
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.http.hooks.http import HttpHook

def use_postgres_hook(**kwargs):
    """Using PostgreSQL Hook"""
    hook = PostgresHook(postgres_conn_id='my_postgres')

    # Execute SQL
    records = hook.get_records("SELECT * FROM users LIMIT 10")

    # Return as DataFrame
    df = hook.get_pandas_df("SELECT * FROM users")

    # Insert rows
    hook.insert_rows(
        table='users',
        rows=[(1, 'John'), (2, 'Jane')],
        target_fields=['id', 'name']
    )

    # Use connection directly
    conn = hook.get_conn()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET active = true")
    conn.commit()


def use_s3_hook(**kwargs):
    """Using S3 Hook"""
    hook = S3Hook(aws_conn_id='my_s3')

    # Upload file
    hook.load_file(
        filename='/tmp/data.csv',
        key='data/output.csv',
        bucket_name='my-bucket',
        replace=True
    )

    # Download file
    hook.download_file(
        key='data/input.csv',
        bucket_name='my-bucket',
        local_path='/tmp/input.csv'
    )

    # List files
    keys = hook.list_keys(
        bucket_name='my-bucket',
        prefix='data/',
        delimiter='/'
    )


def use_http_hook(**kwargs):
    """Using HTTP Hook"""
    hook = HttpHook(http_conn_id='my_api', method='GET')

    response = hook.run(
        endpoint='/api/data',
        headers={'Authorization': 'Bearer token'},
        data={'param': 'value'}
    )

    return response.json()
```

### 4.3 Custom Hook

```python
from airflow.hooks.base import BaseHook
from typing import Any
import requests

class MyCustomHook(BaseHook):
    """Custom API Hook"""

    conn_name_attr = 'my_custom_conn_id'
    default_conn_name = 'my_custom_default'
    conn_type = 'http'
    hook_name = 'My Custom Hook'

    def __init__(self, my_custom_conn_id: str = default_conn_name):
        super().__init__()
        self.my_custom_conn_id = my_custom_conn_id
        self.base_url = None
        self.api_key = None

    def get_conn(self):
        """Load connection configuration"""
        conn = self.get_connection(self.my_custom_conn_id)
        self.base_url = f"https://{conn.host}"
        self.api_key = conn.password
        return conn

    def make_request(self, endpoint: str, method: str = 'GET', data: dict = None) -> Any:
        """Make API request"""
        self.get_conn()

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        url = f"{self.base_url}{endpoint}"

        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=data
        )

        response.raise_for_status()
        return response.json()


# Usage
def call_custom_api(**kwargs):
    hook = MyCustomHook(my_custom_conn_id='my_api')
    result = hook.make_request('/users', method='GET')
    return result
```

---

## 5. TaskGroup

### 5.1 Basic TaskGroup Usage

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime

with DAG('taskgroup_example', start_date=datetime(2024, 1, 1), schedule_interval='@daily') as dag:

    start = EmptyOperator(task_id='start')

    # Group related tasks with TaskGroup
    with TaskGroup(group_id='extract_group') as extract_group:
        extract_users = PythonOperator(
            task_id='extract_users',
            python_callable=lambda: print("Extracting users")
        )
        extract_orders = PythonOperator(
            task_id='extract_orders',
            python_callable=lambda: print("Extracting orders")
        )
        extract_products = PythonOperator(
            task_id='extract_products',
            python_callable=lambda: print("Extracting products")
        )

    with TaskGroup(group_id='transform_group') as transform_group:
        transform_users = PythonOperator(
            task_id='transform_users',
            python_callable=lambda: print("Transforming users")
        )
        transform_orders = PythonOperator(
            task_id='transform_orders',
            python_callable=lambda: print("Transforming orders")
        )

    with TaskGroup(group_id='load_group') as load_group:
        load_warehouse = PythonOperator(
            task_id='load_warehouse',
            python_callable=lambda: print("Loading to warehouse")
        )

    end = EmptyOperator(task_id='end')

    # Dependencies between TaskGroups
    start >> extract_group >> transform_group >> load_group >> end
```

### 5.2 Nested TaskGroups

```python
from airflow.utils.task_group import TaskGroup

with DAG('nested_taskgroup', ...) as dag:

    with TaskGroup(group_id='data_processing') as data_processing:

        with TaskGroup(group_id='source_a') as source_a:
            extract_a = PythonOperator(task_id='extract', ...)
            transform_a = PythonOperator(task_id='transform', ...)
            extract_a >> transform_a

        with TaskGroup(group_id='source_b') as source_b:
            extract_b = PythonOperator(task_id='extract', ...)
            transform_b = PythonOperator(task_id='transform', ...)
            extract_b >> transform_b

        # Parallel execution then join
        join = EmptyOperator(task_id='join')
        [source_a, source_b] >> join
```

### 5.3 Dynamic TaskGroups

```python
from airflow.utils.task_group import TaskGroup

SOURCES = ['mysql', 'postgres', 'mongodb']

with DAG('dynamic_taskgroup', ...) as dag:

    start = EmptyOperator(task_id='start')

    task_groups = []
    for source in SOURCES:
        with TaskGroup(group_id=f'process_{source}') as tg:
            extract = PythonOperator(
                task_id='extract',
                python_callable=lambda s=source: print(f"Extract from {s}")
            )
            load = PythonOperator(
                task_id='load',
                python_callable=lambda s=source: print(f"Load {s}")
            )
            extract >> load

        task_groups.append(tg)

    end = EmptyOperator(task_id='end')

    start >> task_groups >> end
```

---

## 6. Branching and Conditional Execution

### 6.1 BranchPythonOperator

```python
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator

def choose_branch(**kwargs):
    """Choose next task based on condition"""
    ti = kwargs['ti']
    data_count = ti.xcom_pull(task_ids='count_data')

    if data_count > 1000:
        return 'process_large'
    elif data_count > 0:
        return 'process_small'
    else:
        return 'skip_processing'


with DAG('branch_example', ...) as dag:

    count_data = PythonOperator(
        task_id='count_data',
        python_callable=lambda: 500,  # Example return value
    )

    branch = BranchPythonOperator(
        task_id='branch',
        python_callable=choose_branch,
    )

    process_large = EmptyOperator(task_id='process_large')
    process_small = EmptyOperator(task_id='process_small')
    skip_processing = EmptyOperator(task_id='skip_processing')

    # Join after branching
    join = EmptyOperator(
        task_id='join',
        trigger_rule='none_failed_min_one_success'  # Execute if at least one succeeds
    )

    count_data >> branch >> [process_large, process_small, skip_processing] >> join
```

### 6.2 ShortCircuitOperator

```python
from airflow.operators.python import ShortCircuitOperator

def check_condition(**kwargs):
    """Check condition - skip downstream tasks if returns False"""
    ds = kwargs['ds']
    # Skip on weekends
    day_of_week = datetime.strptime(ds, '%Y-%m-%d').weekday()
    return day_of_week < 5  # True only on weekdays


with DAG('shortcircuit_example', ...) as dag:

    check = ShortCircuitOperator(
        task_id='check_weekday',
        python_callable=check_condition,
    )

    # Tasks below are skipped if check returns False
    process = PythonOperator(task_id='process', ...)
    load = PythonOperator(task_id='load', ...)

    check >> process >> load
```

---

## Practice Problems

### Problem 1: Using XCom
Write a DAG with two tasks that each return a number, and a third task that calculates the sum of the two numbers.

### Problem 2: Dynamic DAG
Write a DAG that dynamically generates ETL tasks for each table in a list (users, orders, products).

### Problem 3: Using Sensors
Write a DAG that waits for a file to be created before processing it.

---

## Summary

| Feature | Description |
|---------|-------------|
| **XCom** | Mechanism for sharing data between tasks |
| **Dynamic DAG** | Dynamically generate DAGs/tasks based on configuration |
| **Sensor** | Operator that waits until a condition is met |
| **Hook** | Interface for connecting to external systems |
| **TaskGroup** | Group related tasks for better visualization |
| **Branch** | Conditional branching based on criteria |

---

## References

- [Airflow XCom Guide](https://airflow.apache.org/docs/apache-airflow/stable/concepts/xcoms.html)
- [Dynamic DAGs](https://airflow.apache.org/docs/apache-airflow/stable/howto/dynamic-dag-generation.html)
- [Airflow Sensors](https://airflow.apache.org/docs/apache-airflow/stable/concepts/sensors.html)
