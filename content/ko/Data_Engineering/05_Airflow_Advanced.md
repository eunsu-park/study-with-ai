# Airflow 심화

## 개요

이 문서에서는 Airflow의 고급 기능인 XCom을 통한 데이터 공유, 동적 DAG 생성, Sensor, Hook, TaskGroup 등을 다룹니다. 이러한 기능을 활용하면 더 유연하고 강력한 파이프라인을 구축할 수 있습니다.

---

## 1. XCom (Cross-Communication)

### 1.1 XCom 기본 사용법

XCom은 Task 간에 작은 데이터를 공유하는 메커니즘입니다.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def push_data(**kwargs):
    """XCom으로 데이터 푸시"""
    ti = kwargs['ti']

    # 방법 1: xcom_push 사용
    ti.xcom_push(key='my_key', value={'status': 'success', 'count': 100})

    # 방법 2: return 값 (key='return_value'로 자동 저장)
    return {'result': 'completed', 'rows': 500}


def pull_data(**kwargs):
    """XCom에서 데이터 가져오기"""
    ti = kwargs['ti']

    # 방법 1: 특정 key로 가져오기
    custom_data = ti.xcom_pull(key='my_key', task_ids='push_task')
    print(f"Custom data: {custom_data}")

    # 방법 2: return 값 가져오기
    return_value = ti.xcom_pull(task_ids='push_task')  # key='return_value' 기본값
    print(f"Return value: {return_value}")

    # 방법 3: 여러 Task에서 가져오기
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

### 1.2 Jinja 템플릿에서 XCom 사용

```python
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator

# Bash에서 XCom 사용
bash_task = BashOperator(
    task_id='bash_with_xcom',
    bash_command='echo "Result: {{ ti.xcom_pull(task_ids="push_task") }}"',
)

# SQL에서 XCom 사용
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

### 1.3 XCom 제한 사항 및 대안

```python
# XCom 제한: 기본 1GB (DB에 저장되므로 작은 데이터만 권장)

# 대용량 데이터 처리 방법
class LargeDataHandler:
    """대용량 데이터 전달 패턴"""

    @staticmethod
    def save_to_storage(data, path: str):
        """데이터를 외부 스토리지에 저장하고 경로만 XCom으로 전달"""
        import pandas as pd

        # S3, GCS 등에 저장
        data.to_parquet(path)
        return path  # 경로만 반환

    @staticmethod
    def load_from_storage(path: str):
        """경로에서 데이터 로드"""
        import pandas as pd
        return pd.read_parquet(path)


# 사용 예시
def produce_large_data(**kwargs):
    import pandas as pd

    # 대용량 데이터 생성
    df = pd.DataFrame({'col': range(1000000)})

    # S3에 저장하고 경로만 반환
    path = f"s3://bucket/data/{kwargs['ds']}/output.parquet"
    df.to_parquet(path)

    return path  # XCom에는 경로만 저장


def consume_large_data(**kwargs):
    import pandas as pd

    ti = kwargs['ti']
    path = ti.xcom_pull(task_ids='produce_task')

    # 경로에서 데이터 로드
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} rows from {path}")
```

---

## 2. 동적 DAG 생성

### 2.1 설정 기반 동적 DAG

```python
# dags/dynamic_dag_factory.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# 설정 정의
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
    """설정 기반으로 DAG 생성"""

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


# DAG들을 globals()에 등록 (Airflow가 인식하도록)
for config in DAG_CONFIGS:
    dag_id = config['dag_id']
    globals()[dag_id] = create_dag(config)
```

### 2.2 YAML/JSON 기반 동적 DAG

```python
# dags/yaml_driven_dag.py
import yaml
from pathlib import Path
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# YAML 설정 로드
config_path = Path(__file__).parent / 'configs' / 'dag_configs.yaml'

# configs/dag_configs.yaml 예시:
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
    """함수명으로 callable 생성"""
    def task_func(**kwargs):
        print(f"Executing {func_name} for {kwargs['ds']}")
    return task_func


def create_dag_from_yaml(dag_config: dict) -> DAG:
    """YAML 설정으로 DAG 생성"""

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

        # 순차 의존성 설정
        task_list = list(tasks.values())
        for i in range(len(task_list) - 1):
            task_list[i] >> task_list[i + 1]

    return dag


# DAG 생성 및 등록
try:
    config = load_config()
    for dag_config in config.get('dags', []):
        dag_id = dag_config['id']
        globals()[dag_id] = create_dag_from_yaml(dag_config)
except Exception as e:
    print(f"Error loading DAG config: {e}")
```

### 2.3 동적 Task 생성

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime

# 처리할 테이블 목록
TABLES = ['users', 'orders', 'products', 'reviews', 'inventory']

with DAG(
    dag_id='dynamic_tasks_example',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily',
    catchup=False,
) as dag:

    start = EmptyOperator(task_id='start')
    end = EmptyOperator(task_id='end')

    # 동적으로 Task 생성
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

## 3. Sensor

### 3.1 내장 Sensor

```python
from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.sensors.time_delta import TimeDeltaSensor
from airflow.providers.http.sensors.http import HttpSensor
from airflow.providers.postgres.sensors.postgres import SqlSensor
from datetime import datetime, timedelta

with DAG('sensor_examples', start_date=datetime(2024, 1, 1), schedule_interval='@daily') as dag:

    # 1. FileSensor - 파일 존재 확인
    wait_for_file = FileSensor(
        task_id='wait_for_file',
        filepath='/data/input/{{ ds }}/data.csv',
        poke_interval=60,           # 확인 주기 (초)
        timeout=3600,               # 타임아웃 (초)
        mode='poke',                # poke 또는 reschedule
    )

    # 2. ExternalTaskSensor - 다른 DAG의 Task 완료 대기
    wait_for_upstream = ExternalTaskSensor(
        task_id='wait_for_upstream',
        external_dag_id='upstream_dag',
        external_task_id='final_task',
        execution_delta=timedelta(hours=0),  # 같은 execution_date
        timeout=7200,
        mode='reschedule',          # 워커 반환 후 재스케줄
    )

    # 3. HttpSensor - HTTP 엔드포인트 확인
    wait_for_api = HttpSensor(
        task_id='wait_for_api',
        http_conn_id='my_api',
        endpoint='/health',
        request_params={},
        response_check=lambda response: response.status_code == 200,
        poke_interval=30,
        timeout=600,
    )

    # 4. SqlSensor - SQL 조건 확인
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

    # 5. TimeDeltaSensor - 시간 대기
    wait_30_minutes = TimeDeltaSensor(
        task_id='wait_30_minutes',
        delta=timedelta(minutes=30),
    )
```

### 3.2 커스텀 Sensor

```python
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults
import boto3

class S3KeySensorCustom(BaseSensorOperator):
    """S3 키 존재 확인 커스텀 Sensor"""

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
        """조건 확인 (True 반환 시 성공)"""
        self.log.info(f"Checking for s3://{self.bucket_name}/{self.bucket_key}")

        # S3 클라이언트 생성
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


# 사용
wait_for_s3 = S3KeySensorCustom(
    task_id='wait_for_s3_file',
    bucket_name='my-bucket',
    bucket_key='data/{{ ds }}/input.parquet',
    poke_interval=60,
    timeout=3600,
    mode='reschedule',
)
```

### 3.3 Sensor 모드

```python
# poke vs reschedule 모드 비교
sensor_modes = {
    'poke': {
        'description': '워커 슬롯을 점유하고 대기',
        'pros': '빠른 반응 시간',
        'cons': '워커 리소스 낭비',
        'use_case': '짧은 대기 시간 예상'
    },
    'reschedule': {
        'description': '워커 반환 후 재스케줄',
        'pros': '워커 리소스 효율적 사용',
        'cons': '다소 느린 반응 시간',
        'use_case': '긴 대기 시간 예상'
    }
}

# 권장 설정
wait_for_file = FileSensor(
    task_id='wait_for_file',
    filepath='/data/input.csv',
    poke_interval=300,      # 5분마다 확인
    timeout=86400,          # 24시간 타임아웃
    mode='reschedule',      # 긴 대기에는 reschedule
    soft_fail=True,         # 타임아웃 시 스킵 (실패 대신)
)
```

---

## 4. Hook과 Connection

### 4.1 Connection 설정

```python
# Airflow UI 또는 CLI로 Connection 설정
# Admin > Connections > Add

# CLI로 Connection 추가
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

# 환경 변수로 Connection 설정
# AIRFLOW_CONN_MY_POSTGRES='postgresql://user:password@localhost:5432/mydb'
```

### 4.2 Hook 사용

```python
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.http.hooks.http import HttpHook

def use_postgres_hook(**kwargs):
    """PostgreSQL Hook 사용"""
    hook = PostgresHook(postgres_conn_id='my_postgres')

    # SQL 실행
    records = hook.get_records("SELECT * FROM users LIMIT 10")

    # DataFrame으로 반환
    df = hook.get_pandas_df("SELECT * FROM users")

    # 삽입
    hook.insert_rows(
        table='users',
        rows=[(1, 'John'), (2, 'Jane')],
        target_fields=['id', 'name']
    )

    # 직접 연결 사용
    conn = hook.get_conn()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET active = true")
    conn.commit()


def use_s3_hook(**kwargs):
    """S3 Hook 사용"""
    hook = S3Hook(aws_conn_id='my_s3')

    # 파일 업로드
    hook.load_file(
        filename='/tmp/data.csv',
        key='data/output.csv',
        bucket_name='my-bucket',
        replace=True
    )

    # 파일 다운로드
    hook.download_file(
        key='data/input.csv',
        bucket_name='my-bucket',
        local_path='/tmp/input.csv'
    )

    # 파일 목록 조회
    keys = hook.list_keys(
        bucket_name='my-bucket',
        prefix='data/',
        delimiter='/'
    )


def use_http_hook(**kwargs):
    """HTTP Hook 사용"""
    hook = HttpHook(http_conn_id='my_api', method='GET')

    response = hook.run(
        endpoint='/api/data',
        headers={'Authorization': 'Bearer token'},
        data={'param': 'value'}
    )

    return response.json()
```

### 4.3 커스텀 Hook

```python
from airflow.hooks.base import BaseHook
from typing import Any
import requests

class MyCustomHook(BaseHook):
    """커스텀 API Hook"""

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
        """Connection 설정 로드"""
        conn = self.get_connection(self.my_custom_conn_id)
        self.base_url = f"https://{conn.host}"
        self.api_key = conn.password
        return conn

    def make_request(self, endpoint: str, method: str = 'GET', data: dict = None) -> Any:
        """API 요청"""
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


# 사용
def call_custom_api(**kwargs):
    hook = MyCustomHook(my_custom_conn_id='my_api')
    result = hook.make_request('/users', method='GET')
    return result
```

---

## 5. TaskGroup

### 5.1 TaskGroup 기본 사용

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime

with DAG('taskgroup_example', start_date=datetime(2024, 1, 1), schedule_interval='@daily') as dag:

    start = EmptyOperator(task_id='start')

    # TaskGroup으로 관련 Task 그룹화
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

    # TaskGroup 간 의존성
    start >> extract_group >> transform_group >> load_group >> end
```

### 5.2 중첩 TaskGroup

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

        # 병렬 실행 후 조인
        join = EmptyOperator(task_id='join')
        [source_a, source_b] >> join
```

### 5.3 동적 TaskGroup

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

## 6. 분기 처리와 조건부 실행

### 6.1 BranchPythonOperator

```python
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator

def choose_branch(**kwargs):
    """조건에 따라 다음 Task 선택"""
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
        python_callable=lambda: 500,  # 예시 반환값
    )

    branch = BranchPythonOperator(
        task_id='branch',
        python_callable=choose_branch,
    )

    process_large = EmptyOperator(task_id='process_large')
    process_small = EmptyOperator(task_id='process_small')
    skip_processing = EmptyOperator(task_id='skip_processing')

    # 분기 후 합류
    join = EmptyOperator(
        task_id='join',
        trigger_rule='none_failed_min_one_success'  # 하나라도 성공하면 실행
    )

    count_data >> branch >> [process_large, process_small, skip_processing] >> join
```

### 6.2 ShortCircuitOperator

```python
from airflow.operators.python import ShortCircuitOperator

def check_condition(**kwargs):
    """조건 확인 - False 반환 시 이후 Task 스킵"""
    ds = kwargs['ds']
    # 주말이면 스킵
    day_of_week = datetime.strptime(ds, '%Y-%m-%d').weekday()
    return day_of_week < 5  # 평일만 True


with DAG('shortcircuit_example', ...) as dag:

    check = ShortCircuitOperator(
        task_id='check_weekday',
        python_callable=check_condition,
    )

    # check가 False 반환 시 아래 Task들은 스킵됨
    process = PythonOperator(task_id='process', ...)
    load = PythonOperator(task_id='load', ...)

    check >> process >> load
```

---

## 연습 문제

### 문제 1: XCom 활용
두 개의 Task에서 각각 숫자를 반환하고, 세 번째 Task에서 두 숫자의 합을 계산하는 DAG를 작성하세요.

### 문제 2: 동적 DAG
테이블 목록(users, orders, products)을 기반으로 각 테이블에 대한 ETL Task를 동적으로 생성하는 DAG를 작성하세요.

### 문제 3: Sensor 사용
파일이 생성될 때까지 대기한 후 처리하는 DAG를 작성하세요.

---

## 요약

| 기능 | 설명 |
|------|------|
| **XCom** | Task 간 데이터 공유 메커니즘 |
| **동적 DAG** | 설정 기반으로 DAG/Task 동적 생성 |
| **Sensor** | 조건 충족까지 대기하는 Operator |
| **Hook** | 외부 시스템 연결 인터페이스 |
| **TaskGroup** | 관련 Task를 그룹화하여 시각화 |
| **Branch** | 조건에 따른 분기 처리 |

---

## 참고 자료

- [Airflow XCom Guide](https://airflow.apache.org/docs/apache-airflow/stable/concepts/xcoms.html)
- [Dynamic DAGs](https://airflow.apache.org/docs/apache-airflow/stable/howto/dynamic-dag-generation.html)
- [Airflow Sensors](https://airflow.apache.org/docs/apache-airflow/stable/concepts/sensors.html)
