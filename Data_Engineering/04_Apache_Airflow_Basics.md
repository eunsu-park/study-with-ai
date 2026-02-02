# Apache Airflow 기초

## 개요

Apache Airflow는 워크플로우를 프로그래밍 방식으로 작성, 스케줄링, 모니터링하는 플랫폼입니다. Python으로 DAG(Directed Acyclic Graph)를 정의하여 복잡한 데이터 파이프라인을 관리합니다.

---

## 1. Airflow 아키텍처

### 1.1 핵심 구성 요소

```
┌──────────────────────────────────────────────────────────────┐
│                    Airflow Architecture                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────┐         ┌─────────────┐                   │
│   │  Web Server │         │  Scheduler  │                   │
│   │    (UI)     │         │  (스케줄러) │                   │
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

### 1.2 구성 요소 역할

| 구성 요소 | 역할 |
|-----------|------|
| **Web Server** | UI 제공, DAG 시각화, 로그 조회 |
| **Scheduler** | DAG 파싱, Task 스케줄링, 실행 트리거 |
| **Executor** | Task 실행 방식 결정 (Local, Celery, K8s) |
| **Worker** | 실제 Task 실행 (Celery/K8s Executor) |
| **Metadata DB** | DAG 메타데이터, 실행 이력 저장 |

### 1.3 Executor 유형

```python
# airflow.cfg 설정
executor_types = {
    "SequentialExecutor": "단일 프로세스, 개발용",
    "LocalExecutor": "멀티프로세스, 단일 머신",
    "CeleryExecutor": "분산 처리, 프로덕션",
    "KubernetesExecutor": "K8s Pod으로 실행"
}

# 권장 설정
# 개발: LocalExecutor
# 프로덕션: CeleryExecutor 또는 KubernetesExecutor
```

---

## 2. 설치 및 환경 설정

### 2.1 Docker Compose 설치 (권장)

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

### 2.2 pip 설치 (로컬 개발)

```bash
# 가상 환경 생성
python -m venv airflow-venv
source airflow-venv/bin/activate

# Airflow 설치
pip install "apache-airflow[celery,postgres,redis]==2.7.0" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.0/constraints-3.9.txt"

# 초기화
export AIRFLOW_HOME=~/airflow
airflow db init

# 사용자 생성
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# 서비스 시작
airflow webserver --port 8080 &
airflow scheduler &
```

---

## 3. DAG (Directed Acyclic Graph)

### 3.1 DAG 기본 구조

```python
# dags/simple_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# DAG 기본 인자
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'email': ['data-team@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

# DAG 정의
with DAG(
    dag_id='simple_example_dag',
    default_args=default_args,
    description='간단한 예제 DAG',
    schedule_interval='0 9 * * *',  # 매일 오전 9시
    start_date=datetime(2024, 1, 1),
    catchup=False,  # 과거 실행 건너뛰기
    tags=['example', 'tutorial'],
) as dag:

    # Task 1: Python 함수 실행
    def print_hello():
        print("Hello, Airflow!")
        return "Hello returned"

    task_hello = PythonOperator(
        task_id='print_hello',
        python_callable=print_hello,
    )

    # Task 2: Bash 명령 실행
    task_date = BashOperator(
        task_id='print_date',
        bash_command='date',
    )

    # Task 3: Python 함수 (인자 전달)
    def greet(name, **kwargs):
        execution_date = kwargs['ds']
        print(f"Hello, {name}! Today is {execution_date}")

    task_greet = PythonOperator(
        task_id='greet_user',
        python_callable=greet,
        op_kwargs={'name': 'Data Engineer'},
    )

    # Task 의존성 정의
    task_hello >> task_date >> task_greet
    # 또는: task_hello.set_downstream(task_date)
```

### 3.2 DAG 매개변수

```python
from airflow import DAG

dag = DAG(
    # 필수 매개변수
    dag_id='my_dag',                    # 고유 식별자
    start_date=datetime(2024, 1, 1),    # 시작 날짜

    # 스케줄 관련
    schedule_interval='@daily',         # 실행 주기
    # schedule_interval='0 0 * * *'     # Cron 표현식
    # schedule_interval=timedelta(days=1)

    # 실행 제어
    catchup=False,                      # 과거 실행 여부
    max_active_runs=1,                  # 동시 실행 제한
    max_active_tasks=10,                # 동시 Task 제한

    # 기타
    default_args=default_args,          # 기본 인자
    description='DAG 설명',
    tags=['production', 'etl'],
    doc_md="""
    ## DAG 문서
    이 DAG는 일일 ETL을 수행합니다.
    """
)

# 스케줄 프리셋
schedule_presets = {
    '@once': '한 번만 실행',
    '@hourly': '매시간 (0 * * * *)',
    '@daily': '매일 자정 (0 0 * * *)',
    '@weekly': '매주 일요일 (0 0 * * 0)',
    '@monthly': '매월 1일 (0 0 1 * *)',
    '@yearly': '매년 1월 1일 (0 0 1 1 *)',
    None: '수동 실행만'
}
```

---

## 4. Operator 유형

### 4.1 주요 Operator

```python
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.email import EmailOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.http.operators.http import SimpleHttpOperator

# 1. PythonOperator - Python 함수 실행
def my_function(arg1, arg2):
    return arg1 + arg2

python_task = PythonOperator(
    task_id='python_task',
    python_callable=my_function,
    op_args=[1, 2],              # 위치 인자
    op_kwargs={'arg1': 1},       # 키워드 인자
)


# 2. BashOperator - Bash 명령 실행
bash_task = BashOperator(
    task_id='bash_task',
    bash_command='echo "Hello" && date',
    env={'MY_VAR': 'value'},     # 환경 변수
    cwd='/tmp',                  # 작업 디렉토리
)


# 3. EmptyOperator - 더미 Task (의존성 그룹화)
start = EmptyOperator(task_id='start')
end = EmptyOperator(task_id='end')


# 4. PostgresOperator - SQL 실행
sql_task = PostgresOperator(
    task_id='sql_task',
    postgres_conn_id='my_postgres',
    sql="""
        INSERT INTO logs (message, created_at)
        VALUES ('Task executed', NOW());
    """,
)


# 5. EmailOperator - 이메일 전송
email_task = EmailOperator(
    task_id='send_email',
    to='user@example.com',
    subject='Airflow Notification',
    html_content='<h1>Task completed!</h1>',
)


# 6. SimpleHttpOperator - HTTP 요청
http_task = SimpleHttpOperator(
    task_id='http_task',
    http_conn_id='my_api',
    endpoint='/api/data',
    method='GET',
    response_check=lambda response: response.status_code == 200,
)
```

### 4.2 브랜치 Operator

```python
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator

def choose_branch(**kwargs):
    """조건에 따라 실행할 Task 선택"""
    execution_date = kwargs['ds']
    day_of_week = datetime.strptime(execution_date, '%Y-%m-%d').weekday()

    if day_of_week < 5:  # 평일
        return 'weekday_task'
    else:  # 주말
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

### 4.3 커스텀 Operator

```python
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from typing import Any

class MyCustomOperator(BaseOperator):
    """커스텀 Operator 예시"""

    template_fields = ['param']  # Jinja 템플릿 지원 필드

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
        """Task 실행 로직"""
        self.log.info(f"Executing with param: {self.param}")

        # context에서 실행 정보 접근
        execution_date = context['ds']
        task_instance = context['ti']

        # 비즈니스 로직
        result = f"Processed {self.param} on {execution_date}"

        # XCom으로 결과 반환
        return result


# 사용
custom_task = MyCustomOperator(
    task_id='custom_task',
    param='my_value',
)
```

---

## 5. Task 의존성

### 5.1 의존성 정의 방법

```python
from airflow import DAG
from airflow.operators.empty import EmptyOperator

with DAG('dependency_example', ...) as dag:

    task_a = EmptyOperator(task_id='task_a')
    task_b = EmptyOperator(task_id='task_b')
    task_c = EmptyOperator(task_id='task_c')
    task_d = EmptyOperator(task_id='task_d')
    task_e = EmptyOperator(task_id='task_e')

    # 방법 1: >> 연산자 (권장)
    task_a >> task_b >> task_c

    # 방법 2: << 연산자 (역방향)
    task_c << task_b << task_a  # 위와 동일

    # 방법 3: set_downstream / set_upstream
    task_a.set_downstream(task_b)
    task_b.set_downstream(task_c)

    # 병렬 실행
    task_a >> [task_b, task_c] >> task_d

    # 복잡한 의존성
    #     ┌→ B ─┐
    # A ──┤     ├──→ E
    #     └→ C → D ─┘

    task_a >> task_b >> task_e
    task_a >> task_c >> task_d >> task_e
```

### 5.2 Trigger Rule

```python
from airflow.utils.trigger_rule import TriggerRule

# Trigger Rule 유형
trigger_rules = {
    'all_success': '모든 상위 Task 성공 (기본값)',
    'all_failed': '모든 상위 Task 실패',
    'all_done': '모든 상위 Task 완료 (성공/실패 무관)',
    'one_success': '하나 이상 성공',
    'one_failed': '하나 이상 실패',
    'none_failed': '실패 없음 (스킵 허용)',
    'none_failed_min_one_success': '실패 없고 최소 하나 성공',
    'none_skipped': '스킵 없음',
    'always': '항상 실행',
}

# 사용 예시
task_join = EmptyOperator(
    task_id='join',
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
)

# 에러 핸들링 Task
task_error_handler = EmptyOperator(
    task_id='error_handler',
    trigger_rule=TriggerRule.ONE_FAILED,
)
```

---

## 6. 스케줄링

### 6.1 Cron 표현식

```python
# Cron 형식: 분 시 일 월 요일
cron_examples = {
    '0 0 * * *': '매일 자정',
    '0 9 * * 1-5': '평일 오전 9시',
    '0 */2 * * *': '2시간마다',
    '30 8 1 * *': '매월 1일 오전 8:30',
    '0 0 * * 0': '매주 일요일 자정',
}

# DAG에서 사용
dag = DAG(
    dag_id='scheduled_dag',
    schedule_interval='0 9 * * 1-5',  # 평일 오전 9시
    start_date=datetime(2024, 1, 1),
    ...
)
```

### 6.2 데이터 간격 (Data Interval)

```python
# Airflow 2.0+ 데이터 간격 개념
"""
schedule_interval = @daily, start_date = 2024-01-01

실행 시점: 2024-01-02 00:00
data_interval_start: 2024-01-01 00:00
data_interval_end: 2024-01-02 00:00
logical_date (execution_date): 2024-01-01 00:00

→ 2024-01-01 데이터를 처리하기 위해 2024-01-02에 실행
"""

def process_daily_data(**kwargs):
    # 처리할 데이터 기간
    data_interval_start = kwargs['data_interval_start']
    data_interval_end = kwargs['data_interval_end']

    print(f"Processing data from {data_interval_start} to {data_interval_end}")

# Jinja 템플릿 사용
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

## 7. 기본 DAG 작성 예제

### 7.1 일일 ETL DAG

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
    """데이터 추출"""
    import pandas as pd

    ds = kwargs['ds']  # execution date (YYYY-MM-DD)

    # 소스에서 데이터 추출
    query = f"""
        SELECT * FROM source_table
        WHERE date = '{ds}'
    """

    # df = pd.read_sql(query, source_conn)
    # df.to_parquet(f'/tmp/extract_{ds}.parquet')

    print(f"Extracted data for {ds}")
    return f"/tmp/extract_{ds}.parquet"


def transform_data(**kwargs):
    """데이터 변환"""
    import pandas as pd

    ti = kwargs['ti']
    extract_path = ti.xcom_pull(task_ids='extract')

    # df = pd.read_parquet(extract_path)
    # 변환 로직
    # df['new_column'] = df['column'].apply(transform_func)
    # df.to_parquet(f'/tmp/transform_{kwargs["ds"]}.parquet')

    print("Data transformed")
    return f"/tmp/transform_{kwargs['ds']}.parquet"


with DAG(
    dag_id='daily_etl_pipeline',
    default_args=default_args,
    description='일일 ETL 파이프라인',
    schedule_interval='0 6 * * *',  # 매일 오전 6시
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
                     ELSE 1/0  -- 에러 발생
                END
            FROM target_table
            WHERE date = '{{ ds }}';
        """,
    )

    end = EmptyOperator(task_id='end')

    # 의존성 정의
    start >> extract >> transform >> load >> validate >> end
```

---

## 연습 문제

### 문제 1: 기본 DAG 작성
매시간 실행되는 DAG를 작성하세요. 현재 시간을 로그에 출력하고, 임시 파일을 생성하는 두 개의 Task를 포함해야 합니다.

### 문제 2: 조건부 실행
평일과 주말에 다른 Task를 실행하는 BranchPythonOperator를 사용한 DAG를 작성하세요.

---

## 요약

| 개념 | 설명 |
|------|------|
| **DAG** | Task의 의존성을 정의한 방향성 비순환 그래프 |
| **Operator** | Task의 실행 유형 (Python, Bash, SQL 등) |
| **Task** | DAG 내의 개별 작업 단위 |
| **Scheduler** | DAG 파싱 및 Task 스케줄링 |
| **Executor** | Task 실행 방식 (Local, Celery, K8s) |

---

## 참고 자료

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Astronomer Guides](https://www.astronomer.io/guides/)
