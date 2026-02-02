# Prefect 모던 오케스트레이션

## 개요

Prefect는 현대적인 워크플로우 오케스트레이션 도구로, Python 네이티브 방식으로 데이터 파이프라인을 구축합니다. Airflow와 비교하여 더 간단한 설정과 동적 워크플로우를 지원합니다.

---

## 1. Prefect 개요

### 1.1 Prefect vs Airflow

```
┌────────────────────────────────────────────────────────────────┐
│                   Prefect vs Airflow 비교                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Airflow:                    Prefect:                          │
│  ┌──────────────┐           ┌──────────────┐                  │
│  │ DAG (Static) │           │ Flow (Dynamic)│                  │
│  │              │           │               │                  │
│  │ - 정적 정의  │           │ - 동적 생성   │                  │
│  │ - 파일 기반  │           │ - Python 코드 │                  │
│  │ - Scheduler  │           │ - 이벤트 기반 │                  │
│  └──────────────┘           └──────────────┘                  │
│                                                                │
│  실행 모델:                  실행 모델:                         │
│  Scheduler → Worker         Trigger → Work Pool → Worker       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

| 특성 | Airflow | Prefect |
|------|---------|---------|
| **정의 방식** | DAG 파일 | Python 데코레이터 |
| **스케줄링** | Scheduler 프로세스 | 이벤트 기반, 서버리스 |
| **동적 워크플로우** | 제한적 | 네이티브 지원 |
| **로컬 실행** | 복잡한 설정 | 즉시 가능 |
| **상태 관리** | DB 필수 | 선택적 |
| **학습 곡선** | 가파름 | 완만함 |

### 1.2 Prefect 아키텍처

```
┌────────────────────────────────────────────────────────────────┐
│                    Prefect Architecture                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   ┌─────────────────────────────────────────────┐             │
│   │              Prefect Cloud / Server         │             │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐       │             │
│   │  │  UI     │ │  API    │ │ Automations    │             │
│   │  └─────────┘ └─────────┘ └─────────┘       │             │
│   └─────────────────────────────────────────────┘             │
│                          ↑ ↓                                   │
│   ┌─────────────────────────────────────────────┐             │
│   │               Work Pools                     │             │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐       │             │
│   │  │ Process │ │ Docker  │ │  K8s    │       │             │
│   │  └─────────┘ └─────────┘ └─────────┘       │             │
│   └─────────────────────────────────────────────┘             │
│                          ↑ ↓                                   │
│   ┌─────────────────────────────────────────────┐             │
│   │               Workers                        │             │
│   │         (Flow 실행 에이전트)                  │             │
│   └─────────────────────────────────────────────┘             │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. 설치 및 시작하기

### 2.1 설치

```bash
# 기본 설치
pip install prefect

# 추가 통합 설치
pip install "prefect[aws]"      # AWS 통합
pip install "prefect[gcp]"      # GCP 통합
pip install "prefect[dask]"     # Dask 통합

# 버전 확인
prefect version
```

### 2.2 Prefect Cloud 연결 (선택사항)

```bash
# Prefect Cloud 로그인
prefect cloud login

# 또는 API 키 사용
prefect cloud login --key YOUR_API_KEY

# Self-hosted 서버 연결
prefect config set PREFECT_API_URL="http://localhost:4200/api"
```

### 2.3 로컬 서버 실행

```bash
# Prefect 서버 시작 (UI 포함)
prefect server start

# UI 접속: http://localhost:4200
```

---

## 3. Flow와 Task 기본

### 3.1 기본 Flow 작성

```python
from prefect import flow, task
from prefect.logging import get_run_logger

# Task 정의
@task
def extract_data(source: str) -> dict:
    """데이터 추출 Task"""
    logger = get_run_logger()
    logger.info(f"Extracting from {source}")

    # 실제로는 DB, API 등에서 추출
    data = {"source": source, "records": [1, 2, 3, 4, 5]}
    return data


@task
def transform_data(data: dict) -> dict:
    """데이터 변환 Task"""
    logger = get_run_logger()
    logger.info(f"Transforming {len(data['records'])} records")

    # 변환 로직
    data["records"] = [x * 2 for x in data["records"]]
    data["transformed"] = True
    return data


@task
def load_data(data: dict, destination: str) -> bool:
    """데이터 적재 Task"""
    logger = get_run_logger()
    logger.info(f"Loading to {destination}")

    # 실제로는 DB, 파일 등에 저장
    print(f"Loaded data: {data}")
    return True


# Flow 정의
@flow(name="ETL Pipeline")
def etl_pipeline(source: str = "database", destination: str = "warehouse"):
    """ETL 파이프라인 Flow"""
    # Task 실행 (자동 의존성 관리)
    raw_data = extract_data(source)
    transformed = transform_data(raw_data)
    result = load_data(transformed, destination)
    return result


# 로컬 실행
if __name__ == "__main__":
    etl_pipeline()
```

### 3.2 Task 옵션

```python
from prefect import task
from datetime import timedelta

@task(
    name="My Task",
    description="Task 설명",
    tags=["etl", "production"],
    retries=3,                          # 재시도 횟수
    retry_delay_seconds=60,             # 재시도 대기 시간
    timeout_seconds=3600,               # 타임아웃
    cache_key_fn=lambda: "static_key",  # 캐시 키
    cache_expiration=timedelta(hours=1), # 캐시 만료
    log_prints=True,                    # print 문을 로그로 캡처
)
def my_task(param: str) -> str:
    print(f"Processing: {param}")
    return f"Result: {param}"


# 동적 재시도 (exponential backoff)
from prefect.tasks import exponential_backoff

@task(
    retries=5,
    retry_delay_seconds=exponential_backoff(backoff_factor=10),
)
def flaky_task():
    """불안정한 외부 API 호출"""
    import random
    if random.random() < 0.7:
        raise Exception("Random failure")
    return "Success"
```

### 3.3 Flow 옵션

```python
from prefect import flow
from prefect.task_runners import ConcurrentTaskRunner, SequentialTaskRunner

@flow(
    name="My Flow",
    description="Flow 설명",
    version="1.0.0",
    retries=2,
    retry_delay_seconds=300,
    timeout_seconds=7200,
    task_runner=ConcurrentTaskRunner(),  # 병렬 실행
    log_prints=True,
    persist_result=True,                 # 결과 저장
)
def my_flow():
    pass


# 순차 실행
@flow(task_runner=SequentialTaskRunner())
def sequential_flow():
    pass
```

---

## 4. 동적 워크플로우

### 4.1 동적 Task 생성

```python
from prefect import flow, task

@task
def process_item(item: str) -> str:
    return f"Processed: {item}"


@flow
def dynamic_tasks_flow(items: list[str]):
    """동적으로 Task 수 결정"""
    results = []
    for item in items:
        result = process_item(item)
        results.append(result)
    return results


# 실행
dynamic_tasks_flow(["a", "b", "c", "d"])


# 병렬 실행 (.submit() 사용)
@flow
def parallel_tasks_flow(items: list[str]):
    """병렬로 Task 실행"""
    futures = []
    for item in items:
        # .submit()은 Future 반환 (비동기)
        future = process_item.submit(item)
        futures.append(future)

    # 결과 수집
    results = [f.result() for f in futures]
    return results
```

### 4.2 조건부 실행

```python
from prefect import flow, task

@task
def check_condition(data: dict) -> bool:
    return data.get("count", 0) > 100


@task
def process_large(data: dict):
    print(f"Processing large dataset: {data['count']} records")


@task
def process_small(data: dict):
    print(f"Processing small dataset: {data['count']} records")


@flow
def conditional_flow(data: dict):
    """조건에 따른 분기"""
    is_large = check_condition(data)

    if is_large:
        process_large(data)
    else:
        process_small(data)


# 실행
conditional_flow({"count": 150})  # process_large 실행
conditional_flow({"count": 50})   # process_small 실행
```

### 4.3 서브플로우

```python
from prefect import flow, task

@task
def extract(source: str) -> list:
    return [1, 2, 3, 4, 5]


@task
def transform(data: list) -> list:
    return [x * 2 for x in data]


@task
def load(data: list, target: str):
    print(f"Loading {len(data)} records to {target}")


# 서브플로우 정의
@flow(name="ETL Subflow")
def etl_subflow(source: str, target: str):
    """재사용 가능한 ETL 서브플로우"""
    data = extract(source)
    transformed = transform(data)
    load(transformed, target)
    return len(transformed)


# 메인 플로우
@flow(name="Main Pipeline")
def main_pipeline():
    """여러 서브플로우 오케스트레이션"""
    # 서브플로우 호출
    count_a = etl_subflow("source_a", "target_a")
    count_b = etl_subflow("source_b", "target_b")
    count_c = etl_subflow("source_c", "target_c")

    print(f"Total processed: {count_a + count_b + count_c}")


main_pipeline()
```

---

## 5. 배포 (Deployment)

### 5.1 Deployment 생성

```python
from prefect import flow
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

@flow
def my_etl_flow(date: str = None):
    """일일 ETL 플로우"""
    from datetime import datetime
    date = date or datetime.now().strftime("%Y-%m-%d")
    print(f"Running ETL for {date}")


# 방법 1: Python으로 Deployment 생성
deployment = Deployment.build_from_flow(
    flow=my_etl_flow,
    name="daily-etl",
    version="1.0",
    tags=["production", "etl"],
    schedule=CronSchedule(cron="0 6 * * *"),  # 매일 오전 6시
    parameters={"date": None},
    work_pool_name="default-agent-pool",
)

# Deployment 적용
deployment.apply()
```

### 5.2 CLI로 Deployment 생성

```bash
# prefect.yaml 생성
prefect init

# Deployment 빌드 및 적용
prefect deploy --name daily-etl
```

```yaml
# prefect.yaml 예시
name: my-project
prefect-version: 2.14.0

deployments:
  - name: daily-etl
    entrypoint: flows/etl.py:my_etl_flow
    work_pool:
      name: default-agent-pool
    schedule:
      cron: "0 6 * * *"
    parameters:
      date: null
    tags:
      - production
      - etl
```

### 5.3 Work Pool 및 Worker

```bash
# Work Pool 생성
prefect work-pool create my-pool --type process

# Worker 시작
prefect worker start --pool my-pool

# Docker 기반 Work Pool
prefect work-pool create docker-pool --type docker

# Kubernetes 기반 Work Pool
prefect work-pool create k8s-pool --type kubernetes
```

---

## 6. Airflow와의 비교 예제

### 6.1 Airflow 버전

```python
# Airflow DAG
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def extract(**kwargs):
    ti = kwargs['ti']
    data = [1, 2, 3, 4, 5]
    ti.xcom_push(key='data', value=data)

def transform(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(key='data', task_ids='extract')
    result = [x * 2 for x in data]
    ti.xcom_push(key='result', value=result)

def load(**kwargs):
    ti = kwargs['ti']
    result = ti.xcom_pull(key='result', task_ids='transform')
    print(f"Loading: {result}")

with DAG(
    'etl_airflow',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily',
    catchup=False,
) as dag:
    t1 = PythonOperator(task_id='extract', python_callable=extract)
    t2 = PythonOperator(task_id='transform', python_callable=transform)
    t3 = PythonOperator(task_id='load', python_callable=load)

    t1 >> t2 >> t3
```

### 6.2 Prefect 버전

```python
# Prefect Flow - 훨씬 간단하고 직관적
from prefect import flow, task

@task
def extract() -> list:
    return [1, 2, 3, 4, 5]

@task
def transform(data: list) -> list:
    return [x * 2 for x in data]

@task
def load(data: list):
    print(f"Loading: {data}")

@flow
def etl_prefect():
    data = extract()
    transformed = transform(data)
    load(transformed)

# 로컬 실행
etl_prefect()
```

### 6.3 주요 차이점

```python
"""
1. 데이터 전달:
   - Airflow: XCom 사용 (명시적 push/pull)
   - Prefect: 함수 반환값 직접 사용 (자연스러운 Python)

2. 의존성:
   - Airflow: >> 연산자로 명시
   - Prefect: 함수 호출 순서로 자동 추론

3. 스케줄링:
   - Airflow: Scheduler 프로세스 필수
   - Prefect: 선택적, 이벤트 기반 가능

4. 로컬 테스트:
   - Airflow: 복잡한 설정 필요
   - Prefect: 일반 Python 함수처럼 실행

5. 동적 워크플로우:
   - Airflow: 제한적 지원
   - Prefect: 네이티브 Python 제어문 사용
"""
```

---

## 7. 고급 기능

### 7.1 상태 핸들러

```python
from prefect import flow, task
from prefect.states import State, Completed, Failed

def custom_state_handler(task, task_run, state: State):
    """Task 상태 변경 시 호출"""
    if state.is_failed():
        # 슬랙 알림 등
        print(f"Task {task.name} failed!")
    return state


@task(on_failure=[custom_state_handler])
def risky_task():
    raise ValueError("Something went wrong")


# Flow 레벨 핸들러
@flow(on_failure=[lambda flow, flow_run, state: print("Flow failed!")])
def my_flow():
    risky_task()
```

### 7.2 결과 저장소

```python
from prefect import flow, task
from prefect.filesystems import S3, LocalFileSystem
from prefect.serializers import JSONSerializer

# 로컬 파일 시스템
@task(result_storage=LocalFileSystem(basepath="/tmp/prefect"))
def save_locally():
    return {"data": [1, 2, 3]}


# S3 저장소
@task(
    persist_result=True,
    result_storage=S3(bucket_path="my-bucket/results"),
    result_serializer=JSONSerializer(),
)
def save_to_s3():
    return {"large": "data"}
```

### 7.3 비밀 관리

```python
from prefect.blocks.system import Secret

# Block으로 비밀 저장 (UI 또는 CLI)
# prefect block register -m prefect.blocks.system

# 코드에서 사용
@task
def use_secret():
    api_key = Secret.load("my-api-key").get()
    # API 호출에 사용
    return f"Using key: {api_key[:4]}..."


# 환경 변수 사용
import os

@task
def use_env_var():
    return os.getenv("MY_SECRET")
```

---

## 연습 문제

### 문제 1: 기본 Flow 작성
3개의 Task(데이터 추출, 변환, 적재)로 구성된 ETL Flow를 작성하세요.

### 문제 2: 동적 Task
파일 목록을 입력받아 각 파일을 병렬로 처리하는 Flow를 작성하세요.

### 문제 3: 조건부 실행
데이터 크기에 따라 다른 처리 방식을 선택하는 Flow를 작성하세요.

---

## 요약

| 개념 | 설명 |
|------|------|
| **Flow** | 워크플로우 정의 (Airflow의 DAG) |
| **Task** | 개별 작업 단위 |
| **Deployment** | Flow의 배포 설정 |
| **Work Pool** | Worker 그룹 관리 |
| **Worker** | Flow 실행 에이전트 |

---

## 참고 자료

- [Prefect Documentation](https://docs.prefect.io/)
- [Prefect Tutorials](https://docs.prefect.io/tutorials/)
- [Prefect GitHub](https://github.com/PrefectHQ/prefect)
