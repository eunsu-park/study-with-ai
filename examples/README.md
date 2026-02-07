# Examples (예제 코드)

학습 자료에 대응하는 실행 가능한 예제 코드 모음입니다.

## 디렉토리 구조

```
examples/
├── Algorithm/              # Python, C, C++ 예제 (87개 파일)
│   ├── python/             # Python 구현 (29개)
│   ├── c/                  # C 구현 (29개 + Makefile)
│   └── cpp/                # C++ 구현 (29개 + Makefile)
│
├── C_Programming/          # C 프로젝트 예제 (41개 파일)
│   ├── 02_calculator/
│   ├── 03_number_guess/
│   ├── 04_address_book/    # 주소록 관리
│   ├── 05_dynamic_array/
│   ├── 06_linked_list/
│   ├── 07_file_crypto/     # 파일 암호화
│   ├── 08_stack_queue/     # 스택/큐 구현
│   ├── 09_hash_table/      # 해시 테이블
│   ├── 10_snake_game/      # 뱀 게임
│   ├── 11_minishell/       # 미니 쉘
│   ├── 12_multithread/     # 멀티스레딩
│   ├── 13_embedded_basic/
│   └── 14_bit_operations/
│
├── Computer_Vision/        # OpenCV/Python 예제 (20개 파일)
├── Data_Analysis/          # Pandas/NumPy 예제 (5개 파일)
├── Data_Engineering/       # Airflow/Spark/Kafka 예제 (6개 파일)
├── Deep_Learning/          # PyTorch 예제 (24개 파일)
│   ├── numpy/              # NumPy 기초 구현 (5개)
│   └── pytorch/            # PyTorch 구현 (19개)
│
├── IoT_Embedded/           # Raspberry Pi/MQTT 예제 (12개 파일)
│   ├── edge_ai/            # TFLite, ONNX 추론
│   ├── networking/         # WiFi, BLE, MQTT, HTTP
│   ├── projects/           # 스마트홈, 이미지분석, 클라우드IoT
│   └── raspberry_pi/       # GPIO, 센서
│
├── LLM_and_NLP/            # NLP/HuggingFace 예제 (15개 파일)
├── Machine_Learning/       # sklearn/Jupyter 예제 (14개 파일)
├── MLOps/                  # MLflow/서빙 예제 (5개 파일)
├── Numerical_Simulation/   # 수치해석 Python 예제 (6개 파일)
├── PostgreSQL/             # SQL 예제 (5개 파일)
├── Reinforcement_Learning/ # RL Python 예제 (10개 파일)
└── Web_Development/        # HTML/CSS/JS 프로젝트 (17개 파일)
```

**총 예제 파일: 267개**

## 빌드 방법

### C/C++ 예제 (Algorithm)

```bash
cd examples/Algorithm/c
make          # 전체 빌드
make clean    # 정리

cd examples/Algorithm/cpp
make          # 전체 빌드
```

### C 프로그래밍 예제

```bash
cd examples/C_Programming/<project>
make          # 프로젝트별 빌드
make clean    # 정리
```

### Python 예제

```bash
python examples/Algorithm/python/01_complexity.py
python examples/Reinforcement_Learning/06_q_learning.py
```

### Jupyter 노트북 (Machine_Learning)

```bash
cd examples/Machine_Learning
jupyter notebook
```

## 토픽별 예제 목록

| 토픽 | 파일 수 | 언어 | 설명 |
|------|---------|------|------|
| Algorithm | 87 | Python, C, C++ | 자료구조, 알고리즘 |
| C_Programming | 41 | C | 시스템 프로그래밍 프로젝트 |
| Computer_Vision | 20 | Python | OpenCV, 이미지 처리 |
| Data_Analysis | 5 | Python | NumPy, Pandas, 시각화 |
| Data_Engineering | 6 | Python | Airflow, Spark, Kafka |
| Deep_Learning | 24 | Python | PyTorch, CNN, RNN, Transformer |
| IoT_Embedded | 12 | Python | Raspberry Pi, MQTT, Edge AI |
| LLM_and_NLP | 15 | Python | BERT, GPT, RAG, LangChain |
| Machine_Learning | 14 | Python/Jupyter | sklearn, 분류, 회귀, 클러스터링 |
| MLOps | 5 | Python | MLflow, 모델 서빙, 드리프트 감지 |
| Numerical_Simulation | 6 | Python | 수치해석, ODE, Monte Carlo |
| PostgreSQL | 5 | SQL | CRUD, JOIN, 윈도우 함수 |
| Reinforcement_Learning | 10 | Python | Q-Learning, DQN, PPO, A2C |
| Web_Development | 17 | HTML/CSS/JS/TS | 웹 프로젝트, TypeScript |

## 예제와 학습 자료 매핑

예제 파일은 `content/` 의 학습 자료와 대응됩니다.

예시:
- `content/ko/Algorithm/01_Complexity_Analysis.md` → `examples/Algorithm/python/01_complexity.py`
- `content/ko/C_Programming/05_Project_Address_Book.md` → `examples/C_Programming/04_address_book/`
- `content/ko/Machine_Learning/04_Model_Evaluation.md` → `examples/Machine_Learning/03_model_evaluation.ipynb`
