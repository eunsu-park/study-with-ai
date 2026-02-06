# Study Content (학습 자료)

학습 자료 Markdown 파일 모음입니다. 한국어(ko)와 영어(en) 두 언어로 관리됩니다.

## 디렉토리 구조

```
content/
├── ko/                    # 한국어 자료 (원본)
│   ├── Algorithm/         # 29개 레슨
│   ├── C_Programming/     # 20개 레슨
│   └── ...                # 총 28개 토픽
│
└── en/                    # 영어 자료 (번역본)
    ├── CPP/
    ├── C_Programming/
    └── ...                # 총 17개 토픽 (번역 진행 중)
```

## 토픽 목록

각 토픽의 `00_Overview.md`에서 학습 로드맵과 상세 목차를 확인할 수 있습니다.

| 폴더 | 내용 | 레슨 수 |
|------|------|---------|
| [Algorithm/](./ko/Algorithm/00_Overview.md) | 알고리즘/자료구조, 코딩테스트 준비 | 29개 |
| [C_Programming/](./ko/C_Programming/00_Overview.md) | C 언어 기초 ~ 임베디드, 디버깅, 포인터 심화 | 20개 |
| [Cloud_Computing/](./ko/Cloud_Computing/00_Overview.md) | AWS/GCP, EC2, Lambda, S3, VPC, IAM, Terraform | 17개 |
| [Computer_Architecture/](./ko/Computer_Architecture/00_Overview.md) | 컴퓨터 구조, CPU, 메모리, 병렬처리 | 18개 |
| [Computer_Vision/](./ko/Computer_Vision/00_Overview.md) | OpenCV, 이미지처리, 객체검출, 3D비전, SLAM | 23개 |
| [CPP/](./ko/CPP/00_Overview.md) | C++ 입문 ~ 모던 C++, 디자인패턴 | 18개 |
| [Data_Analysis/](./ko/Data_Analysis/00_Overview.md) | NumPy, Pandas, 시각화, 통계분석 | 11개 |
| [Data_Engineering/](./ko/Data_Engineering/00_Overview.md) | 데이터 파이프라인, Airflow, Spark, Kafka, dbt | 14개 |
| [Deep_Learning/](./ko/Deep_Learning/00_Overview.md) | PyTorch, CNN, RNN, Transformer, GAN, VAE, Diffusion, ViT | 22개 |
| [Docker/](./ko/Docker/00_Overview.md) | Docker, Kubernetes, Helm, CI/CD | 10개 |
| [Foundation_Models/](./ko/Foundation_Models/00_Overview.md) | Scaling Laws, LLaMA, Mistral, DINOv2, SAM, PEFT, RAG | 25개 |
| [Git/](./ko/Git/00_Overview.md) | Git, GitHub, 워크플로우, 모노레포 | 10개 |
| [IoT_Embedded/](./ko/IoT_Embedded/00_Overview.md) | Raspberry Pi, MQTT, Edge AI, 클라우드 IoT | 12개 |
| [Linux/](./ko/Linux/00_Overview.md) | Linux 기초 ~ HA 클러스터, 트러블슈팅 | 26개 |
| [LLM_and_NLP/](./ko/LLM_and_NLP/00_Overview.md) | NLP, BERT, GPT, HuggingFace, RAG, LangChain, RLHF, Agent | 16개 |
| [Machine_Learning/](./ko/Machine_Learning/00_Overview.md) | 회귀, 분류, 앙상블, 클러스터링, sklearn | 14개 |
| [MLOps/](./ko/MLOps/00_Overview.md) | MLflow, Kubeflow, 모델 서빙, 드리프트 감지 | 12개 |
| [Model_Implementations/](./ko/Model_Implementations/00_Overview.md) | 딥러닝 모델 from-scratch 구현 (NumPy, PyTorch) | 12개 모델 |
| [Networking/](./ko/Networking/00_Overview.md) | 네트워크, OSI/TCP-IP, 라우팅, 보안 | 17개 |
| [Numerical_Simulation/](./ko/Numerical_Simulation/00_Overview.md) | 수치해석, ODE/PDE, CFD, FDTD, MHD, 몬테카를로 | 20개 |
| [OS_Theory/](./ko/OS_Theory/00_Overview.md) | 운영체제, 프로세스, 메모리, 파일시스템 | 18개 |
| [PostgreSQL/](./ko/PostgreSQL/00_Overview.md) | SQL 기초 ~ 복제, 파티셔닝 | 18개 |
| [Python/](./ko/Python/00_Overview.md) | 파이썬 고급 문법, 테스트, 패키징 | 14개 |
| [Reinforcement_Learning/](./ko/Reinforcement_Learning/00_Overview.md) | MDP, Q-Learning, DQN, PPO, Multi-Agent RL | 12개 |
| [Spanish/](./ko/Spanish/00_Overview.md) | 스페인어 문법/어휘, 접속법, 지역 변형 | 11개 |
| [Statistics/](./ko/Statistics/00_Overview.md) | 심화통계, 베이지안, 시계열, 실험설계 | 14개 |
| [System_Design/](./ko/System_Design/00_Overview.md) | 시스템 설계, 확장성, 분산시스템 | 18개 |
| [Web_Development/](./ko/Web_Development/00_Overview.md) | HTML, CSS, JS, TypeScript, SEO | 13개 |

## 보기 방법

1. **웹 뷰어**: `viewer/` 폴더의 Flask 앱 실행
2. **직접 열기**: VS Code 등에서 Markdown 파일 열기
3. **GitHub**: GitHub에서 직접 렌더링

## 파일 명명 규칙

- `00_Overview.md` - 토픽 개요 및 목차
- `01_*.md`, `02_*.md`, ... - 순차적 레슨
- 한국어 파일명 사용 (공백은 언더스코어로 대체)

## 예제 코드

예제 코드 파일은 프로젝트 루트의 `examples/` 폴더에 위치합니다.

## 다국어 정책

- **한국어(ko)**: 원본 자료. 모든 토픽에 존재
- **영어(en)**: 번역본. `translate/` 도구를 사용하여 ko → en 번역
- 신규 자료 작성 시 한/영 두 버전 동시 작성 원칙
