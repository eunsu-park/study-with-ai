# Git & GitHub 학습 가이드

## 소개

이 폴더는 Git 버전 관리 시스템과 GitHub 협업 플랫폼을 학습하기 위한 자료를 담고 있습니다. 기본 명령어부터 CI/CD 자동화까지 단계별로 학습할 수 있습니다.

**대상 독자**: 개발자 입문자, 버전 관리를 배우고 싶은 분

---

## 학습 로드맵

```
[Git 기초]            [GitHub]              [고급]
    │                    │                    │
    ▼                    ▼                    ▼
Git 기초 ────────▶ GitHub 시작 ────▶ Git 고급
    │                    │                    │
    ▼                    ▼                    ▼
기본 명령어 ─────▶ GitHub 협업 ────▶ GitHub Actions
    │
    ▼
브랜치
```

---

## 선수 지식

- 터미널/명령줄 기본 사용법
- 텍스트 에디터 사용법

---

## 파일 목록

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [01_Git_기초.md](./01_Git_기초.md) | ⭐ | Git 개념, 설치, 초기 설정 |
| [02_기본_명령어.md](./02_기본_명령어.md) | ⭐ | status, add, commit, log |
| [03_브랜치.md](./03_브랜치.md) | ⭐⭐ | 브랜치 생성, 병합, 충돌 해결 |
| [04_GitHub_시작하기.md](./04_GitHub_시작하기.md) | ⭐ | 원격 저장소, push, pull, clone |
| [05_GitHub_협업.md](./05_GitHub_협업.md) | ⭐⭐ | Pull Request, 코드 리뷰, Fork |
| [06_Git_고급.md](./06_Git_고급.md) | ⭐⭐⭐ | rebase, cherry-pick, stash, reset |
| [07_GitHub_Actions.md](./07_GitHub_Actions.md) | ⭐⭐⭐ | CI/CD, 워크플로우 자동화 |

---

## 추천 학습 순서

### 1단계: Git 기초 (로컬)
1. Git 기초 → 기본 명령어 → 브랜치

### 2단계: GitHub 협업
2. GitHub 시작하기 → GitHub 협업

### 3단계: 고급 활용
3. Git 고급 → GitHub Actions

---

## 빠른 시작

### Git 설치

```bash
# macOS
brew install git

# Ubuntu
sudo apt-get install git

# 설치 확인
git --version
```

### 초기 설정

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

### 첫 커밋

```bash
git init
git add .
git commit -m "Initial commit"
```

---

## 관련 자료

- [Docker 학습](../Docker/00_Overview.md) - 개발 환경 컨테이너화
- [GitHub Actions](./07_GitHub_Actions.md) - CI/CD 자동화
