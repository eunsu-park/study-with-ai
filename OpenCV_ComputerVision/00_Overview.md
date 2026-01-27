# OpenCV / Computer Vision 학습 가이드

## 소개

이 폴더는 OpenCV를 활용한 컴퓨터 비전을 체계적으로 학습하기 위한 자료를 담고 있습니다. 이미지 처리의 기초부터 딥러닝 기반 객체 검출까지 단계별로 학습할 수 있습니다.

**대상 독자**: Python 기초를 아는 개발자, 컴퓨터 비전 입문자, 영상 처리 프로젝트 준비자

---

## 학습 로드맵

```
[기초]                    [중급]                    [고급]
  │                         │                         │
  ▼                         ▼                         ▼
환경설정 (01) ──────▶ 필터링 (05) ──────▶ 특징점 검출 (13)
  │                         │                         │
  ▼                         ▼                         ▼
이미지 기초 (02) ───▶ 모폴로지 (06) ───▶ 특징점 매칭 (14)
  │                         │                         │
  ▼                         ▼                         ▼
색상 공간 (03) ─────▶ 이진화 (07) ─────▶ 객체 검출 (15)
  │                         │                         │
  ▼                         ▼                         ▼
기하 변환 (04) ─────▶ 엣지 검출 (08) ──▶ 얼굴 인식 (16)
                            │                         │
                            ▼                         ▼
                     윤곽선 검출 (09) ──▶ 비디오 처리 (17)
                            │                         │
                            ▼                         ▼
                     도형 분석 (10) ────▶ 캘리브레이션 (18)
                            │                         │
                     ┌──────┴──────┐                  ▼
                     ▼             ▼           DNN 모듈 (19)
               허프 변환 (11)  히스토그램 (12)        │
                                                      ▼
                                              실전 프로젝트 (20)
```

---

## 선수 지식

### 필수

- Python 기초 (변수, 제어문, 함수, 클래스)
- NumPy 기초 (ndarray, 인덱싱, 슬라이싱, 브로드캐스팅)
- 파일 I/O, 예외 처리

### 권장

- 선형대수 기초 (행렬 연산)
- 확률/통계 기초
- 머신러닝 개념 (분류, 학습)

---

## 파일 목록

### 기초 (01-04)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [01_환경_설정_및_기초.md](./01_환경_설정_및_기초.md) | ⭐ | OpenCV 설치, opencv-python vs contrib, 버전 확인 |
| [02_이미지_기초_연산.md](./02_이미지_기초_연산.md) | ⭐ | imread, imshow, imwrite, 픽셀 접근, ROI |
| [03_색상_공간.md](./03_색상_공간.md) | ⭐⭐ | BGR/RGB, HSV, LAB, cvtColor, 채널 분리 |
| [04_기하학적_변환.md](./04_기하학적_변환.md) | ⭐⭐ | resize, rotate, flip, warpAffine, warpPerspective |

### 이미지 처리 (05-08)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [05_이미지_필터링.md](./05_이미지_필터링.md) | ⭐⭐ | blur, GaussianBlur, medianBlur, bilateralFilter |
| [06_모폴로지_연산.md](./06_모폴로지_연산.md) | ⭐⭐ | erode, dilate, opening, closing, gradient |
| [07_이진화_및_임계처리.md](./07_이진화_및_임계처리.md) | ⭐⭐ | threshold, OTSU, adaptiveThreshold |
| [08_엣지_검출.md](./08_엣지_검출.md) | ⭐⭐⭐ | Sobel, Scharr, Laplacian, Canny |

### 객체 분석 (09-12)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [09_윤곽선_검출.md](./09_윤곽선_검출.md) | ⭐⭐⭐ | findContours, drawContours, 계층 구조, approxPolyDP |
| [10_도형_분석.md](./10_도형_분석.md) | ⭐⭐⭐ | moments, boundingRect, convexHull, matchShapes |
| [11_허프_변환.md](./11_허프_변환.md) | ⭐⭐⭐ | HoughLines, HoughLinesP, HoughCircles |
| [12_히스토그램_분석.md](./12_히스토그램_분석.md) | ⭐⭐ | calcHist, equalizeHist, CLAHE, backprojection |

### 특징 및 검출 (13-15)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [13_특징점_검출.md](./13_특징점_검출.md) | ⭐⭐⭐ | Harris, FAST, SIFT, ORB, 키포인트/디스크립터 |
| [14_특징점_매칭.md](./14_특징점_매칭.md) | ⭐⭐⭐ | BFMatcher, FLANN, ratio test, homography |
| [15_객체_검출_기초.md](./15_객체_검출_기초.md) | ⭐⭐⭐ | template matching, Haar cascade, HOG+SVM |

### 고급 주제 (16-18)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [16_얼굴_검출_및_인식.md](./16_얼굴_검출_및_인식.md) | ⭐⭐⭐⭐ | Haar/dlib 얼굴검출, 랜드마크, LBPH, face_recognition |
| [17_비디오_처리.md](./17_비디오_처리.md) | ⭐⭐⭐ | VideoCapture, VideoWriter, 배경차분, 옵티컬플로우 |
| [18_카메라_캘리브레이션.md](./18_카메라_캘리브레이션.md) | ⭐⭐⭐⭐ | 카메라 행렬, 왜곡 보정, 체스보드 캘리브레이션 |

### DNN 및 실전 (19-20)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [19_딥러닝_DNN_모듈.md](./19_딥러닝_DNN_모듈.md) | ⭐⭐⭐⭐ | cv2.dnn, readNet, blobFromImage, YOLO, SSD |
| [20_실전_프로젝트.md](./20_실전_프로젝트.md) | ⭐⭐⭐⭐ | 문서스캐너, 차선검출, AR마커, 얼굴필터 |

---

## 추천 학습 순서

### 빠른 시작 (1주)
```
01_환경_설정 → 02_이미지_기초 → 03_색상_공간 → 05_필터링 → 07_이진화 → 08_엣지_검출 → 09_윤곽선
```

### 기본 완성 (2~3주)
```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12
```

### 중급 완성 (4~5주)
```
기본 완성 + 13 → 14 → 15 → 16 → 17
```

### 전체 마스터 (6~8주)
```
중급 완성 + 18 → 19 → 20
```

---

## 실습 환경

### 설치

```bash
# 기본 설치 (대부분의 기능)
pip install opencv-python numpy matplotlib

# 확장 설치 (SIFT, SURF 등 추가 기능)
pip install opencv-contrib-python

# 얼굴 인식용
pip install dlib face_recognition

# 버전 확인
python -c "import cv2; print(cv2.__version__)"
```

### 추천 환경

```
- Python 3.8 이상
- OpenCV 4.x
- IDE: VSCode, PyCharm, Jupyter Notebook
- OS: Windows, macOS, Linux 모두 지원
```

### 프로젝트 구조 예시

```
my_cv_project/
├── images/           # 입력 이미지
├── output/           # 출력 결과
├── models/           # 학습된 모델 (Haar, DNN 등)
├── src/              # 소스 코드
│   ├── preprocessing.py
│   ├── detection.py
│   └── utils.py
├── notebooks/        # Jupyter 실험
└── requirements.txt
```

---

## OpenCV 주요 함수 빠른 참조

### 이미지 I/O

| 함수 | 설명 | 예시 |
|------|------|------|
| `cv2.imread()` | 이미지 읽기 | `img = cv2.imread('image.jpg')` |
| `cv2.imshow()` | 이미지 표시 | `cv2.imshow('Window', img)` |
| `cv2.imwrite()` | 이미지 저장 | `cv2.imwrite('out.jpg', img)` |

### 색상 변환

| 함수 | 설명 | 예시 |
|------|------|------|
| `cv2.cvtColor()` | 색상 공간 변환 | `gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` |
| `cv2.split()` | 채널 분리 | `b, g, r = cv2.split(img)` |
| `cv2.merge()` | 채널 병합 | `img = cv2.merge([b, g, r])` |

### 기하 변환

| 함수 | 설명 | 예시 |
|------|------|------|
| `cv2.resize()` | 크기 조정 | `resized = cv2.resize(img, (w, h))` |
| `cv2.rotate()` | 회전 | `rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)` |
| `cv2.warpAffine()` | 어파인 변환 | `dst = cv2.warpAffine(img, M, (w, h))` |

### 필터링

| 함수 | 설명 | 예시 |
|------|------|------|
| `cv2.GaussianBlur()` | 가우시안 블러 | `blur = cv2.GaussianBlur(img, (5, 5), 0)` |
| `cv2.Canny()` | 캐니 엣지 | `edges = cv2.Canny(img, 100, 200)` |
| `cv2.threshold()` | 임계값 처리 | `_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)` |

### 윤곽선/도형

| 함수 | 설명 | 예시 |
|------|------|------|
| `cv2.findContours()` | 윤곽선 검출 | `contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)` |
| `cv2.drawContours()` | 윤곽선 그리기 | `cv2.drawContours(img, contours, -1, (0, 255, 0), 2)` |
| `cv2.boundingRect()` | 경계 사각형 | `x, y, w, h = cv2.boundingRect(contour)` |

---

## 관련 자료

### 다른 폴더와의 연계

| 폴더 | 관련 내용 |
|------|----------|
| [Python/](../Python/00_Overview.md) | 고급 Python 문법, 테스트, 패키징 |
| [Algorithm/](../Algorithm/00_Overview.md) | 이미지 처리용 알고리즘 (그래프, DP) |
| [Linux/](../Linux/00_Overview.md) | 개발 환경, 파일 처리 |

### 외부 자료

- [OpenCV 공식 문서](https://docs.opencv.org/)
- [OpenCV-Python 튜토리얼](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [PyImageSearch](https://pyimagesearch.com/) - 실전 튜토리얼
- [Learn OpenCV](https://learnopencv.com/) - 고급 예제

---

## 학습 팁

1. **실습 중심**: 코드를 직접 실행하며 결과 확인
2. **시각화**: matplotlib으로 중간 결과 시각화
3. **파라미터 튜닝**: 트랙바(createTrackbar)로 실시간 조정
4. **단계별 처리**: 복잡한 작업은 파이프라인으로 분해
5. **디버깅**: imshow로 각 단계의 이미지 확인

---

## 면접 대비 핵심 주제

| 주제 | 핵심 질문 |
|------|----------|
| 색상 공간 | RGB vs HSV - 언제 HSV를 사용하는가? |
| 필터링 | Gaussian vs Bilateral - 차이점은? |
| 이진화 | Otsu's method의 원리는? |
| 엣지 검출 | Canny 알고리즘의 단계별 동작은? |
| 특징점 | SIFT vs ORB - 장단점 비교 |
| 객체 검출 | Haar cascade vs HOG+SVM 차이점 |
| DNN | YOLO vs SSD - 속도와 정확도 트레이드오프 |

