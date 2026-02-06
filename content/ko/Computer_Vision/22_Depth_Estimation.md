# 단안 깊이 추정 (Monocular Depth Estimation)

## 개요

단안 깊이 추정은 단일 2D 이미지에서 픽셀별 깊이 정보를 추정하는 기술입니다. MiDaS, DPT 같은 딥러닝 모델과 Structure from Motion (SfM)을 통한 기하학적 접근 방법을 다룹니다.

**난이도**: ⭐⭐⭐⭐

**선수 지식**: DNN 모듈, 특징점 검출/매칭, 카메라 캘리브레이션

---

## 목차

1. [단안 깊이 추정 개요](#1-단안-깊이-추정-개요)
2. [MiDaS 모델](#2-midas-모델)
3. [DPT (Dense Prediction Transformer)](#3-dpt-dense-prediction-transformer)
4. [Structure from Motion (SfM)](#4-structure-from-motion-sfm)
5. [깊이 맵 응용](#5-깊이-맵-응용)
6. [연습 문제](#6-연습-문제)

---

## 1. 단안 깊이 추정 개요

### 왜 단안 깊이 추정인가?

```
스테레오 vs 단안 깊이 추정:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  스테레오 비전                                                  │
│  ┌───────────┐    ┌───────────┐                                 │
│  │   📷      │    │     📷    │                                 │
│  │   Left    │◄──►│   Right   │  두 카메라 필요                 │
│  └───────────┘    └───────────┘                                 │
│                                                                 │
│  장점: 기하학적으로 정확, 절대 깊이 측정 가능                   │
│  단점: 두 카메라 필요, 캘리브레이션 필수                        │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  단안 깊이 추정                                                 │
│  ┌───────────┐                                                  │
│  │    📷     │  단일 카메라로 가능                              │
│  │  Single   │  스마트폰, 드론, 로봇 등에 적합                  │
│  └───────────┘                                                  │
│                                                                 │
│  장점: 단일 카메라, 간단한 설정, 이동 장치에 적합               │
│  단점: 상대적 깊이, 스케일 모호성, 학습 데이터 의존             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 깊이 추정의 어려움

```
단안 깊이 추정의 본질적 모호성:

동일한 2D 이미지를 생성하는 무한히 많은 3D 장면이 존재

                        │
                        │
         ●              │         🎾  작은 공, 가까이
        /│\             │
         │              │
                        │
                        │         🏀  큰 공, 멀리
    ───────────────────[📷]───────────────────

같은 크기로 보임!

해결 방법:
1. 학습된 사전 지식 (딥러닝)
   - 물체의 일반적인 크기
   - 원근감 규칙
   - 텍스처 그래디언트

2. 다중 이미지 (SfM)
   - 시점 변화를 이용
   - 기하학적 제약

3. 추가 센서
   - LiDAR 보조
   - 구조광 보조
```

### 깊이 추정 방법론

```
깊이 추정 접근법:

┌─────────────────────────────────────────────────────────────────┐
│ 1. 지도 학습 (Supervised Learning)                              │
│    - RGB-D 데이터셋으로 학습                                    │
│    - Ground Truth 깊이 필요                                     │
│    - 데이터셋: NYU Depth V2, KITTI, ScanNet                    │
│                                                                 │
│ 2. 자기지도 학습 (Self-supervised Learning)                     │
│    - 스테레오 쌍 또는 연속 프레임으로 학습                      │
│    - Ground Truth 불필요                                        │
│    - Monodepth2, PackNet-SfM                                   │
│                                                                 │
│ 3. 제로샷 학습 (Zero-shot / Cross-domain)                       │
│    - 다양한 데이터셋에서 사전 학습                              │
│    - 새로운 도메인에 일반화                                     │
│    - MiDaS, DPT, ZoeDepth                                      │
│                                                                 │
│ 4. 기하학적 방법 (Geometric Methods)                            │
│    - Structure from Motion                                      │
│    - Multi-View Stereo                                          │
│    - 명시적 기하학적 제약 사용                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. MiDaS 모델

### MiDaS 개요

```
MiDaS (Mixing Datasets for Monocular Depth Estimation):

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  핵심 아이디어: 다양한 데이터셋을 혼합하여 일반화 능력 향상     │
│                                                                 │
│  학습 데이터:                                                   │
│  - ReDWeb (인터넷 이미지)                                       │
│  - DIML (실내)                                                  │
│  - Movies (영화 장면)                                           │
│  - MegaDepth (야외)                                             │
│  - WSVD (비디오)                                                │
│                                                                 │
│  특징:                                                          │
│  - 스케일 불변 (scale-invariant) 손실 함수                      │
│  - 상대적 깊이 예측                                             │
│  - 다양한 백본 (EfficientNet, ResNeXt, ViT)                    │
│                                                                 │
│  모델 버전:                                                     │
│  ┌──────────────────┬───────────┬─────────────────────────┐     │
│  │ 모델             │ 입력 크기 │ 특징                    │     │
│  ├──────────────────┼───────────┼─────────────────────────┤     │
│  │ MiDaS v2.1 Large │ 384x384   │ 고품질, 느림            │     │
│  │ MiDaS v2.1 Small │ 256x256   │ 경량, 빠름              │     │
│  │ MiDaS v3 (DPT)   │ 384x384   │ Transformer 기반        │     │
│  │ MiDaS v3.1 (DPT) │ 다양      │ 최신, 다양한 백본       │     │
│  └──────────────────┴───────────┴─────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### MiDaS 사용하기

```python
import cv2
import numpy as np
import torch

def load_midas_model(model_type='DPT_Large'):
    """MiDaS 모델 로드 (PyTorch Hub)"""

    # 모델 타입:
    # - 'DPT_Large': 가장 정확
    # - 'DPT_Hybrid': 균형
    # - 'MiDaS_small': 가장 빠름

    model = torch.hub.load('intel-isl/MiDaS', model_type)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # 전처리 트랜스폼
    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')

    if model_type in ['DPT_Large', 'DPT_Hybrid']:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return model, transform, device

def estimate_depth_midas(img, model, transform, device):
    """MiDaS로 깊이 추정"""

    # BGR → RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 전처리
    input_batch = transform(img_rgb).to(device)

    # 추론
    with torch.no_grad():
        prediction = model(input_batch)

        # 원본 크기로 리사이즈
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    return depth_map

def normalize_depth(depth_map):
    """깊이 맵 정규화 (시각화용)"""

    depth_min = depth_map.min()
    depth_max = depth_map.max()

    depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
    depth_normalized = (depth_normalized * 255).astype(np.uint8)

    return depth_normalized

def colorize_depth(depth_map, colormap=cv2.COLORMAP_INFERNO):
    """깊이 맵에 컬러맵 적용"""

    depth_norm = normalize_depth(depth_map)
    depth_colored = cv2.applyColorMap(depth_norm, colormap)

    return depth_colored

# 사용 예
def main():
    # 모델 로드
    print("모델 로딩 중...")
    model, transform, device = load_midas_model('DPT_Large')

    # 이미지 로드
    img = cv2.imread('sample.jpg')

    # 깊이 추정
    print("깊이 추정 중...")
    depth = estimate_depth_midas(img, model, transform, device)

    # 시각화
    depth_colored = colorize_depth(depth)

    cv2.imshow('Original', img)
    cv2.imshow('Depth', depth_colored)
    cv2.waitKey(0)
```

### OpenCV DNN으로 MiDaS 실행

```python
import cv2
import numpy as np

class MiDaSDepthEstimator:
    """OpenCV DNN으로 MiDaS 실행"""

    def __init__(self, model_path):
        """
        model_path: ONNX 모델 경로
        다운로드: https://github.com/isl-org/MiDaS/releases
        """
        self.net = cv2.dnn.readNetFromONNX(model_path)

        # GPU 사용 (가능한 경우)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # 입력 크기 (모델에 따라 다름)
        self.input_size = (384, 384)  # DPT_Large
        # self.input_size = (256, 256)  # MiDaS_small

    def estimate(self, img):
        """깊이 추정"""

        h, w = img.shape[:2]

        # 전처리
        blob = cv2.dnn.blobFromImage(
            img,
            scalefactor=1/255.0,
            size=self.input_size,
            mean=(0.485, 0.456, 0.406),  # ImageNet mean
            swapRB=True,
            crop=False
        )

        # 표준편차 정규화 (수동)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        blob = blob / std

        # 추론
        self.net.setInput(blob)
        output = self.net.forward()

        # 후처리
        depth = output[0, 0]

        # 원본 크기로 리사이즈
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)

        return depth

    def visualize(self, depth, colormap=cv2.COLORMAP_MAGMA):
        """깊이 맵 시각화"""

        # 정규화
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)

        # 컬러맵 적용
        depth_colored = cv2.applyColorMap(depth_norm, colormap)

        return depth_colored

# 사용 예
estimator = MiDaSDepthEstimator('midas_v21_384.onnx')

img = cv2.imread('sample.jpg')
depth = estimator.estimate(img)
depth_vis = estimator.visualize(depth)

cv2.imshow('Depth', depth_vis)
cv2.waitKey(0)
```

---

## 3. DPT (Dense Prediction Transformer)

### DPT 아키텍처

```
DPT (Dense Prediction Transformer):

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Vision Transformer (ViT) 기반 밀집 예측 모델                   │
│                                                                 │
│  입력: 이미지 (H × W × 3)                                       │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Patch Embedding                                        │    │
│  │  이미지를 패치로 분할 후 임베딩                         │    │
│  │  패치 크기: 16×16                                       │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Transformer Encoder                                    │    │
│  │  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐               │    │
│  │  │ Block │→│ Block │→│ Block │→│ Block │               │    │
│  │  └───────┘ └───────┘ └───────┘ └───────┘               │    │
│  │     │          │          │          │                  │    │
│  │     └──────────┼──────────┼──────────┘                  │    │
│  │                ▼          ▼          ▼                  │    │
│  │         다중 스케일 특징 추출                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Reassemble + Fusion                                    │    │
│  │  다중 스케일 특징 융합                                  │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Head (Conv Layers)                                     │    │
│  │  최종 깊이 맵 출력                                      │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                     │
│                           ▼                                     │
│  출력: 깊이 맵 (H × W)                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### DPT 구현

```python
import cv2
import numpy as np
import torch
from torchvision import transforms

class DPTDepthEstimator:
    """DPT 깊이 추정기"""

    def __init__(self, model_type='DPT_Large'):
        """
        model_type: 'DPT_Large', 'DPT_Hybrid', 'DPT_SwinV2_L_384'
        """
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # PyTorch Hub에서 모델 로드
        self.model = torch.hub.load('intel-isl/MiDaS', model_type)
        self.model.to(self.device)
        self.model.eval()

        # 전처리 트랜스폼 로드
        midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        self.transform = midas_transforms.dpt_transform

    def estimate(self, img):
        """깊이 추정"""

        h, w = img.shape[:2]

        # BGR → RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 전처리 및 추론
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)

            # 원본 크기로 보간
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode='bicubic',
                align_corners=False
            ).squeeze()

        depth = prediction.cpu().numpy()

        return depth

    def get_metric_depth(self, depth, scale=10.0):
        """상대 깊이 → 미터 단위 변환 (근사)"""

        # MiDaS/DPT는 상대 깊이를 출력
        # 절대 깊이로 변환하려면 스케일 추정 필요

        depth_metric = scale / (depth + 1e-6)

        return depth_metric

def estimate_depth_with_confidence(estimator, img, num_samples=5):
    """몬테카를로 드롭아웃으로 깊이 불확실성 추정"""

    # 참고: 실제로는 드롭아웃이 있는 모델이 필요
    # 여기서는 데이터 증강으로 대체

    depths = []

    for _ in range(num_samples):
        # 약간의 이미지 변형
        augmented = img.copy()

        # 밝기 변화
        factor = np.random.uniform(0.9, 1.1)
        augmented = np.clip(augmented * factor, 0, 255).astype(np.uint8)

        depth = estimator.estimate(augmented)
        depths.append(depth)

    depths = np.stack(depths, axis=0)

    # 평균과 표준편차
    mean_depth = np.mean(depths, axis=0)
    std_depth = np.std(depths, axis=0)

    return mean_depth, std_depth
```

### Depth Anything 모델

```python
# Depth Anything: 더 최신의 SOTA 모델

class DepthAnythingEstimator:
    """Depth Anything 모델 (2024)"""

    def __init__(self, model_size='small'):
        """
        model_size: 'small', 'base', 'large'
        """
        from transformers import pipeline

        model_name = f"LiheYoung/depth-anything-{model_size}-hf"
        self.pipe = pipeline(
            task='depth-estimation',
            model=model_name
        )

    def estimate(self, img):
        """깊이 추정"""

        # BGR → RGB, PIL 변환
        from PIL import Image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # 추론
        result = self.pipe(img_pil)

        # 깊이 맵 추출
        depth = np.array(result['depth'])

        # 원본 크기로 리사이즈
        if depth.shape[:2] != img.shape[:2]:
            depth = cv2.resize(depth, (img.shape[1], img.shape[0]))

        return depth
```

---

## 4. Structure from Motion (SfM)

### SfM 개요

```
Structure from Motion (SfM):
카메라 움직임을 이용해 3D 구조 복원

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  입력: 연속 이미지 (비디오 또는 다중 뷰 이미지)                 │
│                                                                 │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                   │
│  │ t=1 │  │ t=2 │  │ t=3 │  │ t=4 │  │ t=5 │                   │
│  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘                   │
│      │       │       │       │       │                          │
│      └───────┴───────┴───────┴───────┘                          │
│                      │                                          │
│                      ▼                                          │
│          ┌───────────────────────────┐                          │
│          │  1. 특징점 검출 및 매칭   │                          │
│          │     SIFT, ORB, SuperPoint │                          │
│          └───────────────────────────┘                          │
│                      │                                          │
│                      ▼                                          │
│          ┌───────────────────────────┐                          │
│          │  2. 카메라 포즈 추정      │                          │
│          │     Essential Matrix      │                          │
│          │     PnP                   │                          │
│          └───────────────────────────┘                          │
│                      │                                          │
│                      ▼                                          │
│          ┌───────────────────────────┐                          │
│          │  3. 삼각측량              │                          │
│          │     3D 점 복원            │                          │
│          └───────────────────────────┘                          │
│                      │                                          │
│                      ▼                                          │
│          ┌───────────────────────────┐                          │
│          │  4. 번들 조정             │                          │
│          │     전역 최적화           │                          │
│          └───────────────────────────┘                          │
│                      │                                          │
│                      ▼                                          │
│  출력: 3D 포인트 클라우드 + 카메라 궤적                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### SfM 구현 (간단한 버전)

```python
import cv2
import numpy as np

class SimpleSfM:
    """간단한 2-뷰 SfM 구현"""

    def __init__(self, K):
        """
        K: 카메라 내부 파라미터 행렬
        """
        self.K = K
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()

    def detect_and_match(self, img1, img2):
        """특징점 검출 및 매칭"""

        # 특징점 검출
        kp1, desc1 = self.sift.detectAndCompute(img1, None)
        kp2, desc2 = self.sift.detectAndCompute(img2, None)

        # 매칭
        matches = self.bf.knnMatch(desc1, desc2, k=2)

        # 비율 테스트
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # 매칭점 좌표
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        return pts1, pts2, good_matches, kp1, kp2

    def estimate_pose(self, pts1, pts2):
        """Essential Matrix로 상대 포즈 추정"""

        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        # R, t 복구
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K, mask)

        return R, t, mask.ravel().astype(bool)

    def triangulate(self, pts1, pts2, R, t):
        """삼각측량으로 3D 점 복원"""

        # 투영 행렬
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t])

        # 삼각측량
        pts1_h = pts1.T  # (2, N)
        pts2_h = pts2.T

        points_4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)

        # 동차 좌표 → 유클리드 좌표
        points_3d = points_4d[:3] / points_4d[3]

        return points_3d.T  # (N, 3)

    def filter_points(self, pts1, pts2, points_3d, R, t):
        """유효한 3D 점 필터링"""

        # 재투영 오차 계산
        P2 = self.K @ np.hstack([R, t])

        projected = P2 @ np.hstack([points_3d, np.ones((len(points_3d), 1))]).T
        projected = projected[:2] / projected[2]
        projected = projected.T

        errors = np.linalg.norm(pts2 - projected, axis=1)

        # 카메라 앞에 있는지 확인
        # 첫 번째 카메라 기준
        valid_depth1 = points_3d[:, 2] > 0

        # 두 번째 카메라 기준
        points_cam2 = (R @ points_3d.T + t).T
        valid_depth2 = points_cam2[:, 2] > 0

        # 재투영 오차 임계값
        valid_reproj = errors < 2.0

        valid = valid_depth1 & valid_depth2 & valid_reproj

        return points_3d[valid], valid

    def run(self, img1, img2):
        """전체 SfM 파이프라인 실행"""

        # 1. 특징점 매칭
        pts1, pts2, matches, kp1, kp2 = self.detect_and_match(img1, img2)
        print(f"매칭점 수: {len(pts1)}")

        # 2. 포즈 추정
        R, t, inlier_mask = self.estimate_pose(pts1, pts2)
        pts1 = pts1[inlier_mask]
        pts2 = pts2[inlier_mask]
        print(f"인라이어 수: {len(pts1)}")

        # 3. 삼각측량
        points_3d = self.triangulate(pts1, pts2, R, t)

        # 4. 필터링
        points_3d, valid = self.filter_points(pts1, pts2, points_3d, R, t)
        print(f"유효한 3D 점 수: {len(points_3d)}")

        return points_3d, R, t

# 사용 예
K = np.array([
    [800, 0, 320],
    [0, 800, 240],
    [0, 0, 1]
], dtype=np.float32)

sfm = SimpleSfM(K)
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')
points_3d, R, t = sfm.run(img1, img2)
```

### 다중 뷰 SfM

```python
class IncrementalSfM:
    """증분적 SfM"""

    def __init__(self, K):
        self.K = K
        self.sift = cv2.SIFT_create(nfeatures=8000)
        self.bf = cv2.BFMatcher()

        # 전역 데이터
        self.points_3d = None
        self.point_colors = None
        self.camera_poses = []
        self.keypoints_all = []
        self.descriptors_all = []

    def add_image(self, img):
        """새 이미지 추가"""

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = self.sift.detectAndCompute(gray, None)

        self.keypoints_all.append(kp)
        self.descriptors_all.append(desc)

        return len(self.keypoints_all) - 1

    def initialize(self, idx1, idx2):
        """첫 두 이미지로 초기화"""

        # 매칭
        matches = self.bf.knnMatch(
            self.descriptors_all[idx1],
            self.descriptors_all[idx2],
            k=2
        )

        good = [m for m, n in matches if m.distance < 0.7 * n.distance]

        pts1 = np.float32([self.keypoints_all[idx1][m.queryIdx].pt for m in good])
        pts2 = np.float32([self.keypoints_all[idx2][m.trainIdx].pt for m in good])

        # Essential Matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K)
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)

        mask = mask.ravel().astype(bool)
        pts1 = pts1[mask]
        pts2 = pts2[mask]

        # 삼각측량
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t])

        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        self.points_3d = (points_4d[:3] / points_4d[3]).T

        # 카메라 포즈 저장
        self.camera_poses = [
            {'R': np.eye(3), 't': np.zeros((3, 1))},
            {'R': R, 't': t}
        ]

        print(f"초기화 완료: {len(self.points_3d)} 3D 점")

    def register_image(self, idx):
        """새 이미지 등록 (PnP)"""

        if self.points_3d is None or len(self.points_3d) == 0:
            print("먼저 초기화가 필요합니다.")
            return False

        # 마지막으로 추가된 이미지와 매칭
        last_idx = len(self.camera_poses) - 1

        matches = self.bf.knnMatch(
            self.descriptors_all[last_idx],
            self.descriptors_all[idx],
            k=2
        )

        good = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good) < 8:
            print("매칭점 부족")
            return False

        # 3D-2D 대응점 (단순화: 이전 이미지의 매칭점 인덱스 사용)
        # 실제로는 트랙 관리 필요
        obj_points = []
        img_points = []

        for m in good[:len(self.points_3d)]:
            if m.queryIdx < len(self.points_3d):
                obj_points.append(self.points_3d[m.queryIdx])
                img_points.append(
                    self.keypoints_all[idx][m.trainIdx].pt
                )

        if len(obj_points) < 6:
            print("대응점 부족")
            return False

        obj_points = np.array(obj_points, dtype=np.float32)
        img_points = np.array(img_points, dtype=np.float32)

        # PnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_points, img_points, self.K, None
        )

        if not success:
            print("PnP 실패")
            return False

        R, _ = cv2.Rodrigues(rvec)
        self.camera_poses.append({'R': R, 't': tvec})

        print(f"이미지 {idx} 등록 완료")
        return True

    def bundle_adjust(self):
        """번들 조정 (scipy 사용)"""

        from scipy.optimize import least_squares

        # 간단한 번들 조정 구현
        # 실제로는 g2o, Ceres 등 사용 권장

        print("번들 조정은 별도 라이브러리 권장 (g2o, Ceres)")

    def get_point_cloud(self):
        """포인트 클라우드 반환"""
        return self.points_3d

    def get_camera_trajectory(self):
        """카메라 궤적 반환"""
        positions = []
        for pose in self.camera_poses:
            R = pose['R']
            t = pose['t']
            # 카메라 위치 = -R^T * t
            pos = -R.T @ t
            positions.append(pos.ravel())

        return np.array(positions)
```

---

## 5. 깊이 맵 응용

### 깊이 기반 이미지 효과

```python
import cv2
import numpy as np

def apply_bokeh_effect(img, depth, focus_depth=0.5, aperture=0.1):
    """깊이 기반 보케 효과 (피사계 심도 시뮬레이션)"""

    # 깊이 정규화 (0-1)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())

    # 초점 거리에서의 편차 계산
    depth_diff = np.abs(depth_norm - focus_depth)

    # 블러 강도 (초점에서 멀수록 강함)
    blur_strength = (depth_diff / aperture * 30).astype(int)
    blur_strength = np.clip(blur_strength, 0, 31)

    # 블러 적용 (픽셀별로 다른 강도)
    result = np.zeros_like(img, dtype=np.float32)

    for blur_level in range(0, 32, 2):
        mask = (blur_strength >= blur_level) & (blur_strength < blur_level + 2)

        if blur_level == 0:
            blurred = img.astype(np.float32)
        else:
            ksize = blur_level * 2 + 1
            blurred = cv2.GaussianBlur(img, (ksize, ksize), 0).astype(np.float32)

        result += blurred * mask[:, :, np.newaxis]

    return result.astype(np.uint8)

def create_depth_fog(img, depth, fog_color=(200, 200, 200), max_fog=0.8):
    """깊이 기반 안개 효과"""

    # 깊이 정규화
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())

    # 안개 강도 (멀수록 강함)
    fog_factor = depth_norm * max_fog

    # 안개 적용
    fog = np.full_like(img, fog_color, dtype=np.float32)
    result = img.astype(np.float32) * (1 - fog_factor[:, :, np.newaxis])
    result += fog * fog_factor[:, :, np.newaxis]

    return result.astype(np.uint8)

def depth_based_segmentation(img, depth, num_layers=5):
    """깊이 기반 레이어 분할"""

    # 깊이 정규화
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())

    # 깊이 구간으로 분할
    layers = []
    for i in range(num_layers):
        lower = i / num_layers
        upper = (i + 1) / num_layers
        mask = (depth_norm >= lower) & (depth_norm < upper)

        layer = np.zeros_like(img)
        layer[mask] = img[mask]
        layers.append(layer)

    return layers

def remove_background_with_depth(img, depth, threshold=0.5):
    """깊이 기반 배경 제거"""

    # 깊이 정규화
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())

    # 전경 마스크 (임계값보다 가까운 부분)
    foreground_mask = depth_norm < threshold

    # 마스크 정제
    kernel = np.ones((5, 5), np.uint8)
    foreground_mask = cv2.morphologyEx(
        foreground_mask.astype(np.uint8),
        cv2.MORPH_CLOSE, kernel
    )
    foreground_mask = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_OPEN, kernel
    )

    # 배경 제거
    result = np.zeros_like(img)
    result[foreground_mask == 1] = img[foreground_mask == 1]

    return result, foreground_mask
```

### 3D 효과 생성

```python
def create_3d_ken_burns(img, depth, num_frames=60, zoom=0.1):
    """Ken Burns 효과 (3D 카메라 움직임)"""

    h, w = img.shape[:2]
    frames = []

    for i in range(num_frames):
        t = i / (num_frames - 1)

        # 줌 팩터
        scale = 1 + zoom * t

        # 깊이에 따른 시차
        parallax = (depth - depth.mean()) * 0.001 * t

        # 새 좌표 계산
        y_coords, x_coords = np.meshgrid(range(h), range(w), indexing='ij')

        # 중심 기준 스케일링
        new_x = (x_coords - w/2) / scale + w/2 + parallax
        new_y = (y_coords - h/2) / scale + h/2

        # 리맵핑
        map_x = new_x.astype(np.float32)
        map_y = new_y.astype(np.float32)

        frame = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        frames.append(frame)

    return frames

def depth_aware_zoom(img, depth, zoom_center, zoom_factor=2.0):
    """깊이 인식 줌"""

    h, w = img.shape[:2]
    cx, cy = zoom_center

    # 깊이 정규화
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())

    # 깊이에 따라 다른 줌 적용 (가까운 물체는 더 많이 확대)
    depth_factor = 1 - depth_norm * 0.5  # 0.5 ~ 1.0

    # 좌표 그리드
    y_coords, x_coords = np.meshgrid(range(h), range(w), indexing='ij')

    # 줌 변환 (깊이별로 다른 스케일)
    effective_zoom = zoom_factor * depth_factor

    new_x = (x_coords - cx) / effective_zoom + cx
    new_y = (y_coords - cy) / effective_zoom + cy

    # 리맵핑
    map_x = new_x.astype(np.float32)
    map_y = new_y.astype(np.float32)

    result = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    return result
```

---

## 6. 연습 문제

### 문제 1: MiDaS 깊이 추정

MiDaS를 사용하여 이미지의 깊이를 추정하세요.

**요구사항**:
- 모델 로드 및 추론
- 깊이 맵 시각화 (컬러맵)
- 여러 이미지에 대해 테스트

<details>
<summary>힌트</summary>

```python
import torch

model = torch.hub.load('intel-isl/MiDaS', 'DPT_Large')
midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = midas_transforms.dpt_transform
```

</details>

### 문제 2: 깊이 기반 배경 블러

인물 사진에서 배경만 블러 처리하세요.

**요구사항**:
- 깊이 추정
- 전경/배경 분리
- 배경에만 블러 적용
- 자연스러운 경계 처리

<details>
<summary>힌트</summary>

```python
# 깊이 기반 마스크 생성
threshold = np.percentile(depth, 30)  # 가까운 30%를 전경으로
foreground_mask = depth < threshold

# 마스크 블러링 (경계 부드럽게)
mask_blur = cv2.GaussianBlur(
    foreground_mask.astype(np.float32), (21, 21), 0
)

# 배경 블러
background_blur = cv2.GaussianBlur(img, (25, 25), 0)

# 합성
result = img * mask_blur[..., None] + background_blur * (1 - mask_blur[..., None])
```

</details>

### 문제 3: SfM으로 3D 복원

두 이미지에서 3D 포인트 클라우드를 복원하세요.

**요구사항**:
- 특징점 매칭
- Essential Matrix 계산
- 삼각측량
- 포인트 클라우드 시각화

<details>
<summary>힌트</summary>

```python
# Essential Matrix
E, mask = cv2.findEssentialMat(pts1, pts2, K)
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

# 투영 행렬
P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
P2 = K @ np.hstack([R, t])

# 삼각측량
points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
points_3d = points_4d[:3] / points_4d[3]
```

</details>

### 문제 4: 실시간 깊이 추정

웹캠으로 실시간 깊이 추정을 구현하세요.

**요구사항**:
- 경량 모델 사용 (MiDaS small)
- FPS 측정 및 표시
- 깊이 시각화

<details>
<summary>힌트</summary>

```python
# 경량 모델
model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')

while True:
    ret, frame = cap.read()

    start = time.time()
    depth = estimate_depth(frame, model, transform)
    fps = 1.0 / (time.time() - start)

    cv2.putText(depth_vis, f"FPS: {fps:.1f}", ...)
```

</details>

### 문제 5: 깊이 기반 3D 뷰어

깊이 맵을 이용해 간단한 3D 뷰어를 만드세요.

**요구사항**:
- 깊이 맵 → 포인트 클라우드 변환
- Open3D로 시각화
- 마우스로 회전/줌

<details>
<summary>힌트</summary>

```python
import open3d as o3d

# 포인트 클라우드 생성
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d)
pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

# 시각화
o3d.visualization.draw_geometries([pcd])
```

</details>

---

## 다음 단계

- [23_SLAM_Introduction.md](./23_SLAM_Introduction.md) - Visual SLAM, ORB-SLAM, LiDAR SLAM, Loop Closure

---

## 참고 자료

- [MiDaS GitHub](https://github.com/isl-org/MiDaS)
- [DPT Paper](https://arxiv.org/abs/2103.13413)
- [Depth Anything](https://github.com/LiheYoung/Depth-Anything)
- [Structure from Motion Tutorial](https://cmsc426.github.io/sfm/)
- [OpenCV SfM Tutorial](https://docs.opencv.org/4.x/d4/d18/tutorial_sfm_scene_reconstruction.html)
- [Monodepth2](https://github.com/nianticlabs/monodepth2)
