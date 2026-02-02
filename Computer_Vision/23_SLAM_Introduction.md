# SLAM 입문 (Visual SLAM Introduction)

## 개요

SLAM (Simultaneous Localization and Mapping)은 로봇이나 자율주행 시스템이 미지의 환경에서 지도를 작성하면서 동시에 자신의 위치를 추정하는 기술입니다. Visual SLAM, LiDAR SLAM, Loop Closure의 기초를 다룹니다.

**난이도**: ⭐⭐⭐⭐

**선수 지식**: 3D 비전, 특징점 검출/매칭, 카메라 캘리브레이션, 기본 확률론

---

## 목차

1. [SLAM 개요](#1-slam-개요)
2. [Visual Odometry](#2-visual-odometry)
3. [ORB-SLAM](#3-orb-slam)
4. [LiDAR SLAM](#4-lidar-slam)
5. [Loop Closure](#5-loop-closure)
6. [SLAM 구현 실습](#6-slam-구현-실습)
7. [연습 문제](#7-연습-문제)

---

## 1. SLAM 개요

### SLAM이란?

```
SLAM (Simultaneous Localization and Mapping):
동시적 위치추정 및 지도작성

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  핵심 질문:                                                     │
│  "지도 없이 어떻게 위치를 알 수 있는가?"                        │
│  "위치를 모르면서 어떻게 지도를 만들 수 있는가?"                │
│                                                                 │
│  → 둘을 동시에 해결! (닭과 달걀 문제)                           │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐     │
│  │                                                        │     │
│  │     센서 데이터                                        │     │
│  │     (카메라, LiDAR, IMU)                               │     │
│  │            │                                           │     │
│  │            ▼                                           │     │
│  │     ┌──────────────┐                                   │     │
│  │     │    SLAM      │                                   │     │
│  │     │   알고리즘   │                                   │     │
│  │     └──────┬───────┘                                   │     │
│  │            │                                           │     │
│  │     ┌──────┴───────┐                                   │     │
│  │     │              │                                   │     │
│  │     ▼              ▼                                   │     │
│  │  ┌─────────┐  ┌─────────┐                             │     │
│  │  │  지도   │  │  위치   │                             │     │
│  │  │  (Map)  │  │ (Pose)  │                             │     │
│  │  └─────────┘  └─────────┘                             │     │
│  │                                                        │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

응용 분야:
┌─────────────────┬─────────────────────────────────────────┐
│ 분야            │ 예시                                    │
├─────────────────┼─────────────────────────────────────────┤
│ 자율주행        │ 자동차, 드론, 배달 로봇                 │
│ 증강현실        │ ARKit, ARCore, HoloLens                 │
│ 로봇 청소기     │ Roomba, Roborock                        │
│ 3D 스캐닝       │ 건축, 문화재 복원                       │
│ 내비게이션      │ 실내 위치 인식                          │
└─────────────────┴─────────────────────────────────────────┘
```

### SLAM 분류

```
SLAM 방식 분류:

1. 센서 기반 분류
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Visual SLAM (V-SLAM)                                           │
│  - 카메라 (단안, 스테레오, RGB-D)                               │
│  - 특징점 기반 또는 직접 방식                                   │
│  - 예: ORB-SLAM, LSD-SLAM, DSO                                 │
│                                                                 │
│  LiDAR SLAM                                                     │
│  - 레이저 스캐너                                                │
│  - 포인트 클라우드 매칭                                         │
│  - 예: Cartographer, LOAM, LeGO-LOAM                           │
│                                                                 │
│  Visual-Inertial SLAM                                           │
│  - 카메라 + IMU 융합                                            │
│  - 예: VINS-Mono, OKVIS, MSCKF                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

2. 방법론 기반 분류
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  필터 기반 (Filter-based)                                       │
│  - EKF-SLAM, UKF-SLAM                                          │
│  - 실시간 업데이트                                              │
│  - 선형화 오류 누적 문제                                        │
│                                                                 │
│  그래프 기반 (Graph-based)                                      │
│  - 포즈 그래프 최적화                                           │
│  - 번들 조정                                                    │
│  - 더 정확하지만 계산 비용 높음                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

3. 프론트엔드/백엔드
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  프론트엔드 (Front-end)                                         │
│  - 센서 데이터 처리                                             │
│  - 특징 추출 및 매칭                                            │
│  - 초기 포즈 추정                                               │
│  - 루프 클로저 탐지                                             │
│                                                                 │
│  백엔드 (Back-end)                                              │
│  - 전역 최적화                                                  │
│  - 그래프 최적화                                                │
│  - 불확실성 추정                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Visual Odometry

### Visual Odometry 개념

```
Visual Odometry (VO):
연속된 이미지로부터 카메라 움직임 추정

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  프레임 t-1        프레임 t          프레임 t+1                 │
│  ┌───────┐        ┌───────┐        ┌───────┐                   │
│  │   📷  │──T₁───▶│   📷  │──T₂───▶│   📷  │                   │
│  └───────┘        └───────┘        └───────┘                   │
│                                                                 │
│  누적 포즈: P_t = T₁ * T₂ * ... * T_t                           │
│                                                                 │
│  문제점:                                                        │
│  - 누적 오차 (drift)                                            │
│  - 스케일 모호성 (단안 카메라)                                  │
│  - 빠른 움직임에 취약                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

VO 파이프라인:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. 이미지 획득                                                 │
│       ▼                                                         │
│  2. 특징 추출 (ORB, SIFT, Harris corners)                       │
│       ▼                                                         │
│  3. 특징 매칭/추적 (BF Matcher, Optical Flow)                   │
│       ▼                                                         │
│  4. 모션 추정 (Essential Matrix, PnP)                           │
│       ▼                                                         │
│  5. 지역 최적화 (Local BA)                                      │
│       ▼                                                         │
│  6. 포즈 업데이트                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 단안 Visual Odometry 구현

```python
import cv2
import numpy as np

class MonocularVO:
    """단안 Visual Odometry"""

    def __init__(self, K, detector='ORB'):
        """
        K: 카메라 내부 파라미터 행렬
        detector: 특징점 검출기 ('ORB', 'SIFT', 'FAST')
        """
        self.K = K
        self.focal = K[0, 0]
        self.pp = (K[0, 2], K[1, 2])  # principal point

        # 특징점 검출기
        if detector == 'ORB':
            self.detector = cv2.ORB_create(3000)
        elif detector == 'SIFT':
            self.detector = cv2.SIFT_create(3000)
        else:
            self.detector = cv2.FastFeatureDetector_create(threshold=25)

        # 광학 흐름 파라미터
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # 상태
        self.prev_frame = None
        self.prev_pts = None
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        self.trajectory = []

    def detect_features(self, img):
        """특징점 검출"""
        if hasattr(self.detector, 'detectAndCompute'):
            kp, _ = self.detector.detectAndCompute(img, None)
        else:
            kp = self.detector.detect(img, None)

        pts = np.array([p.pt for p in kp], dtype=np.float32)
        return pts.reshape(-1, 1, 2)

    def track_features(self, prev_img, cur_img, prev_pts):
        """광학 흐름으로 특징점 추적"""

        cur_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_img, cur_img, prev_pts, None, **self.lk_params
        )

        status = status.reshape(-1)
        prev_pts = prev_pts[status == 1]
        cur_pts = cur_pts[status == 1]

        return prev_pts, cur_pts

    def estimate_pose(self, pts1, pts2):
        """Essential Matrix로 포즈 추정"""

        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)

        return R, t

    def process_frame(self, frame):
        """프레임 처리"""

        # 그레이스케일 변환
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        if self.prev_frame is None:
            # 첫 프레임
            self.prev_frame = gray
            self.prev_pts = self.detect_features(gray)
            return self.cur_R, self.cur_t

        # 특징점 추적
        if self.prev_pts is not None and len(self.prev_pts) > 0:
            prev_pts, cur_pts = self.track_features(
                self.prev_frame, gray, self.prev_pts
            )

            if len(prev_pts) >= 8:
                # 포즈 추정
                R, t = self.estimate_pose(
                    prev_pts.reshape(-1, 2),
                    cur_pts.reshape(-1, 2)
                )

                # 포즈 누적
                self.cur_t = self.cur_t + self.cur_R @ t
                self.cur_R = R @ self.cur_R

                # 새 특징점이 필요하면 검출
                if len(cur_pts) < 1000:
                    new_pts = self.detect_features(gray)
                    if len(cur_pts) > 0:
                        self.prev_pts = np.vstack([
                            cur_pts.reshape(-1, 1, 2),
                            new_pts
                        ])
                    else:
                        self.prev_pts = new_pts
                else:
                    self.prev_pts = cur_pts.reshape(-1, 1, 2)
            else:
                self.prev_pts = self.detect_features(gray)
        else:
            self.prev_pts = self.detect_features(gray)

        self.prev_frame = gray

        # 궤적 저장
        self.trajectory.append(self.cur_t.copy())

        return self.cur_R, self.cur_t

    def get_trajectory(self):
        """궤적 반환"""
        return np.array([t.ravel() for t in self.trajectory])

# 사용 예
K = np.array([
    [718.856, 0, 607.1928],
    [0, 718.856, 185.2157],
    [0, 0, 1]
], dtype=np.float32)

vo = MonocularVO(K)

cap = cv2.VideoCapture('driving.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    R, t = vo.process_frame(frame)

    # 현재 위치 출력
    x, y, z = t.ravel()
    print(f"Position: x={x:.2f}, y={y:.2f}, z={z:.2f}")

cap.release()

# 궤적 시각화
trajectory = vo.get_trajectory()
```

### 스테레오 Visual Odometry

```python
class StereoVO:
    """스테레오 Visual Odometry"""

    def __init__(self, K, baseline, detector='ORB'):
        self.K = K
        self.baseline = baseline
        self.focal = K[0, 0]

        self.detector = cv2.ORB_create(3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        # 스테레오 매처
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=5,
            P1=8 * 3 * 5 ** 2,
            P2=32 * 3 * 5 ** 2
        )

        self.prev_pts_3d = None
        self.prev_kp = None
        self.prev_desc = None
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))

    def compute_depth(self, left, right):
        """스테레오 매칭으로 깊이 계산"""

        disparity = self.stereo.compute(left, right).astype(np.float32) / 16.0

        # 시차 → 깊이
        depth = np.zeros_like(disparity)
        valid = disparity > 0
        depth[valid] = self.focal * self.baseline / disparity[valid]

        return depth

    def get_3d_points(self, kp, depth):
        """2D 키포인트를 3D로 변환"""

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        pts_3d = []
        valid_indices = []

        for i, pt in enumerate(kp):
            x, y = int(pt.pt[0]), int(pt.pt[1])

            if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
                z = depth[y, x]

                if z > 0 and z < 100:  # 유효한 깊이
                    X = (pt.pt[0] - cx) * z / fx
                    Y = (pt.pt[1] - cy) * z / fy
                    pts_3d.append([X, Y, z])
                    valid_indices.append(i)

        return np.array(pts_3d), valid_indices

    def process_frame(self, left, right):
        """스테레오 프레임 처리"""

        # 그레이스케일
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # 깊이 계산
        depth = self.compute_depth(gray_left, gray_right)

        # 특징 검출
        kp, desc = self.detector.detectAndCompute(gray_left, None)

        # 3D 점 계산
        pts_3d, valid_idx = self.get_3d_points(kp, depth)

        if self.prev_pts_3d is None:
            self.prev_pts_3d = pts_3d
            self.prev_kp = [kp[i] for i in valid_idx]
            self.prev_desc = desc[valid_idx]
            return self.cur_R, self.cur_t

        # 이전 프레임과 매칭
        matches = self.bf.knnMatch(self.prev_desc, desc[valid_idx], k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) >= 6:
            # 3D-2D 대응점
            obj_points = np.array([
                self.prev_pts_3d[m.queryIdx] for m in good_matches
            ])
            img_points = np.array([
                kp[valid_idx[m.trainIdx]].pt for m in good_matches
            ])

            # PnP로 포즈 추정
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_points, img_points, self.K, None
            )

            if success and inliers is not None and len(inliers) > 10:
                R, _ = cv2.Rodrigues(rvec)

                # 포즈 누적
                self.cur_t = self.cur_t + self.cur_R @ tvec
                self.cur_R = R @ self.cur_R

        # 상태 업데이트
        self.prev_pts_3d = pts_3d
        self.prev_kp = [kp[i] for i in valid_idx]
        self.prev_desc = desc[valid_idx]

        return self.cur_R, self.cur_t
```

---

## 3. ORB-SLAM

### ORB-SLAM 개요

```
ORB-SLAM 아키텍처:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ORB-SLAM: 가장 널리 사용되는 Visual SLAM 시스템                │
│                                                                 │
│  버전:                                                          │
│  - ORB-SLAM (2015): 단안                                        │
│  - ORB-SLAM2 (2017): 단안/스테레오/RGB-D                        │
│  - ORB-SLAM3 (2021): Visual-Inertial, 다중 맵                   │
│                                                                 │
│  3개의 병렬 스레드:                                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                                                         │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │    │
│  │  │  Tracking   │  │Local Mapping│  │Loop Closing │     │    │
│  │  │   Thread    │  │   Thread    │  │   Thread    │     │    │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │    │
│  │         │                │                │            │    │
│  │         │    Keyframes   │                │            │    │
│  │         └───────────────▶│                │            │    │
│  │                          │    Keyframes   │            │    │
│  │                          └───────────────▶│            │    │
│  │                                           │            │    │
│  │  ┌───────────────────────────────────────┐│            │    │
│  │  │           Map (MapPoints)             ││            │    │
│  │  │         & Covisibility Graph          ││            │    │
│  │  └───────────────────────────────────────┘│            │    │
│  │                                           │            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Tracking 스레드:
- 매 프레임 처리
- ORB 특징 추출
- 이전 프레임 또는 맵과 매칭
- 초기 포즈 추정
- 키프레임 결정

Local Mapping 스레드:
- 새 키프레임 삽입
- 최근 MapPoint 컬링
- 새 MapPoint 생성
- Local Bundle Adjustment
- 중복 키프레임 제거

Loop Closing 스레드:
- 루프 후보 검출 (DBoW2)
- 루프 검증 및 보정
- Essential Graph 최적화
- 전역 Bundle Adjustment
```

### ORB 특징과 Bag of Words

```python
import cv2
import numpy as np

class ORBVocabulary:
    """ORB 기반 Bag of Words"""

    def __init__(self, num_words=1000):
        self.orb = cv2.ORB_create(1000)
        self.num_words = num_words
        self.vocabulary = None
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def train(self, images):
        """이미지로부터 vocabulary 학습"""

        all_descriptors = []

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, desc = self.orb.detectAndCompute(gray, None)
            if desc is not None:
                all_descriptors.append(desc)

        all_desc = np.vstack(all_descriptors)

        # K-means 클러스터링
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                   100, 0.2)
        _, labels, centers = cv2.kmeans(
            all_desc.astype(np.float32),
            self.num_words,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS
        )

        self.vocabulary = centers.astype(np.uint8)
        print(f"Vocabulary 생성 완료: {self.num_words} words")

    def compute_bow(self, img):
        """이미지의 BoW 벡터 계산"""

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, desc = self.orb.detectAndCompute(gray, None)

        if desc is None:
            return np.zeros(self.num_words)

        # 각 디스크립터를 가장 가까운 vocabulary word에 할당
        matches = self.bf.match(desc, self.vocabulary)

        bow = np.zeros(self.num_words)
        for m in matches:
            bow[m.trainIdx] += 1

        # 정규화
        bow = bow / (np.linalg.norm(bow) + 1e-6)

        return bow

    def compute_similarity(self, bow1, bow2):
        """두 BoW 벡터의 유사도"""
        return np.dot(bow1, bow2)


class SimpleSLAM:
    """간단한 SLAM 시스템 (ORB-SLAM 컨셉)"""

    def __init__(self, K):
        self.K = K
        self.orb = cv2.ORB_create(2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # 맵
        self.keyframes = []      # 키프레임 목록
        self.map_points = []     # 3D 포인트
        self.poses = []          # 키프레임 포즈

        # 현재 상태
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        self.prev_frame = None
        self.prev_kp = None
        self.prev_desc = None

        # 키프레임 기준
        self.kf_threshold = 30   # 최소 매칭 수

    def is_keyframe(self, num_matches, motion):
        """키프레임 여부 결정"""

        # 간단한 기준: 매칭 수가 적거나 움직임이 크면 키프레임
        translation = np.linalg.norm(motion)

        if num_matches < self.kf_threshold or translation > 0.5:
            return True
        return False

    def add_keyframe(self, frame, kp, desc, pose):
        """키프레임 추가"""

        keyframe = {
            'frame': frame.copy(),
            'keypoints': kp,
            'descriptors': desc,
            'pose': pose.copy()
        }

        self.keyframes.append(keyframe)
        self.poses.append(pose)

        print(f"Keyframe 추가: 총 {len(self.keyframes)}개")

    def process_frame(self, frame):
        """프레임 처리 (Tracking)"""

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = self.orb.detectAndCompute(gray, None)

        if self.prev_frame is None:
            # 첫 프레임 → 키프레임
            pose = {'R': np.eye(3), 't': np.zeros((3, 1))}
            self.add_keyframe(gray, kp, desc, pose)
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_desc = desc
            return self.cur_R, self.cur_t

        # 이전 프레임과 매칭
        matches = self.bf.match(self.prev_desc, desc)
        matches = sorted(matches, key=lambda x: x.distance)[:500]

        if len(matches) >= 8:
            # 매칭점 추출
            pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

            # Essential Matrix로 포즈 추정
            E, mask = cv2.findEssentialMat(pts1, pts2, self.K)
            _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)

            # 포즈 누적
            self.cur_t = self.cur_t + self.cur_R @ t
            self.cur_R = R @ self.cur_R

            # 키프레임 체크
            if self.is_keyframe(len(matches), t):
                pose = {'R': self.cur_R.copy(), 't': self.cur_t.copy()}
                self.add_keyframe(gray, kp, desc, pose)

        # 상태 업데이트
        self.prev_frame = gray
        self.prev_kp = kp
        self.prev_desc = desc

        return self.cur_R, self.cur_t

    def get_camera_trajectory(self):
        """카메라 궤적 반환"""
        trajectory = []
        for pose in self.poses:
            R = pose['R']
            t = pose['t']
            # 카메라 위치 = -R^T * t
            pos = -R.T @ t
            trajectory.append(pos.ravel())
        return np.array(trajectory)
```

---

## 4. LiDAR SLAM

### LiDAR SLAM 개요

```
LiDAR SLAM:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  LiDAR 센서 특징:                                               │
│  - 360도 스캔                                                   │
│  - 정확한 거리 측정                                             │
│  - 조명 조건에 강건                                             │
│  - 풍부한 3D 포인트 클라우드                                    │
│                                                                 │
│  LiDAR 종류:                                                    │
│  ┌──────────────────┬─────────────────────────────────────┐     │
│  │ 2D LiDAR         │ 평면 스캔, 저렴, 로봇 청소기        │     │
│  │ (예: RPLiDAR)    │                                     │     │
│  ├──────────────────┼─────────────────────────────────────┤     │
│  │ 3D LiDAR         │ 3D 포인트 클라우드, 자율주행       │     │
│  │ (예: Velodyne)   │                                     │     │
│  ├──────────────────┼─────────────────────────────────────┤     │
│  │ Solid-State      │ 무회전, 소형, 최신 트렌드          │     │
│  │ (예: Livox)      │                                     │     │
│  └──────────────────┴─────────────────────────────────────┘     │
│                                                                 │
│  주요 알고리즘:                                                 │
│  - ICP (Iterative Closest Point)                               │
│  - NDT (Normal Distributions Transform)                        │
│  - LOAM (LiDAR Odometry and Mapping)                           │
│  - LeGO-LOAM (Lightweight Ground-Optimized)                    │
│  - Cartographer (Google)                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### ICP (Iterative Closest Point)

```python
import numpy as np
from scipy.spatial import KDTree

def icp(source, target, max_iterations=50, tolerance=1e-6):
    """
    ICP 알고리즘으로 두 포인트 클라우드 정합

    Parameters:
        source: 소스 포인트 클라우드 (N x 3)
        target: 타겟 포인트 클라우드 (M x 3)

    Returns:
        R: 회전 행렬 (3 x 3)
        t: 평행 이동 벡터 (3,)
        transformed: 변환된 소스 포인트
    """

    src = source.copy()
    prev_error = float('inf')

    R_total = np.eye(3)
    t_total = np.zeros(3)

    # KD-Tree로 효율적인 최근접 탐색
    tree = KDTree(target)

    for i in range(max_iterations):
        # 1. 최근접 대응점 찾기
        distances, indices = tree.query(src)
        correspondences = target[indices]

        # 2. 변환 추정 (SVD)
        src_centroid = np.mean(src, axis=0)
        tgt_centroid = np.mean(correspondences, axis=0)

        src_centered = src - src_centroid
        tgt_centered = correspondences - tgt_centroid

        H = src_centered.T @ tgt_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # 반사 보정
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = tgt_centroid - R @ src_centroid

        # 3. 변환 적용
        src = (R @ src.T).T + t

        # 누적 변환
        R_total = R @ R_total
        t_total = R @ t_total + t

        # 4. 수렴 확인
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            print(f"ICP 수렴: {i+1} 반복, 오차: {mean_error:.6f}")
            break
        prev_error = mean_error

    return R_total, t_total, src

class LiDARSLAM:
    """간단한 2D LiDAR SLAM"""

    def __init__(self, map_resolution=0.05):
        self.resolution = map_resolution
        self.pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.trajectory = [self.pose.copy()]

        # 점유 격자 맵
        self.map_size = 1000
        self.occupancy_map = np.ones((self.map_size, self.map_size)) * 0.5
        self.map_origin = np.array([self.map_size // 2, self.map_size // 2])

    def scan_to_points(self, scan_ranges, scan_angles):
        """스캔 데이터를 2D 포인트로 변환"""

        valid = (scan_ranges > 0.1) & (scan_ranges < 30.0)
        ranges = scan_ranges[valid]
        angles = scan_angles[valid]

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        return np.column_stack([x, y])

    def transform_points(self, points, pose):
        """포인트를 월드 좌표로 변환"""

        x, y, theta = pose
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        transformed = (R @ points.T).T + np.array([x, y])
        return transformed

    def point_to_grid(self, points):
        """포인트를 그리드 좌표로 변환"""

        grid_x = (points[:, 0] / self.resolution + self.map_origin[0]).astype(int)
        grid_y = (points[:, 1] / self.resolution + self.map_origin[1]).astype(int)

        # 맵 범위 내로 제한
        valid = (grid_x >= 0) & (grid_x < self.map_size) & \
                (grid_y >= 0) & (grid_y < self.map_size)

        return grid_x[valid], grid_y[valid], valid

    def update_map(self, scan_points, pose):
        """점유 격자 맵 업데이트"""

        world_points = self.transform_points(scan_points, pose)
        gx, gy, valid = self.point_to_grid(world_points)

        # 점유 확률 업데이트 (로그 오즈)
        self.occupancy_map[gy, gx] = np.clip(
            self.occupancy_map[gy, gx] + 0.1, 0, 1
        )

    def match_scan(self, current_points, previous_points):
        """스캔 매칭으로 상대 이동 추정"""

        if len(previous_points) < 10 or len(current_points) < 10:
            return np.array([0, 0, 0])

        # ICP 적용
        R, t, _ = icp(current_points, previous_points)

        # 2D에서 theta 추출
        theta = np.arctan2(R[1, 0], R[0, 0])

        return np.array([t[0], t[1], theta])

    def process_scan(self, scan_ranges, scan_angles, prev_scan=None):
        """스캔 처리"""

        current_points = self.scan_to_points(scan_ranges, scan_angles)

        if prev_scan is not None:
            prev_points = self.scan_to_points(prev_scan[0], prev_scan[1])

            # 스캔 매칭
            delta_pose = self.match_scan(current_points, prev_points)

            # 포즈 업데이트
            self.pose[2] += delta_pose[2]
            R = np.array([
                [np.cos(self.pose[2]), -np.sin(self.pose[2])],
                [np.sin(self.pose[2]), np.cos(self.pose[2])]
            ])
            self.pose[:2] += R @ delta_pose[:2]

        # 맵 업데이트
        self.update_map(current_points, self.pose)

        # 궤적 저장
        self.trajectory.append(self.pose.copy())

        return self.pose

    def get_occupancy_map(self):
        """점유 맵 반환"""
        return self.occupancy_map

    def get_trajectory(self):
        """궤적 반환"""
        return np.array(self.trajectory)
```

---

## 5. Loop Closure

### Loop Closure 개념

```
Loop Closure (루프 폐합):
이전에 방문한 장소를 재인식하여 누적 오차 보정

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  문제: Drift (누적 오차)                                        │
│                                                                 │
│       실제 경로        추정 경로 (drift 있음)                   │
│       ┌─────────┐      ┌─────────┐                              │
│       │         │      │         ╲                              │
│       │         │      │          ╲                             │
│       │         │      │           ╲                            │
│       └─────────┘      └────────────╲                           │
│       (폐곡선)          (열린 곡선)                             │
│                                                                 │
│  해결: Loop Closure                                             │
│       1. 현재 위치가 이전에 방문한 곳인지 탐지                  │
│       2. 루프 제약 조건 추가                                    │
│       3. 포즈 그래프 최적화                                     │
│                                                                 │
│       ┌─────────┐                                               │
│       │    ●────●  ← 루프 탐지                                  │
│       │    │    │                                               │
│       │    │    │  ← 그래프 최적화                              │
│       │    ●────●                                               │
│       └─────────┘                                               │
│       (보정된 경로)                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Loop Closure 구현

```python
import cv2
import numpy as np
from collections import deque

class LoopClosureDetector:
    """Bag of Words 기반 루프 클로저 탐지"""

    def __init__(self, vocabulary_size=1000, min_score=0.3):
        self.orb = cv2.ORB_create(2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.vocabulary = None
        self.vocabulary_size = vocabulary_size
        self.min_score = min_score

        # 키프레임 데이터베이스
        self.keyframe_bows = []
        self.keyframe_descs = []
        self.keyframe_kps = []

        # 최근 N개 키프레임은 루프 후보에서 제외
        self.temporal_window = 30

    def build_vocabulary(self, training_images):
        """Vocabulary 구축"""

        all_descriptors = []

        for img in training_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, desc = self.orb.detectAndCompute(gray, None)
            if desc is not None:
                all_descriptors.append(desc)

        all_desc = np.vstack(all_descriptors).astype(np.float32)

        # K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                   100, 0.2)
        _, _, self.vocabulary = cv2.kmeans(
            all_desc, self.vocabulary_size, None,
            criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        self.vocabulary = self.vocabulary.astype(np.uint8)

    def compute_bow(self, descriptors):
        """BoW 벡터 계산"""

        if self.vocabulary is None or descriptors is None:
            return None

        matches = self.bf.match(descriptors, self.vocabulary)

        bow = np.zeros(self.vocabulary_size)
        for m in matches:
            bow[m.trainIdx] += 1

        # L2 정규화
        norm = np.linalg.norm(bow)
        if norm > 0:
            bow = bow / norm

        return bow

    def add_keyframe(self, frame):
        """키프레임 추가"""

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = self.orb.detectAndCompute(gray, None)

        if desc is None:
            return -1

        bow = self.compute_bow(desc)

        self.keyframe_bows.append(bow)
        self.keyframe_descs.append(desc)
        self.keyframe_kps.append(kp)

        return len(self.keyframe_bows) - 1

    def detect_loop(self, query_idx):
        """루프 후보 탐지"""

        if query_idx < self.temporal_window + 1:
            return None, 0

        query_bow = self.keyframe_bows[query_idx]

        best_match = -1
        best_score = 0

        # 시간적으로 멀리 떨어진 키프레임만 검색
        for i in range(query_idx - self.temporal_window):
            score = np.dot(query_bow, self.keyframe_bows[i])

            if score > best_score and score > self.min_score:
                best_score = score
                best_match = i

        if best_match >= 0:
            return best_match, best_score

        return None, 0

    def verify_loop(self, query_idx, candidate_idx, min_inliers=50):
        """기하학적 검증으로 루프 확인"""

        desc1 = self.keyframe_descs[query_idx]
        desc2 = self.keyframe_descs[candidate_idx]
        kp1 = self.keyframe_kps[query_idx]
        kp2 = self.keyframe_kps[candidate_idx]

        # 특징점 매칭
        matches = self.bf.knnMatch(desc1, desc2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 8:
            return False, None

        # Fundamental Matrix로 기하학적 검증
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

        if mask is None:
            return False, None

        num_inliers = np.sum(mask)

        if num_inliers >= min_inliers:
            return True, {
                'query_idx': query_idx,
                'match_idx': candidate_idx,
                'inliers': num_inliers,
                'pts1': pts1[mask.ravel() == 1],
                'pts2': pts2[mask.ravel() == 1]
            }

        return False, None


class PoseGraphOptimizer:
    """간단한 포즈 그래프 최적화"""

    def __init__(self):
        self.poses = []         # 노드 (포즈)
        self.edges = []         # 엣지 (상대 변환)
        self.loop_constraints = []  # 루프 제약

    def add_pose(self, pose):
        """포즈 노드 추가"""
        self.poses.append(pose.copy())
        return len(self.poses) - 1

    def add_odometry_edge(self, i, j, relative_pose, info_matrix=None):
        """오도메트리 엣지 추가"""

        if info_matrix is None:
            info_matrix = np.eye(3)

        self.edges.append({
            'from': i,
            'to': j,
            'measurement': relative_pose,
            'info': info_matrix
        })

    def add_loop_constraint(self, i, j, relative_pose, info_matrix=None):
        """루프 제약 추가"""

        if info_matrix is None:
            # 루프 제약은 높은 가중치
            info_matrix = np.eye(3) * 100

        self.loop_constraints.append({
            'from': i,
            'to': j,
            'measurement': relative_pose,
            'info': info_matrix
        })

    def optimize(self, num_iterations=10):
        """그래프 최적화 (Gauss-Newton)"""

        # 간단한 구현 (실제로는 g2o, Ceres 등 사용)
        print("포즈 그래프 최적화는 g2o 등 전문 라이브러리 권장")

        # 루프 제약을 이용한 간단한 보정
        for constraint in self.loop_constraints:
            i = constraint['from']
            j = constraint['to']

            # 누적 드리프트 계산
            drift = self.poses[j][:2] - self.poses[i][:2]
            drift -= constraint['measurement'][:2]

            # 선형 보간으로 드리프트 분배
            for k in range(i, j + 1):
                alpha = (k - i) / (j - i) if j > i else 0
                self.poses[k][:2] -= alpha * drift

        return self.poses
```

---

## 6. SLAM 구현 실습

### 간단한 SLAM 시스템

```python
import cv2
import numpy as np

class SimpleVSLAM:
    """간단한 Visual SLAM 시스템"""

    def __init__(self, K):
        self.K = K

        # 모듈
        self.vo = MonocularVO(K)
        self.loop_detector = LoopClosureDetector()
        self.pose_graph = PoseGraphOptimizer()

        # 상태
        self.frame_count = 0
        self.keyframe_interval = 10

    def process_frame(self, frame):
        """프레임 처리"""

        self.frame_count += 1

        # Visual Odometry
        R, t = self.vo.process_frame(frame)

        # 키프레임 추가
        if self.frame_count % self.keyframe_interval == 0:
            kf_idx = self.loop_detector.add_keyframe(frame)

            # 포즈 그래프에 노드 추가
            pose = np.array([t[0, 0], t[1, 0], 0])  # 2D 근사
            node_idx = self.pose_graph.add_pose(pose)

            # 이전 키프레임과 엣지 연결
            if node_idx > 0:
                prev_pose = self.pose_graph.poses[node_idx - 1]
                relative = pose - prev_pose
                self.pose_graph.add_odometry_edge(
                    node_idx - 1, node_idx, relative
                )

            # 루프 탐지
            if kf_idx > 30:  # 충분한 키프레임 후
                candidate, score = self.loop_detector.detect_loop(kf_idx)

                if candidate is not None:
                    verified, loop_info = self.loop_detector.verify_loop(
                        kf_idx, candidate
                    )

                    if verified:
                        print(f"Loop detected: {kf_idx} -> {candidate}")

                        # 루프 제약 추가
                        relative = pose - self.pose_graph.poses[candidate]
                        self.pose_graph.add_loop_constraint(
                            candidate, node_idx, relative
                        )

                        # 최적화
                        self.pose_graph.optimize()

        return R, t

    def get_map(self):
        """맵 반환"""
        return self.vo.get_trajectory()

    def get_optimized_trajectory(self):
        """최적화된 궤적 반환"""
        return np.array(self.pose_graph.poses)
```

### 시각화

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_slam_result(trajectory, loop_closures=None):
    """SLAM 결과 시각화"""

    fig = plt.figure(figsize=(12, 5))

    # 2D 궤적
    ax1 = fig.add_subplot(121)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=1)
    ax1.scatter(trajectory[0, 0], trajectory[0, 1],
               c='green', s=100, marker='o', label='Start')
    ax1.scatter(trajectory[-1, 0], trajectory[-1, 1],
               c='red', s=100, marker='x', label='End')

    if loop_closures:
        for lc in loop_closures:
            i, j = lc['from'], lc['to']
            ax1.plot([trajectory[i, 0], trajectory[j, 0]],
                    [trajectory[i, 1], trajectory[j, 1]],
                    'g--', linewidth=2, alpha=0.5)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('2D Trajectory')
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True)

    # 3D 궤적
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            'b-', linewidth=1)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D Trajectory')

    plt.tight_layout()
    plt.show()

def visualize_occupancy_map(occupancy_map, trajectory=None):
    """점유 맵 시각화"""

    plt.figure(figsize=(10, 10))

    # 맵 표시
    plt.imshow(occupancy_map, cmap='gray', origin='lower')

    # 궤적 오버레이
    if trajectory is not None:
        # 맵 좌표로 변환
        map_center = occupancy_map.shape[0] // 2
        resolution = 0.05
        traj_map = trajectory / resolution + map_center

        plt.plot(traj_map[:, 0], traj_map[:, 1], 'r-', linewidth=2)
        plt.scatter(traj_map[0, 0], traj_map[0, 1], c='green', s=100)
        plt.scatter(traj_map[-1, 0], traj_map[-1, 1], c='blue', s=100)

    plt.title('Occupancy Grid Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='Occupancy Probability')
    plt.show()
```

---

## 7. 연습 문제

### 문제 1: Visual Odometry 구현

단안 Visual Odometry를 구현하세요.

**요구사항**:
- ORB 특징 검출
- 광학 흐름 또는 디스크립터 매칭
- Essential Matrix로 포즈 추정
- 궤적 시각화

<details>
<summary>힌트</summary>

```python
# Essential Matrix
E, mask = cv2.findEssentialMat(pts1, pts2, K)
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

# 포즈 누적
cur_t = cur_t + cur_R @ t
cur_R = R @ cur_R
```

</details>

### 문제 2: 루프 클로저 탐지

BoW 기반 루프 클로저를 구현하세요.

**요구사항**:
- ORB vocabulary 구축
- BoW 벡터 계산
- 유사도 기반 후보 탐지
- 기하학적 검증

<details>
<summary>힌트</summary>

```python
# BoW 유사도
score = np.dot(bow1, bow2)

# 기하학적 검증
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
inliers = np.sum(mask)
```

</details>

### 문제 3: ICP 구현

ICP 알고리즘을 구현하세요.

**요구사항**:
- 최근접 대응점 검색
- SVD로 변환 추정
- 반복 최적화
- 수렴 조건

<details>
<summary>힌트</summary>

```python
# SVD로 R, t 계산
H = src_centered.T @ tgt_centered
U, _, Vt = np.linalg.svd(H)
R = Vt.T @ U.T
t = tgt_centroid - R @ src_centroid
```

</details>

### 문제 4: 점유 격자 맵

LiDAR 데이터로 점유 격자 맵을 생성하세요.

**요구사항**:
- 스캔 데이터를 포인트로 변환
- 격자 좌표 변환
- 점유 확률 업데이트
- 맵 시각화

<details>
<summary>힌트</summary>

```python
# 로그 오즈 업데이트
log_odds = np.log(p / (1 - p))
log_odds[occupied] += 0.5
log_odds[free] -= 0.2
p = 1 / (1 + np.exp(-log_odds))
```

</details>

### 문제 5: 완전한 SLAM 시스템

VO, 루프 클로저, 맵핑을 통합한 SLAM을 구현하세요.

**요구사항**:
- 키프레임 관리
- 루프 탐지 및 검증
- 포즈 그래프 최적화
- 3D 맵 생성

<details>
<summary>힌트</summary>

```python
# 통합 시스템
class SLAM:
    def process(self, frame):
        # 1. 트래킹
        pose = self.track(frame)

        # 2. 키프레임이면 맵 업데이트
        if self.is_keyframe():
            self.local_mapping()

            # 3. 루프 탐지
            if self.detect_loop():
                self.optimize_graph()
```

</details>

---

## 다음 단계

- 실제 SLAM 라이브러리 사용 (ORB-SLAM3, RTAB-Map)
- ROS 연동
- Visual-Inertial SLAM
- 딥러닝 기반 SLAM

---

## 참고 자료

- [ORB-SLAM3 GitHub](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [SLAM Tutorial - Cyrill Stachniss](https://www.youtube.com/playlist?list=PLgnQpQtFTOGQrZ4O5QzbIHgl3b1JHimN_)
- [Multiple View Geometry in Computer Vision](https://www.robots.ox.ac.uk/~vgg/hzbook/)
- [Probabilistic Robotics (Thrun et al.)](http://www.probabilistic-robotics.org/)
- [LOAM Paper](https://www.ri.cmu.edu/pub_files/2014/7/Ji_LidarMapping_RSS2014_v8.pdf)
- [Cartographer](https://google-cartographer.readthedocs.io/)
