# 3D 비전 기초 (3D Vision Basics)

## 개요

3D 비전은 2D 이미지로부터 3차원 정보를 추출하고 복원하는 기술입니다. 스테레오 비전, 깊이 맵, 포인트 클라우드 처리, 3D 재구성의 기초를 다룹니다.

**난이도**: ⭐⭐⭐⭐

**선수 지식**: 카메라 캘리브레이션, 특징점 검출/매칭, 선형대수

---

## 목차

1. [3D 비전 개요](#1-3d-비전-개요)
2. [스테레오 비전 원리](#2-스테레오-비전-원리)
3. [깊이 맵 생성](#3-깊이-맵-생성)
4. [포인트 클라우드](#4-포인트-클라우드)
5. [Open3D 기초](#5-open3d-기초)
6. [3D 재구성](#6-3d-재구성)
7. [연습 문제](#7-연습-문제)

---

## 1. 3D 비전 개요

### 3D 비전의 목표

```
3D 비전 파이프라인:

┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  2D 이미지 ─────▶ 깊이 추정 ─────▶ 3D 재구성                    │
│      │                                                           │
│      │           ┌─────────────┐                                 │
│      └──────────▶│ 깊이 정보   │──────▶ 포인트 클라우드          │
│                  └─────────────┘            │                    │
│                                             │                    │
│                                             ▼                    │
│                                      ┌─────────────┐             │
│                                      │  3D 메쉬    │             │
│                                      │  3D 모델    │             │
│                                      └─────────────┘             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

깊이 추출 방법:
┌─────────────────────┬──────────────────────────────────────────┐
│ 방법                │ 설명                                     │
├─────────────────────┼──────────────────────────────────────────┤
│ 스테레오 비전       │ 두 카메라의 시차로 깊이 계산             │
│ 구조광 (Structured) │ 알려진 패턴을 투사하여 깊이 측정         │
│ ToF (Time-of-Flight)│ 빛의 비행 시간으로 거리 측정             │
│ 단안 깊이 추정      │ 단일 카메라 + 딥러닝으로 깊이 예측       │
│ LiDAR               │ 레이저 스캐닝으로 정밀 깊이 측정         │
└─────────────────────┴──────────────────────────────────────────┘
```

### 좌표계 이해

```
카메라 좌표계:

        Y (위)
        │
        │
        │
        │_________ X (오른쪽)
       /
      /
     Z (카메라 정면 방향)

월드 좌표계 → 카메라 좌표계 변환:
P_cam = R * P_world + t

이미지 좌표계:
┌─────────────────────▶ u (가로, 픽셀)
│
│   ● (cx, cy) 주점
│
▼
v (세로, 픽셀)

3D → 2D 투영:
u = fx * (X/Z) + cx
v = fy * (Y/Z) + cy
```

---

## 2. 스테레오 비전 원리

### 에피폴라 기하학

```
에피폴라 기하학 (Epipolar Geometry):

             에피폴 (e)
              │
   ┌──────────┼──────────┐
   │          │          │
   │    ●─────┼──────────┼─────● 에피폴라 선
   │   P      │          │   P'
   │          │          │
   └──────────┴──────────┘
       왼쪽         오른쪽
       이미지       이미지

3D 점 P가 왼쪽 이미지의 점 p에 투영되면,
오른쪽 이미지에서는 에피폴라 선 위 어딘가에 p'로 투영됨.

핵심 행렬들:
┌───────────────────┬─────────────────────────────────────────┐
│ 행렬              │ 설명                                    │
├───────────────────┼─────────────────────────────────────────┤
│ Essential Matrix  │ 정규화된 좌표계에서 기하학적 관계       │
│ (E)               │ E = [t]x * R                            │
├───────────────────┼─────────────────────────────────────────┤
│ Fundamental Matrix│ 픽셀 좌표계에서 기하학적 관계           │
│ (F)               │ F = K'^(-T) * E * K^(-1)               │
│                   │ p'^T * F * p = 0                        │
└───────────────────┴─────────────────────────────────────────┘
```

### 시차와 깊이

```
스테레오 시차 (Disparity):

왼쪽 카메라         오른쪽 카메라
    C_L ─────────────── C_R
     │                    │
     │    b (베이스라인)   │
     │    ◄─────────────► │
     │                    │
     │                    │
     ▼                    ▼
    p_L        d        p_R
    ●─────────────────────●
    │                     │
    │     시차 (d)        │
    │     d = x_L - x_R   │

깊이 계산:
Z = (f * b) / d

여기서:
- Z: 깊이 (카메라로부터의 거리)
- f: 초점 거리
- b: 베이스라인 (두 카메라 사이 거리)
- d: 시차 (픽셀 단위)

시차 범위 예시:
┌─────────────────────────────────────────┐
│ 거리    │ 시차 (f=500, b=0.1m)          │
├─────────┼───────────────────────────────┤
│ 1m      │ 50 픽셀                       │
│ 5m      │ 10 픽셀                       │
│ 10m     │ 5 픽셀                        │
│ 무한대  │ 0 픽셀                        │
└─────────────────────────────────────────┘
```

### 스테레오 정합

```python
import cv2
import numpy as np

def stereo_calibrate(obj_points, img_points_left, img_points_right,
                     K1, D1, K2, D2, img_size):
    """스테레오 카메라 캘리브레이션"""

    flags = (cv2.CALIB_FIX_INTRINSIC +
             cv2.CALIB_RATIONAL_MODEL)

    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        img_points_left,
        img_points_right,
        K1, D1,
        K2, D2,
        img_size,
        flags=flags
    )

    print(f"스테레오 캘리브레이션 RMS 오차: {ret:.4f}")
    print(f"\n회전 행렬 R:\n{R}")
    print(f"\n평행 이동 벡터 T:\n{T.ravel()}")
    print(f"\n베이스라인: {np.linalg.norm(T):.4f} 단위")

    return R, T, E, F

def stereo_rectify(K1, D1, K2, D2, img_size, R, T):
    """스테레오 정류 (Rectification)"""

    # 정류 변환 계산
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1,
        K2, D2,
        img_size,
        R, T,
        alpha=0,  # 0: 유효 픽셀만, 1: 모든 픽셀
        newImageSize=img_size
    )

    # Q 행렬: 시차 → 3D 변환에 사용
    # [X Y Z W]^T = Q * [x y disparity 1]^T
    print("Q 행렬 (시차 → 3D 변환):")
    print(Q)

    return R1, R2, P1, P2, Q, roi1, roi2

def create_rectification_maps(K, D, R, P, img_size):
    """정류 맵 생성"""

    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, R, P, img_size, cv2.CV_32FC1
    )

    return map1, map2

def rectify_stereo_pair(img_left, img_right, maps_left, maps_right):
    """스테레오 이미지 쌍 정류"""

    rect_left = cv2.remap(img_left, maps_left[0], maps_left[1],
                          cv2.INTER_LINEAR)
    rect_right = cv2.remap(img_right, maps_right[0], maps_right[1],
                           cv2.INTER_LINEAR)

    return rect_left, rect_right
```

---

## 3. 깊이 맵 생성

### StereoBM (Block Matching)

```python
import cv2
import numpy as np

def compute_disparity_bm(left, right, num_disparities=64, block_size=15):
    """StereoBM을 이용한 시차 맵 계산"""

    # 그레이스케일 변환
    if len(left.shape) == 3:
        left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # StereoBM 생성
    stereo = cv2.StereoBM_create(
        numDisparities=num_disparities,  # 16의 배수
        blockSize=block_size              # 홀수, 5~21
    )

    # 파라미터 조정 (선택)
    stereo.setMinDisparity(0)
    stereo.setSpeckleWindowSize(100)
    stereo.setSpeckleRange(32)
    stereo.setPreFilterType(cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE)
    stereo.setPreFilterSize(9)
    stereo.setPreFilterCap(31)
    stereo.setTextureThreshold(10)
    stereo.setUniquenessRatio(15)

    # 시차 계산
    disparity = stereo.compute(left, right)

    # 시차 값 정규화 (16배로 스케일되어 있음)
    disparity = disparity.astype(np.float32) / 16.0

    return disparity

def visualize_disparity(disparity):
    """시차 맵 시각화"""

    # 유효한 시차만 사용
    valid_mask = disparity > 0

    # 정규화
    disp_vis = np.zeros_like(disparity)
    if np.any(valid_mask):
        disp_min = np.min(disparity[valid_mask])
        disp_max = np.max(disparity[valid_mask])
        disp_vis = (disparity - disp_min) / (disp_max - disp_min) * 255

    disp_vis = disp_vis.astype(np.uint8)

    # 컬러맵 적용
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    # 유효하지 않은 영역은 검은색으로
    disp_color[~valid_mask] = [0, 0, 0]

    return disp_color
```

### StereoSGBM (Semi-Global Block Matching)

```python
def compute_disparity_sgbm(left, right, num_disparities=64, block_size=5):
    """StereoSGBM을 이용한 시차 맵 계산"""

    # 그레이스케일 변환
    if len(left.shape) == 3:
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    else:
        gray_left, gray_right = left, right

    # SGBM 파라미터
    # P1, P2: 인접 픽셀 간 시차 차이에 대한 페널티
    P1 = 8 * 3 * block_size ** 2
    P2 = 32 * 3 * block_size ** 2

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # 시차 계산
    disparity = stereo.compute(gray_left, gray_right)
    disparity = disparity.astype(np.float32) / 16.0

    return disparity

def disparity_to_depth(disparity, Q):
    """시차 맵을 깊이 맵으로 변환"""

    # Q 행렬을 이용한 3D 재투영
    # points_3d[y, x] = [X, Y, Z, W]
    points_3d = cv2.reprojectImageTo3D(disparity, Q)

    # Z 값 (깊이) 추출
    depth = points_3d[:, :, 2]

    # 유효하지 않은 깊이 필터링
    valid_mask = (disparity > 0) & (depth > 0) & (depth < 10000)
    depth[~valid_mask] = 0

    return depth, points_3d

def create_depth_colormap(depth, max_depth=10.0):
    """깊이 맵 시각화"""

    # 깊이 클리핑
    depth_clipped = np.clip(depth, 0, max_depth)

    # 정규화 (0-255)
    depth_norm = (depth_clipped / max_depth * 255).astype(np.uint8)

    # 컬러맵 적용 (가까운 = 빨강, 먼 = 파랑)
    depth_color = cv2.applyColorMap(255 - depth_norm, cv2.COLORMAP_JET)

    # 유효하지 않은 영역 마스킹
    depth_color[depth <= 0] = [0, 0, 0]

    return depth_color
```

### WLS 필터를 이용한 시차 개선

```python
def compute_disparity_with_wls(left, right, num_disparities=64):
    """WLS 필터로 개선된 시차 맵 계산"""

    # 그레이스케일
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # 왼쪽 매처
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=5,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # 오른쪽 매처 (왼쪽-오른쪽 일관성 검사용)
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # 시차 계산
    left_disp = left_matcher.compute(gray_left, gray_right)
    right_disp = right_matcher.compute(gray_right, gray_left)

    # WLS 필터
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(80000)
    wls_filter.setSigmaColor(1.2)

    # 필터 적용
    filtered_disp = wls_filter.filter(left_disp, left, None, right_disp)
    filtered_disp = filtered_disp.astype(np.float32) / 16.0

    return filtered_disp
```

---

## 4. 포인트 클라우드

### 포인트 클라우드 생성

```python
import cv2
import numpy as np

def create_point_cloud(depth, rgb, K):
    """깊이 맵과 RGB 이미지로 포인트 클라우드 생성"""

    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 픽셀 좌표 그리드
    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)

    # 유효한 깊이 마스크
    valid = depth > 0

    # 3D 좌표 계산
    Z = depth[valid]
    X = (u[valid] - cx) * Z / fx
    Y = (v[valid] - cy) * Z / fy

    # 포인트 클라우드 (N x 3)
    points = np.stack([X, Y, Z], axis=-1)

    # 색상 정보 (N x 3)
    if len(rgb.shape) == 3:
        colors = rgb[valid]
    else:
        colors = np.stack([rgb[valid]] * 3, axis=-1)

    return points, colors

def subsample_point_cloud(points, colors, voxel_size=0.01):
    """복셀 그리드로 포인트 클라우드 다운샘플링"""

    # 복셀 인덱스 계산
    voxel_indices = np.floor(points / voxel_size).astype(int)

    # 고유한 복셀만 선택
    _, unique_indices = np.unique(
        voxel_indices, axis=0, return_index=True
    )

    return points[unique_indices], colors[unique_indices]

def save_point_cloud_ply(filename, points, colors):
    """PLY 형식으로 포인트 클라우드 저장"""

    n_points = len(points)

    # PLY 헤더
    header = f"""ply
format ascii 1.0
element vertex {n_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    with open(filename, 'w') as f:
        f.write(header)
        for i in range(n_points):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

    print(f"저장됨: {filename} ({n_points} 포인트)")
```

### 포인트 클라우드 처리

```python
def remove_outliers_statistical(points, colors, nb_neighbors=20, std_ratio=2.0):
    """통계적 이상치 제거"""

    from scipy.spatial import KDTree

    # KD-Tree 구축
    tree = KDTree(points)

    # 각 점의 k-NN 거리 계산
    distances, _ = tree.query(points, k=nb_neighbors + 1)
    mean_distances = np.mean(distances[:, 1:], axis=1)  # 자기 자신 제외

    # 전체 평균과 표준편차
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)

    # 이상치 마스크
    threshold = global_mean + std_ratio * global_std
    inlier_mask = mean_distances < threshold

    print(f"이상치 제거: {len(points)} → {np.sum(inlier_mask)} 포인트")

    return points[inlier_mask], colors[inlier_mask]

def estimate_normals(points, k=30):
    """포인트 클라우드 법선 벡터 추정"""

    from scipy.spatial import KDTree
    from numpy.linalg import eig

    tree = KDTree(points)
    normals = np.zeros_like(points)

    for i, point in enumerate(points):
        # k-NN 검색
        _, indices = tree.query(point, k=k)
        neighbors = points[indices]

        # 공분산 행렬
        centered = neighbors - np.mean(neighbors, axis=0)
        cov = np.dot(centered.T, centered) / k

        # 가장 작은 고유값의 고유벡터가 법선
        eigenvalues, eigenvectors = eig(cov)
        min_idx = np.argmin(eigenvalues)
        normals[i] = eigenvectors[:, min_idx]

    return normals
```

---

## 5. Open3D 기초

### Open3D 설치 및 기본 사용

```python
# pip install open3d

import open3d as o3d
import numpy as np

def create_open3d_point_cloud(points, colors=None):
    """Open3D 포인트 클라우드 생성"""

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        # 색상을 0-1 범위로 정규화
        if colors.max() > 1:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def visualize_point_cloud(pcd):
    """포인트 클라우드 시각화"""

    # 좌표축 추가
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0]
    )

    o3d.visualization.draw_geometries(
        [pcd, coordinate_frame],
        window_name="Point Cloud",
        width=1280,
        height=720,
        point_show_normal=False
    )

def process_point_cloud_open3d(pcd):
    """Open3D로 포인트 클라우드 처리"""

    print(f"원본 포인트 수: {len(pcd.points)}")

    # 1. 다운샘플링
    pcd_down = pcd.voxel_down_sample(voxel_size=0.02)
    print(f"다운샘플링 후: {len(pcd_down.points)}")

    # 2. 이상치 제거
    pcd_clean, _ = pcd_down.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )
    print(f"이상치 제거 후: {len(pcd_clean.points)}")

    # 3. 법선 추정
    pcd_clean.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30
        )
    )

    # 4. 법선 방향 정렬
    pcd_clean.orient_normals_consistent_tangent_plane(k=15)

    return pcd_clean
```

### 메쉬 재구성

```python
def reconstruct_mesh_poisson(pcd, depth=9):
    """포아송 표면 재구성"""

    # 법선이 필요함
    if not pcd.has_normals():
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(k=15)

    # 포아송 재구성
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    # 저밀도 영역 제거
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.01)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    print(f"메쉬 정점 수: {len(mesh.vertices)}")
    print(f"메쉬 삼각형 수: {len(mesh.triangles)}")

    return mesh

def reconstruct_mesh_ball_pivoting(pcd):
    """볼 피벗팅 표면 재구성"""

    if not pcd.has_normals():
        pcd.estimate_normals()

    # 반경 추정
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radii = [avg_dist, avg_dist * 2, avg_dist * 4]

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )

    return mesh

def save_mesh(mesh, filename):
    """메쉬 저장"""
    o3d.io.write_triangle_mesh(filename, mesh)
    print(f"메쉬 저장됨: {filename}")
```

### RGBD 이미지 처리

```python
def create_rgbd_from_opencv(color_img, depth_img, K):
    """OpenCV 이미지를 Open3D RGBD로 변환"""

    # BGR → RGB
    color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

    # Open3D 이미지로 변환
    color_o3d = o3d.geometry.Image(color_rgb)
    depth_o3d = o3d.geometry.Image(depth_img.astype(np.float32))

    # RGBD 이미지 생성
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1000.0,  # mm → m
        depth_trunc=3.0,     # 최대 깊이
        convert_rgb_to_intensity=False
    )

    return rgbd

def rgbd_to_point_cloud(rgbd, K, width, height):
    """RGBD 이미지에서 포인트 클라우드 생성"""

    # Open3D 카메라 파라미터
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height,
        K[0, 0], K[1, 1],  # fx, fy
        K[0, 2], K[1, 2]   # cx, cy
    )

    # 포인트 클라우드 생성
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsic
    )

    return pcd
```

---

## 6. 3D 재구성

### 다중 뷰 스테레오 (MVS) 개념

```
다중 뷰 스테레오 파이프라인:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. 이미지 수집                                                 │
│     여러 각도에서 대상 촬영                                     │
│         📷 📷 📷 📷 📷                                          │
│                                                                 │
│  2. 특징점 검출 및 매칭                                         │
│     SIFT, ORB 등으로 이미지 간 대응점 찾기                      │
│         ● ─────────── ●                                         │
│                                                                 │
│  3. Structure from Motion (SfM)                                 │
│     카메라 포즈 추정 + 희소 포인트 클라우드                     │
│         📷────┐    ●                                            │
│         📷────┼────● ●                                          │
│         📷────┘    ●                                            │
│                                                                 │
│  4. 조밀 재구성 (Dense Reconstruction)                          │
│     모든 픽셀에 대해 깊이 추정                                  │
│         [:::::::::::]                                           │
│                                                                 │
│  5. 메쉬 생성                                                   │
│     포인트 클라우드 → 삼각형 메쉬                               │
│         ▲▲▲▲▲▲▲▲                                              │
│                                                                 │
│  6. 텍스처 매핑                                                 │
│     원본 이미지로 메쉬에 텍스처 적용                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Essential Matrix 기반 포즈 추정

```python
import cv2
import numpy as np

def estimate_pose_from_essential(pts1, pts2, K):
    """Essential Matrix로 상대 포즈 추정"""

    # Essential Matrix 계산
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    print(f"인라이어 비율: {np.sum(mask) / len(mask) * 100:.1f}%")

    # Essential Matrix에서 R, t 복구
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

    print(f"\n회전 행렬 R:\n{R}")
    print(f"\n평행 이동 벡터 t (단위 벡터):\n{t.ravel()}")

    return R, t

def triangulate_points(pts1, pts2, K, R, t):
    """두 뷰에서 3D 점 삼각측량"""

    # 투영 행렬 구성
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    # 삼각측량
    pts1_h = pts1.T  # (2, N)
    pts2_h = pts2.T

    points_4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)

    # 동차 좌표 → 3D 좌표
    points_3d = points_4d[:3] / points_4d[3]

    return points_3d.T  # (N, 3)

def incremental_sfm(images, K):
    """증분적 SfM (간단한 버전)"""

    # SIFT 검출기
    sift = cv2.SIFT_create()

    # 첫 두 이미지로 초기화
    kp1, desc1 = sift.detectAndCompute(images[0], None)
    kp2, desc2 = sift.detectAndCompute(images[1], None)

    # 매칭
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    # 비율 테스트
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # 초기 포즈 및 3D 점
    R, t = estimate_pose_from_essential(pts1, pts2, K)
    points_3d = triangulate_points(pts1, pts2, K, R, t)

    # 카메라 포즈 저장
    camera_poses = [
        {'R': np.eye(3), 't': np.zeros((3, 1))},  # 첫 번째 카메라
        {'R': R, 't': t}                           # 두 번째 카메라
    ]

    print(f"초기 3D 점 수: {len(points_3d)}")

    # 이후 이미지 추가 (PnP로 포즈 추정)
    for i in range(2, len(images)):
        kp_new, desc_new = sift.detectAndCompute(images[i], None)

        # 이전 이미지와 매칭
        matches = bf.knnMatch(desc2, desc_new, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # 3D-2D 대응점
        obj_points = points_3d[[m.queryIdx for m in good_matches]]
        img_points = np.float32([kp_new[m.trainIdx].pt for m in good_matches])

        # PnP로 포즈 추정
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_points, img_points, K, None
        )

        if success:
            R_new, _ = cv2.Rodrigues(rvec)
            camera_poses.append({'R': R_new, 't': tvec})
            print(f"이미지 {i} 등록 완료 (인라이어: {len(inliers)})")

        # 다음 반복을 위해 업데이트
        desc2 = desc_new

    return points_3d, camera_poses
```

### 번들 조정 (Bundle Adjustment)

```
번들 조정 (Bundle Adjustment):
카메라 파라미터와 3D 점 위치를 동시에 최적화

최소화 목표:
E = Σ_i Σ_j || x_ij - π(K, R_i, t_i, X_j) ||²

여기서:
- x_ij: 이미지 i에서 관측된 점 j의 2D 좌표
- π(): 3D → 2D 투영 함수
- K: 카메라 내부 파라미터
- R_i, t_i: 카메라 i의 포즈
- X_j: 3D 점 j의 좌표

최적화 도구:
- Ceres Solver
- g2o
- SciPy (작은 문제용)
```

---

## 7. 연습 문제

### 문제 1: 스테레오 깊이 추정

스테레오 이미지 쌍에서 깊이 맵을 생성하세요.

**요구사항**:
- StereoBM과 StereoSGBM 비교
- 시차 맵 시각화
- 깊이 맵으로 변환
- 품질 개선 (필터링)

<details>
<summary>힌트</summary>

```python
# 파라미터 튜닝 필요
stereo = cv2.StereoSGBM_create(
    numDisparities=128,
    blockSize=5,
    P1=8 * 3 * 5 ** 2,
    P2=32 * 3 * 5 ** 2
)

# WLS 필터로 개선
wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
```

</details>

### 문제 2: 포인트 클라우드 필터링

노이즈가 있는 포인트 클라우드를 정제하세요.

**요구사항**:
- 통계적 이상치 제거
- 복셀 다운샘플링
- 평면 영역 추출
- 결과 시각화

<details>
<summary>힌트</summary>

```python
import open3d as o3d

# 이상치 제거
pcd_clean, _ = pcd.remove_statistical_outlier(
    nb_neighbors=20, std_ratio=2.0
)

# 다운샘플링
pcd_down = pcd_clean.voxel_down_sample(0.02)

# 평면 추출 (RANSAC)
plane_model, inliers = pcd_down.segment_plane(
    distance_threshold=0.01,
    ransac_n=3,
    num_iterations=1000
)
```

</details>

### 문제 3: 두 뷰에서 3D 재구성

두 이미지에서 3D 포인트를 재구성하세요.

**요구사항**:
- 특징점 검출 및 매칭
- Essential Matrix 계산
- 카메라 포즈 복구
- 삼각측량으로 3D 점 생성

<details>
<summary>힌트</summary>

```python
# Essential Matrix
E, mask = cv2.findEssentialMat(pts1, pts2, K)

# 포즈 복구
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

# 삼각측량
points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
points_3d = points_4d[:3] / points_4d[3]
```

</details>

### 문제 4: 메쉬 재구성

포인트 클라우드에서 3D 메쉬를 생성하세요.

**요구사항**:
- 포인트 클라우드 전처리
- 법선 벡터 추정
- 포아송 또는 볼 피벗팅 재구성
- 결과 저장 및 시각화

<details>
<summary>힌트</summary>

```python
# 법선 추정
pcd.estimate_normals()
pcd.orient_normals_consistent_tangent_plane(k=15)

# 포아송 재구성
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9
)

# 저밀도 영역 제거
densities = np.asarray(densities)
mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.01))
```

</details>

### 문제 5: 실시간 스테레오 비전

웹캠 또는 스테레오 카메라로 실시간 깊이 추정을 구현하세요.

**요구사항**:
- 카메라 캘리브레이션 적용
- 실시간 시차 계산
- 깊이 시각화
- FPS 측정

<details>
<summary>힌트</summary>

```python
# 리맵핑 맵 미리 계산
map1_left, map2_left = cv2.initUndistortRectifyMap(...)
map1_right, map2_right = cv2.initUndistortRectifyMap(...)

while True:
    # 정류
    rect_left = cv2.remap(left, map1_left, map2_left, cv2.INTER_LINEAR)
    rect_right = cv2.remap(right, map1_right, map2_right, cv2.INTER_LINEAR)

    # 시차 계산 (SGBM)
    disparity = stereo.compute(rect_left, rect_right)
```

</details>

---

## 다음 단계

- [22_Depth_Estimation.md](./22_Depth_Estimation.md) - 단안 깊이 추정, MiDaS, DPT, Structure from Motion

---

## 참고 자료

- [OpenCV Stereo Vision Tutorial](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)
- [Open3D Documentation](http://www.open3d.org/docs/)
- [Multiple View Geometry in Computer Vision](https://www.robots.ox.ac.uk/~vgg/hzbook/)
- [Structure from Motion Tutorial](https://github.com/colmap/colmap)
- [Stereo Vision: A Tutorial](https://people.cs.rutgers.edu/~elgammal/classes/cs534/lectures/Stereo_2.pdf)
