# 개인의 특성 기반 패션 추천 시스템

## 1. 프로젝트 소개

### 1.1. 연구 배경 및 필요성

현대 사회에서 패션은 개인의 개성을 표현하는 중요한 수단입니다. 그러나 모든 사람에게 적합한 패션을 찾기란 쉽지 않습니다. 특히 **퍼스널 컬러**, **얼굴형**, **체형** 등 신체적 특성에 기반한 맞춤형 패션 추천 시스템은 아직 일반적이지 않으며, 이로 인해 사용자들은 자신에게 적합한 패션을 쉽게 찾지 못하는 어려움이 있습니다.

최근 **개인 맞춤형 서비스에 대한 수요가 증가**하고 있으며, 이를 바탕으로 신체적 특성을 고려한 패션 추천 시스템을 개발하는 것이 중요해지고 있습니다. 본 프로젝트는 이러한 필요성을 바탕으로 개인의 신체적 특성을 분석하여 **맞춤형 패션 아이템을 추천**하는 시스템을 개발하였습니다.

### 1.2. 목표 및 주요 내용

**목표**

- **개인의 신체적 특성 분석**: 사용자의 **퍼스널 컬러**, **얼굴형**, **체형**을 자동으로 분석합니다.
- **맞춤형 패션 아이템 추천**: 분석된 신체적 특성에 기반하여 사용자에게 적합한 패션 아이템을 추천합니다.
- **사용자 피드백 반영**: 사용자의 평가를 반영하여 추천 품질을 지속적으로 향상시킵니다.
- **사용자 친화적인 웹 인터페이스 제공**: 누구나 쉽게 사용할 수 있는 웹 인터페이스를 통해 서비스를 제공합니다.

**주요 내용**

- **이미지 업로드 및 처리**: 얼굴 사진과 전신 사진을 업로드하여 신체적 특성을 자동으로 분석합니다.
- **신체 정보 직접 입력 기능**: 사진 없이도 퍼스널 컬러, 얼굴형, 체형을 직접 입력하여 추천을 받을 수 있습니다.
- **계절별 추천 기능**: 사용자가 선택한 계절에 맞는 패션 아이템을 추천합니다.
- **추천 결과 표시 및 피드백 수집**: 사용자에게 분석 결과와 함께 추천된 패션 아이템 목록을 제공하고, 사용자의 피드백을 수집하여 추천 모델을 개선합니다.

### 모델 개발에 사용한 데이터

- [Kaggle Faceshape Dataset](https://www.kaggle.com/datasets/niten19/face-shape-dataset)
- [연도별 패션 선호도 파악 및 추천 데이터 2019년](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71446)

## 2. 상세설계

### 2.1. 시스템 구성도

시스템은 크게 **프론트엔드**, **백엔드**, **머신러닝 모델**로 구성되어 있습니다.

![시스템 구성도](https://github.com/user-attachments/assets/6db08277-5268-4c8a-898d-6287a4a74e4e)

- **프론트엔드**

  - **역할**: 사용자와의 인터랙션을 담당하며, 사진 업로드, 신체 정보 입력, 추천 결과 표시, 피드백 수집 등의 기능을 제공합니다.
  - **구성요소**:
    - **HTML5**, **CSS3**: 웹 페이지 구조 및 스타일링
    - **JavaScript (ES6)**: 동적 기능 구현 및 API 통신

- **백엔드**

  - **역할**: 프론트엔드에서 요청한 작업을 처리하고, 머신러닝 모델을 통해 신체적 특성을 분석하며, 추천 엔진을 통해 패션 아이템을 추천하고 사용자 피드백을 반영합니다.
  - **구성요소**:
    - **Flask**: 웹 서버 및 API 엔드포인트 제공
    - **Redis & RQ**: 비동기 작업 처리를 위한 큐 시스템
    - **머신러닝 모델 관리 및 업데이트 로직**

- **머신러닝 모델**

  - **퍼스널 컬러 측정 모델**
    - **데이터 전처리**: 얼굴 이미지에서 피부 영역만 추출하여 RGB 값을 얻습니다.
    - **색상 공간 변환 및 분류**: HSV 및 Lab 색상 공간으로 변환한 후, K-Means 클러스터링을 통해 퍼스널 컬러를 분류합니다.
  - **얼굴형 분류 모델**
    - **모델 아키텍처**: EfficientNetB4 모델 기반의 전이학습을 사용한 딥러닝 모델 사용
    - **데이터 증강**: 이미지 회전, 반전 등을 통해 데이터셋을 확장하여 모델의 일반화 성능을 향상시킵니다.
  - **체형 측정 모델**
    - **KeyPoint 추출**: MediaPipe를 사용하여 신체 주요 부위의 좌표를 추출합니다.
    - **허리 및 가슴 위치 추정**: 한국인의 평균 신체 비율 데이터를 활용하여 허리와 가슴의 위치를 예측합니다.
    - **배경 제거 및 길이 측정**: Rembg 라이브러리를 사용하여 배경을 제거하고 신체 부위의 너비를 측정합니다.
  - **추천 시스템 모델**
    - **Random Forest Regressor 기반 모델**
      - 사용자 특성과 옷의 특성을 입력으로 받아 예상 평점을 예측합니다.
    - **콘텐츠 기반 협업 필터링**
      - 옷의 텍스트 특성을 Tf-Idf로 벡터화하여 코사인 유사도를 계산하고, 유사한 아이템을 추천합니다.
    - **사용자 피드백 반영**
      - 사용자의 평가를 모델 학습에 반영하여 추천 품질을 지속적으로 개선합니다.

### 2.2. 사용 기술

#### 백엔드

- **언어 및 프레임워크**
  - **Python 3.11**
  - **Flask**
- **패키지 및 라이브러리**
  - **이미지 처리 및 컴퓨터 비전**
    - **OpenCV**
    - **MediaPipe**
    - **Dlib**
    - **Rembg**
  - **머신러닝 딥러닝 및 데이터 처리**
    - **TensorFlow**
    - **scikit-learn**
    - **Pandas**
    - **Numpy**
    - **Joblib**
  - **기타**
    - **Redis**
    - **RQ (Redis Queue)**
    - **Flask-Session**

#### 프론트엔드

- **언어 및 라이브러리**
  - **HTML5**
  - **CSS3**
  - **JavaScript (ES6)**

#### 머신러닝 모델 및 도구

- **퍼스널 컬러 측정 모델**
  - **K-Means Clustering** (`kmeans_model_L2.pkl`)
- **얼굴형 분류 모델**
  - **TensorFlow Lite** (`faceshape_efficientnetb4_crop.tflite`)
  - **EfficientNetB4**
  - **Haar Cascade Classifier** (`haarcascade_frontalface_default.xml`)
- **체형 측정 모델**
  - **MediaPipe Pose Estimation**
  - **Dlib Face Landmarks** (`shape_predictor_68_face_landmarks.dat`)
- **추천 시스템 모델**
  - **Random Forest Regressor**
  - **콘텐츠 기반 협업 필터링**

## 3. 설치 및 사용 방법

### 3.1. 환경 설정

#### 필수 소프트웨어

- **Python 3.8 이상**
- **Redis 서버**

#### 의존성 설치

1. **프로젝트 클론**

   ```bash
   git clone https://github.com/NextrPlue/PersonalLix.git
   cd PersonalLix
   ```

2. **가상환경 생성 및 활성화**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows의 경우: venv\Scripts\activate
   ```

3. **필요한 패키지 설치**

   ```bash
   pip install -r requirements.txt
   ```

### 3.2. Redis 서버 실행

Redis가 설치되어 있지 않은 경우 [Redis 공식 사이트](https://redis.io/download)에서 설치합니다.

Redis 서버를 실행합니다.

```bash
redis-server
```

### 3.3. 서버 및 워커 실행

1. **Flask 앱 실행**

   ```bash
   python app.py
   ```

2. **워커(worker) 실행**

   ```bash
   python worker.py
   ```

### 3.4. 웹 애플리케이션 사용

웹 브라우저에서 `http://localhost:5000`에 접속하여 애플리케이션을 사용합니다.

#### 사진 업로드 방식

1. **이미지 업로드**

   - **얼굴 사진**
   - **전신 사진**
   - **팔을 올린 전신 사진**

2. **정보 입력**

   - **성별 선택**
   - **나이 입력**

3. **추천 받기**

   "추천 받기" 버튼을 클릭하여 신체적 특성 분석 및 패션 아이템 추천을 받습니다.

#### 신체 정보 직접 입력 방식

1. **신체 정보 입력**

   - **퍼스널 컬러 선택**
   - **얼굴형 선택**
   - **체형 선택**

2. **정보 입력**

   - **성별 선택**
   - **나이 입력**

3. **추천 받기**

   "추천 받기" 버튼을 클릭하여 패션 아이템 추천을 받습니다.

### 3.5. 주요 기능

- **신체적 특성 분석**

  - **퍼스널 컬러 측정**

    - 얼굴 이미지에서 피부 영역을 추출하고, RGB 값을 기반으로 HSV 및 Lab 색상 공간으로 변환하여 퍼스널 컬러를 분류합니다.
    - K-Means 클러스터링을 통해 봄, 여름, 가을, 겨울 타입으로 분류합니다.

  - **얼굴형 분류**

    - EfficientNetB4 기반의 딥러닝 모델을 사용하여 얼굴형을 하트형, 긴형, 타원형, 둥근형, 사각형으로 분류합니다.
    - 데이터 증강 및 정규화를 통해 모델의 성능을 향상시켰습니다.

  - **체형 측정**

    - MediaPipe를 사용하여 신체의 주요 KeyPoint를 추출하고, 한국인의 평균 신체 비율 데이터를 활용하여 허리와 가슴의 위치를 예측합니다.
    - Rembg 라이브러리를 사용하여 배경을 제거하고, 신체 부위의 너비를 정확하게 측정합니다.

- **패션 아이템 추천**

  - **Random Forest 기반 추천 시스템**

    - 사용자 특성(퍼스널 컬러, 얼굴형, 체형)과 옷의 특성을 입력으로 받아 예상 평점을 예측합니다.
    - 사용자 피드백을 반영하여 모델을 지속적으로 업데이트합니다.

  - **콘텐츠 기반 협업 필터링**

    - 옷의 텍스트 특성을 Tf-Idf로 벡터화하고, 코사인 유사도를 계산하여 유사한 아이템을 추천합니다.
    - 사용자 특성에 맞는 옷의 특성을 연결하여 더욱 개인화된 추천을 제공합니다.

- **계절 선택 기능**

  - 계절을 선택하여 해당 계절에 맞는 패션 아이템을 추천받을 수 있습니다.

- **추천 아이템 더 보기**

  - 추천 결과 페이지에서 "더 보기" 버튼을 클릭하여 추가적인 추천 아이템을 확인할 수 있습니다.

- **추천 품질에 대한 피드백 반영**

  - 사용자의 평가를 모델 학습에 반영하여 추천 품질을 지속적으로 개선합니다.

## 4. 소개 및 시연 영상

본 시스템은 개인의 신체적 특성을 분석하여 맞춤형 패션 아이템을 추천하는 웹 애플리케이션입니다.

사용자는 자신의 얼굴 및 전신 사진을 업로드하거나 신체 정보를 직접 입력하여 추천을 받을 수 있습니다. 분석된 퍼스널 컬러, 얼굴형, 체형 정보를 바탕으로 머신러닝 기반의 추천 엔진이 최적의 패션 아이템을 제안합니다.

### 4.1. 시연 과정

1. **웹 애플리케이션 접속**

   - 웹 브라우저에서 `http://localhost:5000`에 접속합니다.

2. **사진 업로드 또는 신체 정보 입력**

   - **사진 업로드 방식**: 얼굴 사진, 전신 사진, 팔을 올린 전신 사진을 업로드합니다.
   - **직접 입력 방식**: 퍼스널 컬러, 얼굴형, 체형을 선택합니다.

3. **성별 및 나이 입력**

   - 성별을 선택하고 나이를 입력합니다.

4. **추천 받기**

   - "추천 받기" 버튼을 클릭하여 분석 결과와 추천 아이템을 확인합니다.

5. **추천 아이템 확인**

   - 추천된 패션 아이템 목록과 각 아이템의 상세 정보를 확인할 수 있습니다.

6. **추가 아이템 확인**

   - 필요한 경우 "더 보기" 버튼을 클릭하여 더 많은 추천 아이템을 볼 수 있습니다.

7. **평점 매기기**

   - 추천된 패션 사진을 클릭하여 평점을 매길 수 있습니다.

### 4.2. 시연 영상

[![시연 영상](http://img.youtube.com/vi/bAQ_bLVpX28/0.jpg)](https://youtu.be/bAQ_bLVpX28)

## 5. 팀 소개

- **이성훈** (201924532)

  - 역할: 퍼스널컬러 측정 개발, 추천 시스템 개발, 백엔드 및 프론트엔드 개발
  - e-mail: p.plue1881@gmail.com
  - github: https://github.com/NextrPlue

- **김태훈** (201924451)

  - 역할: 얼굴형 측정 개발, 체형 측정 개발, 추천 시스템 개발, 백엔드 개발
  - e-mail: bigteach0508@pusan.ac.kr
  - github: https://github.com/minchoCoin

---

## 6. 참고 문헌

1. Yun-Seok Jung, “A Study on the Quantitative Diagnosis Model of Personal Color,” Journal of Convergence for Information Technology, Vol. 11, No. 11, pp. 277-287, 2021. (in Korean)
2. Jong-Suk An, “A Study on Effective Image Making Depending on Hair Style and Neckline,” J Korean Soc Cosmetol, Vol. 15, No. 1, pp.342-351, 2009. (in Korean)
3. Soo-ae Kwon, Fashion and Life, Gyohakyungusa, 2016.
4. “2023 Consumption Trend Series - 03 Personalized Services,” MezzoMedia [Online], 
Available: https://www.mezzomedia.co.kr/data/insight_m_file/insight_m_file_1605.pdf (downloaded 2024, May. 19)
5. So-young Lee, “Personal Color Tone Type and Categorization of Harmonious Colors According to Skin Color,” M.S. thesis, Graduate School of Cultural and Information Policy, Hongik Univ., Seoul, South Korea, 2019. (in Korean)
6. Trotter, Cameron, et al. "Human body shape classification based on a single image." arXiv preprint arXiv:2305.18480 (2023)
