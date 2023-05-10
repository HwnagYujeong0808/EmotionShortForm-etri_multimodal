# ETRI Multimodal Contest - EmotionShortForm Code

- 이 저장소는 한국전자통신연구원(ETRI)이 주최하고 과학기술정보통신부와 국가과학기술연구회(NST)가 후원하는 **"제2회 ETRI 휴먼이해 인공지능 논문경진대회"** 에 제출할 소스코드를 담고 있습니다.
- **팀명**: Emotion숏폼

## 프로젝트 소개

- 본 연구에서는 *음성에서 추출한 감정 정보*와 *영상에서 추출한 이미지*를 동시에 활용하는 **멀티모달 모델**을 제안하고, 이를 활용한 유튜브 하이라이트 자동 추출 모델을 제안한다. 제안하는 핵심 아이디어는 **Vision Transformer(ViT)** 모델로부터 추출한 영상 프레임 특징과 **Long Short-Term Memory(LSTM)** 기반 감정 분류 및 예측 모델을 통해 추출한 감정 특징을 함께 사용하는 멀티모달 모델을 구성한 것이다. 제안하는 방법의 효용성을 보이기 위하여 제안하는 모델과 음성 또는 영상만을 활용했을 때의 모델 성능을 비교한다. 실험 결과, 음성 감정 특징 기반 LSTM 모델 및 영상 특징 기반 ViT 모델보다, 제안한 멀티모달 모델의 **F1 Score**가 각각 약 **26.64%, 16.47%** 만큼 향상됨을 보였다.

## 데이터셋 소개 및 생성

- **데이터 1: ETRI 한국어 감정 데이터셋 KEMDy20 (일반인 대상 자유발화) 데이터셋**

  - **링크**: [KEMDy20\_데이터셋](https://nanum.etri.re.kr/share/kjnoh/KEMDy20?lang=ko_KR)
  - **소개**: 발화 음성, 발화의 문맥적 의미 및 생리반응 신호- 피부전도도, 맥박관련 데이터, 손목 피부온도와 발화자의 감정과의 연관성 분석을 위해 수집한 멀티모달 감정 데이터셋
  - **Train set**
    - 다운 경로: '01.데이터/2.Validation/원천데이터/VS\_유튜브\_04'
    - 가장 용량이 적은 21.4GB 폴더 안에 있는 영상 데이터만 사용함
  - **Test set**
    - 다운 경로: '01.데이터/2.Validation/원천데이터/VS\_유튜브\_01'
    - 폴더 안에 있는 8개의 영상 데이터만 사용함
      - '유튜브*기타\_19843', '유튜브*반려동물및동물*2153', '유튜브*스타일링및뷰티*14630', '유튜브*스포츠*4174', '유튜브*여행*7640', '유튜브*음식*17341', '유튜브*일상*10479', '유튜브*자동차\_0094'

- **데이터 2: AI HUB 동영상 콘텐츠 하이라이트 편집 및 설명(요약) 데이터**
  - **링크**: [AIHUB\_데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=616)
  - **소개**: AI HUB에서 수집한 동영상 콘텐츠 하이라이트 편집 및 설명(요약) 데이터셋은 뉴스 및 유튜브 영상에서 주요 장면의 위치를 레이블링하고 카테고리 항목에 대해 태깅하여 구축한 학습용 데이터셋
  - './audio_baseline/split_data.ipynb'에서 **Train set**, **Test set**을 8:2로 나누어 저장
  - './data_audio/train.csv', './data_audio/test.csv' 생성
  - './data_audio/wav_only/train', './data_audio/wav_only/test' 폴더 내에 wav 파일 나누어 저장

## 프로젝트 구조
<p  align="center"><img src="img.png" height="500px" width="500px"></p>

## 실행 방법

### 모델 학습 및 모델 생성

> 멀티모달 방식에서 쓰일 특징 벡터들을 concatenate 하기 위해 오디오, 비디오 특징 .npy 파일 생성

**1. LSTM 기반 음성 감정 분류 모델**

- run [audio_emotion_baseline_oversampling_SMOTE.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/lstm/audio_emotion_baseline_oversampling_SMOTE.ipynb)
  - _emotion_lstm_features.npy_ 생성 및 음성 감정 특징 저장
- run [youtube\_감정분류모델.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/lstm/youtube_%EA%B0%90%EC%A0%95%EB%B6%84%EB%A5%98%EB%AA%A8%EB%8D%B8.ipynb)
  - _lstm_emotion_classification_model.pt_ 모델 저장

**2. LSTM 기반 음성 각성도 예측 모델**

- run [lstm_arousal.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/lstm/lstm_arousal.ipynb)
  - _arousal_lstm_features.npy_ 생성 및 각성도 특징 저장
- run [youtube_Arousal예측모델.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/lstm/youtube_Arousal%EC%98%88%EC%B8%A1%EB%AA%A8%EB%8D%B8.ipynb)
  - _lstm_arousal_model_best.pt_ 모델 저장

**3. LSTM 기반의 음성 긍/부정도 예측 모델**

- run [lstm_valence.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/lstm/lstm_valence.ipynb)
  - _valence_lstm_features.npy_ 생성 및 긍/부정도 특징 저장
- run [youtube_Valence예측모델.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/lstm/youtube_Valence%EC%98%88%EC%B8%A1%EB%AA%A8%EB%8D%B8.ipynb)
  - _lstm_valence_model.pt_ 모델 저장

**4. 음성 감정 기반 하이라이트 추출 LSTM 모델**

- run [Youtube_feature_concatenate.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/lstm/Youtube_feature_concatenate.ipynb)
  - _concatenate_features_array.npy_ 생성 및 concatenate된 오디오 특징 저장
  - _concatenate_lstm_model.pt_ 모델 저장

**5. 영상 기반 하이라이트 추출 VIT 모델**

- run [final_vit_video.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/vit/final_vit_video.ipynb)
  - _concatenate_vit_features_array.npy_ 생성 및 영상 프레임 특징 저장

**6. 영상과 음성 특징을 모두 사용한 멀티모달 하이라이트 추출 LSTM 모델**

- run [multimodal_lstm_undersampling.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/lstm/multimodal_lstm_undersampling.ipynb)
  - _multimodal_model.pt_ 모델 저장

### 모델 추론
> 8개의 서로 다른 카테고리의 Test 데이터셋에 대해 세 가지 모델의 성능 평가 및 비교
 
- **1) (Baseline) 음성 감정 기반 하이라이트 추출 모델** 
  - run [final_audio.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/lstm/final_audio.ipynb)
  - 모델 불러오기
    1) _lstm_emotion_classification_model.pt_
    2) _lstm_arousal_model_best.pt_
    3) _lstm_valence_model.pt_
    4) _concatenate_lstm_model.pt_
  - 모델 성능 측정
    - **_concatenate_lstm_model.pt_** 모델을 사용해 음성 감정 기반 하이라이트 추출 baseline 모델의 성능 측정
####
 - **2) (Baseline) 영상 기반 하이라이트 추출 모델**
 - run [final_vit_video.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/vit/final_vit_video.ipynb)
 - 모델 성능 측정
    - **사전 학습된 ViT 모델**을 이용해 영상 기반 하이라이트 추출 baseline 모델의 성능 측정 
####
- **3) 영상과 음성 특징을 모두 사용한 멀티모달 하이라이트 추출 모델**
  - run [final_audio+video_best_undersampling.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/lstm/final_audio+video_best_undersampling.ipynb)
  - 모델 불러오기
    1) _lstm_emotion_classification_model.pt_
    2) _lstm_arousal_model_best.pt_
    3) _lstm_valence_model.pt_
    4) _concatenate_lstm_model_0.001.pt_
    5) _multimodal_model.pt_
  - 모델 성능 측정
    - **_multimodal_model.pt_** 모델을 사용해 프로젝트에서 제안하는 음성 감정 및 영상의 특징을 모두 활용한 멀티모달 모델의 성능 측정

## 결과

### LSTM 기반 음성 감정 분류 모델 및 LSTM 기반 음성 각성도 예측 모델 성능
<p  align="center"><img src="audio_result.png" width="500px" ></p>

### 하이라이트 추출 모델 성능 비교 
+ **임계값 그래프**
  + 멀티모달 하이라이트 추출 모델 임계값: 0.5
  + 음성 기반 하이라이트 추출 모델 임계값: 0.3
  + 영상 기반 하이라이트 추출 모델 임계값: 0.3
<p  align="center"><img src="threshold.png" width="500px" ></p>



+ **최종 결과**
<p  align="center"><img src="result_final.png" width="500px" ></p>

+ **Multimodal LSTM Model**:  영상과 음성 특징을 모두 사용한 멀티모달 하이라이트 추출 모델
+ (Baseline) Audio LSTM Model: 음성 감정 기반 하이라이트 추출 모델 
+ (Baseline) ViT Model: 영상 기반 하이라이트 추출 모델

