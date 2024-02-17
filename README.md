# Emotion-based Multimodal Learning Model for Video Highlight Detection


## Project Introduction

- In this research, we propose **a multimodal model** that simultaneously utilizes _emotional information extracted from audio_ and _images extracted from video_ for video highlight detection. The core idea proposed is to construct a Long Short-Term Memory (LSTM) multimodal model that uses both video frame features extracted by the **Vision Transformer (ViT)** model and emotional features extracted through a model based on **Wav2Vec**, including emotions, arousal, and valence. To demonstrate the effectiveness of the proposed method, we utilize AI HUB's YouTube video dataset to compare the performance of the proposed model with models that utilize only audio or video for highlight detection. The experimental results show that the proposed multimodal model improves the **F1 Score** by approximately **16.43% and 51.3%**, respectively, over models that utilize only audio or video.

## Dataset Introduction and Creation

- **Data 1: ETRI Korean Emotional Dataset KEMDy20 (Spontaneous Speech from the General Public)**

  - **Link**: [KEMDy20\ dataset](https://nanum.etri.re.kr/share/kjnoh/KEMDy20?lang=ko_KR)
  - **Introduction**: A multimodal emotional dataset collected for analyzing the relationship between the speaker's emotions and various signals such as speech audio, contextual meaning of speech, physiological response signals - galvanic skin response, heart-related data, and wrist skin temperature.
  - **Train set**
    - Download path: '01.데이터/2.Validation/원천데이터/VS_유튜브_04'
    - Only use video data from the folder with the smallest size, 21.4GB
  - **Test set**
    - Download path: '01.데이터/2.Validation/원천데이터/VS_유튜브_01'
    - Use only 16 video data files (8 Categories) within the folder
      - Categories: 'Others', 'PetsAndAnimals', 'StylingAndBeauty', 'Sports', 'Trave', 'Food', 'DailyLife', 'Automobile'

- **Data 2: AI HUB Video Content Highlights Editing and Description (Summary) Data**
  - **Link**: [AIHUB\_dataset](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=616)
  - **Introduction**: In './audio_baseline/split_data.ipynb', divide **Train set** and **Test set** into an **8:2** ratio.
  - Create './data_audio/train.csv', './data_audio/test.csv'
  - Divide and save wav files in './data_audio/wav_only/train' and './data_audio/wav_only/test' folders

## Project Structure
<p  align="center"><img src="img.png" height="500px" width="500px"></p>

## Execution Method

### Model Training and Model Creation

> To concatenate feature vectors to be used in the multimodal approach, create .npy files for audio and video features.

**1.1 LSTM-based Audio Emotion Classification Model**

- run [audio_emotion_baseline_oversampling_SMOTE.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/lstm/audio_emotion_baseline_oversampling_SMOTE.ipynb)
  - _emotion_lstm_features.npy_ 생성 및 음성 감정 특징 저장
- run [youtube\_감정분류모델.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/lstm/youtube_%EA%B0%90%EC%A0%95%EB%B6%84%EB%A5%98%EB%AA%A8%EB%8D%B8.ipynb)
  - _lstm_emotion_classification_model.pt_ 모델 저장

**1.2 LSTM-based Audio Arousal Prediction Model**

- run [lstm_arousal.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/lstm/lstm_arousal.ipynb)
  - _arousal_lstm_features.npy_ 생성 및 각성도 특징 저장
- run [youtube_Arousal예측모델.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/lstm/youtube_Arousal%EC%98%88%EC%B8%A1%EB%AA%A8%EB%8D%B8.ipynb)
  - _lstm_arousal_model_best.pt_ 모델 저장

**1.3 LSTM-based Audio Valence Prediction Model**

- run [lstm_valence.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/lstm/lstm_valence.ipynb)
  - _valence_lstm_features.npy_ 생성 및 긍/부정도 특징 저장
- run [youtube_Valence예측모델.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/lstm/youtube_Valence%EC%98%88%EC%B8%A1%EB%AA%A8%EB%8D%B8.ipynb)
  - _lstm_valence_model.pt_ 모델 저장

**2.1 Wav2Vec2-based Audio Emotion Classification Model**

- run [audio_emotion_baseline_oversampling_SMOTE.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/lstm/audio_emotion_baseline_oversampling_SMOTE.ipynb)
  - _emotion_lstm_features.npy_ 생성 및 음성 감정 특징 저장
- run [youtube\_감정분류모델.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/lstm/youtube_%EA%B0%90%EC%A0%95%EB%B6%84%EB%A5%98%EB%AA%A8%EB%8D%B8.ipynb)
  - _lstm_emotion_classification_model.pt_ 모델 저장

**2.2 Wav2Vec2-based Audio Arousal/Valence Prediction Model**

- run [lstm_arousal.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/lstm/lstm_arousal.ipynb)
  - _arousal_lstm_features.npy_ 생성 및 각성도 특징 저장
- run [youtube_Arousal예측모델.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/lstm/youtube_Arousal%EC%98%88%EC%B8%A1%EB%AA%A8%EB%8D%B8.ipynb)
  - _lstm_arousal_model_best.pt_ 모델 저장


**3. 음성 감정 기반 하이라이트 추출 LSTM 모델**

- run [Youtube_feature_concatenate.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/lstm/Youtube_feature_concatenate.ipynb)
  - _concatenate_features_array.npy_ 생성 및 concatenate된 오디오 특징 저장
  - _concatenate_lstm_model.pt_ 모델 저장

**4. 영상 기반 하이라이트 추출 VIT 모델**

- run [final_vit_video.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/vit/final_vit_video.ipynb)
  - _concatenate_vit_features_array.npy_ 생성 및 영상 프레임 특징 저장

**5. 영상과 음성 특징을 모두 사용한 멀티모달 하이라이트 추출 LSTM 모델**

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

### LSTM/Wav2Vec 기반 음성 감정 분류 모델 및 음성 각성도 예측 모델 성능 비교
<p  align="center"><img src="audio_result_1.png" width="500px" ></p>

<p  align="center"><img src="audio_result_2.png" width="500px" ></p>

### 하이라이트 추출 모델 성능 비교 
+ **임계값**
  + 멀티모달 하이라이트 추출 모델 임계값: 0.3
  + 음성 기반 하이라이트 추출 모델 임계값: 0.3
  + 영상 기반 하이라이트 추출 모델 임계값: 0.3



+ **최종 결과**
<p  align="center"><img src="result_plot.png" width="500px" ></p>
<p  align="center"><img src="result_final.png" width="500px" ></p>

+ **Multimodal LSTM Model**:  영상과 음성 특징을 모두 사용한 멀티모달 하이라이트 추출 모델
+ (Baseline) Audio LSTM Model: 음성 감정 기반 하이라이트 추출 모델 
+ (Baseline) ViT Model: 영상 기반 하이라이트 추출 모델

