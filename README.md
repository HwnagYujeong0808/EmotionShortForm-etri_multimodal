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
<p  align="center"><img src="architecture.png" height="500px" width="500px"></p>

## Execution Method
# Multimodal Highlight Detection – Implementation Guide

This repository contains the full implementation of our multimodal highlight detection system, integrating video, audio, and emotional cues. The pipeline is modular, reproducible, and ready for real-world use cases.

---

## File Overview

| Notebook | Description |
|----------|-------------|
| `a_0_create_wav.ipynb` | Convert raw input to `.wav` format audio clips |
| `a_1_Wav2Vec2_emotion_classification.ipynb` | Wav2Vec2-based categorical emotion classification |
| `a_1_Wav2Vec2_arousal_valence_prediction.ipynb` | Wav2Vec2-based arousal & valence prediction |
| `a_2_extract_waveform_A_av_mul_data.ipynb` | Extract waveform-only features |
| `a_2_extract_waveform_AE_av_mul_data.ipynb` | Extract waveform + emotion features |
| `a_3_training_evaluation_waveform_A.ipynb` | Train & evaluate highlight model (audio-only) |
| `a_3_training_evaluation_waveform_AE.ipynb` | Train & evaluate highlight model (audio + emotion) |
| `m_1_multimodal_A_V_waveform_pad.ipynb` | Train multimodal (Audio + Video) model |
| `m_1_multimodal_AE_V_waveform_pad.ipynb` | Train multimodal (Audio + Emotion + Video) model |

---

## Implementation Steps

### 1. Audio Preparation
Run: `a_0_create_wav.ipynb`  
→ Convert raw data into `.wav` audio clips (16kHz, mono)

---

### 2. Emotion / Affect Recognition
Run:
- `a_1_Wav2Vec2_emotion_classification.ipynb`
- `a_1_Wav2Vec2_arousal_valence_prediction.ipynb`  
→ Predict emotion class, arousal & valence scores using fine-tuned Wav2Vec2

---

### 3. Feature Extraction
Run:
- `a_2_extract_waveform_A_av_mul_data.ipynb`
- `a_2_extract_waveform_AE_av_mul_data.ipynb`  
→ Extract features for downstream tasks: A (audio-only), AE (audio + emotion)

---

### 4. Unimodal Training & Evaluation
Run:
- `a_3_training_evaluation_waveform_A.ipynb`
- `a_3_training_evaluation_waveform_AE.ipynb`  
→ Train and evaluate on audio-only or audio+emotion features

---

### 5. Multimodal Highlight Detection
Run:
- `m_1_multimodal_A_V_waveform_pad.ipynb`
- `m_1_multimodal_AE_V_waveform_pad.ipynb`  
→ Combine video + audio (A_V) or video + audio + emotion (AE_V)

---

##  Requirements

Install dependencies:
```bash
pip install torch torchaudio transformers pytorch_lightning librosa


### model inference
> Evaluate and compare the performance of three models on eight different categories of Test datasets
 
- **1) (Baseline) Audio emotion-based highlight extraction model** 
  - run [final_audio.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/lstm/final_audio.ipynb)
  - Load model
    1) _lstm_emotion_classification_model.pt_
    2) _lstm_arousal_model_best.pt_
    3) _lstm_valence_model.pt_
    4) _concatenate_lstm_model.pt_
  - Measure model performance
    - **_concatenate_lstm_model.pt_** Measure the performance of the baseline model for highlight extraction based on audio emotion using the model.
####
 - **2) (Baseline) Video-based highlight extraction model**
 - run [final_vit_video.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/vit/final_vit_video.ipynb)
 - Measure model performance
    - Measure the performance of an image-based highlight extraction baseline model using **a pre-trained ViT model**
####
- **3) Multimodal highlight extraction model using both video and audio features**
  - run [final_audio+video_best_undersampling.ipynb](https://github.com/HwnagYujeong0808/EmotionShortForm-etri_multimodal/blob/main/lstm/final_audio+video_best_undersampling.ipynb)
  - Load model
    1) _lstm_emotion_classification_model.pt_
    2) _lstm_arousal_model_best.pt_
    3) _lstm_valence_model.pt_
    4) _concatenate_lstm_model_0.001.pt_
    5) _multimodal_model.pt_
  - Measure model performance
    - Using the **_multimodal_model.pt_** model, measure the performance of a multimodal model that utilizes both voice emotion and video features proposed in the project

## Results

### Performance comparison of LSTM/Wav2Vec based audio emotion classification model and voice arousal prediction model
<p  align="center"><img src="audio_result_1.png" width="500px" ></p>

<p  align="center"><img src="audio_result_2.png" width="500px" ></p>

### Highlight extraction model performance comparison
+ **Threshold**
   + Multimodal highlight extraction model threshold: 0.3
   + Audio-based highlight extraction model threshold: 0.3
   + Video-based highlight extraction model threshold: 0.3


+ **Final result**
<p  align="center"><img src="result_plot.png" width="500px" ></p>
<p  align="center"><img src="result_final.png" width="500px" ></p>

+ **Multimodal LSTM Model**:  Multimodal highlight extraction model using both video and audio features
+ (Baseline) Audio LSTM Model: Audio emotion-based highlight extraction model
+ (Baseline) ViT Model: Video-based highlight extraction model

