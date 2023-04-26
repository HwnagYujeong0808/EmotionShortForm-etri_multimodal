# ETRI Multimodal Contest - EmotionShortForm Code
+ 이 저장소는 한국전자통신연구원(ETRI)이 주최하고 과학기술정보통신부와 국가과학기술연구회(NST)가 후원하눈 **"제2회 ETRI 휴먼이해 인공지능 논문경진대회"** 에 제출할 소스코드를 담고 있습니다.
+ 팀명: EmotionSF

## 프로젝트 소개
+ 본 연구에서는 *음성에서 추출한 감정 정보*와 *영상에서 추출한 이미지*를 동시에 활용하는 **멀티모달 모델**을 제안하고, 이를 활용한 유튜브 하이라이트 자동 추출 모델을 제안한다. 제안하는 핵심 아이디어는 **Vision Transformer(ViT)** 모델로부터 추출한 영상 프레임 특징과 **Long Short-Term Memory(LSTM)** 기반 감정 분류 및 예측 모델을 통해 추출한 감정 특징을 함께 사용하는 멀티모달 모델을 구성한 것이다. 제안하는 방법의 효용성을 보이기 위하여 제안하는 모델과 음성 또는 영상만을 활용했을 때의 모델 성능을 비교한다. 실험 결과, 영상 특징 기반 ViT 모델 및 음성 감정 특징 기반 LSTM 모델보다, 제안한 멀티모달 모델의 **F1 Score**가 각각 약 **7.23%, 16.60%** 정도 향상됨을 보였다.

## 데이터셋 
+ **데이터 1: ETRI 한국어 감정 데이터셋 KEMDy20 (일반인 대상 자유발화) 데이터셋**
  + **링크**: https://nanum.etri.re.kr/share/kjnoh/KEMDy20?lang=ko_KR
  + **소개**: 발화 음성, 발화의 문맥적 의미(lexical) 및 생리반응 신호- 피부전도도(EDA-electrodermal activity), 맥박관련 데이터(IBI-Inter-Beat-Interval), 손목 피부온도와 발화자의 감정과의 연관성 분석을 위해 수집한 멀티모달 감정 데이터셋


+ **데이터 2: AI HUB 동영상 콘텐츠 하이라이트 편집 및 설명(요약) 데이터**
    + **링크**: https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=616
    + **소개**: AI HUB에서 수집한 동영상 콘텐츠 하이라이트 편집 및 설명(요약) 데이터셋은 뉴스 및 유튜브 영상에서 주요 장면의 위치를 레이블링하고 카테고리 항목에 대해 태깅하여 구축한 학습용 데이터셋




## 실행 방법

### 데이터셋 생성

### 모델 학습

### 모델 추론


## 프로젝트 구조
![image](https://user-images.githubusercontent.com/66208800/234458258-45c80130-3fe2-4979-9a5c-073a1f428bba.png)




