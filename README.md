# 📈 Stock Trading Algorithm

> 이동평균선 교차(Golden Cross/Dead Cross)를 활용한 자동 매매 알고리즘 및 머신러닝 기반 주가 예측 시스템

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📌 프로젝트 개요

이 프로젝트는 **2023년 2학기 머신러닝 과목**에서 진행한 주식 투자 알고리즘 개발 프로젝트입니다. 
실시간 주식 데이터를 수집하여 기술적 지표를 계산하고, 골든크로스/데드크로스 패턴을 매매 시그널로 활용하는 자동 트레이딩 시스템을 구현했습니다.

### 🎯 핵심 기능

- **실시간 데이터 수집**: Yahoo Finance API를 통한 글로벌 주식/암호화폐 데이터 수집
- **기술적 분석**: 단기(50일)/장기(200일) 이동평균선 계산 및 교차점 탐지
- **자동 매매 로직**: 
  - 골든크로스 발생 시 → 매수
  - 단기 이평선 하락 시 → 50% 매도 (부분 익절)
  - 데드크로스 발생 시 → 전량 매도 (손절)
- **머신러닝 예측**: TensorFlow 기반 다변량 선형회귀 모델로 가격 예측
- **백테스팅**: 과거 데이터로 전략 검증 및 수익률 계산
- **시각화**: Matplotlib을 활용한 주가 차트, 매매 신호, 학습 곡선 시각화

## 🛠️ 기술 스택

### Core Technologies
- **Python 3.7+**: 메인 개발 언어
- **TensorFlow 2.x**: 딥러닝 모델 구현 및 학습
- **pandas**: 시계열 데이터 처리 및 분석
- **NumPy**: 수치 연산 및 배열 처리

### Data & Visualization
- **yfinance**: 실시간 금융 데이터 API
- **Matplotlib**: 데이터 시각화
- **scikit-learn**: 데이터 전처리 및 평가 지표

## 📊 백테스팅 결과

### 테스트 개요
- **백테스팅 기간**: 2020-01-01 ~ 2023-12-05 (약 4년)
- **전략**: 골든크로스 매수, 데드크로스 매도, 단기MA 하락시 50% 익절
- **초기 자본**: 1,000만원 (KRW) / $10,000 (USD)
- **테스트 종목**: 11개 (주식, 암호화폐, 금 선물)

### 🏆 성과 요약
- **평균 수익률**: +44.93%
- **수익 종목**: 7/11 (63.6%)
- **손실 종목**: 4/11 (36.4%)
- **최고 수익**: 엔비디아 +229.79%

### TOP 3 수익률

| 순위 | 종목 | 수익률 | 최종 자산 | 매매 횟수 |
|------|------|--------|-----------|----------|
| 🥇 | **엔비디아 (NVDA)** | **+229.79%** | $32,978.83 | 9회 |
| 🥈 | **이더리움 (ETH-USD)** | **+187.00%** | $28,699.80 | 8회 |
| 🥉 | **AMD** | **+69.89%** | $16,988.97 | 10회 |

<details>
<summary>📈 전체 종목 상세 결과 보기 (클릭)</summary>

### 전체 11개 종목 백테스팅 결과

| 종목 | 초기자산 | 최종자산 | 수익률 | 매매횟수 | 분석 |
|------|----------|----------|--------|----------|------|
| 엔비디아 (NVDA) | $10,000 | $32,978.83 | **+229.79%** | 9회 | 📈 성장주 + 명확한 상승 추세 |
| 이더리움 (ETH-USD) | $10,000 | $28,699.80 | **+187.00%** | 8회 | 📈 암호화폐 + 높은 변동성 활용 |
| AMD | $10,000 | $16,988.97 | **+69.89%** | 10회 | 📈 반도체 성장 + 안정적 추세 |
| 삼성전기 (009150.KS) | ₩10,000,000 | ₩16,618,582 | **+66.19%** | 10회 | 📈 중간 변동성 + 꾸준한 성장 |
| 금 선물 (GC=F) | $10,000 | $13,293.28 | **+32.93%** | 5회 | 📈 안전자산 + 안정적 수익 |
| 애플 (AAPL) | $10,000 | $10,811.83 | **+8.12%** | 9회 | 📊 대형주 + 완만한 상승 |
| 네이버 (035420.KS) | ₩10,000,000 | ₩10,513,373 | **+5.13%** | 12회 | 📊 횡보장 + 제한적 수익 |
| SM엔터 (041510.KS) | ₩10,000,000 | ₩8,522,108 | **-14.78%** | 12회 | 📉 높은 변동성 + 잦은 손절 |
| 테슬라 (TSLA) | $10,000 | $8,514.43 | **-14.86%** | 9회 | 📉 급등락 반복 + 타이밍 실패 |
| 삼성전자 (005930.KS) | ₩10,000,000 | ₩7,932,892 | **-20.67%** | 11회 | 📉 낮은 변동성 + 횡보장 |
| 비트코인 (BTC-USD) | $10,000 | $4,544.38 | **-54.56%** | 10회 | 📉 과도한 변동성 + 빈번한 손절 |

### 종목별 특성 분석

#### ✅ 성공 요인 (상위 5개 종목)
1. **명확한 상승 추세**: 엔비디아, 이더리움, AMD
   - 장기적으로 우상향하는 차트
   - 골든크로스 신호 후 지속적인 상승
   
2. **적절한 변동성**: 삼성전기, 금 선물
   - 너무 높지도, 낮지도 않은 변동성
   - 잘못된 신호 최소화

3. **섹터 성장성**: 반도체(NVDA, AMD), 암호화폐(ETH)
   - 2020-2023년 성장 산업
   - 산업 전체의 성장세와 동반 상승

#### ❌ 실패 요인 (하위 4개 종목)
1. **과도한 변동성**: 비트코인 (-54.56%)
   - 급등 후 급락 반복
   - 잘못된 골든크로스/데드크로스 신호 빈발
   - 손절 반복으로 자본 잠식

2. **낮은 변동성 + 횡보**: 삼성전자 (-20.67%)
   - 200일 이동평균선 주변에서 횡보
   - 매매 기회 부족
   - 횡보장에서 비효율적

3. **급등락 패턴**: 테슬라 (-14.86%), SM엔터 (-14.78%)
   - 예측 불가능한 급등락
   - 타이밍 포착 실패
   - 변동성 대비 수익 미흡

</details>

### 💡 핵심 인사이트

**✅ 전략이 효과적인 경우:**
- 명확한 상승 추세를 가진 자산 (엔비디아, 이더리움, AMD)
- 적절한 수준의 변동성 (급등락 없이 꾸준한 성장)
- 장기 성장성이 있는 자산 (성장 산업 섹터)

**❌ 전략이 비효율적인 경우:**
- 변동성이 너무 높은 자산 (비트코인 -54.56%)
  - 잦은 골든크로스/데드크로스 신호로 손절 반복
- 변동성이 너무 낮은 자산 (삼성전자 -20.67%)
  - 횡보장에서 매매 기회 부족
- 급등락이 심한 자산 (테슬라 -14.86%)
  - 타이밍 실패로 인한 손실

**📚 프로젝트를 통해 배운 점:**
1. **모든 자산에 동일한 전략 적용 불가**: 자산별 특성에 맞는 파라미터 조정 필요
2. **후행 지표의 한계**: 골든크로스/데드크로스는 추세가 시작된 후 진입하므로 초기 수익 놓침
3. **리스크 관리의 중요성**: 변동성 필터, 손절선 설정 등 추가 안전장치 필요
4. **다중 지표 필요성**: RSI, MACD 등 다른 지표와 조합 시 성과 향상 가능
5. **섹터 선택의 중요성**: 성장 산업(반도체, 암호화폐)에서 더 높은 성과

## 📊 알고리즘 상세

### 1. 이동평균선 전략 (Moving Average Crossover)

```
단기 이동평균 (50일) > 장기 이동평균 (200일) → Golden Cross → 매수 신호
단기 이동평균 (50일) < 장기 이동평균 (200일) → Dead Cross → 매도 신호
```

**장점**: 
- 명확한 진입/청산 시점 제공
- 노이즈 필터링으로 안정적인 신호 생성
- 중장기 추세 파악에 효과적

**한계**:
- 후행성 지표로 늦은 진입
- 횡보장에서 빈번한 잘못된 신호
- 초기 수익 기회 상실

<details>
<summary>🔍 상세 로직 보기 (클릭)</summary>

### 매매 신호 생성 로직

```python
# 골든크로스: 단기MA가 장기MA를 상향 돌파
Golden_Cross = (Short_MA > Long_MA) AND (Short_MA[이전] <= Long_MA[이전])

# 데드크로스: 단기MA가 장기MA를 하향 돌파
Dead_Cross = (Short_MA < Long_MA) AND (Short_MA[이전] >= Long_MA[이전])

# 부분 익절: 단기MA 하락 시
Partial_Sell = (Short_MA < Short_MA[이전]) AND (보유 중)
```

### 포지션 관리

| 상황 | 조건 | 액션 | 보유 비율 |
|------|------|------|-----------|
| 진입 | 골든크로스 발생 | 전량 매수 | 0% → 100% |
| 부분 익절 | 단기MA 하락 | 50% 매도 | 100% → 50% |
| 손절/전량 청산 | 데드크로스 발생 | 전량 매도 | 50% → 0% |

</details>

### 2. 머신러닝 가격 예측

**입력 특성(Features)**:
- Open (시가)
- High (고가)
- Low (저가)
- Close (종가)

**모델**: 
- 다변량 선형회귀 (Multivariate Linear Regression)
- Gradient Descent 최적화
- Cost Function: MSE (Mean Squared Error)

**학습 설정**:
```python
Epochs: 20,000
Learning Rate: 0.000000999
Optimizer: Custom Gradient Descent (TensorFlow)
```

<details>
<summary>🧮 모델 상세 정보 보기 (클릭)</summary>

### 선형회귀 모델 구조

```
입력층: 4개 특성 (Open, High, Low, Close)
    ↓
가중치 행렬 W: [4 x 1]
편향 b: [1]
    ↓
예측값 = W₁×Open + W₂×High + W₃×Low + W₄×Close + b
    ↓
손실 함수: MSE = mean((예측값 - 실제값)²)
    ↓
Gradient Descent로 W, b 업데이트
```

### 학습 과정

- **반복 횟수**: 20,000 epochs
- **학습률**: 0.000000999 (매우 작은 값으로 안정적 학습)
- **배치 크기**: 전체 데이터 (Full Batch)
- **최적화**: 커스텀 Gradient Descent

### 성능 모니터링

1000번 반복마다 다음 정보 출력:
- 현재 epoch 번호
- 가중치 W의 값
- 편향 b의 값
- 현재 손실(Cost) 값

</details>

### 3. 리스크 관리 전략

| 상황 | 액션 | 보유 비율 | 이유 |
|------|------|-----------|------|
| 골든크로스 | 전량 매수 | 0% → 100% | 강한 상승 신호 |
| 단기 이평선 하락 | 50% 매도 | 100% → 50% | 추세 약화 징후, 부분 익절 |
| 데드크로스 | 전량 매도 | 50% → 0% | 강한 하락 신호, 손절 |

## 🚀 사용 방법

### 설치

```bash
# 저장소 클론
git clone https://github.com/LSH1218/stock-trading-algorithm.git
cd stock-trading-algorithm

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 실행

```bash
python stock_predictor.py
```

### 지원 자산

<details>
<summary>📋 지원 종목 전체 목록 보기 (클릭)</summary>

**한국 주식**:
- 삼성전자: `005930.KS`
- 네이버: `035420.KS`
- 삼성전기: `009150.KS`
- SM엔터테인먼트: `041510.KS`
- 카카오: `035720.KS`
- LG전자: `066570.KS`

**미국 주식**:
- 테슬라: `TSLA`
- 애플: `AAPL`
- 엔비디아: `NVDA`
- AMD: `AMD`
- 마이크로소프트: `MSFT`
- 구글: `GOOGL`

**암호화폐**:
- 비트코인: `BTC-USD`
- 이더리움: `ETH-USD`
- 리플: `XRP-USD`

**상품**:
- 금 선물: `GC=F`
- 은 선물: `SI=F`
- 원유 선물: `CL=F`

</details>

### 실행 예시

```bash
yfinance에서 가져올 주식 코드를 입력하세요: NVDA

# 출력 예시:
# ========================================
# Stock Trading Algorithm - Backtesting System
# ========================================
# 
# [차트 표시]
# - 주가 차트 + 50일/200일 이동평균선
# - 골든크로스 포인트 (빨간 원)
# - 데드크로스 포인트 (파란 원)
#
# 백테스팅 중...
# ========================================
# 백테스팅 결과
# ========================================
# 최종 자산 (달러): $32,978.83
# 수익률: +229.79%
# ========================================
```

## 📈 프로젝트 구조

```
stock-trading-algorithm/
│
├── stock_predictor.py          # 메인 알고리즘 (정리된 버전)
├── stock_predictor_original.ipynb  # 원본 Jupyter 노트북
├── README.md                   # 프로젝트 문서 (이 파일)
├── requirements.txt            # 패키지 의존성
├── LICENSE                     # MIT 라이선스
├── .gitignore                  # Git 제외 파일
└── GUIDE.md                    # 상세 사용 가이드
```

## 🔍 핵심 구현 내용

### StockPredictor 클래스

<details>
<summary>📝 클래스 구조 상세 보기 (클릭)</summary>

```python
class StockPredictor:
    """
    주식 가격 예측 및 자동 매매 전략 클래스
    """
    
    def __init__(self, stock_symbol):
        """
        초기화 및 데이터 다운로드
        
        Args:
            stock_symbol (str): 주식 심볼 (예: 'TSLA', '005930.KS')
        """
        self.stock_symbol = stock_symbol
        self.df = self.download_data()
        self.short_window = 50   # 단기 이동평균
        self.long_window = 200   # 장기 이동평균
        
    def download_data(self):
        """Yahoo Finance에서 주식 데이터 다운로드"""
        pass
        
    def calculate_moving_averages(self):
        """50일/200일 이동평균선 계산"""
        pass
        
    def generate_signals(self):
        """골든크로스/데드크로스 신호 생성"""
        pass
        
    def train_linear_regression(self, x_data, y_data):
        """
        TensorFlow 기반 선형회귀 모델 학습
        
        - 20,000 epochs 학습
        - Gradient Descent 최적화
        - MSE 손실 함수
        """
        pass
        
    def trading_strategy(self):
        """
        백테스팅: 과거 데이터로 전략 검증
        
        Returns:
            tuple: (최종자산_원화, 최종자산_달러)
        """
        pass
        
    def plot_moving_averages(self):
        """주가 차트 및 매매 신호 시각화"""
        pass
```

### 주요 메서드 설명

| 메서드 | 기능 | 입력 | 출력 |
|--------|------|------|------|
| `download_data()` | 데이터 수집 | stock_symbol | DataFrame |
| `calculate_moving_averages()` | 이동평균 계산 | - | DataFrame 업데이트 |
| `generate_signals()` | 매매 신호 생성 | - | DataFrame 업데이트 |
| `trading_strategy()` | 백테스팅 실행 | - | (원화자산, 달러자산) |
| `plot_moving_averages()` | 차트 시각화 | - | matplotlib plot |

</details>

## 💡 프로젝트에서 배운 것

### 기술적 성장
- ✅ 실시간 금융 데이터 처리 및 시계열 분석
- ✅ TensorFlow를 활용한 커스텀 학습 루프 구현
- ✅ Gradient Descent 최적화 알고리즘 직접 구현
- ✅ 데이터 전처리 및 결측치 처리 경험
- ✅ 객체지향 프로그래밍으로 재사용 가능한 코드 작성

### 트레이딩 전략
- ✅ 기술적 지표(이동평균선)의 실전 활용
- ✅ 백테스팅의 중요성 및 과최적화(Overfitting) 방지
- ✅ 리스크 관리: 부분 익절 및 손절 전략
- ✅ 다양한 자산군(주식/암호화폐)에 대한 범용적 적용
- ✅ 자산별 특성 분석 및 전략 적합성 평가

### 문제 해결 및 인사이트
- ✅ **한정된 자원으로 구현**: 무료 API와 오픈소스 라이브러리만으로 완성
- ✅ **실패로부터 학습**: 비트코인 -54% 손실 분석을 통한 전략 한계 이해
- ✅ **다중 화폐 지원**: 원화/달러 자동 감지 및 변환 로직 구현
- ✅ **자산별 맞춤화 필요성**: 모든 전략이 모든 자산에 적용되지 않음을 학습
- ✅ **섹터 중요성**: 성장 산업 선택이 수익률에 큰 영향을 미침

## ⚠️ 제한사항 및 개선 방향

### 현재 제한사항
- ⚠️ 단순 선형회귀 모델 → 비선형 패턴 포착 어려움
- ⚠️ 거래 수수료 미반영 (실제 수익률은 더 낮을 수 있음)
- ⚠️ 슬리피지(Slippage) 고려 안 됨
- ⚠️ 단일 전략만 사용 (다중 전략 포트폴리오 부재)
- ⚠️ 후행 지표로 인한 늦은 진입/청산

### 향후 개선 방향
- [ ] **LSTM/GRU 적용**: 순환 신경망으로 시계열 패턴 학습
- [ ] **다중 지표 조합**: RSI, MACD, Bollinger Bands 추가
- [ ] **변동성 기반 포지션 조절**: 자산별 리스크에 따른 투자 비중 조정
- [ ] **멀티 전략 포트폴리오**: 여러 전략을 조합하여 리스크 분산
- [ ] **실시간 API 연동**: Binance, Upbit 등 거래소 API 연동
- [ ] **거래 비용 반영**: 수수료, 슬리피지, 세금 고려
- [ ] **웹 대시보드**: 실시간 모니터링 및 성과 추적 UI 개발
- [ ] **자동 파라미터 최적화**: Grid Search로 최적의 이동평균 기간 탐색

## 📝 학습 자료

- [Moving Average Crossover Strategy](https://www.investopedia.com/terms/m/movingaverage.asp)
- [TensorFlow Custom Training](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch)
- [Algorithmic Trading Basics](https://www.quantstart.com/articles/beginners-guide-to-quantitative-trading/)
- [Backtesting Trading Strategies](https://www.quantstart.com/articles/backtesting-systematic-trading-strategies-in-python-considerations-and-open-source-frameworks/)
- [Technical Analysis Library](https://technical-analysis-library-in-python.readthedocs.io/)

## 👤 작성자

**이석현 (Lee Seokhyun)**
- 동양미래대학교 로봇공학과
- Email: sukhyun1218@gmail.com
- GitHub: [@LSH1218](https://github.com/LSH1218)

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

**⚡ 주의사항**: 
- 이 알고리즘은 **학습 목적**으로 개발되었습니다.
- 백테스팅 결과는 과거 데이터 기반이며, **실제 투자 결과와 다를 수 있습니다**.
- 실제 투자에 사용하기 전에 충분한 검증이 필요합니다.
- **과거 수익률이 미래 수익을 보장하지 않습니다**.
- 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다.
- 거래 수수료, 슬리피지, 세금 등이 반영되지 않았으므로 실제 수익률은 더 낮을 수 있습니다.
