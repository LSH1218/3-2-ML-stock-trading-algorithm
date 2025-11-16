# 📚 사용 가이드

## 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/LSH1218/stock-trading-algorithm.git
cd stock-trading-algorithm

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 기본 실행

```bash
python stock_predictor.py
```

### 3. 예시 실행 결과

#### 테슬라 (TSLA) 백테스팅
```
주식 코드를 입력하세요: TSLA

백테스팅 중...
==============================================================
백테스팅 결과
==============================================================
최종 자산 (달러): $12,567.89
수익률: 25.68%
==============================================================
```

## 코드 구조 설명

### 주요 클래스: `StockPredictor`

```python
predictor = StockPredictor('TSLA')  # 인스턴스 생성
```

#### 1. 데이터 수집
```python
predictor.download_data()  # Yahoo Finance에서 자동 다운로드
```

#### 2. 기술적 분석
```python
predictor.calculate_moving_averages()  # 이동평균선 계산
predictor.generate_signals()           # 매매 신호 생성
```

#### 3. 시각화
```python
predictor.plot_moving_averages()  # 차트 출력
```

#### 4. 백테스팅
```python
final_won, final_usd = predictor.trading_strategy()
```

## 커스터마이징

### 이동평균선 기간 변경

```python
predictor = StockPredictor('AAPL')
predictor.short_window = 20   # 단기: 20일
predictor.long_window = 100   # 장기: 100일
```

### 다른 기간 데이터 사용

`stock_predictor.py` 파일에서 `download_data()` 메서드 수정:

```python
def download_data(self):
    df = yf.download(
        self.stock_symbol, 
        start="2015-01-01",  # 시작일 변경
        end="2023-12-31"     # 종료일 변경
    )
    return df
```

### 초기 자본 변경

`trading_strategy()` 메서드에서:

```python
capital_won = 50000000  # 5,000만원으로 변경
capital_usd = 50000     # $50,000으로 변경
```

## 고급 사용법

### 1. 머신러닝 모델 활성화

메인 함수에서 주석 해제:

```python
# 데이터 준비
x_data, y_data = predictor.prepare_training_data()

# 모델 학습 (주석 해제)
predictor.train_linear_regression(x_data, y_data)
```

### 2. 여러 종목 비교 분석

```python
symbols = ['TSLA', 'AAPL', 'NVDA', 'AMD']
results = {}

for symbol in symbols:
    predictor = StockPredictor(symbol)
    predictor.calculate_moving_averages()
    predictor.generate_signals()
    predictor.preprocess_data()
    
    final_won, final_usd = predictor.trading_strategy()
    results[symbol] = final_usd

# 결과 출력
for symbol, final in results.items():
    returns = (final - 10000) / 10000 * 100
    print(f"{symbol}: ${final:.2f} ({returns:.2f}%)")
```

### 3. Jupyter Notebook에서 사용

```python
from stock_predictor import StockPredictor
import matplotlib.pyplot as plt

# 인스턴스 생성
predictor = StockPredictor('BTC-USD')

# 분석 실행
predictor.calculate_moving_averages()
predictor.generate_signals()

# 인라인 차트 표시
%matplotlib inline
predictor.plot_moving_averages()

# 결과 확인
predictor.df.tail()
```

## 트러블슈팅

### 문제 1: yfinance 오류
```
KeyError: 'Close'
```

**해결**: 주식 심볼이 올바른지 확인하세요. Yahoo Finance에서 지원하는 심볼인지 확인.

```python
# 올바른 예시
'TSLA'       # 테슬라 (O)
'005930.KS'  # 삼성전자 (O)

# 잘못된 예시
'삼성전자'    # 한글 이름 (X)
'TSLA.US'    # 잘못된 접미사 (X)
```

### 문제 2: TensorFlow 경고
```
WARNING:tensorflow:...
```

**해결**: 경고는 무시해도 됩니다. 에러가 아니라면 정상 작동합니다.

### 문제 3: 데이터가 충분하지 않음
```
ValueError: window is larger than array
```

**해결**: 최근에 상장한 종목이거나 데이터가 부족한 경우입니다. `short_window`와 `long_window`를 줄여보세요.

## 성능 최적화

### 1. 데이터 캐싱

매번 다운로드하지 않고 로컬에 저장:

```python
import pickle

# 저장
with open('stock_data.pkl', 'wb') as f:
    pickle.dump(predictor.df, f)

# 불러오기
with open('stock_data.pkl', 'rb') as f:
    predictor.df = pickle.load(f)
```

### 2. 병렬 처리

여러 종목 동시 분석:

```python
from concurrent.futures import ThreadPoolExecutor

def analyze_stock(symbol):
    predictor = StockPredictor(symbol)
    predictor.calculate_moving_averages()
    predictor.generate_signals()
    return predictor.trading_strategy()

symbols = ['TSLA', 'AAPL', 'NVDA']
with ThreadPoolExecutor(max_workers=3) as executor:
    results = executor.map(analyze_stock, symbols)
```

## 다음 단계

1. **전략 개선**: RSI, MACD 등 다른 기술적 지표 추가
2. **실시간 트레이딩**: Binance/Upbit API 연동
3. **딥러닝 적용**: LSTM으로 시계열 예측 고도화
4. **리스크 관리**: 손절/익절 자동화, 포지션 크기 조절

## 참고 자료

- [yfinance 문서](https://pypi.org/project/yfinance/)
- [TensorFlow 튜토리얼](https://www.tensorflow.org/tutorials)
- [Technical Analysis Library](https://technical-analysis-library-in-python.readthedocs.io/)
