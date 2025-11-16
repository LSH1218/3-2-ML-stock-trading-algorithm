"""
Stock Trading Algorithm
========================
이동평균선 교차 전략과 머신러닝을 활용한 자동 매매 시스템

Author: 이석현 (Lee Seokhyun)
Date: 2023
Course: Machine Learning (3학년 2학기)
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import locale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class StockPredictor:
    """
    주식 가격 예측 및 자동 매매 전략 클래스
    
    주요 기능:
    - Yahoo Finance에서 실시간 주식 데이터 수집
    - 이동평균선(MA) 계산 및 골든크로스/데드크로스 신호 생성
    - TensorFlow 기반 선형회귀 모델로 가격 예측
    - 백테스팅을 통한 전략 검증
    """
    
    def __init__(self, stock_symbol):
        """
        StockPredictor 초기화
        
        Args:
            stock_symbol (str): 주식 심볼 (예: 'TSLA', '005930.KS')
        """
        self.stock_symbol = stock_symbol
        self.df = self.download_data()
        
        # 이동평균선 기간 설정
        self.short_window = 50   # 단기 이동평균 (골든크로스용)
        self.long_window = 200   # 장기 이동평균 (골든크로스용)
        
        # TensorFlow 변수 초기화 (선형회귀 모델)
        self.W = tf.Variable(1.0)
        self.b = tf.Variable(0.5)
        self.W_grad = None
        self.b_grad = None
        self.learning_rate = 0.000000999  # 학습률

    def download_data(self):
        """
        Yahoo Finance에서 주식 데이터 다운로드
        
        Returns:
            pd.DataFrame: 2020년 1월 1일부터 현재까지의 주가 데이터
        """
        df = yf.download(
            self.stock_symbol, 
            start="2020-01-01", 
            end=pd.to_datetime('today')
        )
        return df

    def calculate_moving_averages(self):
        """
        단기(50일) 및 장기(200일) 이동평균선 계산
        
        이동평균선은 주가의 평균을 계산하여 추세를 파악하는 기술적 지표
        """
        self.df['Short_MA'] = self.df['Close'].rolling(
            window=self.short_window, 
            min_periods=1
        ).mean()
        
        self.df['Long_MA'] = self.df['Close'].rolling(
            window=self.long_window, 
            min_periods=1
        ).mean()

    def generate_signals(self):
        """
        골든크로스 및 데드크로스 매매 신호 생성
        
        - Golden Cross: 단기 MA가 장기 MA를 상향 돌파 → 매수 신호
        - Dead Cross: 단기 MA가 장기 MA를 하향 돌파 → 매도 신호
        """
        # 골든크로스: 이전에는 Short_MA <= Long_MA였다가 현재 Short_MA > Long_MA
        self.df['Golden_Cross_Signal'] = (
            (self.df['Short_MA'] > self.df['Long_MA']) & 
            (self.df['Short_MA'].shift(1) <= self.df['Long_MA'].shift(1))
        )
        
        # 데드크로스: 이전에는 Short_MA >= Long_MA였다가 현재 Short_MA < Long_MA
        self.df['Dead_Cross_Signal'] = (
            (self.df['Short_MA'] < self.df['Long_MA']) & 
            (self.df['Short_MA'].shift(1) >= self.df['Long_MA'].shift(1))
        )

    def plot_moving_averages(self):
        """
        주가 차트와 이동평균선, 매매 신호를 시각화
        
        - 파란색 실선: 종가
        - 주황색 점선: 50일 이동평균선
        - 초록색 점선: 200일 이동평균선
        - 빨간색 원: 골든크로스 (매수 신호)
        - 파란색 원: 데드크로스 (매도 신호)
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.df.index, self.df['Close'], label='Close Price')
        plt.plot(
            self.df.index, 
            self.df['Short_MA'], 
            label=f'{self.short_window} days MA', 
            linestyle='--'
        )
        plt.plot(
            self.df.index, 
            self.df['Long_MA'], 
            label=f'{self.long_window} days MA', 
            linestyle='--'
        )

        # 매매 신호 표시
        plt.scatter(
            self.df.index[self.df['Golden_Cross_Signal']], 
            self.df['Close'][self.df['Golden_Cross_Signal']], 
            marker='o', 
            color='red', 
            label='Golden Cross',
            s=100
        )
        plt.scatter(
            self.df.index[self.df['Dead_Cross_Signal']], 
            self.df['Close'][self.df['Dead_Cross_Signal']], 
            marker='o', 
            color='blue', 
            label='Dead Cross',
            s=100
        )

        plt.title(f'{self.stock_symbol} Stock Price with Moving Averages and Cross Signals')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def preprocess_data(self):
        """
        데이터 전처리: 일일 수익률 계산 및 결측치 제거
        
        일일 수익률 = (당일 종가 - 전일 종가) / 전일 종가
        """
        self.df['Daily_Return'] = self.df['Close'].pct_change()
        self.df = self.df.dropna()

    def prepare_training_data(self):
        """
        머신러닝 모델 학습을 위한 데이터 준비
        
        Returns:
            tuple: (x_data, y_data)
                - x_data: Open, High, Low, Close 특성
                - y_data: Close 가격 (예측 타겟)
        """
        x_data = tf.dtypes.cast(
            self.df[['Open', 'High', 'Low', 'Close']].to_numpy(), 
            tf.float64
        )
        y_data = tf.dtypes.cast(
            self.df['Close'].values, 
            tf.float64
        )
        return x_data, y_data

    def train_linear_regression(self, x_data, y_data):
        """
        TensorFlow를 사용한 다변량 선형회귀 모델 학습
        
        모델: y = W * x + b
        최적화: Gradient Descent
        손실 함수: Mean Squared Error (MSE)
        
        Args:
            x_data: 입력 특성 (Open, High, Low, Close)
            y_data: 타겟 변수 (Close price)
        """
        # 가중치 및 편향 초기화
        self.W = tf.Variable(tf.random.normal([4, 1], dtype=tf.float64))
        self.b = tf.Variable(tf.random.normal([1], dtype=tf.float64))

        cost_history = []  # 학습 과정에서 비용 변화 추적

        # 20,000번 학습 반복
        for i in range(20001):
            # Gradient Tape: 자동 미분을 위한 컨텍스트
            with tf.GradientTape() as tape:
                # 예측값 계산: hypothesis = X @ W + b
                hypothesis = tf.matmul(x_data, self.W) + self.b
                
                # 손실 함수 계산: MSE
                cost = tf.reduce_mean(tf.square(hypothesis - y_data))

            # 그래디언트 계산 및 가중치 업데이트
            self.W_grad, self.b_grad = tape.gradient(cost, [self.W, self.b])
            self.W.assign_sub(self.learning_rate * self.W_grad)
            self.b.assign_sub(self.learning_rate * self.b_grad)

            # 학습 진행 상황 출력
            if i > 500 and i % 1000 == 0:
                print(
                    f"{i:5}|{self.W.numpy()[0][0]:10.4f}|"
                    f"{self.b.numpy()[0]:10.4f}| cost = {cost:10.6f}"
                )

            # 비용 히스토리 저장
            if i % 1000 == 0:
                cost_history.append(cost.numpy())

        # 학습 곡선 시각화
        if cost_history:
            plt.figure(figsize=(12, 6))
            plt.plot(
                range(0, 20001, 1000), 
                cost_history, 
                marker='o', 
                linestyle='-', 
                color='b'
            )
            plt.title('Cost Function Over Iterations')
            plt.xlabel('Iterations')
            plt.ylabel('Cost (MSE)')
            plt.grid(True, alpha=0.3)
            plt.show()

    def predict_stock_price(self, x_pred):
        """
        학습된 모델로 주가 예측
        
        Args:
            x_pred: 예측할 입력 데이터
            
        Returns:
            float: 예측된 주가
        """
        hypothesis = self.W * x_pred + self.b
        predicted_price = hypothesis.numpy()[0]
        return predicted_price

    def trading_strategy(self):
        """
        백테스팅: 과거 데이터로 매매 전략 검증
        
        전략:
        1. 골든크로스 → 전량 매수
        2. 단기 이평선 하락 → 50% 매도
        3. 데드크로스 → 전량 매도
        
        Returns:
            tuple: (최종자산_원화, 최종자산_달러)
        """
        # 초기 자본 설정
        capital_won = 10000000  # 1,000만원
        capital_usd = 10000     # $10,000
        position = 0            # 보유 주식 수

        # 과거 데이터 반복 (백테스팅)
        for i in range(1, len(self.df)):
            current_price = self.df['Close'].iloc[i]
            
            # 1. 골든크로스 시 매수
            if self.df['Golden_Cross_Signal'].iloc[i] and position == 0:
                if 'KS' in self.stock_symbol:  # 한국 주식
                    position = capital_won // current_price
                    capital_won -= position * current_price
                else:  # 해외 주식/암호화폐
                    position = capital_usd // current_price
                    capital_usd -= position * current_price

            # 2. 단기 이평선 하락 시 절반 매도 (부분 익절)
            if (self.df['Short_MA'].iloc[i] < self.df['Short_MA'].iloc[i-1] 
                and position > 0):
                sell_amount = position // 2
                if 'KS' in self.stock_symbol:
                    capital_won += sell_amount * current_price
                else:
                    capital_usd += sell_amount * current_price
                position -= sell_amount

            # 3. 데드크로스 시 전량 매도 (손절)
            if self.df['Dead_Cross_Signal'].iloc[i] and position > 0:
                if 'KS' in self.stock_symbol:
                    capital_won += position * current_price
                else:
                    capital_usd += position * current_price
                position = 0

        # 최종 자산 계산 (현재 보유 주식 포함)
        final_price = self.df['Close'].iloc[-1]
        final_assets_won = capital_won + position * final_price
        final_assets_usd = capital_usd + position * final_price

        return final_assets_won, final_assets_usd


def main():
    """
    메인 실행 함수
    """
    # 로케일 설정 (화폐 단위 표시용)
    try:
        locale.setlocale(locale.LC_ALL, 'ko_KR.UTF8')
    except locale.Error:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

    # 사용자로부터 주식 심볼 입력받기
    print("=" * 60)
    print("Stock Trading Algorithm - Backtesting System")
    print("=" * 60)
    print("\n지원 자산 예시:")
    print("  한국 주식: 005930.KS (삼성전자), 035420.KS (네이버)")
    print("  미국 주식: TSLA (테슬라), AAPL (애플), NVDA (엔비디아)")
    print("  암호화폐: BTC-USD (비트코인), ETH-USD (이더리움)")
    print("  상품: GC=F (금 선물)")
    print("-" * 60)
    
    stock_symbol = input("\n주식 코드를 입력하세요: ")

    # StockPredictor 인스턴스 생성
    predictor = StockPredictor(stock_symbol)
    
    # 이동평균선 계산 및 매매 신호 생성
    predictor.calculate_moving_averages()
    predictor.generate_signals()
    
    # 차트 시각화
    predictor.plot_moving_averages()
    
    # 데이터 전처리
    predictor.preprocess_data()
    x_data, y_data = predictor.prepare_training_data()
    
    # 선형회귀 모델 학습 (옵션)
    # predictor.train_linear_regression(x_data, y_data)

    # 백테스팅 실행
    print("\n백테스팅 중...")
    final_assets_won, final_assets_usd = predictor.trading_strategy()

    # 결과 출력
    print("\n" + "=" * 60)
    print("백테스팅 결과")
    print("=" * 60)
    
    if 'KS' in stock_symbol:
        # 한국 주식
        print(f"최종 자산 (원화): {locale.currency(final_assets_won, grouping=True, symbol='₩')}")
        returns = (final_assets_won - 10000000) / 10000000 * 100
        print(f"수익률: {returns:.2f}%")
    else:
        # 해외 자산
        print(f"최종 자산 (달러): {locale.currency(final_assets_usd, grouping=True)}")
        returns = (final_assets_usd - 10000) / 10000 * 100
        print(f"수익률: {returns:.2f}%")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
