# 머신러닝 대화 정리

좋아. 지금 대화에서 네가 헷갈려 했던 부분들을 **개념 정리**랑 **scikit-learn import 정리**로 나눠서 한 번에 정리해줄게.

---

# 1. 네가 헷갈려 했던 핵심 개념 정리

## 1) \(\theta\) 는 뭐냐
\[
\theta = \text{모델의 파라미터들}
\]

예를 들어 선형회귀나 로지스틱 회귀에서

\[
y = w_0x_0 + w_1x_1 + b
\]

이면 학습해야 하는 값은

- \(w_0\)
- \(w_1\)
- \(b\)

이 3개고, 이걸 한꺼번에 묶어서 \(\theta\)라고 부르는 거야.

즉

\[
\theta = (w_0, w_1, b)
\]

---

## 2) 비용함수(cost function) 의미
비용함수는 **모델이 얼마나 틀렸는지 숫자로 나타내는 함수**야.

- 학습 목표 = 비용함수를 최소화하는 것
- 즉, 예측 오차가 가장 작아지도록 \(w,b\)를 찾는 것

### 선형회귀
보통 **MSE** 사용

\[
J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(\hat y^{(i)}-y^{(i)})^2
\]

### 로지스틱 회귀
보통 **BCE(Binary Cross Entropy)** 사용

\[
J(\theta)=\frac{1}{m}\sum_{i=1}^{m}
\left[
-y^{(i)}\log(\hat y^{(i)})
-(1-y^{(i)})\log(1-\hat y^{(i)})
\right]
\]

---

## 3) 왜 로지스틱 회귀는 MSE를 안 쓰고 BCE를 쓰냐
로지스틱 회귀의 출력 \(\hat y\)는 **실수값 예측**이 아니라 **확률**이야.

\[
\hat y = P(y=1 \mid x)
\]

그래서 선형회귀처럼 MSE를 쓰기보다는  
**확률적으로 가장 그럴듯한 파라미터를 찾는 방식(MLE)** 을 쓰고,  
그 결과로 BCE가 나와.

즉:

- 로지스틱 회귀 출력 = 확률
- MLE 사용
- negative log likelihood = BCE

---

## 4) \(\hat y\)는 예측값이면서 확률이냐
응, 맞아.

로지스틱 회귀에서는

\[
\hat y = P(y=1 \mid x)
\]

즉 \(\hat y\)는

- y의 예측값이고
- 동시에 클래스 1일 확률

이야.

예:
- \(\hat y=0.9\) → 1일 확률 90%
- \(\hat y=0.2\) → 1일 확률 20%

최종 클래스는 보통 0.5 기준으로 정해.

---

## 5) 벡터/배열 shape 관련

### `(2,)`
이건 **1차원 배열**이야.  
길이 2짜리 벡터라고 보면 돼.

### `(2,1)`
이건 **2차원 배열**이고, 2행 1열 **열벡터**

### `(1,2)`
이건 **2차원 배열**이고, 1행 2열 **행벡터**

---

## 6) `[1,2]`는 2차원 배열이냐?
아니야.

```python
np.array([1,2])
```

는 shape가 `(2,)`니까 **1차원 배열**이야.

배열의 차원은 원소 개수가 아니라 **축(axis)의 개수**로 정해져.

- `[1,2,3]` → 1차원
- `[[1,2,3],[4,5,6]]` → 2차원
- `[[[...]]]` → 3차원

---

## 7) 왜 `np.dot(x,w)`를 쓰냐
수학에서는 \(x, w\)를 열벡터로 써서

\[
x^T w
\]

라고 쓰지만, NumPy에서

```python
x = np.array([1,2,3])
w = np.array([4,5,6])
```

는 둘 다 shape `(3,)`인 **1차원 배열**이야.

그래서 NumPy에서는 이 둘의 내적을

```python
np.dot(x, w)
```

또는

```python
x @ w
```

로 계산하는 거야.

---

## 8) `x*w`는 3×3 배열이 되냐
아니야.

```python
x = np.array([1,2,3])
w = np.array([4,5,6])
x * w
```

는 **원소별 곱**이라서 결과는

```python
[4,10,18]
```

이야.

3×3 배열이 되려면 보통 하나는 `(3,1)`, 하나는 `(1,3)`이어야 해.

---

## 9) `X_train`은 6차원 배열이냐
아니야.  
예제의 `X_train`은 shape `(6,2)`인 **2차원 배열**이야.

뜻:
- 샘플 6개
- feature 2개

즉 **6개의 데이터가 있고 각 데이터마다 특징이 2개 있다**는 뜻이야.

---

## 10) 왜 `np.dot(X,w)+b`의 shape가 `(6,)`냐
예를 들어

- `X.shape = (6,2)`
- `w.shape = (2,)`

이면 `X`의 각 행과 `w`를 내적해서 샘플 6개에 대한 결과 6개를 만들기 때문에

```python
np.dot(X,w).shape == (6,)
```

가 돼.

즉 각 샘플마다 하나씩 \(z^{(i)}\)가 나온 거야.

---

## 11) 왜 `dj_dw = X.T @ err / m`의 shape가 `(2,)`냐
- `X.shape = (6,2)`
- `X.T.shape = (2,6)`
- `err.shape = (6,)`

그래서

```python
X.T @ err
```

의 결과는 `(2,)`가 돼.

이건 feature가 2개니까  
gradient도 2개 나온다는 뜻이야.

---

## 12) `stratify=y`는 왜 쓰냐
train/test를 나눌 때 **클래스 비율을 원래 데이터와 비슷하게 유지하기 위해서** 써.

예를 들어 전체 데이터가

- class 0: 70%
- class 1: 30%

이면 `stratify=y`를 써서 train, test도 대략 7:3 비율이 되게 하는 거야.

---

## 13) `classifier.predict` 와 `contourf`
결정경계 그릴 때:

### `classifier.predict(...)`
격자 위의 모든 점에 대해 **클래스 0/1 예측값**을 계산

### `plt.contourf(...)`
그 예측값을 이용해서 **영역을 색으로 채움**

즉

- `predict` = 계산
- `contourf` = 시각화

---

## 14) `.score()`, `.predict_proba()`, `.predict()`

### `.score(X,y)`
정확도 반환

### `.predict_proba(X)`
각 클래스에 속할 확률 반환

예:
```python
[0.06, 0.94]
```
→ class 0 확률 0.06, class 1 확률 0.94

### `.predict(X)`
최종 클래스 라벨 반환

예:
```python
1
```

---

## 15) OvR과 softmax 차이

### OvR (One vs Rest)
클래스가 3개면 분류기 3개를 따로 만듦.

- A vs Rest
- B vs Rest
- C vs Rest

즉 **이진 분류기 여러 개**로 다중분류를 해결

### Softmax multinomial logistic regression
분류기 하나가 클래스 전체 확률을 **한 번에** 출력

\[
\hat y_k = \frac{e^{z_k}}{\sum_i e^{z_i}}
\]

즉 **하나의 모델이 모든 클래스를 직접 다룸**

---

## 16) multiclass cross entropy 식에서 합이 두 번 보이는 이유
\[
J(\theta)=\sum_i J^{(i)}(\theta)
\]
\[
J^{(i)}(\theta)=-\sum_k y_k \log \hat y_k
\]

여기서

- \(\sum_k\): 한 샘플 안에서 클래스들에 대한 합
- \(\sum_i\): 전체 샘플들에 대한 합

즉 **샘플별 multiclass cross entropy를 다 더한 것이 전체 비용함수**야.

---

## 17) 정규화(regularization)란
비용함수에 penalty를 추가해서 **overfitting을 방지**하는 거야.

---

## 18) L1 vs L2

### L2 regularization (ridge)
\[
\lambda \sum_j w_j^2
\]

- weight를 전반적으로 작게 만듦
- 보통 0으로는 잘 안 감

### L1 regularization (lasso)
\[
\lambda \sum_j |w_j|
\]

- 일부 weight를 아예 0으로 만듦
- feature selection 효과 있음

---

## 19) 왜 정규화가 강하면 weight가 작아지냐
정규화는 cost function에

\[
\lambda \sum_j w_j^2
\quad \text{또는} \quad
\lambda \sum_j |w_j|
\]

같은 벌점을 추가하는 거야.

즉 **큰 weight를 쓰면 cost가 커지기 때문에**  
모델은 weight를 작게 하려는 방향으로 학습돼.

그래서

- weight 작아짐
- 함수가 덜 가파름
- 입력 변화에 덜 민감함
- 더 단순한 모델이 됨

---

## 20) C와 \(\lambda\) 관계
scikit-learn의 `LogisticRegression`에서는 \(\lambda\) 대신 **C**를 써.

\[
C = \frac{1}{\lambda}
\]

즉

- **C 작음** → \(\lambda\) 큼 → 정규화 강함 → weight 작아짐
- **C 큼** → \(\lambda\) 작음 → 정규화 약함 → weight 커질 수 있음

---

## 21) C가 작을수록 무조건 좋은 거냐
아니야.

- C 너무 큼 → overfitting 위험
- C 너무 작음 → underfitting 위험

즉 무조건 작은 게 좋은 게 아니라  
**적절한 C를 찾아야 해.**

---

## 22) 왜 `fit(X_train_std, y_train)` 을 쓰냐
`X_train_std`는 표준화된 데이터야.

정규화가 있는 모델은 feature 스케일에 영향을 받기 때문에  
표준화해서 학습하는 게 더 안정적이야.

즉:

- `X_train` = 원본
- `X_train_std` = 표준화된 train 데이터

그래서 `fit`에 `X_train_std`를 넣는 거야.

---

## 23) “data를 한번에 transform한다”는 말
이건 보통 **샘플 하나씩 처리하지 않고, 전체 데이터 행렬을 한 번에 변환한다**는 뜻이야.

예를 들어 표준화에서:

```python
X_train_std = sc.transform(X_train)
```

이렇게 하면 `X_train` 전체가 한 번에 바뀌는 거야.

---

# 2. sklearn import / 클래스 / 함수 정리

아래는 지금 대화에서 나온 것들 위주로 정리한 거야.

---

## 1) 데이터셋 불러오기
```python
from sklearn import datasets
```

사용 예:
```python
iris = datasets.load_iris()
```

설명:
- `load_iris()`는 `sklearn.datasets` 안에 있음
- 그래서 `datasets.load_iris()` 가능

---

## 2) train/test 분할
```python
from sklearn.model_selection import train_test_split
```

사용 예:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)
```

설명:
- 데이터를 train/test로 나누는 함수

---

## 3) 로지스틱 회귀 모델
```python
from sklearn.linear_model import LogisticRegression
```

사용 예:
```python
lr = LogisticRegression(C=100.0)
lr.fit(X_train, y_train)
```

설명:
- `LogisticRegression`은 `sklearn.linear_model`에 있음
- `from sklearn import datasets`만으로는 바로 못 씀
- 반드시 따로 import해야 함

---

## 4) 표준화
대화에서 직접 import 문을 길게 다루진 않았지만, 표준화 문맥상 보통 이걸 씀:

```python
from sklearn.preprocessing import StandardScaler
```

사용 예:
```python
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
```

또는

```python
X_train_std = sc.fit_transform(X_train)
```

설명:
- `fit()` : 평균/표준편차 학습
- `transform()` : 실제 변환
- `fit_transform()` : 둘을 한 번에

---

# 3. 지금 대화 기준으로 자주 쓰는 코드 세트

## 1) binary logistic regression 기본
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# 이진 분류로 쓸 경우 예시
X = X[0:100]
y = y[0:100]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(X_train, y_train)
```

---

## 2) 표준화 포함
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

lr = LogisticRegression(C=1.0, random_state=1)
lr.fit(X_train_std, y_train)
```

---

## 3) multiclass OvR
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=100.0, random_state=1, multi_class='ovr')
```

---

# 4. 아주 짧은 최종 요약

## 네가 가장 자주 헷갈렸던 포인트
- \(\hat y\)는 로지스틱 회귀에서 **확률**
- `(2,)`는 **1차원 배열**
- `x*w`는 **원소별 곱**, `np.dot(x,w)`는 **내적**
- `X_train`은 `(샘플 수, feature 수)` 모양의 **2차원 배열**
- `stratify=y`는 **클래스 비율 유지**
- OvR와 softmax는 **다른 다중분류 방식**
- 정규화는 **큰 weight에 벌점**을 줘서 overfitting 방지
- scikit-learn에서는 \(\lambda\) 대신 **C**를 쓰고, **작은 C = 강한 정규화**

## import 핵심 3개
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
```

표준화까지 하면:
```python
from sklearn.preprocessing import StandardScaler
```
