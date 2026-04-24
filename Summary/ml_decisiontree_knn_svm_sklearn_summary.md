# 머신러닝 정리: Decision Tree, Random Forest, K-NN, SVM, Kernel Trick, sklearn import

이 문서는 이번 대화에서 헷갈렸던 부분과 관련 sklearn class/import를 한 번에 정리한 파일입니다.

---

# 1. Decision Tree와 Gini / Entropy / Information Gain

## 1.1 Gini Index란?

Gini는 노드 안의 데이터가 얼마나 섞여 있는지 나타내는 **불순도(impurity)** 입니다.

- Gini가 낮다 → 한 클래스가 대부분이다 → 순수하다
- Gini가 높다 → 여러 클래스가 섞여 있다 → 불순하다

이진 분류에서 Gini는 다음과 같습니다.

```text
Gini = 1 - (p1^2 + p2^2)
```

예시:

```text
H, H, L, L
p(H)=1/2, p(L)=1/2
Gini = 1 - ((1/2)^2 + (1/2)^2)
      = 1 - (1/4 + 1/4)
      = 0.5
```

```text
H, H, H, H
p(H)=1, p(L)=0
Gini = 1 - (1^2 + 0^2)
      = 0
```

즉, **완전히 한 클래스만 있으면 Gini = 0** 입니다.

---

## 1.2 Entropy와 Gini의 최대값 차이

이진 분류에서 가장 불순한 경우는 클래스가 반반 섞였을 때입니다.

```text
p = 0.5
```

### Gini

```text
Gini = 1 - (0.5^2 + 0.5^2)
     = 0.5
```

그래서 이진 분류에서 Gini의 최대값은 0.5입니다.

### Entropy

```text
Entropy = -p log2(p) - (1-p) log2(1-p)
```

p = 0.5일 때:

```text
Entropy = -0.5 log2(0.5) - 0.5 log2(0.5)
        = 1
```

그래서 이진 분류에서 Entropy의 최대값은 1입니다.

핵심:

```text
둘 다 p=0.5에서 최대가 된다.
다만 계산식이 달라서 Entropy는 최대 1, Gini는 최대 0.5가 된다.
```

---

## 1.3 Information Gain이란?

Information Gain, IG는 **나누기 전보다 나눈 후 불순도가 얼마나 줄었는지**를 나타냅니다.

```text
IG = 부모 노드의 불순도 - 자식 노드들의 가중 평균 불순도
```

수식:

```text
IG(Dp, f) = I(Dp) - Σ (Nj / Np) I(Dj)
```

의미:

```text
Dp: 부모 노드
Dj: j번째 자식 노드
Np: 부모 노드의 데이터 개수
Nj: j번째 자식 노드의 데이터 개수
I(D): 불순도 함수
```

여기서 `I(D)` 자리에 Gini를 넣으면 Gini 기준이고, Entropy를 넣으면 Entropy 기준입니다.

즉:

```text
I(D) = Gini     → Gini 기준으로 split
I(D) = Entropy  → Entropy 기준으로 split
```

중요한 오해 정리:

```text
Gini가 낮을수록 좋다.
IG가 클수록 좋다.

IG가 크다는 것은 “불순도가 높아서 나눠야 한다”가 아니라,
“이 feature로 나누면 불순도가 많이 줄어든다”는 뜻이다.
```

---

## 1.4 연속형 feature의 split point

연속형 feature는 다음처럼 기준값을 만들어 이진 분할합니다.

예:

```text
x1 = [1, 2, 4, 6.5]
```

먼저 정렬합니다.

```text
1, 2, 4, 6.5
```

인접한 값들의 중간값을 split point 후보로 만듭니다.

```text
1과 2의 중간값 = 1.5
2와 4의 중간값 = 3
4와 6.5의 중간값 = 5.25
```

가능한 split:

```text
x1 <= 1.5
x1 <= 3
x1 <= 5.25
```

각 후보마다 IG를 계산하고, IG가 가장 큰 기준점을 선택합니다.

핵심:

```text
연속형 feature는 정렬 → 인접값 중간점 후보 생성 → 각 후보의 IG 계산 → IG 최대 기준점 선택
```

---

# 2. Decision Tree Pruning

## 2.1 왜 pruning이 필요한가?

Decision Tree는 가지를 너무 많이 만들면 training data를 너무 잘 외울 수 있습니다.

이것을 **overfitting, 과적합**이라고 합니다.

```text
트리가 너무 깊음
→ 훈련 데이터의 노이즈나 이상치까지 학습
→ 새로운 데이터에서는 성능이 떨어질 수 있음
```

그래서 가지치기, 즉 pruning이 필요합니다.

---

## 2.2 Prepruning

Prepruning은 **트리를 만드는 중간에 미리 멈추는 것**입니다.

예:

```text
Information Gain이 threshold보다 작으면 더 이상 split하지 않음
max_depth를 제한함
min_samples_split을 설정함
min_samples_leaf를 설정함
```

`max_depth=2`는 prepruning입니다.

```python
from sklearn.tree import DecisionTreeClassifier

 tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=2,
    random_state=1
)
```

의미:

```text
트리 깊이를 최대 2까지만 허용한다.
너무 복잡한 트리가 되는 것을 막는다.
과적합을 줄이는 효과가 있다.
```

---

## 2.3 Postpruning

Postpruning은 **트리를 일단 크게 만든 다음, 불필요한 가지를 나중에 제거하는 것**입니다.

```text
트리를 완전히 성장시킴
→ 어떤 가지를 제거했을 때 validation/test error가 줄어드는지 확인
→ 성능이 좋아지면 그 가지를 제거
```

정리:

```text
Prepruning  = 만들면서 미리 멈춤
Postpruning = 다 만든 뒤 가지를 자름
```

---

# 3. Decision Tree의 장단점

## 3.1 장점: Interpretability

Decision Tree는 해석 가능성이 좋습니다.

예:

```text
이 사람은 Age > 40이고 Credit = fair이므로 Yes로 분류됨
```

즉, 왜 그런 예측이 나왔는지 규칙으로 설명할 수 있습니다.

---

## 3.2 단점 1: Unstable

Decision Tree는 데이터가 조금만 바뀌어도 트리 구조가 크게 바뀔 수 있습니다.

```text
작은 데이터 변화
→ root split이 바뀔 수 있음
→ 전체 트리 구조가 달라질 수 있음
```

그래서 불안정한 모델이라고 합니다.

---

## 3.3 단점 2: 값이 많은 feature에 편향

Decision Tree는 값의 종류가 많은 feature를 실제보다 좋아 보인다고 판단할 수 있습니다.

예:

```text
gender: 남/여 → 값 2개
age: 20, 21, 22, 23, ... → 값이 많음
student_id: 거의 모든 값이 다름
```

값이 많은 feature는 데이터를 잘게 쪼갤 수 있습니다.
그러면 훈련 데이터에서 우연히 불순도가 많이 줄어드는 것처럼 보일 수 있습니다.

핵심:

```text
값이 많은 feature는 split 후보가 많다.
후보가 많으면 우연히 IG가 크게 나올 가능성이 높다.
그래서 결정트리가 그런 feature를 과하게 선호할 수 있다.
```

---

# 4. Random Forest

## 4.1 Random Forest란?

Random Forest는 여러 개의 Decision Tree를 만들어서 투표시키는 ensemble 방법입니다.

```text
Decision Tree 하나는 불안정하고 과적합될 수 있음
→ 여러 개의 Decision Tree를 만듦
→ 다수결 voting으로 최종 class 결정
```

---

## 4.2 n_estimators=100의 의미

`n_estimators=100`은 데이터를 100개씩 나눈다는 뜻이 아닙니다.

```text
n_estimators=100
= Decision Tree를 100개 만든다는 뜻
```

예측할 때:

```text
Tree 1 → class 0
Tree 2 → class 1
Tree 3 → class 1
...
Tree 100 → class 1
```

가장 많이 나온 class를 최종 예측으로 사용합니다.

---

## 4.3 Bootstrap sampling

각 tree는 원본 training data에서 복원추출한 데이터로 학습합니다.

예:

```text
원본 데이터: A, B, C, D, E
Tree 1 학습 데이터: A, A, C, E, B
Tree 2 학습 데이터: B, D, D, E, A
```

복원추출이므로 같은 데이터가 여러 번 뽑힐 수도 있고, 어떤 데이터는 안 뽑힐 수도 있습니다.

---

## 4.4 max_features="sqrt"

`max_features="sqrt"`는 각 split에서 전체 feature를 다 보지 않고 일부 feature만 랜덤하게 보겠다는 뜻입니다.

예:

```text
전체 feature 개수 n = 4
sqrt(4) = 2
```

그러면 각 split에서 feature 2개만 후보로 보고, 그중에서 가장 좋은 split을 선택합니다.

---

# 5. K-Nearest Neighbors, K-NN

## 5.1 K-NN이란?

K-NN은 새 데이터 주변의 가장 가까운 K개 훈련 데이터를 보고 예측하는 알고리즘입니다.

```text
새 데이터 x가 들어옴
→ training data와의 거리를 계산
→ 가장 가까운 K개 이웃을 찾음
→ 그 이웃들의 label을 보고 예측
```

분류에서는 다수결, 회귀에서는 평균을 사용합니다.

---

## 5.2 Non-parametric method

Non-parametric은 고정된 형태의 수식이나 정해진 개수의 학습 파라미터를 미리 가정하지 않는다는 뜻입니다.

예를 들어 Logistic Regression은 다음과 같은 식을 학습합니다.

```text
z = w1x1 + w2x2 + b
```

여기서는 `w1`, `w2`, `b` 같은 파라미터를 학습합니다.

하지만 K-NN은 이런 식을 학습하지 않습니다.

```text
훈련 단계: 데이터를 저장
예측 단계: 새 데이터와 훈련 데이터 사이 거리 계산
```

그래서 non-parametric method라고 합니다.

주의:

```text
non-parametric = 파라미터가 진짜 하나도 없다는 뜻은 아님
K, distance metric 같은 하이퍼파라미터는 있음
```

---

## 5.3 Lazy learning

K-NN은 lazy learning입니다.

```text
학습 단계에서는 거의 하는 일이 없음
데이터를 저장만 함

예측 단계에서 거리를 계산하고 가까운 이웃을 찾음
```

즉, 학습은 게으르게 하고 예측할 때 계산을 많이 합니다.

---

## 5.4 Distance metric

Distance metric은 거리 계산 방법입니다.

K-NN에서 “가까운 이웃”을 찾기 위해 필요합니다.

대표적인 거리:

### Manhattan distance

```text
d(x, y) = |x1-y1| + |x2-y2| + ...
```

격자 거리라고 생각하면 됩니다.

### Euclidean distance

```text
d(x, y) = sqrt((x1-y1)^2 + (x2-y2)^2 + ...)
```

일반적인 직선거리입니다.

### Minkowski distance

```text
d(x, y) = (Σ |xi - yi|^p)^(1/p)
```

```text
p = 1 → Manhattan distance
p = 2 → Euclidean distance
```

---

## 5.5 Discrete-valued target y

타깃 y가 이산값이면 classification입니다.

예:

```text
y = Yes / No
y = setosa / versicolor / virginica
```

K-NN은 가까운 K개 이웃 중 가장 많이 나온 값을 예측합니다.

예:

```text
9-NN
Yes 5개, No 4개
→ Yes로 예측
```

---

## 5.6 Continuous-valued target y

타깃 y가 연속값이면 regression입니다.

예:

```text
y = 집값
y = 점수
y = 온도
```

K-NN은 가까운 K개 이웃의 y값 평균을 예측값으로 사용합니다.

예:

```text
이웃 3개의 y값: 80, 90, 100
예측값 = (80+90+100)/3 = 90
```

---

## 5.7 Distance-weighted method

기본 K-NN은 가까운 K개 이웃을 모두 똑같이 취급합니다.

하지만 distance-weighted method는 더 가까운 이웃에 더 큰 가중치를 줍니다.

```text
가까운 이웃 → 큰 가중치
먼 이웃 → 작은 가중치
```

가중치 공식:

```text
wi = 1 / d(x, x(i))
```

전체 가중치 합:

```text
W = Σ wi
```

가중 평균:

```text
y_hat = Σ (wi / W) y(i)
```

예:

```text
이웃 1 거리 = 0.5 → w = 2
이웃 2 거리 = 1.0 → w = 1
이웃 3 거리 = 1.0 → w = 1
W = 4
```

Yes를 1, No를 0이라고 하면:

```text
y_hat = 2/4 * 1 + 1/4 * 0 + 1/4 * 1 = 0.75
→ Yes 쪽에 가까우므로 Yes로 분류
```

---

## 5.8 K-NN 예제에서 9, 8, 11의 Yes/No 의미

슬라이드에서 9번, 8번, 11번 데이터가 각각 Yes/No/Yes인 이유는 거리를 계산해서 새로 만든 값이 아닙니다.

그것은 원래 training data에 붙어 있던 정답 label입니다.

```text
9번 training data의 원래 정답 → Yes
8번 training data의 원래 정답 → No
11번 training data의 원래 정답 → Yes
```

K-NN은 새 데이터와 가까운 training data를 찾은 뒤, 그 training data들의 기존 label을 보고 예측합니다.

중요:

```text
거리 계산은 feature로만 한다.
Yes/No label은 거리 계산에 사용하지 않는다.
```

---

## 5.9 Curse of Dimensionality

차원의 저주는 feature 개수가 많아질수록 데이터 공간이 너무 넓어져 데이터가 듬성듬성해지는 현상입니다.

K-NN은 거리 기반 알고리즘이므로 고차원에서 문제가 생길 수 있습니다.

```text
차원이 높아짐
→ 공간의 부피가 급격히 커짐
→ 데이터가 희소해짐
→ 가장 가까운 이웃도 실제로는 멀 수 있음
→ K-NN 성능 저하
```

또한 관련 없는 feature가 많으면 거리 계산이 방해받습니다.

```text
중요 feature: 공부시간, 출석률
관련 없는 feature: 신발 사이즈, 좋아하는 색, 책상 높이
```

K-NN은 모든 feature를 거리 계산에 넣기 때문에 관련 없는 feature가 많으면 가까운 이웃을 잘못 판단할 수 있습니다.

---

# 6. SVM, Support Vector Machine

## 6.1 SVM의 기본 아이디어

SVM은 두 클래스를 나누는 여러 직선 또는 평면 중에서 margin이 가장 큰 것을 선택하는 분류기입니다.

```text
그냥 나누는 선이 아니라,
양쪽 데이터와 가장 멀리 떨어진 가장 안전한 선을 찾는다.
```

결정 경계:

```text
w^T x + b = 0
```

분류:

```text
w^T x + b > 0 → +1 class
w^T x + b < 0 → -1 class
```

---

## 6.2 Margin

Margin은 decision boundary와 가장 가까운 데이터 사이의 거리입니다.

```text
margin이 작음 → 경계가 데이터에 가까움 → 작은 변화에도 오분류 가능
margin이 큼 → 경계가 데이터에서 멂 → 일반화 성능이 좋아질 가능성
```

SVM은 margin을 최대화하려고 합니다.

---

## 6.3 Support Vector

Support Vector는 decision boundary에 가장 가까운 데이터 포인트입니다.

```text
멀리 있는 점들은 이미 확실하게 분류됨
→ decision boundary에 큰 영향이 없음

가까운 점들은 경계 위치를 결정함
→ support vectors
```

즉, SVM의 최종 decision boundary는 support vector들에 의해 결정됩니다.

---

## 6.4 Positive / Negative hyperplane

SVM에서는 decision boundary 양쪽에 다음 두 hyperplane을 둡니다.

```text
positive hyperplane: w^T x + b = +1
negative hyperplane: w^T x + b = -1
```

가운데가 실제 decision boundary입니다.

```text
w^T x + b = 0
```

support vector들은 보통 양쪽 margin boundary 위에 있습니다.

---

## 6.5 왜 ||w||²를 최소화하는가?

점 x_i에서 decision boundary까지의 거리는:

```text
|w^T x_i + b| / ||w||
```

support vector는 `w^T x_i + b = +1` 또는 `-1` 위에 있으므로:

```text
|w^T x_i + b| = 1
```

따라서 support vector에서 decision boundary까지의 거리는:

```text
1 / ||w||
```

margin을 크게 하려면:

```text
1 / ||w||를 크게 해야 함
→ ||w||를 작게 해야 함
→ 계산 편하게 ||w||²를 minimize
```

그래서 SVM 최적화 문제는 다음처럼 됩니다.

```text
Minimize ||w||²
subject to y_i(w^T x_i + b) >= 1
```

---

## 6.6 SVM 예측식과 transpose 이해

SVM에서 최적해는 다음처럼 쓸 수 있습니다.

```text
w = Σ c_i y_i x_i
```

그러면:

```text
w^T x = (Σ c_i y_i x_i)^T x
      = Σ c_i y_i x_i^T x
```

여기서 중요한 점:

```text
c_i, y_i는 숫자라서 transpose해도 그대로이다.
x_i는 벡터라서 transpose되어 x_i^T가 된다.
```

따라서:

```text
w^T x + b = Σ c_i y_i x_i^T x + b
```

여기서 `x_i^T x`는 training data `x_i`와 새 데이터 `x`의 내적입니다.

---

# 7. SVM Kernel Trick

## 7.1 Kernel Trick이 필요한 이유

SVM은 기본적으로 선형 분류기입니다.

하지만 원래 데이터 공간에서 직선이나 평면으로 나누기 어려운 경우가 있습니다.

예:

```text
안쪽은 파란 클래스
바깥쪽은 빨간 클래스
```

이런 경우 2D에서 직선 하나로는 나누기 어렵고 원형 경계 같은 비선형 경계가 필요합니다.

---

## 7.2 고차원 변환

데이터를 고차원 feature space로 보낸다고 생각합니다.

```text
x → φ(x)
```

예:

```text
x = (x1, x2)
φ(x) = (x1, x2, x1² + x2²)
```

원래 2D에서는 직선으로 안 나뉘던 데이터가, 3D에서는 평면으로 나뉠 수 있습니다.

그리고 이 고차원에서의 선형 경계는 원래 저차원에서는 비선형 경계처럼 보입니다.

핵심:

```text
고차원에서는 linear boundary
원래 차원에서는 non-linear boundary
```

---

## 7.3 Kernel function

SVM 예측식에는 고차원 벡터 자체보다 고차원에서의 내적이 필요합니다.

```text
φ(x_i)^T φ(x)
```

Kernel function은 이 값을 직접 계산해주는 함수입니다.

```text
K(x_i, x_j) = φ(x_i) · φ(x_j)
```

즉, 실제로 φ(x)를 만들지 않고도 고차원에서 내적한 것과 같은 결과를 얻습니다.

---

## 7.4 Kernel Trick의 정의

시험용 정의:

```text
Kernel Trick은 φ(x)를 명시적으로 계산하지 않고,
K(x_i, x_j) = φ(x_i) · φ(x_j)를 이용하여
고차원 feature space에서의 inner product를 계산하는 방법이다.
```

쉽게 말하면:

```text
고차원으로 실제로 올리지 않고,
고차원에서 내적한 것처럼 계산하는 기술
```

---

## 7.5 SVM 예측식에서 kernel 사용

고차원에서 예측식은:

```text
w^T φ(x) + b
```

그리고:

```text
w = Σ c_i y_i φ(x_i)
```

따라서:

```text
w^T φ(x) + b
= Σ c_i y_i φ(x_i)^T φ(x) + b
= Σ c_i y_i K(x_i, x) + b
```

즉, 새 데이터 x와 support vector들 사이의 kernel similarity를 계산해서 예측합니다.

---

## 7.6 Polynomial kernel

Polynomial kernel:

```text
K(x_i, x_j) = (x_i · x_j + 1)^p
```

의미:

```text
다항식 형태의 비선형 decision boundary를 만들 수 있다.
p가 커질수록 더 복잡한 경계를 만들 수 있다.
```

---

## 7.7 Gaussian / RBF kernel

Gaussian 또는 RBF kernel:

```text
K(x_i, x_j) = exp(-||x_i - x_j||² / (2σ²))
```

의미:

```text
두 점이 가까우면 K값이 1에 가까움
두 점이 멀면 K값이 0에 가까움
```

즉, 거리 기반 similarity입니다.

RBF kernel은 무한 차원 feature space의 내적처럼 해석할 수 있지만, 실제로는 kernel function 하나로 계산합니다.

---

# 8. sklearn class / import 정리

## 8.1 데이터셋

```python
from sklearn import datasets

iris = datasets.load_iris()
```

또는:

```python
from sklearn.datasets import load_iris, load_wine

iris = load_iris()
wine = load_wine()
```

---

## 8.2 Train/Test split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=1,
    stratify=y
)
```

의미:

```text
test_size=0.3 → 전체 데이터 중 30%를 test set으로 사용
random_state=1 → 결과 재현을 위한 seed
stratify=y → class 비율을 train/test에 비슷하게 유지
```

---

## 8.3 StandardScaler

```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
```

의미:

```text
평균 0, 표준편차 1이 되도록 feature를 표준화한다.
```

중요:

```text
fit은 X_train에만 한다.
test data에는 transform만 한다.
```

이유:

```text
test data 정보가 훈련 과정에 들어가면 data leakage가 발생하기 때문이다.
```

---

## 8.4 Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

 tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=2,
    random_state=1
)

tree.fit(X_train, y_train)
```

주요 파라미터:

```text
criterion="gini"    → Gini impurity 기준
criterion="entropy" → Entropy 기준
max_depth=2         → 트리 최대 깊이 제한, prepruning 효과
random_state=1      → 결과 재현
```

트리 시각화:

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plot_tree(
    tree,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    fontsize=11
)
plt.show()
```

---

## 8.5 Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(
    criterion="gini",
    n_estimators=100,
    max_depth=2,
    max_features="sqrt",
    random_state=1
)

forest.fit(X_train, y_train)
```

주요 파라미터:

```text
n_estimators=100   → decision tree 100개 생성
max_depth=2        → 각 tree의 깊이를 최대 2로 제한
max_features="sqrt" → 각 split에서 sqrt(n_features)개 feature만 랜덤 선택
criterion="gini"   → Gini 기준
```

정확도 확인:

```python
print("Train Accuracy:", forest.score(X_train, y_train))
print("Test Accuracy:", forest.score(X_test, y_test))
```

---

## 8.6 K-NN

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(
    n_neighbors=5,
    metric="minkowski",
    p=2
)

knn.fit(X_train_std, y_train)
```

주요 파라미터:

```text
n_neighbors=5 → K=5
metric="minkowski" → Minkowski distance 사용
p=1 → Manhattan distance
p=2 → Euclidean distance
```

주의:

```text
K-NN은 거리 기반 알고리즘이므로 feature scaling이 중요하다.
따라서 보통 X_train_std, X_test_std를 사용한다.
```

예측:

```python
y_pred = knn.predict(X_test_std)
```

정확도:

```python
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
```

---

## 8.7 SVM

```python
from sklearn.svm import SVC

svm = SVC(
    kernel="linear",
    C=1.0,
    random_state=1
)

svm.fit(X_train_std, y_train)
```

주요 파라미터:

```text
kernel="linear" → 선형 SVM
kernel="rbf"    → RBF kernel SVM
kernel="poly"   → Polynomial kernel SVM
C=1.0            → regularization 강도 조절
random_state=1   → 결과 재현
```

RBF kernel 예:

```python
svm_rbf = SVC(
    kernel="rbf",
    C=1.0,
    gamma=0.2,
    random_state=1
)

svm_rbf.fit(X_train_std, y_train)
```

Polynomial kernel 예:

```python
svm_poly = SVC(
    kernel="poly",
    degree=3,
    C=1.0,
    random_state=1
)

svm_poly.fit(X_train_std, y_train)
```

주의:

```text
SVM도 거리와 내적 기반이므로 scaling이 중요하다.
보통 X_train_std, X_test_std를 사용한다.
```

---

## 8.8 Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    C=1.0,
    random_state=1
)

lr.fit(X_train_std, y_train)
```

주요 파라미터:

```text
C가 작다 → regularization 강함 → weight 작아짐 → 단순한 모델
C가 크다 → regularization 약함 → weight 커질 수 있음 → 복잡한 모델
```

---

## 8.9 Pipeline

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe_lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=1)
)

pipe_lr.fit(X_train, y_train)
```

Pipeline 의미:

```text
전처리와 모델 학습을 하나로 묶는다.
fit할 때 scaler는 train data에만 fit되고,
test data에는 자동으로 transform만 적용된다.
```

---

## 8.10 Validation Curve

```python
from sklearn.model_selection import validation_curve

train_scores, test_scores = validation_curve(
    estimator=pipe_lr,
    X=X_train,
    y=y_train,
    param_name="logisticregression__C",
    param_range=[0.001, 0.01, 0.1, 1, 10, 100],
    cv=10
)
```

의미:

```text
param_name="logisticregression__C"
→ Pipeline 안의 LogisticRegression의 C 값을 바꿔가며 실험한다.

param_range
→ 실험할 C 값 목록

cv=10
→ 10-fold cross validation
```

결과 shape:

```text
train_scores.shape = (파라미터 후보 개수, fold 개수)
```

예:

```text
param_range에 값 6개, cv=10이면
train_scores.shape = (6, 10)
```

---

## 8.11 Bagging / AdaBoost import

Bagging:

```python
from sklearn.ensemble import BaggingClassifier
```

AdaBoost:

```python
from sklearn.ensemble import AdaBoostClassifier
```

Random Forest:

```python
from sklearn.ensemble import RandomForestClassifier
```

---

## 8.12 PCA

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
```

의미:

```text
고차원 데이터를 분산이 큰 방향의 새로운 축으로 변환한다.
n_components=2이면 2차원으로 축소한다.
```

주의:

```text
PCA도 fit은 train data에만 한다.
test data에는 transform만 한다.
```

---

# 9. 자주 헷갈린 핵심 문장 모음

## Decision Tree

```text
Gini는 현재 노드가 얼마나 섞여 있는지를 나타낸다.
Gini는 낮을수록 좋다.

IG는 split 후 불순도가 얼마나 줄었는지를 나타낸다.
IG는 클수록 좋다.

Decision Tree는 모든 feature와 split 후보를 비교하여
IG가 가장 큰 split을 선택한다.
```

## Pruning

```text
max_depth=2는 트리를 최대 깊이 2까지만 만들겠다는 뜻이다.
이는 postpruning이 아니라 prepruning이다.
```

## Random Forest

```text
n_estimators=100은 데이터 100개가 아니라 tree 100개이다.
```

## K-NN

```text
K-NN은 명시적인 모델식을 학습하지 않고,
training data를 저장한 뒤 예측할 때 가까운 이웃을 찾는다.

분류에서는 다수결,
회귀에서는 평균을 사용한다.

distance metric은 가까운 이웃을 찾기 위한 거리 계산 공식이다.
```

## SVM

```text
SVM은 margin이 가장 큰 decision boundary를 찾는다.
Support vector는 decision boundary에 가장 가까운 데이터이다.

margin을 최대화한다는 것은 1/||w||를 크게 하는 것이고,
이는 ||w||²를 최소화하는 문제로 바뀐다.
```

## Kernel Trick

```text
SVM은 원래 선형 분류기이다.
하지만 원래 차원에서 선형 분리가 안 되는 경우가 있다.

이때 데이터를 고차원으로 보낸 것처럼 계산하면
고차원에서는 선형 분리가 가능할 수 있다.

Kernel trick은 φ(x)를 직접 계산하지 않고,
K(xi, xj)=φ(xi)·φ(xj)를 이용해
고차원에서의 내적값을 저차원에서 바로 계산하는 방법이다.

결과적으로 원래 차원에서는 비선형 decision boundary를 얻을 수 있다.
```

---

# 10. 전체 import 한 번에 모음

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

---

# 11. 대표 모델 코드 템플릿

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. 데이터 로드
iris = load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# 2. train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=1,
    stratify=y
)

# 3. scaling: KNN, SVM에는 중요
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 4. Decision Tree
tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=2,
    random_state=1
)
tree.fit(X_train, y_train)

# 5. Random Forest
forest = RandomForestClassifier(
    criterion="gini",
    n_estimators=100,
    max_depth=2,
    max_features="sqrt",
    random_state=1
)
forest.fit(X_train, y_train)

# 6. K-NN
knn = KNeighborsClassifier(
    n_neighbors=5,
    metric="minkowski",
    p=2
)
knn.fit(X_train_std, y_train)

# 7. SVM
svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma=0.2,
    random_state=1
)
svm.fit(X_train_std, y_train)

# 8. 평가
models = {
    "Decision Tree": (tree, X_test),
    "Random Forest": (forest, X_test),
    "K-NN": (knn, X_test_std),
    "SVM": (svm, X_test_std)
}

for name, (model, X_eval) in models.items():
    y_pred = model.predict(X_eval)
    print(name, accuracy_score(y_test, y_pred))
```

---

# 12. 시험 직전 암기용 요약

```text
Decision Tree:
IG가 가장 큰 feature/split을 선택한다.
Gini/Entropy는 불순도 측정 방법이다.

Gini:
낮을수록 순수하다.

Information Gain:
split 전후 불순도 감소량이다.
클수록 좋은 split이다.

Pruning:
과적합을 줄이기 위해 가지를 줄인다.
max_depth는 prepruning이다.

Random Forest:
여러 decision tree를 만들고 voting한다.
n_estimators는 tree 개수이다.

K-NN:
가까운 K개 이웃을 보고 예측한다.
분류는 다수결, 회귀는 평균이다.
거리 기반이므로 scaling이 중요하다.

SVM:
margin을 최대화하는 decision boundary를 찾는다.
support vector가 경계를 결정한다.

Kernel Trick:
고차원으로 직접 변환하지 않고 kernel로 고차원 내적을 계산한다.
원래 차원에서는 비선형 decision boundary를 얻을 수 있다.
```
