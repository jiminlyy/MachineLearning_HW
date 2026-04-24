# 머신러닝 예상 문제 및 정답

---

## Part 1: 개념 설명 문제

---

### 문제 1. Tom Mitchell의 정의

> Tom Mitchell의 머신러닝 정의에서 T, P, E가 각각 무엇을 의미하는지 설명하고, "이메일 스팸 필터"를 예로 들어 T, P, E를 구체적으로 서술하시오.

<details>
<summary>정답</summary>

Tom Mitchell의 정의: "프로그램이 어떤 **과제(T)**에 대해 **경험(E)**을 통해 **성능(P)**이 향상되면, 그 프로그램은 학습한다."

- **T (Task)**: 프로그램이 수행하는 과제
- **P (Performance)**: 과제 수행 성능을 측정하는 지표
- **E (Experience)**: 학습에 사용되는 데이터(경험)

스팸 필터 예시:

- T = 수신된 이메일이 스팸인지 아닌지를 분류하는 것
- P = 분류 정확도 (%)
- E = 이메일 데이터 + 사용자가 스팸/정상으로 표시한 피드백 데이터

</details>

---

### 문제 2. Parameter vs Hyperparameter

> Parameter와 Hyperparameter의 차이를 설명하고, 선형 회귀 모델 ŷ = wx + b를 gradient descent로 학습할 때 각각의 예를 하나씩 제시하시오.

<details>
<summary>정답</summary>

- **Parameter**: 모델이 데이터로부터 **학습하는** 값. 학습 알고리즘이 자동으로 결정한다.
  - 예: w(가중치), b(편향)

- **Hyperparameter**: 학습 알고리즘을 **제어하는 설정값**. 사람이 직접 지정하거나 튜닝해야 한다.
  - 예: α(learning rate), 반복 횟수(iterations), 정규화 파라미터 λ

핵심 차이: Parameter는 학습 과정에서 자동 업데이트되고, Hyperparameter는 학습 **전에** 미리 설정해야 한다.

</details>

---

### 문제 3. Overfitting (과적합)

> 과적합(Overfitting)이 무엇인지 설명하고, 이를 방지하기 위한 방법을 2가지 이상 서술하시오.

<details>
<summary>정답</summary>

**과적합**: 모델이 훈련 데이터에 너무 맞춰져서, **새로운 데이터(테스트 데이터)**에 대한 예측 성능이 떨어지는 현상이다. 예를 들어 단순한 y = ax + b 대신 10차 다항식을 쓰면 훈련 데이터에는 완벽히 맞지만, 새로운 데이터에서는 성능이 나빠진다.

방지 방법:

1. **Regularization (정규화)**: 비용 함수에 가중치의 크기에 대한 패널티 항을 추가하여 모델을 단순화한다 (L1, L2 정규화).
2. **Pruning (가지치기)**: Decision Tree에서 불필요한 가지를 제거하여 모델 복잡도를 줄인다.
3. **K-fold Cross Validation**: 데이터를 k개로 나눠 번갈아 테스트하여, 특정 데이터 분할에 과적합되지 않도록 한다.
4. **더 많은 데이터 확보**: 훈련 데이터가 많을수록 과적합이 줄어든다.
5. **Feature Selection**: 불필요한 feature를 제거하여 모델을 단순화한다.

</details>

---

### 문제 4. Sigmoid 함수의 유도

> 선형 회귀 모델의 출력 범위는 (−∞, +∞)인데, 분류 문제에서는 확률 [0, 1]이 필요하다. Odds → Logit → Sigmoid로 이어지는 흐름을 설명하여, 왜 Sigmoid 함수가 필요한지 서술하시오.

<details>
<summary>정답</summary>

1. **Odds (승산비)**: 성공 확률 대 실패 확률의 비율이다. 범위는 [0, ∞).
   - Odds = p / (1 − p)

2. **Logit**: Odds에 자연로그를 취하면 범위가 **(−∞, +∞)**가 된다.
   - Logit(p) = ln(p / (1 − p))

3. 이제 **Logit(p) = wx + b**로 놓으면, 좌변(−∞, +∞)과 우변(−∞, +∞)의 범위가 일치한다.

4. 이 식을 p에 대해 정리하면:
   - ln(p / (1−p)) = z (z = wx + b)
   - p / (1−p) = eᶻ
   - p = eᶻ(1−p) = eᶻ − p·eᶻ
   - p(1 + eᶻ) = eᶻ
   - **p = eᶻ / (1 + eᶻ) = 1 / (1 + e⁻ᶻ)**

이것이 **Sigmoid 함수** σ(z) = 1 / (1 + e⁻ᶻ)이다. 임의의 실수 z를 (0, 1) 범위로 변환하여 확률로 해석할 수 있게 해준다.

</details>

---

### 문제 5. BCE가 MSE 대신 사용되는 이유

> Logistic Regression의 비용 함수로 MSE 대신 Binary Cross Entropy (BCE)를 사용하는 이유를 설명하시오.

<details>
<summary>정답</summary>

MSE를 sigmoid와 결합하면 비용 함수가 **non-convex**(볼록하지 않은) 형태가 된다. 이 경우 gradient descent가 **local minimum(지역 최솟값)**에 빠져서 global minimum을 찾지 못할 수 있다.

반면 BCE는 **MLE(Maximum Likelihood Estimation)**에서 유도되며, sigmoid와 결합해도 **convex** 함수가 되어 gradient descent가 항상 global minimum으로 수렴할 수 있다.

BCE 함수:

> J(θ) = (1/m) Σ [−y log(ŷ) − (1−y) log(1−ŷ)]

이 함수는 다음과 같이 직관적으로 이해할 수 있다:

- y = 1일 때: ŷ가 0에 가까우면 −log(ŷ) → ∞로 큰 패널티
- y = 0일 때: ŷ가 1에 가까우면 −log(1−ŷ) → ∞로 큰 패널티

즉, 잘못된 예측에 대해 무한대에 가까운 큰 비용을 부과하여 올바른 방향으로 학습을 유도한다.

</details>

---

### 문제 6. L1 vs L2 정규화

> L1 정규화(Lasso)와 L2 정규화(Ridge)의 차이를 설명하고, 왜 L1 정규화가 feature selection 효과를 가지는지 기하학적으로 설명하시오.

<details>
<summary>정답</summary>

| 구분        | L1 (Lasso)                          | L2 (Ridge)                                    |
| ----------- | ----------------------------------- | --------------------------------------------- |
| 패널티      | λ Σ \|wⱼ\| (절대값 합)              | λ Σ wⱼ² (제곱 합)                             |
| 가중치 효과 | 일부 가중치를 **정확히 0**으로 만듦 | 모든 가중치를 **작게** 만들지만 0으로는 안 됨 |
| 특징        | Feature selection 효과 있음         | 모든 feature를 유지하면서 영향을 줄임         |

**기하학적 설명**:

- L2의 제약 조건은 **원(circle)** 형태이다. 비용 함수의 등고선과 원이 만나는 점은 축 위(w = 0)에 떨어지기 어렵다.
- L1의 제약 조건은 **마름모(diamond)** 형태이다. 마름모의 **꼭짓점이 축 위에** 있으므로, 비용 함수의 등고선과 마름모가 꼭짓점에서 만날 확률이 높다. 꼭짓점은 하나 이상의 wⱼ = 0인 점이므로, **일부 가중치가 정확히 0이 되어 해당 feature가 자동으로 제거**된다.

</details>

---

### 문제 7. Decision Tree의 학습 알고리즘

> Decision Tree가 어떤 feature로 분할할지 결정하는 과정에서 사용되는 Entropy와 Information Gain의 개념을 설명하시오.

<details>
<summary>정답</summary>

**Entropy (엔트로피)**: 집합의 **불확실도(불순도)**를 측정하는 지표이다.

> I(S) = −Σ pᵢ log₂(pᵢ)

- pᵢ = 클래스 i의 비율
- 두 클래스가 반반(0.5, 0.5)이면 엔트로피 = 1 (최대 불확실)
- 한 클래스만 있으면(1.0, 0.0) 엔트로피 = 0 (완전 확실)

직관: "하나의 샘플을 분류하는 데 필요한 정보량(비트 수)"이다.

**Information Gain (정보 이득)**: 분할 전후의 엔트로피 감소량이다.

> IG(Dₚ, f) = I(Dₚ) − Σⱼ (Nⱼ/Nₚ) · I(Dⱼ)

- Dₚ: 분할 전 데이터셋
- Dⱼ: 분할 후 각 자식 노드의 데이터셋
- Nⱼ/Nₚ: 각 자식 노드의 데이터 비율 (가중 평균)

IG가 **가장 큰 feature**를 선택하여 분할한다. IG가 크다 = 그 feature로 분할하면 불순도가 가장 많이 줄어든다.

</details>

---

### 문제 8. K-NN과 차원의 저주

> K-NN이 고차원 데이터에서 성능이 떨어지는 이유를 "차원의 저주(Curse of Dimensionality)"의 관점에서 설명하시오.

<details>
<summary>정답</summary>

K-NN은 **거리 기반** 알고리즘으로, 가장 가까운 K개의 이웃을 찾아 분류한다. 그런데 차원이 높아지면 다음 문제가 발생한다:

1. **모든 데이터 간 거리가 비슷해진다**: 고차원 공간에서는 데이터 포인트들이 "모서리"에 분포하게 되어, 가까운 이웃과 먼 이웃의 거리 차이가 줄어든다.

2. **이웃을 찾기 위해 거의 전체 공간을 탐색해야 한다**: 예를 들어 N=1000, k=10일 때,
   - 2차원: 전체 공간의 약 10%만 탐색 (b ≈ 0.1)
   - 100차원: 전체 공간의 약 95%를 탐색해야 함 (b ≈ 0.95)

   즉 거의 모든 데이터가 "이웃"이 되어 **"가장 가까운 이웃"이라는 개념 자체가 의미를 잃는다.**

따라서 고차원 데이터에서 K-NN을 사용하려면 PCA 등으로 **차원 축소**를 먼저 적용하는 것이 필요하다.

</details>

---

### 문제 9. SVM과 Kernel Trick

> SVM에서 Kernel Trick이 필요한 이유와 원리를 설명하고, RBF(Gaussian) 커널의 특징을 서술하시오.

<details>
<summary>정답</summary>

**필요한 이유**: SVM은 선형 결정 경계(hyperplane)를 찾는 알고리즘이다. 하지만 현실의 많은 데이터는 **선형으로 분리되지 않는다**. 예를 들어 2D에서 원형으로 분포된 데이터는 직선으로 나눌 수 없다.

**원리**: 원래 공간의 데이터 x를 **고차원 공간 ϕ(x)로 변환**하면 선형 분리가 가능해질 수 있다. 그러나 실제로 고차원 변환을 계산하면 비용이 매우 크다. **Kernel Trick**은 ϕ(x)를 명시적으로 계산하지 않고, **커널 함수 K(xᵢ, xⱼ) = ϕ(xᵢ)·ϕ(xⱼ)**만으로 내적을 바로 구한다. SVM의 최적화와 예측에는 **내적만 필요**하므로 이것으로 충분하다.

**RBF (Gaussian) 커널**:

> K(xᵢ, xⱼ) = exp(−‖xᵢ − xⱼ‖² / 2σ²)

- 두 데이터의 **유사도**를 측정한다 (가까우면 1, 멀면 0에 가까움)
- Taylor 급수 전개를 통해 **무한 차원**에서의 내적과 동일함을 보일 수 있다
- sklearn: `SVC(kernel='rbf', C=1.0, gamma=0.1)` — gamma는 1/(2σ²)에 해당

</details>

---

### 문제 10. PCA vs LDA

> PCA와 LDA의 차이를 설명하시오. 각각의 목표, 학습 유형, 적합한 상황을 비교하시오.

<details>
<summary>정답</summary>

| 구분          | PCA                                                  | LDA                                                                           |
| ------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------- |
| 학습 유형     | **비지도 학습** (라벨 불필요)                        | **지도 학습** (라벨 필요)                                                     |
| 목표          | 데이터의 **분산이 가장 큰 방향**(주성분)을 찾아 투영 | **클래스 간 분산은 최대화**, **클래스 내 분산은 최소화**하는 방향을 찾아 투영 |
| 적합한 상황   | 클래스 정보가 없거나, 클래스당 샘플 수가 적을 때     | 대규모 데이터 + 다중 클래스 분류 문제                                         |
| 제한 사항     | 특별한 가정 없음                                     | 각 클래스가 **정규분포**이고 **공분산이 같아야** 함                           |
| 사용하는 행렬 | 공분산 행렬의 고유벡터                               | S_W⁻¹S_B의 고유벡터                                                           |

핵심 차이: PCA는 "데이터를 가장 잘 요약하는 방향"을 찾고, LDA는 "클래스를 가장 잘 구분하는 방향"을 찾는다.

</details>

---

### 문제 11. 전처리에서 fit()은 훈련 데이터로만 하는 이유

> sklearn의 전처리 도구(StandardScaler 등)에서 fit()을 반드시 훈련 데이터로만 해야 하는 이유를 설명하시오.

<details>
<summary>정답</summary>

fit()은 데이터의 통계량(평균, 표준편차, 최솟값, 최댓값 등)을 학습하는 과정이다.

만약 **테스트 데이터를 포함하여 fit()**하면, 테스트 데이터의 분포 정보가 전처리 과정에 반영된다. 이는 모델이 미래 데이터(테스트 데이터)의 정보를 미리 "엿보는" 것이 되어 **data leakage(데이터 누출)**가 발생한다.

올바른 절차:

1. `scaler.fit(X_train)` — 훈련 데이터의 통계량만 학습
2. `scaler.transform(X_train)` — 훈련 데이터 변환
3. `scaler.transform(X_test)` — 테스트 데이터도 **훈련 데이터의 통계량으로** 변환

이렇게 해야 모델이 실전에서 처음 보는 데이터에 대해 공정하게 평가된다.

</details>

---

### 문제 12. One-hot Encoding이 필요한 이유

> Nominal feature(순서가 없는 범주형 변수)에 단순히 정수를 부여하면 안 되는 이유를 설명하고, 대안으로 사용하는 One-hot Encoding의 원리를 서술하시오.

<details>
<summary>정답</summary>

예를 들어 색상(red, green, blue)에 대해 red=1, green=2, blue=3으로 정수 인코딩을 하면, 모델이 **"blue(3) > red(1)"이라는 크기 순서**를 학습하게 된다. 실제로는 색상 간에 순서가 없으므로 이는 **잘못된 정보**이다.

**One-hot Encoding**: 각 범주를 별도의 이진(0/1) feature로 변환한다.

| 원래 값 | red | green | blue |
| ------- | --- | ----- | ---- |
| red     | 1   | 0     | 0    |
| green   | 0   | 1     | 0    |
| blue    | 0   | 0     | 1    |

이렇게 하면 범주 간에 크기 관계가 존재하지 않으므로, 모델이 잘못된 순서를 학습하는 문제가 사라진다.

sklearn에서는 `OneHotEncoder()`, Pandas에서는 `pd.get_dummies()`를 사용한다.

</details>

---

## Part 2: 숫자를 가지고 푸는 계산 문제

---

### 문제 13. Gradient Descent 손계산 (Linear Regression)

> 데이터가 2개 있다: (x₁, y₁) = (1, 2), (x₂, y₂) = (2, 4).
> 초기값 w = 0, b = 0, learning rate α = 0.1일 때, gradient descent 1회 업데이트 후의 w와 b를 구하시오.

<details>
<summary>정답</summary>

**Step 1: 예측값 계산** (ŷ = wx + b)

- ŷ₁ = 0·1 + 0 = 0
- ŷ₂ = 0·2 + 0 = 0

**Step 2: Gradient 계산** (m = 2)

- ∂J/∂w = (1/m) Σ (ŷ⁽ⁱ⁾ − y⁽ⁱ⁾) · x⁽ⁱ⁾
  = (1/2) [(0−2)·1 + (0−4)·2]
  = (1/2) [−2 + (−8)]
  = (1/2)(−10) = **−5**

- ∂J/∂b = (1/m) Σ (ŷ⁽ⁱ⁾ − y⁽ⁱ⁾)
  = (1/2) [(0−2) + (0−4)]
  = (1/2)(−6) = **−3**

**Step 3: 업데이트**

- w = w − α · (∂J/∂w) = 0 − 0.1 · (−5) = **0.5**
- b = b − α · (∂J/∂b) = 0 − 0.1 · (−3) = **0.3**

업데이트 후: **w = 0.5, b = 0.3**

</details>

---

### 문제 14. Sigmoid 함수 값 계산

> z = 0, z = 2, z = −2 일 때 각각 σ(z)의 값을 구하시오. 그리고 decision boundary에서의 z 값과 그때의 σ(z) 값을 설명하시오. (e² ≈ 7.389로 계산)

<details>
<summary>정답</summary>

σ(z) = 1 / (1 + e⁻ᶻ)

- **z = 0**: σ(0) = 1 / (1 + e⁰) = 1 / (1 + 1) = **0.5**
- **z = 2**: σ(2) = 1 / (1 + e⁻²) = 1 / (1 + 1/7.389) ≈ 1 / 1.135 ≈ **0.881**
- **z = −2**: σ(−2) = 1 / (1 + e²) = 1 / (1 + 7.389) ≈ 1 / 8.389 ≈ **0.119**

**Decision boundary**: z = 0일 때 σ(z) = 0.5이다.

- σ(z) ≥ 0.5 → class 1 (z ≥ 0과 동치)
- σ(z) < 0.5 → class 0 (z < 0과 동치)

즉 decision boundary는 z = wx + b = 0인 초평면이다.

</details>

---

### 문제 15. BCE 비용 계산

> 2개의 데이터에 대해 실제값과 예측 확률이 다음과 같다:
>
> - 데이터 1: y = 1, ŷ = 0.9
> - 데이터 2: y = 0, ŷ = 0.3
>
> BCE 비용 함수 J를 계산하시오. (ln(0.9) ≈ −0.105, ln(0.1) ≈ −2.303, ln(0.7) ≈ −0.357, ln(0.3) ≈ −1.204로 계산)

<details>
<summary>정답</summary>

BCE: J = (1/m) Σ [−y log(ŷ) − (1−y) log(1−ŷ)]

**데이터 1** (y=1, ŷ=0.9):

- −1·ln(0.9) − 0·ln(0.1) = −ln(0.9) = −(−0.105) = 0.105

**데이터 2** (y=0, ŷ=0.3):

- −0·ln(0.3) − 1·ln(0.7) = −ln(0.7) = −(−0.357) = 0.357

**전체 비용**:

- J = (1/2)(0.105 + 0.357) = (1/2)(0.462) = **0.231**

해석: 두 예측 모두 비교적 정확하므로(y=1일 때 ŷ=0.9, y=0일 때 ŷ=0.3) 비용이 낮다. 만약 y=1인데 ŷ=0.1이었다면 −ln(0.1) = 2.303으로 훨씬 큰 비용이 발생한다.

</details>

---

### 문제 16. Entropy와 Information Gain 계산

> 전체 데이터 10개 중 클래스 A가 6개, 클래스 B가 4개이다. Feature f로 분할하면:
>
> - 왼쪽 자식: 4개 (A=3, B=1)
> - 오른쪽 자식: 6개 (A=3, B=3)
>
> (1) 부모 노드의 엔트로피를 구하시오.
> (2) 각 자식 노드의 엔트로피를 구하시오.
> (3) Information Gain을 구하시오.
> (log₂(0.6) ≈ −0.737, log₂(0.4) ≈ −1.322, log₂(0.75) ≈ −0.415, log₂(0.25) ≈ −2, log₂(0.5) = −1)

<details>
<summary>정답</summary>

**(1) 부모 노드 엔트로피**:

- p_A = 6/10 = 0.6, p_B = 4/10 = 0.4
- I(Dₚ) = −0.6·log₂(0.6) − 0.4·log₂(0.4)
- = −0.6·(−0.737) − 0.4·(−1.322)
- = 0.442 + 0.529
- = **0.971**

**(2) 자식 노드 엔트로피**:

왼쪽 (A=3, B=1, 총 4개):

- p_A = 3/4 = 0.75, p_B = 1/4 = 0.25
- I(D_left) = −0.75·log₂(0.75) − 0.25·log₂(0.25)
- = −0.75·(−0.415) − 0.25·(−2)
- = 0.311 + 0.5
- = **0.811**

오른쪽 (A=3, B=3, 총 6개):

- p_A = 0.5, p_B = 0.5
- I(D_right) = −0.5·log₂(0.5) − 0.5·log₂(0.5)
- = −0.5·(−1) − 0.5·(−1)
- = 0.5 + 0.5
- = **1.0**

**(3) Information Gain**:

- IG = I(Dₚ) − (N_left/N)·I(D_left) − (N_right/N)·I(D_right)
- = 0.971 − (4/10)·0.811 − (6/10)·1.0
- = 0.971 − 0.324 − 0.6
- = **0.047**

IG가 0.047으로 작다 → 이 feature로 분할해도 불순도가 거의 줄지 않는다. 다른 feature와 비교하여 IG가 가장 큰 것을 선택해야 한다.

</details>

---

### 문제 17. K-NN 분류

> 2차원 데이터에서 새로운 점 q = (3, 3)을 분류하려 한다. 훈련 데이터는 다음과 같다:
>
> | 점  | x₁  | x₂  | 클래스 |
> | --- | --- | --- | ------ |
> | A   | 1   | 1   | ●      |
> | B   | 2   | 2   | ●      |
> | C   | 4   | 4   | ▲      |
> | D   | 5   | 5   | ▲      |
> | E   | 2   | 4   | ▲      |
>
> (1) 각 점과 q 사이의 Euclidean 거리를 구하시오.
> (2) K=3일 때 q의 클래스를 결정하시오.

<details>
<summary>정답</summary>

**(1) Euclidean 거리**: d = √((x₁−3)² + (x₂−3)²)

- d(A, q) = √((1−3)² + (1−3)²) = √(4+4) = √8 ≈ **2.83**
- d(B, q) = √((2−3)² + (2−3)²) = √(1+1) = √2 ≈ **1.41**
- d(C, q) = √((4−3)² + (4−3)²) = √(1+1) = √2 ≈ **1.41**
- d(D, q) = √((5−3)² + (5−3)²) = √(4+4) = √8 ≈ **2.83**
- d(E, q) = √((2−3)² + (4−3)²) = √(1+1) = √2 ≈ **1.41**

**(2) K=3일 때**:
가장 가까운 3개: B(●, 1.41), C(▲, 1.41), E(▲, 1.41)

다수결: ● = 1개, ▲ = 2개

**q의 클래스 = ▲**

</details>

---

### 문제 18. Normalization과 Standardization 계산

> Feature 값이 [10, 20, 30, 40, 50]일 때:
> (1) Min-Max Normalization을 적용하여 x = 30의 변환값을 구하시오.
> (2) Standardization(표준화)을 적용하여 x = 30의 변환값을 구하시오.

<details>
<summary>정답</summary>

**(1) Min-Max Normalization**: x_new = (x − x_min) / (x_max − x_min)

- x_min = 10, x_max = 50
- x_new = (30 − 10) / (50 − 10) = 20 / 40 = **0.5**

**(2) Standardization**: x_new = (x − μ) / σ

- 평균 μ = (10+20+30+40+50) / 5 = 150 / 5 = **30**
- 분산 = [(10−30)² + (20−30)² + (30−30)² + (40−30)² + (50−30)²] / 5
  = [400 + 100 + 0 + 100 + 400] / 5 = 1000 / 5 = 200
- 표준편차 σ = √200 ≈ **14.14**
- x_new = (30 − 30) / 14.14 = 0 / 14.14 = **0**

해석: 30은 정확히 평균이므로 표준화 후 0이 되는 것이 맞다.

</details>

---

### 문제 19. Logistic Regression Gradient 계산

> 데이터 1개: x = 2, y = 1. 현재 w = 0.5, b = −1이다.
> (1) z = wx + b를 구하시오.
> (2) ŷ = σ(z)를 구하시오.
> (3) ∂J/∂w와 ∂J/∂b를 구하시오.
> (e⁰ = 1로 계산)

<details>
<summary>정답</summary>

**(1)** z = wx + b = 0.5 · 2 + (−1) = 1 − 1 = **0**

**(2)** ŷ = σ(0) = 1 / (1 + e⁰) = 1 / (1 + 1) = **0.5**

**(3)** Logistic Regression의 gradient는 선형 회귀와 형태가 같다 (단, ŷ = σ(wx+b)):

- ∂J/∂w = (ŷ − y) · x = (0.5 − 1) · 2 = (−0.5) · 2 = **−1**
- ∂J/∂b = (ŷ − y) = 0.5 − 1 = **−0.5**

gradient가 음수이므로 업데이트 시 w와 b가 증가한다 (w = w − α(−1) = w + α). 이는 현재 ŷ = 0.5인데 실제 y = 1이므로 예측을 더 높여야 하기 때문에 올바른 방향이다.

</details>

---

### 문제 20. Regularization이 적용된 비용 함수

> Linear Regression에서 w = [3, −2], λ = 2, m = 10이고, 정규화 없는 비용 J(w) = 5.0이다.
> L2 정규화가 적용된 전체 비용을 구하시오.

<details>
<summary>정답</summary>

L2 정규화된 비용:

> Cost = J(w) + (λ / 2m) Σ wⱼ²

가중치 제곱합:

- Σ wⱼ² = 3² + (−2)² = 9 + 4 = 13

패널티 항:

- (λ / 2m) · Σ wⱼ² = (2 / (2·10)) · 13 = (2/20) · 13 = 0.1 · 13 = 1.3

전체 비용:

- Cost = 5.0 + 1.3 = **6.3**

해석: 정규화에 의해 비용이 5.0에서 6.3으로 증가했다. 이 패널티 때문에 gradient descent가 w의 크기를 줄이는 방향으로 학습하게 된다.

</details>

---

## Part 3: Scikit-learn 관련 문제

---

### 문제 21. sklearn 기본 패턴

> sklearn에서 모델을 학습하고 예측하는 기본 패턴(2단계)을 작성하시오. 그리고 전처리 도구(예: StandardScaler)의 기본 패턴(3단계)도 작성하시오.

<details>
<summary>정답</summary>

**모델 학습/예측 패턴**:

1. `model.fit(X_train, y_train)` — 훈련 데이터로 모델 학습
2. `model.predict(X_test)` — 새로운 데이터에 대한 예측

**전처리 도구 패턴**:

1. `scaler.fit(X_train)` — 훈련 데이터의 통계량 학습
2. `scaler.transform(X_train)` — 훈련 데이터 변환
3. `scaler.transform(X_test)` — 테스트 데이터 변환 (훈련 데이터의 통계량으로)

※ 1+2를 합쳐서 `scaler.fit_transform(X_train)`으로 쓸 수도 있다. 단, 테스트 데이터에는 반드시 `transform()`만 사용해야 한다.

</details>

---

### 문제 22. 알고리즘별 sklearn 클래스

> 다음 각 알고리즘에 대해 sklearn에서 사용하는 클래스 이름을 쓰시오.
>
> (1) 선형 회귀
> (2) L2 정규화가 적용된 선형 회귀
> (3) 로지스틱 회귀
> (4) 다항 feature 생성
> (5) Decision Tree 분류기
> (6) Random Forest 분류기
> (7) K-NN 분류기
> (8) SVM 분류기

<details>
<summary>정답</summary>

| 알고리즘                 | sklearn 클래스           | 모듈                    |
| ------------------------ | ------------------------ | ----------------------- |
| (1) 선형 회귀            | `LinearRegression`       | `sklearn.linear_model`  |
| (2) Ridge (L2 선형 회귀) | `Ridge`                  | `sklearn.linear_model`  |
| (3) 로지스틱 회귀        | `LogisticRegression`     | `sklearn.linear_model`  |
| (4) 다항 feature 생성    | `PolynomialFeatures`     | `sklearn.preprocessing` |
| (5) Decision Tree        | `DecisionTreeClassifier` | `sklearn.tree`          |
| (6) Random Forest        | `RandomForestClassifier` | `sklearn.ensemble`      |
| (7) K-NN                 | `KNeighborsClassifier`   | `sklearn.neighbors`     |
| (8) SVM                  | `SVC`                    | `sklearn.svm`           |

</details>

---

### 문제 23. 전처리 관련 sklearn 클래스

> 다음 각 전처리 작업에 사용하는 sklearn 클래스를 쓰시오.
>
> (1) 결측값을 평균으로 대체
> (2) 순서가 없는 범주형 변수를 이진 벡터로 변환
> (3) 순서가 있는 범주형 변수를 정수로 변환
> (4) Min-Max 정규화 (0~1 범위)
> (5) 표준화 (평균 0, 분산 1)
> (6) 타겟 라벨을 정수로 변환
> (7) PCA 차원 축소

<details>
<summary>정답</summary>

| 작업                 | sklearn 클래스                   | 모듈                    |
| -------------------- | -------------------------------- | ----------------------- |
| (1) 결측값 대체      | `SimpleImputer(strategy='mean')` | `sklearn.impute`        |
| (2) One-hot Encoding | `OneHotEncoder()`                | `sklearn.preprocessing` |
| (3) Ordinal Encoding | `OrdinalEncoder()`               | `sklearn.preprocessing` |
| (4) Min-Max 정규화   | `MinMaxScaler()`                 | `sklearn.preprocessing` |
| (5) 표준화           | `StandardScaler()`               | `sklearn.preprocessing` |
| (6) 라벨 인코딩      | `LabelEncoder()`                 | `sklearn.preprocessing` |
| (7) PCA              | `PCA(n_components=k)`            | `sklearn.decomposition` |

</details>

---

### 문제 24. LogisticRegression의 파라미터 C

> sklearn의 `LogisticRegression(C=0.01)`에서 C의 의미를 설명하시오. C가 작아지면 모델에 어떤 영향이 있는지, 그리고 C와 정규화 파라미터 λ의 관계를 서술하시오.

<details>
<summary>정답</summary>

**C = 1/λ** 관계이다.

- **C가 작으면** → λ가 크다 → **정규화가 강하다** → 가중치가 더 많이 줄어든다 → 모델이 단순해진다 → **underfitting 위험**
- **C가 크면** → λ가 작다 → **정규화가 약하다** → 가중치에 대한 제약이 적다 → 모델이 복잡해질 수 있다 → **overfitting 위험**

따라서 C = 0.01은 매우 강한 정규화를 적용하는 것이며, 모델이 과소적합될 가능성이 있다. 적절한 C 값은 cross validation 등으로 튜닝해야 한다.

</details>

---

### 문제 25. SVC의 주요 하이퍼파라미터

> `SVC(kernel='rbf', C=1.0, gamma=0.1)`에서 각 파라미터의 의미를 설명하시오.

<details>
<summary>정답</summary>

- **kernel='rbf'**: RBF(Gaussian) 커널을 사용한다. 비선형 결정 경계를 만들 수 있으며, 데이터를 무한 차원으로 매핑하는 효과가 있다.

- **C=1.0**: 정규화 파라미터이다. C가 크면 오분류에 대한 패널티가 커서 마진이 좁아지고(과적합 위험), C가 작으면 마진이 넓어지고 일부 오분류를 허용한다(과소적합 위험).

- **gamma=0.1**: RBF 커널의 범위를 결정하는 계수로, σ와 관련된다 (gamma = 1/(2σ²)). gamma가 크면 하나의 학습 데이터의 영향 범위가 좁아져서 결정 경계가 복잡해지고(과적합 위험), gamma가 작으면 영향 범위가 넓어져서 결정 경계가 부드러워진다.

</details>

---

## Part 4: 종합 · 연결 문제

---

### 문제 26. Linear Regression과 Logistic Regression의 Gradient 비교

> Linear Regression과 Logistic Regression의 gradient 식이 형태상 동일하게 ∂J/∂wⱼ = (1/m) Σ (ŷ − y)·xⱼ로 나온다. 그런데 왜 이 둘은 서로 다른 알고리즘인지 설명하시오.

<details>
<summary>정답</summary>

gradient 식의 **형태는 동일**하지만, **ŷ의 정의가 다르다**:

- **Linear Regression**: ŷ = wx + b (선형 함수, 연속 실수값)
- **Logistic Regression**: ŷ = σ(wx + b) = 1/(1+e⁻⁽ʷˣ⁺ᵇ⁾) (sigmoid 적용, 0~1 확률값)

이 차이 때문에:

1. **모델의 출력**: 선형 회귀는 연속값 예측(회귀), 로지스틱 회귀는 확률 예측(분류)
2. **비용 함수**: 선형 회귀는 MSE, 로지스틱 회귀는 BCE를 사용
3. **비용 함수의 볼록성**: BCE + sigmoid는 convex, MSE + sigmoid는 non-convex
4. **Gradient의 행동**: 같은 형태이지만 ŷ 값의 범위가 다르므로 실제 gradient 값이 다르게 계산된다

형태가 같은 이유는 BCE를 sigmoid와 함께 미분(chain rule)하면 sigmoid의 미분 ŷ(1−ŷ)가 깔끔하게 상쇄되기 때문이다.

</details>

---

### 문제 27. 전체 ML 파이프라인 설계

> 주어진 데이터에 다음 특성이 있다:
>
> - Feature 10개 (일부 범주형, 일부 수치형, 결측값 존재)
> - 타겟: 이진 분류 (0 또는 1)
> - 데이터 1000개
>
> 전처리부터 모델 학습, 평가까지의 전체 파이프라인을 순서대로 설명하시오. 각 단계에서 사용할 수 있는 sklearn 클래스도 함께 제시하시오.

<details>
<summary>정답</summary>

**1. 데이터 분할**

- 훈련/테스트 분리: `train_test_split(X, y, test_size=0.2)`

**2. 결측값 처리**

- 수치형: `SimpleImputer(strategy='mean')`
- 범주형: `SimpleImputer(strategy='most_frequent')`

**3. 범주형 데이터 인코딩**

- Ordinal feature: `OrdinalEncoder()`
- Nominal feature: `OneHotEncoder()`

**4. 수치 데이터 스케일링**

- `StandardScaler()` (Logistic Regression, SVM 등에 특히 중요)
- fit()은 반드시 훈련 데이터로만!

**5. (선택) Feature Selection / 차원 축소**

- `PCA(n_components=k)` 또는
- L1 정규화 기반 feature selection

**6. 모델 학습**

- `LogisticRegression()`, `SVC()`, `RandomForestClassifier()` 등 적합한 모델 선택
- `model.fit(X_train, y_train)`

**7. 평가**

- `model.predict(X_test)`로 예측
- `accuracy_score(y_test, y_pred)`로 정확도 측정
- 또는 K-fold Cross Validation: `cross_val_score(model, X, y, cv=5)`

</details>

---

### 문제 28. Gradient Descent의 Learning Rate

> Learning rate α가 너무 크거나 너무 작을 때 각각 어떤 문제가 발생하는지 설명하고, 그래프의 형태를 서술하시오.

<details>
<summary>정답</summary>

**α가 너무 작을 때**:

- 비용이 매우 천천히 감소한다
- 수렴까지 **반복 횟수가 매우 많이** 필요하다
- 그래프: 비용이 매우 완만하게 내려가는 형태

**α가 너무 클 때**:

- 최적점을 지나쳐서(overshoot) **발산**할 수 있다
- 비용이 줄었다 늘었다를 반복하며 수렴하지 않는다
- 그래프: 비용이 위아래로 진동하거나, 점점 커지는 형태

**적절한 α**:

- 비용이 빠르게 감소하다가 수렴하는 형태
- 일반적으로 0.001, 0.01, 0.1 등을 시도하며 튜닝한다

</details>

---

### 문제 29. Random Forest가 Decision Tree보다 좋은 이유

> Decision Tree 하나 대신 Random Forest를 사용하면 어떤 이점이 있는지 설명하시오. Bootstrap과 feature random selection의 역할을 포함하여 서술하시오.

<details>
<summary>정답</summary>

Decision Tree는 **불안정**하다는 단점이 있다. 데이터가 조금만 바뀌어도 트리 구조가 크게 변할 수 있고, 과적합되기 쉽다.

Random Forest는 **K개의 Decision Tree를 앙상블**하여 이 문제를 해결한다:

1. **Bootstrap (복원 추출)**: 원본 데이터에서 복원 추출로 K개의 서로 다른 훈련 세트를 만든다. 각 트리가 서로 다른 데이터로 학습하므로 다양성이 생긴다.

2. **Feature Random Selection**: 각 분할에서 전체 feature 중 **랜덤하게 선택한 일부**(보통 √n개)만 후보로 사용한다. 이렇게 하면 트리들이 서로 다른 feature에 의존하게 되어 상관관계가 줄어든다.

3. **다수결 (Majority Voting)**: K개 트리의 예측을 종합하여 다수결로 최종 예측을 결정한다.

결과적으로 개별 트리의 오류가 서로 상쇄되어 **과적합에 강하고, 안정적인 예측**이 가능하다.

</details>

---

### 문제 30. SVM의 마진과 Support Vector

> SVM에서 마진(margin)의 정의와 마진을 최대화하는 이유를 설명하시오. Support Vector란 무엇이며, 왜 중요한가?

<details>
<summary>정답</summary>

**마진**: Decision boundary(wᵀx + b = 0)와 가장 가까운 데이터 포인트 사이의 거리의 2배이다. 양쪽 경계 wᵀx + b = +1과 wᵀx + b = −1 사이의 거리로, **M = 2/‖w‖**이다.

**마진을 최대화하는 이유**: 마진이 클수록 결정 경계가 두 클래스 사이의 "가운데"에 위치하게 되어, **새로운 데이터에 대한 일반화 성능이 좋아진다**. 마진이 좁으면 약간의 노이즈에도 오분류가 발생할 수 있다.

**Support Vector**: 마진 경계(wᵀx + b = ±1) 위에 놓이는 데이터 포인트들이다. 이들이 중요한 이유:

1. **결정 경계를 결정**한다 — support vector만 바뀌면 경계가 바뀌고, 나머지 데이터는 영향을 주지 않는다.
2. 마진을 최대화하는 최적화는 **‖w‖²를 최소화**하는 것과 동치이며, 제약 조건이 모든 데이터가 y⁽ⁱ⁾(wᵀx⁽ⁱ⁾ + b) ≥ 1을 만족하는 것이다.

</details>
