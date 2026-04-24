# 머신러닝 전처리 · Feature Selection · PCA · LDA 정리

이 문서는 지금까지 대화에서 헷갈렸던 부분과, 수업 슬라이드에 나온 `sklearn` 클래스/import 코드를 함께 정리한 파일이다.

---

## 1. 전체 흐름 정리

지금까지 배운 내용은 크게 다음과 같이 나눌 수 있다.

```text
데이터 준비
↓
결측값 처리
↓
범주형 데이터 인코딩
↓
숫자형 데이터 스케일링
↓
Feature Selection 또는 Dimensionality Reduction
↓
모델 학습
```

중요한 점은 **모든 전처리를 항상 다 하는 것은 아니다**는 것이다. 데이터 상태에 따라 필요한 전처리만 선택해서 사용한다.

---

## 2. fit(), transform(), fit_transform()

### 핵심 개념

| 함수 | 의미 |
|---|---|
| `fit()` | 전처리 기준을 학습한다. 예: 평균, 표준편차, 최솟값, 최댓값, PCA 축 |
| `transform()` | `fit()`에서 학습한 기준으로 실제 값을 변환한다. |
| `fit_transform()` | `fit()`과 `transform()`을 한 번에 수행한다. |

### 가장 중요한 규칙

```python
X_train_processed = transformer.fit_transform(X_train)
X_test_processed = transformer.transform(X_test)
```

`fit()`은 **training data에만** 사용한다.

왜냐하면 test data는 실제 미래 데이터처럼 봐야 하기 때문이다. test data에 `fit()`을 해버리면 test data의 평균, 표준편차, PCA 축 같은 정보를 미리 보는 것이므로 **data leakage**가 발생한다.

### 잘못된 예

```python
X_test_processed = transformer.fit_transform(X_test)  # 잘못된 방식
```

---

## 3. 결측값 처리: SimpleImputer

결측값 `np.nan`을 평균, 중앙값, 최빈값 등으로 채우는 전처리 방법이다.

### Import

```python
import numpy as np
from sklearn.impute import SimpleImputer
```

### 사용 예

```python
imr = SimpleImputer(missing_values=np.nan, strategy='mean')

X_train_imputed = imr.fit_transform(X_train)
X_test_imputed = imr.transform(X_test)
```

### 의미

```python
SimpleImputer(missing_values=np.nan, strategy='mean')
```

뜻:

```text
np.nan으로 표시된 결측값을 각 열의 평균값으로 대체한다.
```

예를 들어 어떤 열의 값이 다음과 같다면:

```text
10, 20, nan, 20, nan
```

평균은:

```text
(10 + 20 + 20) / 3 = 16.6667
```

따라서 `nan`은 모두 `16.6667`로 바뀐다.

---

## 4. Class Label Encoding

앞 슬라이드에서 헷갈렸던 부분은 **pandas 방식과 scikit-learn 방식은 동시에 쓰는 것이 아니라 둘 중 하나를 선택한다**는 점이다.

### 대상

`classlabel`은 보통 정답 label `y`이다.

예:

```text
class1, class2, class3
```

이를 숫자로 바꾼다.

```text
class1 → 0
class2 → 1
class3 → 2
```

---

### 방법 1: pandas `.map()`

```python
label_mapping = {'class1': 0, 'class2': 1, 'class3': 2}
pdf['classlabel'] = pdf['classlabel'].map(label_mapping)
```

장점: 내가 원하는 숫자 매핑을 직접 정할 수 있다.

---

### 방법 2: scikit-learn `LabelEncoder`

#### Import

```python
from sklearn.preprocessing import LabelEncoder
```

#### 사용 예

```python
enc = LabelEncoder()
sdf['classlabel'] = enc.fit_transform(sdf['classlabel'])
```

주의: `LabelEncoder`는 주로 **target label y**에 사용한다.

---

## 5. Feature Encoding: Ordinal Feature vs Nominal Feature

범주형 feature는 크게 두 종류가 있다.

| 종류 | 의미 | 예시 | 인코딩 방식 |
|---|---|---|---|
| Ordinal feature | 순서가 있음 | `M < L < XL` | Ordinal Encoding |
| Nominal feature | 순서가 없음 | `red`, `green`, `blue` | One-Hot Encoding |

---

## 6. Ordinal Encoding

### 예시

`size`는 순서가 있는 feature이다.

```text
M < L < XL
```

따라서 숫자로 바꿀 때도 순서를 반영해야 한다.

```text
M  → 0
L  → 1
XL → 2
```

잘못 바꾸면 모델이 순서를 잘못 이해한다.

예를 들어:

```text
M → 1
L → 0
XL → 2
```

이렇게 하면 모델은 `L < M < XL`처럼 오해할 수 있다.

---

### 방법 1: pandas `.map()`

```python
size_mapping = {'M': 0, 'L': 1, 'XL': 2}
pdf['size'] = pdf['size'].map(size_mapping)
```

---

### 방법 2: scikit-learn `OrdinalEncoder`

#### Import

```python
from sklearn.preprocessing import OrdinalEncoder
```

#### 사용 예

```python
enc = OrdinalEncoder(categories=[['M', 'L', 'XL']])
sdf[['size']] = enc.fit_transform(sdf[['size']])
```

### 왜 `sdf[['size']]`처럼 대괄호를 두 번 쓰나?

```python
sdf['size']      # 1차원 Series
sdf[['size']]    # 2차원 DataFrame
```

`sklearn`의 encoder는 보통 2차원 입력을 기대하기 때문에 `sdf[['size']]`를 사용한다.

---

## 7. One-Hot Encoding

`color`처럼 순서가 없는 nominal feature는 단순히 `0, 1, 2`로 바꾸면 안 된다.

예를 들어:

```text
green → 0
red   → 1
blue  → 2
```

이렇게 하면 모델은 색깔 사이에 크기나 순서가 있다고 오해할 수 있다.

따라서 one-hot encoding을 사용한다.

### 예시

| color | color_blue | color_green | color_red |
|---|---:|---:|---:|
| green | 0 | 1 | 0 |
| red | 0 | 0 | 1 |
| blue | 1 | 0 | 0 |

---

### 방법 1: pandas `get_dummies()`

```python
pd.get_dummies(pdf, columns=['color'])
```

#### Import

```python
import pandas as pd
```

---

### 방법 2: scikit-learn `OneHotEncoder`

#### Import

```python
from sklearn.preprocessing import OneHotEncoder
```

#### 사용 예

```python
enc = OneHotEncoder(sparse=False)

color_enc = enc.fit_transform(sdf[['color']])
sdf_color_enc = pd.DataFrame(color_enc, columns=['color0', 'color1', 'color2'])
sdf = pd.concat([sdf_color_enc, sdf[['size', 'price', 'classlabel']]], axis=1)
```

주의: 최신 scikit-learn에서는 `sparse=False` 대신 `sparse_output=False`를 쓰는 경우가 있다.

```python
enc = OneHotEncoder(sparse_output=False)
```

버전마다 다를 수 있으므로 실행 환경에 따라 확인해야 한다.

---

## 8. Numerical Feature Transformation

숫자형 feature들은 scale을 맞춰야 할 때가 많다.

예:

| feature | 값 범위 |
|---|---:|
| 나이 | 20~70 |
| 소득 | 3000~10000 |
| 시험 점수 | 0~100 |

값의 범위가 너무 다르면 큰 숫자 범위를 가진 feature가 모델에 더 큰 영향을 줄 수 있다.

---

## 9. Normalization

Normalization은 값을 `[0, 1]` 범위로 바꾸는 방법이다.

공식:

```text
x_new = (x - x_min) / (x_max - x_min)
```

### Import

```python
from sklearn.preprocessing import MinMaxScaler
```

### 사용 예

```python
scaler = MinMaxScaler()

X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)
```

### 특징

```text
최솟값 → 0
최댓값 → 1
나머지 값 → 0과 1 사이
```

---

## 10. Standardization

Standardization은 평균을 0, 표준편차를 1로 바꾸는 방법이다.

공식:

```text
x_new = (x - μ) / σ
```

### Import

```python
from sklearn.preprocessing import StandardScaler
```

### 사용 예

```python
scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)
```

### 특징

```text
평균 = 0
표준편차 = 1
값의 범위가 반드시 0~1은 아님
```

---

## 11. 왜 자꾸 `X_train_std`를 쓰는가?

`X_train_std`는 training data를 standardization한 데이터이다.

PCA, LDA, Logistic Regression + regularization 등에서는 `X_train_std`를 자주 사용한다.

이유:

```text
feature scale이 다르면 weight, 거리, 분산 계산이 공정하지 않기 때문이다.
```

특히 PCA는 **분산이 큰 방향**을 찾는 알고리즘이다. 표준화하지 않으면 단위가 큰 feature가 PCA 결과를 지배할 수 있다.

예:

| feature | 값 범위 |
|---|---:|
| Alcohol | 11~14 |
| Proline | 200~1600 |

표준화하지 않으면 `Proline`의 분산이 훨씬 커져서 PCA가 Proline 중심으로 축을 잡을 수 있다.

따라서 PCA 전에는 보통 다음처럼 한다.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)
```

---

## 12. Feature Selection 개념

Feature Selection은 기존 feature 중에서 중요한 feature만 선택하는 것이다.

```text
기존 feature를 그대로 유지하면서 일부만 고른다.
```

예:

```text
원래 feature: Alcohol, Malic acid, Ash, Flavanoids, Proline
선택 후: Flavanoids, Proline
```

Feature Selection은 PCA처럼 새 feature를 만드는 것이 아니다.

---

## 13. Decision Tree의 `feature_importances_`

Decision Tree는 학습 후 각 feature의 중요도를 제공한다.

### Import

```python
from sklearn.tree import DecisionTreeClassifier
```

### 사용 예

```python
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)

importances = tree.feature_importances_
```

### 의미

`feature_importances_`는 각 feature가 tree 분할 과정에서 impurity를 얼마나 줄였는지를 나타낸다.

중요한 feature일수록 entropy 또는 gini 같은 criterion을 많이 감소시킨다.

---

## 14. Information Gain과 Gini 감소량

헷갈렸던 부분:

```text
Information Gain = 나누기 전 불순도 - 나눈 후 자식 노드들의 가중 평균 불순도
```

이 구조는 Gini에서도 똑같다.

### Entropy를 쓰면

```text
IG = Entropy(parent) - weighted average Entropy(children)
```

### Gini를 쓰면

```text
Gini gain = Gini(parent) - weighted average Gini(children)
```

즉, 식의 구조는 같고 **불순도를 무엇으로 계산하느냐**가 다르다.

---

## 15. Feature Importance 코드 해석

```python
feature_labels = df_wine.columns[1:]
importances = tree.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-30s %f" %
          (f + 1, feature_labels[indices[f]], importances[indices[f]]))
```

### Import

```python
import numpy as np
```

### 코드 의미

```python
feature_labels = df_wine.columns[1:]
```

첫 번째 column이 class label이면 제외하고 feature 이름만 가져온다.

```python
importances = tree.feature_importances_
```

각 feature의 중요도를 가져온다.

```python
indices = np.argsort(importances)[::-1]
```

중요도 높은 순서대로 index를 정렬한다.

---

## 16. Regularization으로 Feature Selection

Regularization은 모델 복잡도를 줄이기 위해 weight에 penalty를 주는 방법이다.

### L2 Regularization

```text
L2 = Σ w_j²
```

특징:

```text
큰 weight를 강하게 줄인다.
대부분의 weight를 작게 만든다.
하지만 보통 정확히 0으로 만들지는 않는다.
```

### L1 Regularization

```text
L1 = Σ |w_j|
```

특징:

```text
일부 weight를 정확히 0으로 만들 수 있다.
weight가 0인 feature는 예측에 사용되지 않는다.
따라서 feature selection 효과가 있다.
```

---

## 17. Logistic Regression with L1 Regularization

### Import

```python
from sklearn.linear_model import LogisticRegression
```

### 사용 예

```python
lr = LogisticRegression(penalty='l1', C=1, solver='liblinear')
lr.fit(X_train_std, y_train)

print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))

print(lr.coef_)
```

### 중요 파라미터

| 파라미터 | 의미 |
|---|---|
| `penalty='l1'` | L1 regularization 사용 |
| `C=1` | 정규화 강도 조절. `C`가 작을수록 정규화가 강함 |
| `solver='liblinear'` | L1 penalty를 지원하는 solver |

### `coef_` 의미

`lr.coef_`는 학습된 weight이다.

```text
weight가 0이면 해당 feature는 예측에 사용되지 않는다.
```

따라서 L1에서는 `coef_`를 보고 어떤 feature가 선택되었는지 확인할 수 있다.

---

## 18. Sequential Feature Selection

Sequential Feature Selection은 feature를 하나씩 추가하거나 제거하면서 좋은 feature 조합을 찾는 방법이다.

슬라이드에서는 **Sequential Backward Selection(SBS)** 를 다뤘다.

### SBS 핵심

```text
처음에는 모든 feature를 사용한다.
성능이 가장 덜 떨어지는 feature를 하나씩 제거한다.
원하는 feature 개수가 될 때까지 반복한다.
```

### 관련 sklearn 클래스

```python
from sklearn.feature_selection import SequentialFeatureSelector
```

### 사용 예

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector

knn = KNeighborsClassifier(n_neighbors=5)

sfs = SequentialFeatureSelector(
    knn,
    n_features_to_select=5,
    direction='backward'
)

sfs.fit(X_train_std, y_train)
X_train_sfs = sfs.transform(X_train_std)
X_test_sfs = sfs.transform(X_test_std)
```

주의: 이 슬라이드는 `sklearn.preprocessing`이나 `sklearn.decomposition`이 아니라 `sklearn.feature_selection`에 가까운 내용이다.

---

## 19. Feature Selection vs Dimensionality Reduction

| 구분 | Feature Selection | Dimensionality Reduction / Decomposition |
|---|---|---|
| 방식 | 기존 feature 중 일부 선택 | 기존 feature들을 조합해 새 feature 생성 |
| 결과 | 원래 feature 이름 유지 | PC1, PC2 같은 새 feature 생성 |
| 예시 | Proline 선택 | PC1 = 0.3Alcohol + 0.5Proline + ... |
| 해석 | 쉬움 | 상대적으로 어려움 |
| sklearn 모듈 | `sklearn.feature_selection` | `sklearn.decomposition` |

---

## 20. PCA 개념

PCA는 Principal Component Analysis, 즉 주성분 분석이다.

핵심:

```text
데이터가 가장 많이 퍼져 있는 방향을 새 축으로 잡는다.
그 방향이 PC1이다.
PC1과 수직이면서 남은 방향 중 분산이 가장 큰 방향이 PC2이다.
```

즉:

```text
PC1 = 분산이 가장 큰 방향
PC2 = PC1과 수직이고, 그다음으로 분산이 큰 방향
PC3 = PC1, PC2와 수직이고, 그다음으로 분산이 큰 방향
```

PCA는 기존 feature를 선택하는 것이 아니라 기존 feature들을 선형 조합해서 새로운 feature를 만든다.

---

## 21. 공분산과 PCA

공분산은 두 변수가 같이 어떻게 변하는지를 나타낸다.

```text
cov(x, y) > 0  → x가 커질 때 y도 커지는 경향
cov(x, y) < 0  → x가 커질 때 y는 작아지는 경향
cov(x, y) ≈ 0  → 선형 관계가 약함
```

공분산 행렬은 feature들 사이의 분산과 공분산을 모아놓은 행렬이다.

PCA는 공분산 행렬을 사용해서 데이터가 어느 방향으로 가장 많이 퍼져 있는지 찾는다.

---

## 22. 양의 상관관계와 PCA

양의 상관관계는 `x1`이 커질수록 `x2`도 커지는 경향을 말한다.

PCA 입장에서는:

```text
두 feature가 비슷한 정보를 담고 있다.
하나의 대각선 방향 PC1으로 압축하기 쉽다.
```

중요한 것은 양수냐 음수냐보다 **상관관계가 강한지**이다.

---

## 23. 음의 상관관계와 PCA

음의 상관관계는 `x1`이 커질수록 `x2`는 작아지는 경향을 말한다.

그래프에서는 반대 대각선 방향으로 길게 퍼진다.

PCA는 그 방향을 PC1으로 잡을 수 있다.

정리:

```text
양의 상관관계 → / 방향으로 길게 퍼짐
음의 상관관계 → \ 방향으로 길게 퍼짐
```

둘 다 강한 선형 관계라면 PCA로 압축하기 좋다.

---

## 24. Eigenvector와 Eigenvalue

PCA에서 가장 중요한 수학 개념이다.

```text
A x = λ x
```

| 개념 | 의미 |
|---|---|
| Eigenvector | 변환 후에도 방향이 바뀌지 않는 벡터 |
| Eigenvalue | 그 벡터의 길이가 몇 배 변하는지 나타내는 값 |

PCA에서는 `A`가 공분산 행렬이다.

따라서:

| PCA에서 | 의미 |
|---|---|
| Eigenvector | PCA의 새로운 축, principal component |
| Eigenvalue | 그 축 방향의 분산 크기 |

가장 큰 eigenvalue를 가진 eigenvector가 PC1이다.

---

## 25. PCA Algorithm

PCA의 전체 흐름은 다음과 같다.

```text
1. d차원 데이터를 표준화한다.
2. 공분산 행렬 Σ를 만든다. 크기는 d × d.
3. 공분산 행렬의 eigenvector와 eigenvalue를 구한다.
4. eigenvalue가 큰 순서대로 eigenvector를 정렬한다.
5. 상위 k개의 eigenvector를 선택한다.
6. 선택한 eigenvector들로 projection matrix W를 만든다.
7. X' = XW를 계산해서 d차원 데이터를 k차원 데이터로 변환한다.
```

---

## 26. PCA using NumPy

### Import

```python
import numpy as np
```

### 표준화된 데이터 확인

```python
print(X_train_std.shape)
print(X_train_std[:3])
```

예:

```text
(124, 13)
```

뜻:

```text
124개 training sample, 13개 feature
```

---

### 공분산 행렬 계산

```python
cov_mat = np.cov(X_train_std.T)
print(cov_mat.shape)
print(cov_mat)
```

왜 `.T`를 쓰는가?

```text
X_train_std.shape = (124, 13)
X_train_std.T.shape = (13, 124)
```

`np.cov()`는 기본적으로 row를 변수로 보기 때문에 feature가 row에 오도록 transpose한다.

결과:

```text
cov_mat.shape = (13, 13)
```

feature가 13개이므로 공분산 행렬은 13 × 13이다.

---

### Eigenvalue, Eigenvector 계산

```python
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
```

의미:

```text
eigen_vals = 각 principal component가 설명하는 분산량
eigen_vecs = principal component의 방향
```

대응 관계:

```python
eigen_vals[0]  # 첫 번째 eigenvalue
eigen_vecs[:, 0]  # 첫 번째 eigenvalue에 대응하는 eigenvector
```

---

## 27. PCA using scikit-learn

### Import

```python
from sklearn.decomposition import PCA
```

### 사용 예

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
```

주의:

```text
PCA도 train data에서만 fit한다.
test data에는 transform만 한다.
```

---

## 28. LDA 개념

LDA는 Linear Discriminant Analysis이다.

PCA와 비슷하게 차원 축소를 할 수 있지만 기준이 다르다.

| 구분 | PCA | LDA |
|---|---|---|
| 목적 | 분산을 최대한 보존 | class를 최대한 잘 분리 |
| label 사용 | 사용 안 함 | 사용함 |
| 학습 방식 | 비지도 학습 | 지도 학습 |
| 찾는 축 | 분산이 큰 방향 | class 구분이 잘 되는 방향 |

---

## 29. LDA의 목표

LDA는 다음을 동시에 원한다.

```text
Between-class scatter는 크게
Within-class scatter는 작게
```

즉:

```text
다른 class끼리는 멀리 떨어지게 한다.
같은 class끼리는 가깝게 모이게 한다.
```

LDA가 최대화하는 식:

```text
J(w) = Between class distance / Within class variance
     = (w^T S_B w) / (w^T S_W w)
```

| 기호 | 의미 |
|---|---|
| `S_B` | between-class scatter matrix |
| `S_W` | within-class scatter matrix |
| `w` | projection 방향 |

---

## 30. LDA의 가정

슬라이드에 나온 LDA의 가정:

```text
1. 각 class는 정규분포를 따른다.
2. 각 class는 같은 covariance structure를 가진다.
```

즉, class별 데이터가 비슷한 모양의 타원 분포를 가진다고 보는 것이다.

---

## 31. LDA using scikit-learn

### Import

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
```

또는 별칭 사용:

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
```

### 사용 예

```python
lda = LDA(n_components=2)

X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)
```

PCA와 다른 점:

```python
pca.fit_transform(X_train_std)        # y 필요 없음
lda.fit_transform(X_train_std, y_train)  # y 필요함
```

LDA는 class label을 보고 class가 잘 나뉘는 방향을 찾기 때문이다.

---

## 32. 주요 sklearn import 모음

### 전처리

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
```

### Feature Selection

```python
from sklearn.feature_selection import SequentialFeatureSelector
```

### Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier
```

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
```

### PCA

```python
from sklearn.decomposition import PCA
```

### LDA

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
```

### 기타 자주 쓰는 것

```python
import numpy as np
import pandas as pd
```

---

## 33. 최종 암기 포인트

```text
fit()은 train data에서만 한다.
transform()은 train/test 모두에 적용한다.
```

```text
LabelEncoder는 주로 y label에 사용한다.
OrdinalEncoder는 순서 있는 feature에 사용한다.
OneHotEncoder는 순서 없는 feature에 사용한다.
```

```text
Normalization은 값을 [0, 1]로 만든다.
Standardization은 평균 0, 표준편차 1로 만든다.
```

```text
Feature Selection은 기존 feature 중 일부를 고르는 것이다.
PCA는 기존 feature를 조합해서 새 feature를 만드는 것이다.
```

```text
Decision Tree의 feature_importances_는 impurity 감소량을 기반으로 feature 중요도를 계산한다.
```

```text
L1 regularization은 일부 weight를 0으로 만들어 feature selection 효과가 있다.
L2 regularization은 weight를 전체적으로 작게 만들지만 보통 0으로 만들지는 않는다.
```

```text
PCA는 label 없이 분산이 큰 방향을 찾는다.
LDA는 label을 사용해서 class가 잘 분리되는 방향을 찾는다.
```

```text
PCA의 eigenvector는 principal component이고,
eigenvalue는 그 principal component가 설명하는 분산의 양이다.
```

---

## 34. 시험에서 자주 물을 수 있는 비교

### PCA vs LDA

| 구분 | PCA | LDA |
|---|---|---|
| 종류 | 비지도 차원 축소 | 지도 차원 축소 |
| 사용 데이터 | X만 사용 | X와 y 사용 |
| 목표 | 분산 최대 보존 | class 분리 최대화 |
| 축 이름 | PC1, PC2 | LD1, LD2 |
| sklearn import | `from sklearn.decomposition import PCA` | `from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA` |

### Feature Selection vs PCA

| 구분 | Feature Selection | PCA |
|---|---|---|
| 결과 | 기존 feature 일부 선택 | 새로운 feature 생성 |
| feature 이름 | 유지됨 | PC1, PC2로 바뀜 |
| 예시 | Proline 선택 | PC1 = 여러 feature의 선형 조합 |

### L1 vs L2

| 구분 | L1 | L2 |
|---|---|---|
| penalty | `Σ |w_j|` | `Σ w_j²` |
| 효과 | 일부 weight를 0으로 만듦 | weight를 작게 만듦 |
| feature selection | 가능 | 직접적 feature 제거는 약함 |

---
