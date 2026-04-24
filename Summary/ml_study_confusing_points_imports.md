# 머신러닝 헷갈렸던 부분 + scikit-learn import 정리

## 1. Pipeline

### 핵심 개념

Pipeline은 여러 전처리 단계와 최종 모델을 하나로 묶는 방법이다.

예시:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

pipe_lr = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    LogisticRegression(random_state=1)
)
```

순서는 다음과 같다.

```text
StandardScaler → PCA → LogisticRegression
```

### 왜 LogisticRegression은 fit만 하는가?

`StandardScaler`와 `PCA`는 transformer이다.

```text
StandardScaler: X를 표준화된 X로 변환
PCA: X를 차원 축소된 X로 변환
```

그래서 학습할 때는 내부적으로 다음처럼 동작한다.

```python
X_train_scaled = scaler.fit_transform(X_train)
X_train_pca = pca.fit_transform(X_train_scaled)
lr.fit(X_train_pca, y_train)
```

반면 `LogisticRegression`은 최종 estimator이다. 데이터를 변환하는 역할이 아니라, 변환된 `X_train`과 정답 `y_train`을 보고 분류 규칙을 학습한다.

```text
앞 단계 transformer: fit_transform
마지막 estimator: fit
```

### predict할 때는 왜 transform만 하는가?

```python
y_pred = pipe_lr.predict(X_test)
```

이 코드는 내부적으로 다음처럼 작동한다.

```python
X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)
y_pred = lr.predict(X_test_pca)
```

테스트 데이터에서는 `fit`을 다시 하면 안 된다. 이미 훈련 데이터에서 배운 평균, 표준편차, PCA 축을 그대로 사용해야 한다.

### Pipeline을 써도 import는 필요하다

`make_pipeline`은 여러 단계를 묶는 도구일 뿐이다. 안에 넣는 부품들은 각각 import해야 한다.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
```

---

## 2. Stratified K-fold Cross Validation

### 핵심 개념

Stratified K-fold는 각 fold 안에서 class 비율이 원본 데이터와 비슷하게 유지되도록 나누는 방법이다.

예를 들어 원본 데이터에 class 0과 class 1이 7:3 비율이면, 각 fold에서도 최대한 7:3 비율이 유지된다.

### 직접 사용하는 코드

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(
    n_splits=5,
    shuffle=False
).split(X_train, y_train)

scores = []

for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)

    print(
        f"Fold: {k+1}, "
        f"Class dist.: {np.bincount(y_train[train])}, "
        f"Acc: {score:.3f}"
    )

print(f"CV accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
```

### 필요한 import

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
```

---

## 3. cross_val_score

### 핵심 개념

`cross_val_score()`는 K-fold 과정을 자동으로 해주는 함수이다.

직접 for문으로 fold를 나누고, fit하고, score를 계산하지 않아도 된다.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    estimator=pipe_lr,
    X=X_train,
    y=y_train,
    cv=5,
    n_jobs=1
)
```

내부적으로는 다음과 같은 일을 한다.

```text
1번째 fold로 fit 후 score 계산
2번째 fold로 fit 후 score 계산
...
5번째 fold로 fit 후 score 계산
```

분류 문제에서 `cv=5`를 주면 보통 내부적으로 stratified 방식으로 fold를 나눈다.

### 필요한 import

```python
from sklearn.model_selection import cross_val_score
```

---

## 4. Bias와 Variance

## Bias

Bias는 예측값과 실제값 사이의 체계적인 차이이다.

```text
High bias = underfitting
```

특징:

```text
모델이 너무 단순함
training accuracy 낮음
validation accuracy 낮음
```

해결 방법:

```text
feature 추가
모델 복잡도 증가
regularization 감소
```

## Variance

Variance는 학습 데이터가 바뀔 때 예측값이 얼마나 흔들리는지이다.

```text
High variance = overfitting
```

특징:

```text
training accuracy 높음
validation accuracy 낮음
train-validation gap 큼
```

해결 방법:

```text
feature 감소
regularization 증가
데이터 추가
모델 단순화
```

---

## 5. Learning Curve

### 핵심 개념

Learning curve는 훈련 데이터 개수를 바꿔가며 training accuracy와 validation accuracy를 보는 그래프이다.

```text
x축: training samples 수
y축: accuracy
```

주로 bias와 variance를 진단할 때 사용한다.

### 코드

```python
import numpy as np
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    estimator=pipe_lr,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=10,
    n_jobs=1
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
```

### shape 이해

만약 `train_sizes`가 10개이고 `cv=10`이면:

```text
train_scores.shape = (10, 10)
```

의미:

```text
앞의 10 = training size 종류 10개
뒤의 10 = 10-fold cross validation
```

### test_scores의 의미

여기서 `test_scores`는 최종 test set 점수가 아니라 cross validation 안에서 validation fold에 대한 점수이다.

### 필요한 import

```python
import numpy as np
from sklearn.model_selection import learning_curve
```

---

## 6. Validation Curve

### 핵심 개념

Validation curve는 하이퍼파라미터 값을 바꿔가며 training accuracy와 validation accuracy를 보는 그래프이다.

```text
x축: hyperparameter 값
y축: accuracy
```

예를 들어 Logistic Regression의 `C` 값을 바꿔가며 성능 변화를 확인한다.

### 코드

```python
import numpy as np
from sklearn.model_selection import validation_curve

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

train_scores, test_scores = validation_curve(
    estimator=pipe_lr,
    X=X_train,
    y=y_train,
    param_name='logisticregression__C',
    param_range=param_range,
    cv=10
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
```

### `logisticregression__C`의 의미

Pipeline 안의 LogisticRegression 단계에 있는 `C` 값을 바꾸겠다는 뜻이다.

```text
step 이름__parameter 이름
```

예시:

```python
'logisticregression__C'
```

### C의 의미

`C`는 Logistic Regression의 하이퍼파라미터이고, 정규화 강도의 반대이다.

```text
C 작음 → 정규화 강함 → 모델 단순 → underfitting 가능
C 큼 → 정규화 약함 → 모델 복잡 → overfitting 가능
```

### shape 이해

`param_range`가 6개이고 `cv=10`이면:

```text
train_scores.shape = (6, 10)
```

의미:

```text
앞의 6 = C 값 6개
뒤의 10 = 10-fold cross validation
```

### 필요한 import

```python
import numpy as np
from sklearn.model_selection import validation_curve
```

---

## 7. Grid Search

### 핵심 개념

Grid Search는 여러 하이퍼파라미터 조합을 모두 실험해서 가장 좋은 조합을 찾는 방법이다.

예를 들어:

```python
C = [0.01, 0.1, 1.0, 10.0, 100.0]
penalty = ['l1', 'l2']
```

이면 가능한 조합은:

```text
5개 C 값 × 2개 penalty 값 = 10개 조합
```

각 조합에 대해 cross validation을 수행하고 평균 성능이 가장 좋은 조합을 선택한다.

### parameter와 hyperparameter 차이

Parameter:

```text
모델이 fit하면서 자동으로 학습하는 값
예: weight w, bias b
```

Hyperparameter:

```text
학습 전에 사람이 정하는 값
예: C, penalty, max_depth, n_estimators
```

### 코드

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

param_grid = [{
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'penalty': ['l1', 'l2']
}]

gs = GridSearchCV(
    estimator=LogisticRegression(solver='liblinear'),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=1
)

gs.fit(X, y)
```

### 결과 확인

```python
print(gs.best_score_)
print(gs.best_params_)
print(gs.best_estimator_)
```

### Pipeline과 같이 쓸 때

Pipeline 안의 LogisticRegression에 접근하려면 이름을 이렇게 써야 한다.

```python
param_grid = [{
    'logisticregression__C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'logisticregression__penalty': ['l1', 'l2']
}]
```

### 필요한 import

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
```

Pipeline과 함께 쓰면 추가로 필요할 수 있다.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```

---

## 8. ROC Curve와 AUC

### ROC 뜻

```text
ROC = Receiver Operating Characteristic
```

### 핵심 개념

ROC curve는 threshold를 바꿔가며 TPR과 FPR의 변화를 그린 그래프이다.

```text
x축 = FPR
y축 = TPR
```

좋은 모델일수록 ROC curve가 왼쪽 위에 가까워진다.

### TPR

```text
TPR = TP / (TP + FN)
```

실제 positive 중에서 모델이 positive라고 맞힌 비율이다.

### FPR

```text
FPR = FP / (FP + TN)
```

실제 negative 중에서 모델이 positive라고 잘못 예측한 비율이다.

### AUC

```text
AUC = Area Under the Curve
```

ROC curve 아래 면적이다.

```text
AUC = 1.0 → 완벽
AUC = 0.5 → 랜덤 찍기 수준
```

### 왜 완벽한 경우 ROC가 ㄱ자 모양인가?

Negative와 Positive가 완전히 분리되어 있으면 어떤 threshold에서:

```text
TPR = 1, FPR = 0
```

이 가능하다.

그래서 ROC curve가 왼쪽 위 꼭짓점을 지나고 AUC가 1이 된다.

### 왜 분포가 겹치면 ROC가 아래로 내려오는가?

Negative와 Positive가 겹치면 어떤 threshold를 잡아도 FP나 FN이 생긴다.

```text
TPR을 높이면 FPR도 올라감
FPR을 낮추면 TPR도 낮아짐
```

그래서 ROC curve가 왼쪽 위에 붙지 못하고 아래로 내려온다.

### 코드

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

pipe_lr.fit(X_train2, y_train)
probas = pipe_lr.predict_proba(X_test2)

fpr, tpr, thresholds = roc_curve(
    y_test,
    probas[:, 1],
    pos_label=1
)

roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--', label='random guessing')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc='lower right')
plt.show()
```

### `predict_proba()`와 `probas[:, 1]`

```python
probas = pipe_lr.predict_proba(X_test2)
```

이 코드는 각 클래스에 속할 확률을 반환한다.

이진 분류라면:

```text
probas[:, 0] = class 0일 확률
probas[:, 1] = class 1일 확률
```

`pos_label=1`이면 class 1을 positive로 보겠다는 뜻이므로 `probas[:, 1]`을 사용한다.

### 필요한 import

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
```

---

## 9. Ensemble Learning

### 핵심 개념

Ensemble Learning은 여러 모델을 합쳐서 하나의 모델보다 더 좋은 일반화 성능을 얻는 방법이다.

```text
여러 classifier → meta-classifier → better generalization
```

### Majority Voting

여러 classifier가 예측한 class 중 가장 많이 나온 class를 최종 예측으로 선택한다.

예시:

```text
모델 1 → A
모델 2 → A
모델 3 → B
최종 예측 → A
```

### Bagging

Bagging은 bootstrap sample을 이용해 여러 모델을 독립적으로 학습시키고, 그 결과를 voting 또는 averaging으로 합치는 방법이다.

```text
복원 추출로 여러 training set 생성
각 모델 독립적으로 학습
최종 결과 voting 또는 average
```

주로 variance 감소에 효과가 있다.

대표 예시:

```text
Random Forest
```

### Boosting

Boosting은 이전 모델이 틀린 데이터에 더 집중하면서 약한 모델을 순차적으로 추가하는 방법이다.

```text
앞 모델이 틀린 부분을 다음 모델이 보완
weak learner를 순차적으로 결합
```

대표 예시:

```text
AdaBoost
Gradient Boosting
```

### Random Forest도 Ensemble Learning인가?

그렇다. Random Forest는 Bagging 계열의 ensemble learning이다.

```text
Ensemble Learning
 ├─ Voting
 ├─ Bagging
 │   └─ Random Forest
 └─ Boosting
```

Random Forest는 여러 Decision Tree를 만들고, 분류에서는 majority voting으로 최종 예측한다.

---

## 10. BaggingClassifier

### 핵심 개념

`BaggingClassifier`는 같은 base estimator를 여러 개 만들고, 각 모델을 bootstrap sample로 학습시킨 후 결과를 합친다.

### 코드

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

# base model
tree = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,
    random_state=1
)

tree.fit(X_train, y_train)

# bagging model
bag = BaggingClassifier(
    estimator=tree,
    n_estimators=100,
    max_samples=0.5,
    max_features=1.0,
    n_jobs=1,
    random_state=1
)

bag.fit(X_train, y_train)

# prediction
tree_train_pred = tree.predict(X_train)
tree_test_pred = tree.predict(X_test)

bag_train_pred = bag.predict(X_train)
bag_test_pred = bag.predict(X_test)

# accuracy
tree_train_acc = accuracy_score(y_train, tree_train_pred)
tree_test_acc = accuracy_score(y_test, tree_test_pred)

bag_train_acc = accuracy_score(y_train, bag_train_pred)
bag_test_acc = accuracy_score(y_test, bag_test_pred)

print(f"Tree training/test accuracy: {tree_train_acc:.2f} / {tree_test_acc:.2f}")
print(f"Bag training/test accuracy: {bag_train_acc:.2f} / {bag_test_acc:.2f}")
```

### scikit-learn 버전 주의

예전 버전에서는 `estimator` 대신 `base_estimator`를 쓸 수 있다.

```python
bag = BaggingClassifier(
    base_estimator=tree,
    n_estimators=100,
    max_samples=0.5,
    max_features=1.0,
    n_jobs=1,
    random_state=1
)
```

### 주요 파라미터

```text
estimator 또는 base_estimator: 기본 모델
n_estimators: 만들 base model 개수
max_samples: 각 모델이 사용할 sample 비율
max_features: 각 모델이 사용할 feature 비율
```

### 필요한 import

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
```

---

## 11. AdaBoostClassifier

### 핵심 개념

AdaBoost는 boosting 방식의 ensemble이다.

이전 모델이 틀린 샘플에 더 집중해서 다음 weak learner를 학습한다.

### 코드

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# weak learner
tree = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=1,
    random_state=1
)

tree.fit(X_train, y_train)

# AdaBoost model
ada = AdaBoostClassifier(
    estimator=tree,
    n_estimators=100,
    learning_rate=0.1,
    random_state=1
)

ada.fit(X_train, y_train)
```

### scikit-learn 버전 주의

예전 버전에서는 `estimator` 대신 `base_estimator`를 쓸 수 있다.

```python
ada = AdaBoostClassifier(
    base_estimator=tree,
    n_estimators=100,
    learning_rate=0.1,
    random_state=1
)
```

### 주요 파라미터

```text
estimator 또는 base_estimator: weak learner
n_estimators: weak learner를 최대 몇 개까지 만들지
learning_rate: 각 weak learner의 영향력
```

### 필요한 import

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
```

---

## 12. Regression Tree

### 핵심 개념

Regression tree는 class가 아니라 숫자값을 예측하는 decision tree이다.

```text
Classification tree → class label 예측
Regression tree → numerical value 예측
```

예시:

```text
X = dosage mg
Y = effect %
```

### leaf node의 값

Regression tree의 leaf node에는 class label이 아니라 그 구간에 속한 y 값들의 평균이 들어간다.

예를 들어:

```text
x = 1, 2
y = 15, 20
```

이면 leaf 값은:

```text
(15 + 20) / 2 = 17.5
```

### split 기준

Regression tree는 split 후 각 구간의 제곱오차 합이 작아지도록 데이터를 나눈다.

```text
각 구간에서 실제값 yi와 구간 평균 ŷR 사이의 제곱오차를 최소화
```

### 예측

새로운 x가 들어오면 tree를 따라 leaf node까지 내려가고, 그 leaf node의 평균값을 예측값으로 사용한다.

예시:

```text
x ≤ 2.5        → ŷ = 17.5
2.5 < x ≤ 5.5 → ŷ = 87.7
x > 5.5       → ŷ = 21.5
```

---

## 13. Gradient Boosting

### 핵심 개념

Gradient Boosting은 loss를 줄이는 방향으로 weak learner를 순차적으로 추가하는 boosting 방법이다.

```text
현재 모델의 오차를 다음 모델이 보완
weak learner를 하나씩 더함
전체 loss를 줄임
```

### 식의 의미

슬라이드의 식:

```text
F_{m+1}(x) = F_m(x) - α ∂L/∂F_m
```

뜻:

```text
새 모델 = 현재 모델 - learning rate × loss가 줄어드는 방향
```

일반 gradient descent에서:

```text
w_new = w_old - α × gradient
```

였다면, Gradient Boosting에서는 weight 대신 모델 함수 `F(x)`를 업데이트한다고 보면 된다.

### 최종 모델 구조

```text
F(x) = F0(x) + αh1(x) + αh2(x) + αh3(x) + ...
```

여기서:

```text
F0(x): 초기 모델
h1, h2, h3: 순차적으로 추가되는 weak learners
α: learning rate
```

### `ŷ = F(x)` 의미

`ŷ = F(x)`는 증명해서 성립하는 식이라기보다 기호의 정의이다.

```text
F(x) = 모델이 x를 넣었을 때 출력하는 값
ŷ = 예측값
```

따라서:

```text
ŷ = F(x)
```

즉 모델 함수 F에 입력 x를 넣으면 예측값 ŷ이 나온다는 뜻이다.

### least-squares regression에서의 의미

제곱오차 loss:

```text
L = 1/2(y - ŷ)^2
```

이 경우 다음 weak learner는 현재 모델의 residual, 즉 실제값과 예측값의 차이를 보완하는 방향으로 학습한다.

```text
1번째 모델: 대충 예측
2번째 모델: 1번째 모델이 틀린 오차를 예측
3번째 모델: 아직 남은 오차를 또 예측
...
```

시험에서 수식 전체를 외우기보다 다음 문장을 아는 것이 중요하다.

```text
Gradient Boosting은 이전 모델의 오차를 다음 weak learner가 보완하면서 loss를 줄이는 방법이다.
```

---

## 14. scikit-learn import 전체 정리

### Pipeline 관련

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
```

### Train/test split

```python
from sklearn.model_selection import train_test_split
```

### Cross validation 관련

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
```

### Learning curve / Validation curve

```python
from sklearn.model_selection import learning_curve, validation_curve
```

### Grid Search

```python
from sklearn.model_selection import GridSearchCV
```

### ROC / AUC

```python
from sklearn.metrics import roc_curve, auc
```

### Accuracy

```python
from sklearn.metrics import accuracy_score
```

### Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
```

### Ensemble

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
```

### 기타 자주 쓰는 것

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

---

## 15. 헷갈렸던 포인트 빠른 암기

```text
Pipeline을 써도 안에 들어가는 클래스들은 import해야 한다.
```

```text
make_pipeline은 묶는 도구이고, StandardScaler/PCA/LogisticRegression은 실제 부품이다.
```

```text
Pipeline fit:
transformer는 fit_transform, 마지막 estimator는 fit.
```

```text
Pipeline predict:
transformer는 transform, 마지막 estimator는 predict.
```

```text
cross_val_score는 K-fold를 자동으로 해주는 함수이다.
```

```text
learning_curve는 training size를 바꿔가며 성능을 본다.
```

```text
validation_curve는 hyperparameter 값을 바꿔가며 성능을 본다.
```

```text
GridSearchCV는 여러 hyperparameter 조합을 모두 실험해서 최적 조합을 찾는다.
```

```text
logisticregression__C는 pipeline 안의 LogisticRegression 단계의 C를 의미한다.
```

```text
ROC는 threshold를 바꿔가며 FPR과 TPR을 보는 그래프이다.
```

```text
AUC는 ROC curve 아래 면적이고, 1에 가까울수록 좋다.
```

```text
Random Forest는 Bagging 계열의 Ensemble Learning이다.
```

```text
Bagging은 bootstrap sample로 여러 모델을 독립적으로 학습한다.
```

```text
Boosting은 이전 모델이 틀린 부분을 다음 모델이 보완한다.
```

```text
Regression Tree는 숫자를 예측하고, leaf node 값은 해당 구간 y의 평균이다.
```

```text
Gradient Boosting은 weak learner를 순차적으로 추가해서 loss를 줄인다.
```

```text
ŷ = F(x)는 모델 F에 x를 넣은 출력값이 예측값이라는 정의이다.
```
