# 머신러닝 대화 정리: 헷갈렸던 개념 + scikit-learn import 모음

## 1. 내가 특히 헷갈려했던 부분 정리

### 1-1. Pipeline
핵심:
- `pipeline`은 **전처리 + 모델**을 한 줄로 연결한 것이다.
- 예: `StandardScaler -> PCA -> LogisticRegression`

예시:
```python
pipe_lr = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    LogisticRegression(random_state=1)
)
```

의미:
- 먼저 표준화
- 그다음 PCA로 차원 축소
- 마지막에 Logistic Regression으로 학습

중요:
- `pipe_lr.fit(X_train, y_train)`  
  내부적으로:
  1. `StandardScaler().fit_transform(X_train)`
  2. `PCA().fit_transform(...)`
  3. `LogisticRegression().fit(..., y_train)`

- `y_pred = pipe_lr.predict(X_test)`  
  내부적으로:
  1. `StandardScaler().transform(X_test)`
  2. `PCA().transform(...)`
  3. `LogisticRegression().predict(...)`

헷갈렸던 포인트:
- `y_pred`는 뭔가를 "설명하는 코드"가 아니라 **예측 결과를 저장하는 변수**
- test 데이터에는 `fit`이 아니라 `transform`만 적용된다.
- 이유: 기준(평균, 표준편차, PCA 축 등)은 train에서만 배워야 하기 때문

---

### 1-2. Learning Curve
핵심:
- `learning_curve()`는 **훈련 데이터 크기를 점점 늘려가면서**
  - train accuracy
  - validation accuracy  
  가 어떻게 변하는지 보는 함수이다.

예시 개념:
- `train_sizes=np.linspace(0.1, 1.0, 10)`이면 train size 종류가 10개
- `cv=10`이면 각 train size마다 10-fold CV를 수행

따라서 shape:
- `train_scores.shape = (10, 10)`
- 의미:
  - 첫 번째 10 = **훈련 크기 종류 수**
  - 두 번째 10 = **cv=10의 결과 개수**

교수님 설명 정리:
- "하나의 training size에 대해 10개의 측정 결과가 있다"  
  = **훈련 크기 하나를 정해놓고, 그 조건에서 10-fold CV를 해서 점수 10개가 나온다**는 뜻

평균:
```python
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
```

의미:
- 각 train size마다 CV 결과 10개를 평균내서 대표값 하나로 만든다.

---

### 1-3. `axis=1` 의미
헷갈렸던 점:
- `axis=1`이 무조건 "열 삭제"라고 느껴졌음

정리:
- `axis`는 **방향**을 말할 뿐이고, 무슨 연산인지는 함수가 결정한다.

예:
```python
np.mean(A, axis=1)
```
- 각 **행의 평균**을 구한다.

즉:
- `axis=0`: 세로 방향
- `axis=1`: 가로 방향

`np.mean(..., axis=1)` in learning curve:
- 각 행(= 각 train size 또는 각 parameter 값)에 있는 CV 결과들을 평균내는 것

---

### 1-4. Validation Curve
핵심:
- `validation_curve()`는 **하이퍼파라미터 값을 바꿔가며** 성능 변화를 본다.

예:
```python
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(
    estimator=pipe_lr,
    X=X_train,
    y=y_train,
    param_name='logisticregression__C',
    param_range=param_range,
    cv=10
)
```

중요:
- `param_name='logisticregression__C'`
  - pipeline 안의 `logisticregression` 단계의 `C` 값을 바꾸겠다는 뜻
- `param_range`
  - 시험할 `C` 값 목록

shape:
- `(6, 10)`이면
  - 6 = `param_range` 값 개수
  - 10 = `cv=10`

헷갈렸던 포인트 정리:
- 여기서 6은 **feature 개수**가 아니다.
- 6은 **시험한 하이퍼파라미터 값의 개수**이다.

---

### 1-5. High Bias / High Variance
정리:
- **High bias = underfitting**
  - train score도 낮고 validation score도 낮음
  - 모델이 너무 단순함
- **High variance = overfitting**
  - train score는 높은데 validation score는 낮음
  - 모델이 훈련 데이터에만 지나치게 맞춰짐

교수님 말씀 정리:
- "high bias에 대해 적합한 정도의 complexity"
  - 의미상: **high bias 상태라면 현재 complexity가 부족하므로 적절한 수준까지 complexity를 올려야 한다**는 뜻으로 이해하면 된다.

---

### 1-6. Grid Search
핵심:
- `GridSearchCV`는 **여러 하이퍼파라미터 조합을 전부 시험**해서 가장 좋은 조합을 찾는다.

예:
```python
param_grid = [{
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'penalty': ['l1', 'l2']
}]
```

이 경우:
- `C` 5개
- `penalty` 2개
- 총 10개 조합 평가

중요 속성:
- `cv_results_`: 전체 결과표
- `best_score_`: 최고 평균 CV 점수
- `best_params_`: 최고 성능의 하이퍼파라미터 조합
- `best_estimator_`: 최적 파라미터가 적용된 모델

헷갈렸던 포인트:
- validation curve는 **파라미터 하나**
- grid search는 **파라미터 조합 전체**

---

### 1-7. ROC Curve / AUC
핵심:
- ROC curve는 **threshold를 바꿔가며**
  - FPR
  - TPR  
  을 계산해 그린 그래프다.

중요 코드:
```python
probas = pipe_lr.predict_proba(X_test2)
fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
```

정리:
- `predict_proba()`는 클래스 확률을 반환
- `probas[:, 1]`은 **클래스 1일 확률만 뽑은 것**
- `roc_curve(...)`는 threshold를 여러 개 바꿔가며 FPR, TPR 계산
- `auc(fpr, tpr)`는 ROC curve 아래 면적 계산

중요 개념:
- TPR = `TP / (TP + FN)`
- FPR = `FP / (FP + TN)`

헷갈렸던 포인트:
- 왜 `predict()`가 아니라 `predict_proba()`를 쓰는가?
  - ROC는 threshold를 계속 바꿔야 해서 확률이 필요하기 때문
- 왜 `[:,1]`인가?
  - positive class인 `y=1`의 확률만 필요하기 때문

---

### 1-8. Bagging
핵심:
- **같은 종류의 모델을 여러 개 만들고**
- 각 모델의 예측을 모아
- **투표(voting)** 로 최종 결정하는 방법

예:
```python
tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=1)

bag = BaggingClassifier(
    base_estimator=tree,
    n_estimators=100,
    max_samples=0.5,
    max_features=1.0,
    n_jobs=1,
    random_state=1
)
```

중요:
- `n_estimators=100`: 트리 100개
- `max_samples=0.5`: 각 트리는 샘플 50%씩 사용
- `max_features=1.0`: 모든 feature 사용

헷갈렸던 포인트:
- voting 코드가 안 보였음
- 실제 voting은 `bag.predict(X_test)` 같은 **예측 시점에 내부적으로 자동 수행**된다.

Bagging 장점:
- variance 감소
- test 성능 개선 가능
- 불안정한 모델(특히 decision tree)에 잘 맞음

---

### 1-9. Boosting / AdaBoost
핵심:
- Bagging과 달리, Boosting은 **약한 학습기들을 순차적으로** 학습시킨다.
- 이전 모델이 틀린 샘플에 더 집중하도록 다음 모델을 학습시킨다.

슬라이드 핵심 문장 정리:
- simple classifiers를 **sequentially**
- hard-to-classify samples에 집중
- final classifier는 weak learners의 **weighted combination**

AdaBoost:
- 매 반복마다 샘플 가중치를 조정
- 이전 weak learner가 틀린 샘플의 가중치를 높임
- 다음 learner가 그 샘플을 더 신경 쓰게 됨

예:
```python
tree = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=1)

ada = AdaBoostClassifier(
    base_estimator=tree,
    n_estimators=100,
    learning_rate=0.1,
    random_state=1
)
```

중요:
- `max_depth=1`인 tree는 **decision stump**
- 매우 단순한 weak learner
- `n_estimators=100`: 최대 100개 weak learner
- `learning_rate=0.1`: 각 weak learner의 기여도 조절

Bagging vs Boosting 핵심 차이:
- Bagging: 독립적으로 여러 모델 학습 + voting
- Boosting: 순차적으로 학습 + 이전 실수 보완 + weighted sum

---

### 1-10. Regression Tree
핵심:
- regression tree는 **숫자값을 예측하는 결정트리**
- 리프 노드에는 class label이 아니라 **숫자 평균값**이 들어간다.

기본 아이디어:
1. 어떤 구간에 들어온 데이터의 y 평균을 예측값으로 사용
2. split을 반복하며 오차를 줄임
3. 최종적으로 도달한 리프의 평균값을 예측

기준:
- 제곱오차합(SSE) 또는 MSE를 최소화하는 split 선택

슬라이드 예시 핵심:
- 전체 평균으로 예측하면 거칠다
- `x > 2.5`, `x > 5.5` 같은 분할을 통해
  - 17.5
  - 87.7
  - 21.5  
  같은 구간별 평균 예측값을 얻는다.

중요:
- regression tree는 **piecewise constant** 형태의 예측을 만든다.

---

## 2. scikit-learn class / function import 정리

### 2-1. 데이터 분할
```python
from sklearn.model_selection import train_test_split
```

---

### 2-2. 표준화 / 전처리
```python
from sklearn.preprocessing import StandardScaler
```

---

### 2-3. PCA
```python
from sklearn.decomposition import PCA
```

---

### 2-4. Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
```

---

### 2-5. Pipeline
```python
from sklearn.pipeline import make_pipeline, Pipeline
```

---

### 2-6. Learning Curve / Validation Curve / Grid Search
```python
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
```

---

### 2-7. Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
```

---

### 2-8. Bagging
```python
from sklearn.ensemble import BaggingClassifier
```

---

### 2-9. AdaBoost / Boosting
```python
from sklearn.ensemble import AdaBoostClassifier
```

추가로 boosting 계열에서 자주 보는 것:
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
```

---

### 2-10. ROC / AUC / Accuracy
```python
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
```

추가로 자주 같이 쓰는 것:
```python
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
```

---

### 2-11. Wine dataset 예시에서 자주 쓰는 것
```python
from sklearn.datasets import load_wine
```

---

## 3. 자주 나오는 코드 패턴 모음

### 3-1. Pipeline 생성
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

---

### 3-2. Learning Curve
```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    estimator=pipe_lr,
    X=X,
    y=y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=10,
    n_jobs=1
)
```

---

### 3-3. Validation Curve
```python
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
```

---

### 3-4. Grid Search
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
    n_jobs=-1
)
```

---

### 3-5. ROC / AUC
```python
from sklearn.metrics import roc_curve, auc

probas = pipe_lr.predict_proba(X_test2)
fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
```

---

### 3-6. Bagging
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

tree = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,
    random_state=1
)

bag = BaggingClassifier(
    base_estimator=tree,
    n_estimators=100,
    max_samples=0.5,
    max_features=1.0,
    n_jobs=1,
    random_state=1
)
```

---

### 3-7. AdaBoost
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

tree = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=1,
    random_state=1
)

ada = AdaBoostClassifier(
    base_estimator=tree,
    n_estimators=100,
    learning_rate=0.1,
    random_state=1
)
```

---

## 4. 시험 직전 암기용 핵심 요약

### Pipeline
- 전처리 + 모델을 한 줄로 연결
- train에서는 `fit_transform`, test에서는 `transform`

### Learning Curve
- x축: train size
- y축: score
- high bias / high variance 확인

### Validation Curve
- x축: 하이퍼파라미터 값
- y축: score
- overfitting/underfitting 지점 확인

### Grid Search
- 여러 하이퍼파라미터 조합 전부 시험
- 최고 조합 찾기

### ROC / AUC
- threshold를 바꿔가며 TPR/FPR 계산
- AUC가 클수록 좋음

### Bagging
- bootstrap sample로 같은 모델 여러 개 학습
- 예측 시 voting
- variance 줄이기

### Boosting / AdaBoost
- 약한 학습기를 순차적으로 학습
- 이전 모델이 틀린 샘플에 집중
- weighted combination

### Regression Tree
- 리프 노드에서 숫자 예측
- 각 리프 예측값 = 그 구간 y 평균
- split 기준 = 제곱오차 최소화

---

## 5. 마지막 한 줄 정리
이번 대화에서 가장 중요한 흐름은:

**Pipeline → Learning/Validation Curve → Grid Search → ROC/AUC → Bagging → Boosting → Regression Tree**

이 순서대로,
- 모델을 연결하고
- 성능을 진단하고
- 하이퍼파라미터를 튜닝하고
- 평가 지표를 이해하고
- 앙상블(Bagging/Boosting)과 트리 기반 모델로 확장해 간 것이다.
