# 모델을 선택해서 해당 모델에 대한 최적의 하이퍼 파라미터를 출력하는 함수
# model_select=1 : 랜덤 포레스트, model_select=2 : 의사결정나무 model_select=3 : 로지스틱 
# model_select=4 : Adaboost, model_select=5: XGaboost, 그 외 : 아무것도 안 함
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def param_tuning(x_train, y_train, model_select=1, scoring="precision"):

    from sklearn.model_selection import GridSearchCV

    if model_select == 1:
        from sklearn.ensemble import RandomForestClassifier
        # 하이퍼파라미터 그리드를 설정합니다.
        param_grid = {
            'n_estimators': [100, 200, 500, 1000],      # 트리의 개수
            'max_depth': [None, 10, 20, 30, 40, 50],    # 트리의 최대 깊이
        }
        clf = RandomForestClassifier()
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring=scoring, verbose=2, n_jobs=-1)
        grid_search.fit(x_train, y_train)
        print(grid_search.best_params_)
        return grid_search.best_params_
    
    elif model_select == 2:
        from sklearn.tree import DecisionTreeClassifier
        param_grid = {
            'max_depth': [None, 10, 20, 30],    # 트리의 최대 깊이
            'min_samples_split': [2, 5, 10],    # 노드를 분할하기 위한 최소한의 샘플 수
            'min_samples_leaf' : [1, 2, 4]      # 리프 노드에 필요한 최소 샘플 수
        }
        clf = DecisionTreeClassifier()
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring=scoring, verbose=2, n_jobs=-1)
        grid_search.fit(x_train, y_train)
        print(grid_search.best_params_)
        return grid_search.best_params_
    
    elif model_select == 3:
        from sklearn.linear_model import LogisticRegression
        param_grid = {
            'C': np.logspace(-3, 3, num=42),                 # 규제의 강도. 낮을수록 규제의 강도가 커지며, 모델이 과적합되는 것을 방지할 수 있습니다.
            'penalty': ['l1', 'l2', 'elasticnet', 'none']    # 규제의 종류
        }
        clf = LogisticRegression()
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring=scoring, verbose=2, n_jobs=-1)
        grid_search.fit(x_train, y_train)
        print(grid_search.best_params_)
        return grid_search.best_params_
    
    elif model_select == 4:
        from sklearn.ensemble import AdaBoostClassifier
        param_grid = {
            'n_estimators': [50, 100, 200],              # 약한 학습기(weak learners)의 수. 값이 클수록 모델은 더 복잡해질 수 있지만, 과적합의 위험도 증가
            'learning_rate': [0.01, 0.05, 0.1, 0.5, 1]   # 각 약한 학습기의 기여도를 결정. 이 값이 작을수록 각 약한 학습기의 기여도는 작아짐.
        }
        clf = AdaBoostClassifier()
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring=scoring, verbose=2, n_jobs=-1)
        grid_search.fit(x_train, y_train)
        print(grid_search.best_params_)
        return grid_search.best_params_
    
    elif model_select == 5:
        from xgboost import XGBClassifier
        param_grid = {
            'n_estimators': [100, 200, 500, 1000],   # 앙상블을 구성하는 개별 모델의 수를 결정합니다. 일반적으로 높은 값을 설정하면 모델의 복잡도가 증가하며, 너무 높은 값은 과적합의 위험을 증가
            'learning_rate': [0.01, 0.05, 0.1, 0.3], # 학습률입니다. 낮은 학습률은 더 많은 트리를 필요로 하지만, 일반적으로 더 나은 성능을 제공
            'max_depth': [3, 5, 7, 10],              # 각 개별 모델의 복잡도를 조절하는데 사용됩니다. 과적합을 피하려면 이 값을 줄이는 것이 좋습니다
        }
        clf = XGBClassifier()
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring=scoring, verbose=2, n_jobs=-1)
        grid_search.fit(x_train, y_train)
        print(grid_search.best_params_)
        return grid_search.best_params_

    else:
        return
    

### 최적의 하이퍼 파리미터값을 받으면 이를 활용하여 교차검증, ROC커브그려서 최적의 Threshold 추출, 그리고 그거를 활용한 예측 성능을 측정하는 함수
# model_select=1 : 랜덤 포레스트, model_select=2 : 의사결정나무 model_select=3 : 로지스틱 
# model_select=4 : Adaboost, model_select=5: XGaboost, 그 외 : 아무것도 안 함
def model_performance(x_train, x_test, y_train, y_test, model_select=1, best_grid={}):

    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    if model_select == 1:
        from sklearn.ensemble import RandomForestClassifier

        print('최적의 하이퍼 파라미터 :', best_grid)
        ## 교차 검증 수행
        clf = RandomForestClassifier(**best_grid, random_state=1)
        scores1 = cross_val_score(clf, x_train, y_train, scoring='accuracy', cv=5)
        scores2 = cross_val_score(clf, x_train, y_train, scoring='precision', cv=5)
        scores3 = cross_val_score(clf, x_train, y_train, scoring='recall', cv=5)
        scores4 = cross_val_score(clf, x_train, y_train, scoring='f1', cv=5)
        scores5 = cross_val_score(clf, x_train, y_train, scoring='roc_auc', cv=5)
        print(f"교차 검증 정확도 : {np.round(scores1, 4)}, 평균 정확도 : {np.round(np.mean(scores1), 4)}")
        print(f"교차 검증 정밀도 : {np.round(scores2, 4)}, 평균 정밀도 : {np.round(np.mean(scores2), 4)}")
        print(f"교차 검증 재현율 : {np.round(scores3, 4)}, 평균 재현율 : {np.round(np.mean(scores3), 4)}")
        print(f"교차 검증 f1 스코어 : {np.round(scores4, 4)}, 평균 f1 스코어 : {np.round(np.mean(scores4), 4)}")
        print(f"교차 검증 roc_auc : {np.round(scores5, 4)}, 평균 roc_auc : {np.round(np.mean(scores5), 4)}")

        # 모델 학습
        clf.fit(x_train, y_train)

        ## Precision-Recall Curve 그리고 최적의 Threshold를 추출
        # 예측 확률 계산
        probas = clf.predict_proba(x_test)
        # Precision-Recall curve를 위한 값 계산
        precisions, recalls, thresholds = precision_recall_curve(y_test, probas[:, 1])
        # precisions, recalls의 값이 최소인 지점의 threshold를 구한다.
        arg = np.argmin(abs(precisions - recalls))
        threshold = thresholds[arg]
        # Precision-Recall curve 그리기
        plt.figure(figsize=(10, 8))
        plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
        plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
        plt.xlabel('Threshold')
        plt.legend(loc='upper left')
        plt.ylim([0,1])
        plt.title('Precision-Recall Curve')
        plt.show()

        ## 예측 성능 검증
        # Precision-Recall curve를 바탕으로 적합한 임계값 설정
        print('최적의 Threshold 값 :', threshold)
        # 임계값에 따른 분류 예측값 계산
        predictions = (probas[:, 1] >= threshold).astype('int')
        # 테스트 데이터에 대한 예측 성능 지표를 계산
        test_cm = confusion_matrix(y_test, predictions)
        test_acc = accuracy_score(y_test, predictions)
        test_pre = precision_score(y_test, predictions)
        test_rcll = recall_score(y_test, predictions)
        test_roc_auc = roc_auc_score(y_test, predictions)
        test_f1 = f1_score(y_test, predictions)
        # 테스트 데이터에 대한 예측 성능 출력
        print('혼돈행렬 :', test_cm)
        print('정확도 :', round(test_acc,4))
        print('정밀도 :', round(test_pre,4))
        print('재현율 :', round(test_rcll,4))
        print('roc_auc 스코어 :', round(test_roc_auc,4))
        print('f1 스코어 :', round(test_f1,4))


    elif model_select == 2:
        from sklearn.tree import DecisionTreeClassifier
        
        print('최적의 하이퍼 파라미터 :', best_grid)
        ## 교차 검증 수행
        clf = DecisionTreeClassifier(**best_grid, random_state=1)
        scores1 = cross_val_score(clf, x_train, y_train, scoring='accuracy', cv=5)
        scores2 = cross_val_score(clf, x_train, y_train, scoring='precision', cv=5)
        scores3 = cross_val_score(clf, x_train, y_train, scoring='recall', cv=5)
        scores4 = cross_val_score(clf, x_train, y_train, scoring='f1', cv=5)
        scores5 = cross_val_score(clf, x_train, y_train, scoring='roc_auc', cv=5)
        print(f"교차 검증 정확도 : {np.round(scores1, 4)}, 평균 정확도 : {np.round(np.mean(scores1), 4)}")
        print(f"교차 검증 정밀도 : {np.round(scores2, 4)}, 평균 정밀도 : {np.round(np.mean(scores2), 4)}")
        print(f"교차 검증 재현율 : {np.round(scores3, 4)}, 평균 재현율 : {np.round(np.mean(scores3), 4)}")
        print(f"교차 검증 f1 스코어 : {np.round(scores4, 4)}, 평균 f1 스코어 : {np.round(np.mean(scores4), 4)}")
        print(f"교차 검증 roc_auc : {np.round(scores5, 4)}, 평균 roc_auc : {np.round(np.mean(scores5), 4)}")

        # 모델 학습
        clf.fit(x_train, y_train)

        ## Precision-Recall Curve 그리고 최적의 Threshold를 추출
        # 예측 확률 계산
        probas = clf.predict_proba(x_test)
        # Precision-Recall curve를 위한 값 계산
        precisions, recalls, thresholds = precision_recall_curve(y_test, probas[:, 1])
        # precisions, recalls의 값이 최소인 지점의 threshold를 구한다.
        arg = np.argmin(abs(precisions - recalls))
        threshold = thresholds[arg]
        # Precision-Recall curve 그리기
        plt.figure(figsize=(10, 8))
        plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
        plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
        plt.xlabel('Threshold')
        plt.legend(loc='upper left')
        plt.ylim([0,1])
        plt.title('Precision-Recall Curve')
        plt.show()

        ## 예측 성능 검증
        # Precision-Recall curve를 바탕으로 적합한 임계값 설정
        print('최적의 Threshold 값 :', threshold)
        # 임계값에 따른 분류 예측값 계산
        predictions = (probas[:, 1] >= threshold).astype('int')
        # 테스트 데이터에 대한 예측 성능 지표를 계산
        test_cm = confusion_matrix(y_test, predictions)
        test_acc = accuracy_score(y_test, predictions)
        test_pre = precision_score(y_test, predictions)
        test_rcll = recall_score(y_test, predictions)
        test_roc_auc = roc_auc_score(y_test, predictions)
        test_f1 = f1_score(y_test, predictions)
        # 테스트 데이터에 대한 예측 성능 출력
        print('혼돈행렬 :', test_cm)
        print('정확도 :', round(test_acc,4))
        print('정밀도 :', round(test_pre,4))
        print('재현율 :', round(test_rcll,4))
        print('roc_auc 스코어 :', round(test_roc_auc,4))
        print('f1 스코어 :', round(test_f1,4))

    
    elif model_select == 3:
        from sklearn.linear_model import LogisticRegression
        
        print('최적의 하이퍼 파라미터 :', best_grid)
        ## 교차 검증 수행
        clf = LogisticRegression(**best_grid, random_state=1)
        scores1 = cross_val_score(clf, x_train, y_train, scoring='accuracy', cv=5)
        scores2 = cross_val_score(clf, x_train, y_train, scoring='precision', cv=5)
        scores3 = cross_val_score(clf, x_train, y_train, scoring='recall', cv=5)
        scores4 = cross_val_score(clf, x_train, y_train, scoring='f1', cv=5)
        scores5 = cross_val_score(clf, x_train, y_train, scoring='roc_auc', cv=5)
        print(f"교차 검증 정확도 : {np.round(scores1, 4)}, 평균 정확도 : {np.round(np.mean(scores1), 4)}")
        print(f"교차 검증 정밀도 : {np.round(scores2, 4)}, 평균 정밀도 : {np.round(np.mean(scores2), 4)}")
        print(f"교차 검증 재현율 : {np.round(scores3, 4)}, 평균 재현율 : {np.round(np.mean(scores3), 4)}")
        print(f"교차 검증 f1 스코어 : {np.round(scores4, 4)}, 평균 f1 스코어 : {np.round(np.mean(scores4), 4)}")
        print(f"교차 검증 roc_auc : {np.round(scores5, 4)}, 평균 roc_auc : {np.round(np.mean(scores5), 4)}")

        # 모델 학습
        clf.fit(x_train, y_train)

        ## Precision-Recall Curve 그리고 최적의 Threshold를 추출
        # 예측 확률 계산
        probas = clf.predict_proba(x_test)
        # Precision-Recall curve를 위한 값 계산
        precisions, recalls, thresholds = precision_recall_curve(y_test, probas[:, 1])
        # precisions, recalls의 값이 최소인 지점의 threshold를 구한다.
        arg = np.argmin(abs(precisions - recalls))
        threshold = thresholds[arg]
        # Precision-Recall curve 그리기
        plt.figure(figsize=(10, 8))
        plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
        plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
        plt.xlabel('Threshold')
        plt.legend(loc='upper left')
        plt.ylim([0,1])
        plt.title('Precision-Recall Curve')
        plt.show()

        ## 예측 성능 검증
        # Precision-Recall curve를 바탕으로 적합한 임계값 설정
        print('최적의 Threshold 값 :', threshold)
        # 임계값에 따른 분류 예측값 계산
        predictions = (probas[:, 1] >= threshold).astype('int')
        # 테스트 데이터에 대한 예측 성능 지표를 계산
        test_cm = confusion_matrix(y_test, predictions)
        test_acc = accuracy_score(y_test, predictions)
        test_pre = precision_score(y_test, predictions)
        test_rcll = recall_score(y_test, predictions)
        test_roc_auc = roc_auc_score(y_test, predictions)
        test_f1 = f1_score(y_test, predictions)
        # 테스트 데이터에 대한 예측 성능 출력
        print('혼돈행렬 :', test_cm)
        print('정확도 :', round(test_acc,4))
        print('정밀도 :', round(test_pre,4))
        print('재현율 :', round(test_rcll,4))
        print('roc_auc 스코어 :', round(test_roc_auc,4))
        print('f1 스코어 :', round(test_f1,4))
    
    
    elif model_select == 4:
        from sklearn.ensemble import AdaBoostClassifier
        
        print('최적의 하이퍼 파라미터 :', best_grid)
        ## 교차 검증 수행
        clf = AdaBoostClassifier(**best_grid, random_state=1)
        scores1 = cross_val_score(clf, x_train, y_train, scoring='accuracy', cv=5)
        scores2 = cross_val_score(clf, x_train, y_train, scoring='precision', cv=5)
        scores3 = cross_val_score(clf, x_train, y_train, scoring='recall', cv=5)
        scores4 = cross_val_score(clf, x_train, y_train, scoring='f1', cv=5)
        scores5 = cross_val_score(clf, x_train, y_train, scoring='roc_auc', cv=5)
        print(f"교차 검증 정확도 : {np.round(scores1, 4)}, 평균 정확도 : {np.round(np.mean(scores1), 4)}")
        print(f"교차 검증 정밀도 : {np.round(scores2, 4)}, 평균 정밀도 : {np.round(np.mean(scores2), 4)}")
        print(f"교차 검증 재현율 : {np.round(scores3, 4)}, 평균 재현율 : {np.round(np.mean(scores3), 4)}")
        print(f"교차 검증 f1 스코어 : {np.round(scores4, 4)}, 평균 f1 스코어 : {np.round(np.mean(scores4), 4)}")
        print(f"교차 검증 roc_auc : {np.round(scores5, 4)}, 평균 roc_auc : {np.round(np.mean(scores5), 4)}")

        # 모델 학습
        clf.fit(x_train, y_train)

        ## Precision-Recall Curve 그리고 최적의 Threshold를 추출
        # 예측 확률 계산
        probas = clf.predict_proba(x_test)
        # Precision-Recall curve를 위한 값 계산
        precisions, recalls, thresholds = precision_recall_curve(y_test, probas[:, 1])
        # precisions, recalls의 값이 최소인 지점의 threshold를 구한다.
        arg = np.argmin(abs(precisions - recalls))
        threshold = thresholds[arg]
        # Precision-Recall curve 그리기
        plt.figure(figsize=(10, 8))
        plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
        plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
        plt.xlabel('Threshold')
        plt.legend(loc='upper left')
        plt.ylim([0,1])
        plt.title('Precision-Recall Curve')
        plt.show()

        ## 예측 성능 검증
        # Precision-Recall curve를 바탕으로 적합한 임계값 설정
        print('최적의 Threshold 값 :', threshold)
        # 임계값에 따른 분류 예측값 계산
        predictions = (probas[:, 1] >= threshold).astype('int')
        # 테스트 데이터에 대한 예측 성능 지표를 계산
        test_cm = confusion_matrix(y_test, predictions)
        test_acc = accuracy_score(y_test, predictions)
        test_pre = precision_score(y_test, predictions)
        test_rcll = recall_score(y_test, predictions)
        test_roc_auc = roc_auc_score(y_test, predictions)
        test_f1 = f1_score(y_test, predictions)
        # 테스트 데이터에 대한 예측 성능 출력
        print('혼돈행렬 :', test_cm)
        print('정확도 :', round(test_acc,4))
        print('정밀도 :', round(test_pre,4))
        print('재현율 :', round(test_rcll,4))
        print('roc_auc 스코어 :', round(test_roc_auc,4))
        print('f1 스코어 :', round(test_f1,4))


    elif model_select == 5:
        from xgboost import XGBClassifier
        
        print('최적의 하이퍼 파라미터 :', best_grid)
        ## 교차 검증 수행
        clf = XGBClassifier(**best_grid, random_state=1)
        scores1 = cross_val_score(clf, x_train, y_train, scoring='accuracy', cv=5)
        scores2 = cross_val_score(clf, x_train, y_train, scoring='precision', cv=5)
        scores3 = cross_val_score(clf, x_train, y_train, scoring='recall', cv=5)
        scores4 = cross_val_score(clf, x_train, y_train, scoring='f1', cv=5)
        scores5 = cross_val_score(clf, x_train, y_train, scoring='roc_auc', cv=5)
        print(f"교차 검증 정확도 : {np.round(scores1, 4)}, 평균 정확도 : {np.round(np.mean(scores1), 4)}")
        print(f"교차 검증 정밀도 : {np.round(scores2, 4)}, 평균 정밀도 : {np.round(np.mean(scores2), 4)}")
        print(f"교차 검증 재현율 : {np.round(scores3, 4)}, 평균 재현율 : {np.round(np.mean(scores3), 4)}")
        print(f"교차 검증 f1 스코어 : {np.round(scores4, 4)}, 평균 f1 스코어 : {np.round(np.mean(scores4), 4)}")
        print(f"교차 검증 roc_auc : {np.round(scores5, 4)}, 평균 roc_auc : {np.round(np.mean(scores5), 4)}")

        # 모델 학습
        clf.fit(x_train, y_train)

        ## Precision-Recall Curve 그리고 최적의 Threshold를 추출
        # 예측 확률 계산
        probas = clf.predict_proba(x_test)
        # Precision-Recall curve를 위한 값 계산
        precisions, recalls, thresholds = precision_recall_curve(y_test, probas[:, 1])
        # precisions, recalls의 값이 최소인 지점의 threshold를 구한다.
        arg = np.argmin(abs(precisions - recalls))
        threshold = thresholds[arg]
        # Precision-Recall curve 그리기
        plt.figure(figsize=(10, 8))
        plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
        plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
        plt.xlabel('Threshold')
        plt.legend(loc='upper left')
        plt.ylim([0,1])
        plt.title('Precision-Recall Curve')
        plt.show()

        ## 예측 성능 검증
        # Precision-Recall curve를 바탕으로 적합한 임계값 설정
        print('최적의 Threshold 값 :', threshold)
        # 임계값에 따른 분류 예측값 계산
        predictions = (probas[:, 1] >= threshold).astype('int')
        # 테스트 데이터에 대한 예측 성능 지표를 계산
        test_cm = confusion_matrix(y_test, predictions)
        test_acc = accuracy_score(y_test, predictions)
        test_pre = precision_score(y_test, predictions)
        test_rcll = recall_score(y_test, predictions)
        test_roc_auc = roc_auc_score(y_test, predictions)
        test_f1 = f1_score(y_test, predictions)
        # 테스트 데이터에 대한 예측 성능 출력
        print('혼돈행렬 :', test_cm)
        print('정확도 :', round(test_acc,4))
        print('정밀도 :', round(test_pre,4))
        print('재현율 :', round(test_rcll,4))
        print('roc_auc 스코어 :', round(test_roc_auc,4))
        print('f1 스코어 :', round(test_f1,4))

    else:
        return