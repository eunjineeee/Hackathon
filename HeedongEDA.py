import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 변수들 중 고유값이 한개인 컬럼들을 제거하는 함수
def removeunique(df):

    for col in df.columns:
        if df[col].nunique() == 1:   # nunique()는 데이터에 고유값의 수. 고유값의 수가 1개인 컬럼을 찾아내서
            df.drop(col, axis=1, inplace=True)   # 고유값의 수가 1개인 컬럼을 데이터에서 제거
    return df


# 변수들 중 숫자형 변수와 해당컬럼 혹은 오브젝트형 변수와 해당 컬럼만 추출하는 함수
def getvarcol(df, select = 0):

    if select == 0:
        df_ = df.select_dtypes(include=[np.number])
        df_col = list(df_.columns)
    elif select == 1:
        df_ = df.select_dtypes(include='object')
        df_col = list(df_.columns)
    else :
        df_ = df
        df_col = list(df_.columns)
        print('너 허탕쳤다!!')
    return df_, df_col


# 이상치 분포를 그래프로 나타내는 함수
def plotoutliers(df):

    df_ = df.select_dtypes(include=[np.number])
    num_col = list(df_.columns)
    n = len(num_col)
    ncols = 4
    nrows = n // ncols + (n % ncols > 0)

    # 서브플롯 생성
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows*5))

    for i, var in enumerate(num_col):
        r, c = i // ncols, i % ncols
        ax = axes[r, c]
        df_.boxplot(column=var, ax=ax)
        ax.set_title(f'Box plot for {var}')

    # 빈 서브플롯 제거
    if n % ncols > 0:
        for j in range(n % ncols, ncols):
            fig.delaxes(axes[nrows-1, j])

    plt.tight_layout()  #  plt.tight_layout() 함수는 서브플롯 간의 간격을 적절하게 조절합니다.
    plt.show()


# 각 숫자형 변수들의 분포를 출력하는 함수
def num_plot(df):

    df_ = df.select_dtypes(include=[np.number])

    df1 = df_.copy()
    df1['y'] = df['y']
    df1 = df1.groupby('y').mean()

    ## 타겟 데이터에 따른 각 숫자형 변수들의 평균 분포를 시각화

    n = len(list(df1.columns))
    ncols = 4
    nrows = n // ncols + (n % ncols > 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows*5))

    for i, var in enumerate(list(df1.columns)):
        r, c = i // ncols, i % ncols
        ax = axes[r, c]
        sns.barplot(data=df1, x=df1.index, y=df1.iloc[:,i], ax = ax)
        ax.set_title(f'Bar plot for {var}')

    # 빈 서브플롯 제거
    if n % ncols > 0:
        for j in range(n % ncols, ncols):
            fig.delaxes(axes[nrows-1, j])

    plt.tight_layout()  #  plt.tight_layout() 함수는 서브플롯 간의 간격을 적절하게 조절합니다.
    plt.show()


# 각 숫자형 변수들의 피어슨 상관 분포를 히트맵으로 출력
def plot_cor(df):

    df_ = df.select_dtypes(include=[np.number])
    df_cor = df_.corr(method='pearson')
    plt.figure(figsize = (10,8))

    sns.heatmap(df_cor, 
                xticklabels = df_cor.columns, 
                yticklabels = df_cor.columns, 
                cmap = 'RdBu_r', 
                annot = True,
                annot_kws = {'size':10}, 
                linewidth = 3,
                vmin=-1, vmax=1)
    

## 범주형 변수들을 시각화하는 함수
def obj_plot(df):
    df_ = df.select_dtypes(include='object')
    object_col = list(df_.columns)
    n = len(object_col)
    ncols = 4
    nrows = n // ncols + (n % ncols > 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows*5))

    for i, var in enumerate(object_col):
        r, c = i // ncols, i % ncols
        ax = axes[r, c]
        axes[r][c].barh(df_[var].value_counts().index, df_[var].value_counts().values)
        # sns.barplot(data=marketing_object, x=marketing_object.index, y=marketing_object.iloc[:,i], ax = ax)
        ax.set_title(f'Bar plot for {var}')

    # 빈 서브플롯 제거
    if n % ncols > 0:
        for j in range(n % ncols, ncols):
            fig.delaxes(axes[nrows-1, j])

    plt.tight_layout()  #  plt.tight_layout() 함수는 서브플롯 간의 간격을 적절하게 조절합니다.
    plt.show()


# 윈저라이징 하는 함수
def mywinsorizer(df):

    from scipy.stats import mstats
    # Winsorization을 적용할 분위수 값 설정
    lower_limit = 0.05
    upper_limit = 0.05

    df_ = df.select_dtypes(include=[np.number])
    df_col = list(df_.columns)

    # df_ = df.select_dtypes(include=[np.number])
    # df_col = list(df_.columns)
    # 열 단위로 Winsorization 수행
    df_ = df_.apply(lambda x: mstats.winsorize(x, limits=(lower_limit, upper_limit)))
    df[df_col] = df_
    return df


# 바이너리 인코더로 인코딩을 하고 인코딩 정보가 담긴 딕셔너리를 결과값을 뱉음. 그리고 인코딩 정보가 담긴 파일을 저장
def encoding(df):
    df_ = df.select_dtypes(include='object')
    object_col = list(df_.columns)

    from category_encoders import BinaryEncoder
    encoder = BinaryEncoder(cols=object_col)
    object_encoded = encoder.fit_transform(df_[object_col])
    object_encoded_col = list(object_encoded.columns)

    obj_dict = []

    for col in object_col:
        encoder = BinaryEncoder(cols=col)
        objenc1 = encoder.fit_transform(df_[col])

        x_temp = objenc1.set_index(df_[col].values)
        mask = x_temp.index.duplicated(keep='first')
        x_obj_unique = x_temp[~mask]

        aa = x_obj_unique.to_dict(orient='index')
        obj_dict.append(aa)
    
    # 인코더 딕셔너리 객체 저장
    import pickle
    with open(f'obj_dict.pkl', 'wb') as f:
        pickle.dump(obj_dict, f)

    return object_encoded, object_encoded_col, obj_dict


# 리샘플링을 진행하는 함수
def resampler(df, target, choose=0):
    from collections import Counter
    if choose == 0:

        from imblearn.under_sampling import RandomUnderSampler

        undersample = RandomUnderSampler(sampling_strategy='majority')
        x_sm, y_sm = undersample.fit_resample(df, target)
        counter_res = Counter(y_sm)
        print('Resampled Dataset Shape %s' % counter_res)

        return x_sm, y_sm
    
    elif choose == 1:

        from imblearn.over_sampling import SMOTE
        smote = SMOTE(sampling_strategy='minority')
        # SMOTE 적용
        x_sm, y_sm = smote.fit_resample(df, target)

        # 새로운 데이터셋 클래스 분포 확인
        counter_res = Counter(y_sm)
        print('Resampled Dataset Shape %s' % counter_res)

        return x_sm, y_sm


# 피쳐 중요도를 구해서 그래프로 나타내는 함수
def get_importance(x,y):
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators = 25 , random_state = 1)
    clf.fit(x,y)

    importances = clf.feature_importances_
    feature_importances = {}

    for feature, importance in zip(x.columns, importances):

        # 접미사 제거. _0, _1, _2, _3 순으로..
        if feature.endswith("_0"):
            feature = feature[:-len("_0")]
        if feature.endswith("_1"):
            feature = feature[:-len("_1")]        
        if feature.endswith("_2"):
            feature = feature[:-len("_2")]
        if feature.endswith("_3"):
            feature = feature[:-len("_3")]

        # 만약 original_feature가 딕셔너리에 있으면, 해당 feature의 importance를 기존의 importance에 더합니다.
        if feature in feature_importances:
            feature_importances[feature] = feature_importances[feature] + importance
        # 만약 original_feature가 딕셔너리에 없으면, 새로운 키-값 쌍을 생성하여 딕셔너리에 추가합니다.
        else:
            feature_importances[feature] = importance
    
    featureimp = pd.Series(feature_importances.values() , index = feature_importances.keys()).sort_values(ascending = True)
    plt.figure(figsize = (14,12))
    featureimp.plot(kind = 'barh')


# 스케일러 선택. 숫자형 변수만 넣어야함
def myscaler(x_num, choose=0):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import RobustScaler

    num_col = list(x_num.columns)

    if choose == 0 :

        # Scaler 객체 생성
        scaler = MinMaxScaler()
        x_num_scaled = scaler.fit_transform(x_num)

        # print('\t\t(min, max) (mean, std)')
        print('Scaled (%.2f, %.2f) (%.2f, %.2f)' %(x_num_scaled.min(), x_num_scaled.max(), x_num_scaled.mean(), x_num_scaled.std()))
        x_num_scaled = pd.DataFrame(x_num_scaled, columns=num_col)
        
        return x_num_scaled
    
    elif choose == 1:

        # Scaler 객체 생성
        scaler = StandardScaler()
        x_num_scaled = scaler.fit_transform(x_num)

        # print('\t\t(min, max) (mean, std)')
        print('Train_scaled (%.2f, %.2f) (%.2f, %.2f)' %(x_num_scaled.min(), x_num_scaled.max(), x_num_scaled.mean(), x_num_scaled.std()))
        x_num_scaled = pd.DataFrame(x_num_scaled, columns=num_col)

        return x_num_scaled
    
    elif choose == 2:

        # Scaler 객체 생성
        scaler = RobustScaler()
        x_num_scaled = scaler.fit_transform(x_num)

        # print('\t\t(min, max) (mean, std)')
        print('Train_scaled (%.2f, %.2f) (%.2f, %.2f)' %(x_num_scaled.min(), x_num_scaled.max(), x_num_scaled.mean(), x_num_scaled.std()))
        x_num_scaled = pd.DataFrame(x_num_scaled, columns=num_col)

        return x_num_scaled
    
    else:
        x_num_scaled = x_num
        return x_num_scaled


## 숫자형 변수들의 VIF 지수를 체크하는 함수
def check_vif(x_num):
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # 각 변수별 다중공선성 다시 체크해보자
    vif = pd.DataFrame()

    # VIF 계산
    vif["features"] = x_num.columns
    vif["VIF Factor"] = [variance_inflation_factor(x_num.values, i) for i in range(x_num.shape[1])]

    print(vif)


## PCA를 수행하고 PCA 변환된 정보를 데이터프레임의 형태로 출력하는 함수
def doPCA(x_num, vif_high_columns, cols=['Newcomponent'], train=1):

    from sklearn.decomposition import PCA

    vif_high = x_num[vif_high_columns]
    pca = PCA(n_components = len(cols))  # 1개의 주성분을 추출

    if train == 1:
        x_pca = pca.fit_transform(vif_high)
        print(f'VIF지수가 높은 {len(vif_high_columns)}개의 피처를 {len(cols)}개의 피처로 PCA 변환했을 때 새롭게 나온 피처가 전체의 변동성을 {pca.explained_variance_ratio_*100}%만큼 설명해준다')  
    else:
        x_pca = pca.fit_transform(vif_high)
    
    df_pca = pd.DataFrame(data = x_pca, columns = cols)

    x_df_pca = x_num.copy()
    x_df_pca.drop(vif_high.columns, axis=1, inplace=True)
    x_df_pca[cols] = df_pca

    return x_df_pca


## 교차 검증및 예측 성능을 검증하는 함수. pcaok가 1인 경우 PCA의 성능 검증.
def pca_check(x_train, x_test, y_train, y_test, scoring = "precision", cv = 3, pcaok=0):

    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators = 25 , random_state = 1)
    scores = cross_val_score(clf, x_train, y_train, scoring = scoring, cv = cv)
    scores = np.round(scores, 4)

    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    test_pre = precision_score(y_test, pred)
    test_pre = np.round(test_pre, 4)

    if pcaok == 1:
        print(f"CV={cv} 인경우의 PCA 변환된 개별 Fold 세트별 {scoring} 교차검증 : ", scores)
        print(f"PCA 변환 데이터 셋 평균 {scoring} : {np.round(np.mean(scores), 4)}")
        print(f"{scoring} 예측 성능 :", test_pre)    

    else:
        print(f"CV={cv} 인 경우의 개별 Fold 세트별 {scoring} 교차검증 : ", scores)
        print(f"데이터 셋 평균 {scoring} : {np.round(np.mean(scores), 4)}")
        print(f"{scoring} 예측 성능 :", test_pre)


# # 딕셔너리 로드
# with open('obj_dict.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)

# encoded_data = {'month_0': 0, 'month_1': 0, 'month_2': 1, 'month_3': 1}

# # 디코딩
# for category, encoded_value in loaded_dict[7].items():
#     if encoded_value == encoded_data:
#         decoded_value = category
#         break

# print(f'Decoded value: {decoded_value}')