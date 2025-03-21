import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 기본 설정
n_train = 10
n_test = 100
centers = 3
n_features = 2
random_state = 42
center_box = (-10, 10)

# 결과 저장용 딕셔너리
results = {}

# cluster_std 1부터 10까지
for cluster_std in np.arange(1, 3.2, 0.2):
    # 총 샘플 수
    n_total = n_train + n_test
    
    # 데이터 생성
    X, y = make_blobs(n_samples=n_total, centers=centers, cluster_std=cluster_std, center_box=center_box,
                     n_features=n_features, random_state=random_state)
    
    # 훈련/테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                       train_size=n_train, 
                                                       test_size=n_test, 
                                                       random_state=42)
    
    # k 값 1부터 5까지
    k_results = {}
    for k in range(1, 6):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        k_results[k] = accuracy
    
    results[cluster_std] = k_results

# 결과 출력
for std in results:
    print(f"\ncluster_std = {std}:")
    for k in results[std]:
        print(f"k = {k}: accuracy = {results[std][k]:.3f}")

# 시각화
plt.figure(figsize=(10, 6))

# 각 k값에 대해 그래프 그리기
for k in range(1, 6):
    # x값: cluster_std들
    x_values = list(results.keys())
    # y값: 각 cluster_std에 대한 k값의 accuracy
    y_values = [results[std][k] for std in x_values]
    
    plt.plot(x_values, y_values, marker='o', label=f'k={k}')

plt.xlabel('Cluster Standard Deviation')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy vs Cluster Std for Different k Values')
plt.legend()
plt.grid(True)
plt.savefig('knn_accuracy_vs_cluster_std.png', dpi=300, bbox_inches='tight')
plt.close()