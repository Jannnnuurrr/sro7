import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Загрузка датасета
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели TSVM
tsvm_model = SelfTrainingClassifier(SVC(kernel='linear'))

# Обучение модели
tsvm_model.fit(X_train, y_train)

# Предсказание на тестовых данных
y_pred = tsvm_model.predict(X_test)

# Оценка модели
accuracy = accuracy_score(y_test, y_pred)
print("Точность модели:", accuracy)

print("Отчет по классификации:")
print(classification_report(y_test, y_pred)
