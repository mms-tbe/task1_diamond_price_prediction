import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Загрузка данных
df = pd.read_csv('diamonds.csv')

# Разведочный анализ данных (EDA)
def perform_eda(df):
    print(df.head())
    print(df.info())
    print(df.describe())
    
    # Визуализация распределений
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Распределение {col}')
        plt.show()
    
    # Корреляционная матрица
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm')
    plt.title('Корреляционная матрица')
    plt.show()

# Подготовка данных
def prepare_data(df):
    # Кодирование категориальных переменных
    df = pd.get_dummies(df, columns=['cut', 'color', 'clarity'])
    
    # Разделение на признаки и целевую переменную
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Построение и оценка модели
def build_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'R²: {r2}')
    
    return model, y_pred

# Анализ важности признаков
def analyze_feature_importance(model, X):
    importance = pd.DataFrame({'feature': X.columns, 'importance': model.coef_})
    importance = importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance)
    plt.title('Важность признаков')
    plt.show()

# Визуализация результатов
def visualize_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Фактические цены')
    plt.ylabel('Предсказанные цены')
    plt.title('Фактические vs Предсказанные цены')
    plt.show()
    
    # Визуализация остатков
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Предсказанные цены')
    plt.ylabel('Остатки')
    plt.title('График остатков')
    plt.show()

# Основная функция
def main():
    df = pd.read_csv('diamonds.csv')
    perform_eda(df)
    X_train, X_test, y_train, y_test = prepare_data(df)
    model, y_pred = build_and_evaluate_model(X_train, X_test, y_train, y_test)
    analyze_feature_importance(model, X_train)
    visualize_results(y_test, y_pred)

if __name__ == "__main__":
    main()
