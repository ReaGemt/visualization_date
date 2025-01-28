import logging
from datetime import time
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.io import write_image
import plotly
import kaleido
from selenium import webdriver
import os

print("Plotly version:", plotly.__version__)
print("Kaleido version:", kaleido.__version__)

matplotlib.use('Agg')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sales_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_data(sales_path, funnel_path):
    """Загрузка данных из Excel-файлов с логированием."""
    logger.info(f"Начало загрузки данных: {sales_path}, {funnel_path}")
    try:
        df_sales = pd.read_excel(sales_path)
        df_funnel = pd.read_excel(funnel_path)
        logger.info(f"Успешно загружено {len(df_sales)} строк продаж и {len(df_funnel)} строк воронки")
        return df_sales, df_funnel
    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {str(e)}", exc_info=True)
        raise


def plot_histogram(daily_sales):
    """Построение гистограммы с логированием."""
    logger.info("Начало построения гистограммы продаж по дням")
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(daily_sales['Общая сумма'], bins=10, edgecolor='black', color='skyblue')
        plt.title('Распределение общей суммы продаж по дням')
        plt.xlabel('Общая сумма продаж (в рублях)')
        plt.ylabel('Частота')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('output_histogram.png')
        logger.info("Гистограмма сохранена в output_histogram.png")
        plt.close()
    except Exception as e:
        logger.error(f"Ошибка построения гистограммы: {str(e)}", exc_info=True)
        raise


def plot_boxplot(daily_category_sales):
    """Построение boxplot с логированием."""
    logger.info("Начало построения boxplot по категориям")
    try:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Категория', y='Общая сумма', data=daily_category_sales)
        plt.title('Распределение общей суммы продаж по категориям товаров')
        plt.xlabel('Категория')
        plt.ylabel('Общая сумма продаж (в рублях)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('output_boxplot.png')
        logger.info("Boxplot сохранен в output_boxplot.png")
        plt.close()
    except Exception as e:
        logger.error(f"Ошибка построения boxplot: {str(e)}", exc_info=True)
        raise


def plot_bar_chart(subcategory_sales):
    """Построение столбчатой диаграммы с логированием."""
    logger.info("Начало построения столбчатой диаграммы по подкатегориям")
    try:
        plt.figure(figsize=(12, 8))
        plt.bar(subcategory_sales['Подкатегория'], subcategory_sales['Общая сумма'],
                color='skyblue', edgecolor='black')
        plt.title('Суммарные продажи по подкатегориям')
        plt.xlabel('Подкатегория')
        plt.ylabel('Общая сумма продаж (в рублях)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('output_bar_chart.png')
        logger.info("Столбчатая диаграмма сохранена в output_bar_chart.png")
        plt.close()
    except Exception as e:
        logger.error(f"Ошибка построения столбчатой диаграммы: {str(e)}", exc_info=True)
        raise


def plot_line_chart(pivot_table):
    """Построение линейной диаграммы с логированием."""
    logger.info("Начало построения линейной диаграммы по категориям")
    try:
        plt.figure(figsize=(14, 8))
        pivot_table.plot(kind='line', marker='o', figsize=(14, 8))
        plt.title('Изменение выручки по дням в каждой категории')
        plt.xlabel('Дата продажи')
        plt.ylabel('Общая сумма продаж (в рублях)')
        plt.legend(title='Категория')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('output_line_chart.png')
        logger.info("Линейная диаграмма сохранена в output_line_chart.png")
        plt.close()
    except Exception as e:
        logger.error(f"Ошибка построения линейной диаграммы: {str(e)}", exc_info=True)
        raise


def plot_pie_chart(category_sales):
    """Построение круговой диаграммы с логированием."""
    logger.info("Начало построения круговой диаграммы по категориям")
    try:
        plt.figure(figsize=(10, 8))
        plt.pie(category_sales, labels=category_sales.index,
                autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        plt.title('Доли категорий в общей выручке')
        plt.savefig('output_pie_chart.png')
        logger.info("Круговая диаграмма сохранена в output_pie_chart.png")
        plt.close()
    except Exception as e:
        logger.error(f"Ошибка построения круговой диаграммы: {str(e)}", exc_info=True)
        raise


def plot_funnel_chart(funnel_data, filename="output_funnel_chart.html"):
    """
    Построение и сохранение маркетинговой воронки в HTML с оптимизированными цветами.

    Параметры:
        funnel_data (pd.DataFrame или dict): Данные, содержащие минимум два столбца:
            'Шаг' – названия этапов воронки (ось Y),
            'Количество пользователей' – значения (ось X).
        filename (str): Путь и имя выходного файла HTML. По умолчанию "output_funnel_chart.html".
    """
    logger.info("Начало построения маркетинговой воронки (HTML)")

    required_columns = {'Шаг', 'Количество пользователей'}
    if not required_columns.issubset(funnel_data.columns):
        logger.error(f"Не найдены требуемые столбцы: {required_columns}")
        return

    try:
        # Более мягкая цветовая гамма (пример пастельных или «business-friendly» тонов)
        color_palette = [
            "#87CEEB",  # светло-синий
            "#FFD700",  # золотистый
            "#ADFF2F",  # зеленовато-салатовый
            "#FFB6C1",  # пастельно-розовый
            "#FF8C00",  # оранжевый оттенок
            "#B0C4DE"  # серовато-голубой
        ]

        # Построение фигуры (воронки)
        fig = go.Figure(go.Funnel(
            y=funnel_data['Шаг'],
            x=funnel_data['Количество пользователей'],
            textposition="inside",
            textinfo="value+percent initial",
            marker=dict(color=color_palette),
            connector=dict(line=dict(color="lightgray", width=2, dash="dot"))
        ))

        # Обновление макета для приятного внешнего вида
        fig.update_layout(
            title="Маркетинговая воронка",
            yaxis=dict(title="Этапы воронки"),
            xaxis=dict(title="Количество пользователей"),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=12)
        )

        logger.info(f"Попытка сохранить воронку в файл {filename}")
        fig.write_html(filename, auto_open=False)
        logger.info(f"Воронка успешно сохранена в {filename}")

    except Exception as e:
        logger.error(f"Ошибка при сохранении воронки в HTML: {str(e)}", exc_info=True)
        # В случае ошибки всё ещё можем отобразить график интерактивно
        fig.show()
        logger.info("Воронка отображена интерактивно из-за ошибки.")


def plot_price_distribution(df_sales):
    """Построение распределения цен с логированием."""
    logger.info("Начало построения распределения цен")
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(df_sales['Цена за единицу'], bins=20, edgecolor='black', color='orange')
        plt.title('Распределение цены за единицу')
        plt.xlabel('Цена за единицу (в рублях)')
        plt.ylabel('Частота')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('output_price_distribution.png')
        logger.info("Гистограмма цен сохранена в output_price_distribution.png")
        plt.close()
    except Exception as e:
        logger.error(f"Ошибка построения распределения цен: {str(e)}", exc_info=True)
        raise


def plot_delivery_time_trend(df_sales):
    """Построение тренда времени доставки с логированием."""
    logger.info("Начало анализа времени доставки")
    try:
        avg_delivery_time = df_sales.groupby('Дата продажи')['Время доставки (дни)'].mean().reset_index()
        plt.figure(figsize=(12, 6))
        plt.plot(avg_delivery_time['Дата продажи'],
                 avg_delivery_time['Время доставки (дни)'],
                 marker='o', color='green')
        plt.title('Изменение среднего времени доставки по дням')
        plt.xlabel('Дата продажи')
        plt.ylabel('Среднее время доставки (в днях)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('output_delivery_time_trend.png')
        logger.info("График времени доставки сохранен в output_delivery_time_trend.png")
        plt.close()
    except Exception as e:
        logger.error(f"Ошибка анализа времени доставки: {str(e)}", exc_info=True)
        raise


def plot_rating_boxplot(df_sales):
    """Построение рейтинга товаров с логированием."""
    logger.info("Начало анализа рейтинга товаров")
    try:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Категория', y='Рейтинг товара', data=df_sales)
        plt.title('Размах рейтинга товаров между категориями')
        plt.xlabel('Категория')
        plt.ylabel('Рейтинг товара')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('output_rating_boxplot.png')
        logger.info("График рейтинга сохранен в output_rating_boxplot.png")
        plt.close()
    except Exception as e:
        logger.error(f"Ошибка анализа рейтинга товаров: {str(e)}", exc_info=True)
        raise


def plot_payment_method_pie_chart(df_sales):
    """Построение диаграммы способов оплаты с логированием."""
    logger.info("Начало анализа способов оплаты")
    try:
        payment_sales = df_sales.groupby('Способ оплаты')['Общая сумма'].sum()
        plt.figure(figsize=(10, 8))
        plt.pie(payment_sales, labels=payment_sales.index,
                autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set2'))
        plt.title('Доли способов оплаты в общей сумме продаж')
        plt.savefig('output_payment_method_pie_chart.png')
        logger.info("Диаграмма оплат сохранена в output_payment_method_pie_chart.png")
        plt.close()
    except Exception as e:
        logger.error(f"Ошибка анализа способов оплаты: {str(e)}", exc_info=True)
        raise


def main():
    """Основная функция с полным логированием процесса."""
    logger.info("=== НАЧАЛО РАБОТЫ СКРИПТА ===")
    try:
        sales_path = 'df_sales.xlsx'
        funnel_path = 'df_funnel.xlsx'

        # Загрузка данных
        df_sales, df_funnel = load_data(sales_path, funnel_path)

        # Анализ данных
        logger.info("Этап 1/10: Гистограмма продаж по дням")
        daily_sales = df_sales.groupby('Дата продажи')['Общая сумма'].sum().reset_index()
        plot_histogram(daily_sales)

        logger.info("Этап 2/10: Boxplot по категориям")
        daily_category_sales = df_sales.groupby(['Дата продажи', 'Категория'])['Общая сумма'].sum().reset_index()
        plot_boxplot(daily_category_sales)

        logger.info("Этап 3/10: Столбчатая диаграмма по подкатегориям")
        subcategory_sales = df_sales.groupby('Подкатегория')['Общая сумма'].sum().reset_index()
        subcategory_sales = subcategory_sales.sort_values(by='Общая сумма', ascending=False)
        plot_bar_chart(subcategory_sales)

        logger.info("Этап 4/10: Линейная диаграмма по категориям")
        pivot_table = daily_category_sales.pivot(index='Дата продажи', columns='Категория', values='Общая сумма')
        plot_line_chart(pivot_table)

        logger.info("Этап 5/10: Круговая диаграмма по категориям")
        category_sales = df_sales.groupby('Категория')['Общая сумма'].sum()
        plot_pie_chart(category_sales)

        logger.info("Этап 6/10: Маркетинговая воронка")
        funnel_data = df_funnel.sort_values(by='Количество пользователей', ascending=False)
        logger.info("Перед вызовом plot_funnel_chart")
        plot_funnel_chart(funnel_data)
        logger.info("После вызова plot_funnel_chart")

        logger.info("Этап 7/10: Анализ цен")
        plot_price_distribution(df_sales)

        logger.info("Этап 8/10: Анализ времени доставки")
        plot_delivery_time_trend(df_sales)

        logger.info("Этап 9/10: Анализ рейтинга товаров")
        plot_rating_boxplot(df_sales)

        logger.info("Этап 10/10: Анализ способов оплаты")
        plot_payment_method_pie_chart(df_sales)

        logger.info("=== АНАЛИЗ УСПЕШНО ЗАВЕРШЕН ===")

    except Exception as e:
        logger.critical("КРИТИЧЕСКАЯ ОШИБКА ВЫПОЛНЕНИЯ СКРИПА: %s", str(e), exc_info=True)
        raise


if __name__ == "__main__":
    main()
