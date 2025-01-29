# README

## 🚀 Установка и запуск

### 📥 Клонирование репозитория
Сначала клонируйте репозиторий на локальный компьютер:
```bash
git clone https://github.com/ReaGemt/visualization_date.git
cd visualization_date
```

### 📦 Установка зависимостей
Убедитесь, что у вас установлен Python (версия 3.7+), затем установите зависимости:
```bash
pip install -r requirements.txt
```

### ▶️ Запуск скрипта
```bash
python main.py
```
Скрипт автоматически загрузит данные, построит графики и сохранит их в файлы.

## 📌 Описание проекта
Этот проект предназначен для анализа данных о продажах и маркетинговой воронки. Он включает в себя загрузку данных из Excel-файлов, их обработку и визуализацию в виде различных графиков. 

## 📦 Используемые библиотеки
Для работы скрипта требуются следующие библиотеки:
- `logging` - для логирования выполнения
- `datetime` - для работы с датами
- `matplotlib` - для построения графиков
- `seaborn` - для стилизации графиков
- `pandas` - для работы с таблицами данных
- `plotly` - для интерактивных графиков
- `kaleido` - для сохранения изображений с `plotly`
- `selenium` - для взаимодействия с браузером (если потребуется)
- `os` - для работы с файловой системой

## ⚙️ Основные функции
### 📥 Загрузка данных:
```python
load_data(sales_path, funnel_path)
```
Загружает данные о продажах и маркетинговой воронке из Excel-файлов.

### 📊 Построение графиков:
- 📌 `plot_histogram(daily_sales)` - гистограмма распределения продаж по дням
- 📌 `plot_boxplot(daily_category_sales)` - boxplot по категориям товаров
- 📌 `plot_bar_chart(subcategory_sales)` - столбчатая диаграмма по подкатегориям
- 📌 `plot_line_chart(pivot_table)` - линейная диаграмма динамики продаж
- 📌 `plot_pie_chart(category_sales)` - круговая диаграмма долей категорий в продажах
- 📌 `plot_funnel_chart(funnel_data, filename)` - воронка продаж
- 📌 `plot_price_distribution(df_sales)` - распределение цен товаров
- 📌 `plot_delivery_time_trend(df_sales)` - анализ времени доставки
- 📌 `plot_rating_boxplot(df_sales)` - boxplot рейтинга товаров
- 📌 `plot_payment_method_pie_chart(df_sales)` - анализ способов оплаты

## 📝 Логирование
Вся информация о выполнении скрипта сохраняется в файле `sales_analysis.log`. Ошибки логируются и дублируются в консоль.

## 📂 Выходные файлы
После выполнения скрипта в папке проекта появятся файлы с изображениями построенных графиков, например:
- 🖼️ `output_histogram.png`
- 🖼️ `output_boxplot.png`
- 🖼️ `output_bar_chart.png`
- 🖼️ `output_line_chart.png`
- 🖼️ `output_pie_chart.png`
- 🖼️ `output_funnel_chart.png`
- 🖼️ `output_price_distribution.png`
- 🖼️ `output_delivery_time_trend.png`
- 🖼️ `output_rating_boxplot.png`
- 🖼️ `output_payment_method_pie_chart.png`

## 🔧 Требования
Перед запуском убедитесь, что установлены все зависимости. Для установки используйте:
```bash
pip install -r requirements.txt
```

## 👨‍💻 Автор
Разработчик: **[ReaGemt]**

