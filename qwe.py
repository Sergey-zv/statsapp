import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import math


st.title('Приложение для подготовки по статистике (по вопросам)')

# Создаем боковое меню с вопросами
question = st.sidebar.selectbox(
    "Выберите вопрос:",
    [
        "1. Генеральная и выборочная совокупности",
        "2. Статистическое распределение выборки",
        "3. Точечная и интервальная оценки",
        "4. Генеральная и выборочная характеристики",
        "5. Выборочные моменты и коэффициенты",
        "6. Доверительные интервалы для нормального распределения",
        "7. Доверительный интервал для вероятности",
        "8. Групповые и общие характеристики",
        "9. Структурные характеристики выборки",
        "10. Метод моментов",
        "11. Метод наибольшего правдоподобия",
        "12. Закон распределения вероятностей системы двух дискретных случайных величин. Построение законов распределения составляющих. Условные законы распределения составляющих системы дискретных случайных величин. Функциональная, статистическая и корреляционная зависимости. Корреляционная таблица. Условные средние значения.",
        "13. Теоретическое уравнение регрессии.",
        "14. Выборочное уравнение регрессии. Метод наименьших квадратов (МНК)",
        "15. Выборочный корреляционный момент. Его смысл, расчетные формулы и свойства. Выборочный коэффициент корреляции. Его смысл, расчетные формулы и свойства.",
        "16. Выборочное корреляционное отношение",
        "17. Ранговая корреляция. Коэффициенты ранговой корреляции Спирмена и Кендалла. Предельные значения для коэффициентов ранговой корреляции.",
        "18. Коэффициент конкордации. Смысл. Диапазон изменения. Методика расчета.",
        "19. Случайные числа. Разыгрывание дискретной случайной величины.",
        "20. Разыгрывание непрерывной случайной величины. Метод суперпозиции.",
        "21. Приближённое разыгрывание нормальной случайной величины.",
        "22. Сущность метода Монте-Карло. Оценка его погрешности.",
        "23. Вычисление определённых интегралов с помощью метода Монте-Карло."
    ]
)

if question == "1. Генеральная и выборочная совокупности":
    st.header("1. Генеральная и выборочная совокупности")
    
    st.subheader("Определения")
    st.markdown("""
    - **Генеральная совокупность** - полная совокупность объектов исследования
    - **Выборочная совокупность** - часть генеральной совокупности, отобранная для изучения
    - **Объём совокупности** - число элементов в совокупности (N - генеральной, n - выборочной)
    """)
    
    st.subheader("Типы выборок")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**По способу отбора:**")
        st.markdown("""
        - Повторная (с возвращением)
        - Бесповторная (без возвращения)
        """)
    
    with col2:
        st.markdown("**По методу формирования:**")
        st.markdown("""
        - Простой случайный
        - Типический (стратифицированный)
        - Механический (систематический)
        - Серийный (гнездовой)
        """)
    
    st.subheader("Подробнее о методах формирования выборок")
    st.markdown("""
    - **Простой случайный отбор** – отбор, при котором объекты извлекаются по одному из всей генеральной совокупности
    - **Типический отбор** – отбор, при котором объекты извлекаются не из всей генеральной совокупности, а из каждой её типической части
    - **Механический отбор** – отбор, при котором извлекается каждый n-й объект
    - **Серийный отбор** – отбор, при котором объекты извлекаются сериями из всей генеральной совокупности
    """)
    
    st.subheader("Формулы объема выборки")
    st.latex(r"n = \frac{t^2 \sigma^2 N}{\Delta^2 N + t^2 \sigma^2} \quad \text{(бесповторный отбор)}")
    st.latex(r"n = \frac{t^2 \sigma^2}{\Delta^2} \quad \text{(повторный отбор)}")
    st.markdown("где:")
    st.markdown("- t - коэффициент доверия (например, 1.96 для 95% доверительного интервала)")
    st.markdown("- σ² - дисперсия признака")
    st.markdown("- Δ - предельная ошибка выборки")
    st.markdown("- N - объем генеральной совокупности")
    
    if st.checkbox("Показать практический пример", key="practical_1"):
        st.subheader("Практический пример расчета объема выборки")
        
        st.markdown("**Условие:**")
        st.markdown("""
        - Генеральная совокупность: N = 10,000
        - Ожидаемая дисперсия: σ² = 25
        - Доверительный уровень: 95% (t = 1.96)
        - Предельная ошибка: Δ = 1.5
        """)
        
        st.markdown("**Решение для бесповторного отбора:**")
        N = 10000
        t = 1.96
        sigma2 = 25
        delta = 1.5
        
        numerator = t**2 * sigma2 * N
        denominator = delta**2 * N + t**2 * sigma2
        n = numerator / denominator
        
        st.latex(fr"n = \frac{{{t**2:.4f} \times {sigma2} \times {N}}}{{{delta**2} \times {N} + {t**2:.4f} \times {sigma2}}} = \frac{{{numerator:.2f}}}{{{denominator:.2f}}} = {n:.2f}")
        st.markdown(f"**Результат:** Необходимый объем выборки ≈ {math.ceil(n)} элементов")

elif question == "2. Статистическое распределение выборки":
    st.header("2. Статистическое распределение выборки")
    
    st.subheader("Основные понятия")
    st.markdown("""
    - **Варианты (xᵢ)** - отдельные значения признака в выборке
    - **Частоты (nᵢ)** - количество появлений каждой варианты
    - **Относительные частоты (wᵢ)** - доли частот от общего объема выборки:
    """)
    st.latex(r"w_i = \frac{n_i}{n}")
    
    st.subheader("Эмпирическая функция распределения")
    st.latex(r"F^*(x) = \frac{\text{число } x_i < x}{n}")
    st.markdown("Свойства:")
    st.markdown("- Неубывающая")
    st.markdown("- 0 ≤ F*(x) ≤ 1")
    st.markdown("- F*(-∞) = 0, F*(+∞) = 1")
    
    st.subheader("Графические представления")
    st.markdown("**Многоугольник частот** - ломаная, соединяющая точки (xᵢ, nᵢ)")
    st.markdown("**Гистограмма** - столбчатая диаграмма плотности частот")
    
    if st.checkbox("Показать практический пример", key="practical_2"):
        st.subheader("Практический пример построения распределения")
        
        # Генерация данных
        np.random.seed(42)
        data = np.round(np.random.normal(10, 2, 50), 1)
        
        st.markdown("**Исходные данные (первые 10 значений):**")
        st.write(pd.DataFrame(data[:10], columns=['Значение']))
        
        # Расчет частот
        values, counts = np.unique(data, return_counts=True)
        df = pd.DataFrame({
            'Значение': values, 
            'Частота (n_i)': counts, 
            'Относительная частота (w_i)': np.round(counts/len(data), 3)
        })
        
        st.markdown("**Таблица частотного распределения:**")
        st.write(df)
        
        st.markdown("**Расчет относительных частот:**")
        st.latex(r"w_i = \frac{n_i}{n} = \frac{\text{Частота}}{\text{Объем выборки}}")
        for i in range(len(values)):
            st.latex(fr"w_{{{i}}} = \frac{{{counts[i]}}}{{{len(data)}}} = {counts[i]/len(data):.3f}")
        
        # Визуализация
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Многоугольник частот
        ax1.plot(values, counts, 'bo-')
        ax1.set_title('Многоугольник частот')
        ax1.set_xlabel('Значение')
        ax1.set_ylabel('Частота')
        
        # Гистограмма
        ax2.hist(data, bins=10, edgecolor='black')
        ax2.set_title('Гистограмма частот')
        ax2.set_xlabel('Значение')
        ax2.set_ylabel('Частота')
        
        st.pyplot(fig)

elif question == "3. Точечная и интервальная оценки":
    st.header("3. Точечная и интервальная оценки")
    
    st.subheader("Типы оценок")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Точечная оценка**")
        st.latex(r"\theta^* \text{ - одно число, приближающее } \theta")
    
    with col2:
        st.markdown("**Интервальная оценка**")
        st.latex(r"[\theta_1^*, \theta_2^*] \text{ - интервал, покрывающий } \theta")
    
    st.subheader("Критерии качества оценок")
    
    criteria = st.radio("Выберите критерий:", 
                       ["Несмещённость", "Эффективность", "Состоятельность"])
    
    if criteria == "Несмещённость":
        st.latex(r"M(\theta^*) = \theta")
        st.markdown("Пример: выборочное среднее для генерального среднего")
    
    elif criteria == "Эффективность":
        st.markdown("Оценка с минимальной дисперсией среди всех несмещённых")
        st.latex(r"D(\theta^*) \rightarrow \min")
    
    elif criteria == "Состоятельность":
        st.latex(r"\forall \varepsilon > 0: \lim_{n\to\infty} P(|\theta^* - \theta| > \varepsilon) = 0")
        st.markdown("Сходимость по вероятности при n → ∞")
    
    if st.checkbox("Показать практический пример", key="practical_3"):
        st.subheader("Практический пример точечной оценки")
        
        # Генерация данных
        np.random.seed(42)
        true_mean = 10
        true_std = 2
        data = np.random.normal(true_mean, true_std, 100)
        
        st.markdown("**Исходные данные:** Нормальное распределение с μ = 10, σ = 2")
        st.markdown(f"**Объем выборки:** n = {len(data)}")
        
        # Точечная оценка
        sample_mean = np.mean(data)
        sample_var = np.var(data)
        unbiased_var = np.var(data, ddof=1)
        
        st.markdown("**1. Точечная оценка среднего:**")
        st.latex(r"\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i")
        st.latex(fr"\bar{{x}} = \frac{{1}}{{{len(data)}}}\sum_{{i=1}}^{{{len(data)}}} x_i = {sample_mean:.4f}")
        
        st.markdown("**2. Точечная оценка дисперсии (смещенная):**")
        st.latex(r"D_B = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2")
        st.latex(fr"D_B = \frac{{1}}{{{len(data)}}}\sum_{{i=1}}^{{{len(data)}}} (x_i - {sample_mean:.4f})^2 = {sample_var:.4f}")
        
        st.markdown("**3. Исправленная оценка дисперсии:**")
        st.latex(r"s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2")
        st.latex(fr"s^2 = \frac{{1}}{{{len(data)-1}}}\sum_{{i=1}}^{{{len(data)}}} (x_i - {sample_mean:.4f})^2 = {unbiased_var:.4f}")
        
        st.markdown("**Сравнение с истинными значениями:**")
        comp_df = pd.DataFrame({
            'Параметр': ['Среднее', 'Дисперсия'],
            'Истинное значение': [true_mean, true_std**2],
            'Выборочная оценка': [sample_mean, sample_var],
            'Исправленная оценка': [sample_mean, unbiased_var]
        })
        st.write(comp_df)

elif question == "4. Генеральная и выборочная характеристики":
    st.header("4. Генеральная и выборочная характеристики")
    
    st.subheader("Средние значения")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Генеральная средняя**")
        st.latex(r"\bar{x}_r = \frac{1}{N}\sum_{j=1}^N x_j = M(X)")
    
    with col2:
        st.markdown("**Выборочная средняя**")
        st.latex(r"\bar{x}_B = \frac{1}{n}\sum_{j=1}^n x_j")
    
    st.subheader("Дисперсии")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Генеральная дисперсия**")
        st.latex(r"D_r = \frac{1}{N}\sum_{j=1}^N (x_j - \bar{x}_r)^2 = D(X)")
    
    with col2:
        st.markdown("**Выборочная дисперсия**")
        st.latex(r"D_B = \frac{1}{n}\sum_{j=1}^n (x_j - \bar{x}_B)^2")
        st.markdown("Смещённая оценка:")
        st.latex(r"M(D_B) = \frac{n-1}{n}D_r")
    
    st.subheader("Исправленные оценки")
    st.latex(r"s^2 = \frac{n}{n-1}D_B = \frac{1}{n-1}\sum_{j=1}^n (x_j - \bar{x}_B)^2")
    st.latex(r"s = \sqrt{s^2} = \sqrt{\frac{n}{n-1}}\sigma_B")
    
    if st.checkbox("Показать практический пример", key="practical_4"):
        st.subheader("Практический пример расчета характеристик")
        
        # Генерация данных
        np.random.seed(42)
        true_mean = 10
        true_var = 4
        data = np.random.normal(true_mean, np.sqrt(true_var), 100)
        
        st.markdown("**Исходные данные:** Нормальное распределение с μ = 10, σ² = 4")
        st.markdown(f"**Объем выборки:** n = {len(data)}")
        
        # Расчет характеристик
        sample_mean = np.mean(data)
        sample_var = np.var(data)
        unbiased_var = np.var(data, ddof=1)
        sample_std = np.std(data)
        unbiased_std = np.std(data, ddof=1)
        
        st.markdown("**1. Выборочное среднее:**")
        st.latex(fr"\bar{{x}} = \frac{{1}}{{{len(data)}}}\sum_{{i=1}}^{{{len(data)}}} x_i = {sample_mean:.4f}")
        
        st.markdown("**2. Выборочная дисперсия (смещенная):**")
        st.latex(fr"D_B = \frac{{1}}{{{len(data)}}}\sum_{{i=1}}^{{{len(data)}}} (x_i - \bar{{x}})^2 = {sample_var:.4f}")
        
        st.markdown("**3. Исправленная дисперсия:**")
        st.latex(fr"s^2 = \frac{{1}}{{{len(data)-1}}}\sum_{{i=1}}^{{{len(data)}}} (x_i - \bar{{x}})^2 = {unbiased_var:.4f}")
        
        st.markdown("**4. Выборочное стандартное отклонение:**")
        st.latex(fr"\sigma_B = \sqrt{{D_B}} = \sqrt{{{sample_var:.4f}}} = {sample_std:.4f}")
        
        st.markdown("**5. Исправленное стандартное отклонение:**")
        st.latex(fr"s = \sqrt{{s^2}} = \sqrt{{{unbiased_var:.4f}}} = {unbiased_std:.4f}")
        
        st.markdown("**Сравнение с генеральными параметрами:**")
        comparison = pd.DataFrame({
            'Параметр': ['Среднее', 'Дисперсия', 'Стандартное отклонение'],
            'Генеральное': [true_mean, true_var, np.sqrt(true_var)],
            'Выборочное': [sample_mean, sample_var, sample_std],
            'Исправленное': [sample_mean, unbiased_var, unbiased_std]
        })
        st.write(comparison)
        
        # Визуализация
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.linspace(min(data), max(data), 100)
        ax.hist(data, bins=15, density=True, alpha=0.6, label='Выборка')
        ax.plot(x, stats.norm.pdf(x, true_mean, np.sqrt(true_var)), 
                'r-', lw=2, label='Истинное распределение')
        ax.plot(x, stats.norm.pdf(x, sample_mean, unbiased_std), 
                'g--', lw=2, label='Оценка по выборке')
        ax.legend()
        ax.set_title('Сравнение истинного распределения и оценок')
        st.pyplot(fig)

elif question == "5. Выборочные моменты и коэффициенты":
    st.header("5. Выборочные моменты и коэффициенты")
    
    st.subheader("Выборочные моменты")
    st.markdown("**Начальный момент k-го порядка:**")
    st.latex(r"\nu_k = \frac{1}{n}\sum_{i=1}^n x_i^k")
    
    st.markdown("**Центральный момент k-го порядка:**")
    st.latex(r"\mu_k = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^k")
    
    st.subheader("Коэффициенты")
    st.markdown("**Коэффициент асимметрии:**")
    st.latex(r"A = \frac{\mu_3}{\sigma^3}")
    
    st.markdown("**Коэффициент эксцесса:**")
    st.latex(r"E = \frac{\mu_4}{\sigma^4} - 3")
    
    if st.checkbox("Показать практический пример", key="practical_5"):
        st.subheader("Практический пример расчета моментов")
        
        # Генерация данных
        np.random.seed(42)
        data = np.random.normal(10, 2, 100)
        
        st.markdown("**Исходные данные:** Нормальное распределение с μ = 10, σ = 2")
        st.markdown(f"**Объем выборки:** n = {len(data)}")
        
        # Расчет моментов
        mean = np.mean(data)
        var = np.var(data)
        std = np.std(data)
        
        # Начальные моменты
        nu1 = np.mean(data)
        nu2 = np.mean(data**2)
        nu3 = np.mean(data**3)
        
        # Центральные моменты
        mu2 = np.mean((data - mean)**2)
        mu3 = np.mean((data - mean)**3)
        mu4 = np.mean((data - mean)**4)
        
        # Коэффициенты
        A = mu3 / std**3
        E = mu4 / std**4 - 3
        
        st.markdown("**1. Начальные моменты:**")
        st.latex(fr"\nu_1 = \frac{{1}}{{n}}\sum x_i = {nu1:.4f}")
        st.latex(fr"\nu_2 = \frac{{1}}{{n}}\sum x_i^2 = {nu2:.4f}")
        st.latex(fr"\nu_3 = \frac{{1}}{{n}}\sum x_i^3 = {nu3:.4f}")
        
        st.markdown("**2. Центральные моменты:**")
        st.latex(fr"\mu_2 = \frac{{1}}{{n}}\sum (x_i - \bar{{x}})^2 = {mu2:.4f}")
        st.latex(fr"\mu_3 = \frac{{1}}{{n}}\sum (x_i - \bar{{x}})^3 = {mu3:.4f}")
        st.latex(fr"\mu_4 = \frac{{1}}{{n}}\sum (x_i - \bar{{x}})^4 = {mu4:.4f}")
        
        st.markdown("**3. Коэффициенты:**")
        st.latex(fr"A = \frac{{\mu_3}}{{\sigma^3}} = \frac{{{mu3:.4f}}}{{{std**3:.4f}}} = {A:.4f}")
        st.latex(fr"E = \frac{{\mu_4}}{{\sigma^4}} - 3 = \frac{{{mu4:.4f}}}{{{std**4:.4f}}} - 3 = {E:.4f}")
        
        st.markdown("**Интерпретация:**")
        st.markdown("- Коэффициент асимметрии близок к 0 (симметричное распределение)")
        st.markdown("- Коэффициент эксцесса близок к 0 (нормальное распределение)")



elif question == "6. Доверительные интервалы для нормального распределения":
    st.header("6. Доверительные интервалы для нормального распределения")
    
    st.subheader("Доверительный интервал для среднего")
    st.markdown("**При известной дисперсии σ²:**")
    st.latex(r"\bar{x} \pm z_{1-\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}")
    
    st.markdown("**При неизвестной дисперсии:**")
    st.latex(r"\bar{x} \pm t_{1-\alpha/2}(n-1) \cdot \frac{s}{\sqrt{n}}")
    
    st.subheader("Доверительный интервал для дисперсии")
    st.latex(r"\left( \frac{(n-1)s^2}{\chi^2_{1-\alpha/2}(n-1)}, \frac{(n-1)s^2}{\chi^2_{\alpha/2}(n-1)} \right)")
    
    if st.checkbox("Показать практический пример", key="practical_6"):
        st.subheader("Практический пример расчета доверительных интервалов")
        
        # Генерация данных
        np.random.seed(42)
        true_mean = 10
        true_std = 2
        data = np.random.normal(true_mean, true_std, 30)
        
        st.markdown("**Исходные данные:** Нормальное распределение с μ = 10, σ = 2")
        st.markdown(f"**Объем выборки:** n = {len(data)}")
        
        # Расчет характеристик
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)
        alpha = 0.05
        
        st.markdown("**1. Доверительный интервал для среднего (известная σ):**")
        z = stats.norm.ppf(1 - alpha/2)
        margin = z * true_std / np.sqrt(len(data))
        st.latex(fr"\bar{{x}} \pm z_{{1-\alpha/2}} \cdot \frac{{\sigma}}{{\sqrt{{n}}}}")
        st.latex(fr"{sample_mean:.4f} \pm {z:.4f} \cdot \frac{{{true_std}}}{{\sqrt{{{len(data)}}}}}")
        st.latex(fr"{sample_mean:.4f} \pm {margin:.4f}")
        st.latex(fr"({sample_mean - margin:.4f}, {sample_mean + margin:.4f})")
        
        st.markdown("**2. Доверительный интервал для среднего (неизвестная σ):**")
        t = stats.t.ppf(1 - alpha/2, len(data)-1)
        margin = t * sample_std / np.sqrt(len(data))
        st.latex(fr"\bar{{x}} \pm t_{{1-\alpha/2}}(n-1) \cdot \frac{{s}}{{\sqrt{{n}}}}")
        st.latex(fr"{sample_mean:.4f} \pm {t:.4f} \cdot \frac{{{sample_std:.4f}}}{{\sqrt{{{len(data)}}}}}")
        st.latex(fr"{sample_mean:.4f} \pm {margin:.4f}")
        st.latex(fr"({sample_mean - margin:.4f}, {sample_mean + margin:.4f})")
        
        st.markdown("**3. Доверительный интервал для дисперсии:**")
        chi2_low = stats.chi2.ppf(1 - alpha/2, len(data)-1)
        chi2_high = stats.chi2.ppf(alpha/2, len(data)-1)
        ci_low = (len(data)-1) * sample_std**2 / chi2_low
        ci_high = (len(data)-1) * sample_std**2 / chi2_high
        st.latex(r"\left( \frac{(n-1)s^2}{\chi^2_{1-\alpha/2}(n-1)}, \frac{(n-1)s^2}{\chi^2_{\alpha/2}(n-1)} \right)")
        st.latex(fr"\left( \frac{{{len(data)-1} \times {sample_std**2:.4f}}}{{{chi2_low:.4f}}}, \frac{{{len(data)-1} \times {sample_std**2:.4f}}}{{{chi2_high:.4f}}} \right)")
        st.latex(fr"({ci_low:.4f}, {ci_high:.4f})")

elif question == "7. Доверительный интервал для вероятности":
    st.header("7. Доверительный интервал для вероятности")
    
    st.subheader("Формула доверительного интервала")
    st.latex(r"\hat{p} \pm z_{1 - \alpha/2} \cdot \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}")
    st.markdown("где:")
    st.markdown("- $\hat{p}$ - выборочная доля успехов")
    st.markdown(r"- $z_{1-\alpha/2}$ - квантиль нормального распределения")
    st.markdown("- $n$ - объем выборки")
    
    if st.checkbox("Показать практический пример", key="practical_7"):
        st.subheader("Практический пример расчета ДИ для вероятности")
        
        # Параметры примера
        n = st.slider("Объем выборки (n)", 10, 500, 100)
        k = st.slider("Количество успехов (k)", 0, n, 35)
        confidence = st.slider("Уровень доверия (%)", 90, 99, 95)
        alpha = 1 - confidence/100
        
        st.markdown("**Условие:**")
        st.markdown(f"- Объем выборки: n = {n}")
        st.markdown(f"- Наблюдалось успехов: k = {k}")
        st.markdown(f"- Уровень доверия: {confidence}% (α = {alpha:.3f})")
        
        if k == 0 or k == n:
            st.warning("Для крайних значений (k=0 или k=n) используйте точные методы (Клоппера-Пирсона)")
        else:
            # Расчеты
            p_hat = k / n
            z = stats.norm.ppf(1 - alpha/2)
            margin = z * math.sqrt(p_hat * (1 - p_hat) / n)
            
            st.markdown("**Решение:**")
            st.latex(r"\hat{p} = \frac{k}{n} = \frac{" + str(k) + "}{" + str(n) + "} = " + f"{p_hat:.4f}")
            st.latex(fr"z_{{1-\alpha/2}} = {z:.4f}")
            st.latex(fr"\text{{Погрешность}} = {z:.4f} \cdot \sqrt{{\frac{{{p_hat:.4f} \cdot (1 - {p_hat:.4f})}}{{{n}}}}} = {margin:.4f}")
            st.latex(fr"\text{{ДИ}} = {p_hat:.4f} \pm {margin:.4f}")
            st.latex(fr"({max(0, p_hat - margin):.4f}, {min(1, p_hat + margin):.4f})")
            
            # Визуализация
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.errorbar(p_hat, 0, xerr=margin, fmt='o', 
                      capsize=5, label='Доверительный интервал')
            ax.axvline(x=p_hat, color='blue', linestyle='--', linewidth=0.7)
            ax.set_yticks([])
            ax.set_xlim(0, 1)
            ax.set_xlabel('Вероятность')
            ax.set_title(f'Доверительный интервал для вероятности (уровень доверия {confidence}%)')
            ax.legend()
            st.pyplot(fig)

elif question == "8. Групповые и общие характеристики":
    st.header("8. Групповые и общие характеристики")
    
    st.subheader("Формулы")
    st.markdown("**Общая средняя:**")
    st.latex(r"\bar{x} = \frac{1}{N}\sum_{j=1}^k N_j \bar{x}_j")
    
    st.markdown("**Общая дисперсия (правило сложения):**")
    st.latex(r"D = \frac{1}{N}\sum_{j=1}^k N_j D_j + \frac{1}{N}\sum_{j=1}^k N_j (\bar{x}_j - \bar{x})^2")
    st.markdown("где:")
    st.markdown("- $N_j$ - объем j-й группы")
    st.markdown(r"- $\bar{x}_j$ - среднее j-й группы")
    st.markdown("- $D_j$ - дисперсия j-й группы")
    
    if st.checkbox("Показать практический пример", key="practical_8"):
        st.subheader("Практический пример расчета групповых характеристик")
        
        # Параметры групп
        n_groups = st.slider("Количество групп", 2, 5, 3)
        
        # Создаем данные для групп
        np.random.seed(42)
        groups = []
        means = np.random.uniform(5, 20, n_groups)
        stds = np.random.uniform(1, 5, n_groups)
        sizes = np.random.randint(30, 100, n_groups)
        
        for i in range(n_groups):
            groups.append(np.random.normal(means[i], stds[i], sizes[i]))
        
        st.markdown("**Исходные данные по группам:**")
        df = pd.DataFrame({
            'Группа': [f'Группа {i+1}' for i in range(n_groups)],
            'Объем (N_j)': [len(g) for g in groups],
            'Среднее (x̄_j)': [np.mean(g) for g in groups],
            'Дисперсия (D_j)': [np.var(g, ddof=0) for g in groups]
        })
        st.write(df)
        
        # Расчет общих характеристик
        total_N = sum(len(g) for g in groups)
        total_mean = sum(len(g)*np.mean(g) for g in groups) / total_N
        
        # Внутригрупповая дисперсия
        within_var = sum(len(g)*np.var(g, ddof=0) for g in groups) / total_N
        
        # Межгрупповая дисперсия
        between_var = sum(len(g)*(np.mean(g)-total_mean)**2 for g in groups) / total_N
        
        total_var = within_var + between_var
        
        st.markdown("**Расчет общих характеристик:**")
        st.latex(fr"\bar{{x}} = \frac{{1}}{{N}}\sum N_j \bar{{x}}_j = \frac{{1}}{{{total_N}}} \cdot " + 
                 " + ".join([f"{len(g)} × {np.mean(g):.2f}" for g in groups]) + 
                 fr" = {total_mean:.4f}")
        
        st.markdown("**Разложение дисперсии:**")
        st.latex(fr"\text{{Внутригрупповая дисперсия}} = \frac{{1}}{{N}}\sum N_j D_j = {within_var:.4f}")
        st.latex(fr"\text{{Межгрупповая дисперсия}} = \frac{{1}}{{N}}\sum N_j (\bar{{x}}_j - \bar{{x}})^2 = {between_var:.4f}")
        st.latex(fr"\text{{Общая дисперсия}} = {within_var:.4f} + {between_var:.4f} = {total_var:.4f}")
        
        # Визуализация
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, g in enumerate(groups):
            ax.hist(g, bins=15, alpha=0.5, label=f'Группа {i+1}')
        ax.axvline(total_mean, color='red', linestyle='--', label='Общее среднее')
        ax.set_title('Распределение данных по группам')
        ax.legend()
        st.pyplot(fig)

elif question == "9. Структурные характеристики выборки":
    st.header("9. Структурные характеристики выборки")
    
    st.subheader("Основные характеристики")
    st.markdown("**Медиана:** серединное значение упорядоченного ряда")
    st.markdown("**Квартели:** Q1 (25%), Q2 (медиана), Q3 (75%)")
    st.markdown("**Мода:** наиболее часто встречающееся значение")
    
    st.subheader("Формулы")
    st.markdown("**Для медианы:**")
    st.latex(r"Me = \begin{cases} x_{(n+1)/2} & \text{если } n \text{ нечетное} \\ \frac{x_{n/2} + x_{n/2+1}}{2} & \text{если } n \text{ четное}\end{cases}")
    
    if st.checkbox("Показать практический пример", key="practical_9"):
        st.subheader("Практический пример расчета структурных характеристик")
        
        # Генерация данных
        np.random.seed(42)
        sample_size = st.slider("Объем выборки", 10, 200, 101)
        data = np.random.normal(10, 2, sample_size)
        sorted_data = np.sort(data)
        
        st.markdown("**Исходные данные (первые 10 значений):**")
        st.write(pd.DataFrame(sorted_data[:10], columns=['Значение']))
        
        # Расчет характеристик
        median = np.median(data)
        q1, q3 = np.percentile(data, [25, 75])
        
        # Безопасный расчет моды
        values, counts = np.unique(data, return_counts=True)
        max_count = np.max(counts)
        if max_count == 1:
            st.warning("Все значения уникальны - мода не определена")
            mode_value = None
            mode_count = 0
        else:
            mode_value = values[np.argmax(counts)]
            mode_count = max_count
        
        st.markdown("**1. Медиана:**")
        if sample_size % 2 == 1:
            st.latex(fr"n = {sample_size} \text{{ (нечетное)}}")
            st.latex(fr"Me = x_{{(n+1)/2}} = x_{{{(sample_size+1)//2}}} = {median:.4f}")
        else:
            st.latex(fr"n = {sample_size} \text{{ (четное)}}")
            st.latex(fr"Me = \frac{{x_{{n/2}} + x_{{n/2+1}}}}{{2}} = \frac{{x_{{{sample_size//2}}} + x_{{{sample_size//2 + 1}}}}}{{2}} = {median:.4f}")
        
        st.markdown("**2. Квартели:**")
        st.latex(fr"Q1 = {q1:.4f}, \quad Q3 = {q3:.4f}")
        st.latex(fr"IQR = Q3 - Q1 = {q3 - q1:.4f}")
        
        # Демонстрация расчета квартилей на случайной подвыборке
        st.subheader("\nПошаговый расчет квартилей на случайной подвыборке")
        
        # Случайный выбор 20 значений (без повторений)
        if len(data) >= 20:
            random_indices = np.random.choice(len(data), size=20, replace=False)
            sample_data = np.sort(data[random_indices])
        else:
            sample_data = sorted_data  # Если выборка меньше 20, берем все
            
        st.markdown(f"**Случайно выбранные {len(sample_data)} значений (отсортированные):**")
        st.write(sample_data)

        # Расчет позиций
        n = len(sample_data)
        q1_pos = 0.25 * (n - 1)
        q3_pos = 0.75 * (n - 1)

        st.markdown(f"""
        **Позиции квартилей:**
        - Q1 позиция: 0.25 × (n-1) = 0.25 × {n-1} = {q1_pos:.2f}
        - Q3 позиция: 0.75 × (n-1) = 0.75 × {n-1} = {q3_pos:.2f}
        """)

        # Находим соседние значения
        idx_q1 = int(np.floor(q1_pos))
        idx_q3 = int(np.floor(q3_pos))

        st.markdown(f"""
        **Ближайшие значения:**
        - Для Q1: элементы {idx_q1} и {idx_q1+1} → {sample_data[idx_q1]:.2f} и {sample_data[idx_q1+1]:.2f}
        - Для Q3: элементы {idx_q3} и {idx_q3+1} → {sample_data[idx_q3]:.2f} и {sample_data[idx_q3+1]:.2f}
        """)

        # Линейная интерполяция
        frac_q1 = q1_pos - idx_q1
        frac_q3 = q3_pos - idx_q3

        calc_q1 = sample_data[idx_q1] + frac_q1 * (sample_data[idx_q1+1] - sample_data[idx_q1])
        calc_q3 = sample_data[idx_q3] + frac_q3 * (sample_data[idx_q3+1] - sample_data[idx_q3])

        st.markdown(f"""
        **Итоговый расчет:**
        - Q1 = {sample_data[idx_q1]:.2f} + {frac_q1:.2f} × ({sample_data[idx_q1+1]:.2f} - {sample_data[idx_q1]:.2f}) = {calc_q1:.2f}
        - Q3 = {sample_data[idx_q3]:.2f} + {frac_q3:.2f} × ({sample_data[idx_q3+1]:.2f} - {sample_data[idx_q3]:.2f}) = {calc_q3:.2f}
        """)

        # Сравнение с numpy.percentile
        st.markdown("\n**Проверка через numpy.percentile:**")
        st.markdown(f"- np.percentile: Q1 = {np.percentile(sample_data, 25):.2f}, Q3 = {np.percentile(sample_data, 75):.2f}")
        
        st.markdown("**3. Мода:**")
        if mode_value is not None:
            st.latex(fr"\text{{Mode}} = {mode_value:.4f} \text{{ (встречается {mode_count} раз)}}")
        else:
            st.markdown("Мода не определена (все значения уникальны)")
        
        # Визуализация
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Boxplot
        ax1.boxplot(data, vert=False)
        ax1.set_title('Ящик с усами (Boxplot)')
        ax1.set_yticks([])
        
        # Гистограмма с модой
        ax2.hist(data, bins=20, edgecolor='black')
        if mode_value is not None:
            ax2.axvline(mode_value, color='red', linestyle='--', label=f'Мода = {mode_value:.2f}')
        ax2.set_title('Гистограмма с модой')
        ax2.legend()
        
        st.pyplot(fig)

elif question == "10. Метод моментов":
    st.header("10. Метод моментов")
    
    st.subheader("Суть метода")
    st.markdown("""
    Метод моментов - это способ оценки параметров распределения путем приравнивания выборочных моментов 
    к теоретическим моментам распределения. Основные шаги:
    """)
    
    st.subheader("Алгоритм")
    st.markdown("""
    1. Выражаем параметры распределения через теоретические моменты
    2. Вычисляем соответствующие выборочные моменты по данным
    3. Приравниваем теоретические моменты к выборочным
    4. Решаем полученную систему уравнений относительно параметров
    """)
    
    if st.checkbox("Показать практический пример", key="practical_10"):
        st.subheader("Практический пример: оценка параметров гамма-распределения")
        
        # Параметры для генерации данных
        col1, col2 = st.columns(2)
        with col1:
            true_shape = st.slider("Истинный параметр shape (k)", 1.0, 5.0, 3.0, 0.1)
        with col2:
            true_scale = st.slider("Истинный параметр scale (θ)", 1.0, 5.0, 2.0, 0.1)
        
        # Генерация данных
        np.random.seed(42)
        data = np.random.gamma(true_shape, true_scale, 1000)
        
        st.markdown("### Шаг 1: Анализ исходных данных")
        st.markdown(f"**Объем выборки:** n = {len(data)}")
        
        # Визуализация данных
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.hist(data, bins=30, density=True, alpha=0.6)
        ax1.set_title('Гистограмма исходных данных')
        ax1.set_xlabel('Значение')
        ax1.set_ylabel('Плотность')
        st.pyplot(fig1)
        
        st.markdown("### Шаг 2: Вычисление выборочных моментов")
        
        # Первые два выборочных момента
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=0)
        
        st.latex(r"""
        \begin{aligned}
        \text{Первый момент (среднее):} & \quad \bar{x} = \frac{1}{n}\sum_{i=1}^n x_i = %.4f \\
        \text{Второй центральный момент:} & \quad \mu_2 = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2 = %.4f
        \end{aligned}
        """ % (sample_mean, sample_var))
        
        st.markdown("### Шаг 3: Теоретические моменты гамма-распределения")
        
        st.latex(r"""
        \begin{aligned}
        E[X] &= k\theta \\
        Var(X) &= k\theta^2
        \end{aligned}
        """)
        
        st.markdown("### Шаг 4: Составление системы уравнений")
        
        st.markdown("Приравниваем теоретические моменты к выборочным:")
        st.latex(r"""
        \begin{cases}
        k\theta = \bar{x} \\
        k\theta^2 = \mu_2
        \end{cases}
        """)
        
        st.latex(r"""
        \begin{cases}
        k\theta = %.4f \\
        k\theta^2 = %.4f
        \end{cases}
        """ % (sample_mean, sample_var))
        
        st.markdown("### Шаг 5: Решение системы уравнений")
        
        st.markdown("**Из первого уравнения выражаем k:**")
        st.latex(r"k = \frac{\bar{x}}{\theta}")
        
        st.markdown("**Подставляем во второе уравнение:**")
        st.latex(r"\frac{\bar{x}}{\theta} \cdot \theta^2 = \mu_2")
        st.latex(r"\bar{x} \theta = \mu_2")
        st.latex(r"\theta = \frac{\mu_2}{\bar{x}} = \frac{%.4f}{%.4f} = %.4f" % 
                (sample_var, sample_mean, sample_var/sample_mean))
        
        theta_hat = sample_var / sample_mean
        k_hat = sample_mean / theta_hat
        
        st.markdown("**Теперь находим k:**")
        st.latex(r"k = \frac{\bar{x}}{\theta} = \frac{%.4f}{%.4f} = %.4f" % 
                (sample_mean, theta_hat, k_hat))
        
        st.markdown("### Шаг 6: Сравнение с истинными значениями")
        
        # Создаем таблицу сравнения
        comparison = pd.DataFrame({
            'Параметр': ['Форма (k)', 'Масштаб (θ)'],
            'Истинное значение': [true_shape, true_scale],
            'Оценка методом моментов': [k_hat, theta_hat],
            'Относительная ошибка (%)': [
                abs((k_hat - true_shape)/true_shape*100),
                abs((theta_hat - true_scale)/true_scale*100)
            ]
        })
        
        st.table(comparison.style.format({
            'Истинное значение': '{:.3f}',
            'Оценка методом моментов': '{:.3f}',
            'Относительная ошибка (%)': '{:.2f}%'
        }))
        
        st.markdown("### Шаг 7: Визуализация результатов")
        
        # Визуализация распределений
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        x = np.linspace(0, np.max(data)*1.2, 200)
        
        # Истинное распределение
        ax2.plot(x, stats.gamma.pdf(x, a=true_shape, scale=true_scale), 
                'r-', lw=2, label='Истинное распределение')
        
        # Оценка методом моментов
        ax2.plot(x, stats.gamma.pdf(x, a=k_hat, scale=theta_hat), 
                'g--', lw=2, label='Оценка методом моментов')
        
        # Гистограмма данных
        ax2.hist(data, bins=30, density=True, alpha=0.3, label='Выборка')
        
        ax2.set_title('Сравнение истинного распределения и оценки методом моментов')
        ax2.set_xlabel('Значение')
        ax2.set_ylabel('Плотность вероятности')
        ax2.legend()
        
        st.pyplot(fig2)
        
        st.markdown("### Шаг 8: Анализ точности оценки")
        
        st.markdown("""
        **Факторы, влияющие на точность оценки:**
        - Объем выборки (чем больше, тем лучше)
        - Правильность спецификации распределения
        - Число оцениваемых параметров (чем больше параметров, тем больше моментов нужно использовать)
        """)
        
        st.markdown("**Преимущества метода моментов:**")
        st.markdown("- Простота вычислений\n- Не требует сложных оптимизационных процедур")
        
        st.markdown("**Недостатки метода моментов:**")
        st.markdown("- Может давать менее точные оценки по сравнению с ММП\n- Оценки могут быть за пределами допустимой области")

elif question == "11. Метод наибольшего правдоподобия":
    st.header("11. Метод наибольшего правдоподобия")
    
    st.subheader("Суть метода")
    st.markdown("Находим параметры, максимизирующие функцию правдоподобия:")
    st.latex(r"L(\theta) = \prod_{i=1}^n f(x_i|\theta)")
    
    st.subheader("Алгоритм")
    st.markdown("1. Составляем функцию правдоподобия")
    st.markdown("2. Берем логарифм для получения логарифмической функции правдоподобия")
    st.markdown("3. Находим максимум (обычно через производные)")
    
    if st.checkbox("Показать практический пример", key="practical_11"):
        st.subheader("Практический пример оценки параметров методом МНП")
        
        # Генерация данных из экспоненциального распределения
        np.random.seed(42)
        true_lambda = 0.5
        data = np.random.exponential(1/true_lambda, 100)
        
        st.markdown("**Исходные данные:** Экспоненциальное распределение с λ=0.5")
        st.markdown(f"**Объем выборки:** n = {len(data)}")
        
        st.markdown("**1. Функция правдоподобия для экспоненциального распределения:**")
        st.latex(r"L(\lambda) = \prod_{i=1}^n \lambda e^{-\lambda x_i} = \lambda^n e^{-\lambda \sum x_i}")
        
        st.markdown("**2. Логарифмическая функция правдоподобия:**")
        st.latex(r"\ln L(\lambda) = n \ln \lambda - \lambda \sum x_i")
        
        st.markdown("**3. Берем производную и приравниваем к нулю:**")
        st.latex(r"\frac{d}{d\lambda} \ln L(\lambda) = \frac{n}{\lambda} - \sum x_i = 0")
        
        st.markdown("**4. Решаем уравнение:**")
        lambda_hat = len(data) / sum(data)
        st.latex(fr"\hat{{\lambda}} = \frac{{n}}{{\sum x_i}} = \frac{{{len(data)}}}{{{sum(data):.4f}}} = {lambda_hat:.4f}")
        
        st.markdown("**Сравнение с истинным значением:**")
        comp_df = pd.DataFrame({
            'Параметр': ['λ'],
            'Истинное значение': [true_lambda],
            'Оценка МНП': [lambda_hat]
        })
        st.write(comp_df)
        
        # Визуализация
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(data, bins=20, density=True, alpha=0.6, label='Выборка')
        x = np.linspace(0, 10, 100)
        ax.plot(x, stats.expon.pdf(x, scale=1/true_lambda), 
                'r-', lw=2, label='Истинное распределение')
        ax.plot(x, stats.expon.pdf(x, scale=1/lambda_hat), 
                'g--', lw=2, label='Оценка МНП')
        ax.legend()
        ax.set_title('Сравнение истинного распределения и оценки МНП')
        st.pyplot(fig)

elif question == "12. Закон распределения вероятностей системы двух дискретных случайных величин. Построение законов распределения составляющих. Условные законы распределения составляющих системы дискретных случайных величин. Функциональная, статистическая и корреляционная зависимости. Корреляционная таблица. Условные средние значения.":
    st.header("12. Закон распределения вероятностей системы двух дискретных случайных величин. Построение законов распределения составляющих. Условные законы распределения составляющих системы дискретных случайных величин. Функциональная, статистическая и корреляционная зависимости. Корреляционная таблица. Условные средние значения.")
    
    st.subheader("📚 Основные определения")
    st.markdown("""
    **Совместное распределение** - закон, описывающий вероятность одновременного наступления событий X=xᵢ и Y=yⱼ  
    **Корреляционная таблица** - матрица, отображающая совместные вероятности всех пар значений (xᵢ, yⱼ)  
    **Маргинальные распределения** - распределения отдельных величин, полученные суммированием по строкам/столбцам  
    **Условное распределение** - распределение одной величины при фиксированном значении другой  
    **Условное математическое ожидание** - среднее значение одной величины при заданном значении другой
    """)

    st.subheader("Совместное распределение")
    st.markdown("Для системы двух дискретных случайных величин (X,Y) закон распределения задаётся таблицей вероятностей:")
    st.latex(r"P(x_i, y_j) = P(X=x_i, Y=y_j)")
    
    st.subheader("Корреляционная таблица")
    st.markdown(r"""
    | Y\X | $x_1$ | ... | $x_{n_x}$ | $n_{y_j}$ |
    |-----|-------|-----|----------|----------|
    | $y_1$ | $P(x_1,y_1)$ | ... | $P(x_{n_x},y_1)$ | $P(y_1)$ |
    | ... | ... | ... | ... | ... |
    | $y_{n_y}$ | $P(x_1,y_{n_y})$ | ... | $P(x_{n_x},y_{n_y})$ | $P(y_{n_y})$ |
    | $n_{x_i}$ | $P(x_1)$ | ... | $P(x_{n_x})$ | 1 |
    """)
    
    st.subheader("Маргинальные распределения")
    st.latex(r"P(x_i) = \sum_{j=1}^{n_y} P(x_i, y_j), \quad P(y_j) = \sum_{i=1}^{n_x} P(x_i, y_j)")
    
    st.subheader("Условные распределения")
    st.latex(r"P(x_i|y_j) = \frac{P(x_i, y_j)}{P(y_j)}, \quad P(y_j|x_i) = \frac{P(x_i, y_j)}{P(x_i)}")
    
    st.subheader("Условные средние")
    st.latex(r"M(X|Y=y_j) = \sum_{i=1}^{n_x} x_i P(x_i|y_j)")
    st.latex(r"M(Y|X=x_i) = \sum_{j=1}^{n_y} y_j P(y_j|x_i)")
    
    st.subheader("Типы зависимостей")
    st.markdown("""
    - **Функциональная**: Y = f(X) (однозначная связь)
    - **Статистическая**: P(Y|X) зависит от X
    - **Корреляционная**: M(Y|X) зависит от X
    """)

    if st.checkbox("Показать практический пример", key="practical_12"):
        st.subheader("Практический пример: совместное распределение")
        
        # Создаем случайную таблицу сопряженности
        np.random.seed(42)
        n_x = 3  # количество значений X
        n_y = 4  # количество значений Y
        
        # Генерируем случайные вероятности
        joint_probs = np.random.dirichlet(np.ones(n_x*n_y), size=1).reshape(n_x, n_y)
        joint_probs = np.round(joint_probs, 3)
        
        # Создаем DataFrame для отображения
        df_joint = pd.DataFrame(
            joint_probs,
            index=[f'x{i+1}' for i in range(n_x)],
            columns=[f'y{j+1}' for j in range(n_y)]
        )
        
        # Добавляем маргинальные суммы
        df_joint['P(x_i)'] = df_joint.sum(axis=1)
        df_joint.loc['P(y_j)'] = df_joint.sum(axis=0)
        df_joint.loc['P(y_j)', 'P(x_i)'] = 1.0
        
        st.markdown("**Совместное распределение X и Y:**")
        st.dataframe(df_joint.style.format("{:.3f}"))
        
        st.markdown("**1. Расчет маргинальных вероятностей:**")
        st.latex(r"P(x_1) = \sum_{j=1}^4 P(x_1, y_j) = " + " + ".join([f"{joint_probs[0,j]:.3f}" for j in range(n_y)]) + 
                f" = {df_joint.iloc[0, -1]:.3f}")
        
        st.markdown("**2. Расчет условных вероятностей:**")
        st.latex(r"P(y_1|x_1) = \frac{P(x_1, y_1)}{P(x_1)} = \frac{" + 
            f"{joint_probs[0,0]:.3f}" + r"}{" + f"{df_joint.iloc[0, -1]:.3f}" + 
            r"} = " + f"{joint_probs[0,0]/df_joint.iloc[0, -1]:.3f}")
        st.markdown("**3. Расчет условного математического ожидания:**")
        y_values = [10, 20, 30, 40]  # Примерные значения Y
        cond_expectation = sum(y_values[j] * joint_probs[0,j]/df_joint.iloc[0, -1] for j in range(n_y))
        st.latex(r"M(Y|x_1) = \sum_{j=1}^4 y_j P(y_j|x_1) = " + 
                " + ".join([f"{y_values[j]} × {joint_probs[0,j]/df_joint.iloc[0, -1]:.3f}" for j in range(n_y)]) + 
                f" = {cond_expectation:.3f}")

elif question ==  "13. Теоретическое уравнение регрессии.":
    st.header( "13. Теоретическое уравнение регрессии.")

    st.subheader("📚 Основные определения")
    st.markdown("""
    **Теоретическая регрессия** - условное мат. ожидание Y при заданном X: M(Y|X=x)  
    **Коэффициенты регрессии** - параметры a (наклон) и b (смещение) линейной модели  
    **Остаточная дисперсия** - мера разброса точек вокруг линии регрессии  
    **Критерий МНК** - минимизация математического ожидания квадрата отклонений
    """)

    st.subheader("Линейная регрессия")
    st.latex(r"y(x) = M(Y|X=x) \approx ax + b")

    st.subheader("Коэффициенты (метод наименьших квадратов)")
    st.latex(r"a = \frac{\mu_{xy}}{\sigma_X^2} = r_{xy}\frac{\sigma_Y}{\sigma_X}")
    st.latex(r"b = M(Y) - a M(X)")

    st.subheader("Остаточная дисперсия")
    st.latex(r"\min M[(Y-y(X))^2] = \sigma_Y^2 (1 - r_{xy}^2)")

    if st.checkbox("Показать практический пример", key="practical_13"):
        st.subheader("Практический пример: теоретическая регрессия")
        
        # Параметры двумерного нормального распределения
        col1, col2 = st.columns(2)
        with col1:
            true_mx = st.slider("M(X)", -5.0, 5.0, 0.0)
            true_my = st.slider("M(Y)", -5.0, 5.0, 0.0)
        with col2:
            true_sx = st.slider("σ_X", 0.1, 5.0, 1.0)
            true_sy = st.slider("σ_Y", 0.1, 5.0, 1.0)
        rho = st.slider("ρ (корреляция)", -0.99, 0.99, 0.7)
        
        # Расчет теоретических коэффициентов регрессии
        a_theory = rho * true_sy / true_sx
        b_theory = true_my - a_theory * true_mx
        residual_var = true_sy**2 * (1 - rho**2)
        
        st.markdown("**Теоретические параметры регрессии:**")
        st.latex(fr"a = \rho \frac{{\sigma_Y}}{{\sigma_X}} = {rho:.2f} \times \frac{{{true_sy:.2f}}}{{{true_sx:.2f}}} = {a_theory:.4f}")
        st.latex(fr"b = M(Y) - a M(X) = {true_my:.2f} - {a_theory:.4f} \times {true_mx:.2f} = {b_theory:.4f}")
        st.latex(fr"\text{{Остаточная дисперсия}} = \sigma_Y^2(1-\rho^2) = {true_sy**2:.2f}(1-{rho**2:.4f}) = {residual_var:.4f}")
        
        # Визуализация
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Генерация данных
        cov = [[true_sx**2, rho*true_sx*true_sy], [rho*true_sx*true_sy, true_sy**2]]
        data = np.random.multivariate_normal([true_mx, true_my], cov, 1000)
        
        # Линия регрессии
        x_vals = np.linspace(true_mx-3*true_sx, true_mx+3*true_sx, 100)
        y_vals = a_theory * x_vals + b_theory
        
        ax.scatter(data[:,0], data[:,1], alpha=0.5, label='Случайные точки')
        ax.plot(x_vals, y_vals, 'r-', lw=2, label='Теоретическая регрессия')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Теоретическая линейная регрессия Y на X')
        ax.legend()
        st.pyplot(fig)

elif question == "14. Выборочное уравнение регрессии. Метод наименьших квадратов (МНК)":
    st.header("14. Выборочное уравнение регрессии. Метод наименьших квадратов (МНК)")

    st.subheader("📚 Основные определения")
    st.markdown("""
    **Эмпирическая регрессия** - аппроксимация теоретической регрессии по выборке  
    **МНК-оценки** - коэффициенты, минимизирующие сумму квадратов отклонений  
    **Выборочные моменты** - оценки ковариации и дисперсии на основе данных  
    **Линия регрессии** - график уравнения ŷ = ax + b, наилучшим образом описывающий данные
    """)

    st.subheader("Линейная модель")
    st.latex(r"y_x \approx ax + b")

    st.subheader("Коэффициенты МНК")
    st.latex(r"a = \frac{n\sum x_i y_i - (\sum x_i)(\sum y_i)}{n\sum x_i^2 - (\sum x_i)^2}")
    st.latex(r"b = \bar{y} - a \bar{x}")

    st.subheader("Связь с выборочными характеристиками")
    st.latex(r"a = \frac{\mu_{xy}^*}{\sigma_B(X)^2} = r_{xy}^* \frac{\sigma_B(Y)}{\sigma_B(X)}")

    if st.checkbox("Показать практический пример", key="practical_14"):
        st.subheader("Практический пример: МНК регрессия")
        
        # Генерация данных
        np.random.seed(42)
        n_points = st.slider("Количество точек", 10, 100, 30)
        true_a = st.slider("Истинный наклон (a)", -2.0, 2.0, 1.0, 0.1)
        true_b = st.slider("Истинное смещение (b)", -5.0, 5.0, 2.0, 0.1)
        noise = st.slider("Уровень шума", 0.1, 5.0, 1.0, 0.1)
        
        x = np.random.uniform(-5, 5, n_points)
        y = true_a * x + true_b + np.random.normal(0, noise, n_points)
        
        # Расчет коэффициентов МНК
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x * y)
        sum_x2 = sum(x**2)
        
        a_mnk = (n_points * sum_xy - sum_x * sum_y) / (n_points * sum_x2 - sum_x**2)
        b_mnk = (sum_y - a_mnk * sum_x) / n_points
        
        st.markdown("**Исходные данные:**")
        df = pd.DataFrame({'x': x, 'y': y})
        st.write(df.head())
        
        st.markdown("**Расчет коэффициентов:**")
        st.latex(fr"a = \frac{{{n_points} \times {sum_xy:.2f} - {sum_x:.2f} \times {sum_y:.2f}}}{{{n_points} \times {sum_x2:.2f} - {sum_x:.2f}^2}} = {a_mnk:.4f}")
        st.latex(fr"b = \frac{{{sum_y:.2f} - {a_mnk:.4f} \times {sum_x:.2f}}}{{{n_points}}} = {b_mnk:.4f}")
        
        st.markdown("**Сравнение с истинными значениями:**")
        comp_df = pd.DataFrame({
            'Параметр': ['a (наклон)', 'b (смещение)'],
            'Истинное значение': [true_a, true_b],
            'Оценка МНК': [a_mnk, b_mnk],
            'Ошибка': [abs(a_mnk - true_a), abs(b_mnk - true_b)]
        })
        
        # Исправленное форматирование таблицы
        st.table(comp_df.style.format({
            'Истинное значение': '{:.4f}',
            'Оценка МНК': '{:.4f}',
            'Ошибка': '{:.4f}'
        }))
        
        # Визуализация
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x, y, label='Данные')
        
        # Истинная линия
        x_vals = np.linspace(min(x), max(x), 100)
        ax.plot(x_vals, true_a * x_vals + true_b, 'r-', label='Истинная зависимость')
        
        # Линия МНК
        ax.plot(x_vals, a_mnk * x_vals + b_mnk, 'g--', label='МНК регрессия')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Метод наименьших квадратов')
        ax.legend()
        st.pyplot(fig)

elif question == "15. Выборочный корреляционный момент. Его смысл, расчетные формулы и свойства. Выборочный коэффициент корреляции. Его смысл, расчетные формулы и свойства.":
    st.header("15. Выборочный корреляционный момент и коэффициент корреляции")

    st.subheader("📚 Основные определения")
    st.markdown("""
    **Ковариация (корреляционный момент)** - мера совместной изменчивости величин  
    **Коэффициент корреляции Пирсона** - нормированная мера линейной связи (-1 ≤ r ≤ 1)  
    **Эллипс рассеяния** - визуализация совместного распределения двух величин  
    **Интерпретация r**:  
    ∙ 0.9-1.0 - очень сильная связь  
    ∙ 0.7-0.9 - сильная  
    ∙ 0.4-0.7 - умеренная  
    ∙ 0.2-0.4 - слабая  
    ∙ 0.0-0.2 - отсутствует
    """)

    st.subheader("Корреляционный момент (ковариация)")
    st.latex(r"\mu_{xy}^* = \overline{(x-\bar{x})(y-\bar{y})} = \frac{1}{n}\sum (x_i-\bar{x})(y_i-\bar{y})")

    st.subheader("Расчётная формула")
    st.latex(r"\mu_{xy}^* = \overline{xy} - \bar{x} \cdot \bar{y}")

    st.subheader("Коэффициент корреляции Пирсона")
    st.latex(r"r_{xy}^* = \frac{\mu_{xy}^*}{\sigma_B(X) \sigma_B(Y)}")

    st.subheader("Свойства")
    st.markdown("""
    - $-1 \leq r_{xy}^* \leq 1$
    - $r_{xy}^* = \pm 1$ для линейной функциональной зависимости
    - $r_{xy}^* = 0$ для независимых величин
    """)

    if st.checkbox("Показать практический пример", key="practical_15"):
        st.subheader("Практический пример: корреляционный анализ")
        
        # Генерация данных
        np.random.seed(42)
        n_points = st.slider("Количество точек", 10, 200, 50)
        rho = st.slider("Истинная корреляция ρ", -0.99, 0.99, 0.7)
        
        cov = [[1, rho], [rho, 1]]
        data = np.random.multivariate_normal([0, 0], cov, n_points)
        x, y = data[:,0], data[:,1]
        
        # Расчет характеристик
        mean_x, mean_y = np.mean(x), np.mean(y)
        cov_sample = np.cov(x, y, bias=True)[0,1]
        r_pearson = np.corrcoef(x, y)[0,1]
        sigma_x, sigma_y = np.std(x, ddof=0), np.std(y, ddof=0)
        
        st.markdown("**Исходные данные:**")
        df = pd.DataFrame({'x': x, 'y': y})
        st.write(df.head())
        
        st.markdown("**1. Расчет выборочной ковариации:**")
        st.latex(fr"\mu_{{xy}}^* = \frac{{1}}{{n}}\sum (x_i-\bar{{x}})(y_i-\bar{{y}}) = {cov_sample:.4f}")
        st.latex(fr"\text{{Альтернативно: }} \overline{{xy}} - \bar{{x}}\bar{{y}} = {np.mean(x*y):.4f} - {mean_x:.4f} \times {mean_y:.4f} = {cov_sample:.4f}")
        
        st.markdown("**2. Расчет коэффициента корреляции Пирсона:**")
        st.latex(fr"r_{{xy}}^* = \frac{{\mu_{{xy}}^*}}{{\sigma_B(X)\sigma_B(Y)}} = \frac{{{cov_sample:.4f}}}{{{sigma_x:.4f} \times {sigma_y:.4f}}} = {r_pearson:.4f}")
        
        st.markdown("**Сравнение с истинным значением:**")
        st.latex(fr"\text{{Истинное }} \rho = {rho:.2f}, \quad \text{{Выборочное }} r = {r_pearson:.4f}")
        
        # Визуализация
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x, y, alpha=0.7)
        
        # Линии средних
        ax.axvline(mean_x, color='r', linestyle='--', linewidth=1, label=f'Среднее X = {mean_x:.2f}')
        ax.axhline(mean_y, color='g', linestyle='--', linewidth=1, label=f'Среднее Y = {mean_y:.2f}')
        
        # Эллипс рассеяния
        from matplotlib.patches import Ellipse
        cov_matrix = np.cov(x, y)
        lambda_, v = np.linalg.eig(cov_matrix)
        angle = np.degrees(np.arctan2(v[1,0], v[0,0]))
        width, height = 2 * np.sqrt(lambda_)
        ell = Ellipse(xy=(mean_x, mean_y), width=width, height=height, angle=angle,
                    edgecolor='r', facecolor='none', linewidth=2)
        ax.add_patch(ell)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Диаграмма рассеяния (r = {r_pearson:.2f})')
        ax.legend()
        st.pyplot(fig)

elif question == "16. Выборочное корреляционное отношение":
    st.header("16. Выборочное корреляционное отношение")

    st.subheader("📚 Основные определения")
    st.markdown("""
    **Корреляционное отношение η** - мера любой (не только линейной) статистической связи  
    **Межгрупповая дисперсия** - вариация между групповыми средними  
    **Общая дисперсия** - полная вариация значений признака  
    **Свойства η**:  
    ∙ 0 ≤ η ≤ 1  
    ∙ η ≥ |r|  
    ∙ η = 1 для функциональной зависимости  
    ∙ η = |r| для чисто линейной связи  
    **Интерпретация η**:  
    ∙ >0.9 - очень тесная связь  
    ∙ 0.7-0.9 - сильная  
    ∙ 0.5-0.7 - заметная  
    ∙ <0.5 - слабая
    """)
    
    st.subheader("Определение")
    st.latex(r"\eta_{y|x} = \sqrt{\frac{D_{\text{межгр}}}{D_{\text{общ}}}} = \sqrt{\frac{D_{\bar{y}_x}}{D_y}}")

    st.subheader("Компоненты дисперсии")
    st.latex(r"D_{\bar{y}_x} = \frac{1}{n}\sum n_{x_i} (\bar{y}_{x_i} - \bar{y})^2 \quad \text{(межгрупповая)}")
    st.latex(r"D_y = \frac{1}{n}\sum n_{y_j} (y_j - \bar{y})^2 \quad \text{(общая)}")

    st.subheader("Свойства")
    st.markdown("""
    - $0 \leq \eta_{y|x} \leq 1$
    - $\eta_{y|x} \geq |r_{xy}^*|$
    - $\eta_{y|x} = 1$ для функциональной зависимости
    - $\eta_{y|x} = |r_{xy}^*|$ для линейной зависимости
    """)

    if st.checkbox("Показать практический пример", key="practical_16"):
        st.subheader("Практический пример: корреляционное отношение")
        
        # Создаем искусственные данные с нелинейной зависимостью
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2 * np.sin(x) + np.random.normal(0, 0.5, 100)
        
        # Разбиваем на группы по X
        bins = pd.cut(x, bins=5)
        groups = pd.DataFrame({'x': x, 'y': y, 'group': bins})
        grouped = groups.groupby('group')
        
        # Расчет компонент дисперсии
        overall_mean = y.mean()
        n_total = len(y)
        
        # Межгрупповая дисперсия
        group_means = grouped['y'].mean()
        group_counts = grouped['y'].count()
        between_var = sum(group_counts * (group_means - overall_mean)**2) / n_total
        
        # Общая дисперсия
        total_var = sum((y - overall_mean)**2) / n_total
        
        # Корреляционное отношение
        eta = np.sqrt(between_var / total_var)
        
        # Коэффициент корреляции Пирсона
        r = np.corrcoef(x, y)[0,1]
        
        st.markdown("**Статистики по группам:**")
        group_stats = grouped['y'].agg(['count', 'mean', 'var'])
        st.write(group_stats)
        
        st.markdown("**1. Расчет межгрупповой дисперсии:**")
        st.latex(r"D_{\bar{y}_x} = \frac{1}{n}\sum n_{x_i} (\bar{y}_{x_i} - \bar{y})^2")
        st.latex(f"= \\frac{{1}}{{{n_total}}} \\left[" + 
                " + ".join([f"{cnt} × ({mean:.2f} - {overall_mean:.2f})²" 
                            for cnt, mean in zip(group_counts, group_means)]) + 
                f"\\right] = {between_var:.4f}")
        
        st.markdown("**2. Расчет общей дисперсии:**")
        st.latex(fr"D_y = \frac{{1}}{{n}}\sum (y_i - \bar{{y}})^2 = {total_var:.4f}")
        
        st.markdown("**3. Корреляционное отношение:**")
        st.latex(fr"\eta_{{y|x}} = \sqrt{{\frac{{D_{{\bar{{y}}_x}}}}{{D_y}}}} = \sqrt{{\frac{{{between_var:.4f}}}{{{total_var:.4f}}}}} = {eta:.4f}")
        
        st.markdown("**4. Коэффициент корреляции Пирсона:**")
        st.latex(fr"r = {abs(r):.4f} \quad (\eta_{{y|x}} \geq |r|)")
        
        # Визуализация
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Диаграмма рассеяния
        ax1.scatter(x, y, alpha=0.6)
        for group, data in grouped:
            ax1.axhline(y=data['y'].mean(), color='r', linestyle='--', alpha=0.5)
        ax1.axhline(y=overall_mean, color='k', linestyle='-', linewidth=2)
        ax1.set_title(f'Диаграмма рассеяния (η = {eta:.2f}, r = {r:.2f})')
        
        # Boxplot по группам
        groups.boxplot(column='y', by='group', ax=ax2)
        ax2.axhline(y=overall_mean, color='k', linestyle='-', linewidth=2)
        ax2.set_title('Распределение по группам')
        
        st.pyplot(fig)
        
elif question == "17. Ранговая корреляция. Коэффициенты ранговой корреляции Спирмена и Кендалла. Предельные значения для коэффициентов ранговой корреляции.":
    st.header("17. Ранговая корреляция. Коэффициенты ранговой корреляции Спирмена и Кендалла. Предельные значения для коэффициентов ранговой корреляции.")
    
    st.subheader("📚 Основные определения")
    st.markdown("""
    **Ранговая корреляция** - мера зависимости между рангами объектов по двум признакам.  
    **Ранг** - порядковый номер объекта в упорядоченном ряду.  
    **Связанные ранги** - одинаковые ранги для объектов с одинаковыми значениями признака.  
    **Преимущества**:  
    ∙ Непараметрический метод (не требует нормальности)  
    ∙ Устойчив к выбросам  
    ∙ Работает с порядковыми данными  
    """)
    
    st.subheader("Коэффициент Спирмена")
    st.markdown("**Формула без связанных рангов:**")
    st.latex(r"\tau_S = 1 - \frac{6 \sum_{i=1}^n d_i^2}{n^3 - n}")
    st.markdown("**Формула со связанными рангами:**")
    st.latex(r"\tau_S = 1 - \frac{6 \sum_{i=1}^n d_i^2}{(n^3 - n) - (T_x + T_y)}")
    st.latex(r"T_x = \frac{1}{2} \sum (t_l^3 - t_l), \quad T_y = \frac{1}{2} \sum (t_l^3 - t_l)")
    
    st.subheader("Коэффициент Кендалла")
    st.markdown("**Формула без связанных рангов:**")
    st.latex(r"\tau_K = \frac{2(P - Q)}{n(n-1)}")
    st.markdown("**Формула со связанными рангами:**")
    st.latex(r"\tau_K = \frac{2(P - Q)}{\sqrt{(n(n-1) - T_x)(n(n-1) - T_y)}}")
    
    st.subheader("Предельные значения")
    st.markdown("""
    - Оба коэффициента ∈ [-1, 1]  
    - 1: полная прямая зависимость рангов  
    - -1: полная обратная зависимость рангов  
    - 0: отсутствие монотонной зависимости  
    """)
    
    if st.checkbox("Показать практический пример", key="practical_17"):
        st.subheader("Практический пример: расчет ранговой корреляции")
        
        # Пример данных
        st.markdown("**Пример 1: Нет связанных рангов**")
        df1 = pd.DataFrame({
            'Объект': [1, 2, 3, 4, 5],
            'X': [3, 7, 10, 6, 1],
            'Y': [1, 4, 7, 5, 3],
            'Ранг X': [2, 4, 5, 3, 1],
            'Ранг Y': [1, 3, 5, 4, 2]
        })
        
        st.markdown("**Исходные данные:**")
        st.write(df1)
        
        # Расчет Спирмена
        n = len(df1)
        d = df1['Ранг X'] - df1['Ранг Y']
        sum_d2 = sum(d**2)
        spearman = 1 - 6 * sum_d2 / (n**3 - n)
        
        st.markdown("**1. Коэффициент Спирмена:**")
        st.latex(fr"\sum d_i^2 = {sum_d2}")
        st.latex(fr"\tau_S = 1 - \frac{{6 \times {sum_d2}}}{{{n}^3 - {n}}} = {spearman:.3f}")
        
        # Расчет Кендалла
        from itertools import combinations
        P = 0
        Q = 0
        for (i, j) in combinations(range(n), 2):
            if (df1.loc[i, 'Ранг X'] < df1.loc[j, 'Ранг X']) == (df1.loc[i, 'Ранг Y'] < df1.loc[j, 'Ранг Y']):
                P += 1
            else:
                Q += 1
        kendall = 2 * (P - Q) / (n * (n - 1))
        
        st.markdown("**2. Коэффициент Кендалла:**")
        st.latex(fr"P = {P}, \quad Q = {Q}")
        st.latex(fr"\tau_K = \frac{{2 \times ({P} - {Q})}}{{{n} \times {n-1}}} = {kendall:.3f}")
        
        # Пример со связанными рангами
        st.markdown("---")
        st.markdown("**Пример 2: Есть связанные ранги**")
        df2 = pd.DataFrame({
            'Объект': [1, 2, 3, 4, 5],
            'X': [2, 7, 2, 6, 1],
            'Y': [1, 3, 3, 5, 4],
            'Ранг X': [2.5, 5, 2.5, 4, 1],
            'Ранг Y': [1, 2.5, 2.5, 5, 4]
        })
        
        st.markdown("**Исходные данные:**")
        st.write(df2)
        
        # Расчет Спирмена со связанными рангами
        d = df2['Ранг X'] - df2['Ранг Y']
        sum_d2 = sum(d**2)
        
        # Группы связанных рангов для X
        tx = (2**3 - 2) / 2  # одна группа из 2 элементов
        
        # Группы связанных рангов для Y
        ty = (2**3 - 2) / 2  # одна группа из 2 элементов
        
        spearman = 1 - 6 * sum_d2 / ((n**3 - n) - (tx + ty))
        
        st.markdown("**1. Коэффициент Спирмена (со связанными рангами):**")
        st.latex(fr"T_x = \frac{{1}}{{2}}(2^3 - 2) = {tx}")
        st.latex(fr"T_y = \frac{{1}}{{2}}(2^3 - 2) = {ty}")
        st.latex(fr"\tau_S = 1 - \frac{{6 \times {sum_d2}}}{{({n}^3 - {n}) - ({tx} + {ty})}} = {spearman:.3f}")
        
        # Визуализация
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.scatter(df1['Ранг X'], df1['Ранг Y'])
        ax1.set_title(f'Пример 1: τ_S = {spearman:.2f}, τ_K = {kendall:.2f}')
        ax1.set_xlabel('Ранг X')
        ax1.set_ylabel('Ранг Y')
        
        ax2.scatter(df2['Ранг X'], df2['Ранг Y'])
        ax2.set_title(f'Пример 2 (со связанными рангами): τ_S = {spearman:.2f}')
        ax2.set_xlabel('Ранг X')
        ax2.set_ylabel('Ранг Y')
        
        st.pyplot(fig)

elif question == "18. Коэффициент конкордации. Смысл. Диапазон изменения. Методика расчета.":
    st.header("18. Коэффициент конкордации. Смысл. Диапазон изменения. Методика расчета.")
    
    st.subheader("📚 Основные определения")
    st.markdown("""
    **Коэффициент конкордации (W)** - мера согласованности мнений экспертов:  
    ∙ 0 ≤ W ≤ 1  
    ∙ W = 1: полное согласие экспертов  
    ∙ W = 0: отсутствие согласованности  
    **Применение**:  
    ∙ Оценка согласованности экспертных мнений  
    ∙ Анализ результатов ранжирования  
    ∙ Проверка надежности экспертов  
    """)
    
    st.subheader("Формула расчета")
    st.latex(r"W = \frac{S}{S_{\text{макс}}}")
    st.latex(r"S = \sum_{i=1}^n \left( \sum_{j=1}^m r_{ij} - \frac{m(n+1)}{2} \right)^2")
    st.markdown("**Для связанных рангов:**")
    st.latex(r"S_{\text{макс}} = \frac{m^2(n^3 - n)}{12} - \frac{m}{12} \sum_{j=1}^m \sum (t_l^3 - t_l)")
    
    st.subheader("Интерпретация")
    st.markdown("""
    - 0.9-1.0: отличная согласованность  
    - 0.7-0.9: хорошая  
    - 0.5-0.7: удовлетворительная  
    - <0.5: слабая согласованность  
    """)
    
    if st.checkbox("Показать практический пример", key="practical_18"):
        st.subheader("Практический пример: оценка согласованности экспертов")
        
        # Пример данных (3 эксперта, 5 объектов)
        data = {
            'Объект': ['A', 'B', 'C', 'D', 'E'],
            'Эксперт 1': [1, 2, 3, 4, 5],
            'Эксперт 2': [1, 3, 2, 5, 4],
            'Эксперт 3': [2, 1, 3, 4, 5]
        }
        df = pd.DataFrame(data)
        
        st.markdown("**Ранжирование объектов экспертами:**")
        st.write(df)
        
        # Расчет коэффициента конкордации
        m = 3  # количество экспертов
        n = 5  # количество объектов
        
        # Сумма рангов для каждого объекта
        df['Сумма рангов'] = df.iloc[:, 1:].sum(axis=1)
        
        # Средняя сумма рангов
        T = m * (n + 1) / 2
        
        # Отклонения сумм рангов от среднего
        S = sum((df['Сумма рангов'] - T)**2)
        
        # Максимально возможное S (без связанных рангов)
        S_max = (m**2 * (n**3 - n)) / 12
        
        W = S / S_max
        
        st.markdown("**1. Расчет суммы рангов:**")
        st.write(df)
        
        st.markdown("**2. Расчет средней суммы рангов:**")
        st.latex(fr"T = \frac{{m(n+1)}}{{2}} = \frac{{{m} \times {n+1}}}{{2}} = {T}")
        
        st.markdown("**3. Расчет S:**")
        st.latex(fr"S = \sum (R_i - T)^2 = {S}")
        
        st.markdown("**4. Расчет W:**")
        st.latex(fr"W = \frac{{S}}{{S_{{\text{{макс}}}}}} = \frac{{{S}}}{{{S_max:.2f}}} = {W:.3f}")
        
        st.markdown("**Вывод:**")
        if W > 0.9:
            st.success("Отличная согласованность экспертов (W > 0.9)")
        elif W > 0.7:
            st.success("Хорошая согласованность экспертов (0.7 < W ≤ 0.9)")
        elif W > 0.5:
            st.warning("Удовлетворительная согласованность (0.5 < W ≤ 0.7)")
        else:
            st.error("Слабая согласованность экспертов (W ≤ 0.5)")
        
        # Пример со связанными рангами
        st.markdown("---")
        st.markdown("**Пример со связанными рангами:**")
        
        data_tied = {
            'Объект': ['A', 'B', 'C', 'D', 'E'],
            'Эксперт 1': [1, 2, 2, 4, 5],
            'Эксперт 2': [1, 3, 2, 5, 4],
            'Эксперт 3': [2, 1, 2, 4, 5]
        }
        df_tied = pd.DataFrame(data_tied)
        
        st.markdown("**Ранжирование со связанными рангами:**")
        st.write(df_tied)
        
        # Расчет S (аналогично)
        df_tied['Сумма рангов'] = df_tied.iloc[:, 1:].sum(axis=1)
        S_tied = sum((df_tied['Сумма рангов'] - T)**2)
        
        # Расчет поправок для связанных рангов
        # Эксперт 1: одна группа из 2 элементов (ранг 2)
        t1 = (2**3 - 2)
        
        # Эксперт 2: нет связанных рангов
        t2 = 0
        
        # Эксперт 3: одна группа из 2 элементов (ранг 2)
        t3 = (2**3 - 2)
        
        total_t = t1 + t2 + t3
        S_max_tied = (m**2 * (n**3 - n)) / 12 - m * total_t / 12
        W_tied = S_tied / S_max_tied
        
        st.markdown("**1. Поправки для связанных рангов:**")
        st.latex(fr"\sum (t_l^3 - t_l) = {t1} + {t2} + {t3} = {total_t}")
        
        st.markdown("**2. Расчет S_max со связанными рангами:**")
        st.latex(fr"S_{{\text{{макс}}}} = \frac{{{m}^2({n}^3 - {n})}}{{12}} - \frac{{{m} \times {total_t}}}{{12}} = {S_max_tied:.2f}")
        
        st.markdown("**3. Расчет W:**")
        st.latex(fr"W = \frac{{{S_tied}}}{{{S_max_tied:.2f}}} = {W_tied:.3f}")
        
        # Визуализация
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(['Без связанных рангов', 'Со связанными рангами'], [W, W_tied])
        ax.axhline(0.9, color='g', linestyle='--', label='Отличная согласованность')
        ax.axhline(0.7, color='b', linestyle='--', label='Хорошая согласованность')
        ax.axhline(0.5, color='r', linestyle='--', label='Удовлетворительная')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Коэффициент конкордации W')
        ax.set_title('Сравнение коэффициентов конкордации')
        ax.legend()
        st.pyplot(fig)

elif question == "19. Случайные числа. Разыгрывание дискретной случайной величины.":
    st.header("19. Случайные числа. Разыгрывание дискретной случайной величины.")
    
    st.subheader("📚 Основные определения")
    st.markdown("""
    **Случайные числа** - возможные значения непрерывной равномерно распределенной случайной величины R ∈ [0, 1].  
    **Разыгрывание СВ** - генерация последовательности ее возможных значений.  
    **Квазиравномерная СВ** R* - псевдослучайное число с конечным числом знаков после запятой.  
    """)
    
    st.subheader("Алгоритм разыгрывания дискретной СВ")
    st.markdown("""
    1. Разбить интервал [0, 1] на подынтервалы Δᵢ длиной pᵢ
    2. Сгенерировать случайное число r ∈ [0, 1]
    3. Определить, в какой Δᵢ попало r
    4. Выдать соответствующее xᵢ
    """)
    
    st.latex(r"""
    \begin{array}{cccccc}
    0 & \Delta_1 & \Delta_2 & \Delta_3 & \cdots & \Delta_n \\
    | & p_1 & p_2 & p_3 & \cdots & p_n \\
    \end{array}
    """)
    
    if st.checkbox("Показать практический пример", key="practical_19"):
        st.subheader("Практический пример")
        
        # Задаем распределение
        dist = pd.DataFrame({
            'xᵢ': ['-1', '2', '3', '6'],
            'pᵢ': [0.25, 0.50, 0.15, 0.10]
        })
        
        st.markdown("**Закон распределения:**")
        st.write(dist)
        
        # Визуализация интервалов
        intervals = {
            'Δᵢ': ['(0.00, 0.25]', '(0.25, 0.75]', '(0.75, 0.90]', '(0.90, 1.00]'],
            'xᵢ': ['-1', '2', '3', '6']
        }
        st.markdown("**Соответствующие интервалы:**")
        st.write(pd.DataFrame(intervals))
        
        # Генерация случайного числа
        r = np.random.uniform(0, 1)
        st.markdown(f"**Сгенерированное число r = {r:.4f}**")
        
        # Определение значения
        cum_prob = 0
        result = None
        for i in range(len(dist)):
            cum_prob += dist['pᵢ'][i]
            if r <= cum_prob:
                result = dist['xᵢ'][i]
                break
                
        st.markdown(f"**Результат разыгрывания:** x = {result} (r ∈ {intervals['Δᵢ'][i]})")

elif question == "20. Разыгрывание непрерывной случайной величины. Метод суперпозиции.":
    st.header("20. Разыгрывание непрерывной случайной величины. Метод суперпозиции.")
    
    st.subheader("📚 Основные методы")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Через функцию распределения F(x):**")
        st.latex(r"F(x) = r \Rightarrow x = F^{-1}(r)")
    
    with col2:
        st.markdown("**Через плотность f(x):**")
        st.latex(r"\int_{-\infty}^x f(z)dz = r")
    
    st.subheader("Метод суперпозиции")
    st.markdown("Если F(x) можно представить как:")
    st.latex(r"F(x) = \sum_{j=1}^m C_j F_j(x), \quad \sum C_j = 1")
    st.markdown("Тогда:")
    st.markdown("1. Разыгрываем дискретную СВ Z для выбора компоненты")
    st.markdown("2. Разыгрываем X через выбранную Fⱼ(x) = r")
    
    if st.checkbox("Показать практический пример", key="practical_20"):
        st.subheader("Пример: смесь экспоненциальных распределений")
        
        # Задаем распределение
        st.latex(r"F(x) = 0.6(1-e^{-x}) + 0.4(1-e^{-5x})")
        
        # Генерируем r1 для выбора компоненты
        r1 = np.random.uniform(0, 1)
        component = 1 if r1 <= 0.6 else 2
        st.markdown(f"**Шаг 1:** r₁ = {r1:.4f} → Выбрана компонента {component}")
        
        # Генерируем x через выбранную компоненту
        r2 = np.random.uniform(0, 1)
        if component == 1:
            x = -np.log(1 - r2)
        else:
            x = -np.log(1 - r2)/5
        st.markdown(f"**Шаг 2:** r₂ = {r2:.4f} → x = {x:.4f}")

elif question == "21. Приближённое разыгрывание нормальной случайной величины.":
    st.header("21. Приближённое разыгрывание нормальной случайной величины.")
    
    st.subheader("📚 Центральная предельная теорема")
    st.markdown("Сумма независимых равномерных СВ стремится к нормальному распределению:")
    st.latex(r"Z = \frac{\sum_{j=1}^n R_j - n/2}{\sqrt{n/12}} \sim N(0,1)")
    
    st.subheader("Практическая формула (n=12)")
    st.latex(r"X \approx \sigma \cdot \left( \sum_{j=1}^{12} R_j - 6 \right) + a")
    
    if st.checkbox("Показать практический пример", key="practical_21"):
        st.subheader("Пример генерации N(5, 4)")
        
        # Генерируем 12 случайных чисел
        np.random.seed(42)
        r = np.random.uniform(0, 1, 12)
        st.markdown("**Сгенерированные числа:**")
        st.write(r)
        
        # Вычисляем x
        sum_r = sum(r)
        x = 2 * (sum_r - 6) + 5
        st.markdown(f"**Вычисление:** (2 × ({sum_r:.4f} - 6) + 5 = {x:.4f}")
        
        # Визуализация
        # Gенерация нормального распределения
        fig, ax = plt.subplots()
        ax.hist([2*(np.random.uniform(0,1,12).sum()-6)+5 for _ in range(1000)], bins=20)
        ax.set_title("Гистограмма 1000 разыгранных значений N(5,4)")
        st.pyplot(fig)

elif question == "22. Сущность метода Монте-Карло. Оценка его погрешности.":
    st.header("22. Сущность метода Монте-Карло. Оценка его погрешности.")
    
    st.subheader("📚 Основные принципы метода")
    st.markdown("""
    **Метод Монте-Карло** - численный метод, основанный на:
    - Моделировании случайных величин
    - Статистической оценке искомых величин
    - Законе больших чисел
    
    **Области применения**:
    - Вычисление интегралов
    - Оптимизация
    - Финансовое моделирование
    - Физика частиц
    """)
    
    st.subheader("Алгоритм метода")
    st.markdown("""
    1. **Формулировка задачи**: Определить искомую величину `a` как математическое ожидание случайной величины `X`
    2. **Генерация выборки**: Получить `n` независимых реализаций `x₁, x₂, ..., xₙ`
    3. **Статистическая оценка**: Вычислить выборочное среднее `a* = (x₁ + ... + xₙ)/n`
    4. **Оценка точности**: Рассчитать доверительный интервал
    """)
    
    st.latex(r"a^* = \bar{x}_B = \frac{1}{n}\sum_{i=1}^n x_i")
    
    st.subheader("Оценка погрешности")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**При известной дисперсии σ²:**")
        st.latex(r"\bar{x}_B \pm t_{\gamma}\frac{\sigma}{\sqrt{n}}")
        st.markdown("где `tᵧ` - квантиль нормального распределения")
    
    with col2:
        st.markdown("**При неизвестной дисперсии:**")
        st.latex(r"\bar{x}_B \pm t_{\gamma}\frac{S}{\sqrt{n}}")
        st.markdown("где `S` - выборочное стандартное отклонение")
    
    st.markdown("**Для не нормальных распределений** применяется центральная предельная теорема при n ≥ 30")
    
    if st.checkbox("Показать практический пример", key="practical_22"):
        st.subheader("Пример: оценка числа π")
        
        # Параметры
        n = st.slider("Количество точек", 100, 10000, 1000)
        
        # Генерация точек
        x = np.random.uniform(-1, 1, n)
        y = np.random.uniform(-1, 1, n)
        inside = (x**2 + y**2) <= 1
        pi_estimate = 4 * np.mean(inside)
        
        # Оценка погрешности
        sigma = np.sqrt(pi_estimate * (4 - pi_estimate) / n)
        conf_int = (pi_estimate - 1.96*sigma, pi_estimate + 1.96*sigma)
        
        st.markdown(f"**Оценка π:** {pi_estimate:.4f}")
        st.markdown(f"**95% доверительный интервал:** ({conf_int[0]:.4f}, {conf_int[1]:.4f})")
        
        # Визуализация
        fig, ax = plt.subplots()
        ax.scatter(x[inside], y[inside], color='blue', alpha=0.3, label='Внутри круга')
        ax.scatter(x[~inside], y[~inside], color='red', alpha=0.3, label='Снаружи')
        ax.set_aspect('equal')
        ax.set_title(f"Метод Монте-Карло для π (n={n})")
        ax.legend()
        st.pyplot(fig)

elif question == "23. Вычисление определённых интегралов с помощью метода Монте-Карло.":
    st.header("23. Вычисление определённых интегралов с помощью метода Монте-Карло.")
    
    st.subheader("📚 Основные подходы")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Способ 1 (основной):**")
        st.latex(r"I = \int_a^b \varphi(x)dx \approx \frac{b-a}{n}\sum_{i=1}^n \varphi(x_i)")
        st.markdown("где `xᵢ` - равномерно распределенные точки")
    
    with col2:
        st.markdown("**Способ 2 (геометрический):**")
        st.latex(r"I \approx \frac{m}{n} \cdot S")
        st.markdown("где `m` - число точек под кривой, `S` - площадь области")
    
    st.subheader("Алгоритм основного способа")
    st.markdown("""
    1. Генерируем `n` точек `xᵢ` ∈ [a,b] равномерно
    2. Вычисляем `φ(xᵢ)` для каждой точки
    3. Усредняем значения и умножаем на длину интервала
    """)
    
    st.subheader("Алгоритм геометрического способа")
    st.markdown("""
    1. Генерируем точки (xᵢ,yᵢ) в прямоугольнике [a,b]×[0,d]
    2. Считаем точки под кривой yᵢ ≤ φ(xᵢ)
    3. Вычисляем долю таких точек и умножаем на площадь
    """)
    
    if st.checkbox("Показать практический пример", key="practical_23"):
        st.subheader("Пример вычисления интеграла")
        
        # Функция и параметры
        func = st.selectbox("Функция", ["x² - 7", "sin(x)", "e^(-x²)"])
        a = st.number_input("Нижний предел a", value=1.0)
        b = st.number_input("Верхний предел b", value=3.0)
        n = st.slider("Количество точек", 100, 10000, 1000)
        
        # Аналитическое решение
        if func == "x² - 7":
            true_val = (b**3/3 - 7*b) - (a**3/3 - 7*a)
            def f(x): return x**2 - 7
        elif func == "sin(x)":
            true_val = -np.cos(b) + np.cos(a)
            def f(x): return np.sin(x)
        else:
            true_val = None  # Нет аналитического решения
            def f(x): return np.exp(-x**2)
        
        # Метод 1
        x = np.random.uniform(a, b, n)
        integral_1 = (b-a) * np.mean(f(x))
        
        # Метод 2 (если функция неотрицательна)
        if f(x).min() >= 0:
            d = f(x).max() * 1.1
            y = np.random.uniform(0, d, n)
            m = np.sum(y <= f(x))
            integral_2 = m/n * (b-a)*d
        else:
            integral_2 = None
        
        st.markdown("**Результаты:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Способ 1:**")
            st.latex(fr"I \approx \frac{{{b}-{a}}}{{{n}}} \sum \varphi(x_i) = {integral_1:.4f}")
            if true_val is not None:
                st.markdown(f"**Точное значение:** {true_val:.4f}")
                st.markdown(f"**Погрешность:** {abs(integral_1 - true_val):.4f}")
        
        with col2:
            st.markdown("**Способ 2:**")
            if integral_2 is not None:
                st.latex(fr"I \approx \frac{{{m}}}{{{n}}} \times ({b}-{a}) \times {d:.2f} = {integral_2:.4f}")
                if true_val is not None:
                    st.markdown(f"**Погрешность:** {abs(integral_2 - true_val):.4f}")
            else:
                st.markdown("Неприменим (функция принимает отрицательные значения)")
        
        # Визуализация
        fig, ax = plt.subplots()
        x_plot = np.linspace(a, b, 100)
        ax.plot(x_plot, f(x_plot), 'r-', lw=2, label=f'$\varphi(x)={func}$')
        
        if integral_2 is not None:
            ax.scatter(x, y, color='blue', alpha=0.3)
            ax.axhline(0, color='black')
            ax.set_ylim(0, d)
        else:
            ax.axhline(0, color='black')
            ax.fill_between(x_plot, f(x_plot), where=(f(x_plot)>=0), color='blue', alpha=0.3)
            ax.fill_between(x_plot, f(x_plot), where=(f(x_plot)<0), color='red', alpha=0.3)
        
        ax.set_title("График функции и точек Монте-Карло")
        ax.legend()
        st.pyplot(fig)
