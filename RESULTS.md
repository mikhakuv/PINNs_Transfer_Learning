(это пока черновик, который нужно будет дописать, проставить цитирования, отредактировать и перестроить некоторые графики)
# Введение
## Постановка задачи
В данной работе изучается эффективность методов трансферного обучения в решении дифференциальных уравнений с помощью технологии PINNs.
В качестве дифференциального уравнения рассматривается нелинейное уравнение Шрёдингера:  
$$i q_{t} + q_{xx} +q\cdot |q|^2(1-\alpha|q|^2+\beta |q|^4)=0$$
с начальным условием в виде солитона:  
$$q(x,0) = \sqrt{\frac{\mu e^{\left(x - x_{0}\right) \sqrt{\mu}}}{\left(\frac{1}{2} e^{\left(x - x_{0}\right) \sqrt{\mu}} + 1\right)^{2} - \frac{\alpha_0 \mu}{3} e^{2 \left(x - x_{0}\right) \sqrt{\mu}}}} e^{i \left(k x + \theta_{0}\right)},$$
$$где\ \mu = 4(k^{2} - w)$$
И граничным условием:  
$$q(x_0,t)=q(x_1,t)=0$$
Аналитическое решение такой задачи при $\alpha=\alpha_0$ известно:  
$$q(x,t) = \sqrt{\frac{\mu e^{\left(x - 2 k t - x_{0}\right) \sqrt{\mu}}}{\left(\frac{1}{2} e^{\left(x - 2 k t - x_{0}\right) \sqrt{\mu}} + 1\right)^{2} - \frac{\alpha_0 \mu}{3} e^{2 \left(x - 2 k t - x_{0}\right) \sqrt{\mu}}}} e^{i \left(k x - \omega t + \theta_{0}\right)},$$
$$где\ \mu = 4(k^{2} - w)$$
Значения коэффициентов: $k=1,\ w=0.88,\ x_0=-30,\ \theta_0=0,\ \beta=0$. При этом значения $\alpha$ и $\alpha_0$ совпадают во всех опытах(если не оговорено обратное) и принимают значения из множества $\lbrace 0.2; 0.3; 0.4 \rbrace$.  
Также в данной работе рассматривается пример практического применения методов трансферного обучения для улучшения решений, полученных ранее с помощью технологии FBPINNs.  
## Обзор литературы
Технология PINNs была предложена в 2018 году[1] и по сути предлагала свести задачу решения дифференциального уравнения к задаче оптимизации, которую можно решить с помощью инструментов машинного обучения. С тех пор эта технология была использована для большого числа прикладных задач[2-4] и ещё большее число раз модифицирована [5-7]. И поскольку решение уравнения по сути сводилось к обучению нейросети, было лишь вопросом времени применение идей из классического машинного обучения к этой технологии. В частности идея трансферного обучения была применена к PINNs уже через год её возникновения и есть приличное количество работ на эту тему[7-9]. Уникальность текущей работы заключается именно в рассматриваемом уравнении(до сих пор рассматривались более простые случаи), а также в попутно решаемой задаче об улучшении уже полученного ранее решения.
## План исследования
Основная часть данной работы разделена на 3 части: перенос обучаемых параметров, перенос значений, улучшение ранее полученных решений. В первых двух частях рассматриваются два разных метода трансферного обучения и обсуждается их эффективность для рассматриваемого случая. В последней же части предложен метод улучшения уже готовых решений с помощью трансферного обучения.
## Используемые метрики
В качестве основной метрики используется: $$Rel_h = \frac{\sqrt{ \sum\limits_{i=1}^N (|q_{i}^{truth}|-|q_{i}^{pred}|)^2 }}{\sqrt{\sum\limits_{i=1}^N |q_{i}^{truth}|^2}}$$
# Основная часть
## Перенос обучаемых праметров
Основной идеей метода PINNs является нахождение решения в виде нейросети. Именно поэтому финальным результатом работы этого метода являются параметры (веса и смещения) нейросети, делающие из неё функцию, удовлетворяющую уравнению насколько это возможно. Но процесс поиска таких параметров - обучение нейросети является самой ресурсозатратной частью метода: занимает от нескольких минут до часов. Логично предположить, что это время можно сократить, если начинать обучение PINN не на случайных весах, а на тех, что были получены при решении похожих задач. Для проверки данной гипотезы в случае нелинейного уравнения Шрёдингера было проведено несколько опытов (см. таблицы 1 и 2). Для каждого из значений $\alpha \in [0.2, 0.3, 0.4]$ была обучена PINN с первоначально случайными параметрами(basic learning), а также другая PINN с параметрами, которые были взяты от предыдущего опыта для $\alpha=0.2$(weights transfer). Каждый опыт был повторён 10 раз, а результаты $Rel_h$ усреднены.  

**таблица 1, basic learning**  
|             | alpha=0.2 |          | alpha=0.3 |          | alpha=0.4 |          |
|-------------|-----------|----------|-----------|----------|-----------|----------|
| test_number | Time      | Rel_h    | Time      | Rel_h    | Time      | Rel_h    |
| 1           | 2h 2min   | 4,41E-02 | 2h 2min   | 1,80E-02 | 2h 2min   | 3,93E-02 |
| 2           | 2h 2min   | 4,24E-02 | 2h 2min   | 2,24E-02 | 2h 2min   | 2,31E-02 |
| 3           | 2h 2min   | 3,66E-02 | 2h 2min   | 7,73E-03 | 2h 2min   | 2,19E-02 |
| 4           | 2h 2min   | 1,52E-02 | 2h 2min   | 1,98E-02 | 2h 2min   | 3,97E-02 |
| 5           | 2h 2min   | 7,06E-03 | 2h 2min   | 3,60E-02 | 2h 2min   | 2,98E-02 |
| 6           | 2h 2min   | 1,90E-02 | 2h 2min   | 2,12E-02 | 2h 2min   | 5,85E-03 |
| 7           | 2h 3min   | 3,95E-02 | 2h 3min   | 1,14E-02 | 2h 3min   | 2,45E-02 |
| 8           | 2h 3min   | 1,20E-02 | 2h 3min   | 2,65E-02 | 2h 3min   | 8,21E-03 |
| 9           | 2h 3min   | 2,19E-02 | 2h 3min   | 2,90E-02 | 2h 3min   | 3,51E-02 |
| 10          | 2h 2min   | 2,12E-02 | 2h 2min   | 1,01E-02 | 2h 2min   | 2,78E-02 |
| average     |           | 2,59E-02 |           | 2,02E-02 |           | 2,55E-02 |

<!-- код этой таблицы в латехе
\begin{table}[!ht]
    \centering
    \begin{tabular}{|l|l|l|l|l|l|l|}
    \hline
        ~ & alpha=0.2 & ~ & alpha=0.3 & ~ & alpha=0.4 & ~ \\ \hline
        test\_number & Time & Rel\_h & Time & Rel\_h & Time & Rel\_h \\ \hline
        1 & 2h 2min & 4,41E-02 & 2h 2min & 1,80E-02 & 2h 2min & 3,93E-02 \\ \hline
        2 & 2h 2min & 4,24E-02 & 2h 2min & 2,24E-02 & 2h 2min & 2,31E-02 \\ \hline
        3 & 2h 2min & 3,66E-02 & 2h 2min & 7,73E-03 & 2h 2min & 2,19E-02 \\ \hline
        4 & 2h 2min & 1,52E-02 & 2h 2min & 1,98E-02 & 2h 2min & 3,97E-02 \\ \hline
        5 & 2h 2min & 7,06E-03 & 2h 2min & 3,60E-02 & 2h 2min & 2,98E-02 \\ \hline
        6 & 2h 2min & 1,90E-02 & 2h 2min & 2,12E-02 & 2h 2min & 5,85E-03 \\ \hline
        7 & 2h 3min & 3,95E-02 & 2h 3min & 1,14E-02 & 2h 3min & 2,45E-02 \\ \hline
        8 & 2h 3min & 1,20E-02 & 2h 3min & 2,65E-02 & 2h 3min & 8,21E-03 \\ \hline
        9 & 2h 3min & 2,19E-02 & 2h 3min & 2,90E-02 & 2h 3min & 3,51E-02 \\ \hline
        10 & 2h 2min & 2,12E-02 & 2h 2min & 1,01E-02 & 2h 2min & 2,78E-02 \\ \hline
        average & ~ & 2,59E-02 & ~ & 2,02E-02 & ~ & 2,55E-02 \\ \hline
    \end{tabular}
\end{table}
-->  

**таблица 2, parameters transfer**  
| source alpha=0.2 | alpha=0.2 |          | alpha=0.3 |          | alpha=0.4 |          |
|------------------|-----------|----------|-----------|----------|-----------|----------|
| test_number      | Time      | Rel_h    | Time      | Rel_h    | Time      | Rel_h    |
| 1                | 2h 2min   | 1,22E-02 | 2h 2min   | 1,12E-02 | 2h 2min   | 1,00E-02 |
| 2                | 2h 2min   | 1,36E-02 | 2h 2min   | 1,28E-02 | 2h 2min   | 1,11E-02 |
| 3                | 2h 2min   | 1,47E-02 | 2h 2min   | 1,25E-02 | 2h 2min   | 1,13E-02 |
| 4                | 2h 2min   | 5,56E-03 | 2h 2min   | 5,13E-03 | 2h 2min   | 4,74E-03 |
| 5                | 2h 2min   | 2,61E-03 | 2h 2min   | 2,62E-03 | 2h 2min   | 2,41E-03 |
| 6                | 2h 2min   | 6,88E-03 | 2h 2min   | 6,53E-03 | 2h 2min   | 5,91E-03 |
| 7                | 2h 2min   | 1,53E-02 | 2h 2min   | 1,32E-02 | 2h 2min   | 1,22E-02 |
| 8                | 2h 2min   | 3,77E-03 | 2h 2min   | 3,70E-03 | 2h 2min   | 3,29E-03 |
| 9                | 2h 2min   | 8,86E-03 | 2h 2min   | 8,29E-03 | 2h 2min   | 7,31E-03 |
| 10               | 2h 2min   | 6,29E-03 | 2h 2min   | 5,98E-03 | 2h 2min   | 5,28E-03 |
| average          |           | 8,97E-03 |           | 8,20E-03 |           | 7,35E-03 |

<!--код в латехе
\begin{table}[!ht]
    \centering
    \begin{tabular}{|l|l|l|l|l|l|l|}
    \hline
        source alpha=0.2 & alpha=0.2 & ~ & alpha=0.3 & ~ & alpha=0.4 & ~ \\ \hline
        test\_number & Time & Rel\_h & Time & Rel\_h & Time & Rel\_h \\ \hline
        1 & 2h 2min & 1,22E-02 & 2h 2min & 1,12E-02 & 2h 2min & 1,00E-02 \\ \hline
        2 & 2h 2min & 1,36E-02 & 2h 2min & 1,28E-02 & 2h 2min & 1,11E-02 \\ \hline
        3 & 2h 2min & 1,47E-02 & 2h 2min & 1,25E-02 & 2h 2min & 1,13E-02 \\ \hline
        4 & 2h 2min & 5,56E-03 & 2h 2min & 5,13E-03 & 2h 2min & 4,74E-03 \\ \hline
        5 & 2h 2min & 2,61E-03 & 2h 2min & 2,62E-03 & 2h 2min & 2,41E-03 \\ \hline
        6 & 2h 2min & 6,88E-03 & 2h 2min & 6,53E-03 & 2h 2min & 5,91E-03 \\ \hline
        7 & 2h 2min & 1,53E-02 & 2h 2min & 1,32E-02 & 2h 2min & 1,22E-02 \\ \hline
        8 & 2h 2min & 3,77E-03 & 2h 2min & 3,70E-03 & 2h 2min & 3,29E-03 \\ \hline
        9 & 2h 2min & 8,86E-03 & 2h 2min & 8,29E-03 & 2h 2min & 7,31E-03 \\ \hline
        10 & 2h 2min & 6,29E-03 & 2h 2min & 5,98E-03 & 2h 2min & 5,28E-03 \\ \hline
        average & ~ & 8,97E-03 & ~ & 8,20E-03 & ~ & 7,35E-03 \\ \hline
    \end{tabular}
\end{table}
-->

Таким образом был определён усреднённый рост точности при применении подобного подхода и он составляет не менее **59.5%**. Причём видно, что пока разница в коэффициентах $\alpha$ сравнительно небольшая, увеличение точности от неё практически не зависит(см. график 1).  
<p align="center"><img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/chart_1.PNG"><br><caption>график 1</caption></p>  

Видно, точность при применении подобного подхода действительно возрастает и его результативность подвтерждается критерием Стьюдента(надо разобраться, подходит ли он с математической точки зрения и вычислить).  

## Перенос значений
Но в случае PINNs можно действовать и другим способом. Так как задача оптимизации осложнена не высокой размерностью данных или большим их количеством, а видом функции потерь (которая содержит частные производные), упрощение вида функции потерь приведёт к существенному 
упрощению задачи оптимизации. Поэтому вместо переноса параметров модели можно обучить новую модель на решении, полученном ранее, причём процесс переноса не будет требовать много времени(в результате получилось не больше 5 минут). Применение этой идеи позволило увеличить точность в среднем на **21-39%**(см график 2).  
<p align="center"><img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/chart_2.PNG"><br><caption>график 2</caption></p>  

## Улучшение ранее полученных решений
Рассмотрим задачу улучшения ранее полученных данных, когда полученный результат хорош, но есть потребность сделать его ещё лучше. Хотя видно, что перенос весов работает гораздо лучше переноса значений(см график 3), он не всегда может быть реализован. 
<p align="center"><img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/chart_3.PNG"><br><caption>график 3</caption></p>  

Например если исходные данные получены методом FBPINNs, то можно получить веса, но это будет много весов от большого количества маленьких нейросетей, которые нельзя перенести в одну большую нейросеть. Поэтому в данном случае переносу значений нет альтернатив: нейросеть будет обучаться на данных, которые остались от FBPINN. Дополнительно можно оптимизировать этот процесс, избавившись от областей с околонулевыми значениями и разбив большой массив данных на последовательные части, на каждой из которых обучается отдельная нейросеть. Для данной задачи это приемлемо, ведь на выходе ожидается не нейросеть, а массив улучшенных данных.  
Рассмотрим пример работы данного подхода:  
1. Исходный результат:  
<img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/raw_1_fig.png">  
<img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/raw_1_amplitude.png">  

Разбиение данных:  
<img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/exp_3_decomposition.png">  

Улучшенный результат:  
<img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/exp_3_fig.png">  
<img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/exp_3_amplitude.png">  

Rel_h уменьшилась на **38%**, ошибки на законах - более чем на **70%**  
Можно проделать те же самые действия и для полученных данных:  
<img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/exp_3+_fig.png">  
<img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/exp_3+_amplitude.png">  

Rel_h почему-то увеличилась, но ошибки на законах дополнительно уменьшились более чем на **49%**  

2. Исходный результат:  
<img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/raw_2_fig.png">  
<img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/raw_2_amplitude.png">  

Разбиение данных:  
<img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/exp_4_decomposition.png">  

Улучшенный результат:  
<img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/exp_4_fig.png">  
<img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/exp_4_amplitude.png">  

Rel_h уменьшилась на **55%**, ошибки на законах - более чем на **53%**  
Можно проделать те же самые действия и для полученных данных:  
<img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/exp_4+_fig.png">  
<img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/exp_4+_amplitude.png">  

Rel_h почему-то увеличилась, но ошибки на законах дополнительно уменьшились более чем на **75%**  

Опыты для $\alpha \neq \alpha_0$
---
3. Исходный результат:  
<img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/raw_3_fig.png">  
<img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/raw_3_amplitude.png">  

Разбиение данных:  
<img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/exp_5_decomposition.png">  

Улучшенный результат:  
<img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/exp_5_fig.png">  
<img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/exp_5_amplitude.png">  

Ошибки на законах уменьшились более чем на **36%**  
Можно проделать те же самые действия и для полученных данных:  
<img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/exp_5+_fig.png">  
<img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/exp_5+_amplitude.png">  

Ошибки на законах уменьшатся дополнительно более чем на **39%**

Более подробная статистика доступна в файлах: [performance_table.xlsx](https://github.com/mikhakuv/PINNs/blob/main/statistics/performance_table_transfer_learning.xlsx), [enhancement_stats.xlsx](https://github.com/mikhakuv/PINNs/blob/main/statistics/enhancement_stats.xlsx)

# Заключение
## Обсуждение результатов
## Ссылки на источники
[1] - короче Maziar Raissi, George Em Karniadakis Physical Informed Deep Learning  
[2] - [4] - примеры прикладного использования PINNs  
[5] - [7] - улучшения PINNs: FBPINNs, etc  
[7] - [9] - статьи на тему Transfer Learning in PINNs  
