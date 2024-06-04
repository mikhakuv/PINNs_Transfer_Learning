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
Основная часть данной работы разделена на 3 части: перенос весов, перенос значений, улучшение ранее полученных решений. В первых двух частях рассматриваются два разных метода трансферного обучения и обсуждается их эффективность для рассматриваемого случая. В последней же части предложен метод улучшения уже готовых решений с помощью трансферного обучения.
# Основная часть
## Перенос весов
Обучив нейросеть на уравнении, можно сохранить её обучаемые параметры и при решении уравнения с другими коэффициентами загрузить их. Тогда будет происходить не обучение с нуля, а переучивание, которое может быть более простой задачей.
Применение такой идеи позволило увеличить точность на **65-71%**, причём пока разница в коэффициентах сравнительно небольшая, увеличение точности от неё практически не зависит(см график 1).  
<p align="center"><img src="https://github.com/mikhakuv/PINNs/blob/main/pictures/transfer_learning/chart_1.PNG"><br><caption>график 1</caption></p>  

## Перенос значений
Но в случае PINNs можно действовать другим способом. Так как задача оптимизации осложнена не высокой размерностью данных или большим их количеством, а видом функции потерь (которая содержит частные производные), упрощение вида функции потерь приведёт к существенному 
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
