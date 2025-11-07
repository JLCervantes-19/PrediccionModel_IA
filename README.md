# Modelo Predictivo de Victorias en UEFA Champions League

Este proyecto tiene como objetivo desarrollar un modelo predictivo de Machine Learning para estimar la probabilidad de victoria de los equipos en la UEFA Champions League, utilizando estadísticas agregadas de sus partidos. El modelo se entrenó utilizando datos de las temporadas 2024/25 y 2025/26.

## Enlace a la Presentación

Puedes acceder a la presentación completa del proyecto [aquí](https://gamma.app/docs/MODELO-PREDICTIVO-DE-VICTORIAS-EN-UEFA-CHAMPIONS-LEAGUE-eyyv2o82nbfc5s6).

## Resumen del Proyecto

El modelo desarrollado se enfoca en predecir el *win_rate* (porcentaje de victorias) de los equipos de fútbol basándose en estadísticas de rendimiento ofensivo. A continuación, se describen las fases principales del proyecto:

## Desarrollo del Proyecto

### 1. Adquisición y Exploración de Datos

El dataset utilizado contiene estadísticas de 36 equipos de la UEFA Champions League, abarcando las temporadas 2024/25 y 2025/26. Incluye información sobre partidos disputados, victorias, empates, derrotas, goles, disparos y ataques realizados por los equipos. La fase de exploración inicial reveló una alta variabilidad en los rendimientos de los equipos.

### 2. Preprocesamiento y Limpieza de Datos

Se realizó la normalización de las columnas, se verificaron los valores nulos y duplicados, y se garantizó la coherencia de los datos. La conversión de tipos de datos fue crucial para asegurar que las métricas numéricas fueran correctamente procesadas y que las sumas de victorias, empates y derrotas coincidieran con los partidos disputados.

### 3. Ingeniería de Características

Se derivaron 12 nuevas variables, entre ellas el *win_rate* (porcentaje de victorias), la eficiencia de disparo, el índice de rendimiento (*performance_index*), y métricas de ataque, como goles por partido y disparos por partido. Estas variables fueron claves para alimentar el modelo de predicción.

### 4. Análisis Exploratorio y Modelado

Se realizaron visualizaciones y análisis estadísticos para entender las relaciones entre las variables. Los modelos de regresión lineal y *Gradient Boosting* fueron evaluados, y se eligió *Gradient Boosting* por su mejor desempeño predictivo.

### 5. Evaluación y Predicción

El modelo seleccionado alcanzó un R² de 0.80, lo que indica que el 80% de la variabilidad en las victorias fue explicada por el modelo. Se generaron predicciones para los equipos, que fueron clasificadas en niveles de confianza, como "Muy Alta", "Alta", "Media" y "Baja".

## Resultados del Modelo

El mejor modelo obtenido fue el de *Gradient Boosting*, con las siguientes métricas:

- **R² (Test)**: 0.80
- **RMSE (Error cuadrático medio)**: 8-10 puntos
- **MAE (Error absoluto medio)**: 7 puntos

El modelo logró predecir con alta precisión las probabilidades de victoria para los equipos, destacando a equipos como Bayern München, Inter y Arsenal con una probabilidad de victoria del 100%.

### Equipos con Mayor Probabilidad de Victoria

| Equipo        | Win Rate Real (%) | Probabilidad de Victoria (%) | Nivel de Confianza |
|---------------|-------------------|------------------------------|--------------------|
| Bayern München | 100               | 100                          | Muy Alta           |
| Inter         | 100               | 100                          | Muy Alta           |
| Arsenal       | 100               | 100                          | Muy Alta           |
| Paris         | 75                | 75                           | Muy Alta           |
| Liverpool     | 75                | 75                           | Muy Alta           |

### Equipos con Menor Probabilidad de Victoria

| Equipo        | Win Rate Real (%) | Probabilidad de Victoria (%) | Nivel de Confianza |
|---------------|-------------------|------------------------------|--------------------|
| Benfica       | 0                 | 0                            | Baja               |
| Slavia Praha  | 0                 | 0                            | Baja               |
| Olympiacos    | 0                 | 0                            | Baja               |
| Ajax          | 0                 | 0                            | Baja               |

## Conclusiones

El modelo de *Gradient Boosting* ha demostrado ser eficaz en predecir las probabilidades de victoria en la UEFA Champions League utilizando estadísticas de rendimiento ofensivo. Aunque el modelo presenta una precisión razonable, las limitaciones incluyen la falta de datos defensivos y la no consideración de factores contextuales como lesiones o tácticas específicas.

**Recomendaciones futuras:**
- Incorporar estadísticas defensivas, como goles recibidos y tiros bloqueados.
- Considerar el análisis temporal de rendimiento (últimos 3-5 partidos).
- Incluir datos contextuales como local/visitante y enfrentamientos directos.

## Archivos Generados

El proyecto generó los siguientes archivos para su análisis y visualización:

- `ucl_predictions_final.csv`: Predicciones detalladas.
- `ucl_model_summary.csv`: Resumen del modelo.
- `ucl_feature_importance.csv`: Importancia de las características.
- `ucl_eda_analysis.png`: Análisis exploratorio de datos.
- `ucl_model_evaluation.png`: Evaluación del modelo.
- `ucl_predictions_visualization.png`: Visualización de las predicciones.

## Requisitos Técnicos

El modelo fue desarrollado utilizando las siguientes herramientas y librerías:

- **Lenguaje de programación**: Python 3.x
- **Librerías**: pandas, numpy, scikit-learn, matplotlib, seaborn

## Recomendaciones para Mejoras

Se sugiere que en futuras versiones del modelo se añadan las siguientes mejoras:
1. Incorporar estadísticas defensivas.
2. Añadir análisis de la forma reciente de los equipos.
3. Incluir un análisis de factores psicológicos y tácticos.

---

Este resumen proporciona una visión clara del proyecto, incluyendo las fases técnicas y los resultados obtenidos. Además, incluye recomendaciones y los archivos generados para su posterior análisis.
