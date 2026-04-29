|     | Esquema |     | de Trabajo: |         | Localizaci¾n  |     | de  |     |
| --- | ------- | --- | ----------- | ------- | ------------- | --- | --- | --- |
|     |         |     | Fuentes     | de      | PM10          |     |     |     |
|     |         |     | Problema    | Inverso | de Dispersi¾n |     |     |     |
El problema
El proyecto aborda la problemßtica crÝtica de la dispersi¾n de material particulado
PM10 en el Valle de Aburrß, donde la geografÝa irregular y las condiciones meteoro-
l¾gicas variables dificultan la identificaci¾n precisa de las fuentes de emisi¾n. Para
resolver este problema inverso, se propone el uso de Adaptive Inverse PINNs (Re-
des Neuronales Informadas por la FÝsica), las cuales integran datos del ecosistema
SIATA con la ecuaci¾n matemßtica de advecci¾n-difusi¾n-reacci¾n para modelar el
transporte y la degradaci¾n de partÝculas. Esta metodologÝa utiliza un mecanismo
de pÚrdida adaptativa con pesos dinßmicos y una arquitectura agÚntica para supe-
rar las limitaciones de datos y la naturaleza mal planteada (ill-posed) del sistema
fÝsico, permitiendo localizar ôhotspotsö de contaminaci¾n y diferenciar entre fuentes
| industriales |         | y de trßfico | de manera  | precisa. |     |     |     |     |
| ------------ | ------- | ------------ | ---------- | -------- | --- | --- | --- | --- |
| Inverse      | PINN    |              | Adaptativa |          |     |     |     |     |
| PINN         | Inversa | Estßndar     |            |          |     |     |     |     |
Una Physics-Informed Neural Network (PINN) inversa tiene como objetivo estimar
parßmetros constitutivos ? (coeficientes fÝsicos, velocidades, etc.) que aparecen en
0
la EDP subyacente, a partir de datos observados. El problema de minimizaci¾n toma
la forma
|     |     | (cid:0) | ?å(cid:1)      | (cid:0) | (cid:1) |        |     |     |
| --- | --- | ------- | -------------- | ------- | ------- | ------ | --- | --- |
|     |     |         | ?å, = argm┤?nL | u (?);  | ? +???  | ????2, |     | (1) |
|     |     |         | 0              | NN      | 0       | 0 0    |     |     |
?,?0
donde L combina la pÚrdida de ajuste a datos y la pÚrdida residual de la EDP, ? es
un parßmetro de regularizaci¾n e ?? son parßmetros de referencia. En ausencia de
0
| regularizaci¾n |     | se toma | ? = 0. |     |     |     |     |     |
| -------------- | --- | ------- | ------ | --- | --- | --- | --- | --- |
El principal inconveniente de esta formulaci¾n es que los distintos tÚrminos de L
1

pueden entrar en conflicto durante la retropropagaci¾n, produciendo gradientes des-
equilibrados y dificultando la convergencia, problema que se agrava cuando los datos
| disponibles |         | son escasos |            | [?]. |     |     |     |     |     |     |     |     |
| ----------- | ------- | ----------- | ---------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
| PINN        | Inversa |             | Adaptativa |      |     |     |     |     |     |     |     |     |
Para superar la inestabilidad de la PINN inversa estßndar, Berardi et al., 2025,
| proponen |         | una funci¾n |         | de                 | pÚrdida          | ponderada |     | adaptativa: |                  |     |          |     |
| -------- | ------- | ----------- | ------- | ------------------ | ---------------- | --------- | --- | ----------- | ---------------- | --- | -------- | --- |
|          |         |             |         | (cid:88)(cid:16) M |                  |           |     |             |                  |     | (cid:17) |     |
|          | (cid:0) |             | (cid:1) |                    |                  |           |     |             |                  |     |          |     |
|          | L       | u (?);      | ?       | =                  | ?k?u(x?,t?)?u??2 |           |     | +           | ?k ?R(u(x?,t?);? |     | )?2 ,    | (2) |
|          |         | NN          | 0       |                    | i                | i         | i   | i           | R                | i i | 0        |     |
i=1
donde los pesos ?k y ?k se actualizan en cada Úpoca k de acuerdo con
|     |     |     | i   | R        |     |      |     |          |     |      |     |     |
| --- | --- | --- | --- | -------- | --- | ---- | --- | -------- | --- | ---- | --- | --- |
|     |     |     |     |          | ?êk |      |     |          | ?êk |      |     |     |
|     |     |     | ?k  |          | i   |      | ?k  |          | R   |      |     |     |
|     |     |     |     | =        |     |      | ,   | =        |     | ,    |     | (3) |
|     |     |     |     | i        | M   |      |     | R M      |     |      |     |     |
|     |     |     |     | (cid:88) |     |      |     | (cid:88) |     |      |     |     |
|     |     |     |     |          | ?êk | +?êk |     |          | ?êk | +?êk |     |     |
|     |     |     |     |          | j   | R    |     |          | j   | R    |     |     |
|     |     |     |     | j=1      |     |      |     | j=1      |     |      |     |     |
y los pesos crudos ?êk se asignan seg·n la naturaleza del punto de entrenamiento:
i
?
|     |     |     |     | ??   |     | si x? | ? ??, |     |     |     |     |     |
| --- | --- | --- | --- | ---- | --- | ----- | ----- | --- | --- | --- | --- | --- |
|     |     |     |     | ? BC |     | i     |       |     |     |     |     |     |
? ?
?
|     |     |     |     | ? ?? |     | si t? | = 0, |     |     |     |     |     |
| --- | --- | --- | --- | ---- | --- | ----- | ---- | --- | --- | --- | --- | --- |
IC
|     |     |     | ?êk = |     |     | i   |     |     |     |     |     | (4) |
| --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
i
(x?,t?)
|     |     |     |     | ??(k)? ? |     | si  | es  | punto | de colocaci¾n, |     |     |     |
| --- | --- | --- | --- | -------- | --- | --- | --- | ----- | -------------- | --- | --- | --- |
|     |     |     |     | ?        | u   | i   | i   |       |                |     |     |     |
?
? ?
|     |     |     |     | ?0  |     | en otro | caso. |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------- | ----- | --- | --- | --- | --- | --- |
La funci¾n ?(k) es creciente con la Úpoca k, con ?(0) = 0 y ?(k) ? 1 cuando k ? ?.
| En concreto |     | se adopta |     |          |          |     |          |     |     |     |     |     |
| ----------- | --- | --------- | --- | -------- | -------- | --- | -------- | --- | --- | --- | --- | --- |
|             |     |           |     | (cid:18) | k ?K/2?K |     | (cid:19) |     |     |     |     |     |
0
|     |     |     |     | tanh | 10  |     |     | +1  |     |     |     |     |
| --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
K
|     |     | ?(k) | =   |     |     |     |     | ,   | k   | = 1,...,K, |     | (5) |
| --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | --- |
2
donde K es el n·mero total de Úpocas y K es un umbral a partir del cual los pesos
0
| comienzan |     | a modificarse |     | de  | forma | significativa. |     |     |     |     |     |     |
| --------- | --- | ------------- | --- | --- | ----- | -------------- | --- | --- | --- | --- | --- | --- |
Diferencia radical con la PINN inversa estßndar. La novedad fundamental
| reside | en el | curriculum |     | de entrenamiento |     |     | que | introduce | ?(k): |     |     |     |
| ------ | ----- | ---------- | --- | ---------------- | --- | --- | --- | --------- | ----- | --- | --- | --- |
Fase inicial (k ? K/2): ?(k) ? 0, por lo que los puntos de colocaci¾n reciben
peso nulo. La red se entrena ·nicamente con el residuo de la EDP, aprendiendo
2

primero la dinßmica fÝsica sin interferencia de los datos ruidosos.
Fase tardÝa (k ? K/2): ?(k) ? 1, los datos observados se incorporan pro-
gresivamente a la pÚrdida y los parßmetros fÝsicos ? comienzan a actualizarse
0
|     | de  | forma | efectiva, |     | pues | sus gradientes |     | se  | escalan | por ??(k). |     |
| --- | --- | ----- | --------- | --- | ---- | -------------- | --- | --- | ------- | ---------- | --- |
Este mecanismo evita que los gradientes de los parßmetros fÝsicos corrompan la
soluci¾n antes de que la red haya aprendido la fÝsica del problema, garantizando la
convergencia incluso desde condiciones iniciales aleatorias. La tasa de aprendizaje
se actualiza ademßs con una estrategia de decaimiento exponencial por pasos:
|     |         |          |     |     | ? =    | ? ??k/100?, |              | 0,9 | < ? | < 0,99. | (6) |
| --- | ------- | -------- | --- | --- | ------ | ----------- | ------------ | --- | --- | ------- | --- |
|     |         |          |     |     | k      | 0           |              |     |     |         |     |
| El  | proceso | completo |     | se  | resume | en          | el Algoritmo |     | 1.  |         |     |
Algorithm 1 Entrenamiento con pesos adaptativos y actualizaci¾n de gradientes
| 1:  | epoch | ?   | 0   |     |     |     |     |     |     |     |     |
| --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
2: repeat
| 3:  |     | epoch                 | ? epoch+1 |     |     |     |       |     |        |     |     |
| --- | --- | --------------------- | --------- | --- | --- | --- | ----- | --- | ------ | --- | --- |
| 4:  |     | if do_parameter_train |           |     |     | and | epoch | >   | K then |     |     |
0
|     |     | Calcular |     | ?(epoch) |     | seg·n | Ec. | (5) |     |     |     |
| --- | --- | -------- | --- | -------- | --- | ----- | --- | --- | --- | --- | --- |
5:
| 6:  |     | end        | if         |       |     |          |       |         |     |     |     |
| --- | --- | ---------- | ---------- | ----- | --- | -------- | ----- | ------- | --- | --- | --- |
| 7:  |     | Actualizar |            | pesos | de  | datos    | seg·n | Ec. (4) |     |     |     |
| 8:  |     | Calcular   | gradientes |       | de  | L        |       |         |     |     |     |
| 9:  |     | Reescalar  |            | ? L   | por | ?(epoch) |       |         |     |     |     |
?0
|     |     | Aplicar | gradientes |     | a   | todos | los parßmetros |     | entrenables |     |     |
| --- | --- | ------- | ---------- | --- | --- | ----- | -------------- | --- | ----------- | --- | --- |
10:
| 11: | until      | convergencia |     |                | o epoch | >   | K   |     |     |     |     |
| --- | ---------- | ------------ | --- | -------------- | ------- | --- | --- | --- | --- | --- | --- |
| 1.  | Fundamento |              |     | Arquitect¾nico |         |     |     |     |     |     |     |
La estrategia se basa en disociar la orquestaci¾n semßntica (configuraci¾n, su-
pervisi¾n y validaci¾n) de la soluci¾n de la EDP (c¾mputo numÚrico). Los Large
Language Models (LLMs) se limitarßn al rol de Agentes, y no al cßlculo directo de
ecuaciones diferenciales. El objetivo es minimizar la carga computacional y maximi-
| zar  | la precisi¾n |     | en  | la identificaci¾n  |     |     | de fuentes |     | S(p?, | t). |     |
| ---- | ------------ | --- | --- | ------------------ | --- | --- | ---------- | --- | ----- | --- | --- |
| 1.1. | Adquisici¾n  |     |     | y Preprocesamiento |     |     |            | de  | Datos |     |     |
La alimentaci¾n de la funci¾n de pÚrdida empÝrica L exige la integraci¾n auto-
datos
| matizada |     | de  | datos | de alta | resoluci¾n |     | espacial. |     |     |     |     |
| -------- | --- | --- | ----- | ------- | ---------- | --- | --------- | --- | --- | --- | --- |
3

|        | Cuadro | 1: Fuentes | de datos, | herramientas |     |     | y preprocesamiento. |                  |     |     |
| ------ | ------ | ---------- | --------- | ------------ | --- | --- | ------------------- | ---------------- | --- | --- |
| Fuente | de     | Variables  | Clave     | Herramientas |     |     | /                   | Preprocesamiento |     |     |
| Datos  |        |            |           | Estrategia   |     |     |                     | Crucial          |     |     |
Red SIATA Concentraciones Utilizar Socrata API Adimensionalizaci¾n:
(Monitoreo) de PM10 y datos (sodapy) o peticiones Escalar el dominio
|     |     | meteorol¾gicos |       | al                 | portal         | CKAN        | del    | espacio-temporal |          |          |
| --- | --- | -------------- | ----- | ------------------ | -------------- | ----------- | ------ | ---------------- | -------- | -------- |
|     |     | (frecuencia    | de 10 | ┴rea               | Metropolitana, |             |        | (Valle           | de       | Aburrß   |
|     |     | minutos).      |       | dada               | la             | no          |        | y el             | tiempo)  | a        |
|     |     |                |       | disponibilidad     |                |             | de una | [?1,             | 1]3      | y [0, 1] |
|     |     |                |       | API                | REST           | p·blica     | de     | respectivamen-   |          |          |
|     |     |                |       | SIATA              | para           | descargas   |        | te,              | para     | mitigar  |
|     |     |                |       | masivas            |                | hist¾ricas. |        | el               | problema | de       |
|     |     |                |       | Indagar            |                | en la       | pßgina | gradientes       |          | pa-      |
|     |     |                |       | web                | de             | Calidad     | Aire:  | tol¾gicos        |          | en las   |
|     |     |                |       | https://siata.gov. |                |             |        | Physics-Informed |          |          |
|     |     |                |       | co/CalidadAire/    |                |             |        | Neural           |          | Networks |
(PINN).
| Datos         |     | Aerosol      | Optical | Google        |          | Earth        | Engine  | Collocation |              |          |
| ------------- | --- | ------------ | ------- | ------------- | -------- | ------------ | ------- | ----------- | ------------ | -------- |
| Satelitales   |     | Depth (AOD). |         | (GEE)         | Python   |              | API     | Points      |              | Inte-    |
| (AOD)         |     |              |         | para          | extraer  | series       |         | ligentes:   |              | Im-      |
| (Opcional)    |     |              |         | temporales    |          |              |         | plementar   |              | Latin    |
|               |     |              |         | reproyectadas |          |              | y       | Hypercube   |              | Sam-     |
|               |     |              |         | enmascaradas  |          |              |         | pling       |              | (LHS).   |
|               |     |              |         | (Sentinel-5P, |          |              | MODIS,  | Priorizar   |              | la den-  |
|               |     |              |         | VIIRS),       |          | evitando     | la      | sidad       | de           | puntos   |
|               |     |              |         | descarga      |          | de terabytes |         | cerca       | de           | estacio- |
|               |     |              |         | de            | imßgenes | crudas.      |         | nes         | de monitoreo |          |
|               |     |              |         |               |          |              |         | y zonas     |              | de topo- |
|               |     |              |         |               |          |              |         | grafÝa      | compleja     | en       |
|               |     |              |         |               |          |              |         | lugar       | de           | utilizar |
|               |     |              |         |               |          |              |         | una         | malla        | regular. |
| 2. Ecosistema |     | de Modelado  |         | (Solver       |          | de           | la EDP) |             |              |          |
El motor computacional debe ser eficiente para resolver la ecuaci¾n de Advecci¾n-
| Difusi¾n-Reacci¾n |     | (ADR) y | la optimizaci¾n |     | simultßnea: |     |     |     |     |     |
| ----------------- | --- | ------- | --------------- | --- | ----------- | --- | --- | --- | --- | --- |
|                   |     |         | argm┤?nL(u      |     | (?);        | ? ) |     |     |     |     |
|                   |     |         |                 |     | NN          | 0   |     |     |     |     |
?,?0
4

|            | Cuadro |             | 2: Componentes |       | del   | ecosistema |     | de modelado. |            |               |     |
| ---------- | ------ | ----------- | -------------- | ----- | ----- | ---------- | --- | ------------ | ---------- | ------------- | --- |
| Componente |        | Descripci¾n |                |       |       | Estrategia |     | /            | Frameworks |               |     |
| Motor de   |        | Soluci¾n    |                | de la | EDP y | Opci¾n     |     | 1 (Alto      |            | Rendimiento): |     |
C¾mputo cßlculo de la funci¾n Ecosistema Julia (NeuralPDE.jl,
|                 |     | de             | pÚrdida. |     |     | Lux.jl)      |         | para        | compilar |           | la EDP a  |
| --------------- | --- | -------------- | -------- | --- | --- | ------------ | ------- | ----------- | -------- | --------- | --------- |
|                 |     |                |          |     |     | c¾digo       | mßquina |             | y        | reducir   | el costo. |
|                 |     |                |          |     |     | Opci¾n       |         | 2 (Python): |          | NVIDIA    | Mo-       |
|                 |     |                |          |     |     | dulus        | o       | DeepXDE     |          | (maduros  | para      |
|                 |     |                |          |     |     | problemas    |         | inversos    |          | y soporte | de re-    |
|                 |     |                |          |     |     | ponderaci¾n  |         | adaptativa  |          | de        | pÚrdida). |
| Parametrizaci¾n |     | Representaci¾n |          |     |     | Parametrizar |         |             | como     | una       | suma de   |
de Fuentes eficiente de la fuente distribuciones Gaussianas bi-
|     |     | S(p?, | t). |     |     | dimensionales. |             |                 | La        | PINN         | inversa     |
| --- | --- | ----- | --- | --- | --- | -------------- | ----------- | --------------- | --------- | ------------ | ----------- |
|     |     |       |     |     |     | aprende        |             | las coordenadas |           | del          | centroi-    |
|     |     |       |     |     |     | de             | (Á ,Á       | ) y la          | amplitud, |              | lo que sua- |
|     |     |       |     |     |     |                | x           | y               |           |              |             |
|     |     |       |     |     |     | viza           | el panorama |                 | de        | optimizaci¾n | es-         |
pacial.
| 3. Arquitectura |     | AgÚntica |     | (Flujo |     | de  | Trabajo |     | MLOps) |     |     |
| --------------- | --- | -------- | --- | ------ | --- | --- | ------- | --- | ------ | --- | --- |
Se propone la implementaci¾n de cinco agentes utilizando frameworks open-source
| como CrewAI  | o Microsoft |              | AutoGen. |     |            |     |             |     |     |           |     |
| ------------ | ----------- | ------------ | -------- | --- | ---------- | --- | ----------- | --- | --- | --------- | --- |
| 4. Propuesta |             | de Ejecuci¾n |          |     | Escalonada |     | (Curriculum |     |     | Learning) |     |
Para garantizar la estabilidad y la convergencia del entrenamiento se proponen tres
fases progresivas:
1. Fase Interpolativa. Entrenar una red forward estßndar solo con datos de SIA-
TA, asumiendo fuentes S = 0, para aprender el campo de vientos y concentra-
ciones.
2. Fase Inversa Estßtica. Congelar los pesos de la red anterior. Iniciar la Inverse
PINN, a±adir el tÚrmino S(p?, t) y entrenar exclusivamente para descubrir las
| coordenadas | de  | las | fuentes | y el parßmetro |     | de  | difusi¾n | D.  |     |     |     |
| ----------- | --- | --- | ------- | -------------- | --- | --- | -------- | --- | --- | --- | --- |
3. Ajuste Fino Adaptativo. Aplicar una metodologÝa (e.g., Berardi et al.) para
un afinamiento conjunto de todos los parßmetros, permitiendo que la fÝsica guÝe
el aprendizaje.
5

|        | Cuadro | 3: Agentes    | del sistema   | y sus funcionalidades. |       |     |     |     |
| ------ | ------ | ------------- | ------------- | ---------------------- | ----- | --- | --- | --- |
| Agente |        | Rol Principal | Funcionalidad |                        | Clave |     |     |     |
Agente 1: Physics El Configurador DefineloslÝmitesdeldominioyestablece
| Architect |     |     | las | condiciones | de frontera |     | (Dirichlet | /   |
| --------- | --- | --- | --- | ----------- | ----------- | --- | ---------- | --- |
Neumann)basßndoseendatosdeSIATA
yGEE.Generaelscriptdeconfiguraci¾n
|           |     |             | inicial | de la      | PINN. |           |          |     |
| --------- | --- | ----------- | ------- | ---------- | ----- | --------- | -------- | --- |
| Agente 2: |     | El Ejecutor | No      | es un LLM. | Es    | un script | empaque- |     |
PINN-ADR Solver (Tool) tado (Python / Julia) que el framework
|     |     |     | agÚntico   | invoca. | Entrena  |     | la red, | recibe   |
| --- | --- | --- | ---------- | ------- | -------- | --- | ------- | -------- |
|     |     |     | parßmetros | y       | devuelve | el  | loss y  | los gra- |
dientes.
Agente 3: Reaction El Validador Accede a literatura (vÝa RAG) para veri-
Validator CientÝfico ficar que las tasas de decaimiento k des-
|     |     |     | cubiertas | por | la Inverse | PINN | son | con- |
| --- | --- | --- | --------- | --- | ---------- | ---- | --- | ---- |
sistentesconlatermodinßmicaycinÚtica
del PM10.
Agente 4: Source El Atribuidor Recibe las coordenadas optimizadas p?
Identification Forense y realiza un cruce espacial (GeoPandas
|     |     |     | / OpenStreetMap |     | API) | para | atribuir | la  |
| --- | --- | --- | --------------- | --- | ---- | ---- | -------- | --- |
fuenteazonasindustriales,corredoresde
|     |     |     | trßfico | o factores | topogrßficos. |     |     |     |
| --- | --- | --- | ------- | ---------- | ------------- | --- | --- | --- |
6

Justificaci¾ndelaFaseInterpolativa:Pre-acondicionamientoparalaInverse-
PINN
Atacar el problema de la identificaci¾n de parßmetros fÝsicos directamente con una
Inverse-PINN (que aprende simultßneamente el campo de concentraci¾n y los parß-
metros desconocidos) es una estrategia te¾ricamente vßlida, pero altamente inesta-
ble y computacionalmente ineficiente si se comienza con una inicializaci¾n aleatoria
de pesos. El principal desafÝo radica en la naturaleza de ôproblema mal-puestoö (ill-
posed) debido a la escasez de datos, lo que provoca un conflicto en las contribuciones
| de las funciones | de  | pÚrdida. |     |     |     |     |     |     |     |     |     |
| ---------------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
La fase interpolativa se introduce como un mecanismo de pre-acondicionamiento
| esencial | del espacio | de pesos | para | mitigar | estos | riesgos: |     |     |     |     |     |
| -------- | ----------- | -------- | ---- | ------- | ----- | -------- | --- | --- | --- | --- | --- |
Cuadro 4: Problemas de la Inverse-PINN directa y sus soluciones mediante pre-
acondicionamiento.
| Problema     | con la  |     | Soluci¾n |     | a travÚs | del | Pre-acondicionamiento |     |     |     |     |
| ------------ | ------- | --- | -------- | --- | -------- | --- | --------------------- | --- | --- | --- | --- |
| Inverse-PINN | Inicial |     |          |     |          |     |                       |     |     |     |     |
Optimizaci¾n No La red debe simultßneamente identificar la topologÝa
Convexa Extrema del campo escalar de concentraci¾n C(?) y los parß-
|     |     |     | metros | fÝsicos | ?   | (difusi¾n, | coordenadas |     |     | de fuente), | re- |
| --- | --- | --- | ------ | ------- | --- | ---------- | ----------- | --- | --- | ----------- | --- |
0
|     |     |     | sultando |     | en un    | panorama | de  | optimizaci¾n |     | plagado | de  |
| --- | --- | --- | -------- | --- | -------- | -------- | --- | ------------ | --- | ------- | --- |
|     |     |     | mÝnimos  |     | locales. |          |     |              |     |         |     |
Gradientes Al inicio, el residuo fÝsico (tÚrmino L ) es despro-
EDP
| Dominantes | y   |     | porcionadamentegrande.LosgradientesdeL |     |     |     |     |     |     |     | anu- |
| ---------- | --- | --- | -------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ---- |
EDP
Patol¾gicos lan a L , forzando a la red a soluciones triviales que
datos
|     |     |     | satisfacen |        | la matemßtica |         | de  | la EDP | (e.g., | concentra- |     |
| --- | --- | --- | ---------- | ------ | ------------- | ------- | --- | ------ | ------ | ---------- | --- |
|     |     |     | ci¾n       | nula), | pero          | ignoran | los | datos  | reales | del SIATA. |     |
Alta Susceptibilidad La red es altamente sensible a la discretizaci¾n y el
al Ruido y ruido inherente de los datos puntuales de las estacio-
Sobreajuste nes SIATA. Esto puede llevar a que la red ôsobreajus-
|     |     |     | teö       | los parßmetros |              | fÝsicos     | (e.g.,        | alterar |                | artificialmente |       |
| --- | --- | --- | --------- | -------------- | ------------ | ----------- | ------------- | ------- | -------------- | --------------- | ----- |
|     |     |     | el        | coeficiente    | de           | difusi¾n    | D)            | solo    | para           | justificar      | picos |
|     |     |     | an¾malos  |                | o ruido      | en          | las lecturas, |         | comprometiendo |                 | la    |
|     |     |     | fidelidad |                | de la fÝsica | subyacente. |               |         |                |                 |       |
7

| Funci¾n    | de PÚrdida | Global    |           |      |
| ---------- | ---------- | --------- | --------- | ---- |
| La funci¾n | de pÚrdida | total es: |           |      |
|            |            | L(u(?))   | = L (?)+L | (?). |
|            |            |           | datos EDP |      |
La fase interpolativa busca primero minimizar L para establecer una base de
datos
campo escalar fÝsicamente plausible antes de introducir la penalizaci¾n estricta de
| L   | para la identificaci¾n | de ? | .   |     |
| --- | ---------------------- | ---- | --- | --- |
| EDP |                        | 0    |     |     |
8
