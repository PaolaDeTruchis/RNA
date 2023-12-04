# EDO

En este directorio, probe resolver Ecuaciones Diferenciales Ordinarias.

El trabajo se compoarte en 4 parte para cada de las 4 actividades que tenias que hacer

Part1 : Dise˜nar una capa en keras que transforme imagenes a color en escala de grises.

Part2 : Entrena una red neuronal para que reproduzca las siguientes funciones en el intervalo de [-1,1].

Part3 : Dise˜nar una capa entrenable que represente un polinomio grado 3.

Part4 : Entrenar una red neuronal que de la solucion de ecuaciones diferenciales.
En este parte necesitamos dos class: EDOSlover y PDESolver. 
Necesitaba mucho tiempo para entender lo que tenia que hacer. No habia que entendido que EDOSlver permite reslover la primera ecuacion : xy′ + y = x2 cos x con y(0) = 0 y PDESolver, permite resolver la secunda: d²y/dx² = −y con y(0) = 1, y'(0) = −0.5. Entonces, crei una nueva branch llamada 'EDO2', pero veo que sere mejor de crear Part4bis donde pordia resolver la otra ecuacion. Luega pensaba que mis resultados estaban mal a causa de un error en el codigo. Me tomaba mucho tiempo realizar que el problema estaba que la red no estaba bien entrenada.

Entonces para obtener el grafico de la ecuacion 1 tienes que correr el codigo de la 'branch' main y para obtener el grafico de la ecuacion 2, tienes que correr el codigo de la 'branch' EDO2.



