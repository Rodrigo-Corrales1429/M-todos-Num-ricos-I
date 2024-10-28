# Metodos-Numericos-I
# Este repositorio va a servir como un archivo digital para el curso de Métodos Numéricos I de la maestría en física y tecnología avanzada de la UAEH

%%%
2.5 
%%%
 import numpy as np
import matplotlib.pyplot as plt

#Se define la función
def f(x):
    return x**6 + 0.1 * np.log(np.abs(1+3*(1-x)))

#Creamos el intervalo donde vamos a trabajar con 100 pasos
x_val= np.linspace(0.5,1.5,100)
#Obtenemos los valores de y para la función dada
y_val= f(x_val)
#Gráfica de la función a 100 pasos
plt.figure(figsize=(10,5))
#Características que quiero que tenga
plt.plot(x_val, y_val, label='100 puntos', color='red')
#El nombre de la gráfica
plt.title('Gráfica de f(x) con 100 pasos')
#Ponemos el nombre al eje x
plt.xlabel('x')
#Ponemos el nombre al eje y
plt.ylabel('f(x)')
plt.legend()
plt.show

#Vamos a comparar la gráfica con una a 100,000 pasos
x_val1=np.linspace(0.5,1.5,100000)
y_val1= f(x_val1)

#Gráfica de la otra función para para comparar
plt.figure(figsize=(10,5))
plt.plot(x_val1, y_val1, label='100,000',color='blue')
plt.title('Gráfica de f(x) con 100,000')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show

%%%
2.8
%%%
#a)
import numpy as np
import matplotlib.pyplot as plt

#Se define la función
def f(x):
    return (1-np.cos(x))/x**2

#Valor de las variables. Hacemos una lista para tomar los valores de x.
x_val=[]
for i in range(1,101):
    x_val.append(0.1 * i)
#Usamos array 
x_val=np.array(x_val)
y_val=f(x_val)
#Graficamos la función 
plt.plot(x_val,y_val,label='f(x)',color='purple')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gráfica de (1-cos(x))/x**2')
#Agregamos cuadricula
plt.grid(True)
plt.legend()
plt.show()

#b)
import sympy as sy

#Definimos la variable simbólica x, como lo hicimos en clase
x=sy.symbols('x')

#Definimos la función f(x)
f_x=(1-sy.cos(x))/x**2

#El límite de f(x) cuando x tiende a 0, usando l'Hopital.
lim_f_x=sy.limit(f_x,x,0)
#Imprimimos el resultado
print(f'El límite de f(x) cuando x tiende a 0 es: {lim_f_x}')

#c)
import numpy as np

#El valor de x 
x_val=1.2e-8

#Ponemos la función
f_x_val=(1-np.cos(x_val))/x_val**2

#Imprimimos el resultado
print(f' f(x_val)={f_x_val}')

#El valor es bastante grande, por lo que no tiene sentido. Esto es porque x es muy pequeño y el error se magnifica.

#d)
# La identidad trigonométrica para evitar la cancelación es (1-cos(x))=2*sin^2(x/2)

import numpy as np

#Se define el nuevo valor de x
x_val=1.2e-8

#Sustituimos en la función la nueva identidad trigonométrica
f_x_new=(2*np.sin(x_val/2)**2)/x_val**2

#Imprimimos el resultado
print(f'El nuevo valor de la función, ahora es {f_x_new}')

