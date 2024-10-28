# Metodos-Numericos-I
# Este repositorio va a servir como un archivo digital para el curso de Métodos Numéricos I de la maestría en física y tecnología avanzada de la UAEH

#2.5 
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

#2.8
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

#2.11
def naive(C,x):
    Px=0
    for i in range(len(C)):
        Px=Px+C[i]*x**i
    return Px
coeffs= [(-11)**i for i in reversed(range(8))]
print("El resultado usando naive es: ", naive(coeffs,11.01))
def horner(coeffs,x):
    #Empezamos de atrás a adelante
    result=coeffs[-1]
    x=11.01
    #Iteramos con i del penúltimo al primero
    for i in range(len(coeffs)-2,-1,-1):
    #El algoritmo es
        result=result*x+coeffs[i]
    return result
#El resultado con la regla de Horner
horner_result=horner(coeffs,x)
print("El resultado usando Horner es: ", horner_result)
#Usando Horner es más preciso que de la forma naive.

#2.14
import math

def taylor_sin(x,n=10):
    #Calcula sin(x) con la serie de Taylor hasta n términos.
    result=0
    #Primer término
    term=x
    for n in range(n):
        result +=term
        #Términos nuevos
        term *= -x**2/((2*n+2)*(2*n+3))
    return result
#
#Evaluando en x=0.1
x1=0.1
sin_x1=taylor_sin(x1)
print(f"sin({x1}) usando la serie de Taylor: {sin_x1}")

# Evaluación con x=40
k=40
#Estamos pasando el argumento de sin a radianes
x2=k*(math.pi/180)
sin_x2=taylor_sin(x2)
print(f"sin(40) usando la serie de Taylor con x= {x2}: {sin_x2}")

#Comparamos el método con serie de Taylor y el valor exacto
real_sin_x1=math.sin(x1)
real_sin_x2=math.sin(x2)
print(f"Valor exacto de sin({x1}):",{real_sin_x1})
print(f"Valor exacto de sin(40): ",{real_sin_x2})

#2.17
import numpy as np

# Definir x
x = 0.5

# Funciones iniciales j0(x) y j1(x)
j0 = np.sin(x) / x
j1 = (np.sin(x) / x**2) - (np.cos(x) / x)

# Lista para almacenar los valores de jn(x)
j_val = [j0, j1]

# Recursión hacia adelante para calcular jn(x) hasta j8(x)
for n in range(1, 8):
    j_n = (2*n + 1) / x * j_val[-1] - j_val[-2]
    j_val.append(j_n)

# El resultado "ingenuo" de j8(0.5)
j8_forward = j_val[-1]
j8_forward
# Valor de n inicial para la recursión hacia atrás
n_max = 15

# Valores arbitrarios iniciales
j_max = 1.0
j_max_menos_1 = 1.0

# Lista para almacenar los valores de jn(x) en recursión hacia atrás
j_backward = [j_max, j_max_menos_1]

# Recursión hacia atrás desde n_max hasta n=0
for n in range(n_max, 0, -1):
    j_next = (2*n + 1) / x * j_backward[-1] - j_backward[-2]
    j_backward.append(j_next)

# Invertir la lista para que coincida con el orden de n
j_backward.reverse()

# Normalización
j0_comput = j_backward[0]
j8_comput = j_backward[8]

# Normalización usando j0(0.5) real
j8_normalizada = j8_comput * (j0 / j0_comput)
j8_normalizada
print(f"j8(0.5) calculado por recursión hacia adelante: {j8_forward}")
print(f"j8(0.5) calculado por recursión hacia atrás (normalizado): {j8_normalized}")
