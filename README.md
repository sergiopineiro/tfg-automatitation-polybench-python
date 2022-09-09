# tfg-automatitation-polybench-python
Repositorio del TFG "Optimización automática de PolyBench-Python" de Sergio Piñeiro Bermúdez.


## Resumen

Este trabajo consiste en permitir la ejecución de los bancos de pruebas de PolyBench-Python
con diferentes configuraciones que se obtienen de optimizaciones poliédricas conseguidas
gracias al compilador poliédrico PoCC. De esta manera, se corrige y amplía la herramienta
PolyPy que actúa como intermediador entre los benchmarks que forman PolyBench-Python y
el compilador PoCC. Esto permitirá poder realizar un estudio empírico de las optimizaciones
poliédricas sobre un lenguaje interpretado como Python.
Por otra parte, también se desarrolla, dentro de PolyPy, una capa capaz de transformar el
código de los benchmarks al formato NumPy. Esta configuración también es posible utilizarla
en la ejecución de PolyBench-Python.


## Instalación


Para proceder a la correcta ejecución se debe de en primer lugar realizar las instalaciones de cada herramienta:

1. Instalar el compilador PoCC: https://sourceforge.net/projects/pocc/

2. Instalar los módulos requeridos por PolyBench-Python. En el directorio de PolyBench-Python:

    ~~~
    pip3 install -r  requirements.txt
    ~~~

3. Configurar los ficheros de configuración de PolyBench-Python y PolyPy:
    - PolyBench-Python/polybench-python.conf:
        
            Indicar el path a la ubicación de PolyPy.

    - PolyPy/src/polypy.conf:

            Indicar el path a la ubicación de PoCC.


## Ejecución

Para realizar ejecuciones con optimizaciones automáticas, desde el directorio de PolyBench-Python se puede ejecutar:
- Optimización poliédrica:
    ~~~
    python3 run-benchmark.py --pocc pluto paht/to/benchmark
    ~~~

- Optimización poliédrica con formato NumPy:

    ~~~
    python3 run-benchmark.py --pocc pluto-numpy paht/to/benchmark
    ~~~









    




