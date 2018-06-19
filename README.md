# BackProp

### David Neyra Gutierrez

BackPropagation tarea de Tópicos en Inteligencia Artificial  

Se creo la Clase llamada CBackProp la cual inicializamos con
* numLayers Numero de Layers que le damos
* LSz  Input
* beta cada cuanto va a variar 
* alpha valor de aceptacion

```c
CBackProp *bp = new CBackProp(numLayers, lSz, beta, alpha); 
```

para poder Compilar

```c
g++ NeuralNet.cpp -o BackProp
```

ejecucion

```c
./BackProp
```

Funcion sigmoid

```c
double CBackProp::sigmoid(double in)
{
		return (double)(1/(1+exp(-in)));
}
```
Estructura de la Clase

```c
class CBackProp{

	double **out;
	double **delta;
	double ***weight;
	int numl;
	int *lsize;
	double beta;
	double alpha;
	double ***prevDwt;
	double sigmoid(double in);

public:

	~CBackProp();
	CBackProp(int nl,int *sz,double b,double a);
	void bpgt(double *in,double *tgt);
	void ffwd(double *in);
	double mse(double *tgt) const;	
	double Out(int i) const;
};
```

## Bibliografia
1. A Step by Step Backpropagation Example [Link](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)
2. Neural networks and back-propagation explained in a simple way [Link](https://medium.com/datathings/neural-networks-and-backpropagation-explained-in-a-simple-way-f540a3611f5e)
3. Backpropagation Step by Step [Link](http://hmkcode.github.io/ai/backpropagation-step-by-step/)
