#ifndef backprop_h
#define backprop_h

#include<assert.h>
#include<iostream.h>
#include<stdio.h>
#include<math.h>
using namespace std;

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

#endif