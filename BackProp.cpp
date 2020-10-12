#include "BackProp.hpp"
#include <time.h>
#include <stdlib.h>


CBackProp::CBackProp(int nl,int *sz,double b,double a):beta(b),alpha(a)
{

	numl=nl;
	lsize=new int[numl];

	for(int i=0;i<numl;i++){
		lsize[i]=sz[i];
	}

	out = new double*[numl];

	for( i=0;i<numl;i++){
		out[i]=new double[lsize[i]];
	}

	delta = new double*[numl];

	for(i=1;i<numl;i++){
		delta[i]=new double[lsize[i]];
	}

	weight = new double**[numl];

	for(i=1;i<numl;i++){
		weight[i]=new double*[lsize[i]];
	}
	for(i=1;i<numl;i++){
		for(int j=0;j<lsize[i];j++){
			weight[i][j]=new double[lsize[i-1]+1];
		}
	}

	prevDwt = new double**[numl];

	for(i=1;i<numl;i++){
		prevDwt[i]=new double*[lsize[i]];

	}
	for(i=1;i<numl;i++){
		for(int j=0;j<lsize[i];j++){
			prevDwt[i][j]=new double[lsize[i-1]+1];
		}
	}

	srand((unsigned)(time(NULL)));
	for(i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			for(int k=0;k<lsize[i-1]+1;k++)
				weight[i][j][k]=(double)(rand())/(RAND_MAX/2) - 1;

	for(i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			for(int k=0;k<lsize[i-1]+1;k++)
				prevDwt[i][j][k]=(double)0.0;

}



CBackProp::~CBackProp()
{
	for(int i=0;i<numl;i++)
		delete[] out[i];
	delete[] out;

	for(i=1;i<numl;i++)
		delete[] delta[i];
	delete[] delta;

	for(i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			delete[] weight[i][j];
	for(i=1;i<numl;i++)
		delete[] weight[i];
	delete[] weight;

	for(i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			delete[] prevDwt[i][j];
	for(i=1;i<numl;i++)
		delete[] prevDwt[i];
	delete[] prevDwt;

	//	free layer info
	delete[] lsize;
}

double CBackProp::sigmoid(double in)
{
		return (double)(1/(1+exp(-in)));
}

double CBackProp::mse(double *tgt) const
{
	double mse=0;
	for(int i=0;i<lsize[numl-1];i++){
		mse+=(tgt[i]-out[numl-1][i])*(tgt[i]-out[numl-1][i]);
	}
	return mse/2;
}

double CBackProp::Out(int i) const
{
	return out[numl-1][i];
}

void CBackProp::ffwd(double *in)
{
	double sum;
	for(int i=0;i<lsize[0];i++)
		out[0][i]=in[i];  
	for(i=1;i<numl;i++){				
		for(int j=0;j<lsize[i];j++){		
			sum=0.0;
			for(int k=0;k<lsize[i-1];k++){		
				sum+= out[i-1][k]*weight[i][j][k];	
			}
			sum+=weight[i][j][lsize[i-1]];		
			out[i][j]=sigmoid(sum);				
		}
	}
}

void CBackProp::bpgt(double *in,double *tgt)
{
	double sum;

	ffwd(in);

	for(int i=0;i<lsize[numl-1];i++){
		delta[numl-1][i]=out[numl-1][i]*
		(1-out[numl-1][i])*(tgt[i]-out[numl-1][i]);
	}

	for(i=numl-2;i>0;i--){
		for(int j=0;j<lsize[i];j++){
			sum=0.0;
			for(int k=0;k<lsize[i+1];k++){
				sum+=delta[i+1][k]*weight[i+1][k][j];
			}
			delta[i][j]=out[i][j]*(1-out[i][j])*sum;
		}
	}
	
	for(i=1;i<numl;i++){
		for(int j=0;j<lsize[i];j++){
			for(int k=0;k<lsize[i-1];k++){
				weight[i][j][k]+=alpha*prevDwt[i][j][k];
			}
			weight[i][j][lsize[i-1]]+=alpha*prevDwt[i][j][lsize[i-1]];
		}
	}

	for(i=1;i<numl;i++){
		for(int j=0;j<lsize[i];j++){
			for(int k=0;k<lsize[i-1];k++){
				prevDwt[i][j][k]=beta*delta[i][j]*out[i-1][k];
				weight[i][j][k]+=prevDwt[i][j][k];
			}
			prevDwt[i][j][lsize[i-1]]=beta*delta[i][j];
			weight[i][j][lsize[i-1]]+=prevDwt[i][j][lsize[i-1]];
		}
	}
}

