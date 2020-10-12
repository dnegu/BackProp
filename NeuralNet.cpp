
#include "BackProp.hpp"

int main(int argc, char* argv[])
{


	double data[][4]={
				0,	0,	0,	0,
				0,	0,	1,	1,
				0,	1,	0,	1,
				0,	1,	1,	0,
				1,	0,	0,	1,
				1,	0,	1,	0,
				1,	1,	0,	0,
				1,	1,	1,	1 };

	double testData[][3]={
								0,      0,      0,
                                0,      0,      1,
                                0,      1,      0,
                                0,      1,      1,
                                1,      0,      0,
                                1,      0,      1,
                                1,      1,      0,
                                1,      1,      1};

	

	int numLayers = 4, lSz[4] = {3,3,2,1};

	
	double beta = 0.3, alpha = 0.1, Thresh =  0.00001;
	long num_iter = 2000000;
	CBackProp *bp = new CBackProp(numLayers, lSz, beta, alpha);
	
	cout<< endl <<  "Entrenando..." << endl;	
	for (long i=0; i<num_iter ; i++)
	{
		
		bp->bpgt(data[i%8], &data[i%8][3]);

		if( bp->mse(&data[i%8][3]) < Thresh) {
			cout << endl << "Red entrenada en  " << i << " iteraciones." << endl;
			cout << "MSE:  " << bp->mse(&data[i%8][3]) 
				 <<  endl <<  endl;
			break;
		}
		if ( i%(num_iter/10) == 0 )
			cout<<  endl <<  "MSE:  " << bp->mse(&data[i%8][3]) 
				<< "... Entrenando..." << endl;

	}
	
	if ( i == num_iter )
		cout << endl << i << " Iteraciones Completadas..." 
		<< "MSE: " << bp->mse(&data[(i-1)%8][3]) << endl;  	

	cout<< "Prediciendo con la Red Neuronal Entrenada...." << endl << endl;	
	for ( i = 0 ; i < 8 ; i++ )
	{
		bp->ffwd(testData[i]);
		cout << testData[i][0]<< "  " << testData[i][1]<< "  "  << testData[i][2]<< "  " << bp->Out(0) << endl;
	}

	return 0;
}



