/* 
 * Solves the Panfilov model using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory 
 * and reimplementation by Scott B. Baden, UCSD
 * 
 * Modified and  restructured by Didem Unat, Koc University
 *
 * Futher Modified by Beakal Gizachew Assefa
 * Added : MPI and OMP parallelization
 */
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
#include<omp.h>

using namespace std;

#define NB 1
#define NC_B 2
// Utilities
// 
// Timer
// Make successive calls and take a difference to get the elapsed time.
static const double kMicro = 1.0e-6;
double getTime()
{
	struct timeval TV;
	struct timezone TZ;

	const int RC = gettimeofday(&TV, &TZ);
	if(RC == -1) {
		cerr << "ERROR: Bad call to gettimeofday" << endl;
		return(-1);
	}

	return( ((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec) );

}  // end getTime()

// Allocate a 2D array
double **alloc2D(int m,int n){
	double **E;
	int nx=n, ny=m;
	E = (double**)malloc(sizeof(double*)*ny + sizeof(double)*nx*ny);
	assert(E);
	int j;
	for(j=0;j<ny;j++)
		E[j] = (double*)(E+ny) + j*nx;
	return(E);
}

// Reports statistics about the computation
// These values should not vary (except to within roundoff)
// when we use different numbers of  processes to solve the problem
double stats(double **E, int m, int n, double *_mx){
	double mx = -1;
	double l2norm = 0;
	int i, j;
	for (j=1; j<=m; j++)
		for (i=1; i<=n; i++) {
			l2norm += E[j][i]*E[j][i];
			if (E[j][i] > mx)
				mx = E[j][i];
		}
	*_mx = mx;
	l2norm /= (double) ((m)*(n));
	l2norm = sqrt(l2norm);
	return l2norm;
}
// where m is row and n is column
double MPI_Stats(double **E, int m, int n, double *_mx,int *_size){
	double mx = -1;
	double l2 = 0; //l2 is sum of squares
	int size = m*n;
	int i, j;
	for (j=1; j<=m; j++)
		for (i=1; i<=n; i++) {
			l2 += E[j][i]*E[j][i];
			if (E[j][i] > mx)
				mx = E[j][i];
		}
	*_mx = mx;
	*_size = size;
	return l2;
}

// External functions
extern "C" {
void splot(double **E, double T, int niter, int m, int n);
}
void cmdLine(int argc, char *argv[], double& T, int& n, int& px, int& py, int& plot_freq, int& no_comm, int&num_threads);


//m_total and n_totla and the total number of rows and columns respectively
void simulate (double** E,  double** E_prev,double** R,
		const double alpha, const int n_total, const int m_total, const double kk,
		const double dt, const double a, const double epsilon,
		const double M1,const double  M2, const double b,int n,int m, int px, int py){
	/*******************************************************************/
	double t = 0.0;
	int root = 0;
	int niter;
	int rank =0, np=1;
	MPI_Comm_size(MPI_COMM_WORLD,&np);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Request	send_request,recv_request;
	MPI_Request	send_request2,recv_request2;
	MPI_Request	send_requestc,recv_requestc;
	MPI_Request	send_request2c,recv_request2c;
	int plot_msgsiz = (m+1)*(n+1)*sizeof(double);
	int col_msgsiz = (m+1)*sizeof(double);
	double * sendbuffer = new double [m];
	double * recvbuffer = new double [m];
	double * sendbuffer2 = new double [m];
	double * recvbuffer2 = new double [m];
	double* submatrix = new double[(m+1)*(n+1)];
	double* revmatrix = new double[(m+2)*(n+2)];


	int i, j;
	/*
	 * Copy data from boundary of the computational box
	 * to the padding region, set up for differencing
	 * on the boundary of the computational box
	 * Using mirror boundaries
	 */
    #pragma omp parallel for num_threads(num_threads) private(i)
	for (i=1; i<=n; i++)
			E_prev[0][i] = E_prev[2][i];
		for (i=1; i<=n; i++)
			E_prev[m+1][i] = E_prev[m-1][i];

	#pragma omp parallel for num_threads(num_threads)private(j)
	for (j=1; j<=m; j++)
		E_prev[j][0] = E_prev[j][2];
	for (j=1; j<=m; j++)
		E_prev[j][n+1] = E_prev[j][n-1];


	// Solve for the excitation, a PDE
	int colid = rank % px; //column id
	int rowid = rank / px;//row id of the processes

	#pragma omp parallel for num_threads(num_threads)private(i)
	for (i=1; i<=n; i++) {
		E_prev[0][i] = E_prev[2][i];
		E_prev[m+1][i] = E_prev[m-1][i];
	}

	#pragma omp parallel for num_threads(num_threads)private(j)
	for (j=1; j<=m; j++){
		E_prev[j][0] = E_prev[j][2];
		E_prev[j][n+1] = E_prev[j][n-1];
	}


	if (py > 1) {

		if (rowid == 0) { //top
			//send a row of information
			MPI_Isend(&E_prev[m][1], n , MPI_DOUBLE, rank + px , NB, MPI_COMM_WORLD, &send_request);
			MPI_Irecv(&E_prev[m+1][1], n, MPI_DOUBLE, rank + px, NB, MPI_COMM_WORLD, &recv_request);
			MPI_Wait (&send_request,MPI_STATUS_IGNORE);
			MPI_Wait (&recv_request,MPI_STATUS_IGNORE);
		}

		else if (rowid == py-1) { //bottom

			MPI_Isend(&E_prev[1][1], n, MPI_DOUBLE, rank - px, NB, MPI_COMM_WORLD, &send_request);

			MPI_Irecv(&E_prev[0][1], n, MPI_DOUBLE, rank - px, NB, MPI_COMM_WORLD, &recv_request);
			MPI_Wait (&send_request,MPI_STATUS_IGNORE);
			MPI_Wait (&recv_request,MPI_STATUS_IGNORE);
		}
		else { //middle

			MPI_Isend(&E_prev[m][1], n , MPI_DOUBLE, rank + px , NB, MPI_COMM_WORLD, &send_request);

			MPI_Irecv(&E_prev[m+1][1], n, MPI_DOUBLE, rank + px, NB, MPI_COMM_WORLD, &recv_request);
			MPI_Wait (&send_request,MPI_STATUS_IGNORE);
			MPI_Wait (&recv_request,MPI_STATUS_IGNORE);

			MPI_Isend(&E_prev[1][1], n, MPI_DOUBLE, rank - px, NB, MPI_COMM_WORLD, &send_request);

			MPI_Irecv(&E_prev[0][1], n, MPI_DOUBLE, rank - px, NB, MPI_COMM_WORLD, &recv_request);
			MPI_Wait (&send_request,MPI_STATUS_IGNORE);
			MPI_Wait (&recv_request,MPI_STATUS_IGNORE);
		}
	}
	if (px > 1) {
		if (colid == 0) {
			for (int j=1; j<=m; j++){
				sendbuffer[j-1] = E_prev[j][n];
			}

			MPI_Isend(&sendbuffer[0], m, MPI_DOUBLE,rank+1,NC_B, MPI_COMM_WORLD, &send_requestc);
			MPI_Irecv(&recvbuffer[0], m, MPI_DOUBLE, rank+1,NC_B, MPI_COMM_WORLD, &recv_requestc);

			MPI_Wait (&recv_requestc,MPI_STATUS_IGNORE);
			for (int j=1;j<=m;j++) {
				E_prev[j][n+1] = recvbuffer[j-1];
			}
			MPI_Wait (&send_requestc,MPI_STATUS_IGNORE);
		}
		// Rightmost
		if (colid == px-1) { 	// Packing to send left

			for (int j=1; j<=m; j++){
				sendbuffer[j-1] = E_prev[j][1];

			}

			MPI_Isend(&sendbuffer[0], m, MPI_DOUBLE,rank-1,NC_B, MPI_COMM_WORLD, &send_requestc);
			MPI_Irecv(&recvbuffer[0], m, MPI_DOUBLE, rank-1,NC_B, MPI_COMM_WORLD, &recv_requestc);


			MPI_Wait (&recv_requestc,MPI_STATUS_IGNORE);

			#pragma omp parallel for num_threads(num_threads) private(j)
			for (int j=1; j<=m; j++){
				E_prev[j][0] = recvbuffer[j-1];
			}
			MPI_Wait (&send_requestc,MPI_STATUS_IGNORE);
		}
		// at the middle
		if (colid>0 && colid <px-1)  {

			for (int j=1; j<=m; j++){
				// Packing to send right
				sendbuffer[j-1] = E_prev[j][n];
				// Packing to send left
				sendbuffer2[j-1] = E_prev[j][1];
			}
			MPI_Isend(&sendbuffer[0], m, MPI_DOUBLE,rank+1,NC_B, MPI_COMM_WORLD, &send_requestc);
			MPI_Irecv(&recvbuffer[0], m, MPI_DOUBLE, rank+1,NC_B, MPI_COMM_WORLD, &recv_requestc);
			// Recieve packed array
			MPI_Wait (&recv_requestc,MPI_STATUS_IGNORE);

			MPI_Wait (&send_requestc,MPI_STATUS_IGNORE);


			MPI_Isend(&sendbuffer2[0], m, MPI_DOUBLE,rank-1,NC_B, MPI_COMM_WORLD, &send_request2c);

			MPI_Irecv(&recvbuffer2[0], m, MPI_DOUBLE, rank-1,NC_B, MPI_COMM_WORLD, &recv_request2c);

			MPI_Wait (&recv_request2c,MPI_STATUS_IGNORE);
			// Recieve packed array
			for (int j=1; j<=m; j++){
				E_prev[j][n+1] = recvbuffer[j-1];
				E_prev[j][0] = recvbuffer2[j-1];
			}
			MPI_Wait (&send_request2c,MPI_STATUS_IGNORE);
		}

	}
	/***********************************/
	// Solve for the excitation, the PD

	for (j=1; j<=m; j++){
		for (i=1; i<=n; i++) {
			E[j][i] = E_prev[j][i]+alpha*(E_prev[j][i+1]+E_prev[j][i-1]-4*E_prev[j][i]+E_prev[j+1][i]+E_prev[j-1][i]);
		}
	}

	/*
	 * Solve the ODE, advancing excitation and recovery to the
	 *     next timtestep
	 */
	for (j=1; j<=m; j++){
		for (i=1; i<=n; i++)
			E[j][i] = E[j][i] -dt*(kk* E[j][i]*(E[j][i] - a)*(E[j][i]-1)+ E[j][i] *R[j][i]);
	}

	for (j=1; j<=m; j++){
		for (i=1; i<=n; i++)
			R[j][i] = R[j][i] + dt*(epsilon+M1* R[j][i]/( E[j][i]+M2))*(-R[j][i]-kk* E[j][i]*(E[j][i]-b-1));
	}

}

int main (int argc, char** argv)
{
	/*
	 *  Solution arrays
	 *   E is the "Excitation" variable, a voltage
	 *   R is the "Recovery" variable
	 *   E_prev is the Excitation variable for the previous timestep,
	 *      and is used in time integration
	 */
	double **E, **R, **E_prev;

	// Various constants - these definitions shouldn't change
	const double a=0.1, b=0.1, kk=8.0, M1= 0.07, M2=0.3, epsilon=0.01, d=5e-5;

	double T=1000.0;
	int m=200,n=200;
	int plot_freq = 0;
	int px = 1, py = 1;
	int no_comm = 0;
	int num_threads=1;


	int niters=100;
	int root = 0;


	cmdLine( argc, argv, T, n,px, py, plot_freq, no_comm, num_threads);
	m = n;

	MPI_Init(&argc,&argv);

	int col =0, row = 0;
	int nprocs=1, myrank=0;

	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);



	if(myrank == 0)
		printf("number of processes : %d\n",nprocs);
	// Allocate contiguous memory for solution arrays
	// The computational box is defined on [1:m+1,1:n+1]
	// We pad the arrays in order to facilitate differencing on the
	// boundaries of the computation box


	int ny = py;
	int nx = px;

	row = m / py;
	int rrem = m % py;	// the process executing the last row will compute row + rrem rows.

	col = n / px;
	int crem = n % px;

	int rowID= myrank / nx;
	int colID= myrank % nx;

	if ( rowID == ny-1 )
		row = row + rrem;


	if ( colID == nx-1 )
		col = col + crem;

	E = alloc2D(row+2,col+2);
	E_prev = alloc2D(row+2,col+2);
	R = alloc2D(row+2,col+2);


	int i,j;

	//------------------------------------------ Serial Version
	int rinterval = m / py;
	int cinterval = n / px;

	int m_new = row;
	int n_new = col;
	for (j=1; j<=m_new; j++)
		for (i=1; i<= n_new; i++){
			E_prev[j][i] = R[j][i] = 0;
		}
	//calculating the beginning and ending global indexes of the column
	int cbeg =0 , cend = 0;
	cbeg = colID * (cinterval) + 1;
	cend = cbeg + col -1;


	for (j=1; j<=m_new; j++){
		for (i=1; i<= n_new ; i++){
			int colindex = cbeg + i - 1;
			if (colindex >= n/2+1 && colindex <= n+1)
				E_prev[j][i] = 1.0;
		}
	}

	// calculating the beginning and ending global indexes of the row
	int rbeg =0 , rend = 0;
	rbeg = rowID*(rinterval) + 1;//deadly one:it's starting from 1 !
	rend = rbeg + row -1;

	for ( j = 1 ; j <= m_new;j++){
		//calculate the global index
		int rowindex = rbeg + j - 1;
		// if it falls into the range, change it to one
		if (rowindex >= m/2+1 && rowindex <= m+1){
			for (int i = 1; i<= n_new;i++){
				R[j][i] = 1.0;
			}
		}
	}

	double dx = 1.0/n;
	// For time integration, these values shouldn't change
	double rp= kk*(b+1)*(b+1)/4;
	double dte=(dx*dx)/(d*4+((dx*dx))*(rp+kk));
	double dtr=1/(epsilon+((M1/M2)*rp));
	double dt = (dte<dtr) ? 0.95*dte : 0.95*dtr;
	double alpha = d*dt/(dx*dx);

	if(myrank == 0){
		cout << "Grid Size       : " << n << endl;
		cout << "Duration of Sim : " << T << endl;
		cout << "Time step dt    : " << dt << endl;
		cout << "Process geometry: " << px << " x " << py << endl;
		cout << endl;
	}
	// Start the timer
	double t0 = getTime();

	double t = 0.0;
	// Integer timestep number
	int niter=0;

	while (t<T) {

		t += dt;
		niter++;

		simulate(E, E_prev, R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b,col,row, px,py);
		MPI_Barrier(MPI_COMM_WORLD); //wait till all the processes finish

		double **tmp = E; E = E_prev; E_prev = tmp;


		if (plot_freq){
			if(myrank == 0){
				int k = (int)(t/plot_freq);
				if ((t - k * plot_freq) < dt){
					splot(E,t,niter,m+2,n+2);}
			}
		}
	} //end of while loop


	double time_elapsed = getTime() - t0;

	double Gflops = (double)(niter * (1E-9 * n * n ) * 28.0) / time_elapsed ;
	double BW = (double)(niter * 1E-9 * (n * n * sizeof(double) * 4.0  ))/time_elapsed;
	if(myrank == 0){
		cout << "Size of local grid          : " << row * col << endl;
		cout << "Number of Iterations        : " << niter << endl;
		cout << "Elapsed Time (sec)          : " << time_elapsed << endl;
		cout << "Sustained Gflops Rate       : " << Gflops << endl;
		cout << "Sustained Bandwidth (GB/sec): " << BW << endl << endl;
	}
	double localMx;
	int localSize;
	double localSquareSum = MPI_Stats(E_prev,row,col,&localMx,&localSize);

	double *rMx;
	int *rSize;
	double *rSSum;

	if ( myrank == 0) {

		rMx = (double *)malloc(nprocs*sizeof(double));
		rSize = (int *)malloc(nprocs*sizeof(int));
		rSSum = (double *)malloc(nprocs*sizeof(double));
	}
	MPI_Gather(&localMx,1,MPI_DOUBLE,rMx,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Gather(&localSize,1,MPI_INT,rSize,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Gather(&localSquareSum,1,MPI_DOUBLE,rSSum,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

	double GMax = -1;
	double l2norm = 0;
	int totalSize=0;
	if(myrank == 0)
	{
		for(i=0;i<nprocs;i++)
		{
			if(rMx[i] > GMax)
				GMax = rMx[i];

			l2norm += rSSum[i];
			totalSize += rSize[i];
		}
		l2norm = l2norm/totalSize;
		l2norm = sqrt(l2norm);

		cout << "Max: " << GMax << " L2norm : " << l2norm << endl;
	}

	if (plot_freq){
		cout << "\n\nEnter any input to close the program and the plot..." << endl;
		getchar();
	}

	free (E);
	free (E_prev);
	free (R);

	MPI_Finalize(); //finlaize PMI
	return 0;
}

