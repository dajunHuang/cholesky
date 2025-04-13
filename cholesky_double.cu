#include <stdlib.h>
#include <math.h>

#include <cusolverDn.h>
long int n, k, nb;
int parseArguments(int argc,char *argv[])
{
    if(argc < 4)
    {
        printf("Needs n as inputs\n");
        return -1;
    }
    n = atoi(argv[1]);
    k = atoi(argv[2]);
    nb = atoi(argv[3]);
    return 0;
}

__global__
void clearTri(char uplo, long int m, long int n, double *a, long int lda)
{
	long int i = threadIdx.x + blockDim.x * blockIdx.x;
	long int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i<m && j<n) {
		if (uplo == 'l') {
			if (i>j) {
				a[i+j*lda] = 0;
			}
        } 
        else
        {
            if (i<j)
                a[i+j*lda] = 0;
		}
	}
}

template<typename T>
void printMatrixDeviceBlock(char *filename,int m, int n, T* dA, int lda)
{
    FILE *f = fopen(filename, "w");
	if (f == NULL) {
		printf("fault!\n");
		return;
	}
    //printf("Perform printmatrixdevice\n");
    float *ha;
    ha = (float*)malloc(sizeof(float));

    for(int i = 0;i<m;i++)
    {
        for(int j = 0;j<n;j++)
        {
            cudaMemcpy(&ha[0], &dA[i+j*lda], sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(f, "%lf", ha[0]);
            if (j == n - 1) fprintf(f, "\n");
			else fprintf(f, ",");
        }
    }
    fclose(f);
	//cudaMemcpy(ha, dA, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    //printMatrixFloat(filename, m, n, ha, lda);
    free(ha);
}

__global__
void setEye( long int m, long int n, double *a, long int lda)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
		if (i == j) 
			a[i+j*lda] = 1;
		else
			a[i+j*lda] = 0;
	}
}


cudaEvent_t begin, end;
void startTimer()
{
    cudaEventCreate(&begin);
    cudaEventRecord(begin);
    cudaEventCreate(&end);
}

float stopTimer()
{
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, begin, end);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return milliseconds;
}



float computeFrobeniusNorm(long int n, double *dA)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    double dn;
    int incx = 1;
    cublasDnrm2(handle, n * n, dA, incx, &dn);
    cublasDestroy(handle);
    return dn;
}

int main(int argc,char *argv[])
{
    if(parseArguments(argc, argv)==-1)
        return 0;
    

    // 分配设备矩阵
    double *A;
    cudaMalloc(&A, sizeof(double) * n * n);
    dim3 gridDim((n+15)/16,(n+15)/16);
    dim3 blockDim(16,16);
    
    setEye<<<gridDim, blockDim>>>(n ,n ,A, n);
     cublasHandle_t cublasHandle;
     cublasCreate(&cublasHandle);
       
    
    float orig_norm = computeFrobeniusNorm(n, A);
    double *A_reconstructed;
    cudaMalloc(&A_reconstructed,sizeof(double)*n*n);
    cudaMemcpy(A_reconstructed,A, sizeof(double)*n*n, cudaMemcpyDeviceToDevice); 
   

    cusolverDnHandle_t cusolverHandle;
    cusolverDnCreate(&cusolverHandle);
    int Lwork;
    cusolverDnDpotrf_bufferSize(cusolverHandle,
                                CUBLAS_FILL_MODE_LOWER,
                                nb,
                                A,
                                n,
                                &Lwork);
    
        double *work;
        cudaMalloc((void**)&work, sizeof(double) * Lwork);
        int *devInfo;
        cudaMalloc(&devInfo, sizeof(int));
        startTimer();
        if (n <= 8192) {
            cusolverDnDpotrf(cusolverHandle,
                            CUBLAS_FILL_MODE_LOWER,
                            n,
                            A,
                            n,
                            work,
                            Lwork,
                            devInfo);



        } else {
    
    for (int j=0;j<n;j+=k){

   
    for (int i = j; i < j + k -1 && i < n; i += nb) 
    {
        double ms;
        double snegone = -1.0;
        double sone = 1.0;
        
        cusolverDnDpotrf(cusolverHandle,
                         CUBLAS_FILL_MODE_LOWER,
                         nb,
                         A + i + i * n,
                         n,
                         work,
                         Lwork,
                         devInfo );
       
       
        if (n - i -  nb <= 0) {
        break; 
        }
        
        
        //startTimer();
        cublasDtrsm(cublasHandle,
                    CUBLAS_SIDE_RIGHT,
                    CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_T,
                    CUBLAS_DIAG_NON_UNIT,
                    n - i - nb,
                    nb,
                    &sone,
                    A + i + i * n,
                    n,
                    A + (i + nb) + i * n,
                    n);
        
        if (j + k -1 - i -  nb <= 0) {
            break; 
            }
        //startTimer();
       
        cublasDsyrk(cublasHandle,
                    CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N,
                    nb,
                    i+nb-j,
                    &snegone,
                    A + (i + nb)+ j*n,
                    n,
                    &sone,
                    A + i + nb + (i + nb) * n,
                    n);
       
        if (n - i - 2 * nb <= 0) {
            continue; 
        }
       
        //startTimer();
        cublasDgemm(cublasHandle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    n - i - 2 * nb,  // m
                    nb,              // n
                    i + nb - j,      // k
                    &snegone,        // alpha
                    A + (i + 2 * nb) + j * n,  // A
                    n,               // lda
                    A + (i + nb) + j * n,  // B
                    n,               // ldb
                    &sone,           // beta
                    A + i + 2 * nb + (i + nb) * n, // C
                    n);              // ldc

       
        }
        if (n-k-j<=0) {
            break; 
            }
        double negone = -1.0;
        double one = 1.0;
        cublasDsyrk(cublasHandle,
                    CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N,
                    k,
                    k+j,
                    &negone,
                    A + j + k,
                    n,
                    &one,
                    A + j + k + (j + k) * n,
                    n);
       
        if (n - j - 2 * k <= 0) {
            continue; 
        }
        cublasDgemm(cublasHandle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    n - j - 2 * k,  // m
                    k,              // n
                    j + k,          // k
                    &negone,        // alpha
                    A + j + 2 * k,  // A
                    n,              // lda
                    A + j + k,      // B
                    n,              // ldb
                    &one,           // beta
                    A + (j + 2 * k) + (j + k) * n, // C
                    n);              // ldc

       
    }


}


 double *L;
cudaMalloc(&L, sizeof(double) * n * n);
cudaMemcpy(L, A, sizeof(double) * n * n, cudaMemcpyDeviceToDevice);

clearTri<<<gridDim, blockDim>>>('u', n, n, L, n);
 //printMatrixDeviceBlock<float>("1LL.csv", n, n,  L , n);  


        double a = 1.0f;
        double b = -1.0f;
        cublasDsyrk(cublasHandle,
                    CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N,
                    n,
                    n,
                    &a,
                    L,
                    n,
                    &b,
                    A_reconstructed,
                    n);


float diff_norm = computeFrobeniusNorm(n, A_reconstructed);
printf("diff_norm: %e\n", diff_norm);
printf("orig_norm: %e\n", orig_norm);
float backward_error = diff_norm / orig_norm;
printf("Backward error: %e\n", backward_error);
    float ms = stopTimer();

    printf("Cholesky factorization takes %f ms: %f TFLOPs\n", ms, 1.0/3.0*n*n*n/ms/1e9);

    
  
    cudaFree(A);
    cudaFree(work);
    cudaFree(devInfo);
    // cudaFree(A_reconstructed);  
    cublasDestroy(cublasHandle);
    cusolverDnDestroy(cusolverHandle);

    return 0;
}
