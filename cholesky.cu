#include <curand.h>
#include <cusolverDn.h>
#include <math.h>
#include <stdlib.h>

long int n, k, nb;

int parseArguments(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Needs n as inputs\n");
        return -1;
    }
    n = atoi(argv[1]);
    k = atoi(argv[2]);
    nb = atoi(argv[3]);
    return 0;
}

__global__ void clearTri(char uplo, long int m, long int n, float *a, long int lda) {
    long int i = threadIdx.x + blockDim.x * blockIdx.x;
    long int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < m && j < n) {
        if (uplo == 'l') {
            if (i > j) {
                a[i + j * lda] = 0;
            }
        } else {
            if (i < j) a[i + j * lda] = 0;
        }
    }
}

template <typename T>
void printMatrixDeviceBlock(char *filename, int m, int n, T *dA, int lda) {
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        printf("fault!\n");
        return;
    }
    // printf("Perform printmatrixdevice\n");
    float *ha;
    ha = (float *)malloc(sizeof(float));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cudaMemcpy(&ha[0], &dA[i + j * lda], sizeof(float),
                       cudaMemcpyDeviceToHost);
            fprintf(f, "%lf", ha[0]);
            if (j == n - 1)
                fprintf(f, "\n");
            else
                fprintf(f, ",");
        }
    }
    fclose(f);
    // cudaMemcpy(ha, dA, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    // printMatrixFloat(filename, m, n, ha, lda);
    free(ha);
}

__global__ void setEye(long int m, long int n, float *a, long int lda) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < m && j < n) {
        if (i == j)
            a[i + j * lda] = 1;
        else
            a[i + j * lda] = 0;
    }
}

__global__ void addEye(long int m, long int n, float *a, long int lda) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < m && j < n) {
        if (i == j) a[i + j * lda] += n / 10.0;
        // else
        // 	a[i+j*lda] = 0;
    }
}

cudaEvent_t begin, end;
void startTimer() {
    cudaEventCreate(&begin);
    cudaEventRecord(begin);
    cudaEventCreate(&end);
}

float stopTimer() {
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, begin, end);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return milliseconds;
}

void generateNormalMatrix(float *dA, int m, int n) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    int seed = rand() % 3000;
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormal(gen, dA, m * n, 0, 1);
}
__global__ void symmetricFromProduct(float *A, int n) {
    // 计算线程负责的矩阵元素索引 (i, j)
    int i = threadIdx.x + blockDim.x * blockIdx.x;  // 行索引
    int j = threadIdx.y + blockDim.y * blockIdx.y;  // 列索引

    // 确保线程在有效范围内
    if (i < n && j < n) {
        float value = 0.0f;
        for (int k = 0; k < n; ++k) {
            // 使用列主序的存储规则访问矩阵元素
            value += A[i + k * n] + A[j + k * n];
        }
        A[i + j * n] = value;  // 对称矩阵存储在列主序中
    }
}

float computeFrobeniusNorm(long int n, float *dA) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float dn;
    int incx = 1;
    cublasSnrm2(handle, n * n, dA, incx, &dn);
    cublasDestroy(handle);
    return dn;
}

int main(int argc, char *argv[]) {
    if (parseArguments(argc, argv) == -1) return 0;

    printf("n = %d\n", n);

    // 分配设备矩阵
    float *A;
    cudaMalloc(&A, sizeof(float) * n * n);
    dim3 gridDim((n + 15) / 16, (n + 15) / 16);
    dim3 blockDim(16, 16);
    // generateNormalMatrix(A,n, n);
    setEye<<<gridDim, blockDim>>>(n, n, A, n);
    // float *L;
    // cudaMalloc(&L, sizeof(float) * n * n);
    // float *A_LL;
    // cudaMalloc(&A_LL, sizeof(float) * n * n);
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    //  float ssone = 1.0;
    //  float sszero = 0.0;
    //  cublasSgemmEx( cublasHandle,
    //                 CUBLAS_OP_N,
    //                 CUBLAS_OP_T,
    //                 n,
    //                 n,
    //                 n,
    //                 &ssone,
    //                 A,
    //                 CUDA_R_32F,
    //                 n,
    //                 A,
    //                 CUDA_R_32F,
    //                 n,
    //                 &sszero,
    //                 A_LL,
    //                 CUDA_R_32F,
    //                 n);
    // cudaMemcpy(A , A_LL, sizeof(float)*n*n, cudaMemcpyDeviceToDevice);
    // printMatrixDeviceBlock<float>("1A0.csv", n, n,  A , n);
    // symmetricFromProduct<<<gridDim, blockDim>>>(A, n);

    // cudaDeviceSynchronize();
    // setEye<<<gridDim, blockDim>>>(n ,n ,A, n);
    // printMatrixDeviceBlock<float>("1.csv", n, n, A, n);
    float orig_norm = computeFrobeniusNorm(n, A);
    float *A_reconstructed;
    cudaMalloc(&A_reconstructed, sizeof(float) * n * n);
    cudaMemcpy(A_reconstructed, A, sizeof(float) * n * n, cudaMemcpyDeviceToDevice);

    cusolverDnHandle_t cusolverHandle;
    cusolverDnCreate(&cusolverHandle);
    int Lwork;
    cusolverDnSpotrf_bufferSize(cusolverHandle, CUBLAS_FILL_MODE_LOWER, nb, A, n,
                                &Lwork);

    float *work;
    cudaMalloc((void **)&work, sizeof(float) * Lwork);
    int *devInfo;
    cudaMalloc(&devInfo, sizeof(int));
    startTimer();
    if (n <= 8192) {
        cusolverDnSpotrf(cusolverHandle, CUBLAS_FILL_MODE_LOWER, n, A, n, work,
                         Lwork, devInfo);

    } else {
        for (int j = 0; j < n; j += k) {
            for (int i = j; i < j + k - 1 && i < n; i += nb) {
                float ms;
                float snegone = -1.0;
                float sone = 1.0;

                cusolverDnSpotrf(cusolverHandle, CUBLAS_FILL_MODE_LOWER, nb,
                                 A + i + i * n, n, work, Lwork, devInfo);

                if (n - i - nb <= 0) {
                    break;
                }

                // startTimer();
                cublasStrsm(cublasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n - i - nb, nb, &sone,
                            A + i + i * n, n, A + (i + nb) + i * n, n);

                if (j + k - 1 - i - nb <= 0) {
                    break;
                }
                // startTimer();

                cublasSsyrk(cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, nb,
                            i + nb - j, &snegone, A + (i + nb) + j * n, n, &sone,
                            A + i + nb + (i + nb) * n, n);

                if (n - i - 2 * nb <= 0) {
                    continue;
                }

                // startTimer();
                cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                            n - i - 2 * nb,                 // m
                            nb,                             // n
                            i + nb - j,                     // k
                            &snegone,                       // alpha
                            A + (i + 2 * nb) + j * n,       // A
                            n,                              // lda
                            A + (i + nb) + j * n,           // B
                            n,                              // ldb
                            &sone,                          // beta
                            A + i + 2 * nb + (i + nb) * n,  // CF
                            n);                             // ldc
            }
            if (n - k - j <= 0) {
                break;
            }
            float negone = -1.0;
            float one = 1.0;
            cublasSsyrk(cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, k, k + j,
                        &negone, A + j + k, n, &one, A + j + k + (j + k) * n, n);

            if (n - j - 2 * k <= 0) {
                continue;
            }
            cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                        n - j - 2 * k,                  // m
                        k,                              // n
                        j + k,                          // k
                        &negone,                        // alpha
                        A + j + 2 * k,                  // A
                        n,                              // lda
                        A + j + k,                      // B
                        n,                              // ldb
                        &one,                           // beta
                        A + (j + 2 * k) + (j + k) * n,  // C
                        n);                             // ldc
        }
    }

    float *L;
    cudaMalloc(&L, sizeof(float) * n * n);
    cudaMemcpy(L, A, sizeof(float) * n * n, cudaMemcpyDeviceToDevice);

    clearTri<<<gridDim, blockDim>>>('u', n, n, L, n);
    // printMatrixDeviceBlock<float>("1LL.csv", n, n,  L , n);

    float a = 1.0f;
    float b = -1.0f;
    cublasStatus_t status =
        cublasSsyrk(cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, n, &a, L,
                    n, &b, A_reconstructed, n);
    // printMatrixDeviceBlock<float>("A_reconstructed.csv", n, n,  A_reconstructed ,
    // n);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS Ssyrk failed with status %d\n", status);
        return 1;
    }

    float diff_norm = computeFrobeniusNorm(n, A_reconstructed);
    printf("diff_norm: %e\n", diff_norm);
    printf("orig_norm: %e\n", orig_norm);
    float backward_error = diff_norm / orig_norm;
    printf("Backward error: %e\n", backward_error);
    float ms = stopTimer();

    printf("Cholesky factorization takes %f ms: %f TFLOPs\n", ms,
           1.0 / 3.0 * n * n * n / ms / 1e9);

    cudaFree(A);
    cudaFree(work);
    cudaFree(devInfo);
    // cudaFree(A_reconstructed);
    cublasDestroy(cublasHandle);
    cusolverDnDestroy(cusolverHandle);

    return 0;
}