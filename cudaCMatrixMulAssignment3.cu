#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLOCK_SIZE 16
#define TILE_SIZE 16

__global__ void gpu_matrix_mult_tiling(int *a, int *b, int *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    __shared__ int tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ int tile_b[TILE_SIZE][TILE_SIZE];

    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; tile++)
    {

        int tile_row = threadIdx.y;
        int tile_col = threadIdx.x;

        int global_row = row;
        int global_col = tile * TILE_SIZE + tile_col;

        if (global_row < m && global_col < n)
            tile_a[tile_row][tile_col] = a[global_row * n + global_col];
        else
            tile_a[tile_row][tile_col] = 0;

        global_row = tile * TILE_SIZE + tile_row;
        global_col = col;
        if (global_row < n && global_col < k)
            tile_b[tile_row][tile_col] = b[global_row * k + global_col];
        else
            tile_b[tile_row][tile_col] = 0;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++)
            sum += tile_a[tile_row][i] * tile_b[i][tile_col];

        __syncthreads();
    }

    if (row < m && col < k)
        c[row * k + col] = sum;
}


__global__ void gpu_matrix_mult_normal(int *a,int *b, int *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}  


__global__ void gpu_matrix_transpose(int* mat_in, int* mat_out, unsigned int rows, unsigned int cols) 
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) 
    {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}

void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < k; ++j) 
        {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h) 
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}


int main(int argc, char const *argv[])
{
    int m, n, k;

    printf("Enter values of m, n, and k: ");
    scanf("%d %d %d", &m, &n, &k);//If the scanning does not work input values here in the main
    
    srand(3333);
   
    int *h_a, *h_b, *h_c, *h_cc;
    cudaMallocHost((void **) &h_a, sizeof(int)*m*n);
    cudaMallocHost((void **) &h_b, sizeof(int)*n*k);
    cudaMallocHost((void **) &h_c, sizeof(int)*m*k);
    cudaMallocHost((void **) &h_cc, sizeof(int)*m*k);

    // random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = rand() % 1024;
        }
    }

    // random initialize matrix B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            h_b[i * k + j] = rand() % 1024;
        }
    }

    float gpu1_elapsed_time_ms, cpu_elapsed_time_ms, gpu2_elapsed_time_ms;

    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    
    cudaEventRecord(start, 0);
    

    int *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(int)*m*n);
    cudaMalloc((void **) &d_b, sizeof(int)*n*k);
    cudaMalloc((void **) &d_c, sizeof(int)*m*k);

    
    cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
   
    // normal version 
    
    gpu_matrix_mult_normal<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);    
    
    cudaMemcpy(h_c, d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpu1_elapsed_time_ms, start, stop);
    printf("Time elapsed on normal matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", m, n, n, k, gpu1_elapsed_time_ms);


    cudaEventRecord(start, 0);

    //tiling version
    gpu_matrix_mult_tiling<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);    
    
    cudaMemcpy(h_c, d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpu2_elapsed_time_ms, start, stop);
    printf("Time elapsed on tiling matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", m, n, n, k, gpu2_elapsed_time_ms);
 
    // start the CPU version
    cudaEventRecord(start, 0);

    cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on CPU: %f ms.\n\n", m, n, n, k, cpu_elapsed_time_ms);

    printf("Speedup (CPU/GPU(normal)) = %f\n", cpu_elapsed_time_ms / gpu1_elapsed_time_ms);
 
    printf("Speedup (CPU/GPU(tiling)) = %f\n", cpu_elapsed_time_ms / gpu2_elapsed_time_ms);
 
    printf("Speedup (GPU(normal)/GPU(tiling)) = %f\n", gpu1_elapsed_time_ms / gpu2_elapsed_time_ms);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    return 0;
}