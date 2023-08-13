 __global__ void calculate_forces(void *devX, void *devA)
 {   
    extern __shared__ float4[] shPosition;   
    float4 *globalX = (float4 *)devX;   
    float4 *globalA = (float4 *)devA;

    float4 myPosition;   
    
    int i, tile;   
    
    float3 acc = {0.0f, 0.0f, 0.0f};   
    
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;   
    myPosition = globalX[gtid];   
    
    for (i = 0, tile = 0; i < N; i += p, tile++) 
    {     
        int idx = tile * blockDim.x + threadIdx.x;
        shPosition[threadIdx.x] = globalX[idx];
        
        __syncthreads();     
        
        acc = tile_calculation(myPosition, acc);     
        
        __syncthreads();       
    }   
    
    // Save the result in global memory for the integration step.
    float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};   
    globalA[gtid] = acc4; 
} 