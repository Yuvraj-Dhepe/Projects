#include <stdio.h>

__global__ void hello() {
	printf("Hello from GPU!\n");
}

int main(){
	printf("Hello, world from CPU!\n");

	hello<<<1,1>>>();
	cudaDeviceSynchronize();
	return(0);
}
