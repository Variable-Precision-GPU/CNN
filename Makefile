CUDA_PATH ?= /usr/local/cuda
SRC = *.cu

all::cnn_cuda cnn_gpgpu

cnn_cuda: $(SRC)
	nvcc -o $@ $^ -lcuda -lcublas -gencode arch=compute_60,code=compute_60

cnn_gpgpu: $(SRC)
	nvcc -o $@ $^ -L$(CUDA_PATH)/lib64 -lcudart -lcublas_static -lculibos -gencode arch=compute_60,code=compute_60

clean:
	rm cnn_cuda cnn_gpgpu
