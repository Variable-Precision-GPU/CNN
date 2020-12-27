CUDA_PATH ?= /usr/local/cuda
SRC = *.cu

all::cnn_hardware cnn_sim

cnn_hardware: $(SRC)
	nvcc -o $@ $^ -lcuda -lcublas -gencode arch=compute_60,code=compute_60

cnn_sim: $(SRC)
	nvcc -o $@ $^ -L$(CUDA_PATH)/lib64 -lcudart -lcublas_static -lculibos -gencode arch=compute_60,code=compute_60

clean:
	rm cnn_hardware cnn_sim
