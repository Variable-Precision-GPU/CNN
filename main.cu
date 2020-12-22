#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "layer.h"

#include <cuda.h>
#include <cstdio>
#include <time.h>
#include <assert.h>

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Define layers of CNN
static Layer *l_input;
static Layer *l_c1;
static Layer *l_s1;
static Layer *l_f;

static void learn(int epochs);
static unsigned int classify(double data[28][28]);
static void test();
static double forward_pass(double data[28][28]);
static double back_pass();

static inline void loaddata()
{
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
						 &train_set, &train_cnt);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
						 &test_set, &test_cnt);
}

static void init_layers()
{
	l_input = new Layer(0, 0, 28 * 28);
	l_c1 = new Layer(5 * 5, 6, 24 * 24 * 6);
	l_s1 = new Layer(4 * 4, 1, 6 * 6 * 6);
	l_f = new Layer(6 * 6 * 6, 10, 10);
}

static void init_layers(const char *weights_file)
{
	FILE *file = fopen(weights_file, "r");
	l_input = new Layer(0, 0, 28 * 28, file);
	l_c1 = new Layer(5 * 5, 6, 24 * 24 * 6, file);
	l_s1 = new Layer(4 * 4, 1, 6 * 6 * 6, file);
	l_f = new Layer(6 * 6 * 6, 10, 10, file);
}

static void save_weights(const char *weights_file)
{
	std::ofstream file(weights_file);
	l_input->save(file);
	l_c1->save(file);
	l_s1->save(file);
	l_f->save(file);
}

static void destroy_layers()
{
	delete l_input;
	delete l_c1;
	delete l_s1;
	delete l_f;
}

int main(int argc, const char **argv)
{
	assert(argc > 1 && "Run with mode -both, -train, -train-increment or -test");
	srand(time(NULL));

	if (strcmp(argv[1], "-train") == 0)
	{
		assert(argc == 4 && "Please provide the number of epochs and weights file");
		loaddata();
		init_layers();
		learn(atoi(argv[2]));
		save_weights(argv[3]);
		destroy_layers();
	}
	else if (strcmp(argv[1], "-train-increment") == 0)
	{
		// TODO: Implement
		loaddata();
	}
	else if (strcmp(argv[1], "-test") == 0)
	{
		assert(argc == 3 && "Please provide the weights file");
		loaddata();
		init_layers(argv[2]);
		test();
		destroy_layers();
	}
	else if (strcmp(argv[1], "-both") == 0)
	{
		loaddata();
		init_layers();
		learn(50);
		test();
		destroy_layers();
	}
	else
	{
		assert(0 && "Run with mode -both, -train or -test");
	}

	// [afterdusk] remove driver API initialization (redundant?)
	// CUresult err = cuInit(0);
	// if (err != CUDA_SUCCESS)
	// {
	// 	fprintf(stderr, "CUDA initialisation failed with error code - %d\n", err);
	// 	return 1;
	// }

	return 0;
}

// Forward propagation of a single row in dataset
static double forward_pass(double data[28][28])
{
	float input[28][28];

	for (int i = 0; i < 28; ++i)
	{
		for (int j = 0; j < 28; ++j)
		{
			input[i][j] = data[i][j];
		}
	}

	l_input->clear();
	l_c1->clear();
	l_s1->clear();
	l_f->clear();

	clock_t start, end;
	start = clock();

	l_input->setOutput((float *)input);

	fp_preact_c1<<<64, 64>>>((float(*)[28])l_input->output, (float(*)[24][24])l_c1->preact, (float(*)[5][5])l_c1->weight);
	fp_bias_c1<<<64, 64>>>((float(*)[24][24])l_c1->preact, l_c1->bias);
	apply_step_function<<<64, 64>>>(l_c1->preact, l_c1->output, l_c1->O);

	fp_preact_s1<<<64, 64>>>((float(*)[24][24])l_c1->output, (float(*)[6][6])l_s1->preact, (float(*)[4][4])l_s1->weight);
	fp_bias_s1<<<64, 64>>>((float(*)[6][6])l_s1->preact, l_s1->bias);
	apply_step_function<<<64, 64>>>(l_s1->preact, l_s1->output, l_s1->O);

	fp_preact_f<<<64, 64>>>((float(*)[6][6])l_s1->output, l_f->preact, (float(*)[6][6][6])l_f->weight);
	fp_bias_f<<<64, 64>>>(l_f->preact, l_f->bias);
	apply_step_function<<<64, 64>>>(l_f->preact, l_f->output, l_f->O);

	end = clock();
	return ((double)(end - start)) / CLOCKS_PER_SEC;
}

// Back propagation to update weights
static double back_pass()
{
	clock_t start, end;

	start = clock();

	bp_weight_f<<<64, 64>>>((float(*)[6][6][6])l_f->d_weight, l_f->d_preact, (float(*)[6][6])l_s1->output);
	bp_bias_f<<<64, 64>>>(l_f->bias, l_f->d_preact);

	bp_output_s1<<<64, 64>>>((float(*)[6][6])l_s1->d_output, (float(*)[6][6][6])l_f->weight, l_f->d_preact);
	bp_preact_s1<<<64, 64>>>((float(*)[6][6])l_s1->d_preact, (float(*)[6][6])l_s1->d_output, (float(*)[6][6])l_s1->preact);
	bp_weight_s1<<<64, 64>>>((float(*)[4][4])l_s1->d_weight, (float(*)[6][6])l_s1->d_preact, (float(*)[24][24])l_c1->output);
	bp_bias_s1<<<64, 64>>>(l_s1->bias, (float(*)[6][6])l_s1->d_preact);

	bp_output_c1<<<64, 64>>>((float(*)[24][24])l_c1->d_output, (float(*)[4][4])l_s1->weight, (float(*)[6][6])l_s1->d_preact);
	bp_preact_c1<<<64, 64>>>((float(*)[24][24])l_c1->d_preact, (float(*)[24][24])l_c1->d_output, (float(*)[24][24])l_c1->preact);
	bp_weight_c1<<<64, 64>>>((float(*)[5][5])l_c1->d_weight, (float(*)[24][24])l_c1->d_preact, (float(*)[28])l_input->output);
	bp_bias_c1<<<64, 64>>>(l_c1->bias, (float(*)[24][24])l_c1->d_preact);

	apply_grad<<<64, 64>>>(l_f->weight, l_f->d_weight, l_f->M * l_f->N);
	apply_grad<<<64, 64>>>(l_s1->weight, l_s1->d_weight, l_s1->M * l_s1->N);
	apply_grad<<<64, 64>>>(l_c1->weight, l_c1->d_weight, l_c1->M * l_c1->N);

	end = clock();
	return ((double)(end - start)) / CLOCKS_PER_SEC;
}

// Unfold the input layer
static void unfold_input(double input[28][28], double unfolded[24 * 24][5 * 5])
{
	int a = 0;
	(void)unfold_input;

	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j)
		{
			int b = 0;
			for (int x = i; x < i + 2; ++x)
				for (int y = j; y < j + 2; ++y)
					unfolded[a][b++] = input[x][y];
			a++;
		}
}

static void learn(int epochs)
{
	static cublasHandle_t blas;
	cublasCreate(&blas);

	float err;

	double time_taken = 0.0;

	fprintf(stdout, "Learning\n");

	for (int iter = 1; iter <= epochs; iter++)
	{
		err = 0.0f;

		for (int i = 0; i < train_cnt; ++i)
		{
			float tmp_err;

			time_taken += forward_pass(train_set[i].data);

			l_f->bp_clear();
			l_s1->bp_clear();
			l_c1->bp_clear();

			// Euclid distance of train_set[i]
			makeError<<<10, 1>>>(l_f->d_preact, l_f->output, train_set[i].label, 10);
			cublasSnrm2(blas, 10, l_f->d_preact, 1, &tmp_err);
			err += tmp_err;

			time_taken += back_pass();
		}

		err /= train_cnt;
		fprintf(stdout, "epoch: %d, error: %e, time_on_gpu: %lf\n", iter, err, time_taken);

		if (err < threshold)
		{
			fprintf(stdout, "Training complete, error less than threshold\n\n");
			break;
		}
	}

	fprintf(stdout, "\n Time - %lf\n", time_taken);
}

// Returns label of given data (0-9)
static unsigned int classify(double data[28][28])
{
	float res[10];

	forward_pass(data);

	unsigned int max = 0;

	cudaMemcpy(res, l_f->output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	for (int i = 1; i < 10; ++i)
	{
		if (res[max] < res[i])
		{
			max = i;
		}
	}

	return max;
}

// Perform forward propagation of test data
static void test()
{
	int error = 0;

	for (int i = 0; i < test_cnt; ++i)
	{
		if (classify(test_set[i].data) != test_set[i].label)
		{
			++error;
		}
	}

	fprintf(stdout, "Test Accuracy: %.2lf%%\n",
					double(test_cnt - error) / double(test_cnt) * 100.0);
}
