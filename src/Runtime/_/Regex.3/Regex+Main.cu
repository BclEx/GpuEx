#include "Regex.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <sys/time.h>

//#define debug_simv // uncomment this line for debug information on simulation vectors
#define BLOCK_SZ 32 // should be kept equal to warp size for maximum performance
#define INPUT_BUF_SZ 1024 // size of the input string buffer

// wires up continuation pointers and assigns indexes
void wire_continuations_and_assign_indexes(regex* hexp, int* offset_vector)
{
	switch (hexp->kind)
	{
	case ALT: {
		hexp->idx = offset_vector[ALT]++;
		hexp->e_one->k = hexp->k;
		hexp->e_two->k = hexp->k;
		wire_continuations_and_assign_indexes(hexp->e_one, offset_vector);
		wire_continuations_and_assign_indexes(hexp->e_two, offset_vector);
		break; }
	case CON: {
		hexp->idx = offset_vector[CON]++;
		hexp->e_one->k = hexp->e_two;
		hexp->e_two->k = hexp->k;
		wire_continuations_and_assign_indexes(hexp->e_one, offset_vector);
		wire_continuations_and_assign_indexes(hexp->e_two, offset_vector);
		break; }
	case KLN: {
		hexp->idx = offset_vector[KLN]++;
		hexp->e_one->k = hexp;
		wire_continuations_and_assign_indexes(hexp->e_one, offset_vector);
		break; }
	case LIT: {
		hexp->idx = offset_vector[LIT]++;
		break; }
	case EPS: {
		hexp->idx = offset_vector[EPS]++;
		break; }
	default: { } // Nothing needs to be done.
	}
}

// flattens the given host expression into the target device expression vector
void vectorize_host_expression(regex* hexp, dregex* dexpv)
{
	dregex *dexp = &dexpv[hexp->idx];
	dexp->kind = hexp->kind;
	dexp->k = (hexp->k == NULL ? -1 : hexp->k->idx);
	switch (hexp->kind)
	{
	case ALT:
	case CON: {
		dexp->e_one = hexp->e_one->idx;
		dexp->e_two = hexp->e_two->idx;
		vectorize_host_expression(hexp->e_one, dexpv);
		vectorize_host_expression(hexp->e_two, dexpv);
		break; }
	case KLN: {
		dexp->e_one = hexp->e_one->idx;
		vectorize_host_expression(hexp->e_one, dexpv);
		break; }
	case LIT: {
		dexp->c = hexp->c;
		break; }
	default: { } // Nothing needs to be done.
	}
}

// schedules the given thread for execution for current turn if that is possible
__device__ void schedule_c(long* _simv_c, int idx, int turn)
{
	long val = _simv_c[idx];
	if (val != turn && val != -turn)
		_simv_c[idx] = turn;
}

// simulation kernel
__global__ void simulate(dregex *_dexpv, long *_simv_c, long *_simv_n, int count, char *_w, int *_statv, long turn)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < count && _simv_c[i] == turn)
	{
		dregex dexp = _dexpv[i];
		switch (dexp.kind)
		{
		case ALT:
			schedule_c(_simv_c, dexp.e_one, turn);
			schedule_c(_simv_c, dexp.e_two, turn);
			_statv[0] = 0;
			break;
		case CON: {
			schedule_c(_simv_c, dexp.e_one, turn);
			_statv[0] = 0;
			break; }
		case KLN: {
			schedule_c(_simv_c, dexp.e_one, turn);
			if (dexp.k != -1) {
				schedule_c(_simv_c, dexp.k, turn);
			} else if (*_w == NULL) {
				_statv[2] = 1;
			}
			_statv[0] = 0;
			break;
				  }
		case LIT: {
			if (*_w == dexp.c) {
				if (dexp.k != -1) {
					_simv_n[dexp.k] = turn + 1;
					_statv[1] = 1;
				} else if (*(_w + 1) == NULL) { // match!
					_statv[2] = 1;
				}
			}
			break;
				  }
		case EPS: {
			if (dexp.k != -1) {
				schedule_c(_simv_c, dexp.k, turn);
				_statv[0] = 0;
			} else if (*_w == NULL) {
				_statv[2] = 1;
			}
			break;
				  }
		default: {
			// should not be reached.
				 }
		}
		_simv_c[i] = -turn;
	}
}

/* isolates the matching functionality */
void match(regex* hexp, int simv_sz, char* w, int w_sz, bool* is_success, long* execution_time) {
	/* host varibles */
	dregex* dexpv; // device regular expression vector
	long* simv_c; // simulation vector c
	long* simv_n; // simulation vector n
	int* statv; // status vector
	int i;

	/* device variables */
	dregex* _dexpv;
	char* _w_stat; // points to the begining of the input string in device memory
	char* _w; // points to the current character of the input string in device memory
	long* _simv_c;
	long* _simv_n;
	int* _statv;

	dexpv = (dregex*) malloc(simv_sz * sizeof(dregex));
	vectorize_host_expression(hexp, dexpv);

	simv_c = (long*) malloc(simv_sz * sizeof(long));
	simv_n = (long*) malloc(simv_sz * sizeof(long));
	for (i = 0; i < simv_sz; i++) {
		simv_c[i] = (i == 0 ? 1 : 0); // only the first pointer is active on c list
		simv_n[i] = 0;
	}

	cudaMallocHost((void**) &statv, 3 * sizeof(int)); // page-locked (pinned) memory
	statv[0] = 0; // shouldSwap
	statv[1] = 0; // canSwap
	statv[2] = 0; // isMatch

	/* initialize device data */
	cudaMalloc((void**) &_dexpv, simv_sz * sizeof(dregex));
	cudaMemcpy((void*) _dexpv, (void*) dexpv, simv_sz * sizeof(dregex), cudaMemcpyHostToDevice);

	cudaMalloc((void**) &_w_stat, (w_sz + 1) * sizeof(char)); // need to allocate space for NULL character as well
	cudaMemcpy((void*) _w_stat, (void*) w, (w_sz + 1) * sizeof(char), cudaMemcpyHostToDevice);
	_w = _w_stat; // Initially _w points to the same location as _w_stat

	cudaMalloc((void**) &_simv_c, simv_sz * sizeof(long));
	cudaMemcpy((void*) _simv_c, (void*) simv_c, simv_sz * sizeof(long), cudaMemcpyHostToDevice);

	cudaMalloc((void**) &_simv_n, simv_sz * sizeof(long));
	cudaMemcpy((void*) _simv_n, (void*) simv_n, simv_sz * sizeof(long), cudaMemcpyHostToDevice);

	cudaMalloc((void**) &_statv, 3 * sizeof(int));
	cudaMemcpy((void*) _statv, (void*) statv, 3 * sizeof(int), cudaMemcpyHostToDevice);

#ifdef debug_simv
	long* simv_c_debug;
	cudaMallocHost((void**) &simv_c_debug, simv_sz * sizeof(long));
	long* simv_n_debug;
	cudaMallocHost((void**) &simv_n_debug, simv_sz * sizeof(long));
#endif

	cudaThreadSynchronize(); // wait for device

	int blocks_per_grid = (simv_sz + BLOCK_SZ - 1) / BLOCK_SZ;

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	bool shouldSwap = false;
	bool isMatch = false;
	bool isFail = false;
	int turn = 1;

	/* for timing */
	struct timeval start_time, stop_time;
	gettimeofday(&start_time, NULL);

	while (!isMatch && !isFail) {
		if (shouldSwap) {
			long* _temp = _simv_c;
			_simv_c = _simv_n;
			_simv_n = _temp;

			_w++; // consume one character
			turn++; // increment turn

			statv[1] = 0; // mark as cannot swap
		}

		statv[0] = 1; // mark for swap

		cudaMemcpyAsync((void*) _statv, (void*) statv, 3 * sizeof(int), cudaMemcpyHostToDevice, stream);

		simulate<<<blocks_per_grid, BLOCK_SZ, 0, stream>>>(_dexpv, _simv_c, _simv_n, simv_sz, _w, _statv, turn);

		cudaMemcpyAsync((void*) statv, (void*) _statv, 3 * sizeof(int), cudaMemcpyDeviceToHost, stream);

#ifdef debug_simv
		cudaMemcpyAsync((void*) simv_c_debug, (void*) _simv_c, simv_sz * sizeof(long), cudaMemcpyDeviceToHost, stream);
		cudaMemcpyAsync((void*) simv_n_debug, (void*) _simv_n, simv_sz * sizeof(long), cudaMemcpyDeviceToHost, stream);
#endif

		cudaStreamSynchronize(stream); // wait for the stream

#ifdef debug_simv
		printf("C - ");
		for (i = 0; i < simv_sz; i++) {
			if (simv_c_debug[i] != 0) {
				printf("[%d]", simv_c_debug[i]);
			}
		}
		printf("\n");
		printf("N - ");
		for (i = 0; i < simv_sz; i++) {
			if (simv_n_debug[i] != 0) {
				printf("[%d]", simv_n_debug[i]);
			}
		}
		printf("\n");
#endif

		shouldSwap = (statv[0] == 1);
		isMatch = (statv[2] == 1);
		isFail = (shouldSwap && (statv[1] == 0));
	}
	gettimeofday(&stop_time, NULL);
	cudaStreamDestroy(stream); // done

	*is_success = isMatch;
	*execution_time = stop_time.tv_sec * 1000 * 1000 + stop_time.tv_usec;
	*execution_time -= start_time.tv_sec * 1000 * 1000 + start_time.tv_usec;


	/* free device memory */
	cudaFree(_dexpv);
	cudaFree(_w_stat);
	cudaFree(_simv_c);
	cudaFree(_simv_n);
	cudaFree(_statv);

	/* free host memory */
	free(dexpv);
	free(simv_c);
	free(simv_n);
	cudaFreeHost((void*) statv); // pinned memory
#ifdef debug_simv
	cudaFreeHost((void*) simv_c_debug);
	cudaFreeHost((void*) simv_n_debug);
#endif
}

/* utility method for making test inputs for the (a*^na^n) against (a^n) test */
void make_inputs_test_1(int n, char** input_exp, char** input_string) {
	char* exp = (char*) malloc((3 * n * sizeof(char)) + 1);
	char* str = (char*) malloc((n * sizeof(char)) + 1);
	int i;
	for (i = 0; i < n; i++) {
		exp[i * 2] = 'a';
		exp[(i * 2) + 1] = '*';
		exp[(n * 3) - (i + 1)] = 'a';
		str[i] = 'a';
	}
	exp[3 * n] = '\0';
	str[n] = '\0';

	*input_exp = exp;
	*input_string = str;
}

/* utility method for making test inputs for the (a(*^n)) against (a^c) test */
void make_inputs_test_2(int n, char** input_exp) {
	char* exp = (char*) malloc(((3 * n) + 2) * sizeof(char));
	int i;
	for (i = 0; i < n; i++) {
		exp[i] = '(';
		exp[(n * 3) - (i * 2)] = '*';
		exp[(n * 3) - (i * 2 + 1)] = ')';
	}
	exp[n] = 'a';
	exp[(3 * n) + 1] = '\0';
	*input_exp = exp;
}

/* entry point */
int main(int argc, char** argv) {
	// instruct cuda to suspecd the CPU thread while waiting for kernels
	// cudaSetDeviceFlags(cudaDeviceBlockingSync);

	/*
	// test - 2
	int w_sz = 10;
	char* w = (char*) malloc((w_sz + 1) * sizeof(char));
	int i;
	for (i = 0; i < w_sz; i++) {
	w[i] = 'a';
	}
	w[w_sz] = '\0';

	for (i = 1; i <= 512; i++) {
	char* input_exp;
	make_inputs_test_2(i, &input_exp);

	regex* hexp;
	int* kind_stats;
	if (parse_regex(input_exp, &hexp, &kind_stats)) {
	int offset_vector[TOTAL_KINDS]; // offset for nodes of each kind within simulation vector
	int block_count = 0; // total blocks required
	int j;
	for (j = 0; j < TOTAL_KINDS; j++) {
	offset_vector[j] = block_count * BLOCK_SZ;
	block_count += (kind_stats[j] + BLOCK_SZ - 1) / BLOCK_SZ;
	}

	wire_continuations_and_assign_indexes(hexp, offset_vector); 
	int simv_sz = block_count * BLOCK_SZ;

	bool is_success;
	long execution_time;
	match(hexp, simv_sz, w, w_sz, &is_success, &execution_time);

	printf("%d, %.3f, %s\n", i, execution_time / 1000.0, (is_success ? "Match!" : "No Match!"));

	free(input_exp);
	free_hexp(hexp);
	free(kind_stats);
	}
	}
	free(w);
	*/

	/*
	// test - 1
	int i;
	for (i = 1; i <= 512; i++) {
	char* input_exp;
	char* w;

	make_inputs_test_1(i, &input_exp, &w);

	regex* hexp;
	int* kind_stats;
	if (parse_regex(input_exp, &hexp, &kind_stats)) {
	int offset_vector[TOTAL_KINDS]; // offset for nodes of each kind within simulation vector
	int block_count = 0; // total blocks required
	int j;
	for (j = 0; j < TOTAL_KINDS; j++) {
	offset_vector[j] = block_count * BLOCK_SZ;
	block_count += (kind_stats[j] + BLOCK_SZ - 1) / BLOCK_SZ;
	}

	wire_continuations_and_assign_indexes(hexp, offset_vector); 
	int simv_sz = block_count * BLOCK_SZ;
	int w_sz = strlen(w);

	bool is_success;
	long execution_time;
	match(hexp, simv_sz, w, w_sz, &is_success, &execution_time);

	printf("%d, %.3f, %s\n", i, execution_time / 1000.0, (is_success ? "Match!" : "No Match!"));

	free(input_exp);
	free_hexp(hexp);
	free(kind_stats);
	free(w);
	}
	}
	*/

	printf("Input expression: ");
	char* str_hexp = (char*) malloc(INPUT_BUF_SZ * sizeof(char));
	fgets(str_hexp, INPUT_BUF_SZ, stdin);
	str_hexp[strlen(str_hexp) - 1] = '\0';

	regex* hexp;
	int* kind_stats;
	if (parse_regex(str_hexp, &hexp, &kind_stats)) {
		printf("Input string: ");
		char* w = (char*) malloc(INPUT_BUF_SZ * sizeof(char));
		fgets(w, INPUT_BUF_SZ, stdin);
		w[strlen(w) - 1] = '\0';

		int offset_vector[TOTAL_KINDS]; // offset for nodes of each kind within simulation vector
		int block_count = 0; // total blocks required
		int i;
		for (i = 0; i < TOTAL_KINDS; i++) {
			offset_vector[i] = block_count * BLOCK_SZ;
			block_count += (kind_stats[i] + BLOCK_SZ - 1) / BLOCK_SZ;
		}

		// printf("Offsets: [%d][%d][%d][%d][%d]\n", offset_vector[0], offset_vector[1], offset_vector[2], offset_vector[3], offset_vector[4]);

		wire_continuations_and_assign_indexes(hexp, offset_vector); 
		int simv_sz = block_count * BLOCK_SZ;
		int w_sz = strlen(w);

		bool is_success;
		long execution_time;
		match(hexp, simv_sz, w, w_sz, &is_success, &execution_time);

		printf("Result: %s (%.3f ms)\n", (is_success ? "Match!" : "No Match!"), execution_time / 1000.0);

		free_hexp(hexp);
		free(kind_stats);
		free(w);
	} else {
		return -1;
	}

	free(str_hexp);

	return 0;
}
