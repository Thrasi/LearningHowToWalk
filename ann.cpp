#include <cstdio>
#include <cstdlib>
#include <ode/ode.h>
#include <cmath>


double sigmoid(int u, double c) {
	return 1 / (1 + exp(-u/c));
}

void multiply(int n, int m, int o, bool *A, int *B, int *C) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < o; j++) {
			C[i*o+j] = 0;
			for (int k = 0; k < m; k++) {
				C[i*o+j] += A[i*m+k] * B[k*o+j];
			}
		}
	}
}

void updateSensorNode(dReal angles[3], dReal force, double f_low, double a_low[3], double a_high[3], bool inp[4]) {
	for (int i = 0; i < 3; i++) {
		inp[i] = angles[i] >= a_low[i] && angles[i] <= a_high[i];
	}
	inp[3] = force >= f_low;
}

void updateHiddenNode(int hidden_in[4], bool hidden_out[4]) {
	for (int i = 0; i < 4; i++) {
		hidden_out[i] = hidden_in[i] > 0;
	}
}

void updateActuatorNode(int actuator_in[3], double c[3], double actuator_out[3]) {
	for (int i = 0; i < 3; i++) {
		actuator_out[i] = sigmoid(actuator_in[i], c[i]);
	}
}

void updateANNLeg(dReal angles[3], dReal force, double f_low, double a_low[3], double a_high[3], double c[3],
	int *StoH, int *HtoA, bool sensor_node[4], bool hidden_node[4], double actuator_node[3]) {
	
	updateSensorNode(angles, force, f_low, a_low, a_high, sensor_node);
	int *hidden_in = new int[4];
	multiply(1, 4, 4, sensor_node, StoH, hidden_in);
	updateHiddenNode(hidden_in, hidden_node);
	int *actuator_in = new int[3];
	multiply(1, 4, 3, hidden_node, HtoA, actuator_in);
	updateActuatorNode(actuator_in, c, actuator_node);
}


// int main() {
// 	dReal a[] = {2.95, 3.22, 9.87};
// 	dReal f = 6.66;
// 	bool in[4];
// 	double al[] = {1.9, 3.1, 11.9};
// 	double ah[] = {3.0, 3.2, 12.0};
// 	updateSensorNode(a, f, 5, al, ah, in);
// 	printMat(1, 4, in);

// 	int stoh[] = {0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0};
// 	int *inhi = (int*) malloc(sizeof(int)*1*4);
// 	multiply(1,4,4,in,stoh,inhi);
// 	printMat(1,4,inhi);

// 	bool *outhi = (bool*) malloc(sizeof(bool)*4);
// 	updateHiddenNode(inhi, outhi);
// 	printMat(1, 4, outhi);

// 	int htoa[] = {0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
// 	int *inac = (int*) malloc(sizeof(int)*3);
// 	multiply(1,4,3,outhi, htoa, inac);
// 	printMat(1,3,inac);

// 	double c[] = {2.53, 9.01, 7.2};
// 	double *outac = (double*) malloc(sizeof(double)*3);
// 	updateActuatorNode(inac, c, outac);
// 	printMat(1, 3, outac);
// }


void printMat(int n, int m, int *A) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			printf("%4d", A[i*m+j]);
		}
		printf("\n");
	}
}

void printMat(int n, int m, bool *A) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			printf("%4d", A[i*m+j]);
		}
		printf("\n");
	}
}

void printMat(int n, int m, double *A) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			printf("%8.4f", A[i*m+j]);
		}
		printf("\n");
	}
}