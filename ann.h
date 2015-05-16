#ifndef ANN_H
#define ANN_H

#include <ode/ode.h>

void printMat(int n, int m, int *A);
void printMat(int n, int m, bool *A);
void printMat(int n, int m, double *A);
void updateANNLeg(dReal angles[3], dReal force, double f_low, double a_low[3], double a_high[3], double c[3],
	int *StoH, int *HtoA, bool sensor_node[4], bool hidden_node[4], double actuator_node[3]);

#endif