#include <stdio.h>

void multiply(int n, int m, int o, bool A[n][m], int B[m][o], int C[n][o]) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < o; j++) {
			C[i][j] = 0;
			for (int k = 0; k < m; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

void printMat(int n, int m, int A[n][m]) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			printf("%4d", A[i][j]);
		}
		printf("\n");
	}
}

int main() {
	bool A[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
	int B[4][3] = {{1,2,3}, {5,6,7}, {-1,-2,-3}, {-5,-6,-7}};
	int C[4][3];
	multiply(4, 4, 3, A, B, C);
	printMat(4, 3, C);
}