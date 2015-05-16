#include <cmath>
#include <algorithm>
#include <ctime>
#include <cstdio>

int *indices;
int N;

void initializeIndicesArray(int n) {
	indices = new int[n];
	for (int i = 0; i < n; i++) {
		indices[i] = i;
	}
	N = n;
}

int select(double *result, int n, int k) {
	std::random_shuffle(indices, indices+N);
	double bestValue = 0;
	int bestIdx = 0;
	for (int i = 0; i < k; i++) {
		if (result[indices[i]] > bestValue) {
			bestValue = result[indices[i]];
			bestIdx = indices[i];
		}
	}
	return bestIdx;
}

void cross(bool* a, bool* b, bool* c, bool* d, int n) {
	// int cut = rand() % n;
	// for (int i = 0; i < cut; i++) {
	// 	c[i] = a[i];
	// 	d[i] = b[i];
	// }
	// for (int i = cut; i < n; i++) {
	// 	c[i] = b[i];
	// 	d[i] = a[i];
	// }
	for (int i = 0; i < n; i++) {
		double p = ((double) rand()) / RAND_MAX;
		if (p < 0.5) {
			c[i] = a[i];
			d[i] = b[i];
		} else {
			c[i] = b[i];
			d[i] = a[i];
		}
	}
}

void mutate(bool *a, int n, double p) {
	for (int i = 0; i < n; i++) {
		if (((double) rand()) / RAND_MAX < p) {
			a[i] = !a[i];
		}
	}
}

void copy(bool* from, bool* to, int n) {
	for (int i = 0; i < n; i++) {
		to[i] = from[i];
	}
}

void createPopulation(double *result, int n, bool** ind, int m, int K, double p) {
	initializeIndicesArray(n);

	bool* newInd = new bool[m*n];
	double bestV = 0, sBestV = 0;
	int bestI = 0, sBestI = 0;
	for (int i = 0; i < n; i++) {
		if (result[i] > bestV) {
			sBestV = bestV;
			sBestI = bestI;
			bestV = result[i];
			bestI = i;
		} else if (result[i] > sBestV) {
			sBestV = result[i];
			sBestI = i;
		}
	}

	copy(ind[bestI], newInd, m);
	copy(ind[sBestI], newInd+m, m);
	for (int i = 2; i < n; i+=2) {
		int j = select(result, n, K);
		int k = select(result, n, K);
		cross(ind[i], ind[j], newInd+m*i, newInd+m*(i+1), m);
		mutate(newInd+m*i, m, p);
		mutate(newInd+m*(i+1), m, p);
	}

	for (int i = 0; i < n; i++) {
		copy(newInd+m*i, ind[i], m);
	}
}

void printGene(bool *g, int n) {
	for (int i = 0; i < n; i++) {
		printf("%2d", g[i]);
	}
	printf("\n");
}

// int main() {
// 	srand(time(NULL));
// 	bool g[] = {1, 1, 1, 0, 0, 1, 0, 0, 1, 1};
// 	bool g1[] = {1, 0, 1, 1, 0, 1, 0, 1, 1, 0};
// 	bool *a = new bool[10], *b = new bool[10];
// 	// printGene(g, 10);
// 	// mutate(g, 10, 0.5);
// 	// printGene(g, 10);
// 	// cross(g, g1, a, b, 10);
// 	// printGene(a, 10);
// 	// printGene(b, 10);
// 	return 0;
// }

