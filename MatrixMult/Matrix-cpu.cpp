#include <iostream>
#include <vector>
#include <chrono>

void matrixMultiply(const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b, std::vector<std::vector<int>>& c, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      c[i][j] = 0;
      for (int k = 0; k < N; k++) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}

int main() {
  int N = 1024;  // Matrix size
  std::vector<std::vector<int>> a(N, std::vector<int>(N));
  std::vector<std::vector<int>> b(N, std::vector<int>(N));
  std::vector<std::vector<int>> c(N, std::vector<int>(N));
  
  // Initialize matrices
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      a[i][j] = rand() % 100;
      b[i][j] = rand() % 100;
    }
  }
  
  auto start = std::chrono::high_resolution_clock::now();
  
  matrixMultiply(a, b, c, N);
  
  auto end = std::chrono::high_resolution_clock::now();
  
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "CPU Matrix multiplication time: " << elapsed.count() << " ms." << std::endl;
  
  return 0;
}