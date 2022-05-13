#include <cstdint>
#include <mlir/ExecutionEngine/CRunnerUtils.h>
#include <stdio.h>
#include <vector>

extern "C" {
char *getTensorFilename(uint64_t id);
void *_mlir_ciface_read_coo(char *filename);
void _mlir_ciface_coords(StridedMemRefType<uint64_t, 1> *ref, void *coo,
                         uint64_t dim);
void _mlir_ciface_values(StridedMemRefType<double, 1> *ref, void *coo);
int64_t _mlir_ciface_nano_time();
}

template <typename T, typename V, typename A>
void MttkrpCoo(T &bCoords0, T &bCoords1, T &bCoords2, V &bVals, A &a, A &c, A &d,
            uint64_t nnz, uint64_t J) {
  for (uint64_t iK = 0; iK < nnz; iK++) {
    uint64_t i = bCoords0[iK];
    uint64_t k = bCoords1[iK];
    uint64_t l = bCoords2[iK];
    double bIKL = bVals[iK];
    for (uint64_t j = 0; j < J; j++) {
      a[i * J + j] += bIKL * d[l * J + j] * c[k * J + j];
    }
  }
}

int main() {
  char *filename = getTensorFilename(0);
  void *coo = _mlir_ciface_read_coo(filename);
  StridedMemRefType<uint64_t, 1> bCoord0;
  _mlir_ciface_coords(&bCoord0, coo, 0);
  StridedMemRefType<uint64_t, 1> bCoord1;
  _mlir_ciface_coords(&bCoord1, coo, 1);
  StridedMemRefType<uint64_t, 1> bCoord2;
  _mlir_ciface_coords(&bCoord2, coo, 2);
  StridedMemRefType<double, 1> bVals;
  _mlir_ciface_values(&bVals, coo);
  uint64_t I = 12092;
  uint64_t J = 5000;
  uint64_t K = 9184;
  uint64_t L = 28818;
  uint64_t nnz = 5879419;

  std::vector<double> c = std::vector<double>(K * J);
  for (uint64_t k = 0; k < K; k++) {
    for (uint64_t j = 0; j < J; j++) {
      c[k * J + j] = k * J + j;
    }
  }

  std::vector<double> d = std::vector<double>(L * J);
  for (uint64_t l = 0; l < L; l++) {
    for (uint64_t j = 0; j < J; j++) {
      d[l * J + j] = l * J + j;
    }
  }

  std::vector<double> a = std::vector<double>(I * J);
  std::fill(a.begin(), a.end(), 0.0);

  uint64_t tStartMttkrpCoo = _mlir_ciface_nano_time();
  MttkrpCoo(bCoord0, bCoord1, bCoord2, bVals, a, c, d, nnz, J);
  uint64_t tEndMttkrpCoo = _mlir_ciface_nano_time();
  printf("time: %lu\n", tEndMttkrpCoo - tStartMttkrpCoo);

//  printf("A ( ");
//  for (uint64_t i = 0; i < I; i++) {
//    printf("( ");
//    for (uint64_t j = 0; j < J; j++) {
//      printf(" %f ", a[i*J+j]);
//    }
//    printf(") ");
//  }
//  printf(")\n");
}