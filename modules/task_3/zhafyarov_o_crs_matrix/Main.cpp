  // Copyright 2020 Zhafyarov Oleg
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include "./crs_matrix.h"

TEST(Parallel_Operations_MPI, FirstTest_Matrix_CRS_Random_150x150) {
  int process_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  int Size = 3;
  int Nonzero_in_row = 1;
  int All_No_empty = Nonzero_in_row * Size;

  MatrixCRS A, B;
  MatrixCRS Result;
  MatrixCRS ToCompare;
  InitializeMatrix(Size, All_No_empty, &Result);
  InitializeMatrix(Size, All_No_empty, &A);
  InitializeMatrix(Size, All_No_empty, &B);
  InitializeMatrix(Size, All_No_empty, &ToCompare);
  if (process_rank == 0) {
    RandomMatrixCRS(Size, Nonzero_in_row, &A);
    RandomMatrixCRS(Size, Nonzero_in_row, &B);
    B = Transpose(Size, All_No_empty, Nonzero_in_row, &B);
  }
  ParallelMultiplication(process_rank, Size, &A, &B, &Result);
  Multiplication(&A, &B, &ToCompare);
  bool IsCompare = true;
  if (process_rank == 0) {
    for (int i = 0; i < All_No_empty; i++) {
      if (Result.Value[i] != ToCompare.Value[i]) {
        IsCompare = false;
        break;
      }
    }
    ASSERT_EQ(IsCompare, true);
  }
}

TEST(Parallel_Operations_MPI, SecondTest_Matrix_CRS_Random_500x500) {
  int process_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  int Size = 500;
  int Nonzero_in_row = 1;
  int All_No_empty = Nonzero_in_row * Size;

  MatrixCRS A, B;
  MatrixCRS Result;
  MatrixCRS ToCompare;
  InitializeMatrix(Size, All_No_empty, &Result);
  InitializeMatrix(Size, All_No_empty, &A);
  InitializeMatrix(Size, All_No_empty, &B);
  InitializeMatrix(Size, All_No_empty, &ToCompare);
  if (process_rank == 0) {
    RandomMatrixCRS(Size, Nonzero_in_row, &A);
    RandomMatrixCRS(Size, Nonzero_in_row, &B);
    B = Transpose(Size, All_No_empty, Nonzero_in_row, &B);
  }
  ParallelMultiplication(process_rank, Size, &A, &B, &Result);
  Multiplication(&A, &B, &ToCompare);
  bool IsCompare = true;
  if (process_rank == 0) {
    for (int i = 0; i < All_No_empty; i++) {
      if (Result.Value[i] != ToCompare.Value[i]) {
        IsCompare = false;
        break;
      }
    }
    ASSERT_EQ(IsCompare, true);
  }
}

TEST(Parallel_Operations_MPI, ThirdTest_Matrix_CRS_Random_750x750) {
  int process_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  int Size = 750;
  int Nonzero_in_row = 1;
  int All_No_empty = Nonzero_in_row * Size;

  MatrixCRS A, B;
  MatrixCRS Result;
  MatrixCRS ToCompare;
  InitializeMatrix(Size, All_No_empty, &Result);
  InitializeMatrix(Size, All_No_empty, &A);
  InitializeMatrix(Size, All_No_empty, &B);
  InitializeMatrix(Size, All_No_empty, &ToCompare);
  if (process_rank == 0) {
    RandomMatrixCRS(Size, Nonzero_in_row, &A);
    RandomMatrixCRS(Size, Nonzero_in_row, &B);
    B = Transpose(Size, All_No_empty, Nonzero_in_row, &B);
  }
  ParallelMultiplication(process_rank, Size, &A, &B, &Result);
  Multiplication(&A, &B, &ToCompare);
  bool IsCompare = true;
  if (process_rank == 0) {
    for (int i = 0; i < All_No_empty; i++) {
      if (Result.Value[i] != ToCompare.Value[i]) {
        IsCompare = false;
        break;
      }
    }
    ASSERT_EQ(IsCompare, true);
  }
}

TEST(Parallel_Operations_MPI, FourthTest_Matrix_CRS_Random_1500x1500) {
  int process_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  int Size = 750;
  int Nonzero_in_row = 1;
  int All_No_empty = Nonzero_in_row * Size;

  MatrixCRS A, B;
  MatrixCRS Result;
  MatrixCRS ToCompare;
  InitializeMatrix(Size, All_No_empty, &Result);
  InitializeMatrix(Size, All_No_empty, &A);
  InitializeMatrix(Size, All_No_empty, &B);
  InitializeMatrix(Size, All_No_empty, &ToCompare);
  if (process_rank == 0) {
    RandomMatrixCRS(Size, Nonzero_in_row, &A);
    RandomMatrixCRS(Size, Nonzero_in_row, &B);
    B = Transpose(Size, All_No_empty, Nonzero_in_row, &B);
  }
  ParallelMultiplication(process_rank, Size, &A, &B, &Result);
  Multiplication(&A, &B, &ToCompare);
  bool IsCompare = true;
  if (process_rank == 0) {
    for (int i = 0; i < All_No_empty; i++) {
      if (Result.Value[i] != ToCompare.Value[i]) {
        IsCompare = false;
        break;
      }
    }
    ASSERT_EQ(IsCompare, true);
  }
}

TEST(Parallel_Operations_MPI, FifthTest_Matrix_CRS_Random_3000x3000) {
  int process_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  int Size = 3000;
  int Nonzero_in_row = 1;
  int All_No_empty = Nonzero_in_row * Size;

  MatrixCRS A, B;
  MatrixCRS Result;
  MatrixCRS ToCompare;
  InitializeMatrix(Size, All_No_empty, &Result);
  InitializeMatrix(Size, All_No_empty, &A);
  InitializeMatrix(Size, All_No_empty, &B);
  InitializeMatrix(Size, All_No_empty, &ToCompare);
  if (process_rank == 0) {
    RandomMatrixCRS(Size, Nonzero_in_row, &A);
    RandomMatrixCRS(Size, Nonzero_in_row, &B);
    B = Transpose(Size, All_No_empty, Nonzero_in_row, &B);
  }
  ParallelMultiplication(process_rank, Size, &A, &B, &Result);
  Multiplication(&A, &B, &ToCompare);
  bool IsCompare = true;
  if (process_rank == 0) {
    for (int i = 0; i < All_No_empty; i++) {
      if (Result.Value[i] != ToCompare.Value[i]) {
        IsCompare = false;
        break;
      }
    }
    ASSERT_EQ(IsCompare, true);
  }
}

TEST(Parallel_Operations_MPI, SixthTest_efficiency_2000x2000) {
  int process_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  int Size = 2000;
  int Nonzero_in_row = 1;
  int All_No_empty = Nonzero_in_row * Size;
  double start_time, end_time, parallel, sequantional;
  MatrixCRS A, B;
  MatrixCRS Result;
  MatrixCRS ToCompare;
  InitializeMatrix(Size, All_No_empty, &Result);
  InitializeMatrix(Size, All_No_empty, &A);
  InitializeMatrix(Size, All_No_empty, &B);
  InitializeMatrix(Size, All_No_empty, &ToCompare);
  if (process_rank == 0) {
    RandomMatrixCRS(Size, Nonzero_in_row, &A);
    RandomMatrixCRS(Size, Nonzero_in_row, &B);
    B = Transpose(Size, All_No_empty, Nonzero_in_row, &B);
  }
  start_time = MPI_Wtime();
  ParallelMultiplication(process_rank, Size, &A, &B, &Result);
  end_time = MPI_Wtime();
  parallel = end_time - start_time;
  bool IsCompare = true;
  if (process_rank == 0) {
    start_time = MPI_Wtime();
    Multiplication(&A, &B, &ToCompare);
    end_time = MPI_Wtime();
    sequantional = end_time - start_time;
    for (int i = 0; i < All_No_empty; i++) {
      if (Result.Value[i] != ToCompare.Value[i]) {
        IsCompare = false;
        break;
      }
    }
    std::cout << "Parallel operation = " << parallel << std::endl;
    std::cout << "Sequantional operations = " << sequantional << std::endl;
    ASSERT_EQ(IsCompare, true);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();

  listeners.Release(listeners.default_result_printer());
  listeners.Release(listeners.default_xml_generator());

  listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
  return RUN_ALL_TESTS();
}


