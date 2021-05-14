//                         PyOnlineSVR
//               Copyright 2021 - Sebastian Schmidl
//
// This file is part of PyOnlineSVR.
//
// PyOnlineSVR is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// PyOnlineSVR is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with PyOnlineSVR. If not, see
// <https://www.gnu.org/licenses/gpl-3.0.html>.

%module(package="pyonlinesvr.lib") onlinesvr

%{
#define SWIG_FILE_WITH_INIT
#include "Vector.h"
#include "Matrix.h"
#include "OnlineSVR.h"
%}


class OnlineSVR
{

public:

    // Constants
    enum {
        OPERATION_LEARNING=200,
        OPERATION_UNLEARNING = 201,
        OPERATION_STABILIZING = 202,
        OPERATION_PREDICT = 203,
        OPERATION_MARGIN = 204,

        NO_SET = 10,
        SUPPORT_SET = 11,
        ERROR_SET = 12,
        REMAINING_SET = 13,

        KERNEL_LINEAR = 100,
        KERNEL_POLYNOMIAL = 101,
        KERNEL_RBF = 102,
        KERNEL_RBF_GAUSSIAN = 103,
        KERNEL_RBF_EXPONENTIAL = 104,
        KERNEL_MLP = 105,

        VERBOSITY_NO_MESSAGES = 0,
        VERBOSITY_NORMAL = 1,
        VERBOSITY_DETAILS = 2,
        VERBOSITY_DEBUG = 3
    };

    // Initialization
    OnlineSVR();
    ~OnlineSVR();
    void Clear();
    OnlineSVR* Clone();

    // Attributes Operations
    double GetC ();
    void SetC (double C);
    double GetEpsilon ();
    void SetEpsilon (double Epsilon);
    int GetKernelType ();
    void SetKernelType (int);
    double GetKernelParam (); // gamma
    void SetKernelParam (double KernelParam);
    double GetKernelParam2 (); // coef0
    void SetKernelParam2 (double KernelParam);
    double GetKernelParam3 (); // degree
    void SetKernelParam3 (double KernelParam);
    bool GetAutoErrorTollerance ();
    void SetAutoErrorTollerance (bool AutoErrorTollerance);
    double GetErrorTollerance ();
    void SetErrorTollerance (double ErrorTollerance);
    int GetVerbosity ();
    void SetVerbosity (int Verbosity);
    bool GetStabilizedLearning ();
    void SetStabilizedLearning (bool StabilizedLearning);
    bool GetSaveKernelMatrix ();
    void SetSaveKernelMatrix (bool SaveKernelMatrix);
    int GetSamplesTrainedNumber ();
    int GetSupportSetElementsNumber ();
    int GetErrorSetElementsNumber ();
    int GetRemainingSetElementsNumber ();
    Vector<int>* GetSupportSetIndexes();
    Vector<int>* GetErrorSetIndexes();
    Vector<int>* GetRemainingSetIndexes();
	Matrix<double>* GetSupportVectors();
	float GetBias();

    // Learning Operations
    int Train (Matrix<double>* X, Vector<double>* Y);
    int Train (Matrix<double>* X, Vector<double>* Y, Matrix<double>* TestSetX, Vector<double>* TestSetY);
    int Train (Matrix<double>* X, Vector<double>* Y, int TrainingSize, int TestSize);
    int Train (double**X, double *Y, int ElementsNumber, int ElementsSize);
    int Train (Vector<double>* X, double Y);
    int Forget (Vector<int>* Indexes);
    int Forget (int* Indexes, int N);
    int Forget (int Index);
    int Forget (Vector<double>* Sample);
    int Stabilize ();
    // void SelfLearning (Matrix<double>* TrainingSetX, Vector<double>* TrainingSetY, Matrix<double>* ValidationSetX, Vector<double>* ValidationSetY, double ErrorTollerance);
    static void CrossValidation (Matrix<double>* TrainingSetX, Vector<double>* TrainingSetY, Vector<double>* EpsilonList, Vector<double>* CList, Vector<double>* KernelParamList, int SetNumber, char* ResultsFileName);
    static double CrossValidation (Vector<Matrix<double>*>* SetX, Vector<Vector<double>*>* SetY, double Epsilon, double C, double KernelParam);
    static void LeaveOneOut (Matrix<double>* TrainingSetX, Vector<double>* TrainingSetY, Vector<double>* EpsilonList, Vector<double>* CList, Vector<double>* KernelParamList, char* ResultsFileName);
    static double LeaveOneOut (Matrix<double>* SetX, Vector<double>* SetY, double Epsilon, double C, double KernelParam);

    // Predict/Margin Operations
    double Predict (Vector<double>* X);
    double Predict (double* X, int ElementsSize);
    Vector<double>* Predict (Matrix<double>* X);
    double* Predict (double** X, int ElementsNumber, int ElementsSize);
    double Margin (Vector<double>* X, double Y);
    double Margin (double* X, double Y, int ElementsSize);
    Vector<double>* Margin (Matrix<double>* X, Vector<double>* Y);
    double* Margin (double** X, double* Y, int ElementsNumber, int ElementsSize);

    // Control Operations
    bool VerifyKKTConditions ();
    // void FindError(Matrix<double>* ValidationSetX, Vector<double>* ValidationSetY, double* MinError, double* MeanError, double* MaxError);

    // I/O Operations
    void ShowInfo ();
    void ShowDetails ();
    void LoadOnlineSVR(char* Filename);
    void SaveOnlineSVR(char* Filename);
    static void Import(char* Filename, Matrix<double>** X, Vector<double>** Y);
    static void Import(char* Filename, Matrix<double>** AngularPositions, Matrix<double>** MotorCurrents, Matrix<double>** AppliedVoltages);

};

template<class T>
class Vector
{
public:
	// Attributes
	T* Values;

	// Initialization
	Vector ();
	Vector (T* X, int N);
	Vector (int Length);
	~Vector ();
	Vector<T>* Clone();
	int GetLength ();
	int GetStepSize ();
	void SetStepSize (int X);
	T GetValue (int Index);
	void SetValue (int Index, T Value);
	bool Contains (T Value);

	// Add/Remove Operations
	void Clear ();
	void Add (T X);
	void AddAt (T X, int Index);
	void RemoveAt (int Index);
	Vector<T>* Extract (int FromIndex, int ToIndex);

	// Pre-built Vectors
	static Vector<double>* ZeroVector (int Length);
	static Vector<double>* RandVector (int Length);
	static Vector<T>* GetSequence(T Start, T Step, T End);

	// Mathematical Operations
	void SumScalar (T X);
	void ProductScalar (T X);
	void DivideScalar (T X);
	void PowScalar (T X);
	void SumVector (Vector<T>* V);
	static Vector<T>* SumVector (Vector<T>* V1, Vector<T>* V2);
	void SubtractVector (Vector<T>* V);
	static Vector<T>* SubtractVector (Vector<T>* V1, Vector<T>* V2);
	void ProductVector (Vector<T>* V);
	static Vector<T>* ProductVector (Vector<T>* V1, Vector<T>* V2);
	T ProductVectorScalar (Vector<T>* V);
	static T ProductVectorScalar (Vector<T>* V1, Vector<T>* V2);
	T Sum();
	T AbsSum();

	// Comparison Operations
	T Min();
	void Min(T* MinValue, int*MinIndex);
	T MinAbs();
	void MinAbs(T* MinValue, int*MinIndex);
	T Max();
	void Max(T* MaxValue, int*MaxIndex);
	T MaxAbs();
	void MaxAbs(T* MaxValue, int*MaxIndex);
	T Mean();
	T MeanAbs();
	T Variance();

	// Sorting Operations
	void Sort();
	void RemoveDuplicates();
	int Find(T X);

	// I/O Operations
	static Vector<T>* Load(char* Filename);
	void Save (char* Filename);
	void Print ();
	void Print (char* VectorName);

	// Operators Redefinition
	T operator [] (int Index);
};

%template(DoubleVector) Vector<double>;
%template(IntVector) Vector<int>;

template<class T>
class Matrix
{

public:
	// Attributes
	Vector<Vector<T>*>* Values;

	// Initialization
	Matrix ();
	Matrix (T** X, int Rows, int Cols);
	~Matrix ();
	Matrix<T>* Clone ();
	int GetLengthRows ();
	int GetLengthCols ();

	// Selection Operations
	Vector<T>* GetRowRef (int Index);
	Vector<T>* GetRowCopy (int Index);
	Vector<T>* GetColCopy (int Index);
	T GetValue (int RowIndex, int ColIndex);
	void SetValue (int RowIndex, int ColIndex, T Value);
	int IndexOf (Vector<T> *V);

	// Add/Remove Operations
	void Clear ();
	void AddRowRef (Vector<T>* V);
	void AddRowCopy (Vector<T>* V);
	void AddRowCopy (T* V, int N);
	void AddRowRefAt (Vector<T>* V, int Index);
	void AddRowCopyAt (Vector<T>* V, int Index);
	void AddRowCopyAt (T* V, int N, int Index);
	void AddColCopy (Vector<T>* V);
	void AddColCopy (T* V, int N);
	void AddColCopyAt (Vector<T>* V, int Index);
	void AddColCopyAt (T* V, int N, int Index);
	void RemoveRow (int Index);
	void RemoveCol (int Index);
	Matrix<T>* ExtractRows (int FromRowIndex, int ToRowIndex);
	Matrix<T>* ExtractCols (int FromColIndex, int ToColIndex);

	// Pre-built Matrix
	static Matrix<double>* ZeroMatrix (int RowsNumber, int ColsNumber);
	static Matrix<double>* RandMatrix (int RowsNumber, int ColsNumber);

	// Mathematical Operations
	void SumScalar (T X);
	void ProductScalar (T X);
	void DivideScalar (T X);
	void PowScalar (T X);
	void SumMatrix (Matrix<T>* M);
	void SubtractMatrix (Matrix<T>* M);
	Vector<T>* ProductVector (Vector<T>* V);
	static Vector<T>* ProductVector (Matrix* M, Vector<T>* V);
	static Matrix<T>* ProductVectorVector (Vector<T>* V1, Vector<T>* V2);
	static Matrix<T>* ProductMatrixMatrix (Matrix<T>* M1, Matrix<T>* M2);

	// I/O Operations
	static Matrix<double>* Load(char* Filename);
	void Save (char* Filename);
	void Print ();
	void Print (char* MatrixName);

	// Operators Redefinition
	Vector<T> operator [] (int Index);

};

%template(DoubleMatrix) Matrix<double>;
%template(IntMatrix) Matrix<int>;
