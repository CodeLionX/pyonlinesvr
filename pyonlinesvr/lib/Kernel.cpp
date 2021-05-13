/******************************************************************************
*                       ONLINE SUPPORT VECTOR REGRESSION                      *
*                      Copyright 2006 - Francesco Parrella                    *
*                      Copyright 2021 - Sebastian Schmidl                     *
*                                                                             *
*This program is distributed under the terms of the GNU General Public License*
*                                                                             *
*Changes:                                                                     *
* 2021-05-12 (Sebastian Schmidl): Document formatting, removed namespace      *
* 2021-05-13 (Sebastian Schmidl): Align kernels to sklearn's definition       *
*     (https://sklearn.org/modules/svm.html#kernel-functions)                 *
******************************************************************************/


#ifndef KERNEL_CPP
#define KERNEL_CPP

#include "OnlineSVR.h"
#include <math.h>


// Kernel Operations
double OnlineSVR::Kernel (Vector<double>* V1, Vector<double>* V2)
{
	double K;
	Vector<double>* V;

	switch (this->KernelType) {

		case KERNEL_LINEAR:
			// K = <V1,V2>
			return Vector<double>::ProductVectorScalar(V1,V2);

		case KERNEL_POLYNOMIAL:
			// K = (KernelParam * <V1,V2> + KernelParam2) ^ KernelParam3
			K = Vector<double>::ProductVectorScalar(V1,V2);
			return pow(this->KernelParam * K + this->KernelParam2, this->KernelParam3);

		case KERNEL_RBF:
			// K = exp (-KernelParam * sum(dist(V1,V2))^2)
			V = Vector<double>::SubtractVector(V1,V2);
			V->PowScalar(2);
			K = V->Sum();
			K *= -this->KernelParam;
			delete V;
			return exp(K);

		case KERNEL_RBF_GAUSSIAN:
			// K = exp (-sum(dist(V1,V2)^2 / 2*(KernelParam^2))
			V = Vector<double>::SubtractVector(V1,V2);
			V->PowScalar(2);
			K = V->Sum();
			if (this->KernelParam!=0)
				K /= -(2*pow(this->KernelParam,2));
			else
				K /= -2;
			delete V;
			return exp(K);

		case KERNEL_RBF_EXPONENTIAL:
			// K = exp (-sum(dist(V1,V2) / 2*(KernelParam^2))
			V = Vector<double>::SubtractVector(V1,V2);
			K = V->AbsSum();
			if (this->KernelParam!=0)
				K /= -(2*pow(this->KernelParam,2));
			else
				K /= -2;
			delete V;
			return exp(K);

		case KERNEL_MLP:
			// K = tanh(<V1,V2>*KernelParam + KernelParam2)
			K = Vector<double>::ProductVectorScalar(V1,V2);
			K = tanh(K*this->KernelParam + this->KernelParam2);
			return K;
		}

	return -1;
}

#endif
