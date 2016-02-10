#define BOOST_TEST_MODULE LinAlg_IncompleteCholeskyFactorization
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "IncompleteCholeskyFactorization.h"

#include <shark/LinAlg/rotations.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h> //the used kernel for the SVM


using namespace shark;


const size_t Dimensions=12;
double inputMatrix[Dimensions][Dimensions]=
{ 
    {1.000000 ,0.000000 ,0.000002 ,0.000000 ,0.000000 ,0.000016 ,0.000000 ,0.000042 ,0.000000 ,0.000120 ,0.000000 ,0.559193 },
{0.000000 ,1.000000 ,0.026631 ,0.748935 ,0.127787 ,0.138906 ,0.000035 ,0.000003 ,0.020925 ,0.000013 ,0.255593 ,0.000000 },
{0.000002 ,0.026631 ,1.000000 ,0.012394 ,0.564126 ,0.144666 ,0.000000 ,0.000000 ,0.000009 ,0.119522 ,0.001798 ,0.000001 },
{0.000000 ,0.748935 ,0.012394 ,1.000000 ,0.040296 ,0.238513 ,0.000268 ,0.000098 ,0.114654 ,0.000006 ,0.670119 ,0.000000 },
{0.000000 ,0.127787 ,0.564126 ,0.040296 ,1.000000 ,0.083419 ,0.000000 ,0.000000 ,0.000028 ,0.010363 ,0.004134 ,0.000000 },
{0.000016 ,0.138906 ,0.144666 ,0.238513 ,0.083419 ,1.000000 ,0.000001 ,0.001106 ,0.010605 ,0.001962 ,0.177203 ,0.000051 },
{0.000000 ,0.000035 ,0.000000 ,0.000268 ,0.000000 ,0.000001 ,1.000000 ,0.000002 ,0.051810 ,0.000000 ,0.002053 ,0.000000 },
{0.000042 ,0.000003 ,0.000000 ,0.000098 ,0.000000 ,0.001106 ,0.000002 ,1.000000 ,0.004676 ,0.000000 ,0.001953 ,0.002066 },
{0.000000 ,0.020925 ,0.000009 ,0.114654 ,0.000028 ,0.010605 ,0.051810 ,0.004676 ,1.000000 ,0.000000 ,0.444547 ,0.000000 },
{0.000120 ,0.000013 ,0.119522 ,0.000006 ,0.010363 ,0.001962 ,0.000000 ,0.000000 ,0.000000 ,1.000000 ,0.000001 ,0.000009 },
{0.000000 ,0.255593 ,0.001798 ,0.670119 ,0.004134 ,0.177203 ,0.002053 ,0.001953 ,0.444547 ,0.000001 ,1.000000 ,0.000000 },
{0.559193 ,0.000000 ,0.000001 ,0.000000 ,0.000000 ,0.000051 ,0.000000 ,0.002066 ,0.000000 ,0.000009 ,0.000000 ,1.000000 ,}
};


double decomposedMatrix[Dimensions][Dimensions]=
{
    {1.000000 ,0.018316 ,0.018316 ,0.000335 ,0.135335 ,0.000045 ,0.000045 ,0.000000 ,0.000335 ,0.000000 ,0.000000 ,0.000000 },
{0.018316 ,1.000000 ,0.000335 ,0.018316 ,0.000045 ,0.135335 ,0.000000 ,0.000045 ,0.000000 ,0.000335 ,0.000000 ,0.000000 },
{0.018316 ,0.000335 ,1.000000 ,0.018316 ,0.000045 ,0.000000 ,0.135335 ,0.000045 ,0.000000 ,0.000000 ,0.000335 ,0.000000 },
{0.000335 ,0.018316 ,0.018316 ,1.000000 ,0.000000 ,0.000045 ,0.000045 ,0.135335 ,0.000000 ,0.000000 ,0.000000 ,0.000335 },
{0.135335 ,0.000045 ,0.000045 ,0.000000 ,1.000000 ,0.000000 ,0.000000 ,0.000000 ,0.135335 ,0.000000 ,0.000000 ,0.000000 },
{0.000045 ,0.135335 ,0.000000 ,0.000045 ,0.000000 ,1.000000 ,0.000000 ,0.000000 ,0.000000 ,0.135335 ,0.000000 ,0.000000 },
{0.000045 ,0.000000 ,0.135335 ,0.000045 ,0.000000 ,0.000000 ,1.000000 ,0.000000 ,0.000000 ,0.000000 ,0.135335 ,0.000000 },
{0.000000 ,0.000045 ,0.000045 ,0.135335 ,0.000000 ,0.000000 ,0.000000 ,1.000000 ,0.000000 ,0.000000 ,0.000000 ,0.135335 },
{0.000335 ,0.000000 ,0.000000 ,0.000000 ,0.135335 ,0.000000 ,0.000000 ,0.000000 ,1.000000 ,0.000000 ,0.000000 ,0.000000 },
{0.000000 ,0.000335 ,0.000000 ,0.000000 ,0.000000 ,0.135335 ,0.000000 ,0.000000 ,0.000000 ,1.000000 ,0.000000 ,0.000000 },
{0.000000 ,0.000000 ,0.000335 ,0.000000 ,0.000000 ,0.000000 ,0.135335 ,0.000000 ,0.000000 ,0.000000 ,1.000000 ,0.000000 },
{0.000000 ,0.000000 ,0.000000 ,0.000335 ,0.000000 ,0.000000 ,0.000000 ,0.135335 ,0.000000 ,0.000000 ,0.000000 ,1.000000 }};


BOOST_AUTO_TEST_CASE( LinAlg_IncompleteCholeskyFactorization)
{
	RealMatrix M(Dimensions, Dimensions);   // input matrix
	RealMatrix C(Dimensions, Dimensions);   // matrix for Cholesky Decomposition

	// Initializing matrices
	for (size_t row = 0; row < Dimensions; row++)
	{
		for (size_t col = 0; col < Dimensions; col++)
		{
			M(row, col) = inputMatrix[row][col];
            C(row, col) = 0;
            std::cout << std::setprecision(8) <<std::fixed <<  M(row, col) << " " ;
        }
        std::cout << std::endl;
    }
	//Decompose
	size_t rank = Dimensions;
    double epsilon = 0.0000001;
    GaussianRbfKernel<RealVector> kernel;
    IncompleteCholeskyFactorization<RealVector > icf (&kernel, rank);
    icf.setThreshold (epsilon);
//    icf.adaptToData (M);
    
	//test for equality
	for (size_t row = 0; row < Dimensions; row++)
	{
		for (size_t col = 0; col < Dimensions; col++)
		{
//			BOOST_CHECK_SMALL(C(row, col)-decomposedMatrix[row][col],1.e-14);
            std::cout << std::setprecision(8) <<std::fixed <<  C(row, col) << " " ;
		}
		
		std::cout << std::endl;
	}
}
/*
BOOST_AUTO_TEST_CASE( LinAlg_PivotingCholeskyDecomposition_Base )
{
	RealMatrix M(Dimensions, Dimensions);   // input matrix
	RealMatrix C(Dimensions, Dimensions);   // matrix for Cholesky Decomposition
	PermutationMatrix P(Dimensions);
	// Initializing matrices
	for (size_t row = 0; row < Dimensions; row++)
	{
		for (size_t col = 0; col < Dimensions; col++)
		{
			M(row, col) = inputMatrix[row][col];
			C(row, col) = 0;
		}
	}
	//Decompose
	std::size_t rank = pivotingCholeskyDecomposition(M,P,C);
	swap_rows(P,C);
	
	double error = norm_inf(prod(C,trans(C))-M);
	BOOST_CHECK_SMALL(error,1.e-13);
	BOOST_CHECK_EQUAL(rank,4);
}

RealMatrix createRandomMatrix(RealMatrix const& lambda,std::size_t Dimensions){
	RealMatrix R = blas::randomRotationMatrix(Dimensions);
	RealMatrix Atemp(Dimensions,Dimensions);
	RealMatrix A(Dimensions,Dimensions);
	axpy_prod(R,lambda,Atemp);
	axpy_prod(Atemp,trans(R),A);
	return A;
}

BOOST_AUTO_TEST_CASE( LinAlg_PivotingCholeskyDecomposition_FullRank ){
	std::size_t NumTests = 100;
	std::size_t Dimensions = 48;
	for(std::size_t test = 0; test != NumTests; ++test){
		//first generate a suitable eigenvalue problem matrix A
		RealMatrix lambda(Dimensions,Dimensions);
		lambda.clear();
		for(std::size_t i = 0; i != Dimensions; ++i){
			lambda(i,i) = Rng::uni(1,3.0);
		}
		RealMatrix A = createRandomMatrix(lambda,Dimensions);
		//calculate Cholesky
		RealMatrix C(Dimensions,Dimensions);
		PermutationMatrix P(Dimensions);
		std::size_t rank = pivotingCholeskyDecomposition(A,P,C);
		//test whether result is full rank
		BOOST_CHECK_EQUAL(rank,Dimensions);
		
		//test determinant of C
		double logDetA = trace(log(lambda));
		double logDetC = trace(log(sqr(C)));
		BOOST_CHECK_SMALL(std::abs(logDetA)-std::abs(logDetC),1.e-12);

		//create reconstruction of A
		RealMatrix ATest(Dimensions,Dimensions);
		
		swap_full(P,A);
		
		axpy_prod(C,trans(C),ATest);
		
		//test reconstruction error
		double errorA = norm_inf(A-ATest);
		BOOST_CHECK_SMALL(errorA,1.e-12);
		BOOST_CHECK(!(boost::math::isnan)(norm_frobenius(ATest)));//test for nans
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_PivotingCholeskyDecomposition_RankK ){
	std::size_t NumTests = 100;
	std::size_t Dimensions = 45;
	for(std::size_t test = 0; test != NumTests; ++test){
		std::size_t Rank = Rng::discrete(10,45);
		//first generate a suitable eigenvalue problem matrix A
		RealMatrix lambda(Dimensions,Dimensions);
		lambda.clear();
		for(std::size_t i = 0; i != Rank; ++i){
			lambda(i,i) = Rng::uni(1,3.0);
		}
		RealMatrix A = createRandomMatrix(lambda,Dimensions);
		//calculate Cholesky
		RealMatrix C(Dimensions,Dimensions);
		PermutationMatrix P(Dimensions);
		std::size_t rank = pivotingCholeskyDecomposition(A,P,C);
		
		//test whether result has the correct rank.
		BOOST_CHECK_EQUAL(rank,Rank);

		//create reconstruction of A
		RealMatrix ATest(Dimensions,Dimensions);
		
		swap_full(P,A);
		axpy_prod(C,trans(C),ATest);
		
		//test reconstruction error
		double errorA = norm_inf(A-ATest);
		BOOST_CHECK_SMALL(errorA,1.e-13);
		BOOST_CHECK(!(boost::math::isnan)(norm_frobenius(ATest)));//test for nans
	}
}
*/