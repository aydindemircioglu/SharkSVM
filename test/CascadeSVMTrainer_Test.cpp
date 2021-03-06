//===========================================================================
/*!
 * 
 *
 * \brief       test case for the CascadeSVMTrainer
 * 
 * 
 * 
 *
 * \author      Aydin Demircioglu
 * \date        2014
 *
 *
 * \par Copyright 2014-2016 Aydin Demircioglu
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================
#define BOOST_TEST_MODULE ALGORITHMS_TRAINERS_CASCADESVMTRAINER
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "Algorithms/CascadeSvmTrainer.h"

#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Data/DataDistribution.h>


using namespace shark;

// This test case consists of training SVMs with
// analytically computable solution. This known
// solution is used to validate the trainer.
BOOST_AUTO_TEST_SUITE (Algorithms_Trainers_CascadeSvmTrainer)

BOOST_AUTO_TEST_CASE( CascadeSvm_TRAINER_SIMPLE_TEST )
{
	// simple 5-point dataset
	std::vector<RealVector> input(5);
	std::vector<unsigned int> target(5);
	size_t i;
	for (i=0; i<5; i++) input[i].resize(2);
	input[0](0) =  0.0; input[0](1) =  0.0; target[0] = 0;
	input[1](0) =  2.0; input[1](1) =  2.0; target[1] = 1;
	input[2](0) = -1.0; input[2](1) = -8.0; target[2] = 0;
	input[3](0) = -1.0; input[3](1) = -1.0; target[3] = 0;
	input[4](0) =  3.0; input[4](1) =  3.0; target[4] = 1;
	ClassificationDataset dataset = createLabeledDataFromRange(input, target);

	// hard-margin training with linear kernel
	{
		std::cout << "C-SVM hard margin" << std::endl;
		LinearKernel<> kernel;
		KernelClassifier<RealVector> svm;
        CascadeSvmTrainer<RealVector> trainer(&kernel, 1e100, true);
// 		trainer.sparsify() = false;
// 		trainer.shrinking() = false;
// 		trainer.stoppingCondition().minAccuracy = 1e-8;
		trainer.train(svm, dataset);
		RealVector param = svm.parameterVector();
		BOOST_REQUIRE_EQUAL(param.size(), 6u);
		std::cout << "alpha: "
			<< param(0) << " "
			<< param(1) << " "
			<< param(2) << " "
			<< param(3) << " "
			<< param(4) << std::endl;
		std::cout << "    b: " << param(5) << std::endl;
//		std::cout << "kernel computations: " << trainer.accessCount() << std::endl;
		
		// test against analytically known solution
		BOOST_CHECK_SMALL(param(0) + 0.25, 1e-6);
		BOOST_CHECK_SMALL(param(1) - 0.25, 1e-6);
		BOOST_CHECK_SMALL(param(2), 1e-6);
		BOOST_CHECK_SMALL(param(3), 1e-6);
		BOOST_CHECK_SMALL(param(4), 1e-6);
		BOOST_CHECK_SMALL(param(5) + 1.0, 1e-6);
	}
	
	// hard-margin training with linear kernel
	{
		std::cout << "C-SVM hard margin with shrinking" << std::endl;
		LinearKernel<> kernel;
		KernelClassifier<RealVector> svm;
		CascadeSvmTrainer<RealVector> trainer(&kernel, 1e100,true);
// 		trainer.sparsify() = false;
// 		trainer.shrinking() = true;
// 		trainer.stoppingCondition().minAccuracy = 1e-8;
		trainer.train(svm, dataset);
		RealVector param = svm.parameterVector();
		BOOST_REQUIRE_EQUAL(param.size(), 6u);
		std::cout << "alpha: "
			<< param(0) << " "
			<< param(1) << " "
			<< param(2) << " "
			<< param(3) << " "
			<< param(4) << std::endl;
		std::cout << "    b: " << param(5) << std::endl;
//		std::cout << "kernel computations: " << trainer.accessCount() << std::endl;
		
		// test against analytically known solution
		BOOST_CHECK_SMALL(param(0) + 0.25, 1e-6);
		BOOST_CHECK_SMALL(param(1) - 0.25, 1e-6);
		BOOST_CHECK_SMALL(param(2), 1e-6);
		BOOST_CHECK_SMALL(param(3), 1e-6);
		BOOST_CHECK_SMALL(param(4), 1e-6);
		BOOST_CHECK_SMALL(param(5) + 1.0, 1e-6);
	}

	// soft-margin training with linear kernel
	{
		std::cout << "C-SVM soft margin" << std::endl;
		LinearKernel<> kernel;
		KernelClassifier<RealVector> svm;
		CascadeSvmTrainer<RealVector> trainer(&kernel, 0.1, true);
// 		trainer.sparsify() = false;
// 		trainer.stoppingCondition().minAccuracy = 1e-8;
// 		trainer.train(svm, dataset);
		RealVector param = svm.parameterVector();
		BOOST_REQUIRE_EQUAL(param.size(), 6u);
		std::cout << "alpha: "
			<< param(0) << " "
			<< param(1) << " "
			<< param(2) << " "
			<< param(3) << " "
			<< param(4) << std::endl;
		std::cout << "    b: " << param(5) << std::endl;
//		std::cout << "kernel computations: " << trainer.accessCount() << std::endl;

		// test against analytically known solution
		BOOST_CHECK_SMALL(param(0) + 0.1, 1e-6);
		BOOST_CHECK_SMALL(param(1) - 0.1, 1e-6);
		BOOST_CHECK_SMALL(param(2), 1e-6);
		BOOST_CHECK_SMALL(param(3) + 0.0125, 1e-6);
		BOOST_CHECK_SMALL(param(4) - 0.0125, 1e-6);
		BOOST_CHECK_SMALL(param(5) + 0.5, 1e-6);
	}
	
}


template<class Model1, class Model2, class Dataset>
void checkSVMSolutionsEqual(
	Model1 const& model1, Model2 const& model2, 
	Dataset const& dataset,
	double epsilon
){
	Data<RealVector> decision1 = model1.decisionFunction()(dataset.inputs());
	Data<RealVector> decision2 = model2.decisionFunction()(dataset.inputs());
	
	for(std::size_t i = 0; i != dataset.numberOfElements(); ++i){
		BOOST_CHECK_CLOSE(
			decision1.element(i)(0),
			decision2.element(i)(0),
			epsilon
		);
	}
}




BOOST_AUTO_TEST_SUITE_END()
