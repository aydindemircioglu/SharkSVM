//===========================================================================
/*!
 * 
 *
 * \brief       FeatureTransformation Test
 *
 *
 *
 * \author      Aydin Demircioglu
 * \date        2014
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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

#define BOOST_TEST_MODULE LinAlg_FeatureTransformation
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Models/Kernels/PolynomialKernel.h>

#include "LinAlg/LowRankApproximation.h"
#include "LinAlg/RandomFourierFeatures.h"


using namespace shark;


BOOST_AUTO_TEST_CASE(Needs_Specialization)
{   
    size_t rank = 1;
    try {
        // create kernel
        PolynomialKernel<RealVector> *kernel = new PolynomialKernel<RealVector>();
        
        // create transformation
        RandomFourierFeatures <RealVector> rff (kernel, rank);
        rank = 0;
    }
    catch (...) {
    }
    BOOST_REQUIRE_EQUAL(rank, 1);
}



BOOST_AUTO_TEST_CASE( LinAlg_FeatureTransformation)
{
    size_t Dimensions = 5;
	RealMatrix M(Dimensions, Dimensions);   // input matrix
	RealMatrix C(Dimensions, Dimensions);   // matrix for Cholesky Decomposition
    
    size_t index = 0;
    BOOST_REQUIRE_EQUAL(index, 0);
}

