//===========================================================================
/*!
 *
 *
 * \brief       LabelOrder Test
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

#define BOOST_TEST_MODULE LabelOrder

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "Data/LabelOrder.h"
#include <shark/Data/DataDistribution.h>
#include <shark/LinAlg/KernelMatrix.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/KernelExpansion.h>


#define CHECK_LESS_EQ( a, b ) BOOST_CHECK_PREDICATE( std::less_equal<size_t>(), (a)(b)) 


using namespace shark;

BOOST_AUTO_TEST_CASE( LabelOrder_General )
{
    // create a dataset 
    size_t datasetSize = 64;
    Chessboard problem (2,0);
    LabeledData<RealVector, unsigned int> dataset = problem.generateDataset(datasetSize);
    
    // now map distort every, reversing is enough. make sure we have label 0 in it
    // to test if we can handle that too
    for(std::size_t i = 0; i < dataset.numberOfElements(); ++i)
        dataset.labels().element(i) = 2*(dataset.numberOfElements() - 1) - 2*i;
    
    // create a copy we can compare with later on
    LabeledData<RealVector, unsigned int> datasetCopy = dataset;
    
    // now reorder the dataset
    LabelOrder labelOrder;
    labelOrder.normalizeLabels (dataset);
    
    // obtain the ordering 
    std::vector<unsigned int> internalOrder;
    labelOrder.getLabelOrder(internalOrder);
    
    // check the order 
    for(std::size_t i = 0; i < internalOrder.size(); ++i)
        BOOST_REQUIRE_EQUAL(internalOrder[i], 2*(dataset.numberOfElements() - 1) - 2*i);
    
    
    // finally map the labels back on the copy
    labelOrder.restoreOriginalLabels(dataset);
    
    // make sure we did not loose anything
    for(std::size_t i = 0; i < internalOrder.size(); ++i)
        BOOST_REQUIRE_EQUAL(dataset.labels().element(i), datasetCopy.labels().element(i) );
    
    // now check for some error cases:
    // create labels that are out of range and call the restore function
    LabeledData<RealVector, unsigned int> datasetBroken = dataset;
    for(std::size_t i = 0; i < dataset.numberOfElements(); ++i)
        dataset.labels().element(i) = internalOrder.size() + i;
    
    try {
        labelOrder.restoreOriginalLabels(datasetBroken);
        
        // this should have thrown an error.
        BOOST_REQUIRE_EQUAL(1, 0);
    } 
    catch (...)
    {
        // everything is fine.
        BOOST_REQUIRE_EQUAL(0, 0);
    }
}
	

	
	BOOST_AUTO_TEST_CASE( LabelOrder_Specific )
    {
        bool ULTRAVERBOSE = true;
        
        // create a dataset 
        size_t datasetSize = 12;
        Chessboard problem (2,0);
        LabeledData<RealVector, unsigned int> dataset = problem.generateDataset(datasetSize);
        
        // now map distort every, reversing is enough. make sure we have label 0 in it
        // to test if we can handle that too
        for(std::size_t i = 0; i < dataset.numberOfElements(); ++i)
            dataset.labels().element(i) = Rng::discrete(1, 2)*Rng::discrete(2, 6);

        if (ULTRAVERBOSE) {
            std::cout << "initial dataset is:\n";
            for(std::size_t i = 0; i < dataset.numberOfElements(); ++i) {
                std::cout << dataset.labels().element(i) << " ";
            }
            std::cout << "\n";
        }        
        
        // create a copy we can compare with later on
        LabeledData<RealVector, unsigned int> datasetCopy = dataset;
        
        // now reorder the dataset
        LabelOrder labelOrder;
        labelOrder.normalizeLabels (dataset);
        
        // obtain the ordering 
        std::vector<unsigned int> internalOrder;
        labelOrder.getLabelOrder(internalOrder);
        
        // check the order 'basic style'
        for(std::size_t i = 0; i < internalOrder.size(); ++i)
        {
            // the current label must be the first occurence in the dataset
            size_t firstOccurence = 0;
            for (std::size_t k = 0; k < i; k++)
            {
                firstOccurence = k;
                if (datasetCopy.labels().element(k) == internalOrder[i])
                    break;
            }
            CHECK_LESS_EQ (firstOccurence, i);
        }
        
        
        if (ULTRAVERBOSE) {
            std::cout << "normalized dataset is:\n";
            for(std::size_t i = 0; i < dataset.numberOfElements(); ++i) {
                std::cout << dataset.labels().element(i) << " ";
            }
            std::cout << "\n";
        }        
        
        if (ULTRAVERBOSE) {
            std::cout << "internal order is:\n";
            for(std::size_t i = 0; i < internalOrder.size(); ++i) {
                std::cout << internalOrder[i] << " ";
            }
            std::cout << "\n";
        }        
        
        // finally map the labels back on the copy
        labelOrder.restoreOriginalLabels(dataset);
 
        
        if (ULTRAVERBOSE) {
            std::cout << "re-transformed dataset is:\n";
            for(std::size_t i = 0; i < dataset.numberOfElements(); ++i) {
                std::cout << dataset.labels().element(i) << " ";
            }
            std::cout << "\n";
        }        
        
        
        
        // make sure we did not loose anything
        for(std::size_t i = 0; i < internalOrder.size(); ++i)
            BOOST_REQUIRE_EQUAL(dataset.labels().element(i), datasetCopy.labels().element(i) );
        
        // now check for some error cases:
        // create labels that are out of range and call the restore function
        LabeledData<RealVector, unsigned int> datasetBroken = dataset;
        for(std::size_t i = 0; i < dataset.numberOfElements(); ++i)
            dataset.labels().element(i) = internalOrder.size() + i;
        
        try {
            labelOrder.restoreOriginalLabels(datasetBroken);
            
            // this should have thrown an error.
            BOOST_REQUIRE_EQUAL(1, 0);
        } 
        catch (...)
        {
            // everything is fine.
            BOOST_REQUIRE_EQUAL(0, 0);
        }
    }
    