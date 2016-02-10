//===========================================================================
/*!
 * 
 *
 * \brief       Feature transformed Support Vector Machine Trainer for the standard linear C-SVM
 * 
 * 
 * \par
 * 
 * 
 * 
 *
 * \author      Aydin Demircioglu
 * \date        -
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


#ifndef SHARK_ALGORITHMS_FEATURETRANSFORMEDLINEARTRAINER_H
#define SHARK_ALGORITHMS_FEATURETRANSFORMEDLINEARTRAINER_H


#include <shark/Algorithms/Trainers/AbstractSvmTrainer.h>
#include <shark/Algorithms/Trainers/AbstractWeightedTrainer.h>
#include <shark/Algorithms/QP/BoxConstrainedProblems.h>
#include <shark/Algorithms/QP/SvmProblems.h>
#include <shark/Algorithms/QP/QpBoxLinear.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Models/LinearClassifier.h>
#include <shark/LinAlg/CachedMatrix.h>
#include <shark/LinAlg/GaussianKernelMatrix.h>
#include <shark/LinAlg/KernelMatrix.h>
#include <shark/LinAlg/PrecomputedMatrix.h>
#include <shark/LinAlg/RegularizedKernelMatrix.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>

#include "LinAlg/FeatureTransformation.h"
#include "Models/FeatureTransformedLinearClassifier.h"


namespace shark {


///
/// \brief Training of C-SVMs for feature-transformed binary classification.


template <class InputType>
class FeatureTransformedLinearTrainer : public AbstractLinearSvmTrainer<InputType>
{
public:
    typedef FeatureTransformedLinearTrainer<InputType> base_type;

    FeatureTransformedLinearTrainer(FeatureTransformation<InputType> *featureTransformation,  double C, bool offset = false) 
    : AbstractLinearSvmTrainer<InputType>(C, false){
        // save featureTransformation
        m_featureTransformation = featureTransformation;
        m_offset = offset;
        // FIXME: unconstrained will produce errors..
    }

	
	
	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "FeatureTransformedLinearTrainer"; }

	
	
	virtual void train(LinearClassifier<InputType>& model, LabeledData<InputType, unsigned int> const& dataset)
    {
        // transform dataset
        BOOST_LOG_TRIVIAL (info) << "This should not be called directly.";
        
        
        // train linear model
        
        // return trained model
        
        //model.decisionFunction().setStructure(w);
    }
    
    
	
	virtual void train(FeatureTransformedLinearClassifier<InputType>& model, LabeledData<InputType, unsigned int> const& dataset)
	{
        // transform dataset
        BOOST_LOG_TRIVIAL (info) << "Transforming data.";
        
        LabeledData<InputType, unsigned int> transformedData;
        transformedData = m_featureTransformation->transformData (dataset);
       
        
        //FIXME: where do i put the damn offset?
        
        // train linear model
        BOOST_LOG_TRIVIAL (info) << "Training linear SVM.";
        LinearCSvmTrainer<InputType> linearCSVMTrainer (this->C(), false);
        LinearClassifier<InputType> linearModel;
        linearCSVMTrainer.train (linearModel, transformedData);
        
        // add whatever it takes for the classifier to do whatever it needs
		model.setLandmarks (m_featureTransformation -> landmarks());
	}
	
	
private:
    
    FeatureTransformation<InputType> *m_featureTransformation;
    
    bool m_offset;
};

}
#endif
