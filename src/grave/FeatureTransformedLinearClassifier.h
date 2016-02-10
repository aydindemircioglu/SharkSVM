//===========================================================================
/*!
 * 
 *
 * \brief       FeatureTransformed Linear Classifier 
 * 
 * \par
 * This classifier will apply a linear classifier after
 * transforming the data with a given feature transformation.
 * 
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


#ifndef SHARK_MODELS_FEATURETRANSFORMEDLINEARCLASSIFIER_H
#define SHARK_MODELS_FEATURETRANSFORMEDLINEARCLASSIFIER_H

#include <shark/Models/Converter.h>
#include <shark/Models/LinearClassifier.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/DataView.h>

#include "LinAlg/FeatureTransformation.h"

namespace shark {


///
/// \brief Linear classifier in a feature transformed space.
///
///
/// \tparam InputType Type of basis elements supplied to the kernel
///
    
template<class InputType>
class FeatureTransformedLinearClassifier: public LinearClassifier<InputType>{
public:
    typedef RealVector OutputType;
	typedef KernelExpansion<InputType> KernelExpansionType;
    typedef KernelClassifier<InputType> ClassifierType;
    typedef AbstractModel<InputType, RealVector> base_type;
    typedef typename Batch<InputType>::type BatchInputType;
    typedef typename Batch<OutputType>::type BatchOutputType;

	
	FeatureTransformedLinearClassifier()
	{ }
	
	
	
	Data<InputType> landmarks ()
    {
        return(m_landmarks);
    }
    
    
    
    Data<InputType> setLandmarks (Data<InputType> landmarks)
    {
        m_landmarks = landmarks;
    }
    
    
	
	/// name
	std::string name() const
	{ return "FeatureTransformedLinearClassifier"; }

	
protected:
    
    /// cluster centers
    Data<InputType> m_landmarks;
};


}


#endif
