//===========================================================================
/*!
 * 
 *
 * \brief       Nystrom Classifier 
 * 
 * \par
 * This classifier will apply a linear classifier after
 * transforming the data with the nystrom method.
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


#ifndef SHARK_MODELS_NYSTROMCLASSIFIER_H
#define SHARK_MODELS_NYSTROMCLASSIFIER_H

#include <shark/Models/Converter.h>
#include <shark/Models/LinearClassifier.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/DataView.h>

#include "LinAlg/NystromKernelApproximation.h"


namespace shark {


///
/// \brief Linear classifier in a feature transformed space.
///
///
/// \tparam InputType Type of basis elements supplied to the kernel
///
    
template<class InputType>
class NystromClassifier: public LinearClassifier<InputType>{
public:
    typedef AbstractModel<InputType, RealVector> base_type;
    typedef unsigned int OutputType;
    typedef LinearClassifier<InputType> Model;
    typedef typename Batch<InputType>::type BatchInputType;
    typedef typename Batch<OutputType>::type BatchOutputType;
    typedef typename Model::BatchInputType ModelBatchInputType;
    typedef typename Model::BatchOutputType ModelBatchOutputType;
    typedef AbstractKernelFunction<InputType> KernelType;
    
    typedef typename InputType::value_type InputValueType;
    typedef blas::matrix<InputValueType, blas::row_major > MatrixType;
    
    
    NystromClassifier() 
	{ }
	
 
//    NystromClassifier(KernelType* kernel)
//    : base_type(KernelExpansionType(kernel))
//    { }
    
    
    
    LinearModel<InputType> linearModel () 
    {
        return (m_linearModel);
    }

    
    
    Data<InputType> landmarks ()
    {
        return(m_landmarks);
    }
    
    
    
    
    void setLinearModel (LinearModel<InputType> &model)
    {
        m_linearModel = model;
    }
    
    
    
    void setLandmarks (Data<InputType> landmarks)
    {
        m_landmarks = landmarks;
    }
    

    MatrixType transformationMatrix() {
        return (m_transformationMatrix);
    }
    
    
    
    void setTransformationMatrix (MatrixType transformationMatrix)
    {
        m_transformationMatrix = transformationMatrix;
    }
    
    
    
    void eval(BatchInputType const& input, BatchOutputType& output) const
    {
        BOOST_LOG_TRIVIAL (trace) << "Evaluating input, output batch";

        // transform data first
        BOOST_LOG_TRIVIAL (trace) << "Transforming data into feature space";
        UnlabeledData<InputType> transformedData ;
//        transformedData = transform (input, NystromFeatureTransform<InputType> (m_transformationMatrix, (this->m_kernel), m_landmarks));
    //!!!!!!!!!!!!!!!
        
        
        
        
        ModelBatchOutputType modelResult;
        BatchInputType transformedInput;
        
        // f...k you
        ArgMaxConverter<LinearModel<InputType> > linearClassifier (m_linearModel);
        linearClassifier.eval (transformedInput,modelResult);
    }

    
    
    void eval(BatchInputType const& input, BatchOutputType& output, State& state) const
    {
        BOOST_LOG_TRIVIAL (trace) << "Evaluating input, output batch, state";
        eval(input,output);
    }

    

    void eval(InputType const & pattern, OutputType& output) const
    {
        BOOST_LOG_TRIVIAL (trace) << "Evaluating pattern, output batch";
        typename Model::OutputType modelResult;
        
        eval(pattern,modelResult);
    }
       
    
	
	/// name
	std::string name() const
	{ return "NystromClassifier"; }

	
protected:
    
    /// cluster centers
    Data<InputType> m_landmarks;
    
    /// linear model 
    LinearModel<InputType> m_linearModel;
    
    /// transformation matrix
    MatrixType m_transformationMatrix;
};


}


#endif
