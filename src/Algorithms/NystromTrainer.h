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


#ifndef SHARK_ALGORITHMS_NYSTROMTRAINER_H
#define SHARK_ALGORITHMS_NYSTROMTRAINER_H


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

#include "LinAlg/LowRankApproximation.h"
#include "Models/NystromClassifier.h"

#include <shark/Algorithms/QP/QpMcDecomp.h>
#include <shark/Algorithms/QP/QpMcBoxDecomp.h>
#include <shark/Algorithms/QP/QpMcLinear.h>

#include <shark/LinAlg/KernelMatrix.h>
#include <shark/LinAlg/PrecomputedMatrix.h>
#include <shark/LinAlg/CachedMatrix.h>

namespace shark {


///
/// \brief Training of Nystrom feature transformation.

template <class InputType, class CacheType = float>
class NystromTrainer : public AbstractTrainer< NystromClassifier<InputType>, unsigned int >, public IParameterizable 
{
public:
    typedef NystromClassifier<InputType> ModelType; // again, what the..?
    typedef NystromClassifier<InputType> ClassifierType;
    typedef CacheType QpFloatType;
    typedef AbstractKernelFunction<InputType> KernelType;
    typedef AbstractTrainer<InputType, unsigned int> base_type;
    typedef typename ModelType::OutputType OutputType;
    typedef LabeledData<InputType, OutputType> DatasetType;
    
    typedef typename InputType::value_type InputValueType;
    typedef blas::matrix<InputValueType, blas::row_major > MatrixType;
    
    
    
    //! Constructor
    //! \param  kernel         kernel function to use for training and prediction
    //! \param  C              regularization parameter - always the 'true' value of C, even when unconstrained is set
    //! \param  rank        rank of the nystrom matrix or dimension of the subspace to project down
    //! \param offset    whether to train with offset/bias parameter or not
    //! \param  unconstrained  when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver?
    
    NystromTrainer(KernelType* kernel, size_t rank) 
    {
        // FIXME: make sure unconstrained will not produce errors..
        //FIXME: where do i put the damn offset?
        m_kernel = kernel;
        m_rank = rank;
    }

    
    
    /// \brief From INameable: return the class name.
    std::string name() const
    { return "NystromTrainer"; }


    AbstractTrainer< LinearClassifier<InputType> > *linearTrainer() {
        return m_linearTrainer;
    }
    
    
    void setLinearTrainer (AbstractTrainer< LinearClassifier<InputType> > *linearTrainer) {
        m_linearTrainer = linearTrainer;
    }
    
    
    
    void train (ModelType& model, DatasetType const& dataset) 
    {
        // create a kernel approximator and adapt approximator to given data
        BOOST_LOG_TRIVIAL (info) << "Training Nystrom Model.";
        NystromKernelApproximation <RealVector > nka (m_kernel, m_rank);
        nka.adaptToData (dataset); 

        // next transform our data
        DatasetType transformedData;
        transformedData = nka.transformData (dataset);
        
        // after this we need to transform it with the given flavored classifier
        BOOST_LOG_TRIVIAL (info) << "Training linear SVM.";
        LinearClassifier<InputType> linearClassifier;
        m_linearTrainer->train (linearClassifier, transformedData);
        
        // finalize our model
        model.setLinearModel (linearClassifier.decisionFunction());
        model.setLandmarks (nka.landmarks());
        model.setTransformationMatrix(nka.transformationMatrix());
    }
    
private:

    // rank/dimension of resulting feature space
    size_t m_rank;
    
    KernelType* m_kernel;
    
    // linear classifier that will be called after the transformation
    AbstractTrainer< LinearClassifier<InputType> > *m_linearTrainer;
}; 
    

}
#endif
