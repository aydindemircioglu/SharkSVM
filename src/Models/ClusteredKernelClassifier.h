//===========================================================================
/*!
 * 
 *
 * \brief       Clustered kernel classifier
 * 
 * \par
 * Affine linear kernel expansions resulting from Support
 * vector machine (SVM) training and other kernel methods.
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


#ifndef SHARK_MODELS_CLUSTEREDKERNELCLASSIFIER_H
#define SHARK_MODELS_CLUSTEREDKERNELCLASSIFIER_H

#include <shark/Models/Converter.h>
#include <shark/Models/Kernels/AbstractKernelFunction.h>
#include <shark/Models/Kernels/KernelExpansion.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/DataView.h>


namespace shark {


///
/// \brief Linear classifier in a kernel feature space.
///
/// This model is a simple wrapper for the KernelExpansion calculating the arg max
/// of the outputs of the model. This is the model used by kernel classifier models like SVMs.
///
/// \tparam InputType Type of basis elements supplied to the kernel
///
template<class InputType, class OutputType = unsigned int>
class ClusteredKernelClassifier: public AbstractModel<InputType, OutputType>{
public:
//	typedef ArgMaxConverter<KernelExpansion<InputType> > base_type;
//    typedef unsigned int OutputType;
	typedef AbstractKernelFunction<InputType> KernelType;
	typedef KernelExpansion<InputType> KernelExpansionType;
    typedef KernelClassifier<InputType> ClassifierType;
    typedef AbstractModel<InputType, RealVector> base_type;
//    typedef typename base_type::BatchInputType BatchInputType;
 //   typedef typename base_type::BatchOutputType BatchOutputType;
    /// This could for example be std::vector<InputType> but for example for RealVector it could be RealMatrix
    typedef typename Batch<InputType>::type BatchInputType;
    /// \brief defines the batch type of the output type
    typedef typename Batch<OutputType>::type BatchOutputType;
    
	ClusteredKernelClassifier()
	{ }
	
	
	ClusteredKernelClassifier(KernelType* kernel)
	: base_type(KernelExpansionType(kernel))
	{ }
	
	
	/// will append a given local classifier (=pair of kernelclassifier and a center)
	/// to the cluster kernel classifier
	void appendCluster (InputType clusterCenter, ClassifierType &classifier) 
    {
        // add a cluster to the classifier
        m_classifiers.push_back (classifier);
        m_clusterCenters.push_back(clusterCenter);

        // TODO: some more sanity/integrity checks?
        
        
        SHARK_ASSERT (m_classifiers.size() == m_clusterCenters.size());
    }

    
    
    /// evaluate on a whole batch 
    ///
    void eval(BatchInputType const& input, BatchOutputType& output) const
    {   
        // given the testdata, we just evaluate each point.
        // TODO: speed up

        // sanity check
        if (m_classifiers.size() == 0)
            throw SHARKSVMEXCEPTION ("Cannot evaluate global classifier without any local classifier!");
        
        // make sure output has the correct size
        std::size_t numPatterns = boost::size(input);
        output.resize(numPatterns);

        // now for every element evaluate
        for (std::size_t i=0; i < numPatterns; i++){
            eval(get (input, i), get (output, i));
        }
    }
    
    
    
    /// evaluate a batch with some state, we simply do not use.
    ///
    void eval(BatchInputType const& patterns, BatchOutputType& outputs, State& state)const{
        eval(patterns,outputs);
    }
    
    
    
    /// evaluate a single pattern.
    ///
    void eval(InputType const & pattern, OutputType& output)const{
//        typename Model::OutputType modelResult;
        typename ClassifierType::OutputType modelResult;
        
        // sanity check
        if (m_clusterCenters.size() == 0)
            throw SHARKSVMEXCEPTION ("Cannot evaluate global classifier without any local classifier!");
        
        // given the test pattern, we search for the nearest clustercenter, 
        // then use the local classifier associated with that
        // cluster to obtain a prediction
        
        // find nearest clustercenter
        // TODO: speed up.
        double mDist = std::numeric_limits<double>::infinity();
        ClassifierType mClassifier;
        for (size_t i = 0; i < m_clusterCenters.size(); i++)
        {
            double cDist = blas::norm_2 (m_clusterCenters[i] - pattern);
            if (cDist < mDist)
                mClassifier = m_classifiers[i];
        }
        
        // use the classifier we obtained to yield a prediction
        // as the local classifier is something like a argmax in general,
        // we do not need anything else like argmaxing for ourself.
        mClassifier.eval(pattern,modelResult);
    }
    
    
	/// name
	std::string name() const
	{ return "ClusteredKernelClassifier"; }

	
protected:
    
    /// cluster centers
    std::vector<InputType> m_clusterCenters;
    
    /// classifiers for each cluster
    std::vector<ClassifierType> m_classifiers;
};


}


#endif
