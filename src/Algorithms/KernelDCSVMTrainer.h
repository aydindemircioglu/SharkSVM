//===========================================================================
/*!
 *
 *
 * \brief       DCSVM clone. Hopefully a tad faster.
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


#ifndef SHARK_ALGORITHMS_KERNELDCSVMTRAINER_H
#define SHARK_ALGORITHMS_KERNELDCSVMTRAINER_H

#include <iostream>
#include "GlobalParameters.h"
#include "LibSVMDataModel.h"
#include "SharkSVM.h"
#include "Models/ClusteredKernelClassifier.h"

#include <shark/Algorithms/KMeans.h> //k-means algorithm
#include <shark/Algorithms/Trainers/AbstractTrainer.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h> //normalize
#include <shark/Core/IParameterizable.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/DataView.h>
#include <shark/Models/Clustering/HardClusteringModel.h>//model performing hard clustering of points
#include <shark/Models/Kernels/GaussianRbfKernel.h> //the used kernel for the SVM
#include <shark/Models/Kernels/KernelExpansion.h>
#include <shark/Models/Kernels/KernelHelpers.h>
#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>


namespace shark {


///
/// \brief Divide and Conquer (DC) for kernel-based SVM.
///
/// \par
/// This implementation is an adaptation of the DCSVM algorithm, see the paper
/// <i>    Cho-Jui Hsieh, Si Si, Inderjit Dhillon: A Divide-and-Conquer Solver for Kernel Support Vector Machines; JMLR W&CP 32 (1) :566-574, 2014</i><br/><br/>
///
/// \par 
    /// NOTE: earlyprediction means: level=4, level_stop=4, kk=4, mode=0, method=1
/// \par 
    /// NOTE: exact means: level=4, level_stop=1, kk=4, mode=1, method=0
/// \par
    /// NOTE: max 20000 samples are used for clustering, need to parametrize that
/// NOTE: As the original solver is based on LibSVM, this class depends the usual CSVMTrainer.
/// Basically this solver could be replaced by other solver.
/// NOTE: Though the sourcecode and the paper talk about kernel k-means, they somehow do 
/// not seem to use it in their code? is this on purpose?
///
    template <class InputType>
    class KernelDCSVMTrainer : public AbstractTrainer<ClusteredKernelClassifier<InputType> >, public IParameterizable {
        public:
            typedef AbstractTrainer< KernelExpansion<InputType> > base_type;
            typedef AbstractKernelFunction<InputType> KernelType;
            typedef ClusteredKernelClassifier<InputType> ClassifierType;
            typedef KernelClassifier<InputType> BaseClassifierType;
            typedef KernelExpansion<InputType> ModelType;
            typedef AbstractLoss<unsigned int, RealVector> LossType;
            typedef typename ConstProxyReference<typename Batch<InputType>::type const>::type ConstBatchInputReference;
            typedef LabeledData<InputType, unsigned int> DataType;
            typedef typename DataType::element_type ElementType;
            typedef typename DataType::IndexSet IndexSet;
            
            
            /// \brief Constructor
            ///
            /// \param  kernel          kernel function to use for training and prediction
            /// \param  loss            (sub-)differentiable loss function
            /// \param  C               regularization parameter - always the 'true' value of C, even when unconstrained is set
            /// \param  epsilon         epsilon for the basic SVM solver to  be used (CSVMTrainer currently)
            /// \param  offset          whether to train with offset/bias parameter or not
            /// \param  nCluster        number of clusters to use
            /// \param  earlyPrediction     should we stop early or compute to the bitter end?
            /// \param  unconstrained   when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver?
            KernelDCSVMTrainer (KernelType *kernel, const LossType *loss, double C, double epsilon = 0.0001, bool offset = false,
                                size_t nCluster = 64, bool earlyPrediction = true, std::size_t cacheSize = 0x10000000,
                                bool unconstrained = false)
                : m_kernel (kernel)
                , m_loss (loss)
                , m_C (C)
                , m_epsilon(epsilon)
                , m_offset (offset)
                , m_nCluster(nCluster)
                , m_earlyPrediction (earlyPrediction)
                , m_unconstrained (unconstrained)
                , m_cacheSize(cacheSize)
            { }


            
            /// \brief From INameable: return the class name.
            std::string name() const {
                return "KernelDCSVMTrainer";
            }

            
            
            void train (ClassifierType &classifier, const LabeledData<InputType, unsigned int> &dataset) {

                // for now only support earlyPrediction
                if (m_earlyPrediction == false)
                    throw SHARKSVMEXCEPTION ("Currently only early prediction is supported.");

                
                std::size_t ell = dataset.numberOfElements();
//                unsigned int classes = numberOfClasses (dataset);

                // generate data view
                DataView<const DataType> dataView (dataset);
                
                // first we do an approximate k-means clustering.
                // the number of clusters are given to us,
                // but as we want only an approximate clustering,
                // we take only at most 20000 samples.
                // if we do not have that much, we take all there is.
                size_t maxNSamples = 20000;
                size_t nSamplesForClusterings = std::min (maxNSamples, ell);
                DataView<const DataType> randomSamples = randomSubset(dataView, nSamplesForClusterings);
                
                // cluster the subsampled data
                Centroids centroids;
                //size_t iterations = 
                kMeans(toDataset(randomSamples).inputs(), m_nCluster, centroids);

                // use the centroids to cluster the whole dataset
                HardClusteringModel<InputType> clusterModel(&centroids);
                Data<unsigned> clusterLabel = clusterModel(dataset.inputs());
               
                
                // in early prediction the plain idea is just to cluster the data
                // and apply one svm in each of the data strains. this gives as many
                // models as we have clusters.
                // then at testing time, the nearest cluster is chosen and the point
                // is evaluated there.

                // global classifier will be the set of all local classifiers
                ClassifierType globalClassifier;
                
                // for each of our clusters do train a basic svm.
                for (size_t c = 0; c < m_nCluster; c++)
                {
                    // create a dataset with just the elements in this cluster
                    std::vector<std::size_t> indices;
                    for (size_t e = 0; e < clusterLabel.numberOfElements(); e++)
                    {
                        // this is stupid, but i do not get indexedSubset running, not at all.
                        if (clusterLabel.element(e) == c)
                            indices.push_back(e);
                    }

                    DataType clusterData = toDataset(subset(toView(dataset),indices));
                    
                    // solve subproblem by normal svm trainer
                    CSvmTrainer<InputType> *kernelCSVMTrainer = new CSvmTrainer<InputType> (m_kernel, m_C, m_offset);
                    kernelCSVMTrainer->stoppingCondition().minAccuracy = m_epsilon;
                    kernelCSVMTrainer->setCacheSize (m_cacheSize);
                    
                    // train the svm 
                    BaseClassifierType localClassifier;
                    kernelCSVMTrainer->train(localClassifier, clusterData);
                    
                    // add the cluster model to the overall model
                    globalClassifier.appendCluster (centroids.centroids().element(c), localClassifier); 
                }
                
                // thats all.
            }

            
            /// get the kernel function
            KernelType *kernel() {
                return m_kernel;
            }

            
            /// get the kernel function
            const KernelType *kernel() const {
                return m_kernel;
            }
            
            
            /// set the kernel function
            void setKernel (KernelType *kernel) {
                m_kernel = kernel;
            }

            
            /// check whether the parameter C is represented as log(C), thus,
            /// in a form suitable for unconstrained optimization, in the
            /// parameter vector
            bool isUnconstrained() const {
                return m_unconstrained;
            }

            
            /// return the value of the regularization parameter
            double C() const {
                return m_C;
            }

            
            /// set the value of the regularization parameter (must be positive)
            void setC (double value) {
                RANGE_CHECK (value > 0.0);
                m_C = value;
            }

            
            /// check whether the model to be trained should include an offset term
            bool trainOffset() const {
                return m_offset;
            }

            
            ///\brief  Returns the vector of hyper-parameters.
            RealVector parameterVector() const {
                size_t kp = m_kernel->numberOfParameters();
                RealVector ret (kp + 1);

                if (m_unconstrained)
                    init (ret) << parameters (m_kernel), log (m_C);
                else
                    init (ret) << parameters (m_kernel), m_C;

                return ret;
            }

            
            ///\brief  Sets the vector of hyper-parameters.
            void setParameterVector (RealVector const &newParameters) {
                size_t kp = m_kernel->numberOfParameters();
                SHARK_ASSERT (newParameters.size() == kp + 1);
                init (newParameters) >> parameters (m_kernel), m_C;

                if (m_unconstrained) m_C = exp (m_C);
            }

            ///\brief Returns the number of hyper-parameters.
            size_t numberOfParameters() const {
                return m_kernel->numberOfParameters() + 1;
            }

            
        protected:
            KernelType *m_kernel;                     ///< pointer to kernel function
            const LossType *m_loss;                   ///< pointer to loss function
            double m_C;                               ///< regularization parameter
            double m_epsilon;                       /// < stopping criterion for the basic SVM solver 
            bool m_offset;                            ///< should the resulting model have an offset term?
            std::size_t m_nCluster;                 /// < number of clusters to use for the DC-strategy
            bool m_earlyPrediction;                 /// < early prediction (=stopping) or not?
            bool m_unconstrained;                     ///< should C be stored as log(C) as a parameter?
            std::size_t m_cacheSize;                    /// < cachesize for the basic SVM solver
    };


}
#endif
