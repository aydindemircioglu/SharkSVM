//===========================================================================
/*!
 *
 *
 * \brief       (Naive) Implementation of Cascade SVM
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


#ifndef SHARK_ALGORITHMS_CASCADESVMTRAINER_H
#define SHARK_ALGORITHMS_CASCADESVMTRAINER_H

#include <iostream>
#include "GlobalParameters.h"
#include "LibSVMDataModel.h"
#include "SharkSVM.h"
#include "CSvmTrainer.h"

#include <shark/Algorithms/Trainers/AbstractTrainer.h>
#include <shark/Core/IParameterizable.h>
#include <shark/LinAlg/KernelMatrix.h>
#include <shark/LinAlg/PartlyPrecomputedMatrix.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/KernelExpansion.h>
#include <shark/Models/Kernels/KernelHelpers.h>
#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>


namespace shark {


///
/// \brief One of the oldest and prototypical divide-and-conquer-solvers.
///
/// \par
///
    template <class InputType, class CacheType = float>
    class CascadeSvmTrainer : public AbstractTrainer< KernelClassifier<InputType> >, public IParameterizable {
        public:
            typedef AbstractTrainer< KernelExpansion<InputType> > base_type;
            typedef AbstractKernelFunction<InputType> KernelType;
            typedef KernelClassifier<InputType> ClassifierType;
            typedef KernelExpansion<InputType> ModelType;
            typedef LabeledData<InputType, unsigned int> DataType;
            typedef typename ConstProxyReference<typename Batch<InputType>::type const>::type ConstBatchInputReference;
            typedef CacheType QpFloatType;

            typedef KernelMatrix<InputType, QpFloatType> KernelMatrixType;
            typedef PartlyPrecomputedMatrix< KernelMatrixType > PartlyPrecomputedMatrixType;


            /// \brief Constructor
            ///
            /// \param  kernel          kernel function to use for training and prediction
            /// \param  loss            (sub-)differentiable loss function
            /// \param  C               regularization parameter - always the 'true' value of C, even when unconstrained is set
            /// \param  offset          whether to train with offset/bias parameter or not
            /// \param  unconstrained   when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver?
            CascadeSvmTrainer (KernelType* kernel, double C, bool offset, bool unconstrained = false, size_t cacheSize = 0x4000000)
                : m_kernel (kernel)
                , m_C (C)
                , m_offset (offset)
                , m_unconstrained (unconstrained)
                , m_epochs (0)
                , m_levels (7)
                , m_subClusters (4)
                , m_epsilon (0.001)
                , m_cacheSize (cacheSize)
            { }


            /// return current cachesize
            double cacheSize() const {
                return m_cacheSize;
            }


            void setCacheSize (std::size_t size) {
                m_cacheSize = size;
            }
            
            

            /// \brief From INameable: return the class name.
            std::string name() const {
                return "CascadeSvmTrainer";
            }
            
            
/*            
            
            
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
                
                
                
   */             
                


            // TODOs:
            // -make csvm parametrizable
            // -make clustering parametrizable

            void train (ClassifierType &classifier, const LabeledData<InputType, unsigned int> &dataset) {
                std::size_t ell = dataset.numberOfElements();
                unsigned int classes = numberOfClasses (dataset);
                ModelType &model = classifier.decisionFunction();

                // this will remember which vectors are support vectors
                
                // for cascade svm: the initial splitting does not change over epochs.
                // so we generate a permutation of the data and will just split it 'binary' later on

                // generate data view
                DataView<const DataType> dataView (dataset);
                DataView<const DataType> permutedData = randomSubset(dataView, ell);
                
                // for each epoch
                for (size_t epoch = 0; epoch < m_epochs; epoch++) {
                    
                    // for each level
                    for (size_t level = m_levels; level > 0; level++) {
                        // we need subclusters^{level-1} subsets
                        // here we proceed serially, so we have a simply for loop.
                        
                        for (size_t subset = 0; subset < pow(m_subClusters, level-1); subset++)
                        {
                            // get the dataset
                            // cascade svm: just take the corresponding split
                            DataView<const DataType> dataSplit;

                            // take the alphas
                            if ((epoch > 0) && (level == 0)) {
                                // add global alphas from last epoch
                            }
                            
                            if (level > 0) {
                                // add a
//                                alpha = subset (..)
                            }
                            
                            
                            // wie ist dann die verbindung von alphas und den subsets zu machen???
                            // und notfalls kernel cache etc.
                            
                            // solve subproblem by normal svm trainer
                            CSvmTrainer<InputType> *kernelCSVMTrainer = new CSvmTrainer<InputType> (m_kernel, m_C, m_offset, false ,true, true);
                            kernelCSVMTrainer->stoppingCondition().minAccuracy = m_epsilon;
                            kernelCSVMTrainer->setCacheSize (m_cacheSize);

                            // preinit the alphas
                            
                            //  ALPHAS ARE A MATRIX m_alpha.resize(basis.numberOfElements(), outputs);
                            
                            // train the svm 
                            ClassifierType localClassifier;
                            kernelCSVMTrainer->train(localClassifier, toDataset(dataSplit));
                            
                            // obtain all alphas and put them back into the global array
                            Data<InputType> supportVectors;
                            RealMatrix alphas; // thats how it is. plain reality.
                            supportVectors = localClassifier.decisionFunction().basis();
                            alphas = localClassifier.decisionFunction().alpha();
                        }
                        
                        // cascade svm  tells us that we should check if anything 'changed'
                        // after the first level. if so, we bail out.
                        ClassifierType localClassifier;
                        RealMatrix globalAlphas = localClassifier.decisionFunction().alpha();
                        RealMatrix currentAlphas = localClassifier.decisionFunction().alpha();
                        
                        //if (globalAlphas == currentAlphas) 
                        {
                            // bail out
                        }
                    }
                }
                
                // finally we save the alphas 
                
            }

            
            
            /// Return the number of levels, i.e. 
            std::size_t subClusters() const {
                return m_subClusters;
            }
            
            
            /// Set the number of training epochs.
            /// A value of 0 indicates that the default of max(10, C) should be used.
            void setSubClusters(std::size_t value) {
                // sanity check
                if (m_levels < 2) 
                    throw SHARKSVMEXCEPTION ("Cascade SVM needs at least two subclusters per divide step!");
                
                m_subClusters = value;
            }
            
            
            
            /// Return the number of levels, i.e. 
            std::size_t levels() const {
                return m_levels;
            }
            
            
            /// Set the number of training epochs.
            /// A value of 0 indicates that the default of max(10, C) should be used.
            void setLevels (std::size_t value) {
                // sanity check
                if (m_levels == 0) 
                    throw SHARKSVMEXCEPTION ("Cascade SVM needs at least one layer!");
                
                m_levels = value;
            }
            
            
            
            /// Return the number of training epochs.
            std::size_t epochs() const {
                return m_epochs;
            }

            
            /// Set the number of training epochs.
            void setEpochs (std::size_t value) {
                m_epochs = value;
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
            double m_C;                               ///< regularization parameter
            bool m_offset;                            ///< should the resulting model have an offset term?
            bool m_unconstrained;                     ///< should C be stored as log(C) as a parameter?
            std::size_t m_epochs;                     ///< number of training epochs (sweeps over the data), or 0 for default = max(10, C)
            
            // number of layers. initially the dataset will be divided into 2^{level-1} subsets. Default is 7.
            std::size_t m_levels;

            // number of subclusters. for classical cascade svm we have m_subClusters = 2, DCSVM has 4.
            std::size_t m_subClusters;
            
            // SVM solver parameters-- TODO: allow for any solve
            double m_epsilon;

            // cache size to use.
            std::size_t m_cacheSize;
            
    };


}
#endif
