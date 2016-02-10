/*!
 *
 *
 * \brief       Abstract class for all lowrank linearization methods
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

#ifndef SHARK_LINALG_NYSTROMKERNELAPPROXIMATION_H
#define SHARK_LINALG_NYSTROMKERNELAPPROXIMATION_H


#include <iostream>
#include "GlobalParameters.h"
#include "LibSVMDataModel.h"
#include "SharkSVM.h"
#include "Algorithms/KMedoids.h"
#include "LinAlg/LowRankApproximation.h"
#include "LinAlg/NystromFeatureTransform.h"


#include <shark/Algorithms/KMeans.h>
#include <shark/Algorithms/Trainers/AbstractTrainer.h>
#include <shark/Core/IParameterizable.h>
#include <shark/LinAlg/KernelMatrix.h>
#include <shark/LinAlg/PartlyPrecomputedMatrix.h>
#include <shark/LinAlg/svd.h>
#include <shark/LinAlg/eigenvalues.h>
#include <shark/Models/Kernels/KernelExpansion.h>
#include <shark/Models/Kernels/KernelHelpers.h>
#include <shark/ObjectiveFunctions/KernelBasisDistance.h>
#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>



namespace shark {

    ///
    /// \brief Generic stochastic gradient descent training for kernel-based models.
    ///
    /// Given a differentiable loss function L(f, y) for classification
    /// this trainer solves the regularized risk minimization problem
    /// \f[
    ///     \min \frac{1}{2} \sum_j \|w_j\|^2 + C \sum_i L(y_i, f(x_i)),
    /// \f]

    ///
    /// \par

    template <class InputType, class CacheType = float>
    class NystromKernelApproximation : public LowRankApproximation<InputType> {
        public:
            typedef AbstractKernelFunction<InputType> KernelType;
            typedef typename InputType::value_type InputValueType;
            typedef blas::matrix<InputValueType, blas::row_major > MatrixType;

            typedef CacheType QpFloatType;
            typedef KernelMatrix<InputType, QpFloatType> KernelMatrixType;
            typedef PartlyPrecomputedMatrix< KernelMatrixType > PartlyPrecomputedMatrixType;


            enum LandmarkSelectionStrategy {RANDOM, KMEANS, KMEDOIDS};


            /// \brief Constructor
            ///
            /// \param  kernel          kernel function to use for training and prediction
            /// \param  loss            (sub-)differentiable loss function
            /// \param  C               regularization parameter - always the 'true' value of C, even when unconstrained is set
            /// \param  offset          whether to train with offset/bias parameter or not
            /// \param  unconstrained   when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver?
            NystromKernelApproximation (KernelType* kernel, size_t nComponents) :
                LowRankApproximation<InputType> (kernel, nComponents) {
                m_landmarkSelectionStrategy = KMEANS;
                m_kMeansIterations = 150;
                m_cacheSize = 0x4000000;
            }


            void setLandmarkSelectionStrategy (LandmarkSelectionStrategy landmarkSelectionStrategy) {
                m_landmarkSelectionStrategy = landmarkSelectionStrategy;
            }


            LandmarkSelectionStrategy landmarkSelectionStrategy() {
                return m_landmarkSelectionStrategy;
            }


            void setKMeansIterations (size_t kMeansIterations) {
                m_kMeansIterations = kMeansIterations;
            }


            size_t KMeansIterations() {
                return m_kMeansIterations;
            }


            /// return current cachesize
            double cacheSize() const {
                return m_cacheSize;
            }


            void setCacheSize (std::size_t size) {
                m_cacheSize = size;
            }


            Data<InputType> landmarks() {
                return m_landmarks.centroids();
            }
            
            
            MatrixType transformationMatrix() {
                return m_nystromMatrix;
            }
            
            
            void setTransformationMatrix (MatrixType matrix) {
                m_nystromMatrix = matrix;
            }


            
            /// adapt the transformation to the given dataset
            /// \param[in]      dataset     dataset to adapt to
            /// \note               random fourier features do not adapt to the data, only the dimension matters.
            void adaptToData (UnlabeledData<InputType> dataset, size_t seed = 42)  {
                // seed random generator
                Rng::seed (seed);

                BOOST_LOG_TRIVIAL (trace) << "Adapting Nystrom to given data.";

                // to generate the landmarks, we have different options
                BOOST_LOG_TRIVIAL (debug) << "Creating " << this->m_nComponents << " landmarks.";

                // create landmarks as centroids
                Centroids landmarks;

                // sanity check: if there are more landmarks than data points, we run into a problem.
                if (dataset.inputs().numberOfElements() < this->m_nComponents) {
                    BOOST_LOG_TRIVIAL (debug) << "Number of data points is " << dataset.inputs().numberOfElements();
                    BOOST_LOG_TRIVIAL (debug) << "But number of centroids to find is " << this->m_nComponents;
                    throw SHARKSVMEXCEPTION ("Cannot apply kMeans with more centroids than data points!");
                }
                
                // initialize depending on selection strategy
                switch (m_landmarkSelectionStrategy) {
                    case RANDOM:
                        BOOST_LOG_TRIVIAL (debug) << "Applying random landmark selection strategy.";
                        // copy selected landmarks, luckily all this already exists
                        landmarks.setCentroids (toDataset (randomSubset (toView (dataset), this->m_nComponents)));
                        break;

                        //WRONG USAGE OF KMEANS?
                        
                    case KMEANS:
                        BOOST_LOG_TRIVIAL (debug) << "Applying k-means landmark selection strategy with " << m_kMeansIterations << " iterations.";
                        // cluster the subsampled data
                        kMeans (dataset.inputs(), this->m_nComponents, landmarks, m_kMeansIterations);
                        break;

                    case KMEDOIDS:
                        BOOST_LOG_TRIVIAL (debug) << "Applying k-medoids landmark selection strategy.";
                        // apply KMediods
                        // put centroid indices into permutation array as this will be used to obtain the landmarks
                        kMedoids (dataset.inputs(), this->m_nComponents, landmarks, m_kMeansIterations);
                        break;

                    default:
                        throw SHARKSVMEXCEPTION ("Unknown landmark selection strategy!");
                }

                // save centroids, they are part of the model
                m_landmarks = landmarks;

                // create kernel matrix
                BOOST_LOG_TRIVIAL (trace) << "Creating kernel matrix in memory.";

                // pre-compute the kernel matrix (may change in the future)
                MatrixType K = calculateRegularizedKernelMatrix (* (this->m_kernel), landmarks.centroids(), 0);

                BOOST_LOG_TRIVIAL (trace) << "Matrix size " << K.size1() << "x" << K.size2();

                //calculate eigenvalue decomposition, copied from BudgetedSVM code
                // >> the idea: A = U*V*U' --> A^(-0.5) = U*V.^(-0.5)
                BOOST_LOG_TRIVIAL (trace) << "Creating transformation matrix.";
                InputType eigenValues;
                MatrixType eigenVectors;
                BOOST_LOG_TRIVIAL (trace) << "Computing Eigenzerlegung.";
                eigensymm (K, eigenVectors, eigenValues);
                BOOST_LOG_TRIVIAL (trace) << "Finished computing Eigenzerlegung.";

                m_nystromMatrix.resize (this->m_nComponents, this->m_nComponents);

                // FIXME to something like inputtype::basetype
                double maxEigenValue = eigenValues [std::max_element (eigenValues.begin(), eigenValues.end()).index()];

                for (size_t i = 0; i < eigenValues.size(); i++) {
                    if (eigenValues[i] > maxEigenValue * 1e-5)
                        m_nystromMatrix (i, i) = 1 / sqrt (eigenValues[i]);
                }

                m_nystromMatrix  = eigenVectors * m_nystromMatrix ;
                BOOST_LOG_TRIVIAL (trace) << "Nystrom matrix size " << m_nystromMatrix .size1() << "x" << m_nystromMatrix .size2();
                BOOST_LOG_TRIVIAL (trace) << "Finished creating transformation matrix.";
            }



            UnlabeledData<InputType> transformData (const UnlabeledData<InputType> dataset) const {
                BOOST_LOG_TRIVIAL (trace) << "Transforming given data into Nystrom feature space.";

                UnlabeledData<InputType> transformedData ;
                transformedData = transform (dataset, NystromFeatureTransform<InputType> (m_nystromMatrix, (this->m_kernel), m_landmarks));
                return (transformedData);
            }



            /// adapt the transformation to the given dataset
            void adaptToData (LabeledData<InputType, unsigned int> dataset) {
                // rewind to unlabeled data case
                adaptToData (dataset.inputs());
            }



            /// adapt the transformation to the given dataset
            LabeledData<InputType, unsigned int> transformData (const LabeledData<InputType, unsigned int> dataset) const {
                // rewind to unlabeled data case
                // by transforming the unlabeled data
                UnlabeledData<InputType> transformedData = transformData (dataset.inputs());

                // splice the labels and the transformed data together
                LabeledData<InputType, unsigned int> transformedLabeledData (transformedData, dataset.labels());

                return (transformedLabeledData);
            }




            /// \brief From INameable: return the class name.
            std::string name() const {
                return "NystromKernelApproximation";
            }



            ///\brief  Returns the vector of hyper-parameters.
            RealVector parameterVector() const {
                size_t kp = (this->m_kernel)->numberOfParameters();
                RealVector ret (kp + 1);

                //                 if (m_unconstrained)
                //                     init (ret) << parameters (m_kernel), log (m_C);
                //                 else
                //                     init (ret) << parameters (m_kernel), m_C;
                //
                return ret;
            }



            ///\brief  Sets the vector of hyper-parameters.
            void setParameterVector (RealVector const &newParameters) {
                size_t kp = (this->m_kernel)->numberOfParameters();
                SHARK_ASSERT (newParameters.size() == kp + 1);
                //            init (newParameters) >> parameters (m_kernel), m_C;

                //                if (m_unconstrained) m_C = exp (m_C);
            }



            ///\brief Returns the number of hyper-parameters.
            size_t numberOfParameters() const {
                return (this->m_kernel)->numberOfParameters() + 1;
            }

        protected:
            // transformation itself
            MatrixType m_nystromMatrix;

            // dimension of original data
            size_t m_dataDimension;

            // landmarks
            LandmarkSelectionStrategy m_landmarkSelectionStrategy;
            Centroids m_landmarks;

            // parameter for kmeans
            size_t m_kMeansIterations;

            // cache
            size_t m_cacheSize;

    };

}
#endif
