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

#ifndef SHARK_LINALG_RANDOMFOURIERFEATURES_H
#define SHARK_LINALG_RANDOMFOURIERFEATURES_H


#include <iostream>

#include "GlobalParameters.h"
#include "LibSVMDataModel.h"
#include "SharkSVM.h"
#include "LinAlg/LowRankApproximation.h"

#include <shark/Algorithms/Trainers/AbstractTrainer.h>
#include <shark/Core/IParameterizable.h>
#include <shark/LinAlg/KernelMatrix.h>
#include <shark/LinAlg/PartlyPrecomputedMatrix.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/KernelExpansion.h>
#include <shark/Models/Kernels/KernelHelpers.h>
#include <shark/ObjectiveFunctions/KernelBasisDistance.h>
#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>
#include <shark/Rng/GlobalRng.h>



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
    class RandomFourierFeatures : public LowRankApproximation<InputType>     {
        public:
            typedef AbstractKernelFunction<InputType> KernelType;
            typedef typename InputType::value_type InputValueType;
            typedef blas::matrix<InputValueType, blas::row_major > MatrixType;

            typedef CacheType QpFloatType;
            typedef KernelMatrix<InputType, QpFloatType> KernelMatrixType;



            /// \brief Constructor
            ///
            /// \param  kernel          kernel function to use for training and prediction
            /// \param  loss            (sub-)differentiable loss function
            /// \param  C               regularization parameter - always the 'true' value of C, even when unconstrained is set
            /// \param  offset          whether to train with offset/bias parameter or not
            /// \param  unconstrained   when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver?
            RandomFourierFeatures (KernelType* kernel, size_t nComponents) :
                LowRankApproximation<InputType> (kernel, nComponents) {
                // fixme: disallow anything differnt than gaussian kernels

            }



            /// adapt the transformation to the given dataset
            /// \param[in]      dataset     dataset to adapt to
            /// \note               random fourier features do not adapt to the data, only the dimension matters.
            void adaptToData (UnlabeledData<InputType> dataset, size_t seed = 42)  {
                // seed random generator
                Rng::seed (seed);

                BOOST_LOG_TRIVIAL (trace) << "Adapting RFF to given data.";
                size_t m_dataDimension = dataset.inputs().element (0).size();

                // create random things
                BOOST_LOG_TRIVIAL (trace) << "Creating random matrix of size ." << m_dataDimension << "," << this->m_nComponents;

                // draw random uniform weights as a matrix (with m=0, s=1)
                m_gaussianMatrix (m_dataDimension, this->m_nComponents);

                for (size_t ii = 0; ii < m_gaussianMatrix .size1(); ++ii)
                    for (size_t jj = 0; jj < m_gaussianMatrix .size2(); ++jj)
                        m_gaussianMatrix (ii, jj) = Rng::gauss();

                // draw random offsets
                BOOST_LOG_TRIVIAL (trace) << "Creating random offset vector of size " << this->m_nComponents;
                Uniform< shark::Rng::rng_type> uniform (shark::Rng::globalRng, 0, 2 * M_PI);
                m_offsets (this -> m_nComponents);

                for (size_t ii = 0; ii < m_offsets.size(); ++ii)
                    m_offsets[ii] = uniform();

                // random fourier features are given by taking the pointwise cosine of the matrix
                // cos ().

            }



            template <class myInputType>
            class FourierFeatureTransform {
                public:
                    typedef myInputType result_type;   // do not forget to specify the result type
                    typedef typename myInputType::value_type myInputValueType;
                    typedef blas::matrix<myInputValueType, blas::row_major > myMatrixType;

                    FourierFeatureTransform (myMatrixType randomMatrix, myInputType offset, double normalizationFactor) :
                        m_offset (offset),
                        m_randomMatrix (randomMatrix),
                        m_normalizationFactor (normalizationFactor)
                    {}

                    myInputType operator () (myInputType input) const {

                        myInputType tmp = prod (m_randomMatrix, input) + m_offset;
                        return (m_normalizationFactor * cos (tmp));
                    }

                private:
                    myInputType m_offset;
                    myMatrixType m_randomMatrix;
                    double m_normalizationFactor;
            };



            UnlabeledData<InputType> transformData (const UnlabeledData<InputType> dataset) const {
                BOOST_LOG_TRIVIAL (trace) << "Transforming given data into RFF feature space.";

                // normalization factor
                double normalizationFactor = sqrt (2) / sqrt (m_dataDimension);

                UnlabeledData<InputType> transformedData ;
                transformedData = transform (dataset, FourierFeatureTransform<InputType> (m_gaussianMatrix, m_offsets, normalizationFactor));
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
                return "RandomFourierFeatures";
            }



            ///\brief  Returns the vector of hyper-parameters.
            RealVector parameterVector() const {
                size_t kp = m_kernel->numberOfParameters();
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
                size_t kp = m_kernel->numberOfParameters();
                SHARK_ASSERT (newParameters.size() == kp + 1);
                //            init (newParameters) >> parameters (m_kernel), m_C;

                //                if (m_unconstrained) m_C = exp (m_C);
            }



            ///\brief Returns the number of hyper-parameters.
            size_t numberOfParameters() const {
                return m_kernel->numberOfParameters() + 1;
            }

        protected:
            KernelType *m_kernel;                     ///< pointer to kernel function

            // transformation itself
            MatrixType m_gaussianMatrix;
            InputType m_offsets;

            // dimension of original data
            size_t m_dataDimension;
    };


}
#endif
