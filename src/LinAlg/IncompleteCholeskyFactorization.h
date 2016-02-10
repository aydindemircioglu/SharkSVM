/*!
 *
 *
 * \brief       Incomplete Cholesky Factorization for a Matrix A = LL^T
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

#ifndef SHARK_LINALG_INCOMPLETECHOLESKYFACTORIZATION_H
#define SHARK_LINALG_INCOMPLETECHOLESKYFACTORIZATION_H


#include <iostream>
#include "GlobalParameters.h"
#include "LibSVMDataModel.h"
#include "SharkSVM.h"
#include "LinAlg/LowRankApproximation.h"

#include <shark/Algorithms/Trainers/AbstractTrainer.h>
#include <shark/Core/IParameterizable.h>
#include <shark/LinAlg/KernelMatrix.h>
#include <shark/LinAlg/PartlyPrecomputedMatrix.h>
#include <shark/Models/Kernels/KernelExpansion.h>
#include <shark/Models/Kernels/KernelHelpers.h>
#include <shark/ObjectiveFunctions/KernelBasisDistance.h>
#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>


namespace shark {


    /*!
     *  \brief Incomplete Cholesky decomposition (with Symmetric Pivoting)
     *
     *  Given an \f$ m \times m \f$ symmetric positive definite matrix
     *  \f$A\f$, compute the lower triangular matrix \f$L\f$ such that
     *  \f$A = LL^T \f$.
     *  An exception is thrown if the matrix is not positive definite.
     *  If you suspect the matrix to be positive semi-definite, use
     *  pivotingCholeskyDecomposition instead
     *  The algorithm is given in Figure 2 in Fine, Scheinberg: Efficient SVM Training
     *  Using Low-Rank Kernel Representations, 2001
     *
     *  \param  A \f$ m \times m \f$ matrix, which must be symmetric and positive definite
     *                  (only the lower triangle is accessed)
     *  \param  L \f$ m \times m \f$ matrix, which stores the Cholesky factor. It is a lower
     *                  triangular matrix.
     *  \param  rank    desired rank. L can have a lower rank, but not a higher rank.
     *  \param  epsilon     threshold where to cut off the pivot score
     *  \param  index       array of indicies which columns were selected..
     *  \return none
     *
     *
     */

    template <class InputType>
    class IncompleteCholeskyFactorization: public LowRankApproximation<InputType> {
        public:
            typedef AbstractKernelFunction<InputType> KernelType;
            typedef AbstractTrainer< KernelExpansion<InputType> > base_type;
//            typedef AbstractKernelFunction<InputType> KernelType;
            typedef KernelClassifier<InputType> ClassifierType;
            typedef KernelExpansion<InputType> ModelType;
            typedef AbstractLoss<unsigned int, RealVector> LossType;
            typedef typename ConstProxyReference<typename Batch<InputType>::type const>::type ConstBatchInputReference;
            typedef float QpFloatType;

            typedef KernelMatrix<InputType, QpFloatType> KernelMatrixType;


            /// \brief Constructor
            ///
            /// \param  kernel          kernel function to use for training and prediction
            /// \param  loss            (sub-)differentiable loss function
            /// \param  C               regularization parameter - always the 'true' value of C, even when unconstrained is set
            /// \param  offset          whether to train with offset/bias parameter or not
            /// \param  unconstrained   when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver?
            IncompleteCholeskyFactorization (KernelType* kernel, size_t nComponents) :
                LowRankApproximation<InputType> (kernel, nComponents)
            { }



            void setThreshold (double threshold) {
                m_threshold = threshold;
            }



            /// adapt the transformation to the given dataset
            virtual void adaptToData (UnlabeledData<InputType> dataset, size_t seed = 42)  {

            }



            UnlabeledData<InputType> transformData (const UnlabeledData<InputType> dataset) const {
            }



            /// \brief From INameable: return the class name.
            std::string name() const {
                return "incompleteCholeskyDecomposition";
            }



            void constructLinearization (UnlabeledData<InputType> &dataset) {
                std::size_t ell = dataset.numberOfElements();
                /*
                                // pre-compute the kernel matrix (may change in the future)
                                // and create linear array of labels
                                KernelMatrixType  km (* (this->m_kernel), dataset.inputs());
                                PartlyPrecomputedMatrixType  K (&km, m_cacheSize);
                                UIntVector y = createBatch (dataset.labels().elements());

                                {
                                    size_t n = Q().size1();
                                    ensure_size (G, n, n);
                                    SIZE_CHECK (Q().size1() == Q().size2());

                                    // FIXME: clear result G!

                                    // outer loop
                                    for (size_t i = 0; i < n; i++) {
                                        std::cout << n << "-->";

                                        // pivot scores
                                        for (size_t j = i; j < n; j++) {
                                            std::cout << G() (j, j) << std::endl;
                                            G() (j, j) = Q() (j, j);
                                            std::cout << Q() (j, j) << std::endl;

                                            for (int k = 0; k < i; k++) {
                                                std::cout << "k " << k << ",i" << i << "\n";
                                                G() (j, j)  -= G() (j, k) * G() (j, k);
                                            }
                                        }

                                        // compute sum of pivot elements
                                        // and also find the maximum while we are at it
                                        double sumPivot = 0;
                                        uint32_t jStar = i;
                                        double maxPivot = G() (i, i);

                                        for (size_t j = i; j < n; j++) {
                                            std::cout << "X";
                                            sumPivot += G() (j, j);

                                            if (G() (j, j) > maxPivot) {
                                                jStar = j;
                                                maxPivot = G() (j, j);
                                            }
                                        }

                                        std::cout << "N";


                                        if (maxPivot > epsilon) {
                                            std::cout << "E";

                                            // copy
                                            for (size_t k = i; k < n; k++)
                                                if ( (jStar + k - i) < n)
                                                    G() (k, i) = Q() (jStar + k - i, jStar);

                                            // swap
                                            for (size_t k = 0; k <= i; k++) {
                                                double s = G() (i, k);
                                                G() (i, k) = G() (jStar, k);
                                                G() (jStar, k) = s;
                                            }

                                            // recompute
                                            for (size_t j = 0; j < i; j++) {
                                                for (size_t k = i + 1; k < n; k++)
                                                    G() (k, i) -= G() (k, j) * G() (i, j);
                                            }

                                            // rescale
                                            G() (i, i) = std::sqrt (G() (i, i));

                                            for (size_t k = i + 1; k < n; k++)
                                                G() (k, i) = G() (k, i) / G() (i, i);
                                        } else {
                                            // stop procedure
                                            rank = i - 1;
                                            std::cout << "exit!\n";
                                            return;
                                        }
                                    }

                               }
                */
            }



            ///\brief  Returns the vector of hyper-parameters.
            RealVector parameterVector() const {
                size_t kp = m_kernel->numberOfParameters();
                RealVector ret (kp + 1);

                return ret;
            }



            ///\brief  Sets the vector of hyper-parameters.
            void setParameterVector (RealVector const &newParameters) {
                size_t kp = m_kernel->numberOfParameters();
                SHARK_ASSERT (newParameters.size() == kp + 1);
            }



            ///\brief Returns the number of hyper-parameters.
            size_t numberOfParameters() const {
                return m_kernel->numberOfParameters() + 1;
            }

        protected:
            KernelType *m_kernel;                     ///< pointer to kernel function

            double m_threshold;

            size_t m_rank;

            size_t m_seed;
    };


}
#endif


