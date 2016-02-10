//===========================================================================
/*!
 *
 *
 * \brief       Stochastic Variance Reduced Gradient
 *
 *
 *
 *
 * \author      T. Glasmachers, Aydin Demircioglu
 * \date        2013
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


#ifndef SHARK_ALGORITHMS_KERNELSVRGTRAINER_H
#define SHARK_ALGORITHMS_KERNELSVRGTRAINER_H

#include <iostream>
#include "GlobalParameters.h"
#include "LibSVMDataModel.h"
#include "SharkSVM.h"

#include <shark/Algorithms/Trainers/AbstractTrainer.h>
#include <shark/Core/IParameterizable.h>
#include <shark/LinAlg/KernelMatrix.h>
#include <shark/LinAlg/PartlyPrecomputedMatrix.h>
#include <shark/Models/Kernels/KernelExpansion.h>
#include <shark/Models/Kernels/KernelHelpers.h>
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
    /// where i runs over training data, j over classes, and C > 0 is the
    /// regularization parameter.
    ///
    /// \par
    /// This implementation is an adaptation of the PEGASOS algorithm, see the paper
    /// <i>Shalev-Shwartz et al. "Pegasos: Primal estimated sub-gradient solver for SVM." Mathematical Programming 127.1 (2011): 3-30.</i><br/><br/>
    /// However, the (non-essential) projection step is dropped, and the
    /// algorithm is applied to a kernelized model. The resulting
    /// optimization scheme amounts to plain standard stochastic gradient
    /// descent (SGD) with update steps of the form
    /// \f[
    ///     w_j \leftarrow (1 - 1/t) w_j + \frac{C}{t} \frac{\partial L(y_i, f(x_i))}{\partial w_j}
    /// \f]
    /// for random index i. The only notable trick borrowed from that paper
    /// is the representation of the weight vectors in the form
    /// \f[
    ///     w_j = s \cdot \sum_i \alpha_{i,j} k(x_i, \cdot)
    /// \f]
    /// with a scalar factor s (called alphaScale in the code). This enables
    /// scaling with factor (1 - 1/t) in constant time.
    ///
    /// \par
    /// NOTE: Being an SGD-based solver, this algorithm is relatively fast for
    /// differentiable loss functions such as the logistic loss (class CrossEntropy).
    /// It suffers from significantly slower convergence for non-differentiable
    /// losses, e.g., the hinge loss for SVM training.
    ///
    template <class InputType, class CacheType = float>
    class KernelSVRGTrainer : public AbstractTrainer< KernelClassifier<InputType> >, public IParameterizable {
        public:
            typedef AbstractTrainer< KernelExpansion<InputType> > base_type;
            typedef AbstractKernelFunction<InputType> KernelType;
            typedef KernelClassifier<InputType> ClassifierType;
            typedef KernelExpansion<InputType> ModelType;
            typedef AbstractLoss<unsigned int, RealVector> LossType;
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
            /// \param  updateFrequency     update frequency, will be automatically set to 5*n if frequency is set to zero
            KernelSVRGTrainer (KernelType* kernel, const LossType* loss, double C, bool offset, bool unconstrained = false, size_t updateFrequency = 0, double learningRate = 0.001, size_t cacheSize = 0x4000000)
                : m_kernel (kernel)
                , m_loss (loss)
                , m_C (C)
                , m_offset (offset)
                , m_unconstrained (unconstrained)
                , m_epochs (0)
                , m_updateFrequency (updateFrequency)
                , m_learningRate (learningRate)
                , m_cacheSize (cacheSize)
            { }




            /// return current learning rate
            double updateFrequency() const {
                return m_updateFrequency;
            }


            void setUpdateFrequency (size_t updateFrequency) {
                m_updateFrequency = updateFrequency;
            }


            /// return current learning rate
            double learningRate() const {
                return m_learningRate;
            }


            void setLearningRate (double learningRate) {
                m_learningRate = learningRate;
            }


            /// return current cachesize
            double cacheSize() const {
                return m_cacheSize;
            }


            void setCacheSize (std::size_t size) {
                m_cacheSize = size;
            }

            /// \brief From INameable: return the class name.
            std::string name() const {
                return "KernelSGDTrainer";
            }


            void train (ClassifierType &classifier, const LabeledData<InputType, unsigned int> &dataset) {
                std::size_t ell = dataset.numberOfElements();
                unsigned int classes = numberOfClasses (dataset);
                ModelType &model = classifier.decisionFunction();

                model.setStructure (m_kernel, dataset.inputs(), m_offset, classes);

                // pre-compute the kernel matrix (may change in the future)
                // and create linear array of labels
                KernelMatrixType  km (* (this->m_kernel), dataset.inputs());
                PartlyPrecomputedMatrixType  K (&km, m_cacheSize);
                UIntVector y = createBatch (dataset.labels().elements());

                // FIXME: parameter check is nice.

                // preinitialize everything to prevent costly memory allocations in the loop
                RealVector f_b (classes, 0.0);
                RealVector currentDerivative (classes, 0.0);
                RealVector avgStableDerivative (classes, 0.0);
                RealVector stableDerivative (classes, 0.0);

                // SGD loop
                blas::vector<QpFloatType> kernelRow (ell, 0);

                double lambda = 0.5 / (ell * m_C);
                double eta = 1.0;// / (lambda  + ell);
//                double wScale = 1;

                size_t nPreinitSteps = 10;
                
                // some kind of preinit so we do not start with the zero weight
                // now modify the current model by doing several steps of normal stochastic gradient
                for (std::size_t t = 0; t < nPreinitSteps; t++) {
                    // choose randomly some active variable
                    std::size_t b = Rng::discrete (0, ell - 1);

                    // compute prediction with current model
                    f_b.clear();
                    K.row (b, kernelRow);
                    axpy_prod (trans (model.alpha()), kernelRow, f_b, false);

                    if (m_offset)
                        noalias (f_b) += model.offset();

                    // compute the current derivative, i.e. with the model we work with
                    currentDerivative.clear();
                    m_loss->evalDerivative (y[b], f_b, currentDerivative);

                    // update eta
                    eta = 1.0 / (lambda * (t + 1 + ell));

                    // now do a modified, i.e. stochastic variance reduced step
                    noalias (row (model.alpha(), b)) -= eta * (currentDerivative);

                    if (m_offset)
                        noalias (model.offset()) -= eta * (currentDerivative);;
                }
                
                
                eta = 1.0 / (lambda * (1 + ell));
                
                // SVRG has a stable model. Here is it.
                ModelType stableModel;
                stableModel.setStructure (m_kernel, dataset.inputs(), m_offset, classes);
                
                
                // SVRG main loop
                
                for (std::size_t e = 0; e < m_epochs; e++) {
                    // compute the average and full gradient of the stable model.
                    avgStableDerivative.clear();

                    for (std::size_t iter = 0; iter < dataset.inputs().numberOfElements(); iter++) {
                        f_b.clear();
                        K.row (iter, kernelRow);
                        axpy_prod (trans (stableModel.alpha()), kernelRow, f_b, false);

                        if (m_offset)
                            noalias (f_b) += stableModel.offset();

                        currentDerivative.clear();
                        m_loss->evalDerivative (y[iter], f_b, currentDerivative);
                        avgStableDerivative += currentDerivative;
                    }

                    avgStableDerivative *= 1.0 / ell;
                    std::cout << "A: " << avgStableDerivative << "\n";
                    
                    // now modify the current model by doing several steps of normal stochastic gradient
                    for (std::size_t t = 0; t < m_updateFrequency; t++) {
                        // choose randomly some active variable
                        std::size_t b = Rng::discrete (0, ell - 1);

                        // compute prediction with current model
                        f_b.clear();
                        K.row (b, kernelRow);
                        axpy_prod (trans (model.alpha()), kernelRow, f_b, false);

                        if (m_offset)
                            noalias (f_b) += model.offset();

                        // compute the current derivative, i.e. with the model we work with
                        currentDerivative.clear();
                        m_loss->evalDerivative (y[b], f_b, currentDerivative);
                        

                        // compute prediction with stable model
                        f_b.clear();
                        axpy_prod (trans (stableModel.alpha()), kernelRow, f_b, false);

                        if (m_offset)
                            noalias (f_b) += stableModel.offset();

                        // compute the derivative of the stable model
                        stableDerivative.clear();
                        m_loss->evalDerivative (y[b], f_b, stableDerivative);

                        // update eta
                        eta = 1 - 1.0 / (t+1); //(lambda * (t + 1 + ell));
   
                        // same update rule as in svrg original code
                        // wScale *= (1 - lambda * eta);

                        // scale down model
                        model.alpha() *= eta;
                        
                        std::cout << "C: " << currentDerivative << "\n";
                        std::cout << "S: " << stableDerivative << "\n";
                        std::cout << "A: " << avgStableDerivative << "\n";
                        
                       
                        // now do a modified, i.e. stochastic variance reduced step
                        noalias (row (model.alpha(), b)) += 1/(lambda * (t+1))  * (currentDerivative - stableDerivative + avgStableDerivative);

                        if (m_offset)
                            noalias (model.offset()) -= eta * (currentDerivative - stableDerivative + avgStableDerivative);;
                        
//                        model.alpha() *= wScale;
                        
//                         // rescale ws if we in the numerical instable region
//                         wScale = 1.0; // TURN OFF
//                         if (wScale < 0.00000001) {
//                             model.alpha() *= wScale;
//                             if (m_offset)
//                                 model.offset() *= wScale;
//                             wScale = 1;
//                         }
                    }

                    // two options, we use option I for now.
                    // option I: new stable model is the current model.
                    stableModel = model;
                    
                    // option II: take some model inbetween the updates as stable model.
                    // does not make sense probably, as far as i am concerned.
                    // or to be more precise: this is s2gd.
                }

                // sparsify the model
                model.sparsify();
            }


            /// Return the number of training epochs.
            /// A value of 0 indicates that the default of max(10, C) should be used.
            std::size_t epochs() const {
                return m_epochs;
            }

            /// Set the number of training epochs.
            /// A value of 0 indicates that the default of max(10, C) should be used.
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
            const LossType *m_loss;                   ///< pointer to loss function
            double m_C;                               ///< regularization parameter
            bool m_offset;                            ///< should the resulting model have an offset term?
            bool m_unconstrained;                     ///< should C be stored as log(C) as a parameter?
            std::size_t m_epochs;                     ///< number of training epochs (sweeps over the data), or 0 for default = max(10, C)

            // update frequency, the 'm' parameter
            std::size_t m_updateFrequency;

            // learning rate, the 'eta' parameter
            double m_learningRate;

            // cache size to use.
            std::size_t m_cacheSize;
    };


}
#endif
