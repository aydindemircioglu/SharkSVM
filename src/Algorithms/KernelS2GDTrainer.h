//===========================================================================
/*!
 *
 *
 * \brief       Semi-Stochastic Gradient Descent Method
 *
 *
 *
 *
 * \author      T. Glasmachers, Aydin Demircioglu
 * \date        2013
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


#ifndef SHARK_ALGORITHMS_KERNELS2GDTRAINER_H
#define SHARK_ALGORITHMS_KERNELS2GDTRAINER_H

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
/// \par
///
    template <class InputType, class CacheType = float>
    class KernelS2GDTrainer : public AbstractTrainer< KernelClassifier<InputType> >, public IParameterizable {
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
                , m_updateFrequency (updateFrequency)
                , m_learningRate (learningRate)
                , m_epochs (0)
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

                RealMatrix &alpha = model.alpha();

                // pre-compute the kernel matrix (may change in the future)
                // and create linear array of labels
                KernelMatrixType  km (* (this->m_kernel), dataset.inputs());
                PartlyPrecomputedMatrixType  K (&km, m_cacheSize);
                UIntVector y = createBatch (dataset.labels().elements());
                const double lambda = 0.5 / (ell * m_C);

                double alphaScale = 1.0;
                std::size_t iterations;

                if (m_epochs == 0) iterations = std::max (10 * ell, std::size_t (std::ceil (m_C * ell)));
                else iterations = m_epochs * ell;

                // preinitialize everything to prevent costly memory allocations in the loop
                RealVector f_b (classes, 0.0);
                RealVector derivative (classes, 0.0);

                // get global control parameters
                int wallTime = boost::serialization::singleton<GlobalParameters>::get_const_instance().wallTime;
                int saveTime = boost::serialization::singleton<GlobalParameters>::get_const_instance().saveTime;
                std::string modelPath = boost::serialization::singleton<GlobalParameters>::get_const_instance().modelPath;

                // modify time control parameter
                const double timeOffset = saveTime * TIMEFACTOR;

                double startTime = boost::serialization::singleton<GlobalParameters>::get_const_instance().now();
                double nextSaveTime = startTime + timeOffset;
                std::cout << "Time at start:" << startTime << std::endl;

                // SGD loop
                blas::vector<QpFloatType> kernelRow (ell, 0);

                for (std::size_t iter = 0; iter < iterations; iter++) {

                    // active variable
                    std::size_t b = Rng::discrete (0, ell - 1);

                    // learning rate
                    const double eta = 1.0 / (lambda * (iter + ell));

                    // compute prediction
                    f_b.clear();
                    K.row (b, kernelRow);
                    axpy_prod (trans (alpha), kernelRow, f_b, false, alphaScale);

                    if (m_offset) noalias (f_b) += model.offset();

                    // stochastic gradient descent (SGD) step
                    derivative.clear();
                    m_loss->evalDerivative (y[b], f_b, derivative);

                    // alphaScale *= (1.0 - eta * lambda);
                    alphaScale = (ell - 1.0) / (ell + iter);   // numerically more stable

                    noalias (row (alpha, b)) -= (eta / alphaScale) * derivative;

                    if (m_offset) noalias (model.offset()) -= eta * derivative;

                    // check if we need to save the model
                    if (iter % CHECKINTERVAL == 0) {
                        // check for wallTime
                        double t = boost::serialization::singleton<GlobalParameters>::get_const_instance().now();

                        if ( (wallTime != -1) && (t - startTime >= (wallTime - SAFETYWALLTIME) * TIMEFACTOR)) {
                            // force quit
                            std::cout << "Reached wallTime, quitting" << std::endl;
                            iter = iterations;

                            // do not leave quite yet, save model first
                            t = nextSaveTime;
                            saveTime = 1;
                        }

                        // save intermediate model
                        if ( (saveTime != -1) && (t >= nextSaveTime)) {
                            std::cout << "%" << std::endl;

                            double q = boost::serialization::singleton<GlobalParameters>::get_const_instance().now();

                            std::stringstream currentModelFilename;
                            currentModelFilename << modelPath << "/" << std::setprecision (32) << q << ".kernelsgd.model";


                            // ---- FIXME: assume here RBF kernel
                            // copy and scale alphas
                            {
                                Data<RealVector> supportVectors = model.basis();
                                RealMatrix alphas = model.alpha();
                                alphas *= alphaScale;

                                // create model
                                GaussianRbfKernel<RealVector> *rbf = (GaussianRbfKernel<RealVector> *) (m_kernel);
                                double gamma = rbf -> gamma();
                                double rho = 0.0;

                                if (m_offset) rho = model.offset() [0];

                                LibSVMDataModel libSVMModel;
                                libSVMModel.dataContainer()->setGamma (gamma);
                                libSVMModel.dataContainer()->setBias (rho);
                                libSVMModel.dataContainer()->setKernelType (KernelTypes::RBF);
                                libSVMModel.dataContainer()->setSVMType (SVMTypes::Pegasos);
                                libSVMModel.dataContainer()->setAlphas (alphas);
                                libSVMModel.dataContainer()->setSupportVectors (supportVectors);
                                libSVMModel.save (currentModelFilename.str());
                            }

                            //svm_save_model(tmp.str().c_str(), hack_model);
                            // ---

                            // do not count computation of primal to the save times
                            nextSaveTime = boost::serialization::singleton<GlobalParameters>::get_const_instance().now() + timeOffset;
                        }
                    }
                }

                alpha *= alphaScale;

                // model.sparsify();
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
