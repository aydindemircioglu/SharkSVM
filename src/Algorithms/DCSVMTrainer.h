//===========================================================================
/*!
 *
 *
 * \brief       DCSVM implementation
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


#ifndef SHARK_ALGORITHMS_DCSVMTRAINER_H
#define SHARK_ALGORITHMS_DCSVMTRAINER_H

#include <iostream>
#include "GlobalParameters.h"
#include "LibSVMDataModel.h"
#include "SharkSVM.h"

#include <shark/Algorithms/Trainers/AbstractTrainer.h>
#include <shark/Core/IParameterizable.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h> //the used kernel for the SVM
#include <shark/Models/Kernels/KernelExpansion.h>
#include <shark/Models/Kernels/KernelHelpers.h>
#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>


namespace shark {


/// clustering methods-- or do we provide the algorithms with an clustering object.
    // any linear-time kernel kmeans in shark?? is that an classfier?



///
/// \brief DCSVM implementation. Converted from matlab.

    template <class InputType>
    class DCSVMTrainer : public AbstractTrainer< KernelClassifier<InputType> >, public IParameterizable {
        public:
            typedef AbstractTrainer< KernelExpansion<InputType> > base_type;
            typedef AbstractKernelFunction<InputType> KernelType;
            typedef KernelClassifier<InputType> ClassifierType;
            typedef KernelExpansion<InputType> ModelType;
            typedef AbstractLoss<unsigned int, RealVector> LossType;
            typedef typename ConstProxyReference<typename Batch<InputType>::type const>::type ConstBatchInputReference;

            /// \brief Constructor
            ///
            /// \param  kernel          kernel function to use for training and prediction
            /// \param  loss            (sub-)differentiable loss function
            /// \param  C               regularization parameter - always the 'true' value of C, even when unconstrained is set
            /// \param  offset          whether to train with offset/bias parameter or not
            /// \param  unconstrained   when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver?
            /// \param  earlyStopping    should the algorithm do early stopping?
            /// \param  clusteringMethod    method to use for clustering.
            /// \param  nClusters       number of clusters to use
            DCSVMTrainer (KernelType *kernel, const LossType *loss, double C, bool offset, bool earlyStopping = true, bool unconstrained = false)
                : m_kernel (kernel)
                , m_loss (loss)
                , m_C (C)
                , m_earlyStopping (earlyStopping)
                , m_nClusters (64)
                , m_offset (offset)
                , m_unconstrained (unconstrained)
                , m_epochs (0)
            { }


            /// \brief From INameable: return the class name.
            std::string name() const {
                return "DCSVMTrainer";
            }

            void train (ClassifierType &classifier, const LabeledData<InputType, unsigned int> &dataset) {
                std::size_t ell = dataset.numberOfElements();
                unsigned int classes = numberOfClasses (dataset);
                ModelType &model = classifier.decisionFunction();

                model.setStructure (m_kernel, dataset.inputs(), m_offset, classes);

                RealMatrix &alpha = model.alpha();

                // pre-compute the kernel matrix (may change in the future)
                // and create linear array of labels
                RealMatrix K = calculateRegularizedKernelMatrix (* (this->m_kernel), dataset.inputs(), 0);
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
                for (std::size_t iter = 0; iter < iterations; iter++) {
                    // active variable
                    std::size_t b = Rng::discrete (0, ell - 1);

                    // learning rate
                    const double eta = 1.0 / (lambda * (iter + ell));

                    // compute prediction
                    f_b.clear();
                    axpy_prod (trans (alpha), row (K, b), f_b, false, alphaScale);

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
    };


}
#endif
