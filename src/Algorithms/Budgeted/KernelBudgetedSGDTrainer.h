//===========================================================================
/*!
 *
 *
 * \brief       Budgeted stochastic gradient descent training for kernel-based models.
 *
 * \par This is an implementation  of the BSGD algorithm, developed by 
 *  Wang, Crammer and Vucetic: Breaking the curse of kernelization: 
 *  Budgeted stochastic gradient descent for large-scale SVM training, JMLR 2012.
 * Basically this is pegasos, so something similar to a perceptron. The main
 * difference is that we do restrict the sparsity of the weight vector to a (currently
 * predefined) value. Therefore, whenever this sparsity is reached, we have to
 * decide how to add a new vector to the model, without destroying this
 * sparsity. Several methods have been proposed for this, Wang et al. main
 * insight is that merging two budget vectors (i.e. two vectors in the model).
 * If the first one is searched by norm of its alpha coefficient, the second one
 * can be found by some optimization problem, yielding a roughly optimal pair.
 * This pair can be merged and by doing so the budget has now space for a
 * new vector. Such strategies are called budget maintenance strategies.
 * 
 * \par This implementation owes much to the 'reference' implementation
 * in the BudgetedSVM software.
 *
 *
 * \author      T. Glasmachers, Aydin Demircioglu
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


#ifndef SHARK_ALGORITHMS_KERNELBUDGETEDSGDTRAINER_H
#define SHARK_ALGORITHMS_KERNELBUDGETEDSGDTRAINER_H

#include <iostream>
#include "GlobalParameters.h"
#include "LibSVMDataModel.h"
#include "SharkSVM.h"
#include "Budgeted/AbstractBudgetMaintenanceStrategy.h"

#include <shark/Algorithms/Trainers/AbstractTrainer.h>
#include <shark/Algorithms/KMeans.h>
#include <shark/Core/IParameterizable.h>
#include <shark/LinAlg/KernelMatrix.h>
#include <shark/LinAlg/PartlyPrecomputedMatrix.h>
#include <shark/Models/Kernels/KernelExpansion.h>
#include <shark/Models/Kernels/KernelHelpers.h>
#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>


namespace shark {


    ///
    /// \brief       Budgeted stochastic gradient descent training for kernel-based models.
    ///
    /// \par This is an implementation  of the BSGD algorithm, developed by 
    ///  Wang, Crammer and Vucetic: Breaking the curse of kernelization: 
    ///  Budgeted stochastic gradient descent for large-scale SVM training, JMLR 2012.
    /// Basically this is pegasos, so something similar to a perceptron. The main
    /// difference is that we do restrict the sparsity of the weight vector to a (currently
    /// predefined) value. Therefore, whenever this sparsity is reached, we have to
    /// decide how to add a new vector to the model, without destroying this
    /// sparsity. Several methods have been proposed for this, Wang et al. main
    /// insight is that merging two budget vectors (i.e. two vectors in the model).
    /// If the first one is searched by norm of its alpha coefficient, the second one
    /// can be found by some optimization problem, yielding a roughly optimal pair.
    /// This pair can be merged and by doing so the budget has now space for a
    /// new vector. Such strategies are called budget maintenance strategies.
    /// 
    /// \par This implementation owes much to the 'reference' implementation
    /// in the BudgetedSVM software.
    ///
    /// \par For the documentation of the basic SGD algorithm, please refer to
    /// KernelSGDTrainer.h. Note that we did not take over the special alpha scaling
    /// from that class. Therefore this class is perhaps numerically not as robust as SGD.
    ///
    template <class InputType, class CacheType = float>
    class KernelBudgetedSGDTrainer : public AbstractTrainer< KernelClassifier<InputType> >, public IParameterizable {
        public:
            typedef AbstractTrainer< KernelExpansion<InputType> > base_type;
            typedef AbstractKernelFunction<InputType> KernelType;
            typedef KernelClassifier<InputType> ClassifierType;
            typedef KernelExpansion<InputType> ModelType;
            typedef AbstractLoss<unsigned int, RealVector> LossType;
            typedef typename ConstProxyReference<typename Batch<InputType>::type const>::type ConstBatchInputReference;
            typedef CacheType QpFloatType;
            typedef typename LabeledData<InputType, unsigned int>::element_type ElementType;

            typedef KernelMatrix<InputType, QpFloatType> KernelMatrixType;
            typedef PartlyPrecomputedMatrix< KernelMatrixType > PartlyPrecomputedMatrixType;



            /// preinitialization methods
            enum preInitializationMethod {NONE, RANDOM}; // TODO: add KMEANS

            
            /// data selection strategy
            enum DataSelectionStrategy {UNIFORM, ACF};

            
            // strategy constants
            #define CHANGE_RATE 0.2
            #define PREF_MIN 0.05
            #define PREF_MAX 20.0
            
            
            /// \brief Constructor
            /// Note that there is no cache size involved, as merging vectors will always create new ones,
            /// which makes caching roughly obsolete. 
            /// 
            /// \param[in]  kernel          kernel function to use for training and prediction
            /// \param[in]  loss            (sub-)differentiable loss function
            /// \param[in]  C               regularization parameter - always the 'true' value of C, even when unconstrained is set
            /// \param[in]  offset          whether to train with offset/bias parameter or not
            /// \param[in]  unconstrained   when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver?
            /// \param[in]  budgetSize  size of the budget/model that the final solution will have. Note that it might be smaller though.
            /// \param[in]  budgetMaintenanceStrategy   object that contains the logic for maintaining the budget size.
            /// \param[in]  epochs      number of epochs the SGD solver should run. if zero is given, the size will be the max of 10*datasetsize or C*datasetsize
            /// \param[in]  preInitializationMethod     the method to preinitialize the budget. 
            /// \param[in]  minMargin   the margin every vector has to obey. Usually this is 1.
            ///
            KernelBudgetedSGDTrainer (KernelType* kernel,
                                      const LossType* loss,
                                      double C,
                                      bool offset,
                                      bool unconstrained = false,
                                      size_t budgetSize = 500,
                                      AbstractBudgetMaintenanceStrategy<InputType> *budgetMaintenanceStrategy = NULL,
                                      size_t epochs = 1,
                                      size_t preInitializationMethod = NONE,
                                      size_t dataSelectionStrategy = ACF,
                                      double minMargin = 1.0f)
                : m_kernel (kernel)
                , m_loss (loss)
                , m_C (C)
                , m_offset (offset)
                , m_unconstrained (unconstrained)
                , m_budgetSize (budgetSize)
                , m_budgetMaintenanceStrategy (budgetMaintenanceStrategy)
                , m_epochs (epochs)
                , m_preInitializationMethod (preInitializationMethod)
                , m_dataSelectionStrategy (dataSelectionStrategy)
                , m_minMargin (minMargin){
                    
                // check that the maintenance strategy is not null.
                if (m_budgetMaintenanceStrategy == NULL)
                    throw (SHARKSVMEXCEPTION ("KernelBudgetedSGDTrainer: No budget maintenance strategy provided!"));
            }


            /// get budget size
            /// \return     budget size
            ///
            size_t budgetSize() const {
                return m_budgetSize;
            }


            /// set budget size
            /// \param[in]  budgetSize  size of budget.
            ///
            void setBudgetSize (std::size_t budgetSize) {
                m_budgetSize = budgetSize;
            }


            /// return pointer to the budget maintenance strategy
            /// \return pointer to the budget maintenance strategy.
            ///
            AbstractBudgetMaintenanceStrategy<InputType> *budgetMaintenanceStrategy () const {
                return (m_budgetMaintenanceStrategy);
            }


            /// set budget maintenance strategy
            /// \param[in]  budgetMaintenanceStrategy   set strategy to given object.
            ///
            void setBudgetMaintenanceStrategy (AbstractBudgetMaintenanceStrategy<InputType> *budgetMaintenanceStrategy) {
                m_budgetMaintenanceStrategy = budgetMaintenanceStrategy;
            }


            /// return min margin
            /// \return     current min margin
            ///
            double minMargin() const {
                return m_minMargin;
            }


            /// set min margin
            /// \param[in]  minMargin   new min margin.
            ///
            void setMinMargin (double minMargin) {
                m_minMargin = minMargin;
            }


            /// \brief From INameable: return the class name.
            std::string name() const {
                return "KernelBudgetedSGDTrainer";
            }


            /// Train routine.
            /// \param[in]  classifier      classifier object for the final solution.
            /// \param[in]  dataset     dataset to work with.
            ///
            void train (ClassifierType &classifier, const LabeledData<InputType, unsigned int> &dataset) {

                std::size_t ell = dataset.numberOfElements();
                unsigned int classes = numberOfClasses (dataset);

                // is the budget size larger than reasonable?
                if (m_budgetSize > ell) {
                    // in this case we just set the budgetSize to the given dataset size, so basically
                    // there is an infinite budget.
                    m_budgetSize = ell;
                } 
               
                // we always need one budget vector more than the user specified,
                // as we first have to add any new vector to the budget before applying
                // the maintenance strategy. an alternative would be to keep the budget size
                // correct and test explicitely for the new support vector, but that would
                // create even more hassle on the other side. or one could use a vector of
                // budget vectors instead, but loose the nice framework of kernel expansions.
                // so the last budget vector must always have zero alpha coefficients in
                // the final model. (we do not check for that but roughly assume that in
                // the strategies, e.g. by putting the new vector to the last position in the
                // merge strategy).
                m_budgetSize = m_budgetSize + 1;
                
                // easy access
                UIntVector y = createBatch (dataset.labels().elements());

                // create a preinitialized budget.
                // this is used to initialize the kernelexpansion, we will work with.
                LabeledData<InputType, unsigned int> preinitializedBudgetVectors (m_budgetSize, dataset.element (0));

                // preinit the vectors first
                // we still preinit even for no preinit, as we need the vectors in the
                // constructor of the kernelexpansion. the alphas will be set to zero for none.
                if ( (m_preInitializationMethod == RANDOM) || (m_preInitializationMethod == NONE)) {
                    for (size_t j = 0; j < m_budgetSize; j++) {
                        // choose a random vector
                        std::size_t b = Rng::discrete (0, ell - 1);

                        // copy over the vector
                        preinitializedBudgetVectors.element (j) = dataset.element (b);
                    }
                }

                /*
                // TODO: kmeans initialization
                if (m_preInitializationMethod == KMEANS) {
                    // the negative examples individually. the number of clusters should
                    // then follow the ratio of the classes. then we can set the alphas easily.
                    // TODO: do this multiclass
                    // TODO: maybe Kmedoid makes more sense because of the alphas.
                    // TODO: allow for different maxiters
                    Centroids centroids;
                    size_t maxIterations = 50;
                    kMeans (dataset.inputs(), m_budgetSize, centroids, maxIterations);

                    // copy over to our budget
                    Data<RealVector> const& c = centroids.centroids();

                    for (size_t j = 0; j < m_budgetSize; j++) {
                        preinitializedBudgetVectors.inputs().element (j) = c.element (j);
                        preinitializedBudgetVectors.labels().element (j) = 1; //FIXME
                    }
                }
                */

                // budget is a kernel expansion in its own right
                ModelType &budgetModel = classifier.decisionFunction();
                RealMatrix &budgetAlpha = budgetModel.alpha();
                budgetModel.setStructure (m_kernel, preinitializedBudgetVectors.inputs(), m_offset, classes);


                // variables
                const double lambda = 1.0 / (ell * m_C);
                BOOST_LOG_TRIVIAL (debug) << "Lambda:" << lambda;
                std::size_t iterations;


                // set epoch number
                if (m_epochs == 0)
                    iterations = std::max (10 * ell, std::size_t (std::ceil (m_C * ell)));
                else
                    iterations = m_epochs * ell;


                // set the initial alphas (we do this here, after the array has been initialized by setStructure)
                if (m_preInitializationMethod == RANDOM) {
                    for (size_t j = 0; j < m_budgetSize; j++) {
                        size_t c = preinitializedBudgetVectors.labels().element (j);
                        budgetAlpha (j, c) = 1 / (1 + lambda);
                        budgetAlpha (j, (c + 1) % classes) = -1 / (1 + lambda);
                    }
                }

                
                // init data selection strategy
                std::vector<std::size_t> schedule(ell);

                // example-wise measure of success
                RealVector pref(ell, 1.0);

                // normalization constant
                double prefsum = ell;

                if (m_dataSelectionStrategy == UNIFORM)
                {
                    for (std::size_t i=0; i<ell; i++) 
                        schedule[i] = i;
                }
                
                if (m_dataSelectionStrategy == ACF)
                {
                }
                    
                    
                // gain for ACF
                const double gain_learning_rate = 1.0 / ell;
                double average_gain = 0.0;
                
                // whatever strategy we did use-- the last budget vector needs
                // to be zeroed out, either it was zero anyway (none preinit)
                // or it is the extra budget vector we need for technical reasons
                row(budgetAlpha, m_budgetSize-1) *= 0;
                
                
                // preinitialize everything to prevent costly memory allocations in the loop
                RealVector predictions (classes, 0.0);
                RealVector derivative (classes, 0.0);

                // get global control parameters
                int wallTime = boost::serialization::singleton<GlobalParameters>::get_const_instance().wallTime;
                int saveTime = boost::serialization::singleton<GlobalParameters>::get_const_instance().saveTime;
                std::string modelPath = boost::serialization::singleton<GlobalParameters>::get_const_instance().modelPath;

                // modify time control parameter
                const double timeOffset = saveTime * TIMEFACTOR;

                double startTime = boost::serialization::singleton<GlobalParameters>::get_const_instance().now();
                double nextSaveTime = startTime + timeOffset;
                BOOST_LOG_TRIVIAL (debug) << "Time at start:" << startTime;

                // SGD loop
                std::size_t b = 0;
                
                for (std::size_t iter = 0; iter < iterations; iter++) {
                    // active variable
                    b = Rng::discrete (0, ell - 1);

                    // for smaller datasets instead of choosing randomly a sample
                    // permuting the dataset can be a valid strategy. We do not implement
                    // that here.
                    
                    // compute prediction within the budgeted model
                    // this will compute the predictions for all classes in one step
                    budgetModel.eval (dataset.inputs().element (b), predictions);

                    // now we follow the crammer-singer model as written
                    // in paper (p. 11 top), we compute the scores of the true
                    // class and the runner-up class. for the latter we remove
                    // our true prediction temporarily and redo the argmax.
                    
                    RealVector predictionsCopy = predictions;
                    unsigned int trueClass = y[b];
                    double scoreOfTrueClass = predictions[trueClass];
                    predictions[trueClass] = -std::numeric_limits<double>::infinity();
                    unsigned int runnerupClass = arg_max (predictions);
                    double scoreOfRunnerupClass = predictions[runnerupClass];                    

                    SHARK_ASSERT (trueClass != runnerupClass);
                    
                    // scale alphas
                    budgetModel.alpha() *= ( (long double) (1.0-1.0/(iter+1.0)));
                    
                    // check if there is a margin violation
                    if (scoreOfTrueClass - scoreOfRunnerupClass < m_minMargin ) {
                        // TODO: check if the current vector is already part of our budget

                        // as we do not use the predictions anymore, we use them to push the new alpha values
                        // to the budgeted model
                        predictions.clear();
                        
                        // set the alpha values (see p 11, beta_t^{(i)} formula in wang, crammer, vucetic)
                        // alpha of true class
                        predictions[trueClass] = 1.0/( (long double) (iter+1.0) * lambda);
                        
                        // alpha of runnerup class
                        predictions[runnerupClass] = -1.0/( (long double) (iter + 1.0) * lambda);
                        
                        m_budgetMaintenanceStrategy->addToModel (budgetModel, predictions, dataset.element (b));
                    }

                    // update gain-based preferences
                    if (m_dataSelectionStrategy == ACF)
                    {
                        int epoch = (int) iter/ell;
                        if (epoch == 0) 
                        {
                            //average_gain += gain / (double)ell;
                        }
                        else
                        {      
                            int i = iter % ell;
                            double gain = 1.0;
                            double change = CHANGE_RATE * (gain / average_gain - 1.0);
                            double newpref = std::min(PREF_MAX, std::max(PREF_MIN, pref(i) * std::exp(change)));
                            prefsum += newpref - pref(i);
                            pref(i) = newpref;
                            average_gain = (1.0 - gain_learning_rate) * average_gain + gain_learning_rate * gain;
                        }
                    }
                    
                    
                    // check if we need to save the model
                    if (iter % CHECKINTERVAL == 0) {
                        // check for wallTime
                        double t = boost::serialization::singleton<GlobalParameters>::get_const_instance().now();

                        if ( (wallTime != -1) && (t - startTime >= (wallTime - SAFETYWALLTIME) * TIMEFACTOR)) {
                            // force quit
                            BOOST_LOG_TRIVIAL (info) << "Reached wallTime, quitting";
                            iter = iterations;

                            // TODO: maybe we should not save the final model, only if any intermediate model
                            // was saved.
                            // do not leave quite yet, save model first
                            t = nextSaveTime;
                            saveTime = 1;
                        }

                        // save intermediate model
                        if ( (saveTime != -1) && (t >= nextSaveTime)) {
                            BOOST_LOG_TRIVIAL (debug) << "%";

                            double q = boost::serialization::singleton<GlobalParameters>::get_const_instance().now();

                            std::stringstream currentModelFilename;
                            currentModelFilename << modelPath << "/" << std::setprecision (32) << q << ".kernelsgd.model";


                            // ---- FIXME: assume here RBF kernel
                            // copy and scale alphas
                            {
                                Data<RealVector> supportVectors = budgetModel.basis();
                                RealMatrix alphas = budgetModel.alpha();
//                                alphas *= alphaScale;

                                // create model
                                GaussianRbfKernel<RealVector> *rbf = (GaussianRbfKernel<RealVector> *) (m_kernel);
                                double gamma = rbf -> gamma();
                                RealVector rho; 

                                if (m_offset) rho = budgetModel.offset();

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
                    
                    // every epoch:
                    if ((iter % ell == 0) && (iter > 0)) {
                        // ACF
                        if (m_dataSelectionStrategy == ACF)
                        {
                            // define schedule
                            double psum = prefsum;
                            prefsum = 0.0;
                            std::size_t pos = 0;
                            for (std::size_t i=0; i<ell; i++)
                            {
                                double p = pref(i);
                                double num = (psum < 1e-6) ? ell - pos : std::min((double)(ell - pos), (ell - pos) * p / psum);
                                std::size_t n = (std::size_t)std::floor(num);
                                double prob = num - n;
                                if (Rng::uni() < prob) n++;
                                for (std::size_t j=0; j<n; j++)
                                {
                                    schedule[pos] = i;
                                    pos++;
                                }
                                psum -= p;
                                prefsum += p;
                            }
                            SHARK_ASSERT(pos == ell);
                        }
                    }
                }
                
                double finishTime = boost::serialization::singleton<GlobalParameters>::get_const_instance().now();
                BOOST_LOG_TRIVIAL (debug) << "Time needed to find solution:" << finishTime - startTime;

                // finally we need to get rid of zero supportvectors.
                budgetModel.sparsify();
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

            // budget size
            size_t m_budgetSize;

            // budget maintenance strategy
            AbstractBudgetMaintenanceStrategy<InputType> *m_budgetMaintenanceStrategy;

            std::size_t m_epochs;                     ///< number of training epochs (sweeps over the data), or 0 for default = max(10, C)

            /// method to preinitialize budget
            size_t m_preInitializationMethod;
            
            /// method to select next data point 
            size_t m_dataSelectionStrategy;

            // needed margin below which we update the model, also called beta sometimes
            double m_minMargin;
    };

}
#endif
