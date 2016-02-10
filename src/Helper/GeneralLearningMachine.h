//===========================================================================
/*!
 *
 *
 * \brief       helper for defining the 'general' learning machine.
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


#ifndef SHARK_GENERALLEARNINGMACHINE_H
#define SHARK_GENERALLEARNINGMACHINE_H

#include "CommandLineParameters.h"
#include "LinAlg/RandomFourierFeatures.h"
#include "Models/ClusteredKernelClassifier.h"

#include "Algorithms/NystromTrainer.h"
#include "LinAlg/NystromKernelApproximation.h"
#include "Models/NystromClassifier.h"

#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/Libsvm.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h> //the used kernel for the SVM
#include <shark/Models/LinearClassifier.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/ObjectiveFunctions/CrossValidationError.h>



namespace shark {


//! \brief .
//!
//! \par
//!
//! \sa


/// this was created just to unclutter the main routine.
/// templates and OO suck, the truth lies in duck-typing.
/// 
class GeneralLearningMachine {
public:
    
    // it needs a parameter object to construct itself
    GeneralLearningMachine (CommandLineParameters &commandLineParameters)
    {
        // copy over
        clp = commandLineParameters;
        
        // some default values
        budgetMaintenanceStrategyClass = NULL;
        kernelTrainer = NULL;
        linearTrainer = NULL;
        nystromTrainer = NULL;
        clusteredKernelTrainer = NULL;
        loss = NULL;
    }

    
    ~GeneralLearningMachine () {
        if (budgetMaintenanceStrategyClass != NULL)
            delete budgetMaintenanceStrategyClass;
        
        if (kernelTrainer != NULL)
            delete kernelTrainer;

        if (linearTrainer != NULL)
            delete linearTrainer;
        
        if (clusteredKernelTrainer != NULL)
            delete clusteredKernelTrainer;

        if (loss != NULL)
            delete loss;
        
        if (nystromTrainer!= NULL)
            delete nystromTrainer;
    }

    
    /// this will read the training data by looking up the file
    /// given in the command line parameters.
    void readTrainingData ()
    {
        // need to read the training file
        BOOST_LOG_TRIVIAL (info) << "Reading training data from " << clp.trainingDataPath;
        
        // for now only sparse data
        SparseDataModel<RealVector> sparseDataHandler;
        
        // reading the data will simultaneously normalize the labels.
        trainingData = sparseDataHandler.importData(clp.trainingDataPath, labelOrder);
        BOOST_LOG_TRIVIAL (info) << "Data has " << trainingData.numberOfElements() << " points, input dimension " << inputDimension (trainingData);
        
        // report label order
        std::vector<int> lb;
        labelOrder.getLabelOrder(lb);
        std::stringstream tmp;
        for (unsigned int t = 0; t < lb.size(); t++) {
            tmp << " " << lb[t] ;
        }
        BOOST_LOG_TRIVIAL (debug) << "Labelorder: " << tmp.str();
    }


    
    void createBSGDTrainer()
    {
        // create pegasos trainer
        if (clp.svmType == SVMTypes::BSGD) {
            
            // ask the factory to produce a maintenance strategy
            budgetMaintenanceStrategyClass = &BudgetMaintenanceStrategyFactory<RealVector>::createBudgetMaintenanceStrategy (clp.budgetMaintenanceStrategy);
            
            if (budgetMaintenanceStrategyClass == NULL) {
                BOOST_LOG_TRIVIAL (error) << "Error occured while creating the budget maintenance strategy. NULL returned.";
                throw SHARKSVMEXCEPTION ("Error occured while creating the budget maintenance strategy. NULL returned.");
            }
            
            KernelBudgetedSGDTrainer<RealVector> *kernelBudgetedSGDtrainer = new KernelBudgetedSGDTrainer<RealVector> (&kernel, &hingeLoss, clp.cost, clp.bias, false, clp.budgetSize, budgetMaintenanceStrategyClass);
            kernelBudgetedSGDtrainer -> setEpochs (clp.epochs);
            kernelBudgetedSGDtrainer -> setMinMargin (clp.minMargin);
            
            // set our kernel classifier to what we have created
            kernelTrainer = kernelBudgetedSGDtrainer;
        }
    }
    

    
    void createPegasosTrainer ()
    {
        // create pegasos trainer
        if (clp.svmType == SVMTypes::Pegasos) {
            KernelSGDTrainer<RealVector> *kernelSGDtrainer = new KernelSGDTrainer<RealVector> (&kernel, &hingeLoss, clp.cost, clp.bias, false, clp.cacheSize);
            kernelSGDtrainer -> setEpochs (clp.epochs);
            
            // set our kernel classifier to what we have created
            kernelTrainer = kernelSGDtrainer;
        }
    }
    
    
    void createSVRGTrainer ()
    {
        // create SVRG trainer
        if (clp.svmType == SVMTypes::SVRG) {
            KernelSVRGTrainer<RealVector> *kernelSVRGTrainer = new KernelSVRGTrainer<RealVector> (&kernel, &hingeLoss, clp.cost, clp.bias, false, clp.updateFrequency);
            kernelSVRGTrainer -> setEpochs (clp.epochs);
            
            // set our kernel classifier to what we have created
            kernelTrainer = kernelSVRGTrainer;
        }
    }
    
    
    void createDCSVMTrainer() 
    {
        // create DCSVM trainer
        if (clp.svmType == SVMTypes::DCSVM) {
            // FIXME: move to parameters
            
            bool earlyPrediction = true;
            size_t nCluster = 10; // rethink hwo to make a parameter out of it
            KernelDCSVMTrainer<RealVector> *kernelDCSVMTrainer = new KernelDCSVMTrainer<RealVector> (&kernel, &hingeLoss, clp.cost, clp.epsilon, clp.bias, nCluster, earlyPrediction, false);
            
            // set our kernel classifier to what we have created
            clusteredKernelTrainer = kernelDCSVMTrainer;
        }
    }
    
    
    
    void createCPATrainer ()
    {
        // create pegasos trainer
        if (clp.svmType == SVMTypes::CPA) {
            KernelCPATrainer<RealVector> *kernelCPATrainer = new KernelCPATrainer<RealVector> (&kernel, &hingeLoss, clp.cost, clp.bias);
            kernelCPATrainer -> setEpochs (clp.epochs);
            
            // set our kernel classifier to what we have created
            kernelTrainer = kernelCPATrainer;
        }
    }

    
    
    void createCSVCTrainer() 
    {
        // create c-svm trainer
        if (clp.svmType == SVMTypes::CSVC) {
            
            // check if we have a binary problem
            // FIXME: for other SVCs, fix this too.
            if (numberOfClasses (trainingData) != 2) {
                BOOST_LOG_TRIVIAL (error) << "Data has " << numberOfClasses (trainingData) << " classes.";
                throw SHARKSVMEXCEPTION ("Only binary data is supported for binary C-SVCs. Please use one of the multi-class solvers for multi-class problem or something else for outlier detection.");
            }
            
            // FIXME: many more switches depending on kernel blabla
            switch (clp.kernelType) {
                case KernelTypes::RBF: {
                    CSvmTrainer<RealVector> *kernelCSVMTrainer = new CSvmTrainer<RealVector> (&kernel, clp.cost, clp.bias);
                    kernelCSVMTrainer->stoppingCondition().minAccuracy = clp.epsilon;
                    kernelCSVMTrainer->setCacheSize (clp.cacheSize);
                    
                    // set our kernel classifier to what we have created
                    kernelTrainer = kernelCSVMTrainer;
                }
                break;
                
                case KernelTypes::LINEAR: {
                    LinearCSvmTrainer<RealVector> *linearCSVMTrainer = new LinearCSvmTrainer<RealVector> (clp.cost, clp.bias);
                    // TODO: any other parameter like cache?
                    
                    // set our linear classifier to what we have created
                    linearTrainer = linearCSVMTrainer;
                }
                break;
                
                default:
                   throw SHARKSVMEXCEPTION ("Kernel Type is not supported currently by C-SVMs!");
            }
        }
    }
    
    
    
    void createFastfoodTrainer() 
    {
        // create fastfood trainer
        if (clp.svmType == SVMTypes::Fastfood) {
            ;
        }
    } 
    
    
    
    void createRandomFourierFeaturesTrainer ()
    {
        // create fastfood trainer
        if (clp.svmType == SVMTypes::RandomFourierFeatures) {
            // create kernel
            switch (clp.kernelType) {
                case KernelTypes::RBF: {
                    // create kernel
                    GaussianRbfKernel<RealVector> *kernel = new GaussianRbfKernel<RealVector>(clp.gamma);
                    
                    // create transformation
                    size_t rank = clp.budgetSize;
                    RandomFourierFeatures <RealVector > rff (kernel, rank);
                    rff.adaptToData (trainingData); // can also be empty for RandomFourierFeatures
                    trainingData = rff.transformData (trainingData);
                    
                    // now we need a linear trainer
                    LinearCSvmTrainer<RealVector> *linearCSVMTrainer = new LinearCSvmTrainer<RealVector> (clp.cost, clp.bias);
                    linearTrainer = linearCSVMTrainer;
                }
                break;
                    
                case KernelTypes::LINEAR: {
                    // does this really make  sense?
                    throw SHARKSVMEXCEPTION ("Training linear SVM with RKS features does not make much sense, does it?");
                }
                break;
                
                default:
                    throw SHARKSVMEXCEPTION ("Kernel Type is not supported currently by Incomplete Cholesky Decomposition!");
            }
            
            // create svm flavor
            ;
        }
    }

    
    
    
        
    void createNystromTrainer ()
    {
        // create nystrom trainer
        if (clp.svmType == SVMTypes::Nystrom) {
            // create kernel
            switch (clp.kernelType) {
                case KernelTypes::RBF: {
                    // kernel already exists upon construction of class

                    // create SVM flavor
                    LinearCSvmTrainer<RealVector> *flavoredTrainer = new LinearCSvmTrainer<RealVector> (clp.cost, clp.bias);
                    
                    // create trainer FIXME: budgetsize --> rank
                    NystromTrainer<RealVector> *nT = new NystromTrainer<RealVector>(&kernel, clp.budgetSize);
                    // FIXME: clp.cacheSize
                    nT->setLinearTrainer (flavoredTrainer);
                    nystromTrainer = nT;
                }
                break;
                
                case KernelTypes::LINEAR: {
                    // does this really make  sense?
                    throw SHARKSVMEXCEPTION ("Training linear SVM with Nystrom features does make some, but not so much sense, does it?");
                }
                break;
                
                default:
                    throw SHARKSVMEXCEPTION ("Kernel Type is not supported currently by Incomplete Cholesky Decomposition!");
            }
        }
    }
    
    
    
    void createIncompleteCholeskyTrainer() 
    {
        // create incomplete cholesky trainer
        if (clp.svmType == SVMTypes::IncompleteCholesky) {
            // create kernel
            switch (clp.kernelType) {
                case KernelTypes::RBF: {
                    // create kernel
                    //              GaussianRbfKernel<> *kernel = new GaussianRbfKernel<>(gamma);
                    
                    //            KernelClassifier<RealVector> kernelClassifier;
                    //                IncompleteCholeskyDecomposition->train(kernelClassifier, trainingData);
                    
                }
                break;
                
                case KernelTypes::LINEAR: {
                    // does this really make  sense?
                    throw SHARKSVMEXCEPTION ("Training linear SVM with incomplete Cholesky features does not make much sense, does it?");
                }
                break;
                
                default:
                   throw SHARKSVMEXCEPTION ("Kernel Type is not supported currently by Incomplete Cholesky Decomposition!");
            }
            
            // create svm flavor
            ;
        }
    }
    
    
    void createMCSVMCSTrainer () 
    {
        // create multi-class CS trainer
        if (clp.svmType == SVMTypes::MCSVMCS) {
            switch (clp.kernelType) {
                case KernelTypes::RBF: {
                    McSvmCSTrainer<RealVector> *kernelMCSVMCSTrainer = new McSvmCSTrainer<RealVector> (&kernel, clp.cost, clp.bias);
                    kernelMCSVMCSTrainer ->stoppingCondition().minAccuracy = clp.epsilon;
                    kernelMCSVMCSTrainer ->setCacheSize (clp.cacheSize);
                    
                    // set our kernel classifier to what we have created
                    kernelTrainer = kernelMCSVMCSTrainer;
                }
                break;
                
                case KernelTypes::LINEAR: {
                    LinearMcSvmCSTrainer<RealVector> *linearMCSVMCSTrainer = new LinearMcSvmCSTrainer<RealVector> (clp.cost, clp.bias);
                    linearMCSVMCSTrainer ->stoppingCondition().minAccuracy = clp.epsilon;
                    
                    // set our linear classifier to what we have created
                    linearTrainer = linearMCSVMCSTrainer;
                }
                break;
                
                default:
                   throw SHARKSVMEXCEPTION ("Kernel Type is not supported currently by MultiClass Max Margin Regression SVM!");
            }
        }
    }
    
    

    void createMCSVMOVATrainer () 
    {
        // create multi-class OVA trainer
        if (clp.svmType == SVMTypes::MCSVMOVA) {
            switch (clp.kernelType) {
                case KernelTypes::RBF: {
                    McSvmOVATrainer<RealVector> *kernelMCSVMOVATrainer = new McSvmOVATrainer<RealVector> (&kernel, clp.cost, clp.bias);
                    kernelMCSVMOVATrainer ->stoppingCondition().minAccuracy = clp.epsilon;
                    kernelMCSVMOVATrainer ->setCacheSize (clp.cacheSize);
                    
                    // set our kernel classifier to what we have created
                    kernelTrainer = kernelMCSVMOVATrainer;
                }
                break;
                
                case KernelTypes::LINEAR: {
                    LinearMcSvmOVATrainer<RealVector> *linearMCSVMOVATrainer = new LinearMcSvmOVATrainer<RealVector> (clp.cost, clp.bias);
                    linearMCSVMOVATrainer ->stoppingCondition().minAccuracy = clp.epsilon;
                    
                    // set our linear classifier to what we have created
                    linearTrainer = linearMCSVMOVATrainer;
                }
                break;
                
                default:
                   throw SHARKSVMEXCEPTION ("Kernel Type is not supported currently by MultiClass Max Margin Regression SVM!");
            }
        }
    }
    
    
    
    void createMCSVMLLWTrainer () 
    {
        // create multi-class LLW trainer
        if (clp.svmType == SVMTypes::MCSVMLLW) {
            // LLW does not support any bias so we turn it off 
            if (clp.bias == true)
            {
                BOOST_LOG_TRIVIAL (error) << "Bias cannot be activated with LLW type multi-class classification. Will turn it off..";
                clp.bias = false;
            }
            
            switch (clp.kernelType) {
                case KernelTypes::RBF: {
                    McSvmLLWTrainer<RealVector> *kernelMCSVMLLWTrainer = new McSvmLLWTrainer<RealVector> (&kernel, clp.cost);
                    kernelMCSVMLLWTrainer ->stoppingCondition().minAccuracy = clp.epsilon;
                    kernelMCSVMLLWTrainer ->setCacheSize (clp.cacheSize);
                    
                    // set our kernel classifier to what we have created
                    kernelTrainer = kernelMCSVMLLWTrainer;
                }
                break;
                
                case KernelTypes::LINEAR: {
                    LinearMcSvmLLWTrainer<RealVector> *linearMCSVMLLWTrainer = new LinearMcSvmLLWTrainer<RealVector> (clp.cost);
                    linearMCSVMLLWTrainer ->stoppingCondition().minAccuracy = clp.epsilon;
                    
                    // set our linear classifier to what we have created
                    linearTrainer = linearMCSVMLLWTrainer;
                }
                break;
                
                default:
                   throw SHARKSVMEXCEPTION ("Kernel Type is not supported currently by MultiClass Max Margin Regression SVM!");
            }
        }
    }
    
    
    
    void createMCSVMWWTrainer () 
    {
        // create multi-class WW trainer
        if (clp.svmType == SVMTypes::MCSVMWW) {
            switch (clp.kernelType) {
                case KernelTypes::RBF: {
                    McSvmWWTrainer<RealVector> *kernelMCSVMWWTrainer = new McSvmWWTrainer<RealVector> (&kernel, clp.cost, clp.bias);
                    kernelMCSVMWWTrainer ->stoppingCondition().minAccuracy = clp.epsilon;
                    kernelMCSVMWWTrainer ->setCacheSize (clp.cacheSize);
                    
                    // set our kernel classifier to what we have created
                    kernelTrainer = kernelMCSVMWWTrainer;
                }
                break;
                
                case KernelTypes::LINEAR: {
                    LinearMcSvmWWTrainer<RealVector> *linearMCSVMWWTrainer = new LinearMcSvmWWTrainer<RealVector> (clp.cost, clp.bias);
                    linearMCSVMWWTrainer ->stoppingCondition().minAccuracy = clp.epsilon;
                    
                    // set our linear classifier to what we have created
                    linearTrainer = linearMCSVMWWTrainer;
                }
                break;
                
                default:
                   throw SHARKSVMEXCEPTION ("Kernel Type is not supported currently by MultiClass Max Margin Regression SVM!");
            }
        }
    }
    
    
    void createMCSVMMMRTrainer () 
    {
        // create multi-class MMR trainer
        if (clp.svmType == SVMTypes::MCSVMMMR) {
            switch (clp.kernelType) {
                case KernelTypes::RBF: {
                    McSvmMMRTrainer<RealVector> *kernelMCSVMMMRTrainer = new McSvmMMRTrainer<RealVector> (&kernel, clp.cost, clp.bias);
                    kernelMCSVMMMRTrainer ->stoppingCondition().minAccuracy = clp.epsilon;
                    kernelMCSVMMMRTrainer ->setCacheSize (clp.cacheSize);
                    
                    // set our kernel classifier to what we have created
                    kernelTrainer = kernelMCSVMMMRTrainer;
                }
                break;
                
                case KernelTypes::LINEAR: {
                    LinearMcSvmMMRTrainer<RealVector> *linearMCSVMMMRTrainer = new LinearMcSvmMMRTrainer<RealVector> (clp.cost, clp.bias);
                    linearMCSVMMMRTrainer ->stoppingCondition().minAccuracy = clp.epsilon;
                    
                    // set our linear classifier to what we have created
                    linearTrainer = linearMCSVMMMRTrainer;
                }
                break;
                
                default:
                   throw SHARKSVMEXCEPTION ("Kernel Type is not supported currently by MultiClass Max Margin Regression SVM!");
            }
        }
    }        
    
    
    
    void createMCSVMADMTrainer () 
    {    
        // create multi-class ADM trainer
        if (clp.svmType == SVMTypes::MCSVMADM) {
            
            // ADM does not support any bias so we turn it off 
            if (clp.bias == true)
            {
                BOOST_LOG_TRIVIAL (error) << "Bias cannot be activated with ADM type multi-class classification. Will turn it off..";
                clp.bias = false;
            }
            
            switch (clp.kernelType) {
                case KernelTypes::RBF: {
                    McSvmADMTrainer<RealVector> *kernelMCSVMADMTrainer = new McSvmADMTrainer<RealVector> (&kernel, clp.cost);
                    kernelMCSVMADMTrainer ->stoppingCondition().minAccuracy = clp.epsilon;
                    kernelMCSVMADMTrainer ->setCacheSize (clp.cacheSize);
                    
                    // set our kernel classifier to what we have created
                    kernelTrainer = kernelMCSVMADMTrainer;
                }
                break;
                
                case KernelTypes::LINEAR: {
                    LinearMcSvmADMTrainer<RealVector> *linearMCSVMADMTrainer = new LinearMcSvmADMTrainer<RealVector> (clp.cost, clp.bias);
                    linearMCSVMADMTrainer ->stoppingCondition().minAccuracy = clp.epsilon;
                    
                    // set our linear classifier to what we have created
                    linearTrainer = linearMCSVMADMTrainer;
                }
                break;
                
                default:
                   throw SHARKSVMEXCEPTION ("Kernel Type is not supported currently by MultiClass Max Margin Regression SVM!");
            }
        }
    }    
    
    
    
    void createMCSVMATMTrainer () 
    {
        // create multi-class ATM trainer
        if (clp.svmType == SVMTypes::MCSVMATM) {
            switch (clp.kernelType) {
                case KernelTypes::RBF: {
                    McSvmATMTrainer<RealVector> *kernelMCSVMATMTrainer = new McSvmATMTrainer<RealVector> (&kernel, clp.cost, clp.bias);
                    kernelMCSVMATMTrainer ->stoppingCondition().minAccuracy = clp.epsilon;
                    kernelMCSVMATMTrainer ->setCacheSize (clp.cacheSize);
                    
                    // set our kernel classifier to what we have created
                    kernelTrainer = kernelMCSVMATMTrainer;
                }
                break;
                
                case KernelTypes::LINEAR: {
                    LinearMcSvmATMTrainer<RealVector> *linearMCSVMATMTrainer = new LinearMcSvmATMTrainer<RealVector> (clp.cost, clp.bias);
                    linearMCSVMATMTrainer ->stoppingCondition().minAccuracy = clp.epsilon;
                    
                    // set our linear classifier to what we have created
                    linearTrainer = linearMCSVMATMTrainer;
                }
                break;
                
                default:
                   throw SHARKSVMEXCEPTION ("Kernel Type is not supported currently by MultiClass Max Margin Regression SVM!");
            }
        }
    }        
    
    
    
    void createMCSVMATSTrainer () 
    {
        // create multi-class ATS trainer
        if (clp.svmType == SVMTypes::MCSVMATS) {
            switch (clp.kernelType) {
                case KernelTypes::RBF: {
                    McSvmATSTrainer<RealVector> *kernelMCSVMATSTrainer = new McSvmATSTrainer<RealVector> (&kernel, clp.cost, clp.bias);
                    kernelMCSVMATSTrainer ->stoppingCondition().minAccuracy = clp.epsilon;
                    kernelMCSVMATSTrainer ->setCacheSize (clp.cacheSize);
                    
                    // set our kernel classifier to what we have created
                    kernelTrainer = kernelMCSVMATSTrainer;
                }
                break;
                
                case KernelTypes::LINEAR: {
                    LinearMcSvmATSTrainer<RealVector> *linearMCSVMATSTrainer = new LinearMcSvmATSTrainer<RealVector> (clp.cost, clp.bias);
                    linearMCSVMATSTrainer ->stoppingCondition().minAccuracy = clp.epsilon;
                    
                    // set our linear classifier to what we have created
                    linearTrainer = linearMCSVMATSTrainer;
                }
                break;
                
                default:
                   throw SHARKSVMEXCEPTION ("Kernel Type is not supported currently by MultiClass Max Margin Regression SVM!");
            }
        }
    }    
    
    
    
    // create a classifier 
    void createClassifier()
    {
        // we always create some things
        kernel.setGamma(clp.gamma);
        
        // create all trainers. we do not need to check the functions do that themselves
        createBSGDTrainer();
        createPegasosTrainer();
        createSVRGTrainer();
        createDCSVMTrainer();
        createCPATrainer();
        createCSVCTrainer();
        createFastfoodTrainer();
        createRandomFourierFeaturesTrainer();
        createNystromTrainer();
        createIncompleteCholeskyTrainer();
        
        // multi-class SVM trainer
        createMCSVMCSTrainer(); 
        createMCSVMOVATrainer(); 
        createMCSVMLLWTrainer(); 
        createMCSVMWWTrainer(); 
        createMCSVMMMRTrainer(); 
        createMCSVMADMTrainer(); 
        createMCSVMATMTrainer(); 
        createMCSVMATSTrainer(); 
    }        
    

    void internalTraining(LabeledData<RealVector, unsigned int> trainingData)
    {
        // normal training
        if (linearTrainer != NULL) {
            linearTrainer->train (linearClassifier, trainingData);
        }
        
        if (nystromTrainer != NULL) {
            nystromTrainer ->train (nystromClassifier, trainingData);
        }
        
        if (kernelTrainer != NULL) {
            kernelTrainer->train (kernelClassifier, trainingData);
        }
        
        if (clusteredKernelTrainer != NULL) {
            clusteredKernelTrainer->train (clusteredKernelClassifier, trainingData);
        }
    }
    

    Data<unsigned int> internalPredict(Data<RealVector> testData)
    {
        // normal training
        if (linearTrainer != NULL) {
            return linearClassifier (testData);
        }
        
        if (nystromTrainer != NULL) {
            return nystromClassifier (testData);
        }
        
        if (kernelTrainer != NULL) {
            return kernelClassifier (testData);
        }
        
        if (clusteredKernelTrainer != NULL) {
            return clusteredKernelClassifier (testData);
        }
    }
    
    
    // now do training
    void train() 
    {
        // now do training
        BOOST_LOG_TRIVIAL (info) << "Starting training (Depending on the dataset, this might take several hours, days or month)\n";

        // do check if any of the trainers have been initialized
        if ((linearTrainer == NULL) && 
            (nystromTrainer == NULL) && 
            (kernelTrainer == NULL) && 
            (clusteredKernelTrainer == NULL))
            throw SHARKSVMEXCEPTION("Unable to initialize the given SVM type.\nEither this is an internal error, or you have chosen an unsupported set of command options.\nPlease report this to the developers.\n");

        if (clp.crossValidationFolds > 1) {
            BOOST_LOG_TRIVIAL (info) << "Applying Cross Validation";
        
            // randomize data
            if (clp.crossValidationRandomize == true) {
                // hopefully this will not take too much time
                BOOST_LOG_TRIVIAL (info) << "Randomizing data";
                trainingData = toDataset (randomSubset (toView (trainingData), trainingData.numberOfElements()));
            }
            
            // stratified sampling
            if (clp.crossValidationStratifiedSampling == true) {
                throw SHARKSVMEXCEPTION("Stratified Sampling is not yet supported. Sorry.\n");
            }

            // do we need to save the model?
            if (clp.crossValidationSaveModels == true) {
                // create a nice filename for this model
            }
               
            // create CV. as CrossValidationError will not return any kind of object to work with, 
            // we need to do it by ourselves.
            
            // TODO: this is fixed for now.
            ZeroOneLoss<unsigned int> zeroOneLoss;
            
            // create folds 
            CVFolds<ClassificationDataset> folds = createCVSameSizeBalanced(trainingData, clp.crossValidationFolds);
            std::vector<double> cvFoldErrors;
            
            for (size_t f = 0; f < folds.size(); f++) {
                // train on fold
                BOOST_LOG_TRIVIAL (info) << "Training on fold " << f+1 << "/" << folds.size() << "..";
                LabeledData<RealVector, unsigned int> fold =  folds.training(f);
                LabeledData<RealVector, unsigned int> validation =  folds.validation(f);
                internalTraining(fold);
                
                Data<unsigned int> output = internalPredict(validation.inputs());
                
                // we have now a trained classifier, extract all we need
                if (clp.crossValidationSaveModels == true) {
                    std::stringstream tmpS;
                    tmpS << clp.modelDataPath << f+1 << "_" << folds.size() << ".model";
                    std::string currentModelFilename = tmpS.str();
                    BOOST_LOG_TRIVIAL (info) << "Saving model as " << currentModelFilename;
                    saveModel(currentModelFilename);
                }
                
                //  some stats
                solutionProperties ();
                    
                double foldError = zeroOneLoss.eval (validation.labels(), output);
                BOOST_LOG_TRIVIAL (info) << "Fold error:\t" <<  setprecision (6) << foldError << endl;
                cvFoldErrors.push_back (foldError);
            }
            
            // now  we have trained all these CVs, and we need to print out the average error at this point
            BOOST_LOG_TRIVIAL (info) << "Finished Cross Validation.";
            
            // TODO: replace with boost something
            double meanError = 0.0;
            for (int x = 0; x < cvFoldErrors.size(); x++) {
                meanError += cvFoldErrors[x];
            }
            meanError = meanError/cvFoldErrors.size();
            BOOST_LOG_TRIVIAL (info) << "Overall Training Error: " << meanError;
            
        } else {
            // normal training
            internalTraining (trainingData);
        }
    }

    

    void solutionProperties ()
    {
        // get some data, if available
        if (clp.svmType == SVMTypes::CSVC) {
            switch (clp.kernelType) {
                case KernelTypes::RBF: {
                    BOOST_LOG_TRIVIAL (info) << "Needed " << ( (CSvmTrainer<RealVector>*) kernelTrainer)->solutionProperties().seconds << " seconds to reach a dual of "
                    << std::setprecision (16) << ( (CSvmTrainer<RealVector>*) kernelTrainer)->solutionProperties().value;
                }
                break;
                
                case KernelTypes::LINEAR: {
                    BOOST_LOG_TRIVIAL (info) << "Needed " << ( (LinearCSvmTrainer<RealVector>*) linearTrainer)->solutionProperties().seconds << " seconds to reach a dual of "
                    << std::setprecision (16) << ( (LinearCSvmTrainer<RealVector>*) linearTrainer)->solutionProperties().value;
                }
                break;
                
                default:
                    //??;
                    break;
            }
        }
        
        
        // FIXME: also evaluate on trainingset??
    }
    
    
    
    void saveModel() {
        saveModel (clp.modelDataPath);
    }

    
    
    // save model file
    void saveModel(std::string modelFilename) 
    {
        // if bias is used, get value FIXME
        RealVector rho;
        
        if (clp.bias == true) {
            BOOST_LOG_TRIVIAL (debug) << "Size of bias vector " << kernelClassifier.decisionFunction().offset().size();
            
            for (size_t b = 0; b < kernelClassifier.decisionFunction().offset().size(); b++)
                BOOST_LOG_TRIVIAL (debug) << "b[" << b << "]=" << kernelClassifier.decisionFunction().offset() [b];
            
            if (clp.kernelType == KernelTypes::RBF)
                rho = kernelClassifier.decisionFunction().offset();
            if (clp.kernelType == KernelTypes::LINEAR)
                rho = linearClassifier.decisionFunction().offset();
        }
        
        
        // save model file
        if (modelFilename != "") {
            // prepare the model
            Data<RealVector> supportVectors;
            RealMatrix alphas;
            Data<RealVector> landmarks;

            if (clp.svmType == SVMTypes::Nystrom)  {
                supportVectors = nystromClassifier.landmarks();
                alphas = nystromClassifier.transformationMatrix();
            }
            else if (clp.svmType == SVMTypes::LLSVM)  {
            }
            else if (clp.svmType == SVMTypes::RandomFourierFeatures) 
            {
            }
            else { 
                if (clp.kernelType == KernelTypes::RBF) {
                    supportVectors = kernelClassifier.decisionFunction().basis();
                    alphas = kernelClassifier.decisionFunction().alpha();
                } 
                
                if (clp.kernelType == KernelTypes::LINEAR) {
                    alphas = linearClassifier.decisionFunction().matrix();
                }
            }
            
            
            // create model
            SVMDataModel svmModel = SVMDataModelFactory::createFromType (clp.svmType);
            svmModel->dataContainer()->setGamma (clp.gamma);
            svmModel->dataContainer()->setLabelOrder (labelOrder);
            svmModel->dataContainer()->setBias (rho);
            svmModel->dataContainer()->setKernelType (clp.kernelType);
            svmModel->dataContainer()->setSVMType (clp.svmType);
            svmModel->dataContainer()->setAlphas (alphas);
            svmModel->dataContainer()->setSupportVectors (supportVectors);
//            svmModel->dataContainer()->setLandmarks (landmarks);
            svmModel->save (modelFilename);
        }
    }
    

    // finally do some reporting
    void evaluateDataOnTrainingSet ()
    {
        // classify the training data
        
        // finally do some reporting
        Data<unsigned int> output;

        if (clp.svmType == SVMTypes::Nystrom)  {
            output = nystromClassifier (trainingData.inputs());
        }
        else {
            if (clp.kernelType == KernelTypes::RBF)
                output = kernelClassifier (trainingData.inputs());
            if (clp.kernelType == KernelTypes::LINEAR)
                output = linearClassifier (trainingData.inputs());
        }
        
        // evaluate using zero one loss 
        ZeroOneLoss<unsigned int> zeroOneLoss; // 0-1 loss
        double train_error = zeroOneLoss.eval (trainingData.labels(), output);
        BOOST_LOG_TRIVIAL (info) << "Training error:\t" <<  setprecision (6) << train_error << endl;
    }
    
    
public:
    
    // the parameters it works on
    CommandLineParameters clp;

    // a learning machine has data
    LabeledData<RealVector, unsigned int> trainingData;
    
    // the data has some labeling order  we also need to consider
    LabelOrder labelOrder;
    
    // and create extra classes we might probably need
    AbstractBudgetMaintenanceStrategy<RealVector> *budgetMaintenanceStrategyClass;

    // some things for some things
    GaussianRbfKernel<> kernel;
    HingeLoss hingeLoss;

    // create abstract trainer
    AbstractTrainer< KernelClassifier<RealVector> > *kernelTrainer;
    AbstractTrainer< LinearClassifier<RealVector> > *linearTrainer;
    AbstractTrainer< ClusteredKernelClassifier<RealVector> > *clusteredKernelTrainer;
    NystromTrainer<RealVector> *nystromTrainer;
    //FeatureTransformedLinearTrainer<RealVector> *featureTransformedLinearTrainer;
    AbstractLoss< unsigned int, RealVector> *loss;    
    
    LinearClassifier<RealVector> linearClassifier;
    KernelClassifier<RealVector> kernelClassifier;
    // FeatureTransformedLinearClassifier<RealVector> featureTransformedLinearClassifier;
    NystromClassifier<RealVector> nystromClassifier;
    ClusteredKernelClassifier<RealVector> clusteredKernelClassifier;
};




}

#endif

