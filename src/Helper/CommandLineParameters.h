//===========================================================================
/*!
 *
 *
 * \brief       helper for the parameters coming from the command line
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


#ifndef SHARK_COMMANDLINEPARAMETER_H
#define SHARK_COMMANDLINEPARAMETER_H

#include <limits>

#include <shark/Data/Dataset.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Data/Libsvm.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h> //the used kernel for the SVM
#include <shark/Models/LinearClassifier.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>


#include "Data/LabelOrder.h"
#include "Data/SparseData.h"

#include "AbstractSVMDataModel.h"
#include "GlobalParameters.h"
//#include "IncompleteCholeskyDecomposition.h"
#include "Budgeted/KernelBudgetedSGDTrainer.h"
#include "Budgeted/BudgetMaintenanceStrategyFactory.h"
#include "KernelCPATrainer.h"
#include "KernelDCSVMTrainer.h"
#include "KernelSGDTrainer.h"
#include "KernelSVRGTrainer.h"
#include "LibSVMDataModel.h"
#include "SVMDataModelFactory.h"


using namespace std;


namespace shark {


//! \brief .
//!
//! \par
//!
//! \sa



    
class CommandLineParameters 
{
public:
    
    CommandLineParameters ()
    {
        // default parameters
        
        // control parameter
        crossValidationFolds = -1;
        crossValidationRandomize = true;
        crossValidationStratifiedSampling = false;
        
        // paths
        trainingDataPath = "";
        modelDataPath = "";
        modelName = "";
        
        // hyper-parameter
        gamma = numeric_limits<double>::infinity();
        cost = 0.0f;
        epochs = 1;
        
        // meta-parameter
        svmType = SVMTypes::CSVC;
        kernelType = KernelTypes::RBF;
        cacheSize = 1024; // in MB
        svmFlavor = SVMTypes::ACF;
        
        // optimization parameter
        epsilon = 0.001;
        bias = false;
        
        // budget parameter
        budgetMaintenanceStrategy = BudgetMaintenanceStrategy ::MERGE;
        budgetSize = 500;
        minMargin = 1.0f;
        updateFrequency = 0; // for SVR
        
        // control parameter
        modelPath = ".";
        wallTime = -1;
        saveTime = -1;
    }

    
    
    int interprete (int argc, char** argv) 
    {
        namespace po = boost::program_options;
        po::options_description desc ("Options");
        desc.add_options()

        (",s", po::value<int>()->default_value (-1), 
         "svm_type : set type of SVM (default 0)\n"
         "        0 -- C-SVC\n"        
         "        11 -- Pegasos\n"
         "        12 -- Budgeted SGD\n"
         "        13 -- Cutting Plane\n"
         "        14 -- Random Kitchen Sinks\n"
         "        15 -- Nystrom\n"
         "        16 -- Fastfood\n"
         "        17 -- Incomplete Cholesky\n"
         "        18 -- LibLINEAR\n"
         "        19 -- Acceleared Coordinate Frequency\n"
         "        20 -- Stochastic Variance Reduced Gradient\n"
         "        21 -- Divide-and-Conquer SVM\n"
         "        100 -- MCSVMOVA\n"
         "        101 -- MCSVMCS\n"
         "        102 -- MCSVMWW\n"
         "        103 -- MCSVMLLW\n"
         "        104 -- MCSVMMMR")
        (",t", po::value<int>()->default_value (KernelTypes::RBF), "kernel_type : set type of kernel function (default 2)\n        0 -- linear: u'*v\n        2 -- radial basis function: exp(-gamma*|u-v|^2)")
        (",g", po::value<double>(), "gamma : set gamma in kernel function (default 1/num_features)")
        (",c", po::value<double>()->default_value (1.0), "cost : set the parameter C of C-SVC (default 1)")
        (",e", po::value<double>()->default_value (0.001), "epsilon : set tolerance of termination criterion (default 0.001)")
        (",i", po::value<bool>()->default_value (true), "bias : if set, a bias term will be used (default true)")
        (",m", po::value<size_t>()->default_value (1024), "cachesize : set cache memory size in MB (default 640MB)")
        (",p", po::value<uint32_t>()->default_value (1), "epochs : number of epochs, for Pegasos only (default 3)")

        ("cv", po::value<uint32_t>()->default_value (1), "cross validation: number of folds (default 1=no cross validation)")
        ("cvrand", po::value<bool>()->default_value (true), "cross validation randomize data beforehand (default true)")
        ("cvstratified", po::value<bool>()->default_value (false), "cross validation: apply stratified sampling of folds (default false)")
        ("cvsavemodels", po::value<bool>()->default_value (false), "cross validation: save fold models (default false)")
        
        (",a", po::value<int32_t>()->default_value (-1), "savetime : set time interval in minutes to save the current model (default -1)")
        (",l", po::value<int32_t>()->default_value (-1), "walltime : set maximum walltime in minutes (default -1)")
        (",x", po::value<std::string>()->default_value ("."), "modelpath : path to save walltime models (default '.')")
        ("s", po::value<int32_t>(), "linearsolver : linear solver for incomplete cholesky or nystrom methods (default 0)\n        0 -- LibLinear\n        1 -- Acceleared Frequency (ACF)")
        ("budgetstrategy", po::value<size_t>()->default_value (BudgetMaintenanceStrategy::MERGE), "budget maintenance strategy: set type of maintenance strategy (default 0)\n        0 -- Remove (budget vector with smallest coefficient)\n        1 -- Project \n        2 -- Merge")
        ("budgetsize", po::value<size_t>()->default_value (500), "budgetsize : size of the budget (default 500)")
        ("minmargin", po::value<double>()->default_value (1.0f), "minimum margin : minimum margin violation needed (default 0.0)")
        ("learningrate", po::value<double>()->default_value (1.0), "update frequency: frequency for inner loop of SVRG. If 0, then the frequency is set to twice the training data size (default 0.0)")
        (",v", po::value<int>()->default_value (2), "verbosity : set boost::log verbosity (default 2=info)")
        ("modelformat,", po::value<std::string> (&modelName), "format of model file")
        ("training_set_file,", po::value<std::string> (&trainingDataPath)->required(), "path to training data file")
        ("model_file,", po::value<std::string> (&modelDataPath), "path to model file");
        
        po::positional_options_description positionalOptions;
        positionalOptions.add ("training_set_file", 1);
        positionalOptions.add ("model_file", 1);
        
        po::variables_map vm;

        for (int i = 0; i < argc; i++)
        {
            std::string  argument (argv[i]);
            if ((argument.compare("--help") == 0) || (argument.compare("-help") == 0) || (argument.compare("help") == 0))
            {
                BOOST_LOG_TRIVIAL (info) << "Basic Command Line Parameter App" << std::endl << desc << std::endl;
                exit (ErrorCodes::SUCCESS);
            }
        }
        
        po::store (po::command_line_parser (argc, argv).options (desc).positional (positionalOptions).run(),  vm);
        po::notify (vm);
            
        /** --help option
            */
        if (vm.count ("help")) 
        {
            BOOST_LOG_TRIVIAL (info) << "Basic Command Line Parameter App" << std::endl << desc << std::endl;
            return (ErrorCodes::SUCCESS);
        }
        
        // set log verbosity
        if (vm.count ("-v")) 
        {
            int logLevel = vm["-v"].as<int>();
            
#ifndef REPLACE_BOOST_LOG
            boost::log::core::get()->set_filter (boost::log::trivial::severity >= logLevel);
#else
            // FIXME for non-boost logging
#endif 
        }
        
        // svm type
        if (vm.count ("-s")) {
            svmType = vm["-s"].as<int>();
            
            switch (svmType) {
                case SVMTypes::CSVC:
                    svmType = SVMTypes::CSVC;
                    BOOST_LOG_TRIVIAL (info) << "Type of SVM: C-SVC";
                    break;
                    
                case SVMTypes::Pegasos:
                    svmType = SVMTypes::Pegasos;
                    BOOST_LOG_TRIVIAL (info) << "Type of SVM: Pegasos";
                    break;

                case SVMTypes::BSGD:
                    svmType = SVMTypes::BSGD;
                    BOOST_LOG_TRIVIAL (info) << "Type of SVM: Budgeted SGD";
                    break;
                    
                case SVMTypes::CPA:
                    svmType = SVMTypes::CPA;
                    BOOST_LOG_TRIVIAL (info) << "Type of SVM: CPA";
                    break;
                    
                case SVMTypes::RandomFourierFeatures:
                    svmType = SVMTypes::RandomFourierFeatures;
                    BOOST_LOG_TRIVIAL (info) << "Type of SVM: Random Fourier Features";
                    break;
                    
                case SVMTypes::Nystrom:
                    svmType = SVMTypes::Nystrom;
                    BOOST_LOG_TRIVIAL (info) << "Type of SVM: Nystrom";
                    break;

                case SVMTypes::Fastfood:
                    svmType = SVMTypes::Fastfood;
                    BOOST_LOG_TRIVIAL (info) << "Type of SVM: Fastfood";
                    break;

                case SVMTypes::IncompleteCholesky:
                    svmType = SVMTypes::IncompleteCholesky;
                    BOOST_LOG_TRIVIAL (info) << "Type of SVM: Incomplete Cholesky";
                    break;
                    
                    
                    // LIBLINEAR/ACF..
                    
                    
                case SVMTypes::SVRG:
                    svmType = SVMTypes::SVRG;
                    BOOST_LOG_TRIVIAL (info) << "Type of SVM: SVRG";
                    break;
                    
                case SVMTypes::DCSVM:
                    svmType = SVMTypes::DCSVM;
                    BOOST_LOG_TRIVIAL (info) << "Type of SVM: DCSVM";
                    break;
                    
                case SVMTypes::MCSVMOVA:
                    svmType = SVMTypes::MCSVMOVA;
                    BOOST_LOG_TRIVIAL (info) << "Type of SVM: Multi-class One versus All";
                    break;
                    
                case SVMTypes::MCSVMCS:
                    svmType = SVMTypes::MCSVMCS;
                    BOOST_LOG_TRIVIAL (info) << "Type of SVM: Multi-class Crammer & Singer";
                    break;
                    
                case SVMTypes::MCSVMWW:
                    svmType = SVMTypes::MCSVMWW;
                    BOOST_LOG_TRIVIAL (info) << "Type of SVM: Multi-class Weston & Watkins";
                    break;
                    
                case SVMTypes::MCSVMLLW:
                    svmType = SVMTypes::MCSVMLLW;
                    BOOST_LOG_TRIVIAL (info) << "Type of SVM: Multi-class Lee, Lin & Wahba";
                    break;
                    
                case SVMTypes::MCSVMMMR:
                    svmType = SVMTypes::MCSVMMMR;
                    BOOST_LOG_TRIVIAL (info) << "Type of SVM: Multi-class Maximum Margin Regression";
                    break;
                    
                case SVMTypes::MCSVMADM:
                    svmType = SVMTypes::MCSVMADM;
                    BOOST_LOG_TRIVIAL (info) << "Type of SVM: Multi-class Maximum Margin Regression";
                    break;
                    
                case SVMTypes::MCSVMATM:
                    svmType = SVMTypes::MCSVMATM;
                    BOOST_LOG_TRIVIAL (info) << "Type of SVM: Multi-class Maximum Margin Regression";
                    break;
                    
                case SVMTypes::MCSVMATS:
                    svmType = SVMTypes::MCSVMATS;
                    BOOST_LOG_TRIVIAL (info) << "Type of SVM: Multi-class Maximum Margin Regression";
                    break;
                    
                default:
                    BOOST_LOG_TRIVIAL (error) << "Unknown SVM type. Refer to help to see the possible options. ";
                    return ErrorCodes::ERROR_WRONG_SVM_TYPE;
            }
        }
        
        // did we obtained that parameter?
        if (svmType == -1) {
            BOOST_LOG_TRIVIAL (error) << "Unknown SVM type or did not specify any SVM type. Refer to help to see the possible options. ";
            return ErrorCodes::ERROR_WRONG_SVM_TYPE;
        }
        
        // subsolver: linear svm type
        if (vm.count ("s")) {
            // cannot choose flavor if we have not the correct SVM type
            if ( (svmType != SVMTypes::IncompleteCholesky) && (svmType != SVMTypes::Nystrom)) {
                BOOST_LOG_TRIVIAL (error) << "Cannot choose SVM Flavor for given SVM solver. Need Incomplete Cholesky or Nystrom solver.\nRefer to help to see the possible options. ";
                return ErrorCodes::ERROR_WRONG_SVM_TYPE;
            }
            
            svmFlavor = vm["s"].as<int>();
            
            switch (svmFlavor) {
                case SVMFlavors::LIBLINEAR:
                    svmFlavor = SVMTypes::LIBLINEAR;
                    BOOST_LOG_TRIVIAL (info) << "Flavor of SVM solver: LIBLINEAR";
                    break;
                    
                case SVMFlavors::ACF:
                    svmFlavor = SVMTypes::ACF;
                    BOOST_LOG_TRIVIAL (info) << "Flavor of SVM solver: ACF";
                    break;
                    
                default:
                    BOOST_LOG_TRIVIAL (error) << "Unknown SVM flavor. Refer to help to see the possible options. ";
                    return ErrorCodes::ERROR_WRONG_SVM_FLAVOR;
            }
        }
        
        
        // model format
        if (vm.count ("modelformat")) {
            modelName = vm["modelformat"].as<size_t>();
            BOOST_LOG_TRIVIAL (info) << "Saving model in " << modelName << " format.";
        }
        
        
        // updatefrequency
        if (vm.count ("updatefrequency")) {
            updateFrequency= vm["updatefrequency"].as<size_t>();
            if (updateFrequency > 0)
                BOOST_LOG_TRIVIAL (info) << "Update Frequency: " << updateFrequency;
        }
        
        
        // budget strategy
        if (vm.count ("budgetstrategy")) {
            budgetMaintenanceStrategy = vm["budgetstrategy"].as<size_t>();
            
            // check for a valid strategy
            if ( (budgetMaintenanceStrategy != BudgetMaintenanceStrategy::REMOVE) &&
                (budgetMaintenanceStrategy != BudgetMaintenanceStrategy::PROJECT) &&
                (budgetMaintenanceStrategy != BudgetMaintenanceStrategy::MERGE)) {
                BOOST_LOG_TRIVIAL (error) << "Unknown budget maintenance strategy selected. Obtained: " <<
                budgetMaintenanceStrategy << "\nRefer to help to see the possible options. ";
            return ErrorCodes::ERROR_WRONG_BUDGET_STRATEGY;
                }
                
                stringstream tmp;
                
                if (budgetMaintenanceStrategy == BudgetMaintenanceStrategy::REMOVE)
                    tmp << "Remove";
                
                if (budgetMaintenanceStrategy == BudgetMaintenanceStrategy::PROJECT)
                    tmp << "Project";
                
                if (budgetMaintenanceStrategy == BudgetMaintenanceStrategy::MERGE)
                    tmp << "Merge";
                
                BOOST_LOG_TRIVIAL (info) << "Budget maintenance strategy: " << tmp.str();
        }
        
        
        // budget size
        if (vm.count ("budgetsize")) {
            budgetSize = vm["budgetsize"].as<size_t>();
            
            // check for a valid budget size
            if (budgetSize < 1) {
                BOOST_LOG_TRIVIAL (error) << "Budget size is invalid. Obtained: " <<  budgetSize;
                return ErrorCodes::ERROR_WRONG_BUDGET_SIZE;
            }
            
            BOOST_LOG_TRIVIAL (info) << "Budget size: " << budgetSize;
        }
        
        
        // budget strategy
        if (vm.count ("minMargin")) {
            minMargin = vm["minMargin"].as<int>();
            
            // check for a min margin
            if (minMargin < 0.0f) {
                BOOST_LOG_TRIVIAL (error) << "Minimum margin is invalid. Obtained: " <<  minMargin;
                return ErrorCodes::ERROR_WRONG_MIN_MARGIN;
            }
            
            BOOST_LOG_TRIVIAL (info) << "Min Margin: " << minMargin;
        }
        
        
        // kernel type
        if (vm.count ("-t")) {
            kernelType = vm["-t"].as<int>();
            
            switch (kernelType) {
                case KernelTypes::RBF:
                    kernelType = KernelTypes::RBF;
                    BOOST_LOG_TRIVIAL (info) << "Type of Kernel: RBF";
                    break;
                    
                case KernelTypes::LINEAR:
                    kernelType = KernelTypes::LINEAR;
                    BOOST_LOG_TRIVIAL (info) << "Type of Kernel: LINEAR";
                    break;
                    
                default:
                    std::cerr << "Unknown kernel type. Refer to help to see the possible options. ";
                    return ErrorCodes::ERROR_WRONG_KERNEL_TYPE;
            }
        }
        
        // bias term
        if (vm.count ("-i")) {
            bias = vm["-i"].as<bool>();
            BOOST_LOG_TRIVIAL (info) << "Bias-term: " << bias;
        }
        
        // gamma
        if (vm.count ("-g")) {
            gamma = vm["-g"].as<double>();
            BOOST_LOG_TRIVIAL (info) << "Gamma: " << gamma;
        }
        
        // cost
        if (vm.count ("-c")) {
            cost = vm["-c"].as<double>();
            BOOST_LOG_TRIVIAL (info) << "Cost: " << cost;
        }
        
        
        // epochs
        if (vm.count ("-p")) {
            epochs = vm["-p"].as<uint32_t>();
            BOOST_LOG_TRIVIAL (info) << "Epochs: " << epochs;
        }
        
        // modelpath
        if (vm.count ("-x")) {
            modelPath = vm["-x"].as<std::string>();
            BOOST_LOG_TRIVIAL (info) << "Path for saving modelfiles: " << modelPath;
        }
        
        // savetime
        if (vm.count ("-a")) {
            saveTime = vm["-a"].as<int32_t>();
            BOOST_LOG_TRIVIAL (info) << "Savetime: " << saveTime;
        }
        
        // walltime
        if (vm.count ("-l")) {
            wallTime = vm["-l"].as<int32_t>();
            BOOST_LOG_TRIVIAL (info) << "Walltime: " << wallTime;
        }

        
        // cv-folds
        if (vm.count ("cv")) {
            crossValidationFolds = vm["cv"].as<uint32_t>();

            if (crossValidationFolds < 1) {
                std::cerr << "Number of Folds for Cross Validation must be either 1 (no CV) or greater than 1. Refer to help to see the possible options. ";
                return ErrorCodes::ERROR_WRONG_CV_FOLD_NUMBER;
            }
            if (crossValidationFolds > 1) {
                BOOST_LOG_TRIVIAL (info) << "Will apply cross validation with " << crossValidationFolds << " folds.";
            } else {
                BOOST_LOG_TRIVIAL (debug) << "Will not apply cross validation.";
            }
        }
        
        // cv stratified
        if (vm.count ("cvstratified")) {
            crossValidationStratifiedSampling= vm["cvstratified"].as<bool>();
            if (crossValidationStratifiedSampling == true) {
                if (crossValidationFolds <= 1) {
                    BOOST_LOG_TRIVIAL (info) << "Warning: Specified Stratified Sampling for Cross Validation, but Cross Validation is not turned on. This option will be ignored.";
                }
                BOOST_LOG_TRIVIAL (info) << "Will apply Stratified Sampling for Cross Validation.";
            } else {
                if (crossValidationFolds > 1) {
                    BOOST_LOG_TRIVIAL (info) << "Will not apply Stratified Sampling.";
                }
            }
        }
        
        // cv randomization
        if (vm.count ("cvrand")) {
            crossValidationRandomize = vm["cvrand"].as<bool>();
            if (crossValidationRandomize == true) {
                if (crossValidationFolds > 1) {
                    BOOST_LOG_TRIVIAL (info) << "Will randomize data before applying Cross Validation.";
                }
            } else {
                if (crossValidationFolds <= 1) {
                    BOOST_LOG_TRIVIAL (info) << "Warning: Specified Randomization of Data for Cross Validation, but Cross Validation is not turned on. This option will be ignored.";
                }
                BOOST_LOG_TRIVIAL (info) << "Will not randomize data.";
            }
        }
        
        
        // cv randomization
        if (vm.count ("cvsavemodels")) {
            crossValidationSaveModels = vm["cvsavemodels"].as<bool>();
            if (crossValidationSaveModels== false) {
                if (crossValidationFolds > 1) {
                    BOOST_LOG_TRIVIAL (info) << "Will not save intermediate models while applying Cross Validation.";
                }
            } else {
                if (crossValidationFolds <= 1) {
                    BOOST_LOG_TRIVIAL (info) << "Warning: Specified Saving Intermediate Models for Cross Validation, but Cross Validation is not turned on. This option will be ignored.";
                }
                BOOST_LOG_TRIVIAL (info) << "Will save intermediate models while applying Cross Validation.";
            }
        }
        
        if (vm.count ("-e")) {
            epsilon = vm["-e"].as<double>();
            BOOST_LOG_TRIVIAL (info) << "Epsilon: " << epsilon;
        }
        
        // cache size
        if (vm.count ("-m")) {
            // keep computations at float, just like libSVM
            cacheSize = vm["-m"].as<size_t>();
            cacheSize *= 0x10000000/1024;
            BOOST_LOG_TRIVIAL (info) << "CacheSize: " << std::setprecision (2) << (cacheSize * sizeof (float) / 1024 / 1024) << "MB";
        }
        
        // grep other parameters
        trainingDataPath = vm["training_set_file"].as<std::string>();

        if (vm.count ("model_file")) {
            modelDataPath = vm["model_file"].as<std::string>();
        }
        
        // make sure the global parameters land where they should
        boost::serialization::singleton<GlobalParameters>::get_mutable_instance().wallTime = wallTime;
        boost::serialization::singleton<GlobalParameters>::get_mutable_instance().saveTime = saveTime;
        boost::serialization::singleton<GlobalParameters>::get_mutable_instance().modelPath = modelPath;
        
                
        return (ErrorCodes::SUCCESS);
    }
    
    
    // there are some parameter we can only set automatically
    // after seeing the data, e.g. a default gamma depends on the
    // distances of the data
    void adapt( LabeledData<RealVector, unsigned int>  &trainingData)
    {
        // adapt update frequency
        if (updateFrequency == 0) {
            // set it to 2*ell
            updateFrequency = 2 * trainingData.numberOfElements();
            BOOST_LOG_TRIVIAL (info) << "Update Frequency was automatically set to: " << updateFrequency;
        }
        
        
        // if we had no gamma, we need to set it
        if (gamma == numeric_limits<double>::infinity()) {
            gamma = 1. / trainingData.numberOfElements();
            BOOST_LOG_TRIVIAL (info) << "No gamma specified, will compute it from training data.";
            BOOST_LOG_TRIVIAL (info) << "Gamma: " << gamma;
        }
    }
    
    
public:
    
    
    // file parameter
    std::string trainingDataPath;
    std::string modelDataPath;
    std::string modelName;
    
    // hyper-parameter
    double gamma;  
    double cost;  
    uint32_t epochs; 
    
    // meta-parameter
    int svmType;  
    int kernelType; 
    size_t cacheSize;  
    int svmFlavor;
    
    // optimization parameter
    double epsilon; 
    bool bias;
    
    // budget parameter
    size_t budgetMaintenanceStrategy;
    size_t budgetSize;
    double minMargin;
    size_t updateFrequency;
    
    // control parameter
    std::string modelPath;
    int32_t wallTime;
    int32_t saveTime;

    // cv control parameter
    int crossValidationFolds;
    bool crossValidationRandomize;
    bool crossValidationStratifiedSampling;
    bool crossValidationSaveModels;
};




}

#endif

