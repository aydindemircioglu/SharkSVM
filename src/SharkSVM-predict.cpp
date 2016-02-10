//===========================================================================
/*!
 *
 *
 * \brief       Shark SVM predict wrapper for binary linear and non-linear SVMs
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

// WTF??
#define BOOST_SPIRIT_USE_PHOENIX_V3


#include "boost/filesystem.hpp"
//#include <boost/log/core.hpp>
//#include <boost/log/trivial.hpp>
//#include <boost/log/expressions.hpp>
#include "boost/program_options.hpp"

#include <iostream>
#include <string>
#include <sstream>

#include <shark/Data/Dataset.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Data/Libsvm.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h> //the used kernel for the SVM
#include <shark/Models/LinearClassifier.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>

#include "SharkSVM.h"
#include "Data/LabelOrder.h"
#include "Data/SparseData.h"
#include "DataModels/AbstractSVMDataModel.h"
#include "DataModels/LibSVMDataModel.h"
#include "DataModels/SVMDataModelFactory.h"
#include "Helper/ModelStatistics.h"


using namespace shark;
using namespace std;


#ifndef REPLACE_BOOST_LOG
namespace logging = boost::log;
#endif

void initLogger (int level) {
#ifndef REPLACE_BOOST_LOG
    logging::core::get()->set_filter (logging::trivial::severity >= level);
#endif
}





int main (int argc, char** argv) {
    // setup boost log
#ifndef REPLACE_BOOST_LOG
    boost::log::add_console_log (std::cout, boost::log::keywords::format = "%Message%");
#endif

    // hello world!
    BOOST_LOG_TRIVIAL (info) << "SharkSVM Predict v" << SHARKSVM_VERSION_MAJOR << "." << SHARKSVM_VERSION_MINOR << "." <<  SHARKSVM_VERSION_PATCH << " -- " << SHARKSVM_BUILD_TYPE << ".";
    BOOST_LOG_TRIVIAL (info) << "Copyright 1995-2014 Shark Development Team" << std::endl;

    try {
        std::string appName = boost::filesystem::basename (argv[0]);

        // parameter
        std::string testDataPath;
        std::string modelDataPath;
        std::string predictionScoreFilePath;

        bool probabilityEstimates = false;
        bool statistics = false;
        double cost = -42.0;

        namespace po = boost::program_options;
        po::options_description desc ("Options");
        desc.add_options()
        (",b", po::value<bool>()->default_value (false), "probability_estimates: whether to predict probability estimates (default false); not yet supported.")
        (",s", po::value<bool>()->default_value (false), "compute statistics for given file. makes only sense if given test set is the training set (default false)")
        (",c", po::value<double>(), "compute statistics for given file. makes only sense if given test set is the training set (default false)")
        (",v", po::value<unsigned int>()->default_value (2), "set log level, 0 = trace, debug, info, warning, error, fatal (default 2)")
        ("test_file,", po::value<std::string> (&testDataPath)->required(), "path to test file")
        ("model_file,", po::value<std::string> (&modelDataPath)->required(), "path to model file")
        ("prediction_score_file,", po::value<std::string> (&predictionScoreFilePath), "path to prediction score output file");

        po::positional_options_description positionalOptions;
        positionalOptions.add ("test_file", 1);
        positionalOptions.add ("model_file", 1);
        positionalOptions.add ("prediction_score_file", 1);

        po::variables_map vm;

        try {
            po::store (po::command_line_parser (argc, argv).options (desc).positional (positionalOptions).run(),  vm);
            po::notify (vm);

            /** --help option
             */
            if (vm.count ("help")) {
                std::cout << "Basic Command Line Parameter App" << std::endl
                          << desc << std::endl;
                return ErrorCodes::SUCCESS;
            }

        } catch (boost::program_options::required_option& e) {
            std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
            return ErrorCodes::ERROR_IN_COMMAND_LINE;
        } catch (boost::program_options::error& e) {
            std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
            std::cerr << desc << std::endl;
            return ErrorCodes::ERROR_IN_COMMAND_LINE;
        }


        // get log level
        if (vm.count ("-v")) {
            unsigned int logLevel = vm["-v"].as<unsigned int>();

            // assume the levels are actually integers
            BOOST_LOG_TRIVIAL (info) << "Loglevel: " << logLevel;

#ifndef REPLACE_BOOST_LOG

            if ( (logLevel < logging::trivial::trace) || (logLevel > logging::trivial::fatal))
                throw SHARKSVMEXCEPTION ("Unknown log level!");

#endif

            initLogger (logLevel);
        }

        // get cost first
        if (vm.count ("-c")) {
            cost = vm["-c"].as<double>();

            if (cost <= 0.0)
                throw SHARKSVMEXCEPTION ("Cost cannot be negative!");

            std::cout << "Cost: " << cost << std::endl;
        }

        // statistics
        if (vm.count ("-s")) {
            statistics = vm["-s"].as<bool>();
            std::cout << "Computing statistics: " << statistics << std::endl;

            // check if a valid cost was specified
            if ( (statistics == true) && (cost == -42.0))
                throw SHARKSVMEXCEPTION ("If statistics are specified, cost must be given!");
        }

        // bias term
        if (vm.count ("-b")) {
            // FIXME: not supported yet.
            probabilityEstimates = vm["-b"].as<bool>();
            std::cout << "probabilityEstimates: " << probabilityEstimates << std::endl;
        }

        modelDataPath = vm["model_file"].as<std::string>();
        std::cout << "Reading model data from " << modelDataPath << std::endl;

        // read model
        SVMDataModel svmModel = SVMDataModelFactory::createFromFile (modelDataPath);
        svmModel->load (modelDataPath);



        // read general model properties
        int kernelType = -1;
        svmModel->dataContainer()->getKernelType (kernelType);
        int svmType = -1;
        svmModel->dataContainer()->getSVMType (svmType);

        // FIXME: for now only RBF kernel
        if (kernelType != KernelTypes::RBF)
            throw SHARKSVMEXCEPTION ("Model has an unsupported Kernel Type!");

        if (kernelType == KernelTypes::RBF) {
            // read model parameter
            double gamma = 1.0;
            RealVector rho;
            svmModel->dataContainer()->getGamma (gamma);
            svmModel->dataContainer()->getBias (rho);

            // read model = alphas, SVs
            RealMatrix alphas;
            Data<RealVector> supportVectors;
            LabelOrder labelOrder;
            svmModel->dataContainer()->getAlphas (alphas);
            svmModel->dataContainer()->getSupportVectors (supportVectors);
            svmModel->dataContainer()->getLabelOrder (labelOrder);


            // get first number of classes, are we in a multiclass or binary setting?
            // for sanity check reasons we do this in two different ways
            // count the number of bias terms
            size_t nBiasTerms = rho.size();

            // count the size of the alpha vectors
            size_t nClasses = alphas.size2();

            BOOST_LOG_TRIVIAL (debug) << "Found  " << nClasses << " alpha-coeefficients per SV, and will work with " << nBiasTerms << " bias terms. ";

            // sanity check
            if (nClasses != nBiasTerms)
                throw SHARKSVMEXCEPTION ("Sanity check for number of classes failed.");


            // need to convert the model to something  shark understands
            GaussianRbfKernel<> kernel (gamma);
            KernelExpansion<RealVector> sharkModel (&kernel, supportVectors, true, nClasses);

            // we need as many biases as we have classes
            // but pegasos and similar need two bias terms, though the model might
            // specify only one. in this case we need to extend the size
            RealVector parameters (alphas.size1() *alphas.size2() + nBiasTerms);

            // now copy over all alphas into the parameter vector
            for (size_t c = 0; c < alphas.size2(); c++) {
                for (size_t r = 0; r < alphas.size1(); r++) {
                    parameters[c + r * alphas.size2()] = alphas (r, c);
                }
            }

            BOOST_LOG_TRIVIAL (debug) << alphas.size1() <<  " alpha coefficients, each row has " << alphas.size2() << " entries.";


            // put all bias terms at the end
            for (size_t c = 0; c < rho.size(); c++)
                parameters[parameters.size() - rho.size() + c] = rho[c];

            // add the removed bias term for other class for binary problems
            if (rho.size() == 1)
                parameters[parameters.size() - 1] = -rho[0];

            // have model its parameters.
            sharkModel.setParameterVector (parameters);

            BOOST_LOG_TRIVIAL (info) << "Model has Offset? " << sharkModel.hasOffset();

            if (sharkModel.hasOffset()) {
                BOOST_LOG_TRIVIAL (debug) << "Size of offset " << sharkModel.offset().size();

                for (size_t c = 0; c < nBiasTerms; c++) {
                    BOOST_LOG_TRIVIAL (trace) << "Offset is " << sharkModel.offset() [c];
                }
            }



            // create a classifier with the model
            KernelClassifier<RealVector> kc (sharkModel);


            // read test data
            testDataPath = vm["test_file"].as<std::string>();
            BOOST_LOG_TRIVIAL (info) << std::endl << "Reading test data from " << testDataPath << std::endl;

            LabeledData<RealVector, unsigned int> testData;
            LabelOrder testLabelOrder;
            SparseDataModel<RealVector> sparseDataHandler;
            testData = sparseDataHandler.importData (testDataPath, testLabelOrder, false);

            BOOST_LOG_TRIVIAL (info) << "Data has " << testData.numberOfElements() << " points, input dimension " << inputDimension (testData);
            BOOST_LOG_TRIVIAL (info) << "Number of inputs: " << testData.inputs().numberOfElements();
            BOOST_LOG_TRIVIAL (info) << "Number of labels: " << testData.labels().numberOfElements();


            // map testData labels to our classes.
            // i.e. if we are told the correct label is 6 in ordering "4 1 3 6 2 5",
            // then we must change 6 to 4.
            std::vector<int> tmpLabelOrder;
            labelOrder.getLabelOrder (tmpLabelOrder);

            std::stringstream tmpS;

            for (size_t x = 0; x < tmpLabelOrder.size(); x++)
                tmpS << " " << tmpLabelOrder[x];

            BOOST_LOG_TRIVIAL (debug) << "Label Ordering: " << tmpS.str();

            for (size_t i = 0; i < testData.labels().numberOfElements(); i++) {
                // obtain label, correct and put back
                int currentLabel = testData.labels().element (i);

                std::vector<int>::iterator pos = std::find (tmpLabelOrder.begin(), tmpLabelOrder.end(), currentLabel);

                if (pos ==  tmpLabelOrder.end())
                    throw SHARKSVMEXCEPTION ("Unable to find label of element in the label ordering of the model file!");

                testData.labels().element (i) = pos - tmpLabelOrder.begin ();
            }

            // do prediction
            ZeroOneLoss<unsigned int> loss;
            Data<unsigned int> prediction = kc (testData.inputs());
            double error_rate = loss (testData.labels(), prediction);

            BOOST_LOG_TRIVIAL (info) << std::endl << "Test error rate: " << error_rate << std::endl;


            // compute primal
            if (statistics) {
                BOOST_LOG_TRIVIAL (info) << "";
                double primalValue = ModelStatistics::primalValue (kc, testData, cost);
                BOOST_LOG_TRIVIAL (info) << "Primal: " << setprecision (16) << primalValue;
                double dualValue = ModelStatistics::dualValue (kc, testData);
                BOOST_LOG_TRIVIAL (info) << "Dual: " << setprecision (16) << dualValue;
                double aucValue = ModelStatistics::aucValue (kc, testData);
                BOOST_LOG_TRIVIAL (info) << "AUC: " << setprecision (16) << aucValue;
            }

            // save predictionScore

            // TODO: in this case we have to re-vert the labelordering again.
        }
    } catch (std::exception& e) {
        std::cerr << "Unhandled Exception occured." << std::endl
                  << e.what() << ", application will now exit" << std::endl;
        return ErrorCodes::ERROR_UNHANDLED_EXCEPTION;

    }

    return ErrorCodes::SUCCESS;
}
