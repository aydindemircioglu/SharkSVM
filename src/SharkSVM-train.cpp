//===========================================================================
/*!
 *
 *
 * \brief       Shark SVM training wrapper for SVMs
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


#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/singleton.hpp>
#ifndef REPLACE_BOOST_LOG
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/console.hpp>
#endif

#include <iostream>
#include <string>

#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Algorithms/Trainers/McSvmCSTrainer.h>
#include <shark/Algorithms/Trainers/McSvmLLWTrainer.h>
#include <shark/Algorithms/Trainers/McSvmMMRTrainer.h>
#include <shark/Algorithms/Trainers/McSvmOVATrainer.h>
#include <shark/Algorithms/Trainers/McSvmWWTrainer.h>
#include <shark/Algorithms/Trainers/McSvmADMTrainer.h>
#include <shark/Algorithms/Trainers/McSvmATMTrainer.h>
#include <shark/Algorithms/Trainers/McSvmATSTrainer.h>

#include <shark/Data/Dataset.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/LinearClassifier.h>
#include <shark/ObjectiveFunctions/Loss/HingeLoss.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>

#include "SharkSVM.h"
#include "SharkSVMConfig.h"

#include "CommandLineParameters.h"
#include "GeneralLearningMachine.h"

#include "GlobalParameters.h"

using namespace shark;
using namespace std;



int main (int argc, char** argv) {
    // setup boost log
#ifndef REPLACE_BOOST_LOG
    boost::log::add_console_log (std::cout, boost::log::keywords::format = "%Message%");
#endif

    // hello world!
    BOOST_LOG_TRIVIAL (info) << "SharkSVM Train v" << SHARKSVM_VERSION_MAJOR << "." << SHARKSVM_VERSION_MINOR << "." <<    SHARKSVM_VERSION_PATCH << " -- " << SHARKSVM_BUILD_TYPE << ".";
    BOOST_LOG_TRIVIAL (info) << "Copyright 1995-2014 Shark Development Team" << std::endl;

    // parameter
    try {
        std::string appName = boost::filesystem::basename (argv[0]);

        // create a (strange) parameter object and let it parse.
        CommandLineParameters parameter;
        int status = parameter.interprete (argc, argv);

        if (status != ErrorCodes::SUCCESS)
            throw SHARKSVMEXCEPTION ("Error while parsing command line");

        // read data
        GeneralLearningMachine learningMachine (parameter);
        learningMachine.readTrainingData();

        // there are some parameter we can only set automatically
        // after seeing the data, e.g. a default gamma depends on the
        // distances of the data
        parameter.adapt (learningMachine.trainingData);

        // create a classifier
        learningMachine.createClassifier();

        // now do training
        learningMachine.train();

        // save model file
        learningMachine.saveModel();

        // finally do some reporting
        learningMachine.evaluateDataOnTrainingSet ();
    } catch (boost::program_options::required_option& e) {
        BOOST_LOG_TRIVIAL (error) << e.what() << std::endl << std::endl;
        return ErrorCodes::ERROR_IN_COMMAND_LINE;
    } catch (boost::program_options::error& e) {
        BOOST_LOG_TRIVIAL (error) << e.what() << std::endl << std::endl;
//        BOOST_LOG_TRIVIAL (error) << desc << std::endl;
        return ErrorCodes::ERROR_IN_COMMAND_LINE;
    } catch (std::exception& e) {
        std::cerr << "Unhandled Exception occured." << std::endl << e.what() << ", application will now exit" << std::endl;
        return ErrorCodes::ERROR_UNHANDLED_EXCEPTION;
    }

    return ErrorCodes::SUCCESS;
}

