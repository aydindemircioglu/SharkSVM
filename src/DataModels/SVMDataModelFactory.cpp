//===========================================================================
/*!
 *
 *
 * \brief       Factory for SVM data models
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


#include <shark/Data/Dataset.h>
#include <shark/Data/Libsvm.h>

#include "SharkSVM.h"

#include "SVMDataModelFactory.h"
#include "BudgetedSVMDataModel.h"
#include "LibSVMDataModel.h"
#include "LocalSVMDataModel.h"
#include "LLSVMDataModel.h"
#include "SVMLightDataModel.h"

#include <boost/regex.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/spirit/include/qi.hpp>



namespace shark {

    SVMDataModel SVMDataModelFactory::createFromFile (std::string filePath) {
        // open file for input
        std::ifstream stream (filePath.c_str());

        if (!stream.good())
            throw SHARKSVMEXCEPTION ("[LibSVMDataModel::load] failed to open file for input");

        SVMDataModel model;

        // check for a 'keyword', that is somewhat unique within the header of 24 lines
        BOOST_LOG_TRIVIAL (info) << "Trying to determine mode fileformat...";

        for (size_t i = 0; i < 24; i++) {
            std::string line;
            std::getline (stream, line);
            BOOST_LOG_TRIVIAL (trace) << line;

            // no header has empty lines.
            if (line.empty()) break;

            boost::cmatch matchedStrings;

            // check for BSGD/LLSVM
            boost::regex BSGDExpression ("KERNEL_GAMMA_PARAM:");

            if (boost::regex_search (line.c_str(), matchedStrings, BSGDExpression)) {
                // need to check for LLSVM, as LLSVM is a different beast..
                BOOST_LOG_TRIVIAL (info) << "Found BudgetedSVM Model";
                model = SVMDataModel (new BudgetedSVMDataModel);
                break;
            }

            // check for SVMLight
            boost::regex SVMLightExpression ("SVM-light");

            if (boost::regex_search (line.c_str(), matchedStrings, SVMLightExpression)) {
                BOOST_LOG_TRIVIAL (info) << "Found SVMLight Model";
                model = SVMDataModel (new SVMLightDataModel);
                break;
            }

            // check for LibSVM
            boost::regex LibSVMExpression ("total_sv");

            if (boost::regex_search (line.c_str(), matchedStrings, LibSVMExpression)) {
                BOOST_LOG_TRIVIAL (info) << "Found LibSVM Model";
                model = SVMDataModel (new LibSVMDataModel);
                break;
            }

        }


        return model;
    };

    
    
    SVMDataModel SVMDataModelFactory::createFromString (std::string modelName)
    {
        BOOST_LOG_TRIVIAL (debug) << "SVM DataModelFactory: creating model for given name " << modelName;
        
        SVMDataModel model;
        if (modelName == "LibSVM") {
            BOOST_LOG_TRIVIAL (debug) << "SVM DataModelFactory: creating LibSVM model.";
            model = SVMDataModel (new LibSVMDataModel);
            return (model);
        }
        
        if (modelName == "LocalSVM") {
            BOOST_LOG_TRIVIAL (debug) << "SVM DataModelFactory: creating LocalSVM model.";
            model = SVMDataModel (new LocalSVMDataModel);
            return (model);
        }
        
        if (modelName == "LLSVM") {
            BOOST_LOG_TRIVIAL (debug) << "SVM DataModelFactory: creating LLSVM model.";
            model = SVMDataModel (new LLSVMDataModel);
            return (model);
        }
        
        // unknown strategy, need to throw something
        throw (SHARKSVMEXCEPTION ("SVMDataModelFactory: Unknown model name!"));
    };

    

    SVMDataModel  SVMDataModelFactory::createFromType (int svmType) 
    {
        BOOST_LOG_TRIVIAL (debug) << "SVM DataModelFactory: creating model for given type " << svmType;
        
        SVMDataModel model;
        if ((svmType == SVMTypes::CSVC) || (svmType == SVMTypes::Pegasos) ||
            (svmType == SVMTypes::MCSVMOVA) || (svmType == SVMTypes::MCSVMCS) || 
            (svmType == SVMTypes::MCSVMWW) || (svmType == SVMTypes::MCSVMLLW) || 
            (svmType == SVMTypes::MCSVMMMR) || (svmType == SVMTypes::MCSVMADM) || 
            (svmType == SVMTypes::MCSVMATM) || (svmType == SVMTypes::MCSVMATS) ||
            (svmType  == SVMTypes::LARANK) ) {
            BOOST_LOG_TRIVIAL (debug) << "factory: creating LibSVM model.";
            model = SVMDataModel (new LibSVMDataModel);
            return (model);
        }
        
        if ((svmType == SVMTypes::DCSVM)) {
            BOOST_LOG_TRIVIAL (debug) << "factory: creating LocalSVM model.";
            model = SVMDataModel (new LocalSVMDataModel);
            return (model);
        }
        
        if ((svmType == SVMTypes::Nystrom) || (svmType == SVMTypes::LLSVM)) {
            BOOST_LOG_TRIVIAL (debug) << "factory: creating LLSVM model.";
            model = SVMDataModel (new LLSVMDataModel);
            return (model);
        }
        
        // unknown strategy, need to throw something
        throw (SHARKSVMEXCEPTION ("SVMDataModelFactory: Unknown model name!"));
        
    }
    
    
}

