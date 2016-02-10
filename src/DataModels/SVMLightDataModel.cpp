//===========================================================================
/*!
 *
 *
 * \brief       SVMLight data model
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


#include <fstream>
#include <iostream>

#include "SharkSVM.h"

#ifndef REPLACE_BOOST_LOG
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#endif

#include <boost/regex.hpp>

#include <shark/Core/Exception.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/Libsvm.h>

#include "AbstractSVMDataModel.h"
#include "SVMLightDataModel.h"



using namespace std;


namespace shark {


    SVMLightDataModel::SVMLightDataModel() {
    }



    SVMLightDataModel::~SVMLightDataModel() {
    }



    void SVMLightDataModel::loadHeader (std::ifstream &modelDataStream) {
        BOOST_LOG_TRIVIAL (info) << "Loading headers for SVMLight DataModel... ";

        // we simply discard the data container we might have had
        container = DataModelContainerPtr (new DataModelContainer());

        // TODO: probably assuming fixed header data order is better than
        // parsing it 'randomly' as done below.

        //unsigned int nr_classes = 0;
        //  unsigned int total_sv = 0;
        //    unsigned int nr_sv = 0;

        // discard first 'header line' = svmlight text
        std::string line;
        std::getline (modelDataStream, line);


        while (modelDataStream) {
            using namespace boost::spirit::qi;

            std::string line;
            std::getline (modelDataStream, line);

            if (line.empty())
                continue;

            std::string::const_iterator first = line.begin();
            std::string::const_iterator last = line.end();

            vector<std::string> contents;

            const bool result = parse (first, last,
                                       + (char_ - ' ') % space,
                                       contents);

            // FIXME: this is stupid, but works.
            if (!contents.empty()) {
                // when we have less than 4 entries, something is weird.
                if (contents.size() < 4)
                    SHARKSVMEXCEPTION ("Unexpected format occured. ?Syntax Error");

                BOOST_LOG_TRIVIAL (trace) << line;


                boost::cmatch matchedStrings;

                // threshold
                boost::regex thresholdExpression ("threshold b");

                if (boost::regex_search (line.c_str(), matchedStrings, thresholdExpression)) {
                    BOOST_LOG_TRIVIAL (trace) << "rho:";
                    RealVector biasTerm(contents.size() - 1);
                    for (size_t m = 1; m < contents.size(); m++) {
                        // we need to take care:
                        // BSGD takes negative bias, not positive, as we do-- so take -rho..
                        double currentRho = -boost::lexical_cast<double> (contents[m]);
                        biasTerm[m] = currentRho;
                        BOOST_LOG_TRIVIAL (trace) << currentRho;
                    }
                    container -> setBias(biasTerm);
                
                    // this is the sign to quit reading the header
                    break;
                }


                // kernel type
                boost::regex kernelTypeExpression ("kernel type");

                if (boost::regex_search (line.c_str(), matchedStrings, kernelTypeExpression)) {
                    // TODO: more than this
                    container -> m_kernelType = -42;

                    if (contents[0] == "2")
                        container -> m_kernelType = KernelTypes::RBF;

                    if (contents[0] == "0")
                        container -> m_kernelType = KernelTypes::LINEAR;

                    if (container -> m_kernelType == -42)
                        throw SHARKSVMEXCEPTION ("unsupported kernel type!");
                }


                // gamma kernel parameter
                boost::regex gammaExpression ("kernel parameter -g");

                if (boost::regex_search (line.c_str(), matchedStrings, gammaExpression)) {
                    container -> m_gamma = boost::lexical_cast<double> (contents[0]);
                    std::cout << "gamma:" << container -> m_gamma << endl;
                }


                // FIXME:  so many things..

                continue;
            }

            if (!result || first != last)
                throw SHARKSVMEXCEPTION ("[SVMLight read header] problems parsing file");
        }

        // need to set svmtype
        container -> m_svmType = SVMTypes::CSVC;

    }



    void SVMLightDataModel::load (std::string filePath) {
        std::cout << "Loading from " << filePath << endl;

        // stupid thing, need a filestream, or how else
        // to remember the position we saved the header?
        std::ifstream stream (filePath.c_str());

        if (!stream.good())
            throw SHARKSVMEXCEPTION ("[SVMLightDataModel::load] failed to open file for input");


        // read header first
        loadHeader (stream);

        // load alphas and SVs in sparse format
        container -> loadSparseLabelAndData (stream);
    }



    void SVMLightDataModel::save (std::string filePath) {
        std::cout << "Saving model to " << filePath << std::endl;

        // create new datastream
        std::ofstream ofs;
        ofs.open (filePath.c_str());

        if (!ofs)
            throw (SHARKSVMEXCEPTION ("[export_SVMLight] file can not be opened for writing"));


        // create header first
        saveHeader (ofs);
        RealMatrix preparedAlphas = container -> m_alphas;

        // prepare for binary case

        switch (container -> m_svmType) {
            case SVMTypes::CSVC: {
                // nothing to prepare for CSVC
                break;
            }

            case SVMTypes::Pegasos: {
                // FIXME: multiclass
                // throw away the alphas we do not need
                preparedAlphas.resize (container -> m_alphas.size1(), 1);

                // FIXME: better copy...
                for (size_t j = 0; j < container -> m_alphas.size1(); ++j) {
                    preparedAlphas (j, 0) = container -> m_alphas (j, 1);
                }

                break;
            }

            default: {
                throw (SHARKSVMEXCEPTION ("[SVMLightDataModel::save] Unsupported SVM type."));
            }
        }


        // then save data in SVMLight format
        container -> saveSparseLabelAndData (ofs);

        ofs.close();
    }



    unsigned int SVMLightDataModel::numberOfPositiveSV() {
        // FIXME: for multiclass this is .....
        // TODO: sanity check maybe
        // TODO: can this be done better?
        unsigned int totalPosSV = 0;

        for (size_t j = 0; j < container -> m_alphas.size1(); ++j) {
            RealVector currentRow = row (container -> m_alphas, j);

            // FIXME: this is for the binary case only...
            if (currentRow[0] > 0)
                totalPosSV++;
        }

        return (totalPosSV);
    }



    unsigned int SVMLightDataModel::numberOfNegativeSV() {
        unsigned int totalSV = container -> m_supportVectors.numberOfElements();
        return (totalSV - numberOfPositiveSV());
    }



    void SVMLightDataModel::saveHeader (std::ofstream &modelDataStream) {
        std::cout << "Saving headers..." << endl;

        // sanity check for size
        unsigned int totalSV = container -> m_supportVectors.numberOfElements();

        if (totalSV != container -> m_alphas.size1()) {
            throw (SHARKSVMEXCEPTION ("[createHeader] label dimension and data dimension mismatch."));
        }

        std::cout << "Total number of support vectors: " << totalSV << std::endl;


        if (!modelDataStream)
            throw (SHARKSVMEXCEPTION ("[createHeader] Cannot open modeldata file for writing!"));


        // svm type TODO: refactor
        modelDataStream << "svcontainer -> m_type ";

        switch (container -> m_svmType) {
            case SVMTypes::CSVC: {
                modelDataStream << "c_svc" << endl;
                break;
            }

            case SVMTypes::Pegasos: {
                // for pegasos we pretend we are CSVC and modify the alphas accordingly (take alphas for first class only)
                modelDataStream << "c_svc" << endl;
                break;
            }

            default: {
                throw (SHARKSVMEXCEPTION ("[createHeader] SVMLight format does not support the specified SVM type!"));
                break;
            }
        }

        // kernel type TODO: refactor
        modelDataStream << "kernel_type ";

        switch (container -> m_kernelType) {
            case KernelTypes::RBF: {
                modelDataStream << "rbf" << endl;
                break;
            }

            case KernelTypes::LINEAR: {
                modelDataStream << "linear" << endl;
                break;
            }

            default: {
                throw (SHARKSVMEXCEPTION ("[createHeader] SVMLight format does not support the specified kernel type!"));
                break;
            }
        }

        // gamma
        modelDataStream << "gamma " << container -> m_gamma << std::endl;


        modelDataStream << "nr_class " << 2 << std::endl;
        modelDataStream << "total_sv " << totalSV << std::endl;

        // make sure the bias is compatible with the way SVMLight computes it-- i.e. -b not +b.
        modelDataStream << "rho " << -container -> m_rho << std::endl;

        // FIXME: multiclass.
        modelDataStream << "label -1 1" << std::endl;
        unsigned int posSV = numberOfPositiveSV();
        modelDataStream << "nr_sv " << posSV << " " << totalSV - posSV << std::endl;
        modelDataStream << "SV" << std::endl;
    }



}