//===========================================================================
/*!
 *
 *
 * \brief       BudgetedSVM data model
 *
 *
 *
 * \author      Aydin Demircioglu
 * \date        2014
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


#include <fstream>
#include <iostream>

// WTF??
#define BOOST_SPIRIT_USE_PHOENIX_V3

#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/spirit/include/qi.hpp>


#ifndef REPLACE_BOOST_LOG
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#endif


#include <shark/Core/Exception.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/Libsvm.h>

#include "AbstractSVMDataModel.h"
#include "BudgetedSVMDataModel.h"
#include "DataModelContainer.h"
#include "SharkSVM.h"



using namespace std;


namespace shark {


    BudgetedSVMDataModel::BudgetedSVMDataModel() {
    }



    BudgetedSVMDataModel::~BudgetedSVMDataModel() {
    }



    void BudgetedSVMDataModel::loadHeader (std::ifstream &modelDataStream) {
        BOOST_LOG_TRIVIAL (info) << "Loading headers for BSGD DataModel... ";

        // we simply discard the data container we might have had
        container = DataModelContainerPtr (new DataModelContainer());

        //unsigned int nr_classes = 0;
        //  unsigned int total_sv = 0;
//    unsigned int nr_sv = 0;

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
                BOOST_LOG_TRIVIAL (trace) << contents[0];

                if (contents.size() > 1)
                    BOOST_LOG_TRIVIAL (trace) << " " << contents[1] << std::endl;

                if (contents[0] == "MODEL:") {
                    break;
                }

                if (contents[0] == "BIAS_TERM:") {
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
                }

                if (contents[0] == "ALGORITHM:") {
                    // TODO: more than this
                    container -> m_svmType = -42;

                    if (contents[1] == "4")
                        container -> m_svmType = SVMTypes::BSGD;

                    if (contents[1] == "3")
                        container -> m_svmType = SVMTypes::LLSVM;

                    if (container -> m_svmType == -42)
                        throw SHARKSVMEXCEPTION ("[import_BSGD_reader] unsupported svm type!");
                }

                if (contents[0] == "KERNEL_FUNCTION:") {
                    // TODO: more than this
                    container -> m_kernelType = -42;

                    if (contents[1] == "0")
                        container -> m_kernelType = KernelTypes::RBF;

                    if (contents[1] == "3")
                        container -> m_kernelType = KernelTypes::LINEAR;

                    if (container -> m_kernelType == -42)
                        throw SHARKSVMEXCEPTION ("[import_BSGD_reader] unsupported kernel type!");
                }

                if (contents[0] == "KERNEL_GAMMA_PARAM:") {
                    container -> m_gamma = boost::lexical_cast<double> (contents[1]);
                    std::cout << "gamma:" << container -> m_gamma << endl;
                    BOOST_LOG_TRIVIAL (info) << "Gamma in Modelfile: " << container -> m_gamma;

                    // make sure the gammas do fit-- bsgd does divide by 2
                    container -> m_gamma *= .5;
                    BOOST_LOG_TRIVIAL (info) << "Corrected gamma: " << container -> m_gamma;
                }

                if (contents[0] == "NUMBER_OF_CLASSES:") {
                    int nClasses = boost::lexical_cast<double> (contents[1]);
                    BOOST_LOG_TRIVIAL (info) << "Number of classes: " << nClasses;
                }

                if (contents[0] == "LABELS:") {
                    // here is the ordering

                    std::vector <unsigned int> labelOrder;

                    if (contents.size() == 3) {
                        // binary, it is -1 1 or 1 -1 as far as we know.
                        if (contents[1] == "1") {
                            BOOST_LOG_TRIVIAL (trace) << "detected 1 -1 ordering ";
                            labelOrder.push_back (1);
                            labelOrder.push_back (0);
                        } else {
                            BOOST_LOG_TRIVIAL (trace) << "detected -1 1 ordering ";
                            labelOrder.push_back (0);
                            labelOrder.push_back (1);
                        }
                    }

                    if (contents.size() > 3) {
                        // multiclass
                        BOOST_LOG_TRIVIAL (trace) << "detected multiclass ordering ";

                        for (size_t m = 1; m < contents.size(); m++) {
                            labelOrder.push_back (boost::lexical_cast<double> (contents[m]));
                        }
                    }

                    if (contents.size() < 3) {
                        throw SHARKSVMEXCEPTION ("Class ordering has only zero or one entry.");
                    }
                    
                    // set order
                    container->m_labelOrder.setLabelOrder(labelOrder);
                    
                    BOOST_LOG_TRIVIAL (trace) << "Ordering of classes (total: " << labelOrder.size() << "): ";

                    container->m_labelOrder.getLabelOrder(labelOrder);
                    for (size_t m = 0; m < labelOrder.size(); m++) {
                        BOOST_LOG_TRIVIAL (trace) << labelOrder[m];
                    }
                }

                continue;
            }

            if (!result || first != last)
                throw SHARKSVMEXCEPTION ("[import_BSGD_reader] problems parsing file");
        }
    }



    void BudgetedSVMDataModel::load (std::string filePath) {
        std::cout << "Loading BSGD Model from " << filePath << endl;

        // stupid thing, need a filestream, or how else
        // to remember the position we saved the header?
        std::ifstream stream (filePath.c_str());

        if (!stream.good())
            throw SHARKSVMEXCEPTION ("[BudgetedSVMDataModel::load] failed to open file for input");


        // read header first
        loadHeader (stream);

        // load alphas and SVs in sparse format, depending on the subtype
        if (container->m_svmType == SVMTypes::BSGD)
            loadSparseLabelAndData (container, stream);
        else
            throw SHARKSVMEXCEPTION ("Trying to save a non-BSGD model as BSGD. Either correct or hack this.");
    }



    void BudgetedSVMDataModel::save (std::string filePath) {
        std::cout << "Saving model to " << filePath << std::endl;

        // create new datastream
        std::ofstream ofs;
        ofs.open (filePath.c_str());

        if (!ofs)
            throw (SHARKSVMEXCEPTION ("[export_BSGD] file can not be opened for writing"));


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
                throw (SHARKSVMEXCEPTION ("[BudgetedSVMDataModel::save] Unsupported SVM type."));
            }
        }


        // then save data in BSGD format
        container -> saveSparseLabelAndData (ofs);

        ofs.close();
    }




    unsigned int BudgetedSVMDataModel::numberOfPositiveSV() {
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



    unsigned int BudgetedSVMDataModel::numberOfNegativeSV() {
        unsigned int totalSV = container -> m_supportVectors.numberOfElements();
        return (totalSV - numberOfPositiveSV());
    }



    void BudgetedSVMDataModel::saveHeader (std::ofstream &modelDataStream) {
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
        modelDataStream << "svm_type ";

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
                throw (SHARKSVMEXCEPTION ("[createHeader] BSGD format does not support the specified SVM type!"));
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
                throw (SHARKSVMEXCEPTION ("[createHeader] BSGD format does not support the specified kernel type!"));
                break;
            }
        }

        // gamma
        modelDataStream << "gamma " << container -> m_gamma << std::endl;


        modelDataStream << "nr_class " << 2 << std::endl;
        modelDataStream << "total_sv " << totalSV << std::endl;

        // make sure the bias is compatible with the way BSGD computes it-- i.e. -b not +b.
        modelDataStream << "rho " << -container -> m_rho << std::endl;

        // FIXME: multiclass.
        modelDataStream << "label -1 1" << std::endl;
        unsigned int posSV = numberOfPositiveSV();
        modelDataStream << "nr_sv " << posSV << " " << totalSV - posSV << std::endl;
        modelDataStream << "SV" << std::endl;
    }



    void BudgetedSVMDataModel::loadSparseLabelAndData (DataModelContainerPtr container, std::ifstream &modelDataStream) {
        BOOST_LOG_TRIVIAL (debug) << "Loading BSGD sparse label and data...";

        // read the alphas and the support vectors now..
        BOOST_LOG_TRIVIAL (trace) << "Reading SVs..";

        // FIXME
        typedef std::vector<std::pair<int32_t, double> >  LibSVMPoint;
        std::vector<LibSVMPoint> contents;

        while (modelDataStream) {
            std::string line;
            std::getline (modelDataStream, line);

            if (line.empty())
                continue;

            using namespace boost::spirit::qi;
            std::string::const_iterator first = line.begin();
            std::string::const_iterator last = line.end();

            LibSVMPoint newPoint;
            bool r = phrase_parse (
                         first, last,
                         * (int_ >> ':' >> double_),
                         space , newPoint
                     );

            if (!r || first != last) {
                BOOST_LOG_TRIVIAL (fatal) << std::string (first, last);
                throw SHARKSVMEXCEPTION ("[import_libsvm_reader] problems parsing file");

            }

            // for (int x = 0; x < newPoint.size(); x++)
            //    std::cout << newPoint[x].first <<  ": " << newPoint[x].second << "   ";
            // std::cout << std::endl;
            //BOOST_LOG_TRIVIAL (trace) << newPoint;

            contents.push_back (newPoint);
        }

        //read contents of stream
        std::size_t numPoints = contents.size();
        BOOST_LOG_TRIVIAL (debug) << "Found " << numPoints << " SV.";

        // sanity check
        if (numPoints < 1)
            throw SHARKSVMEXCEPTION ("Zero data points read?");

        // the alphas are now negative, we can simply search for the smallest index
        // to get their size

        // find dimension of alphas
        int alphaDimension = 0;

        LibSVMPoint firstPoint = contents[0];

        for (std::size_t i = 0; i < firstPoint.size(); ++i) {
            if (firstPoint[i].first <  alphaDimension)
                alphaDimension = firstPoint[i].first;
        }

        // check if something valid is there
        if (alphaDimension >= 0)
            throw SHARKSVMEXCEPTION ("No alpha labels found?");

        alphaDimension = -alphaDimension;
        BOOST_LOG_TRIVIAL (info) << "Dimension of alphas (Number of classes): " << alphaDimension;


        // need really to check all those entries to obtain data dimension
        int dataDimension = 0;

        // check for feature index zero (non-standard, but it happens)
        bool haszero = false;


        // FIXME: could not get it to work otherwise.
        container -> m_alphas.resize (numPoints, alphaDimension);

        for (unsigned int l = 0; l < contents.size(); l++) {
            // FIXME: slow copy
            for (unsigned int c = 0; c < contents[l].size(); c++) {
                int curIndex = contents[l][c].first;

                if (curIndex < 0)
                    container -> m_alphas (l, -curIndex - 1) = contents[l][c].second;

                // while we are at it: check if there is a "zero" feature, non-standard thing and the data dimension
                if (curIndex == 0)
                    haszero = true;

                if (curIndex > dataDimension)
                    dataDimension = curIndex;
            }
        }


        // assume data indices start with 1:... in that case when we copy the data
        // we need to shift, as arrays start with 0.
        int indexShift = 1;

        if (haszero == true) {
            // ok, our data starts with 0:... so no indexshift, but maxindex != dimension anymore
            indexShift = 0;
            dataDimension += 1;
            BOOST_LOG_TRIVIAL (debug) << "Found label with index 0, adding an index shift offset.";
        }

        BOOST_LOG_TRIVIAL (info) << "Dimension of data (Dimension of problem): " << dataDimension;



        // if its binary, we need to throw away one lane of alphas
        // as budgetedsvm format saves alphas for both classes
        // FIXME: actually reorder the alphas and then do whatever.
        BOOST_LOG_TRIVIAL (trace) << "Reordering labels. ";

        std::vector<unsigned int> labelOrder;
        container ->m_labelOrder.getLabelOrder(labelOrder);
        if (labelOrder[0] == 0) 
            swap_columns (container->m_alphas, 0, 1);

        // FIXME: for now it is not multiclass. do it by hand
        shark::RealMatrix tmpAlphas = container->m_alphas;
        container-> m_alphas.resize (numPoints, alphaDimension - 1);

        for (size_t i = 0; i < tmpAlphas.size1(); i++)
            container -> m_alphas (i, 0) = tmpAlphas (i, 0);


        BOOST_LOG_TRIVIAL (trace) << "Finished reading labels. ";


        std::vector <RealVector> tmp_supportVectors;

        for (size_t l = 0; l < contents.size(); l++) {
            RealVector tmp (dataDimension);

            // copy features
            LibSVMPoint const &inputs = contents[l];

            for (std::size_t j = 0; j < inputs.size(); ++j) {
                if (inputs[j].first >= 0)
                    tmp[inputs[j].first - indexShift] = inputs[j].second;
            }

            tmp_supportVectors.push_back (tmp);
        }

        BOOST_LOG_TRIVIAL (trace) << "Finished reading support vectors. ";

        container -> m_supportVectors = createDataFromRange (tmp_supportVectors);
    }

    
    
    void BudgetedSVMDataModel::saveSparseLabelAndData (DataModelContainerPtr container, std::ifstream &modelDataStream) {
        BOOST_LOG_TRIVIAL (debug) << "Saving BSGD sparse label and data...";
        
    }
    
    
}


