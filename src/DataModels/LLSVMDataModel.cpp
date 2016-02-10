//===========================================================================
/*!
 *
 *
 * \brief       LLSVM data model
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
#include "LLSVMDataModel.h"
#include "DataModelContainer.h"
#include "SharkSVM.h"



using namespace std;


namespace shark {


    LLSVMDataModel::LLSVMDataModel() {
    }



    LLSVMDataModel::~LLSVMDataModel() {
    }



    void LLSVMDataModel::loadHeader (std::ifstream &modelDataStream) {
        BOOST_LOG_TRIVIAL (info) << "Loading headers for LLSVM DataModel... ";

        // we simply discard the data container we might have had
        container = DataModelContainerPtr (new DataModelContainer());

        while (modelDataStream) {
            using namespace boost::spirit::qi;

            std::string line;
            std::getline (modelDataStream, line);

            if (line.empty())
                continue;

            std::string::const_iterator first = line.begin();
            std::string::const_iterator last = line.end();

            vector<std::string> contents;

            const bool result = parse (first, last, + (char_ - ' ') % space, contents);

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
                        throw SHARKSVMEXCEPTION ("Unsupported SVM type!");
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



    void LLSVMDataModel::load (std::string filePath) {
        std::cout << "Loading LLSVM Model from " << filePath << endl;

        // stupid thing, need a filestream, or how else
        // to remember the position we saved the header?
        std::ifstream stream (filePath.c_str());

        if (!stream.good())
            throw SHARKSVMEXCEPTION ("[LLSVMDataModel::load] failed to open file for input");

        // read header and data
        loadHeader (stream);
        loadSparseLabelAndData (container, stream);
    }



    void LLSVMDataModel::save (std::string filePath) {
        std::cout << "Saving model to " << filePath << std::endl;

        // create new datastream
        std::ofstream ofs;
        ofs.open (filePath.c_str());

        if (!ofs)
            throw (SHARKSVMEXCEPTION ("[export_BSGD] file can not be opened for writing"));


        // create header and save data
        saveHeader (ofs);
        RealMatrix preparedAlphas = container -> m_alphas;
        container -> saveSparseLabelAndData (ofs);
        ofs.close();
    }



    void LLSVMDataModel::saveHeader (std::ofstream &modelDataStream) {
        BOOST_LOG_TRIVIAL (debug)  << "Saving LLSVM headers." << endl;

        // sanity check for size
        unsigned int totalSV = container -> m_supportVectors.numberOfElements();

        if (totalSV != container -> m_alphas.size1()) {
            BOOST_LOG_TRIVIAL (trace) << "Size of alphas: " << container -> m_alphas.size1() << ", " << container -> m_alphas.size2();
            BOOST_LOG_TRIVIAL (trace) << "Number of SVs: " << container -> m_supportVectors.numberOfElements();
            throw (SHARKSVMEXCEPTION ("Label dimension and data dimension mismatch."));
        }

        BOOST_LOG_TRIVIAL (debug) << "Total number of support vectors: " << totalSV << std::endl;


        if (!modelDataStream)
            throw (SHARKSVMEXCEPTION ("Cannot open modeldata file for writing!"));


        // svm type TODO: refactor
        modelDataStream << "svm_type LLSVM\n";

        if (container -> m_svmType == SVMTypes::LLSVM) {
            throw (SHARKSVMEXCEPTION ("LLSVM format does not support the specified SVM type!"));
        }

        if (container -> m_kernelType != KernelTypes::RBF) {
            throw (SHARKSVMEXCEPTION ("LLSVM format does not support the specified kernel type!"));
        }

        // gamma
        modelDataStream << "gamma " << container -> m_gamma << std::endl;

        // number of classes
        // for now there is only binary LLSVM.
        size_t nClasses = 2;
        
        // fix our sparse saving
        if (nClasses == 1)
            nClasses = 2;
        
        BOOST_LOG_TRIVIAL (debug) << "Total classes: " << nClasses;
        modelDataStream << "nr_class " << nClasses << std::endl;

        // total number of SVs is just the budget size.
        size_t total_sv = container -> m_supportVectors.numberOfElements();
        modelDataStream << "total_sv " << totalSV << std::endl;

        // FIXME?
        // make sure the bias is compatible with the way libsvm computes it-- i.e. -b not +b.
        // and also if we only have a binary problem we only write out one of the two
        modelDataStream << "rho";
        size_t nBiasTerms = container -> m_rho.size();
        if (nBiasTerms == 2)
            nBiasTerms = 1;
        
        // there is another possiblity: the model does not support bias terms at all.
        // in this case we want to fake bias terms by setting them to zero.
        if (nBiasTerms == 0)
        {
            nBiasTerms = nClasses;
            
            // even in this case we need to fix the number of written rhos for the binary case.
            if (nBiasTerms == 2)
                nBiasTerms = 1;
            
            
            for (size_t c = 0; c < nBiasTerms; c++)
                modelDataStream << " " << "0.0";
        }
        else
        {
            // normal case: we have bias terms available
            for (size_t c = 0; c < nBiasTerms; c++)
                modelDataStream << " " << - container -> m_rho[c];
        }
        modelDataStream  << std::endl;
        
        
        // dump labels
        std::vector<int> labelOrder;
        container ->m_labelOrder.getLabelOrder(labelOrder);
        modelDataStream <<  "label";
        for (size_t c = 0; c < labelOrder.size(); c++)
            modelDataStream << " " << labelOrder[c];
        modelDataStream << std::endl;
        
        // dump number of SVs
        modelDataStream << "nr_sv";
        size_t countedTotalSV = 0;
        
        modelDataStream << "SV" << std::endl;
    }



    void LLSVMDataModel::loadSparseLabelAndData (DataModelContainerPtr container, std::ifstream &modelDataStream) {
        BOOST_LOG_TRIVIAL (debug) << "Loading LLSVM sparse label and data...";

        // read the alphas and the support vectors now..
        BOOST_LOG_TRIVIAL (trace) << "Reading SVs..";

        // FIXME
        typedef std::vector<std::pair<int32_t, double> >  LibSVMPoint;
        std::vector<LibSVMPoint> contents;
        std::vector<std::vector<double> > weightMatrix;

        while (modelDataStream) {
            std::string line;
            std::getline (modelDataStream, line);

            if (line.empty())
                continue;

            using namespace boost::spirit::qi;
            using boost::phoenix::push_back;
            using boost::phoenix::ref;

            std::string::const_iterator first = line.begin();
            std::string::const_iterator last = line.end();

            std::vector<double> weights;
            std::vector<int32_t> pointIndex;
            std::vector<double> pointValue;

            // FIXME: fixme, fixme hard. this sucks. completely... never show this to anyone.
            bool r = phrase_parse (
                         first, last,
                         * (
                             (int_[push_back (ref (pointIndex), boost::spirit::_1)] >> ':' >>
                              double_[push_back (ref (pointValue), boost::spirit::_1)])
                             |
                             double_[push_back (ref (weights), boost::spirit::_1)])
                         ,
                         space);


            if (!r || first != last) {
                BOOST_LOG_TRIVIAL (fatal) << std::string (first, last);
                throw SHARKSVMEXCEPTION ("[import_libsvm_reader] problems parsing file");
            }

            // add support vectors to content as usual
            LibSVMPoint newPoint;

            for (size_t t = 0; t < pointValue.size(); t++) {
                // create a pair
                std::pair<int32_t, double> entry;
                entry.first = pointIndex[pointIndex.size() - pointValue.size() + t  ];
                entry.second = pointValue[t];
                newPoint.push_back (entry);
            }

            weightMatrix.push_back (weights);
            contents.push_back (newPoint);
        }

        //read contents of stream
        std::size_t numPoints = contents.size();
        BOOST_LOG_TRIVIAL (debug) << "Found " << numPoints << " SV.";

        // sanity check
        if (numPoints < 1)
            throw SHARKSVMEXCEPTION ("Zero data points read?");

        // here the alphas are the first X rows of the weight 'matrix'
        // so we divide it into parts
        int alphaDimension = weightMatrix[0].size() - weightMatrix.size();

        if (alphaDimension <= 0)
            throw SHARKSVMEXCEPTION ("No alpha labels found?");

        BOOST_LOG_TRIVIAL (info) << "Dimension of alphas (Number of classes): " << alphaDimension;


        // create alphas and weights
        container-> m_alphas.resize (numPoints, alphaDimension);
        container -> m_weights.resize (numPoints, numPoints);

        for (size_t i = 0; i < weightMatrix.size(); i++) {
            // alpha
            std::vector<double> currentRow = weightMatrix[i];

            for (int j = 0; j < alphaDimension; j++)
                container -> m_alphas (i, j) = currentRow[j];

            // weights
            for (size_t j = alphaDimension; j < currentRow.size(); j++)
                container -> m_weights (i, j - alphaDimension) = currentRow[j];
        }

        BOOST_LOG_TRIVIAL (trace) << "Copied alphas and weights.";


        // need really to check all those entries to obtain data dimension
        int dataDimension = 0;

        // check for feature index zero (non-standard, but it happens)
        bool haszero = false;


        for (unsigned int l = 0; l < contents.size(); l++) {
            // FIXME: slow copy
            for (unsigned int c = 0; c < contents[l].size(); c++) {
                int curIndex = contents[l][c].first;

//                        std::cout << curIndex << ":" << contents[l][c].second << " " ;

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
        /*
        BOOST_LOG_TRIVIAL(trace) << "Reordering labels. ";
        if ( (container -> m_classOrdering)[0] == 1)
            swap_columns( container->m_alphas, 0, 1);

        // FIXME: for now it is not multiclass. do it by hand
        shark::RealMatrix tmpAlphas = container->m_alphas;
        container-> m_alphas.resize(numPoints, alphaDimension - 1);
        for (int i = 0; i < tmpAlphas.size1(); i++)
            container -> m_alphas(i,0) = tmpAlphas(i,0);
        */

        /*
        // -- 8< -- cut here --


        // damn debug
        for(unsigned int  l = 0; l < container -> m_alphas.size1(); l++)
        {
            std::cout << std::endl << l << "   ";
            // FIXME: slow copy
            for(unsigned int c = 0; c < container -> m_alphas.size2(); c++)
            {
                std::cout << c << ":" << container -> m_alphas(l, c) << " " ;
            }
            size_t s = container -> m_alphas.size2();
            double alphaCoefficient = (s == 1) ? container->m_alphas (l, 0) : 0.5 * (container->m_alphas(l, 1) - container->m_alphas(l, 0));
            std::cout << endl << "    -- " << alphaCoefficient;
        }


        size_t s = container -> m_alphas.size2();
        double maxAlpha = (-std::numeric_limits<double>::infinity());
        double sumAlpha = 0.0;
        double maxYAlpha = (-std::numeric_limits<double>::infinity());
        double sumYAlpha = 0.0;
        for (size_t i = 0; i < container -> m_alphas.size1(); i++)
        {
            // FIXME: refactor me. it hurts.
            double alphaCoefficient = (s == 1) ? container->m_alphas (i, 0) : 0.5 * (container->m_alphas(i, 1) - container->m_alphas(i, 0));
            sumAlpha += abs(alphaCoefficient);
            if (abs(alphaCoefficient)> maxAlpha)
                maxAlpha = abs(alphaCoefficient);
            sumYAlpha += (alphaCoefficient);
            if ((alphaCoefficient)> maxYAlpha)
                maxYAlpha = (alphaCoefficient);
        }
        BOOST_LOG_TRIVIAL(debug) << "Sum of Alphas: " << setprecision(16) <<  sumAlpha;
        BOOST_LOG_TRIVIAL(debug) << "Max of Alphas: " << setprecision(16) << maxAlpha;


        //--

        */


        // sad truth is:
        // read alpha = 'weights'
        // read weights = 'M'
        // and more true alphas = Mw
        std::cout << "Alphas size: " << container->m_alphas.size1() << " x " << container->m_alphas.size2() << std::endl;
        std::cout << "Weights size: " << container->m_weights.size1() << " x " << container->m_weights.size2() << std::endl;
        //RealMatrix e = container -> m_weights * container->m_alphas;
        RealMatrix e = container->m_alphas;
        e.clear();
//        container->m_weights = shark::blas::trans(container->m_weights);
        shark::blas::axpy_prod (container->m_weights, container->m_alphas, e);

        std::cout << "Result size: " << e.size1() << " x " << e.size2() << std::endl;

        // copy over, but axpy has wrong sign, should read the docs i guess
        std::vector<unsigned int> labelOrder;
        container ->m_labelOrder.getLabelOrder(labelOrder);
        
        if (labelOrder[0] == 0) {
            BOOST_LOG_TRIVIAL (trace) << "Normal ordering. ";
            container -> m_alphas = e;
        } else {
            BOOST_LOG_TRIVIAL (trace) << "Odd ordering. ";
            container -> m_alphas = -e;
        }

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


    void LLSVMDataModel::saveSparseLabelAndData (DataModelContainerPtr container, std::ifstream &modelDataStream) {
        BOOST_LOG_TRIVIAL (debug) << "Saving LLSVM sparse label and data...";
    }        

}


