//===========================================================================
/*!
 *
 *
 * \brief       A smart container that has all the data any model can possibly hold
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

#include "DataModelContainer.h"
#include "SharkSVM.h"


#include <boost/spirit/include/qi.hpp>
#include <boost/regex.hpp>
#include <boost/shared_ptr.hpp>


#ifndef REPLACE_BOOST_LOG
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#endif


namespace shark {

    void DataModelContainer::getKernelType (int &kernelType) {
        kernelType = m_kernelType;
        BOOST_LOG_TRIVIAL (trace) << "Retrieving kernel type: " << kernelType;
    };



    void DataModelContainer::getSVMType (int &svmType) {
        svmType = m_svmType;
        BOOST_LOG_TRIVIAL (trace) << "Retrieving SVM type: " << svmType;
    };



    void DataModelContainer::getAlphas (RealMatrix &alphas) {
        alphas = m_alphas;
        BOOST_LOG_TRIVIAL (trace) << "Retrieving alphas";
    }



    void DataModelContainer::getSupportVectors (Data<RealVector> &supportVectors) {
        supportVectors = m_supportVectors;
        BOOST_LOG_TRIVIAL (trace) << "Retrieving supportVectors";
    }



    void DataModelContainer::getGamma (double &gamma) {
        gamma = m_gamma;
        BOOST_LOG_TRIVIAL (trace) << "Retrieving gamma: " << gamma;
    }



    void DataModelContainer::getBias (RealVector &rho) {
        rho = m_rho;
        BOOST_LOG_TRIVIAL (trace) << "Retrieving bias rho.";
    }


    
    void DataModelContainer::getLabelOrder (LabelOrder &labelOrder) {
        labelOrder = m_labelOrder;
        BOOST_LOG_TRIVIAL (trace) << "Getting label order.";
    }
    
    
    
    void DataModelContainer::getLandmarks (Data<RealVector> &landmarks) {
        landmarks = m_landmarks;
        BOOST_LOG_TRIVIAL (trace) << "Retrieving landmarks.";
    }

    
    
    void DataModelContainer::setLandmarks(Data<RealVector> const &landmarks) {
        m_landmarks = landmarks;
        BOOST_LOG_TRIVIAL (trace) << "Setting landmarks.";
    }
    
    
    
    void DataModelContainer::setLabelOrder (LabelOrder &labelOrder) {
        m_labelOrder = labelOrder;
        BOOST_LOG_TRIVIAL (trace) << "Setting label order.";
    }
    
    

    void DataModelContainer::setKernelType (int kernelType) {
        m_kernelType = kernelType;
        BOOST_LOG_TRIVIAL (trace) << "Setting kernel type to " << m_kernelType;
    };



    void DataModelContainer::setSVMType (int svmType) {
        m_svmType = svmType;
        BOOST_LOG_TRIVIAL (trace) << "Setting svm type to " << m_svmType;
    };



    void DataModelContainer::setAlphas (RealMatrix alphas) {
        BOOST_LOG_TRIVIAL (trace) << "Setting alphas";
        m_alphas = alphas;
    }



    void DataModelContainer::setSupportVectors (Data<RealVector> const &supportVectors) {
        BOOST_LOG_TRIVIAL (trace)  << "Setting supportVectors";
        m_supportVectors = supportVectors;
    }



    void DataModelContainer::setGamma (double gamma) {
        m_gamma = gamma;
        BOOST_LOG_TRIVIAL (trace)  << "Setting gamma to " << m_gamma;
    }



    void DataModelContainer::setBias (RealVector rho) {
        m_rho = rho;
        BOOST_LOG_TRIVIAL (trace) << "Setting bias vector rho.";
    }



    void DataModelContainer::loadSparseLabelAndData (std::ifstream &stream) {
        BOOST_LOG_TRIVIAL (debug) << "Loading sparse label and data...";

        // read the alphas and the support vectors now..
        BOOST_LOG_TRIVIAL (trace) << "Reading SVs..";

        // FIXME
        typedef std::pair<std::vector<double>, std::vector<std::pair<std::size_t, double> > > LibSVMPoint;
        std::vector<LibSVMPoint> contents;

        while (stream) {
            std::string line;
            std::getline (stream, line);

            if (line.empty()) continue;

            using namespace boost::spirit::qi;
            std::string::const_iterator first = line.begin();
            std::string::const_iterator last = line.end();

            // FIXME: does only read one alpha..
            LibSVMPoint newPoint;
            bool r = phrase_parse (
                         first, last,
                         *(double_ >> !(lit(':') >> double_) ) 
                         >> *(uint_ >> ':' >> double_) >> *char_ >> * ('#' >> char_),
                         space , newPoint
                     );

            if (!r || first != last) {
                bool lineOK = false;
                boost::cmatch matchedStrings;

                // now this line could be one of many:
                // CVM/BVM have a "CPU Time = 0.050000 second" line at the end.
                boost::regex CPUTimeExpression ("CPU Time =");

                if (boost::regex_search (line.c_str(), matchedStrings, CPUTimeExpression))
                    lineOK = true;

                if (lineOK == false) {
                    BOOST_LOG_TRIVIAL (fatal) << std::string (first, last);
                    throw SHARKSVMEXCEPTION ("Problem parsing file at line: ");
                }

                BOOST_LOG_TRIVIAL (debug) << "Ignored line " << std::string (first, last);
            }

            contents.push_back (newPoint);
        }

        //read contents of stream
        std::size_t numPoints = contents.size();
        BOOST_LOG_TRIVIAL (info) << "Found " << numPoints << " SV.";


        // find dimension of alphas
        std::size_t labelDimension = 0;

        for (std::size_t i = 0; i != numPoints; ++i) {
            std::vector<double> const& inputs = contents[i].first;

            if (!inputs.empty())
                labelDimension = std::max (labelDimension, inputs.size());
        }
        BOOST_LOG_TRIVIAL (debug) << "Found " << labelDimension << " alpha coefficients per vector..";

        
        // find dimension of data
        std::size_t dataDimension = 0;

        for (std::size_t i = 0; i != numPoints; ++i) {
            std::vector<std::pair<std::size_t, double> > const& inputs = contents[i].second;

            if (!inputs.empty())
                dataDimension = std::max (dataDimension, inputs.back().first);
        }
        BOOST_LOG_TRIVIAL (debug) << "Found " << dataDimension << "  coefficients per vector..";
        

        // check for feature index zero (non-standard, but it happens)
        bool haszero = false;

        for (std::size_t i = 0; i < numPoints; i++) {
            std::vector<std::pair<std::size_t, double> > const& input = contents[i].second;

            if (input.empty()) continue;

            if (input[0].first == 0) {
                haszero = true;
                break;
            }
        }

        // feature 0 means more input dimensions
        if (haszero == true)
            dataDimension = dataDimension + 1;


        // copy contents into a new dataset


        // FIXME: could not get it to work otherwise.
        m_alphas.resize (numPoints, labelDimension);

        for (size_t l = 0; l < contents.size(); l++) {
            // FIXME: slow copy
            for (size_t c = 0; c < contents[l].first.size(); c++) {
                m_alphas (l, c) = contents[l].first[c];
            }
        }


        std::vector <RealVector> tmp_supportVectors;
        size_t delta = (haszero ? 0 : 1);

        for (size_t l = 0; l < contents.size(); l++) {
            RealVector tmp (dataDimension);

            // copy features
            std::vector<std::pair<std::size_t, double> > const& inputs = contents[l].second;

            for (std::size_t j = 0; j != inputs.size(); ++j) {
                tmp[inputs[j].first - delta] = inputs[j].second;
            }

            tmp_supportVectors.push_back (tmp);
        }

        m_supportVectors = createDataFromRange (tmp_supportVectors);
    }



    void DataModelContainer::saveSparseData (std::ofstream &ofs) {
        BOOST_LOG_TRIVIAL (info) << "Saving sparse data";

        // sanity check for size
        unsigned int nPoints = m_supportVectors.numberOfElements();

        if (nPoints != m_alphas.size1()) {
            throw (SHARKSVMEXCEPTION ("Label dimension and data dimension mismatch."));
        }

        // check for stream
        if (!ofs)
            throw (SHARKSVMEXCEPTION ("File can not be opened for reading"));


        // write all elements one by one
        for (std::size_t r = 0; r < nPoints; r++) {
            // write current labels to file
            // (we do not care about binary here, we dump everything)
            size_t labelDimension = m_alphas.size2();
            RealVector currentRow = row (m_alphas, r);

            // check if we have somewhere a non-zero entry
            bool allZero = true;
            for (std::size_t q = 0; q < labelDimension; q++)
            {
                if (currentRow[q] != 0.0)
                    allZero = false;
            }
            
            // only save support vectors
            if (allZero == true)
                continue;
            
            
            // save all coefficients
            for (std::size_t q = 0; q < labelDimension; q++)
            {
                ofs << std::setprecision (16) << currentRow[q] << " ";
            }

            // write current input data to file
            RealVector currentData = m_supportVectors.element (r);

            for (std::size_t j = 0; j < currentData.size(); j++) {
                // write sparse
                if (currentData[j] != 0)
                    ofs << j + 1 << ":" << std::setprecision (16) << currentData[j] << " ";
            }

            ofs << std::endl;
        }
    };



    void DataModelContainer::saveSparseLabelAndData (std::ofstream &ofs) {
        BOOST_LOG_TRIVIAL(info) << "Saving sparse label and data";

        // sanity check for size
        unsigned int nPoints = m_supportVectors.numberOfElements();

        if (nPoints != m_alphas.size1()) {
            throw (SHARKSVMEXCEPTION ("Label dimension and data dimension mismatch."));
        }


        if (!ofs)
            throw (SHARKSVMEXCEPTION ("File can not be opened for reading!"));


        // write all elements one by one
        for (std::size_t r = 0; r < nPoints; r++) {
            // write current labels to file
            // (we do not care about binary here, we dump everything)
            size_t labelDimension = m_alphas.size2();
            RealVector currentRow = row (m_alphas, r);

            for (std::size_t c = 0; c < labelDimension; c++) {
                // TODO; actually Label = vector of matrizes, some int matrices, some real matrices......
                ofs << std::setprecision (16) << currentRow[c] << " ";
            }

            // write current input data to file
            RealVector currentData = m_supportVectors.element (r);

            for (std::size_t j = 0; j < currentData.size(); j++) {
                // write sparse
                if (currentData[j] != 0)
                    ofs << j + 1 << ":" << std::setprecision (16) << currentData[j] << " ";
            }

            ofs << std::endl;
        }
    };


}

