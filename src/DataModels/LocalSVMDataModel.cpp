//===========================================================================
/*!
 *
 *
 * \brief       Local SVM data model
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
#include <sstream>
#include <iostream>

#include "SharkSVM.h"

#ifndef REPLACE_BOOST_LOG
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#endif


#include <shark/Core/Exception.h>
#include <shark/Data/Dataset.h>

#include "Data/SparseData.h"

#include "AbstractSVMDataModel.h"
#include "LocalSVMDataModel.h"



using namespace std;


namespace shark {


    LocalSVMDataModel::LocalSVMDataModel() {
    }




    LocalSVMDataModel::~LocalSVMDataModel() {
    }



    void LocalSVMDataModel::loadHeader (std::ifstream &modelDataStream) {
        BOOST_LOG_TRIVIAL (debug) << "Loading LocalSVM headers.. ";

        while (modelDataStream) {
            using namespace boost::spirit::qi;

            std::string line;
            std::getline (modelDataStream, line);

            if (line.empty()) continue;

            std::string::const_iterator first = line.begin();
            std::string::const_iterator last = line.end();

            vector<std::string> contents;

            const bool result = parse (first, last,
                                       + (char_ - ' ') % space,
                                       contents);

            if (!contents.empty()) {
                BOOST_LOG_TRIVIAL (trace) << line;

                /*
                if (contents.size() > 1){
                    std::stringstream tmpS;
                    for (size_t m = 0; m < contents.size(); m++)
                        tmpS << " " << contents[m];
                    BOOST_LOG_TRIVIAL (trace)  << tmpS.str();
                }
                */
                
                if (contents[0] == "SV")
                    break;

                if (contents[0] == "rho") {
                    BOOST_LOG_TRIVIAL (trace) << "rho:";
                    RealVector biasTerm(contents.size() - 1);
                    for (size_t m = 1; m < contents.size(); m++) {
                        // we need to take care:
                        // BSGD takes negative bias, not positive, as we do-- so take -rho..
                        double currentRho = -boost::lexical_cast<double> (contents[m]);
                        biasTerm[m-1] = currentRho;
                        //BOOST_LOG_TRIVIAL (trace) << currentRho;
                    }
                    container -> setBias(biasTerm);
                }

                if (contents[0] == "svm_type") {
                    // TODO: more than this
                    container -> m_svmType = -42;

                    if (contents[1] == "c_svc")
                        container -> m_svmType = SVMTypes::CSVC;

                    if (contents[1] == "pegasos")
                        container -> m_svmType = SVMTypes::Pegasos;

                    // this is created by AE-SVM
                    if (contents[1] == "svm")
                        container -> m_svmType = SVMTypes::CSVC;

                    // this is created by BVM
                    if (contents[1] == "bvm")
                        container -> m_svmType = SVMTypes::CSVC;

                    // this is created by CVM
                    if (contents[1] == "cvc")
                        container -> m_svmType = SVMTypes::CSVC;
                    
                    // this is created by several multiclass programs
                    if (contents[1] == "c_svc_cs")
                        container -> m_svmType = SVMTypes::MCSVMCS;
                    if (contents[1] == "c_svc_llw")
                        container -> m_svmType = SVMTypes::MCSVMLLW;
                    if (contents[1] == "c_svc_ova")
                        container -> m_svmType = SVMTypes::MCSVMOVA;
                    if (contents[1] == "c_svc_ww")
                        container -> m_svmType = SVMTypes::MCSVMWW;
                    if (contents[1] == "c_svc_mmr")
                        container -> m_svmType = SVMTypes::MCSVMMMR;
                    if (contents[1] == "c_svc_adm")
                        container -> m_svmType = SVMTypes::MCSVMADM;
                    if (contents[1] == "c_svc_atm")
                        container -> m_svmType = SVMTypes::MCSVMATM;
                    if (contents[1] == "c_svc_ats")
                        container -> m_svmType = SVMTypes::MCSVMATS;
                    
                    if (container -> m_svmType == -42)
                        throw SHARKSVMEXCEPTION ("Unsupported SVM type!");

                    BOOST_LOG_TRIVIAL (debug) << "Detected svm_type:" << container -> m_svmType;
                }


                if (contents[0] == "kernel_type") {
                    // TODO: more than this
                    container -> m_kernelType = -42;

                    if (contents[1] == "rbf")
                        container -> m_kernelType = KernelTypes::RBF;

                    if (contents[1] == "linear")
                        container -> m_kernelType = KernelTypes::LINEAR;

                    if (container -> m_kernelType == -42)
                        throw SHARKSVMEXCEPTION ("Unsupported Kernel type!");

                    BOOST_LOG_TRIVIAL (debug) << "Detected kernel_type:" << container -> m_kernelType;
                }

                if (contents[0] == "gamma") {
                    container -> m_gamma = boost::lexical_cast<double> (contents[1]);
                    BOOST_LOG_TRIVIAL (debug) << "gamma:" << container -> m_gamma;
                }
                
                if (contents[0] == "label")
                {
                    // read order into a vector
                    std::vector<int> order;
                    BOOST_LOG_TRIVIAL (trace) << "Reading label order.";
                    for (size_t m = 1; m < contents.size(); m++) {
                        // add labels to ordering
                        order.push_back (boost::lexical_cast<int>(contents[m]));
                    }

                    // get handle to label order and set it to our order
                    BOOST_LOG_TRIVIAL (trace) << "Setting label order.";
                    LabelOrder labelOrder;
                    container -> getLabelOrder(labelOrder);
                    labelOrder.setLabelOrder(order);
                    // TODO: access here is somewhat not nice., 
                    container -> setLabelOrder(labelOrder);
                }
                
                
                // FIXME: ignore for now
                /*
                if(contents[0] == "nr_classes")
                {
                    nr_classes = boost::lexical_cast<unsigned int>(contents[1]);
                }

                if(contents[0] == "total_sv")
                {
                    total_sv = boost::lexical_cast<unsigned int>(contents[1]);
                }

                if(contents[0] == "nr_sv")
                {
                    nr_sv = boost::lexical_cast<unsigned int>(contents[1]);
                }
                */
                continue;
            }

            if (!result || first != last)
                throw SHARKSVMEXCEPTION ("Problems parsing file!");
        }

    }



    void LocalSVMDataModel::load (std::string filePath) {
        BOOST_LOG_TRIVIAL (debug) << "Loading from " << filePath << endl;

        // stupid thing, need a filestream, or how else
        // to remember the position we saved the header?
        std::ifstream stream (filePath.c_str());

        if (!stream.good())
            throw SHARKSVMEXCEPTION ("Failed to open file for input!");


        // read header first
        loadHeader (stream);

        // load alphas and SVs in sparse format
        container -> loadSparseLabelAndData (stream);
    }



    void LocalSVMDataModel::save (std::string filePath) {
        BOOST_LOG_TRIVIAL (debug) << "Saving model to " << filePath;

        // create new datastream
        std::ofstream ofs;
        ofs.open (filePath.c_str());

        if (!ofs)
            throw (SHARKSVMEXCEPTION ("File can not be opened for writing!"));


        // create header first
        saveHeader (ofs);
        RealMatrix preparedAlphas = container -> m_alphas;

        // prepare for binary case

        switch (container -> m_svmType) {
            case SVMTypes::CSVC: {
                // nothing to prepare for CSVC
                break;
            }

            // FIXME: can we apply binary data to multiclass, if so, what happens to the data model?
            case SVMTypes::MCSVMCS:
            case SVMTypes::MCSVMLLW:
            case SVMTypes::MCSVMMMR:
            case SVMTypes::MCSVMOVA:
            case SVMTypes::MCSVMADM:
            case SVMTypes::MCSVMATM:
            case SVMTypes::MCSVMATS:
            case SVMTypes::MCSVMWW:{
                // nothing to prepare
                break;
            }
                
                
            case SVMTypes::SVRG:
            case SVMTypes::BSGD:
            case SVMTypes::Pegasos: {
                BOOST_LOG_TRIVIAL (debug) << "Preparing alpha coefficients from Pegasos/BSGD to LocalSVM...";
             
                // we need to throw away the alphas we do not need,
                // for the binary case
                if (container -> m_alphas.size2() == 2) {
                    BOOST_LOG_TRIVIAL (debug) << "Found two alphas, so one of them is not needed, removing it.";
                    preparedAlphas.resize (container -> m_alphas.size1(), 1);

                    for (size_t j = 0; j < container -> m_alphas.size1(); ++j) {
                        preparedAlphas (j, 0) = container -> m_alphas (j, 1);
                    }
                }
                break;
            }

            default: {
                throw (SHARKSVMEXCEPTION ("Unsupported SVM type!"));
            }
        }

        // copy over our prepared alphas
        container -> m_alphas = preparedAlphas;

        // sanity check
        if (container -> m_alphas.size2() == 2) 
            throw SHARKSVMEXCEPTION ("Removing extra alpha coefficients in binary case failed!");
            
        
        // then save data in libsvm format
        container -> saveSparseData (ofs);

        ofs.close();
    }


    
    unsigned int LocalSVMDataModel::getSVCountForClass (size_t classLabel, int originalLabel) {
        // for each row we check if the argmax is the same as the given class
        size_t nSV = 0;
        
        // but in binary case we need to care about the sign directly
        if (container->m_alphas.size2() == 1)
        {
            for (size_t j = 0; j < container -> m_alphas.size1(); ++j) {
                RealVector currentRow = row (container -> m_alphas, j);
                if (isSupportVector (currentRow) == false)
                    continue;
                
                double alpha = currentRow[0];
                if (alpha * originalLabel > 0)
                    nSV++;
            }
                
            return nSV;
        }
            
        for (size_t j = 0; j < container -> m_alphas.size1(); ++j) {
            RealVector currentRow = row (container -> m_alphas, j);
            if (isSupportVector (currentRow) == false)
                continue;
            
            size_t maxClass = 0;
            double maxValue = -std::numeric_limits<double>::infinity();
            
            for (size_t c = 0; c < currentRow.size(); c++) {
                if (currentRow[c] > maxValue) {
                    maxClass =c ;
                    maxValue = currentRow[c];
                }
            }
            
            if (maxClass == classLabel)
                nSV++;
        }
        
        return (nSV);
    }

    

    bool LocalSVMDataModel::isSupportVector(RealVector alphas) {
        for (size_t i = 0; i < alphas.size(); i++) {
            if (alphas[i] != 0)
                return true;
        }
        return false;
    }

    
    
    unsigned int LocalSVMDataModel::numberOfSV() {
        unsigned int totalSV = 0;

        for (size_t j = 0; j < container -> m_alphas.size1(); ++j) {
            if (isSupportVector (row (container -> m_alphas, j)))
                totalSV++;
        }

        return (totalSV);
    }




    void LocalSVMDataModel::saveHeader (std::ofstream &modelDataStream) {
        BOOST_LOG_TRIVIAL (debug) << "Saving LocalSVM headers...";

        // sanity check for size
        unsigned int totalSV = numberOfSV();
        BOOST_LOG_TRIVIAL (debug) << "Total number of support vectors: " << totalSV;

        if (!modelDataStream)
            throw (SHARKSVMEXCEPTION ("Cannot open modeldata file for writing!"));

        // svm type TODO: refactor
        modelDataStream << "svm_type ";

        switch (container -> m_svmType) {
            case SVMTypes::CSVC: {
                modelDataStream << "c_svc" << endl;
                break;
            }

            case SVMTypes::SVRG:
            case SVMTypes::BSGD:
            case SVMTypes::Pegasos: {
                // for pegasos we pretend we are CSVC and modify the alphas accordingly (take alphas for first class only)
                modelDataStream << "c_svc" << endl;
                break;
            }

            case SVMTypes::MCSVMCS: {
                modelDataStream << "c_svc_cs" << endl;
                break;
            }
            
            case SVMTypes::MCSVMLLW: {
                modelDataStream << "c_svc_llw" << endl;
                break;
            }
            
            case SVMTypes::MCSVMOVA: {
                modelDataStream << "c_svc_ova" << endl;
                break;
            }
            
            case SVMTypes::MCSVMWW: {
                modelDataStream << "c_svc_ww" << endl;
                break;
            }
            
            case SVMTypes::MCSVMMMR: {
                modelDataStream << "c_svc_mmr" << endl;
                break;
            }
            
            case SVMTypes::MCSVMADM: {
                modelDataStream << "c_svc_adm" << endl;
                break;
            }

            case SVMTypes::MCSVMATM: {
                modelDataStream << "c_svc_atm" << endl;
                break;
            }

            case SVMTypes::MCSVMATS: {
                modelDataStream << "c_svc_ats" << endl;
                break;
            }
            
            default: {
                throw (SHARKSVMEXCEPTION ("LIBSVM format does not support the specified SVM type!"));
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
                throw (SHARKSVMEXCEPTION ("LIBSVM format does not support the specified kernel type!"));
                break;
            }
        }

        // parameters FIXME: multiclass
        modelDataStream << "gamma " << container -> m_gamma << std::endl;

        size_t nClasses = container -> m_alphas.size2();
        
        // fix our sparse saving
        if (nClasses == 1)
            nClasses = 2;
        
        BOOST_LOG_TRIVIAL (debug) << "Total classes: " << nClasses;
        modelDataStream << "nr_class " << nClasses << std::endl;
        
        modelDataStream << "total_sv " << totalSV << std::endl;

        // make sure the bias is compatible with the way libsvm computes it-- i.e. -b not +b.
        // and also if we only have a binary problem we only write out one of the two
        modelDataStream << "rho";
        size_t nBiasTerms = container -> m_rho.size();
        if (nBiasTerms == 2)
            nBiasTerms = 1;
        
        // there is another possiblity: the model does not support bias terms at all.
        // in this case we want to fake bias terms by setting them to zero.
        // probably this can be done better? TODO?
        if (nBiasTerms == 0)
        {
            nBiasTerms = nClasses;
            for (size_t c = 0; c < nBiasTerms; c++)
                modelDataStream << " " << "0.0";
        }
        else
        {
            // normal case: we have bias terms available
            for (size_t c = 0; c < nBiasTerms; c++)
                modelDataStream << " " << -container -> m_rho[c];
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
        for (size_t c = 0; c < labelOrder.size(); c++)
        {
            size_t nSVforCurrentClass = getSVCountForClass (c, labelOrder[c]); 
            modelDataStream << " " << nSVforCurrentClass;
            countedTotalSV += nSVforCurrentClass;
        }
        
        // end of header
        modelDataStream << std::endl;
        modelDataStream << "SV" << std::endl;

        // sanity check for number of SVs
        if (countedTotalSV != totalSV) 
            throw SHARKSVMEXCEPTION ("Sanity check failed: Number of total SVs and sum of SVs for each class differ!");
    }



}