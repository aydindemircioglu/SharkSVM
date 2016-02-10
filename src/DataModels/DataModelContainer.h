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


#ifndef SHARK_DATAMODELCONTAINER_H
#define SHARK_DATAMODELCONTAINER_H

#include <shark/Core/INameable.h>
#include <shark/Core/ISerializable.h>

#include <shark/Data/Dataset.h>
#include <shark/Data/Libsvm.h>

#include "SharkSVM.h"

#include "Data/LabelOrder.h"

#include <boost/spirit/include/qi.hpp>


namespace shark {


//! \brief .
//!
//! \par
//!
//! \sa


    class DataModelContainer : public INameable, ISerializable {
        private:

        public:


            DataModelContainer() : m_gamma (-1.0),
                m_rho (0.0),
                m_useOffset (false),
                m_kernelType (-42),
                m_svmType (-42) {};


            virtual ~DataModelContainer() {};


            /// \brief From INameable: return the class name.
            std::string name() const
            { return "DataModelContainer"; }


            /// \brief
            void getKernelType (int &kernelType);



            /// \brief
            void getSVMType (int &svmType);



            /// \brief
            void getAlphas (RealMatrix &alphas);



            /// \brief
            void getSupportVectors (Data<RealVector> &supportVectors);



            /// \brief
            void getGamma (double &gamma);



            /// \brief
            void getBias (RealVector &rho);


            
            /// \brief
            void getLabelOrder (LabelOrder &labelOrder);

            
            
            /// \brief
            void getLandmarks (Data<RealVector> &landmarks);

            
            
            /// \brief
            void setLabelOrder (LabelOrder &labelOrder);
                


            /// \brief
            void setKernelType (int kernelType);



            /// \brief
            void setSVMType (int svmType);



            /// \brief
            void setGamma (double gamma);



            /// \brief
            void setBias (RealVector rho);



            /// \brief
            void setAlphas (RealMatrix alphas);



            /// \brief
            void setSupportVectors (Data<RealVector> const &supportVectors);


            
            /// \brief
            void setLandmarks(Data<RealVector> const &landmarks);
            
            

            // FIXME: refactor

            /// \brief Export data to LIBSVM format.
            ///
            /// \param  todo.
            void loadSparseLabelAndData (std::ifstream &stream);



            // FIXME: refactor

            /// \brief Export data to LIBSVM format.
            ///
            void saveSparseLabelAndData (std::ofstream &ofs);



            // FIXME: refactor

            /// \brief Export data to LIBSVM format.
            ///
            void saveSparseData (std::ofstream &ofs);



            /// From ISerializable, reads a model from an archive
            virtual void read (InArchive & archive) {
                archive >> m_gamma;
                archive >> m_rho;
                archive >> m_useOffset;
                archive >> m_alphas;
                archive >> m_supportVectors;
            };

            /// From ISerializable, writes a model to an archive
            virtual void write (OutArchive & archive) const {
                archive << m_gamma;
                archive << m_rho;
                archive << m_useOffset;
                archive << m_alphas;
                archive << m_supportVectors;
            };


            // FIXME
//protected:

            double m_gamma;


            RealVector m_rho;


            bool m_useOffset; ///< whether or not to allow non-zero offset values


            RealMatrix m_alphas;


            Data<RealVector> m_supportVectors;


            // llsvm
            RealMatrix m_weights;

            
            // landmarks for nystrom and other methods
            Data<RealVector> m_landmarks;
            
            
            int m_kernelType;


            int m_svmType;


            LabelOrder m_labelOrder;
    };

    typedef boost::shared_ptr<DataModelContainer> DataModelContainerPtr;


}

#endif

