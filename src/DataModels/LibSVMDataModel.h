//===========================================================================
/*!
 *
 *
 * \brief       LIBSVM data model
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


#ifndef SHARK_LIBSVMDATAMODEL_H
#define SHARK_LIBSVMDATAMODEL_H

#include "AbstractSVMDataModel.h"
#include "SharkSVM.h"

#include <shark/Data/Dataset.h>
#include <shark/Data/Libsvm.h>


namespace shark {


//! \brief
//!
//! \par
//!
//! \sa


    class LibSVMDataModel : public AbstractSVMDataModel {

        public:

            LibSVMDataModel();


            virtual ~LibSVMDataModel();


            /// \brief From INameable: return the class name.
            std::string name() const
            { return "LibSVMDataModel"; }


            /// \brief
            virtual void load (std::string filePath);


            /// \brief
            virtual void save (std::string filePath);


            /// From ISerializable, reads a model from an archive
            virtual void read (InArchive & archive) {
            };

            /// From ISerializable, writes a model to an archive
            virtual void write (OutArchive & archive) const {
            };

            /// \brief  set the model to be saved
            virtual void setModel(AbstractModel<RealVector, unsigned int> &model) 
            {
            };
            
            
        protected:

            /// \brief
            void loadHeader (std::ifstream &modelDataStream);


            /// \brief
            void saveHeader (std::ofstream &modelDataStream);


            
            /// \brief count the number of SVs for a given class.
            /// \param[in]  classLabel  the class to count. this is something between 0..N-1 as we
            ///                                         assume normalized labels. 
            /// \param[in]  originalLabel   in the binary case, we only save one alpha. thus we cannot
            ///                                                 know if this belongs to the positive or negative class. therefore
            ///                                                 this parameter tells us if need to report the negatives or the positives.
            /// \return number of support vectors for the given class.
            ///
            unsigned int getSVCountForClass (size_t classLabel, int originalLabel);
                
            

            /// \brief count number of support vectors.
            /// \return number of vectors that have at least one non-zero component.
            ///
            unsigned int numberOfSV();

            
            
            /// \brief makes the given alpha a vector to a support vector?
            /// \param[in]  alphas      vector of alpha values.
            /// \return true, if any one of the alphas is != 0, or else false
            ///
            bool isSupportVector(RealVector alphas);
                
    };

}

#endif

