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


#ifndef SHARK_SVMLightDATAMODEL_H
#define SHARK_SVMLightDATAMODEL_H

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


    class SVMLightDataModel : public AbstractSVMDataModel {
        private:

            unsigned int numberOfPositiveSV();

            unsigned int numberOfNegativeSV();


        public:

            SVMLightDataModel();


            virtual ~SVMLightDataModel();


            /// \brief From INameable: return the class name.
            std::string name() const
            { return "SVMLightDataModel"; }


            /// \brief
            virtual void load (std::string filePath);


            /// \brief
            virtual void save (std::string filePath);



            /// \brief
            void loadHeader (std::ifstream &modelDataStream);



            /// \brief
            void saveHeader (std::ofstream &modelDataStream);



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

    };

}

#endif

