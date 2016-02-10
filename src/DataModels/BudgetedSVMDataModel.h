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


#ifndef SHARK_BSGDDATAMODEL_H
#define SHARK_BSGDDATAMODEL_H

#include "AbstractSVMDataModel.h"
#include "SharkSVM.h"

#include <shark/Data/Dataset.h>
#include <shark/Data/Libsvm.h>

#ifndef REPLACE_BOOST_LOG
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#endif


namespace shark {


//! \brief
//!
//! \par
//!
//! \sa


    class BudgetedSVMDataModel : public AbstractSVMDataModel {
        private:

            unsigned int numberOfPositiveSV();

            unsigned int numberOfNegativeSV();


        public:

            BudgetedSVMDataModel();


            virtual ~BudgetedSVMDataModel();


            /// \brief From INameable: return the class name.
            std::string name() const {
                return "BudgetedSVMDataModel";
            }


            /// \brief
            virtual void load (std::string filePath);


            /// \brief
            virtual void save (std::string filePath);


            /// From ISerializable, reads a model from an archive
            virtual void read (InArchive &archive) {
            };

            /// From ISerializable, writes a model to an archive
            virtual void write (OutArchive &archive) const {
            };

            
            /// \brief  set the model to be saved
            virtual void setModel(AbstractModel<RealVector, unsigned int> &model) 
            {
            };

            
        protected:


            /// \brief
            virtual void loadHeader (std::ifstream &modelDataStream);



            /// \brief
            virtual void saveHeader (std::ofstream &modelDataStream);



            /// \brief
            virtual void loadSparseLabelAndData (DataModelContainerPtr container, std::ifstream &modelDataStream);


            
            /// \brief
            virtual void saveSparseLabelAndData (DataModelContainerPtr container, std::ifstream &modelDataStream);
    };

}

#endif

