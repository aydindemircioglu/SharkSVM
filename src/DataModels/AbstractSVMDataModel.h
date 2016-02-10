//===========================================================================
/*!
 *
 *
 * \brief       Handle SVM data models, abstract class
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


#ifndef SHARK_ABSTRACTSVMDATAMODEL_H
#define SHARK_ABSTRACTSVMDATAMODEL_H

#include "SharkSVM.h"

#include <shark/Core/INameable.h>
#include <shark/Core/ISerializable.h>

#include <shark/Data/Dataset.h>
#include <shark/Data/Libsvm.h>
#include <shark/Models/AbstractModel.h>
#include "DataModelContainer.h"

#include <boost/spirit/include/qi.hpp>


namespace shark {


//! \brief .
//!
//! \par
//!
//! \sa


    class AbstractSVMDataModel : public INameable, ISerializable {
        private:

        public:

            AbstractSVMDataModel() {
                // FIXME?
                container = DataModelContainerPtr (new DataModelContainer);
            };


            virtual ~AbstractSVMDataModel() {};


            /// \brief From INameable: return the class name.
            std::string name() const {
                return "AbstractSVMDataModel";
            }


            /// \brief
            virtual void load (std::string filePath) = 0;


            /// \brief
            virtual void save (std::string filePath) = 0;


            /// From ISerializable, reads a model from an archive
            virtual void read (InArchive &archive) = 0;


            /// From ISerializable, writes a model to an archive
            virtual void write (OutArchive &archive) const = 0;

            
            /// \brief  set the model to be saved
            virtual void setModel(AbstractModel<RealVector, unsigned int> &model) = 0;
            

            // get a hold of the data inside
            DataModelContainerPtr dataContainer() {
                return container;
            }


        protected:

            DataModelContainerPtr container;

    };

    typedef boost::shared_ptr<AbstractSVMDataModel> SVMDataModel;


}

#endif

