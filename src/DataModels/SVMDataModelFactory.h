//===========================================================================
/*!
 *
 *
 * \brief       Factory for SVM data models
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


#ifndef SHARK_SVMDATAMODELFACTORY_H
#define SHARK_SVMDATAMODELFACTORY_H

#include "AbstractSVMDataModel.h"
#include "SharkSVM.h"

#include <shark/Core/INameable.h>
#include <shark/Core/ISerializable.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/Libsvm.h>

#include <boost/shared_ptr.hpp>
#include <boost/spirit/include/qi.hpp>


namespace shark {


//! \brief .
//!
//! \par
//!
//! \sa



    class SVMDataModelFactory : public INameable {
        private:

        public:

            SVMDataModelFactory() {};


            virtual ~SVMDataModelFactory() {};


            /// \brief From INameable: return the class name.
            std::string name() const
            { return "SVMDataModelFactory"; }


            /// \brief determine the model from the file format and return
            /// an object that can read/write the model.
            static SVMDataModel  createFromFile (std::string filePath);

            
            /// \brief Create an object for the given model name
            static SVMDataModel  createFromString (std::string modelName);
    
            
            /// \brief Create an object from svm type enum
            static SVMDataModel  createFromType (int svmType);
    protected:


    };

}

#endif

