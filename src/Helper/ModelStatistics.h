//===========================================================================
/*!
 *
 *
 * \brief       Compute some statistics for a given learned classifer
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


#ifndef SHARK_MODELSTATISTICS_H
#define SHARK_MODELSTATISTICS_H

#include <shark/Core/INameable.h>
#include <shark/Core/ISerializable.h>

#include "SharkSVM.h"

#include <shark/Data/Dataset.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Data/Libsvm.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h> //the used kernel for the SVM
#include <shark/Models/LinearClassifier.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>


namespace shark {


//! \brief .
//!
//! \par
//!
//! \sa


    class ModelStatistics : public INameable, ISerializable {
        private:

        public:

            ModelStatistics() {
            };


            virtual ~ModelStatistics() {};


            /// \brief From INameable: return the class name.
            std::string name() const {
                return "ModelStatistics";
            }

            
            /// \brief
            static double aucValue (KernelClassifier<RealVector> const &model, ClassificationDataset const &data);
            
            
            
            /// \brief
            static double primalValue (KernelClassifier<RealVector> const &model, ClassificationDataset const &data, double C);



            /// \brief
            static double dualValue (KernelClassifier<RealVector> const &model, ClassificationDataset const &data);


        protected:



    };




}

#endif

