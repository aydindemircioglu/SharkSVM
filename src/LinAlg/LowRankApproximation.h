/*!
 *
 *
 * \brief       Abstract interface for any feature transformation like nystrom, random fourier.
 *
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

#ifndef SHARK_LINALG_FEATURETRANSFORMATION_H
#define SHARK_LINALG_FEATURETRANSFORMATION_H


#include <iostream>
#include "GlobalParameters.h"
#include "LibSVMDataModel.h"
#include "SharkSVM.h"

#include <shark/Algorithms/Trainers/AbstractTrainer.h>
#include <shark/Core/IParameterizable.h>
#include <shark/Data/Dataset.h>
#include <shark/LinAlg/KernelMatrix.h>
#include <shark/LinAlg/PartlyPrecomputedMatrix.h>
#include <shark/Models/Kernels/KernelExpansion.h>
#include <shark/Models/Kernels/KernelHelpers.h>
#include <shark/ObjectiveFunctions/KernelBasisDistance.h>
#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>
#include <shark/Models/Kernels/AbstractKernelFunction.h>


namespace shark {


    /*!
     *  \brief Abstract feature transformation class.
     *
     *  Roughly it should be used in the following way.
     *  Initialize the derived feature transformation class, e.g. RandomFourierFeatures
     *
     */

    template <class InputType>
    class LowRankApproximation: public IParameterizable {
        public:
            typedef AbstractKernelFunction<InputType> KernelType;


            /// \brief Constructor
            ///
            /// \param  kernel          kernel function to use for training and prediction
            /// \param  loss            (sub-)differentiable loss function
            /// \param  C               regularization parameter - always the 'true' value of C, even when unconstrained is set
            /// \param  offset          whether to train with offset/bias parameter or not
            /// \param  unconstrained   when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver?
            LowRankApproximation (KernelType* kernel, size_t nComponents)
                : m_kernel (kernel)
                , m_nComponents (nComponents)
            { }



            /// adapt the transformation to the given dataset
            virtual void adaptToData (UnlabeledData<InputType> dataset, size_t seed = 42) = 0;



            virtual UnlabeledData<InputType> transformData (const UnlabeledData<InputType> dataset) const = 0;



            //virtual transformData (UnlabeledData<InputType> &dataset) const = 0;



            // we also can handle labeled data, some kind of nice wrapper



            /// adapt the transformation to the given dataset
            void adaptToData (LabeledData<InputType, unsigned int> dataset, size_t seed = 42) {
                // rewind to unlabeled data case
                adaptToData (dataset.inputs(), seed);
            }



            //
            // adaptToKERNEL????
            //



            /// adapt the transformation to the given dataset
            LabeledData<InputType, unsigned int> transformData (const LabeledData<InputType, unsigned int> dataset) const {
                // rewind to unlabeled data case
                // by transforming the unlabeled data
                UnlabeledData<InputType> transformedData = transformData (dataset.inputs());

                // splice the labels and the transformed data together
                LabeledData<InputType, unsigned int> transformedLabeledData (transformedData, dataset.labels());

                return (transformedLabeledData);
            }



            /// \brief From INameable: return the class name.
            virtual std::string name() const {
                return "FeatureTransformation";
            }



            ///\brief  Returns the vector of hyper-parameters.
            virtual RealVector parameterVector() const = 0;



            ///\brief  Sets the vector of hyper-parameters.
            virtual void setParameterVector (RealVector const &newParameters) = 0;



            ///\brief Returns the number of hyper-parameters.
            size_t numberOfParameters() const {
                return m_kernel->numberOfParameters() + 1;
            }


        protected:
            KernelType *m_kernel;                     ///< pointer to kernel function

            size_t m_nComponents ;
    };


}
#endif


