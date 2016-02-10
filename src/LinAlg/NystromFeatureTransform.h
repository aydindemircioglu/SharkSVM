/*!
 * 
 *
 * \brief       Transformation of data into nystrom feature space
 *
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

#ifndef SHARK_LINALG_NYSTROMFEATURETRANSFORM_H
#define SHARK_LINALG_NYSTROMFEATURETRANSFORM_H


#include "LinAlg/LowRankApproximation.h"

#include <shark/Algorithms/KMeans.h>
#include <shark/LinAlg/KernelMatrix.h>




namespace shark {
        
    template <class myInputType>
    class NystromFeatureTransform {
    public:
        typedef myInputType result_type;   // do not forget to specify the result type
        typedef typename myInputType::value_type myInputValueType;
        typedef blas::matrix<myInputValueType, blas::row_major > myMatrixType;
        typedef AbstractKernelFunction<myInputType> myKernelType;
        
        NystromFeatureTransform (myMatrixType nystromMatrix, myKernelType *kernel, Centroids landmarks) :
        m_nystromMatrix (nystromMatrix),
        m_kernel (kernel),
        m_landmarks (landmarks)
        {}
        
        myInputType operator () (myInputType input) const {
            // compute kernel 'distance' to each landmark
            myInputType transformedInput (m_landmarks.centroids().numberOfElements());
            
            for (size_t t = 0; t < m_landmarks.centroids().numberOfElements(); t++) {
                transformedInput[t] = (*m_kernel) (m_landmarks.centroids().element (t), input);
            }
            
            // transform
            myInputType tmp = prod (transformedInput, m_nystromMatrix); // FIXME: bias term// + m_offset;
            return (tmp);
        }
        
    private:
        myMatrixType m_nystromMatrix;
        myKernelType *m_kernel;
        Centroids m_landmarks;
    };
}

#endif