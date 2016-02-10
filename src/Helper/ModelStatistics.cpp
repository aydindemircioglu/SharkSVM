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


#include "ModelStatistics.h"
#include "SharkSVM.h"

#include <iostream>

#ifndef REPLACE_BOOST_LOG
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#endif



using namespace std;


namespace shark {

    double ModelStatistics::aucValue (KernelClassifier<RealVector> const &model, ClassificationDataset const &data) {
        /*
        size_t s = model.decisionFunction().outputSize();
        Data<RealVector> output_data = model.decisionFunction() (data.inputs());
        Data<RealVector> output_basis = model.decisionFunction() (model.decisionFunction().basis());
        RealMatrix const &alpha = model.decisionFunction().alpha();

        if (output_basis.numberOfElements() != alpha.size1())
            throw SHARKSVMEXCEPTION ("[primal] something is seriously wrong");

        // squared norm
        double norm2 = 0.0;

        for (size_t i = 0; i < output_basis.numberOfElements(); i++) {
            const double a = (s == 1) ? alpha (i, 0) : 0.5 * (alpha (i, 1) - alpha (i, 0));
            const double prediction = (s == 1) ? output_basis.element (i) (0) : 0.5 * (output_basis.element (i) (1) - output_basis.element (i) (0));
            norm2 += a * prediction;
        }

        // empirical error
        double loss = 0.0;

        for (size_t i = 0; i < output_data.numberOfElements(); i++) {
            const double prediction = (s == 1) ? output_data.element (i) (0) : 0.5 * (output_data.element (i) (1) - output_data.element (i) (0));
            const double y = (data.labels().element (i) == 0) ? -1.0 : +1.0;
            const double margin = y * prediction;

            if (margin < 1.0)
                loss += (1.0 - margin);
        }

        {
            // debug output of norm^2 and loss
            std::cout << " [" << norm2 << "|" << loss << "]";

            // output a few alpha values
            std::cout << " (";

            /*
            for (size_t i = 0; i < 4; i++)
                {
                    if (i != 0)
                        cout << ",";

                    const double a = (s == 1) ? alpha (i, 0) : 0.5 * (alpha (i, 1) - alpha (i, 0));
                    cout << a;
                }

            cout << ")";
        }
        */
        double loss = -1;
        double norm2 = -2;
        BOOST_LOG_TRIVIAL (debug) << "Loss: " << setprecision (16) << loss;


        return (0.5 * norm2 );
    }


    
    double ModelStatistics::primalValue (KernelClassifier<RealVector> const &model, ClassificationDataset const &data, double C) {
        size_t s = model.decisionFunction().outputSize();
        Data<RealVector> output_data = model.decisionFunction() (data.inputs());
        Data<RealVector> output_basis = model.decisionFunction() (model.decisionFunction().basis());
        RealMatrix const &alpha = model.decisionFunction().alpha();
        
        if (output_basis.numberOfElements() != alpha.size1())
            throw SHARKSVMEXCEPTION ("[primal] something is seriously wrong");
        
        // squared norm
        double norm2 = 0.0;
        
        for (size_t i = 0; i < output_basis.numberOfElements(); i++) {
            const double a = (s == 1) ? alpha (i, 0) : 0.5 * (alpha (i, 1) - alpha (i, 0));
            const double prediction = (s == 1) ? output_basis.element (i) (0) : 0.5 * (output_basis.element (i) (1) - output_basis.element (i) (0));
            norm2 += a * prediction;
        }
        
        // empirical error
        double loss = 0.0;
        
        for (size_t i = 0; i < output_data.numberOfElements(); i++) {
            const double prediction = (s == 1) ? output_data.element (i) (0) : 0.5 * (output_data.element (i) (1) - output_data.element (i) (0));
            const double y = (data.labels().element (i) == 0) ? -1.0 : +1.0;
            const double margin = y * prediction;
            
            if (margin < 1.0)
                loss += (1.0 - margin);
        }
        
        BOOST_LOG_TRIVIAL (debug) << "Squared Norm: " << setprecision (16) << norm2;
        BOOST_LOG_TRIVIAL (debug) << "Loss: " << setprecision (16) << loss;
        
        return (0.5 * norm2 + C * loss);
    }
    
    
    
    double ModelStatistics::dualValue (KernelClassifier<RealVector> const &model, ClassificationDataset const &data) {
        // depends on what we actually do...
        size_t s = model.decisionFunction().outputSize();
        Data<RealVector> output_data = model.decisionFunction() (data.inputs());
        Data<RealVector> output_basis = model.decisionFunction() (model.decisionFunction().basis());
        RealMatrix const &alpha = model.decisionFunction().alpha();

        if (output_basis.numberOfElements() != alpha.size1())
            throw SHARKSVMEXCEPTION ("[primal] something is seriously wrong");

        // squared norm
        double norm2 = 0.0;

        for (size_t i = 0; i < output_basis.numberOfElements(); i++) {
            const double a = (s == 1) ? alpha (i, 0) : 0.5 * (alpha (i, 1) - alpha (i, 0));
            const double prediction = (s == 1) ? output_basis.element (i) (0) : 0.5 * (output_basis.element (i) (1) - output_basis.element (i) (0));
            norm2 += a * prediction;
        }

        // this is w^T w, not SPARTA!
        BOOST_LOG_TRIVIAL (debug) << "Weight vector length: " << setprecision (16) << sqrt (norm2);

        // this is ATHEN.
        BOOST_LOG_TRIVIAL (debug) << "0.5 Weight vector length squared: " << setprecision (16) << (0.5 * norm2);


        double maxAlpha = (-std::numeric_limits<double>::infinity());
        double sumAlpha = 0.0;
        double maxYAlpha = (-std::numeric_limits<double>::infinity());
        double sumYAlpha = 0.0;

        for (size_t i = 0; i < output_basis.numberOfElements(); i++) {
            // FIXME: refactor me. it hurts.
            double alphaCoefficient = (s == 1) ? alpha (i, 0) : 0.5 * (alpha (i, 1) - alpha (i, 0));
            sumAlpha += abs (alphaCoefficient);

            if (abs (alphaCoefficient) > maxAlpha)
                maxAlpha = abs (alphaCoefficient);

            sumYAlpha += (alphaCoefficient);

            if ( (alphaCoefficient) > maxYAlpha)
                maxYAlpha = (alphaCoefficient);
        }

        BOOST_LOG_TRIVIAL (debug) << "Sum of Alphas: " << setprecision (16) <<  sumAlpha;
        BOOST_LOG_TRIVIAL (debug) << "Max of Alphas: " << setprecision (16) << maxAlpha;
        BOOST_LOG_TRIVIAL (debug) << "Sum of labeled Alphas: " << setprecision (16) << sumYAlpha;
        BOOST_LOG_TRIVIAL (debug) << "Max of labeled Alphas: " << setprecision (16) << maxYAlpha;

        return (- (0.5 * norm2 - sumAlpha));
    }

}

