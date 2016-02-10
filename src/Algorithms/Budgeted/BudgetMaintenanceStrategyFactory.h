//===========================================================================
/*!
 *
 *
 * \brief       Factory for Budget maintenance strategies
 *
 * \par
 * Given a string, create a proper budget strategy
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
//===========================================================================


#ifndef SHARK_MODELS_BUDGETMAINTENANCESTRATEGYFACTORY_H
#define SHARK_MODELS_BUDGETMAINTENANCESTRATEGYFACTORY_H

#include <iostream>
#include "SharkSVM.h"
#include "Budgeted/MergeBudgetMaintenanceStrategy.h"
#include "Budgeted/ProjectBudgetMaintenanceStrategy.h"
#include "Budgeted/RemoveBudgetMaintenanceStrategy.h"


namespace shark {

///
/// Simple factory to create an strategy object given its name.
///
///
    template<class InputType>
    class BudgetMaintenanceStrategyFactory {
            typedef typename Data<InputType>::element_reference ElementType;
        public:

            /// constructor.
            BudgetMaintenanceStrategyFactory() {
            }


            /// Create an budget maintenance strategy object by string.
            /// \param[in]  budgetMaintenanceStrategy   string with the name of the strategy.
            /// \return object for specified strategy.
            ///
            static AbstractBudgetMaintenanceStrategy<InputType> &createBudgetMaintenanceStrategy (std::string budgetMaintenanceStrategy) {
                BOOST_LOG_TRIVIAL (info) << "factory: creating budget strategy: " << std::endl;

                if (budgetMaintenanceStrategy == "RemoveSmallest") {
                    BOOST_LOG_TRIVIAL (info) << "factory: remove smallest " << std::endl;
                    RemoveBudgetMaintenanceStrategy<InputType> *strategy = new RemoveBudgetMaintenanceStrategy<InputType> (RemoveBudgetMaintenanceStrategy<InputType>::SMALLEST);
                    return (*strategy);
                }

                if (budgetMaintenanceStrategy == "RemoveRandom") {
                    BOOST_LOG_TRIVIAL (info) << "factory: remove  random " << std::endl;
                    RemoveBudgetMaintenanceStrategy<InputType> *strategy = new RemoveBudgetMaintenanceStrategy<InputType> (RemoveBudgetMaintenanceStrategy<InputType>::RANDOM);
                    return (*strategy);
                }

                if (budgetMaintenanceStrategy == "Merge") {
                    BOOST_LOG_TRIVIAL (info) << "factory: merge" << std::endl;
                    MergeBudgetMaintenanceStrategy<InputType> *strategy = new MergeBudgetMaintenanceStrategy<InputType>();
                    return (*strategy);
                }

                if (budgetMaintenanceStrategy == "Project") {
                    BOOST_LOG_TRIVIAL (info) << "factory: project" << std::endl;
                    ProjectBudgetMaintenanceStrategy<InputType> *strategy = new ProjectBudgetMaintenanceStrategy<InputType>();
                    return (*strategy);
                }

                // unknown strategy, need to throw something
                throw (SHARKSVMEXCEPTION ("BudgetMaintenanceStrategyFactory: Unknown budget maintenance strategy!"));
            }


            /// Create an budget maintenance strategy object by enum value.
            /// \param[in]  budgetMaintenanceStrategy   enum value for the strategy.
            /// \return object for specified strategy.
            ///
            static AbstractBudgetMaintenanceStrategy<InputType> &createBudgetMaintenanceStrategy (size_t budgetMaintenanceStrategy) {
                std::string humanReadable = "";

                switch (budgetMaintenanceStrategy) {
                    case BudgetMaintenanceStrategy::REMOVE:
                        BOOST_LOG_TRIVIAL(debug) << "creating remove strategy";
                        humanReadable = "RemoveSmallest"; 
                        break;

                    case BudgetMaintenanceStrategy::MERGE:
                        BOOST_LOG_TRIVIAL(debug) << "creating merge strategy";
                        humanReadable = "Merge";
                        break;

                    case BudgetMaintenanceStrategy::PROJECT:
                        BOOST_LOG_TRIVIAL(debug) << "creating project strategy";
                        humanReadable = "Project";
                        break;

                    default:
                        throw (SHARKSVMEXCEPTION ("BudgetMaintenanceStrategyFactory: Unknown budget maintenance strategy!"));
                }

                return createBudgetMaintenanceStrategy (humanReadable);
            }


            /// class name.
            std::string name() const
            { return "BudgetMaintenanceStrategyFactory"; }
    };


}
#endif
