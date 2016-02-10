
# ---
# ---   SharkSVM 
# ---
# ---   (C) 2014, Aydin Demircioglu
# --- 
# ---   This file is part of SharkSVM.
# ---   <http://www.sharksvm.de>
# --- 
# ---   SharkSVM is free software: you can redistribute it and/or modify
# ---   it under the terms of the GNU Lesser General Public License as published
# ---   by the Free Software Foundation, either version 3 of the License, or
# ---   (at your option) any later version.
# --- 
# ---   SharkSVM is distributed in the hope that it will be useful,
# ---   but WITHOUT ANY WARRANTY; without even the implied warranty of
# ---   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# ---   GNU Lesser General Public License for more details.
# --- 
# ---   You should have received a copy of the GNU Lesser General Public License
# ---   along with Shark.  If not, see <http://www.gnu.org/licenses/>.
# 



# --- add sources 
#
    include_directories (${Boost_INCLUDE_DIRS} ${SHARK_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
    link_directories (${Boost_LIBRARY_DIRS} ${SHARK_LIBRARIES})

    set (Algorithms_Source 
			CascadeSvmTrainer.h
			CSvmTrainer.h
            KernelCPATrainer.h
            KernelDCSVMTrainer.h
            KernelSGDTrainer.h
            KernelSVRGTrainer.h
            KMedoids.cpp
            KMedoids.h
            Budgeted/AbstractBudgetMaintenanceStrategy.h
            Budgeted/BudgetMaintenanceStrategyFactory.h
            Budgeted/KernelBudgetedSGDTrainer.h
            Budgeted/MergeBudgetMaintenanceStrategy.h
            Budgeted/ProjectBudgetMaintenanceStrategy.h
            Budgeted/RemoveBudgetMaintenanceStrategy.h
    )

    add_library (Algorithms ${Algorithms_Source})
    set_target_properties(Algorithms PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries (Algorithms ${Boost_LIBRARIES}  ${SHARK_LIBRARIES})
    