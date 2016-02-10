
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

 
    set (DATAMODELS_SOURCE 
            # --
            BudgetedSVMDataModel.cpp  
            DataModelContainer.cpp
            LibSVMDataModel.cpp  
            LLSVMDataModel.cpp
            LocalSVMDataModel.cpp  
            SVMDataModelFactory.cpp
            SVMLightDataModel.cpp  
            # --
            AbstractSVMDataModel.h
            BudgetedSVMDataModel.h  
            DataModelContainer.h
            LibSVMDataModel.h  
            LLSVMDataModel.h
            LocalSVMDataModel.h
            SVMDataModelFactory.h
            SVMLightDataModel.h
    )

    add_library (DataModels STATIC ${DATAMODELS_SOURCE})
    target_link_libraries (DataModels ${Boost_LIBRARIES}  ${SHARK_LIBRARIES})
    set_target_properties(DataModels PROPERTIES LINKER_LANGUAGE CXX)

    