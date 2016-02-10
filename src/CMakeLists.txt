
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


# --- cmake settings
#
#
    cmake_minimum_required (VERSION 2.8)
    project (SharkSVM_bin)
    
    

# --- add libraries
#
    
    include_directories (${Boost_INCLUDE_DIRS} ${SHARK_INCLUDE_DIRS})
    link_directories (${Boost_LIBRARY_DIRS} ${SHARK_LIBRARIES})

 

# --- add subdirectories
#

    include_directories(Algorithms)
    add_subdirectory(Algorithms)
    include_directories(DataModels)
    add_subdirectory(DataModels)
    include_directories(Helper)
    add_subdirectory(Helper)
    include_directories(LinAlg)
    add_subdirectory(LinAlg)
    include_directories(Models)
    add_subdirectory(Models)

    set (SHARKSVM_LIBRARIES
            Algorithms
            DataModels
            Helper
            LinAlg
            Models)
    
  

# --- Add files for SharkSVM main
#

    set (SHARKSVM_SOURCE ${SHARKSVM_SOURCE}
    )

    set (SHARKSVM_TRAIN_SOURCE ${SHARKSVM_SOURCE}
        SharkSVM-train.cpp 
    )

    set (SHARKSVM_PREDICT_SOURCE ${SHARKSVM_SOURCE}
        SharkSVM-predict.cpp 
    )


    
    
# --- compiler flags
#

    add_definitions("-DHAVE_CONFIG_H")
    add_definitions("-DBOOST_LOG_DYN_LINK")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -fno-strict-aliasing")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl -subsystem,console,debug")

    #if(NOT APPLE)
    # apple build fails on deprecated warnings..
    #  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
    #endif(NOT APPLE)

    if(NOT WIN32)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lpthread")
    endif(NOT WIN32)



# --- add SharkSVM executable
#

    include_directories (${SHARK_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
    link_directories (${SHARK_LIBRARY_DIRS} ${SHARK_LIBRARIES} ${Boost_LIBRARIES}  )

    add_executable (SharkSVM-train ${SHARKSVM_TRAIN_SOURCE})
    target_link_libraries(SharkSVM-train ${Boost_LIBRARIES} ${SHARK_LIBRARIES} ${SHARKSVM_LIBRARIES} -lpthread -lcblas  -fopenmp)

    add_executable (SharkSVM-predict ${SHARKSVM_PREDICT_SOURCE})
    target_link_libraries(SharkSVM-predict ${Boost_LIBRARIES} ${SHARK_LIBRARIES} ${SHARKSVM_LIBRARIES} -lpthread -lcblas  -fopenmp)




# --- install SharkSVM
#


