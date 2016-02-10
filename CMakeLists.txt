
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



# --- project settings
#
#

    project (SharkSVM)
        set (SHARKSVM_VERSION_MAJOR 0)
        set (SHARKSVM_VERSION_MINOR 2)
        set (SHARKSVM_VERSION_PATCH 0)
    message("\nConfiguring SharkSVM v" ${SHARKSVM_VERSION_MAJOR}  "." ${SHARKSVM_VERSION_MINOR} "." ${SHARKSVM_VERSION_PATCH} "\n")
 
    # path for extra modules
        set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
        
        cmake_policy(SET CMP0003 NEW)
        enable_language(C)
        enable_testing()


        
# --- project options 

    # should doc-generation (doxygen, manual) be run? 
        set (compile_docs OFF)

    # set default build type
        if(NOT CMAKE_BUILD_TYPE)
            set(CMAKE_BUILD_TYPE "Release")
        endif(NOT CMAKE_BUILD_TYPE)
        
    # fix case differences
        if((CMAKE_BUILD_TYPE MATCHES "debug") OR (CMAKE_BUILD_TYPE MATCHES "DEBUG"))
            set(CMAKE_BUILD_TYPE "Debug")
        endif((CMAKE_BUILD_TYPE MATCHES "debug") OR (CMAKE_BUILD_TYPE MATCHES "DEBUG"))

        if((CMAKE_BUILD_TYPE MATCHES "release") OR (CMAKE_BUILD_TYPE MATCHES "RELEASE"))
            set(CMAKE_BUILD_TYPE "Release")
        endif((CMAKE_BUILD_TYPE MATCHES "release") OR (CMAKE_BUILD_TYPE MATCHES "RELEASE"))

    # add debug stuff
        if(CMAKE_BUILD_TYPE MATCHES "Debug")
            add_definitions(-DDEBUG )
            add_definitions("-DBOOST_UBLAS_TYPE_CHECK=0" )
            #we want to be warned in debug mode
            if(UNIX)
                if(CMAKE_COMPILER_IS_GNUCC)
                    add_definitions(-Wall)
                endif(CMAKE_COMPILER_IS_GNUCC)
            endif(UNIX)
            message("-- Build type: DEBUG.")
        endif (CMAKE_BUILD_TYPE MATCHES "Debug")
        
    # add release stuff
        if(CMAKE_BUILD_TYPE MATCHES "Release")
            add_definitions(-DNDEBUG )
            add_definitions(-DBOOST_UBLAS_NDEBUG )
            message("-- Build type: RELEASE.")
        endif (CMAKE_BUILD_TYPE MATCHES "Release")


        
# --- avoid source tree pollution
#
#

    if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
        message(FATAL_ERROR "In-source builds are not permitted. Make a separate folder for building:\nmkdir build; cd build; cmake ..\nBefore that, remove the files already created:\nrm -rf CMakeCache.txt CMakeFiles")
    endif (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    
    set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")


        
# --- required packages
#
#

    find_package (Shark REQUIRED)
    find_library (PTHREAD_LIBRARY pthread )

    #set (Boost_USE_STATIC_LIBS ON)
    add_definitions("-DBOOST_LOG_DYN_LINK")

    # this is needed for LIDO, FIXME 
    set (BOOST_ROOT  "/sysdata/shared/sfw/boost/1.53")
    set (BOOST_MIN_VERSION "1.53.0")

    # ?? boost, sorry, i do not get this, calling find_package( ..) twice overwrites the first call at linking time
    # find and test for boost log, it is not always there
    find_package (Boost COMPONENTS filesystem program_options regex serialization system thread unit_test_framework REQUIRED)
    find_package (Boost COMPONENTS log_setup log filesystem program_options regex serialization system thread unit_test_framework)
    if (NOT Boost_LOG_FOUND)
        set (REPLACE_BOOST_LOG "define REPLACE_BOOST_LOG")
        message ("-- Could not find Boost Log, will replace it with simple std::cout ")
        add_definitions( -DREPLACE_BOOST_LOG=${REPLACE_BOOST_LOG})
    else (NOT Boost_LOG_FOUND)
        set (REPLACE_BOOST_LOG "undef REPLACE_BOOST_LOG")
        add_definitions( -DREPLACE_BOOST_LOG=${REPLACE_BOOST_LOG})
    endif (NOT Boost_LOG_FOUND)
    
    
    
# --- generate extra source
#
#

    # set version numbers
    add_definitions( -DSHARKSVM_VERSION_MAJOR=${SHARKSVM_VERSION_MAJOR} )
    add_definitions( -DSHARKSVM_VERSION_MINOR=${SHARKSVM_VERSION_MINOR} )
    add_definitions( -DSHARKSVM_VERSION_PATCH=${SHARKSVM_VERSION_PATCH} )
    configure_file( config/SharkSVMConfig.h.in ${CMAKE_BINARY_DIR}/generated/SharkSVMConfig.h )
    include_directories( ${CMAKE_BINARY_DIR}/generated/ ) 


    
    
# --- add subdirectories
#
#

    set(PROJECT_BINARY_DIR ${PROJECT_SOURCE_DIR}/bin)
    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

    include_directories (${Boost_INCLUDE_DIRS} ${SHARK_INCLUDE_DIR})
    link_directories (${Boost_LIBRARY_DIRS} ${SHARK_LIBRARIES})

    INCLUDE_DIRECTORIES( src )
    add_subdirectory(src)


    if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
        message(FATAL_ERROR "In-source builds are not permitted. Make a separate folder for building:\nmkdir build; cd build; cmake ..\nBefore that, remove the files already created:\nrm -rf CMakeCache.txt CMakeFiles")
    endif (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

    

# --- Include Unit Tests
#
#

    option( opt_make_tests "compile unit tests." on )

    if( opt_make_tests)
        add_subdirectory( test )
    endif( opt_make_tests )



# --- Documentation
#
#

    option( opt_make_docs "compile documentation." off )

    if (opt_make_docs)
        find_package(Doxygen)
        if(DOXYGEN_FOUND)
            message("-- Doxygen template file is ${PROJECT_SOURCE_DIR}/doc/Doxyfile.in")
            configure_file(${PROJECT_SOURCE_DIR}/doc/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
          
            add_custom_target(doxygen ALL ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile 
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen" VERBATIM)
            
        endif(DOXYGEN_FOUND)

        add_subdirectory(doc)
    endif (opt_make_docs)
