
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

    option( opt_log_test_output "log test output to xml files." off )


     
# --- add test macro
#
#

    macro( shark_add_test src name)
 
        if( opt_log_test_output )
            set( xml_logging_command_line_args "--log_level=test_suite --log_format=XML --log_sink=${name}_log.xml --report_level=no" )
        endif( opt_log_test_output )

        add_definitions(-DBOOST_TEST_DYN_LINK) 

        add_executable( ${name} ${src} )

        # i dont get it, how to do it correctly.......
        set( link_libraries ${BOOST_LIBRARY_DIRS} LinAlg)
        include_directories (${PROJECT_SOURCE_DIR}/src/DataModels
                                            ${PROJECT_SOURCE_DIR}/src/Helper
                                            ${PROJECT_SOURCE_DIR}/src/LinAlg
                                            ${PROJECT_SOURCE_DIR}/src/Algorithms
                                            ${PROJECT_SOURCE_DIR}/src/Algorithms/Budgeted
				            ${SHARK_INCLUDE_DIRS}
                                            ${Boost_INCLUDE_DIRS})
                                            
        link_directories (${Boost_LIBRARY_DIRS} ${SHARK_LIBRARIES})
#        message(STATUS ${PROJECT_SOURCE_DIR}/src)
        target_link_libraries(  ${name} ${link_libraries}
                                            ${boost_filesystem_library}
                                            ${boost_system_library}
                                            ${boost_unit_test_framework_library} 
                                            -lpthread 
                                            -lcblas)
        link_directories (${Boost_LIBRARY_DIRS} ${SHARK_LIBRARIES})

        set(EXECUTABLE_OUTPUT_PATH bin)

        add_test( ${name} ${EXECUTABLE_OUTPUT_PATH}/${name} ${xml_logging_command_line_args} )
    endmacro()


# --- tests 
#
# 

    shark_add_test( AbstractSVMDataModel_Test.cpp AbstractSVMDataModel_Test)
    shark_add_test( AbstractBudgetMaintenanceStrategy_Test.cpp AbstractBudgetMaintenanceStrategy_Test)
    shark_add_test( CascadeSVMTrainer_Test.cpp CascadeSVMTrainer_Test)
    shark_add_test( MergeBudgetMaintenanceStrategy_Test.cpp MergeBudgetMaintenanceStrategy_Test)
    shark_add_test( IncompleteCholeskyFactorization_Test.cpp IncompleteCholeskyFactorization_Test)
    shark_add_test( LabelOrder_Test.cpp LabelOrder_Test)
    shark_add_test( NystromKernelApproximation_Test.cpp NystromKernelApproximation_Test)
    shark_add_test( RandomFourierFeatures_Test.cpp RandomFourierFeatures_Test)
    shark_add_test( RemoveBudgetMaintenanceStrategy_Test.cpp RemoveBudgetMaintenanceStrategy_Test)

