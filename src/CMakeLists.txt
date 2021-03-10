project( Tesis_Rodrigo_Vilca )

message( "Configuring main application: " ${PROJECT_NAME} )

file(GLOB SOURCES "*.cpp" )
file(GLOB HEADERS "*.h" )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_EXECUTABLE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
message( "RUNTIME Output: " "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}" )
message( "EXECUTABLE Output: " "${CMAKE_EXECUTABLE_OUTPUT_DIRECTORY}" )

find_package( OpenCV REQUIRED core imgproc calib3d highgui )

set( DEPENDENCY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/Calibration_Ring/)

# include_directories( 
# ${DEPENDENCY_DIR}/ring/ 
# ${OpenCV_INCLUDE_DIRS} )
 
add_executable( ${PROJECT_NAME}  
				${HEADERS} 
				${SOURCES})
				
target_link_libraries( ${PROJECT_NAME} 
                       ${OpenCV_LIBS} )
                           