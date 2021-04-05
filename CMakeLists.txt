cmake_minimum_required( VERSION 3.10 )

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set( CMAKE_BUILD_TYPE Release )
#set( CMAKE_BUILD_TYPE Debug )

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
  message(STATUS "\nBuild type not specified: Use Debug by default")
endif(NOT CMAKE_BUILD_TYPE)



if ( WIN32 )
	set ( OpenCV_DIR D:/opencv/build)
	message( "\nNOT NEED CONFIGURING PATH FOR WINDOWS" )

else()
	set( OpenCV_DIR 
	/home/manuel/Documents/sdk/OpenCV/installation/OpenCV-4.4.0/lib/cmake/opencv4/ )

	message( "\nCONFIGURING PATH OPENCV FOR LINUX" )
	
endif()


message( "\nCurrent directory is: ${CMAKE_CURRENT_SOURCE_DIR}" )
message( "\nDependency directory is: ${DEPENDENCY_DIR}" )
message( "\nBuild mode is: ${CMAKE_BUILD_TYPE}" )


project( Rodrigo_Project_Tests )

if ( WIN32 )
	message( "CONFIGURING FOR WINDOWS" )

else()
	message( "CONFIGURING FOR LINUX" )
	
endif()

add_subdirectory( src )