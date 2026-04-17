set(LIBRARY_NAME torch_fl)

file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

target_link_libraries(${LIBRARY_NAME} PRIVATE torch_cpu_library flagos ${CUDA_RUNTIME_LIB})
target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link FlagGems C++ operators if available for low-overhead C++ dispatch registration
if(FLAGGEMS_AVAILABLE)
  target_link_libraries(${LIBRARY_NAME} PRIVATE FlagGems::operators)
  target_compile_definitions(${LIBRARY_NAME} PRIVATE FLAGGEMS_AVAILABLE)
  message(STATUS "torch_fl: FlagGems C++ operator registration enabled")
endif()

install(TARGETS ${LIBRARY_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
