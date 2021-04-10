# CMakeList.txt: проект CMake для dqn; включите исходный код и определения,
# укажите здесь логику для конкретного проекта.
#
cmake_minimum_required (VERSION 3.10)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


# Добавьте источник в исполняемый файл этого проекта.
add_executable (dqn "main.cpp" "dqn.cpp" "dqn.h" "agent.cpp" "agent.h" "experience.cpp" "experience.h" "enviroment.cpp" "enviroment.h")
#target_compile_features(dqn PUBLIC cxx_std_17)

target_link_libraries(dqn "${TORCH_LIBRARIES}")

set_property(TARGET dqn PROPERTY CXX_STANDARD 14)
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET dqn
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:dqn>)
endif (MSVC)

# TODO: Добавьте тесты и целевые объекты, если это необходимо.
