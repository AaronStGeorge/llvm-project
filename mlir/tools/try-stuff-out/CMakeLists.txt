add_executable(try-stuff-out main.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

target_link_libraries(try-stuff-out
        ${dialect_libs}
        MLIRIR
        )
