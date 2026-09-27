set(COMMON ${CMAKE_CURRENT_LIST_DIR}/../../common)
set(C_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../c)
idf_component_register(SRCS "main.c" "specs.c" "${COMMON}/mcubench.c" "${COMMON}/blob.c" "${C_DIR}/maps.c" "${C_DIR}/baselines.c"
                       INCLUDE_DIRS "." "${COMMON}" "${C_DIR}")
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -ffp-contract=off -Wno-unused-variable)
target_compile_definitions(${COMPONENT_LIB} PRIVATE MCU_VENDOR_ESP)
# the map kernels, sparse grids among them, at -O3 as registered for the sparse-grid timing
set_source_files_properties("${C_DIR}/maps.c" PROPERTIES COMPILE_OPTIONS "-O3")
