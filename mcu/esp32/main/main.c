#include <stdio.h>

#include "esp_chip_info.h"
#include "esp_cpu.h"
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "mcubench.h"
#include "soc/rtc.h"

extern const char *const SPECS[];
extern const int NSPECS;

uint32_t mcu_cycles(void) { return esp_cpu_get_cycle_count(); }

void app_main(void) {
    esp_chip_info_t chip;
    esp_chip_info(&chip);
    rtc_cpu_freq_config_t f;
    rtc_clk_cpu_freq_get_config(&f);
    uint32_t flash = 0;
    esp_flash_get_size(NULL, &flash);
    vTaskDelay(pdMS_TO_TICKS(1500));
    const size_t largest = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);
    printf("\nboard esp32 rev %d cores %d cpu_mhz %lu flash %lu specs %d heap_free %lu largest_block %lu\n", chip.revision,
           chip.cores, (unsigned long)f.freq_mhz, (unsigned long)flash, NSPECS,
           (unsigned long)heap_caps_get_free_size(MALLOC_CAP_8BIT), (unsigned long)largest);
    for (int i = 0; i < NSPECS; i++) {
        mcu_bench(SPECS[i], 31, 16);
        mcu_bench_ram(SPECS[i], 31, 16, largest - 512);
    }
    printf("DONE\n");
    for (;;) vTaskDelay(pdMS_TO_TICKS(1000));
}
