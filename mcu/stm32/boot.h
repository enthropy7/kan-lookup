// A RAM word the startup code does not touch. Set it and reset: main enters the ROM DFU bootloader.
#define DFU_REQUEST (*(volatile uint32_t *)0x20010000u)
#define DFU_MAGIC 0xDF00B007u
