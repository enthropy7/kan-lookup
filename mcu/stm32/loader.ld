/* above the RAM the ROM bootloader uses; the top 40 KB of the 96 KB of an F401 for data and stack */
MEMORY
{
    rom (rx) : ORIGIN = 0x20008000, LENGTH = 24K
    ram (rwx) : ORIGIN = 0x2000E000, LENGTH = 40K
}
INCLUDE cortex-m-generic.ld
