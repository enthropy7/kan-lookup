// Flash writer that runs from RAM. The ROM DFU bootloader loads it: its RAM writes check out, its flash writes do
// not on this board. Vendor-class USB with one bulk pair. A request is 16 bytes {cmd, addr, len, crc}, followed for a
// write by len bytes; the reply is 8 bytes {status, value}.
#include <string.h>

#include <libopencm3/cm3/scb.h>
#include <libopencm3/cm3/systick.h>
#include <libopencm3/cm3/vector.h>
#include <libopencm3/stm32/flash.h>
#include <libopencm3/stm32/gpio.h>
#include <libopencm3/stm32/rcc.h>
#include <libopencm3/usb/dwc/otg_fs.h>
#include <libopencm3/usb/usbd.h>

#include "boot.h"

#define REG(a) (*(volatile uint32_t *)(a))

static const struct usb_device_descriptor dev = {
    .bLength = USB_DT_DEVICE_SIZE, .bDescriptorType = USB_DT_DEVICE, .bcdUSB = 0x0200, .bDeviceClass = 0xFF,
    .bMaxPacketSize0 = 64, .idVendor = 0x0483, .idProduct = 0x5741, .bcdDevice = 0x0100, .iManufacturer = 1,
    .iProduct = 2, .bNumConfigurations = 1,
};

static const struct usb_endpoint_descriptor endp[] = {{
    .bLength = USB_DT_ENDPOINT_SIZE, .bDescriptorType = USB_DT_ENDPOINT, .bEndpointAddress = 0x01,
    .bmAttributes = USB_ENDPOINT_ATTR_BULK, .wMaxPacketSize = 64,
}, {
    .bLength = USB_DT_ENDPOINT_SIZE, .bDescriptorType = USB_DT_ENDPOINT, .bEndpointAddress = 0x81,
    .bmAttributes = USB_ENDPOINT_ATTR_BULK, .wMaxPacketSize = 64,
}};

static const struct usb_interface_descriptor iface[] = {{
    .bLength = USB_DT_INTERFACE_SIZE, .bDescriptorType = USB_DT_INTERFACE, .bInterfaceNumber = 0,
    .bNumEndpoints = 2, .bInterfaceClass = 0xFF, .endpoint = endp,
}};

static const struct usb_interface ifaces[] = {{.num_altsetting = 1, .altsetting = iface}};

static const struct usb_config_descriptor config = {
    .bLength = USB_DT_CONFIGURATION_SIZE, .bDescriptorType = USB_DT_CONFIGURATION, .bNumInterfaces = 1,
    .bConfigurationValue = 1, .bmAttributes = 0x80, .bMaxPower = 0x32, .interface = ifaces,
};

static const char *strings[] = {"kan-lookup", "loader"};
static uint8_t control_buffer[128];
static usbd_device *usbd;

enum { MAX = 4096 };
static uint8_t buf[16 + MAX] __attribute__((aligned(4)));
static uint32_t got;

static void rx(usbd_device *d, uint8_t ep) {
    (void)ep;
    uint8_t pkt[64];
    const int n = usbd_ep_read_packet(d, 0x01, pkt, sizeof pkt);
    for (int i = 0; i < n && got < sizeof buf; i++) buf[got++] = pkt[i];
}

static void set_config(usbd_device *d, uint16_t value) {
    (void)value;
    usbd_ep_setup(d, 0x01, USB_ENDPOINT_ATTR_BULK, 64, rx);
    usbd_ep_setup(d, 0x81, USB_ENDPOINT_ATTR_BULK, 64, NULL);
}

static uint32_t crc32(const uint8_t *p, uint32_t n) {
    uint32_t c = 0xFFFFFFFFu;
    while (n--) {
        c ^= *p++;
        for (int k = 0; k < 8; k++) c = c >> 1 ^ (0xEDB88320u & -(c & 1));
    }
    return ~c;
}

static void reply(uint32_t status, uint32_t value) {
    const uint32_t r[2] = {status, value};
    while (usbd_ep_write_packet(usbd, 0x81, r, sizeof r) == 0) usbd_poll(usbd);
}

// OPERR, WRPERR, PGAERR, PGPERR, PGSERR
static uint32_t flash_errors(void) { return FLASH_SR & 0xF2u; }

// the bootloader leaves flash unlocked, and a key write to unlocked flash is a bus fault
static void unlock(void) {
    if (FLASH_CR & FLASH_CR_LOCK) flash_unlock();
    flash_clear_status_flags();
}

static void handle(const uint32_t *q, const uint8_t *data) {
    const uint32_t cmd = q[0], addr = q[1], len = q[2];
    switch (cmd) {
    case 'I':  // device ID and flash size in KB
        reply(0, (REG(0xE0042000u) & 0xFFFu) | (uint32_t)*(volatile uint16_t *)0x1FFF7A22u << 16);
        break;
    case 'E':  // erase sector addr
        unlock();
        flash_erase_sector((uint8_t)addr, FLASH_CR_PROGRAM_X32);
        reply(flash_errors(), 0);
        flash_lock();
        break;
    case 'W': {
        if (crc32(data, len) != q[3]) { reply(1, 0); break; }
        unlock();
        for (uint32_t o = 0; o < len; o += 4) {
            uint32_t w;
            memcpy(&w, data + o, 4);
            flash_program_word(addr + o, w);
        }
        const uint32_t err = flash_errors();
        flash_lock();
        if (err) reply(2, err);
        else if (memcmp((const void *)addr, data, len)) reply(3, 0);
        else reply(0, 0);
        break;
    }
    case 'C':  // CRC of a memory range
        reply(0, crc32((const uint8_t *)addr, len));
        break;
    case 'B':  // back to the ROM bootloader through the firmware in flash
        DFU_REQUEST = DFU_MAGIC;
        /* fall through */
    case 'G':
        reply(0, 0);
        for (int i = 0; i < 100000; i++) usbd_poll(usbd);
        scb_reset_system();
    default:
        reply(0xFF, cmd);
    }
}

int main(void) {
    // the bootloader jumps here with its interrupts, SysTick and USB still set up
    __asm__ volatile("cpsid i");
    SCB_VTOR = (uint32_t)&vector_table;
    STK_CSR = 0;
    rcc_clock_setup_pll(&rcc_hse_25mhz_3v3[RCC_CLOCK_3V3_84MHZ]);
    // flash is read back right after programming: no caches
    flash_dcache_disable();
    flash_icache_disable();
    flash_dcache_reset();
    flash_icache_reset();

    rcc_periph_clock_enable(RCC_GPIOC);
    gpio_mode_setup(GPIOC, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, GPIO13);
    gpio_clear(GPIOC, GPIO13);

    rcc_periph_clock_enable(RCC_GPIOA);
    rcc_periph_clock_enable(RCC_OTGFS);
    rcc_periph_reset_pulse(RST_OTGFS);
    // drop off the bus long enough for the host to see a disconnect
    for (volatile int i = 0; i < 3000000; i++) {}
    gpio_mode_setup(GPIOA, GPIO_MODE_AF, GPIO_PUPD_NONE, GPIO11 | GPIO12);
    gpio_set_af(GPIOA, GPIO_AF10, GPIO11 | GPIO12);
    usbd = usbd_init(&otgfs_usb_driver, &dev, &config, strings, 2, control_buffer, sizeof control_buffer);
    usbd_register_set_config_callback(usbd, set_config);
    OTG_FS_GCCFG = (OTG_FS_GCCFG | OTG_GCCFG_NOVBUSSENS) & ~(OTG_GCCFG_VBUSBSEN | OTG_GCCFG_VBUSASEN);

    for (;;) {
        usbd_poll(usbd);
        if (got < 16) continue;
        uint32_t q[4];
        memcpy(q, buf, 16);
        if (q[0] == 'W' && (q[2] > MAX || q[2] % 4)) { got = 0; reply(4, q[2]); continue; }
        if (q[0] == 'W' && got < 16 + q[2]) continue;
        handle(q, buf + 16);
        got = 0;
    }
}
