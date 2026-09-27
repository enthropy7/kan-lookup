#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

#include <libopencm3/cm3/scb.h>
#include <libopencm3/stm32/flash.h>
#include <libopencm3/stm32/gpio.h>
#include <libopencm3/stm32/iwdg.h>
#include <libopencm3/stm32/rcc.h>
#include <libopencm3/stm32/syscfg.h>
#include <libopencm3/usb/cdc.h>
#include <libopencm3/usb/dwc/otg_fs.h>
#include <libopencm3/usb/usbd.h>

#include "boot.h"
#include "mcubench.h"

extern const char *const SPECS[];
extern const int NSPECS;

#define REG(a) (*(volatile uint32_t *)(a))

uint32_t mcu_cycles(void) { return REG(0xE0001004u); }

static const struct usb_device_descriptor dev = {
    .bLength = USB_DT_DEVICE_SIZE, .bDescriptorType = USB_DT_DEVICE, .bcdUSB = 0x0200,
    .bDeviceClass = USB_CLASS_CDC, .bMaxPacketSize0 = 64, .idVendor = 0x0483, .idProduct = 0x5740,
    .bcdDevice = 0x0200, .iManufacturer = 1, .iProduct = 2, .iSerialNumber = 3, .bNumConfigurations = 1,
};

static const struct usb_endpoint_descriptor comm_endp[] = {{
    .bLength = USB_DT_ENDPOINT_SIZE, .bDescriptorType = USB_DT_ENDPOINT, .bEndpointAddress = 0x83,
    .bmAttributes = USB_ENDPOINT_ATTR_INTERRUPT, .wMaxPacketSize = 16, .bInterval = 255,
}};

static const struct usb_endpoint_descriptor data_endp[] = {{
    .bLength = USB_DT_ENDPOINT_SIZE, .bDescriptorType = USB_DT_ENDPOINT, .bEndpointAddress = 0x01,
    .bmAttributes = USB_ENDPOINT_ATTR_BULK, .wMaxPacketSize = 64,
}, {
    .bLength = USB_DT_ENDPOINT_SIZE, .bDescriptorType = USB_DT_ENDPOINT, .bEndpointAddress = 0x82,
    .bmAttributes = USB_ENDPOINT_ATTR_BULK, .wMaxPacketSize = 64,
}};

static const struct {
    struct usb_cdc_header_descriptor header;
    struct usb_cdc_call_management_descriptor call_mgmt;
    struct usb_cdc_acm_descriptor acm;
    struct usb_cdc_union_descriptor cdc_union;
} __attribute__((packed)) cdcacm_functional = {
    .header = {sizeof(struct usb_cdc_header_descriptor), CS_INTERFACE, USB_CDC_TYPE_HEADER, 0x0110},
    .call_mgmt = {sizeof(struct usb_cdc_call_management_descriptor), CS_INTERFACE, USB_CDC_TYPE_CALL_MANAGEMENT, 0, 1},
    .acm = {sizeof(struct usb_cdc_acm_descriptor), CS_INTERFACE, USB_CDC_TYPE_ACM, 0},
    .cdc_union = {sizeof(struct usb_cdc_union_descriptor), CS_INTERFACE, USB_CDC_TYPE_UNION, 0, 1},
};

static const struct usb_interface_descriptor comm_iface[] = {{
    .bLength = USB_DT_INTERFACE_SIZE, .bDescriptorType = USB_DT_INTERFACE, .bInterfaceNumber = 0,
    .bNumEndpoints = 1, .bInterfaceClass = USB_CLASS_CDC, .bInterfaceSubClass = USB_CDC_SUBCLASS_ACM,
    .bInterfaceProtocol = USB_CDC_PROTOCOL_NONE, .endpoint = comm_endp,
    .extra = &cdcacm_functional, .extralen = sizeof(cdcacm_functional),
}};

static const struct usb_interface_descriptor data_iface[] = {{
    .bLength = USB_DT_INTERFACE_SIZE, .bDescriptorType = USB_DT_INTERFACE, .bInterfaceNumber = 1,
    .bNumEndpoints = 2, .bInterfaceClass = USB_CLASS_DATA, .endpoint = data_endp,
}};

static const struct usb_interface ifaces[] = {{.num_altsetting = 1, .altsetting = comm_iface},
                                              {.num_altsetting = 1, .altsetting = data_iface}};

static const struct usb_config_descriptor config = {
    .bLength = USB_DT_CONFIGURATION_SIZE, .bDescriptorType = USB_DT_CONFIGURATION, .bNumInterfaces = 2,
    .bConfigurationValue = 1, .bmAttributes = 0x80, .bMaxPower = 0x32, .interface = ifaces,
};

static const char *strings[] = {"kan-lookup", "mcubench", "stm32f4"};
static uint8_t control_buffer[128];
static usbd_device *usbd;
static volatile int started, ping, sending;
/* a spec sent over USB ("sgrid:..."), which spans packets */
static char spec_line[10240];
static volatile int spec_len, spec_ready;

static enum usbd_request_return_codes control_request(usbd_device *d, struct usb_setup_data *req, uint8_t **buf,
                                                      uint16_t *len, usbd_control_complete_callback *complete) {
    (void)d; (void)buf; (void)complete;
    if (req->bRequest == USB_CDC_REQ_SET_CONTROL_LINE_STATE) return USBD_REQ_HANDLED;
    if (req->bRequest == USB_CDC_REQ_SET_LINE_CODING && *len >= sizeof(struct usb_cdc_line_coding)) return USBD_REQ_HANDLED;
    return USBD_REQ_NOTSUPP;
}

static void data_rx(usbd_device *d, uint8_t ep) {
    (void)ep;
    char buf[64];
    // 'b' reboots into the ROM DFU bootloader, 'p' asks for "pong", 'g' starts the run, 's' sends a spec line to time;
    // a modem probe does nothing
    const int n = usbd_ep_read_packet(d, 0x01, buf, sizeof buf);
    if (n <= 0) return;
    if (spec_len > 0 || buf[0] == 's') {
        for (int i = 0; i < n && !spec_ready; i++) {
            if (buf[i] == '\n') { spec_line[spec_len] = 0; spec_ready = 1; }
            else if (spec_len < (int)sizeof spec_line - 1) spec_line[spec_len++] = buf[i];
        }
        return;
    }
    if (buf[0] == 'b') { DFU_REQUEST = DFU_MAGIC; scb_reset_system(); }
    if (buf[0] == 'p') ping = 1;
    if (buf[0] == 'g') started = 1;
}

static void sent(usbd_device *d, uint8_t ep) {
    (void)d; (void)ep;
    sending = 0;
}

static void set_config(usbd_device *d, uint16_t value) {
    (void)value;
    sending = 0;
    usbd_ep_setup(d, 0x01, USB_ENDPOINT_ATTR_BULK, 64, data_rx);
    usbd_ep_setup(d, 0x82, USB_ENDPOINT_ATTR_BULK, 64, sent);
    usbd_ep_setup(d, 0x83, USB_ENDPOINT_ATTR_INTERRUPT, 16, NULL);
    usbd_register_control_callback(d, USB_REQ_TYPE_CLASS | USB_REQ_TYPE_INTERFACE,
                                   USB_REQ_TYPE_TYPE | USB_REQ_TYPE_RECIPIENT, control_request);
}

int _write(int fd, const char *p, int len) {
    (void)fd;
    for (int done = 0; done < len;) {
        const int n = len - done > 64 ? 64 : len - done;
        // libopencm3 NAKs the endpoint when it handles a completed packet; a packet written before that is NAKed
        // for good, so wait for the completion of the last one
        while (sending) usbd_poll(usbd);
        sending = 1;
        usbd_ep_write_packet(usbd, 0x82, p + done, (uint16_t)n);
        iwdg_reset();
        done += n;
    }
    return len;
}

extern char end, _stack;
static char *brk = &end;

void *_sbrk(ptrdiff_t n) {
    // leave 8 KB for the stack
    if (brk + n > &_stack - 8192) { errno = ENOMEM; return (void *)-1; }
    char *p = brk;
    brk += n;
    return p;
}

static void hex(char *s, uint32_t v) {
    for (int k = 7; k >= 0; k--, v >>= 4) s[k] = "0123456789abcdef"[v & 15];
}

static size_t largest_block(void) {
    size_t lo = 0, hi = 256 * 1024;
    while (hi - lo > 256) {
        const size_t mid = (lo + hi) / 2;
        // volatile: GCC drops an unused malloc and assumes it succeeded
        void *volatile p = malloc(mid);
        if (p) { free(p); lo = mid; } else hi = mid;
    }
    return lo;
}

static void led(int on) {
    if (on) gpio_clear(GPIOC, GPIO13);
    else gpio_set(GPIOC, GPIO13);
}

// USB is the only way out of the board: a fault keeps it running and reports where it happened
void fault_report(const uint32_t *frame) {
    char s[] = "fault pc ........ lr ........ cfsr ........ bfar ........\n";
    hex(s + 9, frame[6]);
    hex(s + 21, frame[5]);
    hex(s + 35, SCB_CFSR);
    hex(s + 49, SCB_BFAR);
    for (uint32_t i = 0;; i++) {
        usbd_poll(usbd);
        iwdg_reset();
        if ((i & 0xFFFFF) == 0) _write(1, s, sizeof s - 1);
        led(i >> 15 & 1);
    }
}

__attribute__((naked)) void hard_fault_handler(void) {
    __asm__ volatile("tst lr, #4\n\tite eq\n\tmrseq r0, msp\n\tmrsne r0, psp\n\tb fault_report\n");
}

static void rom_bootloader_if_requested(void) {
    if (DFU_REQUEST != DFU_MAGIC) return;
    DFU_REQUEST = 0;
    rcc_periph_clock_enable(RCC_SYSCFG);
    SYSCFG_MEMRM = 1;
    const uint32_t *rom = (const uint32_t *)0x1FFF0000u;
    __asm__ volatile("msr msp, %0\n\tbx %1" : : "r"(rom[0]), "r"(rom[1]));
}

int main(void) {
    rom_bootloader_if_requested();
    // a hang must not need the reset button: the watchdog restarts the board, which then takes 'b' again
    const uint32_t reset_cause = RCC_CSR;
    RCC_CSR |= RCC_CSR_RMVF;
    iwdg_set_period_ms(30000);
    iwdg_start();
    // PC13 LED: on until the clock runs, blinking until the host sends 'g', on during the run, off at the end,
    // blinking fast on a fault
    rcc_periph_clock_enable(RCC_GPIOC);
    gpio_mode_setup(GPIOC, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, GPIO13);
    led(1);
    rcc_clock_setup_pll(&rcc_hse_25mhz_3v3[RCC_CLOCK_3V3_84MHZ]);
    flash_prefetch_enable();
    flash_icache_enable();
    flash_dcache_enable();
    REG(0xE000EDFCu) |= 1u << 24;
    REG(0xE0001004u) = 0;
    REG(0xE0001000u) |= 1u;

    rcc_periph_clock_enable(RCC_GPIOA);
    rcc_periph_clock_enable(RCC_OTGFS);
    gpio_mode_setup(GPIOA, GPIO_MODE_AF, GPIO_PUPD_NONE, GPIO11 | GPIO12);
    gpio_set_af(GPIOA, GPIO_AF10, GPIO11 | GPIO12);
    usbd = usbd_init(&otgfs_usb_driver, &dev, &config, strings, 3, control_buffer, sizeof control_buffer);
    usbd_register_set_config_callback(usbd, set_config);
    // PA9 (VBUS) is not wired on the Black Pill: force the session valid
    OTG_FS_GCCFG = (OTG_FS_GCCFG | OTG_GCCFG_NOVBUSSENS) & ~(OTG_GCCFG_VBUSBSEN | OTG_GCCFG_VBUSASEN);
    static char outbuf[256];
    setvbuf(stdout, outbuf, _IOLBF, sizeof outbuf);
    const size_t largest = largest_block();
    for (uint32_t i = 0; !started; i++) {
        usbd_poll(usbd);
        if (spec_ready) {
            led(1);
            mcu_bench(spec_line, 31, 16);
            mcu_bench_ram(spec_line, 31, 16, largest - 512);
            printf("sgrid end\n");
            spec_len = spec_ready = 0;
        }
        if (ping) {
            char s[] = "pong csr ........\n";
            hex(s + 9, reset_cause);
            ping = 0;
            _write(1, s, sizeof s - 1);
        }
        iwdg_reset();
        led(i >> 17 & 1);
    }
    led(1);
    printf("board stm32 dev 0x%03lx cpu_hz %lu specs %d largest_block %u\n", (unsigned long)(REG(0xE0042000u) & 0xFFFu),
           (unsigned long)rcc_ahb_frequency, NSPECS, (unsigned)largest);
    for (int i = 0; i < NSPECS; i++) {
        mcu_bench(SPECS[i], 31, 16);
        mcu_bench_ram(SPECS[i], 31, 16, largest - 512);
    }
    printf("DONE\n");
    led(0);
    for (;;) usbd_poll(usbd);
}
