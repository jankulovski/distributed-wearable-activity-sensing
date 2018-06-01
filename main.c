﻿#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <float.h>
#include <stdbool.h>
#include <lwip/dhcp.h>
#include <lwip/sockets.h>
#include <espressif/sdk_private.h>

#include "espressif/esp_common.h"
#include "esp/uart.h"
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "esp8266.h"
#include "lwip/udp.h"
#include "i2c/i2c.h"
#include "lis3dh/lis3dh.h"
#include "espressif/esp_common.h"
#include "esp/uart.h"
#include "queue.h"
#include "espressif/user_interface.h"
#include "semphr.h"


#define FREQ_DIVIDER 0

#define WIFI_LED_IO_MUX     PERIPHS_IO_MUX_GPIO2_U
#define WIFI_LED_IO_NUM     2
#define WIFI_LED_IO_FUNC    FUNC_GPIO2

const uint8_t i2c_bus = 0;
const uint8_t scl_pin = 14;
const uint8_t sda_pin = 12;
const uint8_t mpu9250_adress = 0x68;

ip_addr_t dstaddr;

struct udp_pcb *nefastor_pcb;

// Global variable needed for calibration mpu9250_calib()
int x_acc_sum = 0;
int y_acc_sum = 0;
int z_acc_sum = 0;

struct FloatIntUnion {
    float f;
    int tmp;
};

// Custom sqrt function
float sqrt_(float z)
{
    if (z == 0) return 0;
    struct FloatIntUnion u;
    u.tmp = 0;
    u.f = z;
    u.tmp -= 1 << 23; /* Subtract 2^m. */
    u.tmp >>= 1; /* Divide by 2. */
    u.tmp += 1 << 29; /* Add ((b + 1) / 2) * 2^m. */
    return u.f;
}

void mpu9250_init(){
    uint8_t pwr_mng_address = 0x6c;
    uint8_t acc_conf0_address = 0x1c;
    uint8_t acc_conf1_address = 0x1d;
    uint8_t pwr_mng_payload = 0x00;
    uint8_t acc_conf0_payload = 0x00;
    uint8_t acc_conf1_payload = 0x05;

    i2c_slave_write(i2c_bus, mpu9250_adress, &pwr_mng_address, &pwr_mng_payload, 1);
    i2c_slave_write(i2c_bus, mpu9250_adress, &acc_conf0_address, &acc_conf0_payload, 1);
    i2c_slave_write(i2c_bus, mpu9250_adress, &acc_conf1_address, &acc_conf1_payload, 1);
}

void mpu9250_read_acc(void *pvParameters){
    // Readings are in second complement
    // e.g.
    // -2g -> -2^15
    // 2g -> 2^15 - 1

    uint8_t data_start_address = 0x3b;
    uint8_t data[6];

    // acceleration in 2's complement
    short x_acc = 0;
    short y_acc = 0;
    short z_acc = 0;

    // acceleration in g units
    float x_acc_g = 0;
    float y_acc_g = 0;
    float z_acc_g = 0;

    float accelerations[10];

    vTaskDelay(5000 / portTICK_PERIOD_MS);

    while(1){
        for(uint8_t i=0; i<10; i++){

            data_start_address = 0x3b;

            for (uint8_t j=0;j<6;j++){
                i2c_slave_read(i2c_bus, mpu9250_adress, &data_start_address, &data[j], 1);
                data_start_address++;
            }

            x_acc = data[1] + ((short)data[0] << 8);
            y_acc = data[3] + ((short)data[2] << 8);
            z_acc = data[5] + ((short)data[4] << 8);

            x_acc_g = (float)(x_acc - (short)x_acc_sum) / (1 << 14); // 2^14
            y_acc_g = (float)(y_acc - (short)y_acc_sum) / (1 << 14); // 2^14
            z_acc_g = (float)(z_acc - (short)z_acc_sum) / (1 << 14); // 2^14

            accelerations[i] = sqrt_((x_acc_g * x_acc_g)+(y_acc_g * y_acc_g)+ (z_acc_g * z_acc_g) );

            vTaskDelay(50 / portTICK_PERIOD_MS);
        }

        // Construct buffer and send to udp
        struct pbuf * go;
        go = pbuf_alloc(PBUF_TRANSPORT, 1, PBUF_REF);
        go->payload = &accelerations;
        go->len = go->tot_len = sizeof(accelerations);
        udp_sendto(nefastor_pcb, go, &dstaddr, 8888);
        pbuf_free(go);

        // zero-reset accelerations
        for(uint8_t y=0; y<10; y++) {
            accelerations[y] = 0;
        }
    }
}

void mpu9250_calib(){
    // collect 100 measurements and average it
    // This is the template for calibration, but it isn't needed
    // When the Gy-91 is still, then you have acc_z = 1g and acc_x = acc_y = 0
    // in other words readings from sensor are acc_z_out = 2^14, acc_x_out = acc_x_out \approx 0
    // After finding mean you need also to store calibration value in X,Y and Z_offs_register
    // but implemented here with global variable

    uint8_t data_start_address = 0x3b;
    uint8_t data[6];
    uint8_t i,j;

    for(i=0;i<100;i++){

        data_start_address = 0x3b;

        for (j=0;j<6;j++){
            i2c_slave_read(i2c_bus, mpu9250_adress, &data_start_address, &data[j], 1);
            data_start_address++;
        }

        x_acc_sum += data[1] + ((int)data[0] << 8);
        y_acc_sum += data[3] + ((int)data[2] << 8);
        z_acc_sum += data[5] + ((int)data[4] << 8);
    }

    x_acc_sum /= 100;
    y_acc_sum /= 100;
    z_acc_sum /= 100;
    z_acc_sum = z_acc_sum - (1<<14) ;  // when standing, sensors measures 1g in z-axis and that corrensponds to 2^14
}


// Callback function for UDP
void _is_esp_connected_to_wifi(void){
    int status = sdk_wifi_station_get_connect_status();
    while(status != STATION_GOT_IP){
        vTaskDelay(100);
        status = sdk_wifi_station_get_connect_status();
        printf("%d \n",status);
    }
    return;
}


void user_init(void)
{
    uart_set_baud(0, 115200);

    i2c_init(i2c_bus, scl_pin, sda_pin, I2C_FREQ_400K);

    mpu9250_init();
    vTaskDelay(500 / portTICK_PERIOD_MS); // sleep 100ms

    mpu9250_calib();
    vTaskDelay(1000 / portTICK_PERIOD_MS); // sleep 100ms

    xTaskCreate(mpu9250_read_acc, "mpu_read", 256, NULL, 2, NULL);

    // set params for network
    char ssid[] = "ssid";
    char password[] = "password";
    struct sdk_station_config _station_info;

    strcpy((char *)_station_info.ssid, ssid);
    strcpy((char *)_station_info.password, password);
    _station_info.bssid_set = 0;

    //Must call sdk_wifi_set_opmode before sdk_wifi_station_set_config
    sdk_wifi_set_opmode(STATION_MODE);
    sdk_wifi_station_set_config(&_station_info);

    // setting static IP adress
    struct ip_info info;
    sdk_wifi_station_dhcpc_stop();
    IP4_ADDR(&info.ip, 192, 168, 0, 108);
    IP4_ADDR(&info.gw, 192, 168, 0, 1);
    IP4_ADDR(&info.netmask, 255, 255, 255, 0);
    sdk_wifi_set_ip_info(STATION_IF, &info);

    // Connect esp8266 to a network
    sdk_wifi_station_connect();

    // Set receiver
    IP4_ADDR(&dstaddr, 192, 168, 0, 103);

    // Listening for UDP
    nefastor_pcb = udp_new();
    udp_bind(nefastor_pcb, IP_ADDR_ANY, 8888);
}