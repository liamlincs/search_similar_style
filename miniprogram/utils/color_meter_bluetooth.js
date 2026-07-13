const { ColorMeterCommand } = require("./color_meter_command");
const {
  bufferToString,
  uint8ArrayToFloat32,
  uint8ArrayToHex,
  uint8ArrayToUint16,
  waitFor,
} = require("./color_meter_utils");

function getErrorText(err) {
  return String((err && (err.errMsg || err.message)) || "");
}

function isBluetoothAlreadyOpened(err) {
  return /already opened/i.test(getErrorText(err));
}

function isBluetoothPermissionError(err) {
  const text = getErrorText(err).toLowerCase();
  return /auth|authorize|permission|privacy|deny|denied|unauthorized/.test(text);
}

function shouldOpenBluetoothSetting(err) {
  const text = getErrorText(err).toLowerCase();
  if (/privacy/.test(text)) return false;
  return /auth|authorize|permission|deny|denied|unauthorized/.test(text);
}

function getBluetoothErrorMessage(err) {
  const code = err && err.errCode;
  if (isBluetoothAlreadyOpened(err)) return "";
  const text = getErrorText(err);
  if (/appid privacy|api scope is not declared in the privacy agreement/i.test(text)) {
    return "蓝牙接口未通过正式版隐私校验，请在小程序后台用户隐私保护指引中声明蓝牙用途后重新发布";
  }
  if (code === 10001) return "请打开手机蓝牙，并允许微信使用蓝牙";
  if (code === 10009) return "当前系统版本不支持低功耗蓝牙";
  if (code === 10000) return "蓝牙初始化失败，请关闭页面后重试";
  if (code === 10008) return "蓝牙系统错误，请重启蓝牙后重试";
  if (isBluetoothPermissionError(err)) {
    return text ? `蓝牙授权失败：${text}` : "蓝牙授权失败，请检查微信蓝牙权限";
  }
  if (text) return `蓝牙不可用：${text}`;
  return "蓝牙不可用或未授权";
}

class ColorMeterBluetooth {
  constructor() {
    this.listeners = new Set();
    this.discovering = false;
    this.available = true;
    this.connected = null;
    this.connecting = null;
    this.serviceRule = /^0000FFE0/i;
    this.serviceId = null;
    this.characteristicRule = /^0000FFE1/i;
    this.characteristicId = null;
    this.command = null;
    this.responseResolve = null;
    this.responseReject = null;
    this.responseTimer = null;
    this.debug = false;
    this.inited = false;
  }

  init() {
    if (this.inited) return this.openAdapter();
    this.inited = true;
    wx.onBluetoothAdapterStateChange((res) => {
      this.discovering = !!res.discovering;
      this.available = !!res.available;
      this.emit({ type: "stateUpdate", detail: res });
    });
    wx.onBLEConnectionStateChange((res) => {
      if (res.connected) {
        if (this.connecting) {
          this.connected = this.connecting;
          this.connecting = null;
        }
        this.emit({ type: "connected", detail: res });
      } else {
        if (this.connected && this.connected.deviceId === res.deviceId) {
          this.connected = null;
          this.resetCommand();
        }
        this.emit({ type: "disconnect", detail: res });
      }
    });
    wx.onBLECharacteristicValueChange(({ value }) => this.notifySubscriber(value));
    return this.openAdapter();
  }

  subscribe(cb) {
    if (cb) this.listeners.add(cb);
  }

  unsubscribe(cb) {
    if (cb) this.listeners.delete(cb);
  }

  emit(event) {
    this.listeners.forEach((cb) => cb && cb(event));
  }

  requirePrivacyAuthorize() {
    if (!wx.requirePrivacyAuthorize) return Promise.resolve();
    return new Promise((resolve, reject) => {
      wx.requirePrivacyAuthorize({
        success: resolve,
        fail: reject,
      });
    });
  }

  async openAdapter() {
    await this.requirePrivacyAuthorize();
    return new Promise((resolve, reject) => {
      wx.openBluetoothAdapter({
        success: resolve,
        fail: (err) => {
          if (isBluetoothAlreadyOpened(err)) resolve(err);
          else reject(err);
        },
      });
    });
  }

  getErrorMessage(err) {
    return getBluetoothErrorMessage(err);
  }

  isPermissionError(err) {
    return isBluetoothPermissionError(err);
  }

  shouldOpenSetting(err) {
    return shouldOpenBluetoothSetting(err);
  }

  getAdapterState() {
    return new Promise((resolve, reject) => {
      wx.getBluetoothAdapterState({ success: resolve, fail: reject });
    });
  }

  startScan(cb, duration) {
    const timeout = duration === undefined ? 10000 : Number(duration || 0);
    wx.onBluetoothDeviceFound(cb);
    return new Promise((resolve, reject) => {
      wx.startBluetoothDevicesDiscovery({
        allowDuplicatesKey: true,
        success: resolve,
        fail: reject,
      });
      if (timeout > 0) {
        setTimeout(() => this.stopScan(cb), timeout);
      }
    });
  }

  stopScan(cb) {
    if (cb) wx.offBluetoothDeviceFound(cb);
    wx.stopBluetoothDevicesDiscovery();
    this.discovering = false;
  }

  async connect(device) {
    this.connecting = device;
    try {
      await this.createConnection(device.deviceId);
      await this.discoverService(device.deviceId);
      await this.discoverCharacteristic(device.deviceId);
      await this.notifyCharacteristicValueChange(device.deviceId);
      this.connected = device;
      this.connecting = null;
      return device;
    } catch (e) {
      this.connecting = null;
      throw e;
    }
  }

  async disconnect() {
    const device = this.connected || this.connecting;
    if (!device) return;
    await this.closeConnection(device.deviceId).catch(() => null);
    this.connected = null;
    this.connecting = null;
    this.serviceId = null;
    this.characteristicId = null;
    this.resetCommand();
  }

  createConnection(deviceId) {
    return new Promise((resolve, reject) => {
      wx.createBLEConnection({ deviceId, timeout: 5000, success: resolve, fail: reject });
    });
  }

  closeConnection(deviceId) {
    return new Promise((resolve, reject) => {
      wx.closeBLEConnection({ deviceId, success: resolve, fail: reject });
    });
  }

  discoverService(deviceId) {
    return new Promise((resolve, reject) => {
      wx.getBLEDeviceServices({
        deviceId,
        success: ({ services }) => {
          const service = (services || []).find((i) => this.serviceRule.test(i.uuid));
          if (!service) reject(new Error("未找到色差仪 BLE 服务 FFE0"));
          else {
            this.serviceId = service.uuid;
            resolve(service);
          }
        },
        fail: reject,
      });
    });
  }

  discoverCharacteristic(deviceId) {
    return new Promise((resolve, reject) => {
      wx.getBLEDeviceCharacteristics({
        deviceId,
        serviceId: this.serviceId,
        success: ({ characteristics }) => {
          const characteristic = (characteristics || []).find((i) => this.characteristicRule.test(i.uuid));
          if (!characteristic) reject(new Error("未找到色差仪 BLE 特征 FFE1"));
          else {
            this.characteristicId = characteristic.uuid;
            resolve(characteristic);
          }
        },
        fail: reject,
      });
    });
  }

  notifyCharacteristicValueChange(deviceId, state) {
    return new Promise((resolve, reject) => {
      wx.notifyBLECharacteristicValueChange({
        deviceId,
        serviceId: this.serviceId,
        characteristicId: this.characteristicId,
        state: state !== false,
        success: resolve,
        fail: reject,
      });
    });
  }

  notifySubscriber(buffer) {
    if (this.command) {
      if (this.debug) console.log(`[BLE RESP] ${uint8ArrayToHex(new Uint8Array(buffer))}`);
      this.command.fillResponse(buffer);
      if (!this.command.isComplete) return;
      if (this.command.isValid && this.responseResolve) {
        this.responseResolve(this.command.response);
      } else if (!this.command.isValid && this.responseReject) {
        this.responseReject(new Error("无效数据"));
      }
      this.resetCommand();
      return;
    }

    const bytes = new Uint8Array(buffer);
    if (bytes[0] === 0xbb && bytes[1] === 1 && bytes[3] === 0) {
      this.emit({ type: "measure", detail: { mode: bytes[2] } });
    }
  }

  exec(command) {
    return new Promise(async (resolve, reject) => {
      if (!this.connected) {
        reject(new Error("色差仪未连接"));
        return;
      }
      if (this.command) {
        reject(new Error("正在执行其他蓝牙命令"));
        return;
      }
      try {
        this.command = command;
        const data = command.data;
        for (let i = 0; i < data.length; i += 1) {
          await this.sendData(data[i]);
        }
        if (command.responseSize <= 0) {
          resolve();
          this.resetCommand();
        } else {
          this.responseReject = reject;
          this.responseResolve = resolve;
          this.responseTimer = setTimeout(() => {
            reject(new Error("命令响应超时"));
            this.resetCommand();
          }, command.timeout);
        }
      } catch (e) {
        this.resetCommand();
        reject(e);
      }
    });
  }

  sendData(buffer) {
    if (this.debug) console.log(`[BLE SEND] ${uint8ArrayToHex(new Uint8Array(buffer))}`);
    return new Promise((resolve, reject) => {
      wx.writeBLECharacteristicValue({
        deviceId: this.connected.deviceId,
        serviceId: this.serviceId,
        characteristicId: this.characteristicId,
        value: buffer,
        success: resolve,
        fail: reject,
      });
    });
  }

  resetCommand() {
    if (this.responseTimer) clearTimeout(this.responseTimer);
    this.command = null;
    this.responseResolve = null;
    this.responseReject = null;
    this.responseTimer = null;
  }

  async measure(mode) {
    await this.exec(ColorMeterCommand.WakeUp);
    await waitFor(50);
    return this.exec(ColorMeterCommand.measure(mode || 0));
  }

  async getLab(mode) {
    await this.exec(ColorMeterCommand.WakeUp);
    await waitFor(50);
    const data = await this.exec(ColorMeterCommand.getLab(mode || 0));
    return {
      L: uint8ArrayToFloat32(data.slice(5, 9)),
      a: uint8ArrayToFloat32(data.slice(9, 13)),
      b: uint8ArrayToFloat32(data.slice(13, 17)),
    };
  }

  async measureAndGetLab(mode) {
    await this.measure(mode || 0);
    await waitFor(50);
    return this.getLab(mode || 0);
  }

  async getRGB(mode) {
    await this.exec(ColorMeterCommand.WakeUp);
    await waitFor(50);
    const data = await this.exec(ColorMeterCommand.getRGB(mode || 0));
    return {
      R: uint8ArrayToUint16(data.slice(5, 7)),
      G: uint8ArrayToUint16(data.slice(7, 9)),
      B: uint8ArrayToUint16(data.slice(9, 11)),
    };
  }

  async measureAndGetRGB(mode) {
    await this.measure(mode || 0);
    await waitFor(50);
    return this.getRGB(mode || 0);
  }

  async getDeviceInfo() {
    await waitFor(50);
    const data = await this.exec(ColorMeterCommand.getDeviceInfo());
    const softwareVersion = bufferToString(data.slice(97, 127));
    return {
      serial: bufferToString(data.slice(67, 97)),
      device_alias: bufferToString(data.slice(37, 67)),
      softwareVersion,
      deviceCode: uint8ArrayToUint16(data.slice(5, 7)),
    };
  }

  async getBatteryInfo() {
    await this.exec(ColorMeterCommand.WakeUp);
    await waitFor(50);
    const data = await this.exec(ColorMeterCommand.getBatteryInfo());
    return data[2];
  }
}

const shared = new ColorMeterBluetooth();

module.exports = {
  ColorMeterBluetooth,
  ColorMeter: shared,
  getBluetoothErrorMessage,
  isBluetoothPermissionError,
  shouldOpenBluetoothSetting,
};
