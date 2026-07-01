function uint32ToUint8Array(n) {
  return new Uint8Array(new Uint32Array([n]).buffer);
}

class ColorMeterCommand {
  constructor(content, responseSize, timeout, needSign) {
    this.content = content instanceof Uint8Array ? content : new Uint8Array(content);
    this.responseSize = Number(responseSize || 0);
    this.timeout = typeof timeout === "number" && timeout >= 0 ? timeout : 3000;
    this.needSign = needSign !== false;
    this.response = new Uint8Array(0);
  }

  get data() {
    if (!this.content.length) throw new Error("正文内容不能为空");
    const data = [];
    const bytes = new Uint8Array(this.content.buffer);
    if (this.needSign) {
      bytes[bytes.length - 1] = ColorMeterCommand.getSign(bytes);
    }
    for (let i = 0; i < bytes.length; i += 20) {
      data.push(bytes.slice(i, i + 20).buffer);
    }
    return data;
  }

  get isComplete() {
    return this.response.length >= this.responseSize;
  }

  get isValid() {
    return ColorMeterCommand.getSign(this.response) === this.response[this.response.length - 1];
  }

  fillResponse(buffer) {
    this.response = new Uint8Array([...this.response, ...new Uint8Array(buffer)]);
  }

  static getSign(buffer) {
    const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
    let sum = 0;
    bytes.slice(0, bytes.length - 1).forEach((i) => {
      sum += i;
    });
    return new Uint8Array([sum])[0];
  }

  static measure(mode) {
    ColorMeterCommand.measureId += 1;
    const measureId = uint32ToUint8Array(ColorMeterCommand.measureId);
    return new ColorMeterCommand([0xbb, 1, mode || 0, ...measureId, 0, 0xff, 0], 10, 5000);
  }

  static getLab(mode) {
    return new ColorMeterCommand([0xbb, 3, mode || 0, 0, 0, 0, 0, 0, 0xff, 0], 20, 3000);
  }

  static getRGB(mode) {
    return new ColorMeterCommand([0xbb, 4, mode || 0, 0, 0, 0, 0, 0, 0xff, 0], 20, 3000);
  }

  static getDeviceInfo() {
    return new ColorMeterCommand([0xbb, 0x12, 0x01, 0, 0, 0, 0, 0, 0xff, 0], 200, 5000);
  }

  static getBatteryInfo() {
    return new ColorMeterCommand([0xbb, 0x1d, 0, 0, 0, 0, 0, 0, 0xff, 0], 10, 5000);
  }
}

ColorMeterCommand.measureId = 1;
ColorMeterCommand.WakeUp = new ColorMeterCommand([0xf0], 0, 0, false);

module.exports = {
  ColorMeterCommand,
};
