function waitFor(duration) {
  return new Promise((resolve) => setTimeout(resolve, duration));
}

function uint8ArrayToFloat32(raw) {
  const bytes = raw instanceof Uint8Array ? raw : new Uint8Array(raw);
  return new Float32Array(bytes.buffer)[0];
}

function uint8ArrayToUint16(raw) {
  const bytes = raw instanceof Uint8Array ? raw : new Uint8Array(raw);
  return new Uint16Array(bytes.buffer)[0];
}

function uint8ArrayToHex(raw) {
  const parts = [];
  raw.forEach((i) => {
    const b = i.toString(16);
    parts.push(b.length > 1 ? b : `0${b}`);
  });
  return parts.join(" ");
}

function utf82string(code) {
  if (code >= 0xf0000000) {
    const utf8code = (((code >> 24) & 7) << 21) + (((code >> 16) & 0x3f) << 12) + (((code >> 8) & 0x3f) << 6) + (code & 0x3f);
    return String.fromCharCode(utf8code);
  }
  if (code >= 0xe00000 && code < 0xf0000000) {
    const utf8code = (((code >> 16) & 0xf) << 12) + (((code >> 8) & 0x3f) << 6) + (code & 0x3f);
    return String.fromCharCode(utf8code);
  }
  if (code >= 0xc000 && code < 0xe00000) {
    const utf8code = (((code >> 8) & 0x1f) << 6) + (code & 0x3f);
    return String.fromCharCode(utf8code);
  }
  if (code < 0xc000) return String.fromCharCode(code);
  return "";
}

function bufferToString(buffer) {
  let str = "";
  for (const code of buffer) {
    if (code === 0) break;
    str += utf82string(code);
  }
  return str;
}

function labToRgb(lab) {
  const l = Number(lab.L ?? lab.l ?? lab[0] ?? 0);
  const a = Number(lab.a ?? lab[1] ?? 0);
  const b = Number(lab.b ?? lab[2] ?? 0);

  let y = (l + 16) / 116;
  let x = a / 500 + y;
  let z = y - b / 200;

  y = y > 6 / 29 ? Math.pow(y, 3) : (y - 16 / 116) / 7.787;
  x = x > 6 / 29 ? Math.pow(x, 3) : (x - 16 / 116) / 7.787;
  z = z > 6 / 29 ? Math.pow(z, 3) : (z - 16 / 116) / 7.787;

  x *= 0.95047;
  z *= 1.08883;

  let red = 3.2406 * x - 1.5372 * y - 0.4986 * z;
  let green = -0.9689 * x + 1.8758 * y + 0.0415 * z;
  let blue = 0.0557 * x - 0.2040 * y + 1.0570 * z;

  red = red > 0.0031308 ? 1.055 * Math.pow(red, 1 / 2.4) - 0.055 : 12.92 * red;
  green = green > 0.0031308 ? 1.055 * Math.pow(green, 1 / 2.4) - 0.055 : 12.92 * green;
  blue = blue > 0.0031308 ? 1.055 * Math.pow(blue, 1 / 2.4) - 0.055 : 12.92 * blue;

  return {
    r: Math.max(Math.min(Math.round(red * 255), 255), 0),
    g: Math.max(Math.min(Math.round(green * 255), 255), 0),
    b: Math.max(Math.min(Math.round(blue * 255), 255), 0),
  };
}

function hex2(n) {
  return Math.max(0, Math.min(255, Number(n) || 0)).toString(16).padStart(2, "0");
}

function labToHex(lab) {
  const rgb = labToRgb(lab);
  return `${hex2(rgb.r)}${hex2(rgb.g)}${hex2(rgb.b)}`.toUpperCase();
}

async function retry(cb, times) {
  const max = Number(times || 1);
  for (let i = 0; i < max + 1; i += 1) {
    try {
      return await cb();
    } catch (e) {
      if (i === max) throw e;
      await waitFor(80);
    }
  }
  return null;
}

module.exports = {
  bufferToString,
  labToHex,
  labToRgb,
  retry,
  uint8ArrayToFloat32,
  uint8ArrayToHex,
  uint8ArrayToUint16,
  waitFor,
};
